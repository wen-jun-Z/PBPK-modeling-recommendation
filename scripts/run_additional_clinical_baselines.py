#!/usr/bin/env python3
"""Add train-only-selected SVR/XGBoost priors to the frozen ten-drug PBPK batch.

The original RF, Ours, clinical references and core solver are immutable. This
runner changes the molecular parameter estimator only, then uses the same
direct PBPK construction and native audits as RF. The cohort is developmental,
not a new blinded test. Clinical outcomes never select hyperparameters.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
import xgboost
from xgboost import XGBRegressor

from run_complete_reference_pilot import ROOT, WHEEL, read, ref, verify, now
from pbpk_kb.blueprints.source_workflow import save, sha256
from pbpk_kb.blueprints.pk_parameter_learning import (
    load_parameter_dataset, normalize_structure, _features, _metrics,
)

BASE = ROOT / 'experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug'
OUT = BASE / 'additional_baselines/ml'
METHODS = ('svr', 'xgboost')
PARAMETERS = {'human_CL_mL_min_kg': 'clearance_ml_min_kg',
              'human_VDss_L_kg': 'vss_l_kg', 'human_fup': 'fraction_unbound'}


def estimator(method, config, seed, device):
    if method == 'svr':
        return make_pipeline(StandardScaler(), SVR(kernel='rbf', C=config['C'],
                             gamma='scale', epsilon=.1, cache_size=512))
    return XGBRegressor(n_estimators=400, max_depth=config['max_depth'],
        learning_rate=.05, min_child_weight=5, subsample=.9, colsample_bytree=.8,
        reg_lambda=5., objective='reg:squarederror', tree_method='hist', device=device,
        random_state=seed, n_jobs=4, verbosity=0)


def train(summary):
    dataset = load_parameter_dataset(WHEEL)
    split = read(BASE / 'run/parameter_split.json')
    assert dataset.provenance['csv_sha256'] == split['source']['csv_sha256']
    train_ids, test_ids = split['train_indices'], split['test_indices']
    assert set(train_ids).isdisjoint(test_ids)
    external = {x['scaffold_group'] for x in split['external_exclusions'].values()}
    train_groups = {dataset.rows[i]['scaffold_group'] for i in train_ids}
    test_groups = {dataset.rows[i]['scaffold_group'] for i in test_ids}
    assert not train_groups & test_groups and not external & (train_groups | test_groups)
    ids = sorted(train_ids + test_ids)
    features = _features([dataset.rows[i]['identity_smiles'] for i in ids])
    position = {i: k for k, i in enumerate(ids)}
    models, audit = {}, {'train_count': len(train_ids), 'test_count': len(test_ids),
                        'same_split_as_rf': True, 'endpoints': {}, 'software': {'xgboost': xgboost.__version__}}
    for method in METHODS:
        models[method] = {}; audit['endpoints'][method] = {}
        for endpoint in PARAMETERS:
            labeled = [i for i in train_ids if dataset.rows[i]['labels_log10'][endpoint] is not None]
            test = [i for i in test_ids if dataset.rows[i]['labels_log10'][endpoint] is not None]
            X = features[[position[i] for i in labeled]]
            y = np.array([dataset.rows[i]['labels_log10'][endpoint] for i in labeled])
            groups = [dataset.rows[i]['scaffold_group'] for i in labeled]
            configs = summary['protocol']['candidate_configs'][method]
            candidates = []
            for config in configs:
                errors, folds = [], []
                for fold_train, fold_valid in GroupKFold(3).split(X, y, groups):
                    model = estimator(method, config, split['seed'], summary['protocol']['xgboost_device'])
                    model.fit(X[fold_train], y[fold_train])
                    pred = model.predict(X[fold_valid])
                    errors.extend(np.abs(pred-y[fold_valid]).tolist())
                    folds.append({'train_n': len(fold_train), 'valid_n': len(fold_valid),
                                  'scaffold_overlap': sorted(set(np.array(groups)[fold_train]) & set(np.array(groups)[fold_valid]))})
                candidates.append({'parameters': config, 'cv_log10_mae': float(np.mean(errors)), 'folds': folds})
            best = min(candidates, key=lambda x: x['cv_log10_mae'])
            model = estimator(method, best['parameters'], split['seed'], summary['protocol']['xgboost_device'])
            model.fit(X, y); models[method][endpoint] = model
            yt = [dataset.rows[i]['labels_log10'][endpoint] for i in test]
            pt = model.predict(features[[position[i] for i in test]])
            audit['endpoints'][method][endpoint] = {'train_n': len(labeled), 'test_n': len(test),
                'selected': best['parameters'], 'candidates': candidates,
                'parameter_test_metrics': _metrics(yt, pt),
                'parameter_test_predictions': [{'row_index': i, 'observed_log10': a, 'predicted_log10': float(b)} for i, a, b in zip(test, yt, pt)]}
            if method == 'xgboost':
                audit['endpoints'][method][endpoint]['actual_device'] = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
            summary['training_progress'] = {'method': method, 'endpoint': endpoint, 'selected': best['parameters']}
            save(OUT / 'summary.json', summary)
            print('trained', method, endpoint, best['parameters'], flush=True)
    joblib.dump(models, OUT / 'parameter_models.joblib', compress=3)
    save(OUT / 'parameter_training.json', audit)
    summary['training'] = audit
    summary['locks']['models'] = ref(OUT / 'parameter_models.joblib', 'SVR/XGBoost selected on training-only scaffold CV')
    summary['locks']['training'] = ref(OUT / 'parameter_training.json', 'Training-only selection and untouched parameter test')
    summary['status'] = 'models_locked'; save(OUT / 'summary.json', summary)
    return models


def construct(case, method, models, summary):
    folder = OUT / method / case['case_id']; folder.mkdir(parents=True, exist_ok=False)
    req = read(BASE / 'run/inputs' / case['case_id'] / 'target_request.json')
    physical = read(ROOT / case['physchem'])
    structure = normalize_structure(physical['identity']['isomeric_smiles'])
    split = read(BASE / 'run/parameter_split.json')
    training = [r for r in split['rows'] if r['status'] == 'train']
    assert not any(r['identity_smiles'] == structure['identity_smiles'] or
                   r['scaffold_group'] == structure['scaffold_group'] for r in training)
    X = _features([structure['identity_smiles']])
    values, raw = {}, {}
    for endpoint, key in PARAMETERS.items():
        value = float(10 ** float(models[method][endpoint].predict(X)[0]))
        assert math.isfinite(value) and value > 0
        raw[key] = value
        values[key] = min(1., value) if key == 'fraction_unbound' else value
    record = {'method': method, 'compound_id': case['compound_id'], 'predictions': values,
              'raw_predictions': raw, 'fraction_cap_policy': 'Prespecified physical cap at 1, no outcome-based adjustment',
              'training_overlap': False, 'same_scaffold_training_overlap': False,
              'provenance': {'model': summary['locks']['models'], 'split': ref(BASE / 'run/parameter_split.json', 'Identical RF scaffold split')},
              'clinical_observations_used': False, 'predicted_vss_used': False}
    save(folder / 'predicted_pk.json', record)
    source = ref(folder / 'predicted_pk.json', 'Locked molecular '+method+' PK predictions')
    for evidence in req['evidence']:
        key = {'total_clearance': 'clearance_ml_min_kg', 'fraction_unbound': 'fraction_unbound'}.get(evidence['parameter'])
        if key:
            evidence.update(value=values[key], source=source, basis='predicted')
            evidence['prediction_method'].update(name=method, version='additional-clinical-baselines-1')
    save(folder / 'evidence_ledger.json', {'evidence': req['evidence']})
    req['evidence_ledger'] = ref(folder / 'evidence_ledger.json', 'Shared physical inputs with '+method+' molecular priors')
    req['assumptions'] = [s.replace('RF does not', method.upper()+' does not') for s in req['assumptions']]
    save(folder / 'request.json', req)
    return folder, req


def execute_one(folder, req, method):
    from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint
    from pbpk_kb.blueprints.pk_anchored_blueprint import mapped_snapshot, _run
    req = EvidenceBlueprintRequest.model_validate(req).model_dump(mode='json', exclude_none=True)
    blueprint = assemble_evidence_blueprint(req, ROOT)
    if blueprint['status'] != 'assembled':
        raise ValueError('Incomplete PBPK construction: '+str(blueprint.get('gaps')))
    logp = next(e['value'] for e in req['evidence'] if e['parameter'] == 'logp')
    snapshot = mapped_snapshot(blueprint, logp, 1.)
    result = _run(snapshot, req, folder, ROOT, audit=True)
    ep = folder / 'endpoints.json'
    if ep.exists():
        endpoints = read(ep)
        endpoints.update(parameters_fitted=False, construction_parameters_fitted=False,
            parameter_construction='direct_'+method+'_inputs_without_CL_Vss_mapping',
            observations_used_for_parameter_construction=False, construction_target_basis='predicted', clinical_validation=False)
        save(ep, endpoints)
    result.update(method=method, construction='Same direct PBPK recipe and native audits as RF',
                  predicted_vss_used=False, observations_used=False)
    if result['status'] == 'predicted':
        result['prediction'] = ref(folder / 'prediction_locked.json', 'Frozen native '+method+' PBPK prediction')
    save(folder / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: raise ValueError('Use 1 through 8 project workers')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        raise FileExistsError('Existing run is immutable; inspect its status instead of overwriting')
    base = read(BASE / 'run/summary.json')
    verify(base['protocol']['pins']); verify(base['locks']['inputs'])
    summary = {'status': 'training', 'created_at': now(), 'protocol': {
        'methods': list(METHODS), 'cases': base['protocol']['cases'],
        'candidate_configs': {'svr': [{'C': c} for c in (1., 10., 100.)],
                              'xgboost': [{'max_depth': d} for d in (3, 5)]},
        'selection': '3-fold scaffold-group CV within the existing RF training rows, pooled log10 MAE',
        'clinical_outcomes_used_for_selection': False, 'xgboost_device': args.device,
        'workers': args.workers, 'interpretation': 'Additional exploratory comparators on the already-viewed ten-drug cohort; not a blinded external test',
        'information': 'Molecular features only, like RF; Ours and 2-compartment additionally use development clinical curves',
        'reference': 'Independent clinical observations are not decoded until all new predictions are locked',
        'pins': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), WHEEL, BASE / 'run/parameter_split.json']}},
        'locks': {}, 'results': {}}
    save(OUT / 'summary.json', summary)
    pid = OUT / 'runner.pid'; pid.write_text(str(os.getpid())+'\n')
    started = time.monotonic()
    try:
        with threadpool_limits(limits=4):
            models = train(summary)
        inputs = {(m, c['case_id']): construct(c, m, models, summary) for m in METHODS for c in summary['protocol']['cases']}
        summary['locks']['inputs'] = {str(p.relative_to(ROOT)): sha256(p) for folder, _ in inputs.values() for p in folder.glob('*.json')}
        summary['status'] = 'native_predictions'; save(OUT / 'summary.json', summary)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(execute_one, folder, req, m): (m, cid) for (m, cid), (folder, req) in inputs.items()}
            for job in as_completed(jobs):
                m, cid = jobs[job]
                try: result = job.result()
                except Exception as exc: result = {'status': 'failed', 'reason': type(exc).__name__+': '+str(exc)}
                summary['results'].setdefault(cid, {})[m] = result
                summary['elapsed_seconds'] = round(time.monotonic()-started, 1)
                save(OUT / 'summary.json', summary)
                print('prediction', m, cid, result['status'], flush=True)
        verify(summary['protocol']['pins']); verify(summary['locks']['inputs'])
        summary['status'] = 'predictions_locked'; summary['finished_at'] = now()
        summary['failures'] = [{'case': cid, 'method': m, 'result': v} for cid, methods in summary['results'].items() for m,v in methods.items() if v['status'] != 'predicted']
        save(OUT / 'summary.json', summary)
    finally:
        if pid.exists() and pid.read_text().strip() == str(os.getpid()): pid.unlink()


if __name__ == '__main__': main()
