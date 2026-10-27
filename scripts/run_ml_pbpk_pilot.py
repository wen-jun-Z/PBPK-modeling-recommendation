#!/usr/bin/env python3
"""Bounded, molecule-held-out parameter-learning and native PBPK pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import joblib

from pbpk_kb.blueprints.evidence_evaluation import score_case
from pbpk_kb.blueprints.extended_prediction import IV_ENDPOINTS
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/blueprint_design/ml_hybrid_pilot_20260928'
WHEEL = ROOT / 'artifacts/pk_predictors/pksmart/pksmart-3.0.1-py3-none-any.whl'
METHODS = ('median_direct', 'rf_direct', 'rf_vss_mapping')
SEED = 20260928
ACTIVE_OUTPUT = None


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def ref(path, location):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'location': location}


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predicted_request(case, prediction_source, output, method):
    request = deepcopy(case['request'])
    request['partition_method'] = 'PKSimStandard'
    request['target']['endpoints'] = list(IV_ENDPOINTS)
    request['target']['primary_endpoint'] = 'plasma_concentration_time'
    request['target'].pop('terminal_fit', None)
    prediction = read(ROOT/prediction_source['path'])
    # Remove every existing clinical clearance and protein-binding input.
    remove = {'fraction_unbound', 'total_clearance', 'hepatic_clearance', 'renal_clearance', 'elimination_scope'}
    evidence = [deepcopy(row) for row in request['evidence'] if row['parameter'] not in remove]
    provenance = {'name': prediction['method'], 'version': 'ml-parameter-pilot-1',
        'training_overlap': False,
        'applicability': {'human_IV_parameter_model': True,
            'target_structure_and_scaffold_excluded': True,
            'prospective_clinical_validation': False}}
    for parameter, key, unit in [('total_clearance', 'clearance_ml_min_kg', 'ml/min/kg'),
                                  ('fraction_unbound', 'fraction_unbound', 'fraction')]:
        evidence.append({'parameter': parameter, 'compound_id': request['compound_id'],
            'value': float(prediction['predictions'][key]), 'unit': unit, 'basis': 'predicted',
            'source': prediction_source, 'context': {'species': 'human', 'matrix': 'plasma', 'route': 'intravenous'},
            'prediction_method': provenance,
            'notes': 'Structure-only learner; target-drug PK labels and scaffold are withheld.'})
    allocation = output/'allocation_assumption.json'
    if not allocation.exists():
        save(allocation, {'elimination_scope': 'hepatic_only',
            'basis': 'lumped total systemic clearance assigned to liver for this limited pilot',
            'not_verified_organ_specific_clearance': True,
            'same_assumption_for_all_three_methods': True})
    evidence.append({'parameter': 'elimination_scope', 'compound_id': request['compound_id'],
        'value': 'hepatic_only', 'unit': 'text', 'basis': 'assumption',
        'source': ref(allocation, 'declared pilot clearance allocation'),
        'context': {}, 'notes': 'Systemic exposure approximation; not an organ-specific mechanistic claim.'})
    request['evidence'] = evidence
    ledger = output/'ledger.json'
    save(ledger, {'evidence': evidence})
    request['evidence_ledger'] = ref(ledger, 'immutable ML-driven drug evidence')
    request['assumptions'] = [
        'Historical clinical cases were seen in previous development; this is not a blinded prospective validation.',
        'Clinical target compound and its scaffold are excluded from this learner training and parameter-test split.',
        'CL and fu are predicted from structure; predicted Vss is used only by rf_vss_mapping.',
        'Other independent physicochemical evidence is retained; this is not a purely SMILES-only PBPK system.',
        'PKSimStandard is fixed before any clinical outcome scoring; no clinical partition selection or fit is reused.',
        'One representative adult approximates a published clinical group; demographic defaults remain approximate.',
        'Total predicted plasma clearance is lumped into the liver; renal/hepatic fractions are not validated.',
    ]
    obs_path = ROOT/case['observations_path']
    forbidden = {x['path']: x for x in request.get('forbidden_sources', [])}
    forbidden[str(obs_path.relative_to(ROOT))] = ref(obs_path, 'clinical outcomes for scoring only')
    request['forbidden_sources'] = list(forbidden.values())
    save(output/'request.json', request)
    return request


def main():
    global ACTIVE_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE/'run')
    parser.add_argument('--candidates', type=Path, default=BASE/'inputs/clinical_candidates.json')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    output = inside(args.output, ROOT)
    if args.status:
        print(json.dumps(read(output/'progress.json'), ensure_ascii=False, indent=2))
        return 0
    if output.exists():
        raise FileExistsError('Use a new directory; a previous run is immutable')
    from pbpk_kb.blueprints.pk_parameter_learning import run_parameter_pilot
    from pbpk_kb.blueprints.ml_hybrid_blueprint import run_ml_hybrid_case, run_ml_direct_case
    candidates_path = inside(args.candidates, ROOT, exists=True)
    candidates = read(candidates_path)
    cases = deepcopy(candidates['cases'])
    for case in cases:
        request_path = inside(case['request_path'], ROOT, exists=True)
        if sha256(request_path) != case['request_source']['sha256']:
            raise ValueError('Clinical input inventory no longer matches: '+case['drug'])
        case['request'] = read(request_path)
    assert len(cases) == 3 and len({c['drug'] for c in cases}) == 3
    output.mkdir(parents=True)
    ACTIVE_OUTPUT = output
    started = time.monotonic()
    results = {}
    def progress(stage, **extra):
        save(output/'progress.json', {'stage': stage, 'updated_at': now(), 'pid': os.getpid(),
            'elapsed_seconds': round(time.monotonic()-started, 1),
            'clinical_jobs_completed': len(results), 'clinical_jobs_total': 9, **extra})
    sources = [WHEEL, candidates_path, Path(__file__)]
    sources += list((ROOT/'src/pbpk_kb/blueprints').glob('*.py'))
    sources += list((ROOT/'src/pbpk_kb/model_runtime/R').glob('*.R'))
    for case in cases:
        sources.append(ROOT/case['request_path'])
        sources.append(ROOT/case['observations_path'])
        sources += [ROOT/e['source']['path'] for e in case['request']['evidence']]
    pins = {str(p.relative_to(ROOT)): sha256(p) for p in set(sources)}
    save(output/'protocol.json', {'started_at': now(), 'seed': SEED,
        'train_limit': 400, 'parameter_test_limit': 100,
        'clinical_cases': [{'case_id': c['case_id'], 'drug': c['drug'], 'smiles': c['smiles'],
            'observations_path': c['observations_path']} for c in cases],
        'methods': list(METHODS), 'split_rule': 'exclude clinical identities/scaffolds; split other scaffold groups before training',
        'label_policy': 'use measured literature labels per endpoint; no imputed target values',
        'all_clinical_predictions_locked_before_scoring': True, 'hyperparameter_tuning': False,
        'maximum_mapping_evaluations_per_drug': 12, 'maximum_native_calls': 45,
        'clinical_concentrations_used_for_fitting': False, 'published_clinical_endpoints_used_for_fitting': False,
        'physicochemical_evidence_retained': True, 'partition_method': 'PKSimStandard',
        'historical_case_pilot_not_blinded_validation': True,
        'resources': {'workers': args.workers, 'native_cpu_threads': 2,
                      'native_solver': 'CPU CVODES; no GPU implementation', 'LLM_calls': 0},
        'pins': pins})
    save(output/'runner.json', {'pid': os.getpid(), 'project': str(ROOT), 'started_at': now()})
    progress('train_parameter_models')
    learned = run_parameter_pilot(WHEEL,
        exclude_smiles={c['drug']: c['smiles'] for c in cases}, train_limit=400, test_limit=100,
        seed=SEED, n_jobs=4)
    save(output/'parameter_split.json', learned.audit)
    save(output/'parameter_metrics.json', learned.metrics)
    save(output/'parameter_test_predictions.json', learned.predictions)
    csv_write(output/'parameter_test_predictions.csv', learned.predictions)
    joblib.dump(learned.models, output/'parameter_models.joblib', compress=3)
    save(output/'model_manifest.json', {'model_file': ref(output/'parameter_models.joblib', 'newly trained fixed RFs'),
        'training_medians_log10': learned.medians_log10,
        'no_pretrained_PKSmart_weights_used': True, 'seed': SEED})
    parameters = {}
    for case in cases:
        parameters[case['drug']] = learned.predict(case['smiles'])
        assert not parameters[case['drug']]['training_structure_overlap']
        assert not parameters[case['drug']]['training_scaffold_overlap']
    save(output/'clinical_parameter_predictions_locked.json', parameters)
    jobs = []
    for case in cases:
        for method in METHODS:
            folder = output/'inputs'/case['drug']/method
            folder.mkdir(parents=True)
            learner_method = 'training_median' if method == 'median_direct' else 'random_forest'
            values = parameters[case['drug']]['predictions']
            prediction = {'compound_id': case['compound_id'], 'method': learner_method,
                'predictions': {'clearance_ml_min_kg': values['human_CL_mL_min_kg'][learner_method],
                    'vss_l_kg': values['human_VDss_L_kg'][learner_method],
                    'fraction_unbound': values['human_fup'][learner_method]},
                'training_overlap': False, 'same_scaffold_training_overlap': False,
                'provenance': {'split': ref(output/'parameter_split.json', 'scaffold split and exclusions'),
                    'model': ref(output/'parameter_models.joblib', 'fixed RFs trained for this run')},
                'label_sources': 'PKSmart published human PK parameter dataset; only training subset',
                'no_clinical_outcomes_used': True}
            save(folder/'predicted_pk.json', prediction)
            source = ref(folder/'predicted_pk.json', 'locked structure-derived parameter predictions')
            request = predicted_request(case, source, folder, method)
            from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
            EvidenceBlueprintRequest.model_validate(request)
            jobs.append((case, method, request, source))
    progress('native_pbpk_predictions', parameter_models_completed=True)
    def run(job):
        case, method, request, source = job
        try:
            kwargs = dict(request=request, predicted_pk_source=source,
                output_dir=output/'cases'/case['drug']/method, project_root=ROOT)
            result = run_ml_hybrid_case(**kwargs, max_evaluations=12) if method == 'rf_vss_mapping' else run_ml_direct_case(**kwargs)
        except Exception as error:
            result = {'status': 'failed', 'reason': type(error).__name__+': '+str(error)}
        return {'case_id': case['case_id'], 'drug': case['drug'], 'method': method, 'result': result}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result();key = row['drug']+'/'+row['method'];results[key] = row
            save(output/'job_results.json', results)
            progress('native_pbpk_predictions', latest={'job': key, 'status': row['result']['status']})
            print(json.dumps({'job': key, 'status': row['result']['status']}, ensure_ascii=False), flush=True)
    locks = {}
    for key, row in results.items():
        result = row['result']
        if result['status'] == 'predicted':
            path = inside(result['prediction_path'], ROOT, exists=True)
            endpoint_path = inside(result['endpoints_path'], ROOT, exists=True)
            locks[key] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                'endpoints': ref(endpoint_path, 'locked native predicted endpoints')}
    save(output/'predictions_lock.json', {'locked_at': now(), 'predictions': locks,
        'all_attempts_completed_before_scoring': True})
    for name, digest in pins.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Frozen input/code changed: '+name)
    progress('score_clinical_outcomes')
    clinical_rows = []
    for case, method, request, source in jobs:
        key = case['drug']+'/'+method
        row = results[key]
        if key in locks:
            path = ROOT/locks[key]['path']
            assert sha256(path) == locks[key]['sha256']
            observation = read(ROOT/case['observations_path'])
            if 'observation_dataset_id' in case:
                observation = next(d for d in observation['datasets'] if d['dataset_id'] == case['observation_dataset_id'])
            score_request = deepcopy(request)
            times = [p['time_min'] for p in observation['observations']]
            score_request['target']['terminal_fit'] = {'start_h': times[-4]/60, 'end_h': times[-1]/60,
                'min_points': 4, 'min_adjusted_r2': .98, 'min_span_half_lives': 1.,
                'max_auc_extrapolated_fraction': .2, 'max_aumc_extrapolated_fraction': .2}
            try:
                row['metrics'] = score_case(read(path), observation, score_request)
                row['scoring_status'] = 'completed'
            except Exception as error:
                row['scoring_status'] = 'failed';row['scoring_error'] = str(error)
        else:
            row['scoring_status'] = 'prediction_unavailable'
        metrics = row.get('metrics', {})
        clinical_rows.append({'drug': case['drug'], 'study_id': case['study_id'], 'method': method,
            'prediction_status': row['result']['status'], 'scoring_status': row['scoring_status'],
            'point_count': metrics.get('point_count'), 'gmfe': metrics.get('gmfe'),
            'ln_rmse': metrics.get('log_rmse'),
            'auc_fold_error': metrics.get('auc', {}).get('fold_error'),
            'sampled_cmax_fold_error': metrics.get('sampled_cmax', {}).get('fold_error'),
            'c_last_fold_error': metrics.get('c_last', {}).get('fold_error'),
            'sampled_tmax_absolute_error_h': metrics.get('sampled_tmax', {}).get('absolute_error_h'),
            'half_life_fold_error': metrics.get('terminal_half_life', {}).get('fold_error'),
            'failure_reason': row.get('scoring_error', row['result'].get('reason'))})
    scored = sum(row.get('scoring_status') == 'completed' for row in results.values())
    summary = {'status': 'completed' if len(locks) == scored == 9 else 'completed_with_failures',
        'finished_at': now(), 'elapsed_seconds': round(time.monotonic()-started, 1),
        'clinical_drugs': 3, 'attempted_clinical_predictions': 9, 'successful_native_predictions': len(locks),
        'successful_clinical_scores': scored,
        'clinical_cases_previously_seen': True, 'all_11_clinical_endpoints_validated': False,
        'parameter_metrics': learned.metrics, 'results': results,
        'limitations': ['Three historical clinical drugs do not establish generalization or clinical suitability.',
            'One representative adult is compared with group mean concentrations.',
            'Clinical study endpoint reports must remain separate from curve-recomputed endpoints.',
            'Small fixed training subset and one scaffold split; no model-selection claim.',
            'No confidence intervals from individual timepoints; clinical drugs are the independent units.']}
    save(output/'summary.json', summary)
    csv_write(output/'clinical_comparison.csv', clinical_rows)
    progress(summary['status'], successful_native_predictions=len(locks))
    print(json.dumps({'status': summary['status'], 'seconds': summary['elapsed_seconds'], 'native_predictions': len(locks)}), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        if ACTIVE_OUTPUT is not None:
            previous = read(ACTIVE_OUTPUT/'progress.json') if (ACTIVE_OUTPUT/'progress.json').exists() else {}
            save(ACTIVE_OUTPUT/'progress.json', {**previous, 'stage': 'failed', 'updated_at': now(),
                'failure_reason': type(error).__name__+': '+str(error)})
        raise
