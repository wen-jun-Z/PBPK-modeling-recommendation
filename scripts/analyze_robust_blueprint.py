#!/usr/bin/env python3
"""Source-locked development replay and exploratory cluster-held-out risk audit.

This analysis never fits PBPK parameters, exports physiology, or launches a
native process. Only analysis.json is created, after the experiment completes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprint_policy import predictive_risk as risk
from pbpk_kb.blueprint_policy.perfusion_calibration import prediction_grid
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.blueprints.source_workflow import save, sha256
from report_blueprint_validation import Store
import run_robust_blueprint as batch

VERSION = 'robust-blueprint-risk-analysis-1'
HISTORICAL = 'runtime/cache/robust_structure/risk_baseline.json'
HISTORICAL_SHA256 = 'c03cb78a00545dd9d250cdf57a2a277bb356edb0240dcf8daa8424d3a0a104db'
RISK_SHA256 = '9051ae4b1109a7ef34a41e9c3b1cabd65911ebaacc30dae0fdb40b797140bb17'
CONFIG = {'primary': 'Three methods, available inputs, full development observations; each scored separately with nine study clusters.',
    'missing_input_diagnostics': 'Apply the available-input outer-fold transformations, coefficients and nested thresholds to the same held-out drugs under missing inputs; never retrain on scenario aliases.',
    'replay': 'Selected mapping development physiology and parameters only; pure linear simulation and in-memory dense NCA, no native build/export/refit.',
    'rmse_replay_tolerance': {'rtol': 1e-9, 'atol': 1e-10},
    'historical': 'Previously inspected original always-three full/five 106 endpoints; separate historical development analysis, not pooled with current full-only comparisons.',
    'interpretation': 'All model choices use a previously inspected ten-drug development cohort. Cross-fitting prevents same-cluster label fitting, but does not create blind clinical validation.'}


def verify_pins(store, pins):
    if not isinstance(pins, dict):
        raise ValueError('Expected explicit source hashes')
    for name, digest in pins.items():
        store.checksum(name, digest)


def _ref(store, path):
    path = store.checksum(path)
    return {'path': str(path.relative_to(store.root)), 'sha256': store.pins[str(path.relative_to(store.root))]}


def audit_rf_provenance(store, cases):
    """Follow and verify prediction -> split/weights without unpickling weights."""
    references, splits, records = {}, {}, []
    def locked(reference, *, decode=True):
        references[reference['path']] = reference['sha256']
        return store.ref(reference) if decode else store.checksum(reference['path'], reference['sha256'])
    for case in cases:
        request = locked(case['requests']['rf']['development'])
        selected = {r['parameter']: r for r in request['evidence']
                    if r['parameter'] in ('fraction_unbound', 'total_clearance')}
        if set(selected) != {'fraction_unbound', 'total_clearance'}:
            raise ValueError('Both authorized RF evidence inputs are required')
        if selected['fraction_unbound']['source'] != selected['total_clearance']['source']:
            raise ValueError('RF evidence was not sourced from one locked prediction')
        predicted = locked(selected['fraction_unbound']['source'])
        split_ref = predicted['provenance']['split']; locked(predicted['provenance']['model'], decode=False)
        split_document = locked(split_ref)
        if split_ref['path'] not in splits:
            split = split_document
            source = split['source']
            archive = store.checksum(source['source_path'], source['source_sha256'])
            with zipfile.ZipFile(archive) as z:
                if hashlib.sha256(z.read(source['csv_member'])).hexdigest() != source['csv_sha256']:
                    raise ValueError('RF training CSV differs from its locked archive member')
            indices = split['train_indices']; by_index = {r['index']: r for r in split['rows']}
            if len(indices) != len(set(indices)) or len(indices) != split['train_count']:
                raise ValueError('RF train indices and declared denominator disagree')
            train = [by_index[i] for i in indices]
            if any(r['status'] != 'train' for r in train):
                raise ValueError('RF training membership differs from split rows')
            splits[split_ref['path']] = (split, train)
        split, train = splits[split_ref['path']]
        exclusion = split['external_exclusions'][case['case_id']]
        identity_overlap = sum(r['identity_smiles'] == exclusion['identity_smiles'] for r in train)
        scaffold_overlap = sum(r['scaffold_group'] == exclusion['scaffold_group'] for r in train)
        if identity_overlap or scaffold_overlap:
            raise ValueError('Clinical case structure or scaffold occurs in RF training')
        values = {'fraction_unbound': predicted['predictions']['fraction_unbound'],
                  'total_clearance': predicted['predictions']['clearance_ml_min_kg']}
        for parameter, value in values.items():
            evidence = selected[parameter]
            if (evidence['basis'] != 'predicted' or evidence['value'] != value
                    or evidence['prediction_method']['training_overlap'] is not False):
                raise ValueError('RF evidence value or overlap declaration differs from its source')
        if predicted['training_overlap'] is not False or predicted['same_scaffold_training_overlap'] is not False:
            raise ValueError('RF prediction declares a training overlap')
        records.append({'case_id': case['case_id'], 'values_checked': values, 'training_rows': len(train),
            'identity_overlap_count': identity_overlap, 'scaffold_overlap_count': scaffold_overlap,
            'prediction_source': selected['fraction_unbound']['source'], 'split_source': split_ref,
            'model_source': predicted['provenance']['model']})
    return {'records': records, 'evidence_values_verified': 2 * len(records),
        'request_prediction_split_model_refs': references,
        'unique_request_prediction_split_model_files': len(references),
        'weights_deserialized': False, 'rf_refitted': False,
        'interpretation': 'Source and recorded split-membership audit; it does not establish predictive accuracy.'}


def replay_development(store, mapping_ref):
    """Read an already exported native physiology and run the linear solver only."""
    mapping = store.ref(mapping_ref)
    mapping_path = store.path(mapping_ref['path'])
    for name in ('source_pins', 'code_pins', 'retained_artifact_pins'):
        verify_pins(store, mapping[name])
    fixed_path = mapping_path.parent / 'input.json'
    fixed = store.read(fixed_path, mapping['input_sha256'])
    physiology_path = mapping_path.parent / 'physiology/perfusion_physiology.json'
    physiology_key = str(physiology_path.relative_to(store.root))
    if physiology_key not in mapping['retained_artifact_pins']:
        raise ValueError('Development physiology is not retained by the selected mapping')
    physiology = store.read(physiology_path, mapping['retained_artifact_pins'][physiology_key])
    verify_pins(store, physiology['source_pins'])
    verify_pins(store, physiology['artifact_pins'])
    if (physiology.get('observations_used') is not False
            or physiology.get('source_kind') != 'fresh_evidence_native_model'):
        raise ValueError('Development physiology lacks its input-only source audit')
    module_name = ('transport_calibration' if mapping['variant'] == 'flow_exchange'
                   else 'robust_exchange_calibration')
    engine = importlib.import_module('pbpk_kb.blueprint_policy.' + module_name)
    engine._validated(mapping, mapping_path, store.root)
    if mapping['status'] != 'fitted' or not mapping['eligible'] or not mapping['decision']:
        raise ValueError('Cannot replay an unsuccessful development fit')
    request = fixed['request']; training = store.ref(mapping['training_source'])
    if training['human'] != request['human'] or training['dosing'] != request['dosing']:
        raise ValueError('Development observations and physiology conditions disagree')
    keys = engine.PARAMETERS[mapping['variant']]
    log_parameters = mapping['decision']['log_parameters']
    mw = next(float(r['value']) for r in request['evidence'] if r['parameter'] == 'molecular_weight')
    observed_times = np.asarray([r['time_min'] for r in training['observations']], float) / 60
    observed = np.asarray([r['concentration_umol_per_l'] for r in training['observations']], float)
    check = engine._simulate(physiology, request, observed_times, mapping['variant'], keys, log_parameters)
    fitted = np.asarray(check['concentration_mg_l'], float) * 1000 / mw
    if not np.isfinite(fitted).all() or np.any(fitted <= 0) or np.any(observed <= 0):
        raise ValueError('Invalid development observation replay')
    rmse = float(np.sqrt(np.mean(np.log(fitted / observed)**2)))
    expected = mapping['decision']['data_only_ln_rmse']
    if not np.isclose(rmse, expected, **CONFIG['rmse_replay_tolerance']):
        raise ValueError('Pure development replay disagrees with locked fit RMSE')
    times = prediction_grid(request)
    predicted = engine._simulate(physiology, request, times, mapping['variant'], keys, log_parameters)
    curve = {'time_min': (times * 60).tolist(),
        'concentration': (np.asarray(predicted['concentration_mg_l']) * 1000 / mw).tolist(),
        'time_unit': 'min', 'unit': 'umol/l', 'observation': predicted['observation']}
    endpoints = summarize_extended_prediction(curve, request)
    bounds = (mapping['optimizer_config']['bounds']['clearance_scale'] if mapping['variant'] == 'flow_exchange'
              else mapping['clearance_prior_policy']['scale_bounds'])
    features, support = risk.development_features(training=training, prediction=curve, endpoints=endpoints,
        mapping=mapping, request=request, clearance_bounds=bounds)
    audit = {'mapping_source': deepcopy(mapping_ref), 'training_source': mapping['training_source'],
        'physiology_source': _ref(store, physiology_path), 'development_request_source': _ref(store, fixed_path),
        'variant': mapping['variant'], 'construction_partition_method': mapping['construction_partition_method'],
        'development_points': len(observed_times), 'dense_points': len(times),
        'locked_development_ln_rmse': expected, 'replayed_development_ln_rmse': rmse,
        'rmse_absolute_difference': abs(rmse - expected), 'linear_solver_calls': 2, 'native_calls': 0,
        'mass_balance': predicted['mass_balance'], 'observation': predicted['observation'],
        'retained_artifacts_verified': len(mapping['retained_artifact_pins']), 'support': support,
        'curve_sha256': hashlib.sha256(json.dumps(curve, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        'curve_saved': False, 'target_observations_used_for_features': False,
        'physiology_export_implies_scientific_acceptance': physiology['scientific_accepted']}
    return features, audit


def locked_target_gate(store, computation):
    prediction_ref = computation['prediction']; receipt_ref = computation['completion']
    dense_ref = computation['dense_endpoints']; request_ref = computation['request']
    store.ref(prediction_ref)
    receipt, dense, request = store.ref(receipt_ref), store.ref(dense_ref), store.ref(request_ref)
    folder = store.path(prediction_ref['path']).parent
    if store.path(receipt_ref['path']) != folder / 'completion.json' or store.path(dense_ref['path']) != folder / 'endpoints.json':
        raise ValueError('Target gate is not bound to the locked prediction')
    verify_pins(store, receipt['artifact_pins'])
    for reference in (prediction_ref, dense_ref, request_ref):
        if receipt['artifact_pins'].get(reference['path']) != reference['sha256']:
            raise ValueError('Target prediction/gate/request disagrees with its completion receipt')
    construction = receipt['construction']
    if construction['mapping'] != computation['mapping'] or construction['request'] != request:
        raise ValueError('Target completion is bound to another mapping or request')
    if (dense.get('endpoint_estimator_uses_observations') is not False
            or dense.get('comparison_observations_used') is not False):
        raise ValueError('Dense warning uses target observations')
    return {endpoint: dense['endpoints'][endpoint]['status'] != 'completed' for endpoint in risk.ENDPOINT_FAMILY}


def collect_records(store, protocol, state, result):
    """One solver replay per scientific computation; aliases retain explicit IDs."""
    cases = {c['case_id']: c for c in protocol['cases']}
    expected = {(u['case_id'], u['method'], u['scenario']) for u in protocol['units']}
    expected_computations = {(u['case_id'], u['method'], u['scenario']): u['computation_id'] for u in protocol['units']}
    actual = [(r['case_id'], r['method'], r['scenario']) for r in result['units']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Result comparison matrix differs from the protocol')
    replay, gates, records, excluded = {}, {}, [], []
    for row in result['units']:
        cid = row['computation_id']; comp = state['computations'][cid]
        if cid != expected_computations[row['case_id'], row['method'], row['scenario']]:
            raise ValueError('Scientific computation alias changed after preparation')
        if row['mapping_source'] != comp.get('mapping') or row['prediction_source'] != comp.get('prediction'):
            raise ValueError('Scored result is bound to another mapping/prediction')
        if comp['status'] != 'predicted':
            excluded.append({'case_id': row['case_id'], 'method': row['method'], 'scenario': row['scenario'],
                'reason': 'scientific_prediction_failed', 'requested_endpoints': list(risk.ENDPOINT_FAMILY)})
            continue
        if cid not in replay:
            features, audit = replay_development(store, comp['mapping'])
            replay[cid] = {'features': features, 'audit': audit}
            gates[cid] = locked_target_gate(store, comp)
        case = cases[row['case_id']]
        for endpoint in risk.ENDPOINT_FAMILY:
            score = row['endpoints'].get(endpoint, {})
            record = {'case_id': row['case_id'], 'drug': case['drug'], 'cluster_id': case['cluster_id'],
                'sampling': 'full', 'method': row['method'], 'scenario': row['scenario'], 'endpoint': endpoint,
                'computation_id': cid, 'features': replay[cid]['features'][endpoint],
                'dense_warning': gates[cid][endpoint], 'scientific_inputs_changed': row['scientific_inputs_changed'],
                'matched_prediction_weak_tail': bool(score.get('weak_tail'))}
            error = score.get('error'); reference = score.get('reference')
            if (score.get('error_metric') != 'fold_error' or not risk._positive(reference)
                    or not risk._finite(error) or error < 1):
                excluded.append({**record, 'reason': 'missing_or_invalid_clinical_reference_or_matched_error'})
            else:
                records.append({**record, 'matched_fold_error': error})
    return records, excluded, replay


def fixed_fold_predictions(evaluation, record):
    """Apply an available-input outer model to its same held-out study only."""
    fold = next(f for f in evaluation['folds'] if f['held_out_cluster'] == record['cluster_id'])
    nested = next(f for f in evaluation['nested_rmse_policy']['folds'] if f['held_out_cluster'] == record['cluster_id'])
    output = {'training_prevalence': fold['smoothed_training_prevalence'],
        'dense_gate': float(record['dense_warning']), 'always_warn': 1.}
    for name, fitted in fold['models'].items():
        if fold['fallback']:
            probability = fold['smoothed_training_prevalence']
        else:
            raw = np.asarray([record['features'][f] if record['features'][f] is not None else m
                for f, m in zip(fitted['features'], fitted['training_weighted_medians'])], float)
            x = (raw - fitted['training_weighted_means']) / fitted['training_weighted_scales']
            if fitted['endpoint_categories']:
                x = np.r_[x, [float(risk.ENDPOINT_FAMILY[record['endpoint']] == category)
                              for category in fitted['endpoint_categories']]]
            z = float(np.dot(x, fitted['coefficients']) + fitted['intercept'])
            probability = float(1 / (1 + np.exp(-np.clip(z, -700, 700))))
        output[name] = probability
    return {'probabilities': output, 'warnings_at_0_35': {name: p >= .35 for name, p in output.items()},
        'nested_warning': True if nested['fallback'] else output['development_rmse_only'] >= nested['threshold'],
        'nested_threshold': nested['threshold'], 'nested_fallback': nested['fallback'],
        'training_clusters': fold['training_clusters']}


def scenario_diagnostics(records, evaluations, excluded=()):
    feature_records = records + [r for r in excluded if 'features' in r and 'endpoint' in r]
    available = {(r['method'], r['case_id'], r['endpoint']): r for r in feature_records if r['scenario'] == 'available'}
    output = []
    for row in feature_records:
        if row['scenario'] == 'available':
            continue
        baseline = available.get((row['method'], row['case_id'], row['endpoint']))
        evaluation = evaluations[row['method']]
        has_fold = any(f['held_out_cluster'] == row['cluster_id'] for f in evaluation.get('folds', []))
        initial = fixed_fold_predictions(evaluation, baseline) if has_fold and baseline is not None else None
        changed = fixed_fold_predictions(evaluation, row) if has_fold else None
        output.append({**row, 'available_features': None if baseline is None else baseline['features'], 'available_scores': initial,
            'missing_input_scores': changed,
            'paired_feature_diagnostic_status': 'compared' if initial is not None and changed is not None else
                'available_features_missing' if baseline is None else 'available_outer_fold_missing',
            'clinical_reference_pair_complete': baseline is not None and 'matched_fold_error' in baseline and 'matched_fold_error' in row,
            'available_reference_missing_reason': 'available_prediction_missing' if baseline is None else baseline.get('reason'),
            'probability_change': None if initial is None or changed is None else
                {k: changed['probabilities'][k] - initial['probabilities'][k] for k in initial['probabilities']},
            'interpretation': 'Paired input-perturbation diagnostic from a fixed outer model; not another independent validation sample.'})
    return output


def analyze(run, *, remove_historical=True):
    run = batch.own_run(run)
    protocol, state = batch.load(run, locked=True)
    if state['status'] != 'complete':
        raise ValueError('All fits, predictions and results must be complete before analysis')
    if batch.paper_pins() != protocol['paper_guard']:
        raise ValueError('Publication files changed or their file set changed')
    output_path = run / 'analysis.json'
    store = Store(ROOT)
    store.read(run / 'state.json'); store.ref(state['protocol'])
    result = store.ref(state['results'])
    for pins in (protocol['source_pins'], protocol['paper_guard'], result['source_pins']):
        verify_pins(store, pins)
    store.verify_refs([state['fits'], state['computations'], state['formula_selection']])
    own_code = 'scripts/analyze_robust_blueprint.py'
    store.checksum(own_code)
    store.checksum('src/pbpk_kb/blueprint_policy/predictive_risk.py', RISK_SHA256)
    store.checksum('tests/test_blueprints/test_predictive_risk.py')
    store.checksum('tests/test_blueprints/test_robust_analysis.py')
    if output_path.exists():
        previous = json.loads(output_path.read_text())
        if previous['version'] != VERSION or previous['config'] != CONFIG:
            raise ValueError('An existing analysis must not be overwritten by a different protocol')
        verify_pins(store, previous['source_pins'])
        if remove_historical and (ROOT / HISTORICAL).exists():
            if sha256(ROOT / HISTORICAL) != HISTORICAL_SHA256:
                raise ValueError('Historical cache changed before its authorized removal')
            (ROOT / HISTORICAL).unlink()
        return previous
    history_path = ROOT / HISTORICAL
    if sha256(history_path) != HISTORICAL_SHA256:
        raise ValueError('Historical risk result changed')
    historical = json.loads(history_path.read_text())
    verify_pins(store, historical['source_pins'])
    rf_provenance = audit_rf_provenance(store, protocol['cases'])
    records, excluded, replay = collect_records(store, protocol, state, result)
    evaluations = {}
    for method in batch.METHODS:
        selected = [r for r in records if r['method'] == method and r['scenario'] == 'available']
        scored_cases = {r['case_id'] for r in selected}; scored_clusters = {r['cluster_id'] for r in selected}
        expected_cases = {c['case_id'] for c in protocol['cases']}
        expected_clusters = {c['cluster_id'] for c in protocol['cases']}
        denominator = {'requested_drugs': len(expected_cases), 'scored_drugs': len(scored_cases),
            'requested_study_clusters': len(expected_clusters), 'scored_study_clusters': len(scored_clusters),
            'requested_endpoints': len(expected_cases) * len(risk.ENDPOINT_FAMILY), 'scored_endpoints': len(selected),
            'unscored_endpoints': len(expected_cases) * len(risk.ENDPOINT_FAMILY) - len(selected),
            'unscored_drugs': sorted(expected_cases - scored_cases), 'unscored_study_clusters': sorted(expected_clusters - scored_clusters),
            'denominator_loss': bool(expected_cases - scored_cases or expected_clusters - scored_clusters)}
        if len(scored_clusters) < 2:
            evaluation = {'status': 'insufficient_labeled_study_clusters', 'records': selected, 'folds': [], 'metrics': None}
        else:
            evaluation = risk.evaluate_records(selected)
            evaluation['status'] = 'cross_validated_with_denominator_loss' if denominator['denominator_loss'] else 'cross_validated'
        evaluations[method] = {**evaluation, 'denominators': denominator}
    diagnostic = scenario_diagnostics(records, evaluations, excluded)
    document = {'version': VERSION, 'config': deepcopy(CONFIG), 'protocol': state['protocol'],
        'results_source': state['results'], 'primary_risk': evaluations,
        'rf_provenance': rf_provenance,
        'scenario_diagnostics': diagnostic, 'excluded': excluded,
        'replay_audit': {cid: item['audit'] for cid, item in replay.items()},
        'cost': {'unique_development_replays': len(replay), 'linear_solver_calls': 2 * len(replay),
                 'native_calls': 0, 'new_fits': 0, 'curve_files_written': 0},
        'historical_risk': {'original_path': HISTORICAL, 'original_sha256': HISTORICAL_SHA256,
            'scope': CONFIG['historical'], 'result': historical, 'standalone_cache_removed_after_embedding': remove_historical},
        'source_pins': dict(store.pins), 'paper_guard_verified': True, 'paper_updated': False,
        'limitations': [CONFIG['interpretation'],
            'Dense numerical gate and matched clinical endpoint errors use different estimation windows.',
            'The former full/five risk model and nested attempt remain visible, including unfavorable results.',
            'Neither good discrimination nor an inner recall target guarantees clinical safety.']}
    store.unchanged()
    if batch.paper_pins() != protocol['paper_guard']:
        raise ValueError('Publication files changed during analysis')
    document = json.loads(json.dumps(document, allow_nan=False))
    save(output_path, document)
    if json.loads(output_path.read_text()) != document:
        raise ValueError('Analysis serialization failed')
    if remove_historical:
        if sha256(history_path) != HISTORICAL_SHA256:
            raise ValueError('Historical cache changed before its authorized removal')
        history_path.unlink()
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=batch.DEFAULT)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    print(json.dumps({'analysis': str((batch.own_run(args.run_dir) / 'analysis.json').relative_to(ROOT)),
        'cost': result['cost'], 'primary_records': {m: len(v['records']) for m, v in result['primary_risk'].items()},
        'historical_cache_embedded': True}, indent=2))


if __name__ == '__main__':
    main()
