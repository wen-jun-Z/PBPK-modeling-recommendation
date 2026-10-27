#!/usr/bin/env python3
"""Fixed-threshold risk audit for the separate tail-refinement experiment."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprint_policy import robust_exchange_calibration as baseline_engine
from pbpk_kb.blueprint_policy import tail_exchange_calibration as tail_engine
from pbpk_kb.blueprints.source_workflow import save, sha256
from report_blueprint_validation import Store
from run_tail_blueprint import create_runner

VERSION = 'tail-blueprint-fixed-risk-analysis-1'
RISK_SHA256 = '9051ae4b1109a7ef34a41e9c3b1cabd65911ebaacc30dae0fdb40b797140bb17'
PREVIOUS_ANALYSIS_SHA256 = '0aea157bcd7252223770e5e8bc19b5cb93c9b76e0a2f8b83b0ee702b3bf0d478'
CONFIG = {'primary': 'Three methods, available inputs, full development observations; whole-study-cluster leave-one-out.',
    'baseline': 'current is the previous evidence_flexible, not the original fixed-prior flow_exchange.',
    'risk': 'Unchanged three-feature L2 logistic, train-only weighted preprocessing and all original fixed-threshold comparators.',
    'thresholds': [.35, .5, .8], 'main_threshold': .35,
    'nested_threshold_selection': 'Not run in this batch. Prior negative nested results remain in the linked previous analysis.',
    'missing_inputs': 'Fixed available-input outer models applied to scenario features; no new training on aliases.',
    'replay': 'Source-pinned selected development physiology; linear solver only; verify unweighted development lnRMSE.',
    'interpretation': 'Previously inspected ten-drug development refinement, not blind clinical qualification.'}


def _private_module(path, label):
    spec = importlib.util.spec_from_file_location('_tail_analysis_' + label + '_' + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReplayEngines:
    """Stateless routing adapter for the frozen collector's replay interface."""
    PARAMETERS = {**baseline_engine.PARAMETERS, **tail_engine.PARAMETERS}

    @staticmethod
    def engine(variant):
        if variant in tail_engine.PARAMETERS:
            return tail_engine
        if variant in baseline_engine.PARAMETERS:
            return baseline_engine
        raise ValueError('Unsupported replay variant')

    @staticmethod
    def _validated(mapping, path, root):
        return ReplayEngines.engine(mapping['variant'])._validated(mapping, path, root)

    @staticmethod
    def _simulate(physiology, request, times, variant, keys, parameters):
        return ReplayEngines.engine(variant)._simulate(physiology, request, times, variant, keys, parameters)


def analysis_helpers():
    """Reuse frozen functions in isolated namespaces, leaving imports unchanged."""
    collector = _private_module(ROOT / 'scripts/analyze_robust_blueprint.py', 'collector')
    original_import = collector.importlib.import_module
    collector.importlib = SimpleNamespace(import_module=lambda name: ReplayEngines
        if name == 'pbpk_kb.blueprint_policy.robust_exchange_calibration' else original_import(name))
    model = _private_module(ROOT / 'src/pbpk_kb/blueprint_policy/predictive_risk.py', 'fixed_risk')
    # evaluate_records performs the unchanged fixed-probability cross-fit first.
    # Only its extra nested-threshold experiment is disabled in this private
    # module. No weights, regularization, features, bins or thresholds change.
    model.evaluate_nested_rmse_policy = lambda records: {'status': 'not_run_in_this_batch'}
    return collector, model


def fixed_risk_evaluation(model, records):
    result = model.evaluate_records(records)
    if result.pop('nested_rmse_policy') != {'status': 'not_run_in_this_batch'}:
        raise ValueError('An undeclared threshold-selection experiment was executed')
    result['nested_threshold_selection_performed'] = False
    return result


def scenario_diagnostics(collector, records, excluded, evaluations):
    """Reuse the frozen scenario scorer while expressly disabling nested output."""
    adapted = deepcopy(evaluations)
    for evaluation in adapted.values():
        evaluation['nested_rmse_policy'] = {'folds': [
            {'held_out_cluster': f['held_out_cluster'], 'fallback': 'not_run_in_this_batch', 'threshold': None}
            for f in evaluation.get('folds', [])]}
    rows = collector.scenario_diagnostics(records, adapted, excluded)
    for row in rows:
        for name in ('available_scores', 'missing_input_scores'):
            if row[name] is not None:
                for key in ('nested_warning', 'nested_threshold', 'nested_fallback'):
                    row[name].pop(key)
    return rows


def actual_costs(store, protocol, state):
    counts = Counter(); fit_cost = {}
    for identity, slot in state['fits'].items():
        mapping = store.ref(slot['mapping'])
        path = store.path(slot['mapping']['path'])
        tail_engine._validated(mapping, path, store.root)
        for field in ('source_pins', 'code_pins', 'retained_artifact_pins'):
            for name, digest in mapping[field].items():
                store.checksum(name, digest)
        counts['full_formula_slots_locked'] += 1
        counts['full_formula_slots_fitted' if mapping['eligible'] else 'full_formula_slots_failed'] += 1
        fit_cost[identity] = mapping['total_native_calls'], mapping['linear_solver_calls']
        for name, stage in mapping['stages'].items():
            kind = 'full' if name == 'full' else 'cv'
            if stage['status'] != 'complete':
                raise ValueError('An optimization stage is not complete')
            counts[kind + '_stages_completed'] += 1
            counts[kind + '_optimizer_starts_recorded'] += len(stage['starts'])
            if kind == 'cv':
                counts['cv_stages_without_score'] += int(stage['heldout_score'] is None)
        counts['full_stages_skipped_after_cv_failure'] += int('full' not in mapping['stages'])
        counts['fits_with_uncertain_cost'] += int(not mapping.get('cost_count_exact', True))
    for cid in protocol['computations']:
        computation = state['computations'][cid]; resources = computation['resources']
        native_fit = sum(fit_cost[cid + '__' + formula][0] for formula in protocol['formulas'])
        linear_fit = sum(fit_cost[cid + '__' + formula][1] for formula in protocol['formulas'])
        if resources['total_native_calls'] < native_fit or resources['linear_solver_calls'] < linear_fit:
            raise ValueError('Prediction receipt undercounts its formula-fit costs')
        counts['new_native_calls'] += resources['total_native_calls']
        counts['new_linear_solver_calls'] += resources['linear_solver_calls']
        counts['new_target_native_calls'] += resources['total_native_calls'] - native_fit
        counts['new_target_linear_solver_calls'] += resources['linear_solver_calls'] - linear_fit
        counts['predictions_with_unknown_prior_incomplete_cost'] += int(resources.get('prior_incomplete_costs_unknown', False))
    counts['new_fit_native_calls'] = sum(x[0] for x in fit_cost.values())
    counts['new_fit_linear_solver_calls'] = sum(x[1] for x in fit_cost.values())
    return {'planned': protocol['optimization_budget'], 'actual': dict(counts),
        'reused_predictions': len(protocol['cached_ids']),
        'interpretation': 'Each unique computation counted once; interrupted unknown costs remain flagged. Pure analysis replay costs are separate.'}


def analyze(run):
    runner = create_runner(); run = runner.own_run(run)
    protocol, state = runner.load(run, locked=True)
    if state['status'] != 'complete':
        raise ValueError('The full tail batch must be complete before risk analysis')
    store = Store(ROOT)
    store.read(run / 'state.json'); store.ref(state['protocol'])
    result = store.ref(state['results'])
    for pins in (protocol['source_pins'], protocol['paper_guard'], result['source_pins']):
        for name, digest in pins.items():
            store.checksum(name, digest)
    store.verify_refs([state['fits'], state['computations'], state['formula_selection']])
    for path in ('scripts/analyze_tail_blueprint.py', 'scripts/analyze_robust_blueprint.py',
                 'tests/test_blueprints/test_tail_analysis.py'):
        store.checksum(path)
    store.checksum('src/pbpk_kb/blueprint_policy/predictive_risk.py', RISK_SHA256)
    output_path = run / 'analysis.json'
    if output_path.exists():
        previous = json.loads(output_path.read_text())
        if previous['version'] != VERSION or previous['config'] != CONFIG:
            raise ValueError('A completed analysis cannot be overwritten by a changed protocol')
        for name, digest in previous['source_pins'].items():
            store.checksum(name, digest)
        return previous
    historical_path = store.path(protocol['source_protocol']['path']).parent / 'analysis.json'
    historical = store.read(historical_path, PREVIOUS_ANALYSIS_SHA256)
    if historical['protocol'] != protocol['source_protocol']:
        raise ValueError('Historical analysis belongs to another source experiment')
    for name, digest in historical['source_pins'].items():
        store.checksum(name, digest)
    collector, model = analysis_helpers()
    rf_provenance = collector.audit_rf_provenance(store, protocol['cases'])
    costs = actual_costs(store, protocol, state)
    records, excluded, replay = collector.collect_records(store, protocol, state, result)
    evaluations = {}
    for method in protocol['methods']:
        selected = [r for r in records if r['method'] == method and r['scenario'] == 'available']
        case_ids = {r['case_id'] for r in selected}; clusters = {r['cluster_id'] for r in selected}
        expected_cases = {c['case_id'] for c in protocol['cases']}; expected_clusters = {c['cluster_id'] for c in protocol['cases']}
        denominators = {'requested_drugs': len(expected_cases), 'scored_drugs': len(case_ids),
            'requested_study_clusters': len(expected_clusters), 'scored_study_clusters': len(clusters),
            'requested_endpoints': 6 * len(expected_cases), 'scored_endpoints': len(selected),
            'unscored_endpoints': 6 * len(expected_cases) - len(selected),
            'unscored_drugs': sorted(expected_cases - case_ids), 'unscored_study_clusters': sorted(expected_clusters - clusters)}
        if len(clusters) < 2:
            evaluation = {'status': 'insufficient_labeled_study_clusters', 'records': selected, 'folds': [], 'metrics': None}
        else:
            evaluation = fixed_risk_evaluation(model, selected)
            evaluation['status'] = 'cross_validated_with_denominator_loss' if expected_cases - case_ids else 'cross_validated'
        evaluations[method] = {**evaluation, 'denominators': denominators}
    # The current method is exactly reused evidence_flexible; any difference in
    # its fixed-risk scores indicates a collector/weighting/provenance mistake.
    control = historical['primary_risk']['evidence_flexible']
    if evaluations['current']['metrics'] != control['metrics']:
        raise ValueError('Reused baseline fixed-risk scores changed')
    diagnostics = scenario_diagnostics(collector, records, excluded, evaluations)
    document = {'version': VERSION, 'config': deepcopy(CONFIG), 'protocol': state['protocol'],
        'results_source': state['results'], 'primary_risk': evaluations, 'scenario_diagnostics': diagnostics,
        'excluded': excluded, 'replay_audit': {cid: item['audit'] for cid, item in replay.items()},
        'cost': costs, 'analysis_replay_cost': {'unique_computations': len(replay), 'linear_solver_calls': 2 * len(replay),
            'native_calls': 0, 'new_fits': 0, 'curve_files_written': 0}, 'rf_provenance': rf_provenance,
        'historical_analysis': {'path': str(historical_path.relative_to(ROOT)), 'sha256': PREVIOUS_ANALYSIS_SHA256},
        'source_pins': dict(store.pins), 'paper_guard_verified': True, 'paper_updated': False,
        'limitations': [CONFIG['interpretation'], 'Nine research clusters; correlated endpoint aliases are not independent samples.',
            'Target scalar FE labels use protocol-matched NCA; dense numerical warnings use a different window.',
            'Neither fixed warning probabilities nor previous nested recall targets establish clinical safety.']}
    store.unchanged()
    if runner.paper_pins() != protocol['paper_guard']:
        raise ValueError('Publication files changed during analysis')
    document = json.loads(json.dumps(document, allow_nan=False))
    save(output_path, document)
    if json.loads(output_path.read_text()) != document:
        raise ValueError('Risk analysis serialization changed')
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'experiments/blueprint_design/robust_structure_tail_20260930')
    args = parser.parse_args(); result = analyze(args.run_dir)
    print(json.dumps({'status': 'analyzed', 'primary_records': {m: len(v['records']) for m, v in result['primary_risk'].items()},
        'cost': result['cost'], 'analysis_replay_cost': result['analysis_replay_cost']}, indent=2))


if __name__ == '__main__':
    main()
