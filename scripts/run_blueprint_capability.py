#!/usr/bin/env python3
"""Five scoped capability checks of the frozen vascular–tissue blueprint.

Evidence faults are engineered interventions, not clinical observations. Only
development inputs drive calibration; every new prediction locks before the
original clinical outcomes are decoded for scoring.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprint_policy.evidence_decision import decide
from run_blueprint_learning import Controller, Paused, now, read, ref, checked, verify
import run_blueprint_evidence_policy as policy_inputs
import run_blueprint_transport as transport_runner

VERSION = 'blueprint-capability-1'
DEFAULT = ROOT / 'experiments/blueprint_design/blueprint_capability_20260930'
BASE = ROOT / 'experiments/blueprint_design/vascular_tissue_ten_drug_20260930'
POLICIES = ('evidence_policy', 'fixed_template', 'rf_fill', 'no_conflict_guard')
SCENARIOS = policy_inputs.SCENARIOS
SAMPLING = ('full', 'five')
FORMULAS = ('PKSimStandard', 'RodgersRowland')
AXES = ('clearance_scale', 'partition_scale', 'bidirectional_exchange_scale')
ENGINES = {'always_three': 'pbpk_kb.blueprint_policy.transport_calibration',
           'adaptive_axes': 'pbpk_kb.blueprint_policy.capability_calibration'}
PLASMA_ENDPOINTS = ('plasma_concentration_time', 'auc', 'cmax', 'tmax', 'c_last',
    'terminal_half_life', 'auc_inf', 'clearance', 'vz', 'mrt', 'vss')
# Independent engineering expectations, fixed before simulation. Clinical
# quality is scored separately and cannot alter this requested safety policy.
EXPECTED_READY = {'available': True, 'missing_fu': True, 'wrong_species_fu': True,
    'conflicting_fu': False, 'missing_required_pbpk_formula': False, 'unsupported_ddi': False}
api = transport_runner.create_runner()


def own_run(path):
    raw = Path(path) if Path(path).is_absolute() else ROOT / path
    result = inside(raw, ROOT)
    if raw.is_symlink() or result.parent != DEFAULT.parent or not result.name.startswith('blueprint_capability_'):
        raise ValueError('Use an immediate blueprint_capability_ project experiment directory')
    return result


def module(name):
    return importlib.import_module(ENGINES[name])


def flow_decision(request, fallback, *, purpose=None, method='evidence_policy'):
    result = decide(request, fallback, purpose=purpose, method=method)
    legacy = result.pop('calibration')
    result['source_policy_calibration_metadata'] = legacy
    result['calibration'] = {'adapter_version': VERSION, 'model_version': 'linear-finite-exchange-model-1',
        'active_parameters': list(AXES), 'fraction_unbound_role': 'fixed_selected_construction_input',
        'protect_net_renal': legacy['protect_net_renal'],
        'clearance_scale_scope': legacy['clearance_scale_scope'],
        'note': 'Evidence selection is reused; the old lipophilicity calibration is not executed.'}
    return result


def values(request):
    """Scientific evidence identity; source spelling/order is not a model feature."""
    return sorted([policy_inputs.visible_record(e) for e in request['evidence']], key=lambda e: e['parameter'])


def signature(request, target, training, engine):
    return policy_inputs.digest({'engine': engine, 'values': values(request), 'human': request['human'],
        'dosing': request['dosing'], 'target': request['target'],
        'target_human': target['human'], 'target_dosing': target['dosing'], 'target_endpoints': target['target'],
        'observed_development': policy_inputs.clean_training(training),
        'optimizer_budget': {'starts': 16, 'max_nfev_per_start': 400}, 'formulas': FORMULAS})


def pin_all(value, pins):
    api.pin_refs(value, pins)


def fit_key(compid, formula):
    return compid + '__' + formula


def selected_values(decision):
    return sorted([(e['parameter'], e['value'], e['unit']) for e in decision['selected_evidence']], key=lambda x: x[0])


def prepare(run, workers=14):
    from pbpk_kb.blueprint_policy.capability_evaluation import CONFIG as TAIL_GATE_CONFIG
    run = own_run(run)
    if run.exists():
        raise FileExistsError('Preserve existing experiment; resume instead')
    if type(workers) is not int or not 1 <= workers <= 14:
        raise ValueError('Use 1–14 native workers')
    original, completed = read(BASE / 'protocol.json'), read(BASE / 'state.json')
    if completed['status'] != 'complete' or completed['protocol']['sha256'] != sha256(BASE / 'protocol.json'):
        raise ValueError('The current ten-drug experiment is not complete and locked')
    verify(original['source_pins'])
    contracts = {}
    pins = dict(original['source_pins'])
    for engine in ENGINES:
        implementation = module(engine)
        contracts[engine] = {'module': ENGINES[engine], 'version': implementation.VERSION,
            'config': deepcopy(implementation.CONFIG), 'code_pins': implementation._code_pins(ROOT)}
        if implementation.CONFIG['starts'] != 16 or implementation.CONFIG['max_evaluations_per_start'] != 400:
            raise ValueError('Both calibrators require the frozen 16×400 budget')
        verify(contracts[engine]['code_pins']); pins.update(contracts[engine]['code_pins'])
    run.mkdir(); (run / 'inputs').mkdir()
    cases = deepcopy(original['cases']); computations, reused, baseline_ids = {}, {}, {}
    decisions, units, audit, permutations, changed_fu = [], [], [], [], []
    for case in cases:
        cid = case['case_id']
        base = checked(case['requests']['evidence']['development'])
        target = checked(case['requests']['evidence']['target'])
        rf = checked(case['requests']['rf']['development'])
        if next(e['value'] for e in base['evidence'] if e['parameter'] == 'fraction_unbound') != next(
                e['value'] for e in rf['evidence'] if e['parameter'] == 'fraction_unbound'):
            changed_fu.append(cid)
        training = {s: checked(original['computations'][cid + '__' + s + '__flow_exchange']['training_source']) for s in SAMPLING}
        training_refs = {s: {**policy_inputs.store_input(run, policy_inputs.clean_training(training[s])),
                            'role': 'calibration_data_reconstruction'} for s in SAMPLING}
        for sampling in SAMPLING:
            compid = 'fixed_' + signature(base, target, training[sampling], 'always_three')[:24]
            result = deepcopy(completed['computations'][cid + '__' + sampling + '__flow_exchange'])
            baseline_folder = (ROOT / result['prediction']['path']).parent
            result['completion'] = ref(baseline_folder / 'completion.json')
            result['dense_endpoints'] = ref(baseline_folder / 'endpoints.json')
            result['original_resources'] = deepcopy(result.get('resources', {}))
            result['resources'] = {'total_native_calls': 0, 'linear_solver_calls': 0, 'wall_seconds': 0, 'cache_reused': True}
            result['reused_from'] = ref(BASE / 'state.json')
            reused[compid] = result
            baseline_ids[cid + '__' + sampling] = compid
            units.append({'id': cid + '__' + sampling + '__always_three', 'case_id': cid,
                'sampling': sampling, 'scenario': 'calibration', 'method': 'always_three',
                'status': 'planned', 'computation_id': compid})
            adaptive = 'adaptive_' + signature(base, target, training[sampling], 'adaptive_axes')[:24]
            computations[adaptive] = {'case_id': cid, 'sampling': sampling, 'engine': 'adaptive_axes',
                'request': case['requests']['evidence']['development'], 'target_request': case['requests']['evidence']['target'],
                'training_source': training_refs[sampling]}
            units.append({'id': cid + '__' + sampling + '__adaptive_axes', 'case_id': cid,
                'sampling': sampling, 'scenario': 'calibration', 'method': 'adaptive_axes',
                'status': 'planned', 'computation_id': adaptive})
        for scenario in SCENARIOS:
            rows, fallback, provenance = policy_inputs.scenario_evidence(base, rf, scenario, cid)
            rows, fallback, view = policy_inputs.bind_view(run, rows, fallback)
            request = policy_inputs.selected_request(run, base, rows)
            purpose = {'endpoints': list(base['target']['endpoints']), 'population_condition': 'reference_adult',
                'mechanism': 'ddi' if scenario == 'unsupported_ddi' else 'parent_plasma'}
            audit.append({'case_id': cid, 'scenario': scenario, 'visible_source': view, **provenance})
            for policy in POLICIES:
                decision = flow_decision(request, fallback, purpose=purpose, method=policy)
                did = '__'.join((cid, scenario, policy))
                decisions.append({'id': did, 'case_id': cid, 'drug': case['drug'], 'scenario': scenario,
                    'method': policy, 'status': decision['status'], 'expected_ready': EXPECTED_READY[scenario],
                    'decision': decision, 'visible_evidence': view})
                if scenario in ('available', 'conflicting_fu'):
                    permuted = deepcopy(request); permuted['evidence'].reverse()
                    alternate = flow_decision(permuted, list(reversed(fallback)), purpose=purpose, method=policy)
                    permutations.append({'case_id': cid, 'scenario': scenario, 'method': policy,
                        'original_status': decision['status'], 'permuted_status': alternate['status'],
                        'status_equal': alternate['status'] == decision['status'],
                        'selected_values_equal': selected_values(alternate) == selected_values(decision),
                        'original_selected': selected_values(decision), 'permuted_selected': selected_values(alternate),
                        'perturbation': 'reverse all primary records and fallback records; no values changed'})
                for sampling in SAMPLING:
                    item = {'id': '__'.join((cid, scenario, sampling, policy)), 'case_id': cid,
                        'sampling': sampling, 'scenario': scenario, 'method': policy, 'decision_id': did,
                        'status': 'planned' if decision['status'] == 'ready' else 'blocked',
                        'reasons': decision['reasons']}
                    if decision['status'] == 'ready':
                        selected = policy_inputs.selected_request(run, base, decision['selected_evidence'])
                        predicted = deepcopy(selected)
                        for field in ('human', 'dosing', 'target'):
                            predicted[field] = deepcopy(target[field])
                        compid = 'fixed_' + signature(selected, predicted, training[sampling], 'always_three')[:24]
                        item['computation_id'] = compid
                        if compid not in reused and compid not in computations:
                            computations[compid] = {'case_id': cid, 'sampling': sampling, 'engine': 'always_three',
                                'request': policy_inputs.store_input(run, selected),
                                'target_request': policy_inputs.store_input(run, predicted), 'training_source': training_refs[sampling]}
                    units.append(item)
    if (len(decisions) != 240 or len(units) != 520 or len(computations) != 38 or len(reused) != 20):
        raise ValueError('Unexpected predeclared logical/deduplicated experiment matrix: ' + str(
            (len(decisions), len(units), len(computations), len(reused))))
    for value in (cases, decisions, computations, reused, audit):
        pin_all(value, pins)
    for path in (Path(__file__), BASE / 'protocol.json', BASE / 'state.json',
            ROOT / 'scripts/run_blueprint_evidence_policy.py', ROOT / 'scripts/run_blueprint_learning.py',
            ROOT / 'scripts/run_blueprint_transport.py', ROOT / 'scripts/report_blueprint_validation.py',
            ROOT / 'src/pbpk_kb/blueprint_policy/evidence_decision.py',
            ROOT / 'src/pbpk_kb/blueprint_policy/capability_evaluation.py'):
        pins[str(path.relative_to(ROOT))] = sha256(path)
    protocol = {'version': VERSION, 'created_at': now(), 'cases': cases, 'decisions': decisions,
        'units': units, 'computations': computations, 'baseline_ids': baseline_ids, 'permutations': permutations,
        'policies': list(POLICIES), 'scenarios': list(SCENARIOS), 'sampling': list(SAMPLING),
        'formulas': list(FORMULAS), 'contracts': contracts, 'source_pins': pins,
        'source_protocol': ref(BASE / 'protocol.json'), 'source_state': ref(BASE / 'state.json'),
        'audit_not_passed_to_methods': audit, 'engineering_expected_ready': EXPECTED_READY,
        'changed_fu_case_ids': changed_fu, 'unchanged_fu_case_ids': [c['case_id'] for c in cases if c['case_id'] not in changed_fu],
        'tail_gate': deepcopy(TAIL_GATE_CONFIG),
        'workers': workers, 'native_threads': 2, 'native_timeout_seconds': 180,
        'monitor_seconds': 1200, 'directory_limit_bytes': 10 * 1024**3,
        'bootstrap': {'resamples': 2000, 'seed': 20260930, 'cluster': 'study_id; Pletz2004 shared'},
        'scope': 'Ten previously inspected development drugs; evidence faults and request negatives are engineered controls.',
        'calibration_rules': {'screening': 'Development-only identity Jacobian; at least 2 points per retained parameter; relative singular threshold 1e-3; CL,K,PS order',
            'fixed_axes': 'three fitted scales', 'excluded_axis_value': 1, 'selection': 'Development objective only; both formulas complete before selection'},
        'replay_rule': {'rtol': 1e-10, 'atol_umol_l': 1e-12, 'source': 'Same model equations, rebuilt trajectory from locked physiology and parameters; no fit'},
        'diagnostic_rule': {'bound_fraction_of_log_range': .01, 'near_optimal_delta_objective': .01,
            'interpretation': 'Local data-Jacobian and start dispersion diagnostics are not proof of global identifiability'},
        'limitations': ['No new blind clinical cohort.', 'Synthetic faults and their replicas are not independent clinical samples.',
            'Blocking establishes a configured safety policy, not accurate prediction or independent novelty.',
            'Ready means admissible input; model-tail gates can still make an endpoint unavailable.',
            'Moxifloxacin PK prior is from the development study; five curve points are not the entire information budget.',
            'RF fallback is pre-existing and excludes the target compound/scaffold; no learner is retrained here.',
            'Missing reported clinical endpoints remain missing; weak clinical NCA estimates remain explicitly flagged.']}
    verify(pins); save(run / 'protocol.json', protocol)
    save(run / 'state.json', {'version': VERSION, 'status': 'prepared', 'protocol': ref(run / 'protocol.json'),
        'fits': {}, 'computations': reused, 'monitor': {}, 'created_at': now()})
    return {'decisions': 240, 'logical_clinical_units': 520, 'new_fits': 76, 'new_predictions': 38, 'reused_baselines': 20}


def load(run, locked=False):
    protocol, state = read(run / 'protocol.json'), read(run / 'state.json')
    if state['protocol']['sha256'] != sha256(run / 'protocol.json') or protocol['version'] != VERSION or state['version'] != VERSION:
        raise ValueError('Experiment protocol changed')
    verify(protocol['source_pins'])
    source = checked(protocol['source_protocol'])
    if protocol['cases'] != source['cases']:
        raise ValueError('Original ten cases or clinical policies changed')
    ids = [c['case_id'] for c in protocol['cases']]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError('Ten distinct original drug cases required')
    expected_decisions = {(cid, s, m) for cid in ids for s in SCENARIOS for m in POLICIES}
    observed_decisions = [(d['case_id'], d['scenario'], d['method']) for d in protocol['decisions']]
    if len(observed_decisions) != len(set(observed_decisions)) or set(observed_decisions) != expected_decisions:
        raise ValueError('Decision matrix has duplicates or missing conditions')
    expected_units = {(cid, s, sampling, m) for cid in ids for s in SCENARIOS for sampling in SAMPLING for m in POLICIES}
    expected_units |= {(cid, 'calibration', sampling, m) for cid in ids for sampling in SAMPLING for m in ('always_three', 'adaptive_axes')}
    observed_units = [(u['case_id'], u['scenario'], u['sampling'], u['method']) for u in protocol['units']]
    if len(observed_units) != len(set(observed_units)) or set(observed_units) != expected_units:
        raise ValueError('Clinical matrix has duplicates or missing conditions')
    if set(protocol['baseline_ids']) != {cid + '__' + sampling for cid in ids for sampling in SAMPLING}:
        raise ValueError('Baseline reuse matrix differs')
    for compid, spec in protocol['computations'].items():
        request, target, training = (checked(spec[k]) for k in ('request', 'target_request', 'training_source'))
        expected_training = checked(source['computations'][spec['case_id'] + '__' + spec['sampling'] + '__flow_exchange']['training_source'])
        if policy_inputs.clean_training(training) != policy_inputs.clean_training(expected_training):
            raise ValueError('Development subset changed or full curve leaked into five-point inputs')
        prefix = 'adaptive_' if spec['engine'] == 'adaptive_axes' else 'fixed_'
        if compid != prefix + signature(request, target, training, spec['engine'])[:24]:
            raise ValueError('Unique computation identity differs from the exact scientific inputs')
    expected = set(protocol['computations']) | set(protocol['baseline_ids'].values())
    if len(protocol['units']) != 520 or len(protocol['decisions']) != 240 or len(expected) != 58:
        raise ValueError('Comparison matrix changed')
    fits = {fit_key(cid, formula) for cid in protocol['computations'] for formula in FORMULAS}
    if not set(state['fits']) <= fits or not set(protocol['baseline_ids'].values()) <= set(state['computations']) <= expected:
        raise ValueError('Unexpected checkpoints')
    pin_all([state['fits'], state['computations']], {})
    if locked:
        if (set(state['fits']) != fits or set(state['computations']) != expected
                or state['status'] not in ('predictions_locked', 'complete')
                or not state.get('all_attempts_locked_before_scoring') or not state.get('monitor_stopped_before_scoring')
                or any(r['status'] not in ('predicted', 'failed') for r in state['computations'].values())):
            raise ValueError('Every fit and prediction must lock before outcomes are decoded')
        choice = checked(state['formula_selection'])
        if set(choice['choices']) != set(protocol['computations']) or choice['basis'] != 'development_objective_only':
            raise ValueError('Formula choices are not complete and development-only')
    return protocol, state


def execute(run):
    run = own_run(run)
    with (run / '.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, state = load(run)
        if state['status'] == 'complete':
            report(run); return {'status': 'complete', 'reused': True}
        for engine, contract in protocol['contracts'].items():
            if module(engine).VERSION != contract['version'] or module(engine).CONFIG != contract['config']:
                raise ValueError('Calibrator contract changed')
        controller = Controller(run, protocol, state); controller.check(force=True)
        monitor = threading.Thread(target=controller.watch, daemon=True); monitor.start()
        controller.persist(status='running', started_at=state.get('started_at', now()), pause_reason=None)
        def fit_one(cid, spec, formula):
            controller.check(); key = fit_key(cid, formula)
            if key in state['fits']:
                checked(state['fits'][key]['mapping']); return
            req = checked(spec['request']); req['partition_method'] = formula
            folder = run / 'work' / cid / formula
            mapping = module(spec['engine']).fit_mapping(reference_request=req, training_source=spec['training_source'],
                output_dir=folder, project_root=ROOT, variant='flow_exchange', before_trial=controller.check)
            api.fit_costs(mapping, 'flow_exchange')
            if mapping['construction_partition_method'] != formula:
                raise ValueError('Fit formula changed')
            with controller.lock:
                state['fits'][key] = {'mapping': mapping['mapping'], 'status': mapping['status'], 'eligible': mapping['eligible'],
                    'native_calls': mapping['total_native_calls'], 'actual_solver_calls': mapping['actual_solver_calls'],
                    'linear_solver_calls': mapping['linear_solver_calls'], 'engine': spec['engine']}
                controller.persist()
        def predict_one(cid, spec):
            controller.check()
            if cid in state['computations']:
                pin_all(state['computations'][cid], {}); return
            choice = checked(state['formula_selection'])['choices'][cid]
            mappings = [checked(state['fits'][fit_key(cid, f)]['mapping']) for f in FORMULAS]
            resources = {'total_native_calls': sum(m['total_native_calls'] for m in mappings),
                'linear_solver_calls': sum(m['linear_solver_calls'] for m in mappings),
                'wall_seconds': sum(m['elapsed_seconds'] for m in mappings), 'cache_reused': False,
                'fit_cost_count_exact': all(m.get('cost_count_exact', True) for m in mappings)}
            if choice is None:
                outcome = {'status': 'failed', 'failure': 'No eligible formula'}
            else:
                request = checked(spec['target_request']); request['partition_method'] = choice['formula']
                folder = run / 'work' / cid / 'target'
                def incomplete(path):
                    return path.exists() and not (path / 'completion.json').exists() and any(p.name != '.calibration.lock' for p in path.iterdir())
                if incomplete(folder):
                    folder = folder.with_name('target_retry_1')
                    if incomplete(folder): raise Paused('Two incomplete prediction attempts retained')
                predicted = module(spec['engine']).predict_mapping(request=request, mapping_path=ROOT / choice['mapping']['path'],
                    output_dir=folder, project_root=ROOT, before_trial=controller.check)
                cost = read(folder / 'result.json')
                resources['total_native_calls'] += cost['actual_new_calls']
                resources['linear_solver_calls'] += cost['linear_solver_calls']
                resources['wall_seconds'] += cost['elapsed_seconds']
                resources['prior_incomplete_costs_unknown'] = folder.name == 'target_retry_1'
                outcome = {'status': predicted['status'], 'request': ref(folder / 'request.json'),
                    'mapping': choice['mapping'], 'selected_formula': choice['formula'], 'engine': spec['engine']}
                if predicted['status'] == 'predicted':
                    outcome['prediction'] = ref(folder / 'prediction_locked.json')
                    outcome['completion'] = ref(folder / 'completion.json')
                    outcome['dense_endpoints'] = ref(folder / 'endpoints.json')
                else: outcome['failure'] = predicted
            outcome['resources'] = resources
            with controller.lock:
                state['computations'][cid] = outcome; controller.persist()
        def phase(items, function):
            with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
                futures = {pool.submit(function, *args): args[0] for args in items}
                for future in as_completed(futures):
                    try: future.result()
                    except Exception as exc:
                        with controller.lock:
                            if not controller.halted.is_set():
                                state.setdefault('interruptions', []).append({'unit': futures[future], 'error': str(exc), 'time': now()})
                                controller.pause(str(exc))
            controller.check(force=True)
        try:
            items = sorted(protocol['computations'].items())
            phase([(cid, spec, formula) for cid, spec in items for formula in FORMULAS], fit_one)
            choices = {}
            for cid, spec in items:
                mappings = [checked(state['fits'][fit_key(cid, f)]['mapping']) for f in FORMULAS]
                best = api.choose_formula(mappings)
                choices[cid] = None if best is None else {'formula': best['construction_partition_method'],
                    'mapping': state['fits'][fit_key(cid, best['construction_partition_method'])]['mapping']}
            document = {'version': VERSION, 'basis': 'development_objective_only', 'choices': choices}
            locked_path = run / 'formulas_locked.json'
            if locked_path.exists() and read(locked_path) != document: raise ValueError('Formula choice changed')
            save(locked_path, document)
            controller.persist(status='predicting', formula_selection=ref(locked_path), formulas_locked_before_target_prediction=True)
            phase(items, predict_one)
            controller.closed.set(); monitor.join(timeout=30)
            if monitor.is_alive(): raise Paused('Monitor has not stopped')
            controller.persist(status='predictions_locked', all_attempts_locked_before_scoring=True,
                monitor_stopped_before_scoring=True, locked_at=now())
            result = report(run)
            controller.persist(status='complete', completed_at=now(), results=ref(run / 'results.json'))
            return {'status': 'complete', 'units': len(result['units'])}
        except Exception as exc:
            if not controller.halted.is_set(): controller.pause(str(exc))
            raise
        finally:
            controller.closed.set(); monitor.join(timeout=30)


def replay_and_protection(protocol, state):
    import numpy as np
    from pbpk_kb.blueprint_policy.perfusion_export import export_physiology
    from pbpk_kb.blueprint_policy import transport_calibration as engine
    from pbpk_kb.blueprint_policy.permeability_model import build_system
    replay, protection, pins = [], [], {}
    cases = {c['case_id']: c for c in protocol['cases']}
    entries = [(key, compid, 'always_three') for key, compid in sorted(protocol['baseline_ids'].items())]
    entries.extend((u['case_id'] + '__' + u['sampling'], u['computation_id'], 'adaptive_axes')
        for u in protocol.get('units', []) if u['scenario'] == 'calibration' and u['method'] == 'adaptive_axes')
    for key, compid, method in entries:
        cid, sampling = key.rsplit('__', 1); case = cases[cid]
        item = state['computations'][compid]
        if item['status'] != 'predicted':
            source_request = checked(case['requests']['evidence']['target'])
            for parameter in ('fraction_unbound', 'renal_clearance'):
                source = next((e for e in source_request['evidence'] if e['parameter'] == parameter and e['basis'] in ('measured', 'derived')), None)
                if source:
                    protection.append({'case_id': cid, 'drug': case['drug'], 'sampling': sampling, 'method': method,
                        'parameter': parameter, 'passed': False, 'status': 'prediction_failed', 'source': source['source']})
            replay.append({'case_id': cid, 'sampling': sampling, 'method': method,
                'passed': False, 'status': 'prediction_failed', 'new_linear_solver_calls': 0})
            continue
        request = checked(item['request']); mapping = checked(item['mapping'])
        original = checked(item['prediction']); folder = (ROOT / item['prediction']['path']).parent
        manifest = folder / 'base/native/execution_manifest.json'
        physiology = export_physiology(execution_manifest_path=manifest, source_manifest_sha256=sha256(manifest),
            output_dir=folder / 'physiology', project_root=ROOT)
        pin_all(physiology, pins)
        keys = engine.PARAMETERS['flow_exchange']
        times = np.array(original['time_min']) / 60
        started = time.perf_counter()
        prediction = engine._simulate(physiology, request, times, 'flow_exchange', keys, mapping['decision']['log_parameters'])
        mw = next(e['value'] for e in request['evidence'] if e['parameter'] == 'molecular_weight')
        p = np.asarray(prediction['concentration_mg_l']) * 1000 / mw; old = np.array(original['concentration'])
        absolute = float(np.max(np.abs(p - old)))
        relative = float(np.max(np.abs(p - old) / np.maximum(np.abs(old), 1e-12)))
        replay.append({'case_id': cid, 'drug': case['drug'], 'sampling': sampling, 'method': method, 'prediction_source': item['prediction'],
            'mapping_source': item['mapping'], 'native_readback_verified': True,
            'passed': bool(np.allclose(p, old, rtol=1e-10, atol=1e-12)), 'max_absolute_difference_umol_l': absolute,
            'max_relative_difference': relative, 'points': len(p), 'wall_seconds': time.perf_counter() - started,
            'new_native_calls': 0, 'new_linear_solver_calls': 1, 'replayed_array_sha256': hashlib.sha256(p.tobytes()).hexdigest(),
            'mass_balance_relative_error': prediction['mass_balance_relative_error']})
        cl = engine.clearance_inputs(request)
        system = build_system(physiology, total_clearance_l_h=cl['total_clearance_l_h'],
            renal_clearance_l_h=cl['renal_clearance_l_h'], family='exchange',
            log_parameters=dict(zip(keys, mapping['decision']['log_parameters'])), observation='auto')
        fu = next(e for e in request['evidence'] if e['parameter'] == 'fraction_unbound')
        if fu['basis'] in ('measured', 'derived'):
            protection.append({'case_id': cid, 'drug': case['drug'], 'sampling': sampling, 'method': method, 'parameter': 'fraction_unbound',
                'source_value': fu['value'], 'executed_value': physiology['fraction_unbound_plasma'], 'unit': 'fraction',
                'passed': math.isclose(fu['value'], physiology['fraction_unbound_plasma'], rel_tol=1e-10, abs_tol=1e-12),
                'source': fu['source'], 'readback': physiology['manifest']})
        if cl['renal_clearance_l_h'] is not None:
            actual = system.metadata['clearance_components_l_h']['qualified_renal_fixed']
            scale = mapping['decision']['parameters']['clearance_scale']
            protection.append({'case_id': cid, 'drug': case['drug'], 'sampling': sampling, 'method': method, 'parameter': 'renal_clearance',
                'source_value': cl['renal_clearance_l_h'], 'executed_value': actual, 'unit': 'L/h',
                'nonrenal_scale': scale, 'effective_total_clearance_l_h': system.metadata['effective_total_clearance_l_h'],
                'passed': math.isclose(actual, cl['renal_clearance_l_h'], rel_tol=1e-12) and math.isclose(
                    system.metadata['effective_total_clearance_l_h'], actual + (cl['total_clearance_l_h'] - actual) * scale, rel_tol=1e-12),
                'readback': physiology['manifest'], 'interpretation': 'Fixed engineering construction input, not verified target-patient renal CL.'})
    return replay, protection, pins


def applicability(protocol, state):
    checks = []
    for case in protocol['cases']:
        base = checked(case['requests']['evidence']['development'])
        rf = checked(case['requests']['rf']['development'])
        fallback = [r for r in rf['evidence'] if r['parameter'] == 'fraction_unbound']
        for endpoint in PLASMA_ENDPOINTS:
            request = deepcopy(base); request['target']['endpoints'] = [endpoint]; request['target']['primary_endpoint'] = endpoint
            outcome = flow_decision(request, fallback, purpose={'endpoints': [endpoint]})
            baseline = flow_decision(request, fallback, purpose={'endpoints': [endpoint]}, method='fixed_template')
            checks.append({'case_id': case['case_id'], 'request_type': endpoint, 'expected_supported': True,
                'supported': outcome['status'] == 'ready', 'reasons': outcome['reasons'],
                'fixed_gate_baseline_supported': baseline['status'] == 'ready', 'fixed_gate_baseline_reasons': baseline['reasons'],
                'interpretation': 'Admissible request only; finite prediction and model-tail quality must still be checked.'})
        for category in ('ddi', 'organ', 'oral', 'repeated', 'pediatric'):
            request = deepcopy(base); purpose = {'endpoints': ['plasma_concentration_time']}
            request['target']['endpoints'] = ['plasma_concentration_time']; request['target']['primary_endpoint'] = 'plasma_concentration_time'
            if category == 'ddi': purpose['mechanism'] = 'ddi'
            elif category == 'organ': purpose['endpoints'] = ['liver_concentration_time']
            elif category == 'oral': request['dosing']['route'] = 'oral'
            elif category == 'repeated': purpose['dosing_regimen'] = 'repeated_dose'
            else: purpose['population_condition'] = 'pediatric'; request['human']['age_years'] = 12.
            outcome = flow_decision(request, fallback, purpose=purpose)
            baseline = flow_decision(request, fallback, purpose=purpose, method='fixed_template')
            checks.append({'case_id': case['case_id'], 'request_type': category, 'expected_supported': False,
                'supported': outcome['status'] == 'ready', 'reasons': outcome['reasons'],
                'fixed_gate_baseline_supported': baseline['status'] == 'ready', 'fixed_gate_baseline_reasons': baseline['reasons']})
    return checks


def calibration_diagnostics(protocol, state):
    import numpy as np
    cases = {c['case_id']: c for c in protocol['cases']}; rows = []
    for unit in protocol['units']:
        if unit['scenario'] != 'calibration': continue
        outcome = state['computations'][unit['computation_id']]
        row = {k: unit[k] for k in ('case_id', 'sampling', 'method')}
        row['drug'] = cases[unit['case_id']]['drug']; row['status'] = outcome['status']
        if outcome['status'] != 'predicted': rows.append(row); continue
        mapping = checked(outcome['mapping']); choice = mapping['decision']
        singular = choice.get('data_jacobian_singular_values', [])
        near = [r for r in mapping['starts'] if r.get('eligible') and r['objective'] <= choice['objective'] + .01]
        keys = list(AXES); parameters = choice['parameters']
        bounds = ((.25, 4.), (.1, 10.), (.01, 100.))
        hits = [key for key, (lo, hi) in zip(keys, bounds) if min(
            abs(math.log(parameters[key] / lo)), abs(math.log(parameters[key] / hi))) <= .01 * math.log(hi / lo)]
        row.update(mapping=outcome['mapping'], parameters=parameters, data_jacobian_singular_values=singular,
            local_condition_number=singular[0] / singular[-1] if singular and singular[-1] > 0 else None,
            bound_hits=hits, near_optimal_starts=len(near), selected_start=choice['index'],
            active_parameters=mapping.get('active_parameters', keys),
            near_optimal_parameter_ranges={key: [min(r['parameters'][key] for r in near), max(r['parameters'][key] for r in near)] for key in keys},
            development_objective=choice['objective'], development_ln_rmse=choice['data_only_ln_rmse'])
        for name in ('screening', 'parameter_selection', 'axis_selection', 'calibration_screening'):
            if name in mapping: row[name] = mapping[name]
        rows.append(row)
    return rows


def report(run):
    from report_blueprint_validation import Store, _case_reference, _score_prediction, positive, clustered_geomean, paired_ratio
    run = own_run(run); protocol, state = load(run, locked=True)
    if state['status'] == 'complete':
        result = checked(state['results']); verify(result['source_pins']); return result
    store = Store(ROOT)
    for path, digest in protocol['source_pins'].items(): store.checksum(path, digest)
    store.ref(state['protocol']); generated = {}; pin_all([state['fits'], state['computations'], state['formula_selection']], generated)
    for path, digest in generated.items(): store.checksum(path, digest)
    replay, protection, replay_pins = replay_and_protection(protocol, state)
    for path, digest in replay_pins.items(): store.checksum(path, digest)
    purpose = applicability(protocol, state); calibration = calibration_diagnostics(protocol, state)
    # First target outcome decode after all 76 fits and 38 new predictions lock.
    references = {c['case_id']: _case_reference(c, store) for c in protocol['cases']}
    cases = {c['case_id']: c for c in protocol['cases']}; records, scored_cache = [], {}
    for unit in protocol['units']:
        case = cases[unit['case_id']]; clinical, curve, endpoints = references[unit['case_id']]
        record = {k: unit[k] for k in ('id', 'case_id', 'scenario', 'sampling', 'method')}
        record.update(drug=case['drug'], study_id=case['study_id'], cluster_id=case['cluster_id'],
            clinical_references=deepcopy(endpoints), expected_observation_missing=sum(e['value'] is None for e in endpoints.values()))
        if unit['status'] == 'blocked':
            record.update(attempt_status='blocked', scoring_status='not_attempted', reasons=unit['reasons'])
        else:
            compid = unit['computation_id']; computation = state['computations'][compid]
            record.update(computation_id=compid, attempt_status=computation['status'],
                prediction_source=computation.get('prediction'), mapping_source=computation.get('mapping'),
                resources=computation.get('resources'), reused_baseline=compid in set(protocol['baseline_ids'].values()))
            if compid not in scored_cache:
                try:
                    if computation['status'] != 'predicted': raise ValueError(str(computation.get('failure')))
                    scored, _, _ = _score_prediction(case, clinical, curve, endpoints, computation, store)
                    scored.pop('curve', None); scored['scoring_status'] = 'scored' if positive(scored['curve_metrics'].get('gmfe')) else 'partial'
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    scored = {'scoring_status': 'failed', 'failure': str(exc)}
                scored_cache[compid] = scored
            record.update(deepcopy(scored_cache[compid]))
        records.append(record)
    summary = []
    for scenario in (*SCENARIOS, 'calibration'):
        methods = ('always_three', 'adaptive_axes') if scenario == 'calibration' else POLICIES
        for sampling in SAMPLING:
            for method in methods:
                rows = [r for r in records if (r['scenario'], r['sampling'], r['method']) == (scenario, sampling, method)]
                observed = {r['case_id']: r.get('curve_metrics', {}).get('gmfe') for r in rows if positive(r.get('curve_metrics', {}).get('gmfe'))}
                estimate = clustered_geomean(observed, {cid: cases[cid]['cluster_id'] for cid in observed}, seed=20260930)
                summary.append({'scenario': scenario, 'sampling': sampling, 'method': method, 'attempted': len(rows),
                    'blocked': sum(r['attempt_status'] == 'blocked' for r in rows),
                    'failed': sum(r['attempt_status'] == 'failed' or r['scoring_status'] == 'failed' for r in rows),
                    'scored': len(observed), 'coverage': len(observed) / len(rows), 'curve_gmfe': estimate,
                    'within_twofold': sum(v <= 2 for v in observed.values()),
                    'scoring_note': 'Clinical metric conditions on scored cases; always read with coverage.'})
    paired = {}
    for sampling in SAMPLING:
        def collect(method):
            return {r['case_id']: r.get('curve_metrics', {}).get('gmfe') for r in records if
                (r['scenario'], r['sampling'], r['method']) == ('calibration', sampling, method)}
        paired[sampling] = paired_ratio(collect('adaptive_axes'), collect('always_three'),
            {cid: c['cluster_id'] for cid, c in cases.items()}, expected_ids=list(cases), seed=20260930)
    result = {'version': VERSION, 'scope': protocol['scope'], 'all_attempts_locked_before_scoring': True,
        'protocol': ref(run / 'protocol.json'), 'decisions': protocol['decisions'], 'units': records,
        'clinical_summary': summary, 'adaptive_to_always_three_curve_ratio': paired,
        'protection': protection, 'replay': replay, 'applicability': purpose, 'calibration': calibration,
        'permutations': protocol['permutations'], 'unique_scored_computations': len(scored_cache),
        'source_pins': dict(store.pins), 'limitations': protocol['limitations']}
    missing_effect = {}
    for sampling in SAMPLING:
        missing_effect[sampling] = {}
        for method in POLICIES:
            def scores(scenario):
                return {r['case_id']: r.get('curve_metrics', {}).get('gmfe') for r in records if
                    (r['scenario'], r['sampling'], r['method']) == (scenario, sampling, method)}
            missing_effect[sampling][method] = {}
            for label, case_ids in (('changed_four_drugs', protocol['changed_fu_case_ids']),
                                   ('unchanged_six_controls', protocol['unchanged_fu_case_ids'])):
                missing_effect[sampling][method][label] = paired_ratio(scores('missing_fu'), scores('available'),
                    {cid: cases[cid]['cluster_id'] for cid in case_ids}, expected_ids=case_ids, seed=20260930)
    result['missing_fu_effect'] = missing_effect
    result['protection_counts'] = {method: {parameter: {
        'records': sum(r['parameter'] == parameter and r['method'] == method for r in protection),
        'unique_drugs': len({r['case_id'] for r in protection if r['parameter'] == parameter and r['method'] == method}),
        'passed_records': sum(r['parameter'] == parameter and r['method'] == method and r['passed'] for r in protection)}
        for parameter in ('fraction_unbound', 'renal_clearance')} for method in ('always_three', 'adaptive_axes')}
    from pbpk_kb.blueprint_policy.capability_evaluation import evaluate_tail_gate
    result['tail_gate'] = evaluate_tail_gate(records, state['computations'], ROOT)
    for path, digest in result['tail_gate']['source_pins'].items(): store.checksum(path, digest)
    result['source_pins'] = dict(store.pins)
    new_costs = [state['computations'][cid].get('resources', {}) for cid in protocol['computations']]
    result['resources'] = {'unique_new_computations': len(new_costs),
        'native_calls_recorded': sum(c.get('total_native_calls', 0) for c in new_costs),
        'linear_solver_calls_recorded': sum(c.get('linear_solver_calls', 0) for c in new_costs),
        'sum_wall_seconds': sum(c.get('wall_seconds', 0) for c in new_costs),
        'costs_exact': not any(c.get('prior_incomplete_costs_unknown') or not c.get('fit_cost_count_exact', True) for c in new_costs),
        'replay_extra_linear_calls': sum(r['new_linear_solver_calls'] for r in replay), 'cache_aliases_not_counted_as_new_computation': True}
    store.unchanged(); save(run / 'results.json', result)
    table = [{**{k: v for k, v in row.items() if k != 'curve_gmfe'},
        'curve_gmfe': row['curve_gmfe']['estimate'], 'ci95_low': (row['curve_gmfe']['ci95'] or [None, None])[0],
        'ci95_high': (row['curve_gmfe']['ci95'] or [None, None])[1]} for row in summary]
    with (run / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    return result


def status(run):
    import psutil
    run = own_run(run); protocol, state = read(run / 'protocol.json'), read(run / 'state.json')
    result = {'status': state['status'], 'fits_completed': len(state['fits']), 'fits_total': 2 * len(protocol['computations']),
        'new_predictions_completed': sum(cid in state['computations'] for cid in protocol['computations']),
        'new_predictions_total': len(protocol['computations']), 'reused_predictions': 20,
        'monitor': state.get('monitor'), 'pause_reason': state.get('pause_reason')}
    if (run / 'pid.json').exists():
        process_ref = read(run / 'pid.json'); result.update(pid=process_ref['pid'], alive=False)
        try:
            process = psutil.Process(process_ref['pid'])
            result['alive'] = process.uids().real == os.getuid() and abs(process.create_time() - process_ref['create_time']) < .01 and process.status() != 'zombie'
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    return result


def background(run):
    import psutil
    if status(run).get('alive'): raise ValueError('Existing project run is active')
    with (run / 'run.log').open('a') as log:
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--run-dir', str(run)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    save(run / 'pid.json', {'pid': process.pid, 'uid': os.getuid(), 'create_time': psutil.Process(process.pid).create_time(),
        'script': str(Path(__file__).resolve().relative_to(ROOT)), 'started_at': now()})
    return {'pid': process.pid, 'log': str((run / 'run.log').relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'report', 'cleanup', 'pause', 'resume'))
    parser.add_argument('--run-dir', type=Path, default=DEFAULT); parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--background', action='store_true'); args = parser.parse_args(); run = own_run(args.run_dir)
    if args.command == 'prepare': result = prepare(run, args.workers)
    elif args.command == 'status': result = status(run)
    elif args.command == 'run': result = background(run) if args.background else execute(run)
    elif args.command == 'pause': save(run / 'PAUSE', {'time': now()}); result = {'status': 'pause_requested'}
    elif args.command == 'resume':
        if status(run).get('alive'): raise ValueError('Existing process is active')
        (run / 'PAUSE').unlink(missing_ok=True); result = background(run) if args.background else execute(run)
    elif args.command == 'cleanup':
        protocol, state = load(run)
        result = {'fits_checked': len(state['fits']), 'removed_count': 0,
            'reason': 'No per-evaluation artifacts are produced; selected mappings and source-bound native artifacts remain required.'}
    else:
        with (run / '.run.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = {'status': 'reported', 'units': len(report(run)['units'])}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
