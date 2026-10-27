#!/usr/bin/env python3
"""Two predeclared perfusion architectures against frozen PBPK and two-compartment controls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from run_blueprint_learning import Controller, Paused, now
import run_blueprint_refinement as frozen

VERSION = 'blueprint-perfusion-pilot-1'
GROUP = ROOT / 'experiments/blueprint_design/architecture_20260930'
BASE = GROUP / 'clinical_refinement_v2'
DEFAULT = GROUP / 'clinical_perfusion_v1'
VARIANTS = ('perfusion_global', 'perfusion_grouped')
BASELINES = frozen.METHODS
METHODS = (*VARIANTS, *BASELINES)
PILOT_DRUGS, SAMPLING, FORMULAS = frozen.PILOT_DRUGS, frozen.SAMPLING, frozen.FORMULAS
OPTIMIZER_BUDGET = {'fixed_starts': 16, 'max_evaluations_per_start': 400}
unit_id, fit_id = frozen.unit_id, frozen.fit_id
target_signature, choose_formula = frozen.target_signature, frozen.choose_formula


def read(path):
    return json.loads(inside(path, ROOT, exists=True).read_text())


def ref(path):
    path = inside(path, ROOT, exists=True)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def verify(pins):
    for name, digest in pins.items():
        if sha256(inside(name, ROOT, exists=True)) != digest:
            raise ValueError('Frozen source/code changed: ' + name)


def checked(record):
    verify({record['path']: record['sha256']})
    return read(record['path'])


def pin_refs(value, pins):
    """Verify nested references and pin dictionaries without decoding documents."""
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            verify({value['path']: value['sha256']})
            pins[value['path']] = value['sha256']
        for field, child in value.items():
            if field in ('source_pins', 'code_pins') and isinstance(child, dict):
                verify(child)
                pins.update(child)
            else:
                pin_refs(child, pins)
    elif isinstance(value, list):
        for child in value:
            pin_refs(child, pins)


def own_run(path):
    raw = Path(path) if Path(path).is_absolute() else ROOT / path
    if raw.is_symlink() or GROUP.is_symlink():
        raise ValueError('Perfusion runs cannot be symbolic links')
    run = inside(raw, ROOT)
    if run.parent != GROUP.resolve() or not run.name.startswith('clinical_perfusion_') or run == BASE.resolve():
        raise ValueError('Use a dedicated immediate clinical_perfusion_ child of the architecture group')
    return run


def engine():
    from pbpk_kb.blueprint_policy import perfusion_calibration
    return perfusion_calibration


def engine_contract():
    module = engine()
    if not isinstance(module.VERSION, str) or not module.VERSION:
        raise ValueError('The perfusion engine must declare its version')
    config = json.loads(json.dumps(module.CONFIG, allow_nan=False))
    if config.get('starts') != OPTIMIZER_BUDGET['fixed_starts'] or config.get('max_evaluations_per_start') != OPTIMIZER_BUDGET['max_evaluations_per_start']:
        raise ValueError('Perfusion engine does not implement the declared fixed-start budget')
    pins = module._code_pins(ROOT)
    verify(pins)
    source = inside(module.__file__, ROOT, exists=True)
    if pins.get(str(source.relative_to(ROOT))) != sha256(source):
        raise ValueError('Perfusion engine source is missing from its dependency pins')
    return {'module': 'pbpk_kb.blueprint_policy.perfusion_calibration',
        'version': module.VERSION, 'config': config, 'code_pins': pins}


def fit_costs(mapping, variant=None):
    for field in ('total_native_calls', 'actual_solver_calls', 'linear_solver_calls', 'failed_evaluation_count'):
        if type(mapping.get(field)) is not int or mapping[field] < 0:
            raise ValueError('Fit must separately report nonnegative native and linear costs: ' + field)
    if mapping['actual_solver_calls'] > mapping['total_native_calls']:
        raise ValueError('Native solver calls exceed native construction attempts')
    if variant is not None and mapping.get('variant') != variant:
        raise ValueError('Fit returned a different perfusion variant')


def validate_source(source, completed):
    """Check the completed refinement matrix without decoding old mappings or scores."""
    if source.get('version') != frozen.VERSION or completed.get('version') != frozen.VERSION:
        raise ValueError('Frozen controls must come from the refinement protocol')
    for field, value in (('methods', BASELINES), ('sampling', SAMPLING), ('formulas', FORMULAS)):
        if source.get(field) != list(value):
            raise ValueError('Frozen refinement conditions differ: ' + field)
    if completed.get('status') != 'complete' or not completed.get('all_attempts_locked_before_scoring') or not completed.get('monitor_stopped_before_scoring'):
        raise ValueError('The frozen refinement must be complete after the prediction lock')
    ids = [case['case_id'] for case in source['cases']]
    expected = {unit_id(cid, sampling, method) for cid in ids for sampling in SAMPLING for method in BASELINES}
    native = {unit_id(cid, sampling, frozen.METHOD) for cid in ids for sampling in SAMPLING}
    if (len(ids) != 4 or len(set(ids)) != 4 or len(source['units']) != 56
            or {unit['id'] for unit in source['units']} != expected
            or set(completed['computations']) != expected
            or set(source['computations']) != native
            or set(completed['fits']) != {fit_id(identity, formula) for identity in native for formula in FORMULAS}
            or any(value.get('status') not in ('predicted', 'failed') for value in completed['computations'].values())):
        raise ValueError('The complete frozen 56-unit comparison is required')
    if not completed.get('formula_selection') or not completed.get('results'):
        raise ValueError('The frozen refinement requires sealed formula selection and report')


def prepare(run, workers=14):
    run = own_run(run)
    if run.exists():
        raise FileExistsError('Preserve this batch; resume it or choose another perfusion directory')
    if type(workers) is not int or not 1 <= workers <= 14:
        raise ValueError('Use one to fourteen native workers')
    source, completed = read(BASE / 'protocol.json'), read(BASE / 'state.json')
    validate_source(source, completed)
    if completed['protocol'] != ref(BASE / 'protocol.json'):
        raise ValueError('Frozen state does not bind its source protocol')
    pins = {}
    # The old score and mappings remain opaque; hashing seals their provenance.
    pin_refs(source, pins)
    pin_refs(completed, pins)
    if 'results' not in completed:
        raise ValueError('The frozen refinement has no sealed report')
    cases = deepcopy(source['cases'])
    if sorted(c['drug'] for c in cases) != sorted(PILOT_DRUGS):
        raise ValueError('The four frozen pilot drugs must match exactly')
    contract = engine_contract()
    pins.update(contract['code_pins'])
    units, specifications, reused = [], {}, {}
    source_state = ref(BASE / 'state.json')
    for case in cases:
        for sampling in SAMPLING:
            cid = case['case_id']
            native = source['computations'][unit_id(cid, sampling, frozen.METHOD)]
            request, target, training = (checked(native[name]) for name in ('request', 'target_request', 'training_source'))
            if sampling == 'five' and len(training['observations']) != 5:
                raise ValueError('Five-point development input differs from the frozen subset')
            if any(key in training for key in ('raw_observations', 'full_curve', 'fit', 'selection')):
                raise ValueError('Training input contains forbidden full-curve/fit metadata')
            if training['human'] != request['human'] or training['dosing'] != request['dosing']:
                raise ValueError('Development human/dosing differs from the construction request')
            for variant in VARIANTS:
                identity = unit_id(cid, sampling, variant)
                specifications[identity] = {name: deepcopy(native[name]) for name in ('request', 'target_request', 'training_source')}
                specifications[identity].update(case_id=cid, sampling=sampling, method=variant,
                    optimizer_budget=deepcopy(OPTIMIZER_BUDGET))
            for method in METHODS:
                identity = unit_id(cid, sampling, method)
                units.append({'id': identity, 'case_id': cid, 'sampling': sampling, 'method': method,
                    'kind': 'perfusion' if method in VARIANTS else 'frozen_baseline'})
                if method in VARIANTS:
                    continue
                result = deepcopy(completed['computations'][identity])
                if method in ('two_compartment', 'two_compartment_prior'):
                    training_ref = native['training_source']
                    if result.get('source_pins', {}).get(training_ref['path']) != training_ref['sha256']:
                        raise ValueError('A frozen two-compartment control used a different development subset')
                if result['status'] == 'predicted':
                    actual = checked(result['request'])
                    for key in ('human', 'dosing', 'target'):
                        left, right = actual[key], target[key]
                        if key == 'target':
                            left, right = target_signature(left), target_signature(right)
                        if left != right:
                            raise ValueError('A frozen prediction used a different target request')
                history = deepcopy(result.get('original_resources', result.get('resources', {})))
                result.update(frozen_computation_id=identity, reused_from=source_state,
                    original_resources=history, resources={'actual_new_native_calls': 0, 'linear_solver_calls': 0, 'wall_seconds': 0,
                        'cached_native_calls': history.get('total_native_calls', history.get('native_execution_calls', 0)),
                        'cache_reused': True, 'resource_scope': 'Frozen prediction reuse; original resources retained separately'})
                reused[identity] = result
    for value in (cases, specifications, reused):
        pin_refs(value, pins)
    for path in (Path(__file__), ROOT / 'scripts/run_blueprint_refinement.py', ROOT / 'scripts/run_blueprint_architecture.py',
            ROOT / 'scripts/run_blueprint_learning.py', ROOT / 'scripts/report_blueprint_validation.py',
            BASE / 'protocol.json', BASE / 'state.json'):
        pins.update({str(path.relative_to(ROOT)): sha256(path)})
    protocol = {'version': VERSION, 'created_at': now(), 'cases': cases, 'methods': list(METHODS),
        'sampling': list(SAMPLING), 'formulas': list(FORMULAS), 'units': units, 'computations': specifications,
        'engine': contract, 'source_protocol': ref(BASE / 'protocol.json'), 'source_state': source_state,
        'source_results': deepcopy(completed['results']), 'source_pins': pins, 'workers': workers,
        'native_threads': 2, 'native_timeout_seconds': 180, 'optimizer_budget': deepcopy(OPTIMIZER_BUDGET),
        'monitor_seconds': 1200, 'directory_limit_bytes': 10 * 1024**3,
        'scope': 'Previously viewed four-drug development cohort; two predeclared perfusion architectures, not blinded validation',
        'fairness': ['Both new architectures use the same frozen full/five observed development subset and frozen PK evidence as the completed refinement.',
            'Moxifloxacin reported PK priors come from the same development study Wise1999; five points are not its entire human information budget and this prior is not independent.',
            'Remdesivir prior PK evidence comes from study 4231, development from 1954, and target from 1812.',
            'Perfusion optimization has 16 fixed starts with at most 400 optimizer evaluations per start, matching the classical comparator ceiling; actual linear solves, native construction calls and time are reported separately.',
            'The native 48-attempt cap of historical PBPK optimization is not applied to linear perfusion solves; these are different computational methods.',
            'All seven old methods reuse immutable predictions; historical resource costs are retained separately from zero new reuse calls.',
            'An identical frozen request may reuse an existing evidence base; only the construction that actually runs is charged. No fitted parameters are reused.',
            'The two-compartment control without a prior intentionally omits PK prior evidence; the prior control retains its original evidence.',
            'The global two-parameter and grouped three-parameter tissue-partition architectures are predetermined for every drug and both subsets; fixed group definitions come from the frozen engine configuration.',
            'Neither target scores nor data-rank diagnostics choose the architecture for an individual drug.',
            'Formula choice uses only each architecture development objective; all 32 fits and all 16 choices finish before any new target prediction.',
            'All predictions lock and the monitor stops before target observations are decoded. No old fit or target score supplies a warm start.',
            'Previous target results informed continued development; the new run is not an independent or blind validation.'],
        'bootstrap': {'seed': 20260929, 'resamples': 2000},
        'retention': 'Keep one aggregate report, all predictions, protocols, best models and compact fit/failure records; clean only engine-declared successful intermediate work in this run.'}
    # Do not publish a protocol if the frozen source changed during preparation.
    verify(pins)
    run.mkdir(parents=True)
    save(run / 'protocol.json', protocol)
    save(run / 'state.json', {'version': VERSION, 'status': 'prepared', 'protocol': ref(run / 'protocol.json'),
        'created_at': now(), 'fits': {}, 'computations': reused, 'monitor': {}})
    return {'drugs': len(cases), 'new_formula_fits': len(specifications) * len(FORMULAS),
        'new_predictions': len(specifications), 'reused_predictions': len(reused), 'comparison_units': len(units)}


def validate_matrix(protocol, state, locked=False):
    if protocol.get('version') != VERSION or state.get('version') != VERSION:
        raise ValueError('Wrong perfusion protocol version')
    for field, expected in (('methods', METHODS), ('sampling', SAMPLING), ('formulas', FORMULAS)):
        if protocol.get(field) != list(expected):
            raise ValueError('Changed perfusion conditions: ' + field)
    if protocol.get('optimizer_budget') != OPTIMIZER_BUDGET:
        raise ValueError('Changed perfusion optimizer budget')
    cases = protocol['cases']
    ids = [case['case_id'] for case in cases]
    if len(set(ids)) != 4 or sorted(c['drug'] for c in cases) != sorted(PILOT_DRUGS):
        raise ValueError('The four pilot cases must be unique')
    expected = {unit_id(cid, s, m) for cid in ids for s in SAMPLING for m in METHODS}
    if (len(protocol['units']) != len(expected) or {u['id'] for u in protocol['units']} != expected
            or any(u['id'] != unit_id(u['case_id'], u['sampling'], u['method']) for u in protocol['units'])):
        raise ValueError('Incomplete or duplicated perfusion comparison matrix')
    native = {unit_id(cid, s, method) for cid in ids for s in SAMPLING for method in VARIANTS}
    if (set(protocol['computations']) != native or any(c['method'] not in VARIANTS
            or key != unit_id(c['case_id'], c['sampling'], c['method'])
            or c.get('optimizer_budget') != OPTIMIZER_BUDGET for key, c in protocol['computations'].items())):
        raise ValueError('Only the sixteen declared perfusion computations may execute')
    fits = {fit_id(identity, formula) for identity in native for formula in FORMULAS}
    if not set(state['fits']) <= fits or not expected - native <= set(state['computations']) <= expected:
        raise ValueError('Unexpected fit or missing frozen baseline checkpoint')
    if locked:
        if set(state['fits']) != fits or set(state['computations']) != expected:
            raise ValueError('Every fit and prediction attempt must finish before scoring')
        if any(c.get('status') not in ('predicted', 'failed') for c in state['computations'].values()):
            raise ValueError('All predictions must be terminal before scoring')
        if (state.get('status') not in ('predictions_locked', 'complete')
                or not state.get('all_attempts_locked_before_scoring') or not state.get('monitor_stopped_before_scoring')):
            raise ValueError('Prediction lock and stopped monitor are required before scoring')
        selection = checked(state['formula_selection'])
        if set(selection['choices']) != native or selection.get('basis') != 'development_objective_only':
            raise ValueError('Every formula choice must be locked before target prediction')


def load_batch(run):
    protocol, state = read(run / 'protocol.json'), read(run / 'state.json')
    if state['protocol'] != ref(run / 'protocol.json'):
        raise ValueError('Perfusion protocol changed after preparation')
    validate_matrix(protocol, state)
    verify(protocol['source_pins'])
    pin_refs(state['fits'], {})
    pin_refs(state['computations'], {})
    return protocol, state


def lock_formulas(run, protocol, state):
    choices = {}
    for identity in sorted(protocol['computations']):
        mappings = [checked(state['fits'][fit_id(identity, formula)]['mapping']) for formula in FORMULAS]
        for mapping in mappings:
            fit_costs(mapping, protocol['computations'][identity]['method'])
        chosen = choose_formula(mappings)
        choices[identity] = None if chosen is None else {
            'formula': chosen['construction_partition_method'],
            'mapping': state['fits'][fit_id(identity, chosen['construction_partition_method'])]['mapping']}
    document = {'version': VERSION, 'basis': 'development_objective_only', 'choices': choices}
    path = run / 'formulas_locked.json'
    if path.exists():
        if read(path) != document:
            raise ValueError('Development formula selection changed after locking')
    else:
        save(path, document)
    if state.get('formula_selection') and checked(state['formula_selection']) != document:
        raise ValueError('Formula selection checkpoint changed')
    return document, ref(path)


def execute(run):
    from pbpk_kb.blueprints.evidence_preserving_calibration import _infrastructure_problem
    run = own_run(run)
    with (run / '.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, state = load_batch(run)
        if state['status'] == 'complete':
            report(run)
            return {'status': 'complete', 'reused': True}
        if engine_contract() != protocol['engine']:
            raise ValueError('Perfusion engine configuration changed after preparation')
        controller = Controller(run, protocol, state)
        controller.check(force=True)
        monitor = threading.Thread(target=controller.watch, daemon=True)
        monitor.start()
        controller.persist(status='running', pause_reason=None, started_at=state.get('started_at', now()))

        def fit_one(identity, config, formula):
            controller.check()
            key, folder = fit_id(identity, formula), run / 'work' / identity / formula
            if key in state['fits']:
                fit_costs(checked(state['fits'][key]['mapping']), config['method'])
                return
            request = checked(config['request'])
            request['partition_method'] = formula
            mapping = engine().fit_mapping(reference_request=request, training_source=config['training_source'],
                output_dir=folder, project_root=ROOT, variant=config['method'], before_trial=controller.check)
            fit_costs(mapping, config['method'])
            if mapping.get('construction_partition_method') != formula:
                raise ValueError('Fit returned a different construction formula')
            cleaned = engine().cleanup_unselected(output_dir=folder, project_root=ROOT)
            with controller.lock:
                state['fits'][key] = {'mapping': mapping['mapping'], 'status': mapping['status'], 'eligible': mapping['eligible'],
                    'native_calls': mapping['total_native_calls'], 'actual_solver_calls': mapping['actual_solver_calls'],
                    'linear_solver_calls': mapping['linear_solver_calls'], 'cleanup': cleaned}
                controller.persist()

        def predict_one(identity, config):
            controller.check()
            if identity in state['computations']:
                pin_refs(state['computations'][identity], {})
                return
            mappings = [checked(state['fits'][fit_id(identity, formula)]['mapping']) for formula in FORMULAS]
            resources = {'total_native_calls': sum(m['total_native_calls'] for m in mappings),
                'actual_solver_calls': sum(m['actual_solver_calls'] for m in mappings),
                'linear_solver_calls': sum(m['linear_solver_calls'] for m in mappings),
                'linear_cost_count_exact': all(m.get('cost_count_exact', True) for m in mappings),
                'optimizer_budget_per_formula': deepcopy(OPTIMIZER_BUDGET), 'cache_reused': False,
                'wall_seconds': sum(m.get('elapsed_seconds', 0) for m in mappings),
                'failed_evaluations': sum(m['failed_evaluation_count'] for m in mappings),
                'resource_scope': 'Only actual evidence-base construction is counted as native work; linear perfusion evaluations are counted separately.'}
            choice = checked(state['formula_selection'])['choices'][identity]
            if choice is None:
                result = {'status': 'failed', 'failure': 'No eligible formula fit'}
            else:
                request = checked(config['target_request'])
                request['partition_method'] = choice['formula']
                folder = run / 'work' / identity / 'target'
                def incomplete(path):
                    return path.exists() and not (path / 'completion.json').exists() and any(p.name != '.calibration.lock' for p in path.iterdir())
                if incomplete(folder):
                    folder = folder.with_name('target_retry_1')
                    if incomplete(folder):
                        raise Paused('Two incomplete target attempts retained; inspect before retry')
                native = engine().predict_mapping(request=request, mapping_path=ROOT / choice['mapping']['path'],
                    output_dir=folder, project_root=ROOT, before_trial=controller.check)
                if _infrastructure_problem(native):
                    raise Paused('Target native infrastructure failure')
                cost = read(folder / 'result.json') if native.get('completed_prediction_reused') or native.get('actual_new_calls') == 0 else native
                charged = cost.get('actual_new_calls')
                solved = cost.get('actual_solver_calls', cost.get('actual_native_solver_calls'))
                linear = cost.get('linear_solver_calls')
                if type(charged) is not int or charged < 0:
                    raise ValueError('Target must report its actual native construction calls')
                if type(solved) is not int or not 0 <= solved <= charged:
                    raise ValueError('Target must separately report valid actual solver calls')
                if type(linear) is not int or linear < 0:
                    raise ValueError('Target must separately report its linear solver calls')
                resources.update(target_native_calls=charged, target_linear_solver_calls=linear)
                resources['total_native_calls'] += charged
                resources['actual_solver_calls'] += solved
                resources['linear_solver_calls'] += linear
                if folder.name == 'target_retry_1':
                    # Never invent a native cap or hide work from an interrupted attempt.
                    resources['prior_incomplete_target_costs_unknown'] = True
                    resources['linear_cost_count_exact'] = False
                    for name in ('total_native_calls', 'actual_solver_calls', 'linear_solver_calls'):
                        resources['recorded_' + name] = resources[name]
                        resources[name] = None
                resources['wall_seconds'] += cost.get('elapsed_seconds', 0)
                result = {'status': 'predicted' if native.get('status') == 'predicted' else 'failed',
                    'native_status': native.get('status'), 'mapping': choice['mapping'], 'request': ref(folder / 'request.json'),
                    'selected_formula': choice['formula'], 'selection_basis': 'Locked development objective only'}
                if result['status'] == 'predicted':
                    result['prediction'] = ref(folder / 'prediction_locked.json')
                else:
                    result['failure'] = native
            resources['actual_new_native_calls'] = resources['total_native_calls']
            result['resources'] = resources
            with controller.lock:
                state['computations'][identity] = result
                controller.persist()

        def phase(jobs, function):
            with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
                futures = {pool.submit(function, *args): args[0] for args in jobs}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        with controller.lock:
                            if not controller.halted.is_set():
                                state.setdefault('interruptions', []).append({'unit': futures[future], 'time': now(), 'error': str(exc)})
                                controller.pause(type(exc).__name__ + ': ' + str(exc))
            controller.check(force=True)

        try:
            items = sorted(protocol['computations'].items())
            phase([(identity, config, formula) for identity, config in items for formula in FORMULAS], fit_one)
            _, selection = lock_formulas(run, protocol, state)
            controller.persist(status='predicting_targets', formula_selection=selection, formulas_locked_before_target_prediction=True)
            phase(items, predict_one)
            controller.closed.set()
            monitor.join(timeout=30)
            if monitor.is_alive():
                raise Paused('Monitor did not stop; defer scoring')
            controller.persist(status='predictions_locked', all_attempts_locked_before_scoring=True,
                monitor_stopped_before_scoring=True, locked_at=now())
            result = report(run)
            controller.persist(status='complete', completed_at=now(), results=ref(run / 'results.json'))
            return {'status': 'complete', 'units': len(result['units'])}
        except Exception as exc:
            if not controller.halted.is_set():
                controller.pause(type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            controller.closed.set()
            monitor.join(timeout=30)


def statistics(records, cases):
    from report_blueprint_validation import UNITS, clustered_geomean, paired_ratio, positive
    ids = [c['case_id'] for c in cases]
    clusters = {c['case_id']: c['cluster_id'] for c in cases}
    lookup = {(r['case_id'], r['sampling'], r['method']): r for r in records}
    def values(sampling, method):
        return {cid: lookup[cid, sampling, method].get('curve_metrics', {}).get('gmfe') for cid in ids}
    groups = {s: {} for s in SAMPLING}
    for sampling in SAMPLING:
        for method in METHODS:
            selected = [lookup[cid, sampling, method] for cid in ids]
            scores = {cid: value for cid, value in values(sampling, method).items() if positive(value)}
            curve = clustered_geomean(scores, {cid: clusters[cid] for cid in scores})
            curve.update(attempted=len(ids), failed=sum(r['attempt_status'] == 'failed' for r in selected),
                scored=len(scores), coverage=len(scores) / len(ids), within_twofold=sum(x <= 2 for x in scores.values()))
            endpoints = {}
            for endpoint in UNITS:
                errors = {r['case_id']: r.get('endpoints', {}).get(endpoint, {}).get('error') for r in selected}
                errors = {cid: error for cid, error in errors.items() if error is not None}
                item = {'scored': len(errors), 'reference_available': sum(r['clinical_references'][endpoint]['value'] is not None for r in selected),
                    'weak_scored': sum(r.get('endpoints', {}).get(endpoint, {}).get('error') is not None and
                        r['endpoints'][endpoint].get('weak_tail', False) for r in selected)}
                if endpoint == 'tmax':
                    item['mean_absolute_error_h'] = math.fsum(errors.values()) / len(errors) if errors else None
                else:
                    item['fold_error'] = clustered_geomean(errors, {cid: clusters[cid] for cid in errors})
                    item['within_twofold'] = sum(error <= 2 for error in errors.values())
                endpoints[endpoint] = item
            groups[sampling][method] = {'curve': curve, 'endpoints': endpoints}
    return {'by_sampling': groups,
        'paired_curve_gmfe_ratios': {s: {method: {base: paired_ratio(values(s, method), values(s, base), clusters, expected_ids=ids)
            for base in METHODS if base != method} for method in VARIANTS} for s in SAMPLING},
        'five_to_full_curve_gmfe_ratios': {m: paired_ratio(values('five', m), values('full', m), clusters, expected_ids=ids) for m in METHODS},
        'interpretation': 'Study-cluster paired bootstrap, 2000 resamples; four-drug development pilot, no confirmatory significance claim.'}


def comparison_table(records, summary):
    """One aggregate table; detailed case and endpoint values remain in results JSON."""
    def total(rows, field, original=False):
        values = [row.get('original_resources' if original else 'resources', {}).get(field) for row in rows]
        return sum(values) if all(type(value) in (int, float) and math.isfinite(value) for value in values) else None
    table = []
    for sampling in SAMPLING:
        for method in METHODS:
            rows = [row for row in records if row['sampling'] == sampling and row['method'] == method]
            group = summary['by_sampling'][sampling][method]
            curve, endpoints = group['curve'], group['endpoints']
            interval = curve.get('ci95') or (None, None)
            table.append({'method': method, 'sampling': sampling, 'curve_gmfe': curve['estimate'],
                'curve_ci95_low': interval[0], 'curve_ci95_high': interval[1],
                'scored_cases': curve['scored'], 'attempted_cases': curve['attempted'], 'failed_cases': curve['failed'],
                'auc_gmfe': endpoints['auc']['fold_error']['estimate'], 'cmax_gmfe': endpoints['cmax']['fold_error']['estimate'],
                'new_native_construction_calls': total(rows, 'actual_new_native_calls'),
                'new_linear_solver_calls': total(rows, 'linear_solver_calls'),
                'linear_count_exact': all(row.get('resources', {}).get('linear_cost_count_exact', True) for row in rows),
                'sum_fit_and_target_wall_seconds': total(rows, 'wall_seconds'),
                'historical_native_calls': total(rows, 'total_native_calls', original=True),
                'historical_analytic_residual_calls': total(rows, 'analytic_cpu_residual_calls', original=True),
                'historical_wall_seconds': total(rows, 'wall_seconds', original=True),
                'reused_frozen_predictions': method in BASELINES})
    return table


def report(run):
    from report_blueprint_validation import CURVE, UNITS, Store, _case_reference, _score_prediction, _unit_row, positive
    run = own_run(run)
    protocol, state = load_batch(run)
    validate_matrix(protocol, state, locked=True)
    store = Store(ROOT)
    store.ref(state['protocol'])
    for name, digest in protocol['source_pins'].items():
        store.checksum(name, digest)
    generated = {}
    for value in (state['fits'], state['computations'], state['formula_selection']):
        pin_refs(value, generated)
    for name, digest in generated.items():
        store.checksum(name, digest)
    if state['status'] == 'complete':
        result = store.ref(state['results'])
        for name, digest in result['source_pins'].items():
            store.checksum(name, digest)
        store.unchanged()
        return result
    # The first target observation read is deliberately below every lock check.
    references = {c['case_id']: _case_reference(c, store) for c in protocol['cases']}
    cases = {c['case_id']: c for c in protocol['cases']}
    records, rows = [], []
    for unit in protocol['units']:
        case = cases[unit['case_id']]
        clinical, curve, endpoints = references[unit['case_id']]
        computation = state['computations'][unit['id']]
        record = {'unit_id': unit['id'], 'case_id': unit['case_id'], 'drug': case['drug'], 'study_id': case['study_id'],
            'cluster_id': case['cluster_id'], 'sampling': unit['sampling'], 'method': unit['method'],
            'attempt_status': computation['status'], 'scoring_status': 'failed', 'failure': computation.get('failure', ''),
            'resources': deepcopy(computation.get('resources', {})), 'original_resources': deepcopy(computation.get('original_resources', {})),
            'reused_from': computation.get('reused_from'), 'prediction_source': computation.get('prediction', {}),
            'clinical_references': deepcopy(endpoints), 'source_approximations': deepcopy(clinical.get('approximations', []))}
        if computation['status'] == 'predicted':
            try:
                scored, _, _ = _score_prediction(case, clinical, curve, endpoints, computation, store)
                scored.pop('curve', None)
                record.update(scored)
                record['scoring_status'] = 'scored' if positive(record.get('curve_metrics', {}).get('gmfe')) else 'partial'
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                record.update(failure=str(exc), failure_category=getattr(exc, 'category', 'clinical_scoring_error'))
        records.append(record)
        rows.extend(_unit_row(record, endpoint, {'curve': curve, 'endpoints': endpoints}) for endpoint in (CURVE, *UNITS))
    result = {'version': VERSION, 'scope': protocol['scope'], 'engine': protocol['engine'],
        'all_attempts_locked_before_scoring': True, 'units': records, 'statistics': statistics(records, protocol['cases']),
        'fits': deepcopy(state['fits']), 'formula_selection': state['formula_selection'],
        'sensitivity_without_moxifloxacin': statistics([r for r in records if r['drug'] != 'Moxifloxacin'],
            [c for c in protocol['cases'] if c['drug'] != 'Moxifloxacin']),
        'source_pins': dict(store.pins), 'limitations': protocol['fairness'] + ['Missing clinical values are not imputed.',
            'No aggregate winner is selected across incompatible endpoint dimensions.']}
    result['comparison_table'] = comparison_table(records, result['statistics'])
    result['endpoint_rows'] = rows
    store.unchanged()
    with tempfile.TemporaryDirectory(prefix='.perfusion-report-', dir=run) as temporary:
        temporary = Path(temporary)
        (temporary / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        with (temporary / 'results.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result['comparison_table'][0]))
            writer.writeheader()
            writer.writerows(result['comparison_table'])
        store.unchanged()
        for name in ('results.csv', 'results.json'):
            (temporary / name).replace(run / name)
    return result


def status(run):
    run = own_run(run)
    state, protocol = read(run / 'state.json'), read(run / 'protocol.json')
    progress = []
    for path in (run / 'work').glob('*/*/progress.json'):
        try:
            progress.append(read(path))
        except (ValueError, FileNotFoundError):
            pass
    predicted = sum(key in state['computations'] for key in protocol['computations'])
    result = {'status': state['status'], 'comparison_units': len(protocol['units']), 'fits_completed': len(state['fits']),
        'fits_total': len(protocol['computations']) * len(FORMULAS), 'new_predictions_completed': predicted,
        'new_predictions_total': len(protocol['computations']), 'frozen_baselines_reused': len(state['computations']) - predicted,
        'native_fit_calls_recorded': sum(item.get('total_native_calls', 0) for item in progress),
        'linear_fit_calls_recorded': sum(item.get('linear_solver_calls', 0) for item in progress),
        'monitor': state.get('monitor'), 'pause_reason': state.get('pause_reason')}
    if (run / 'pid.json').exists():
        import psutil
        pid = read(run / 'pid.json')
        result.update(pid=pid['pid'], alive=False)
        try:
            process = psutil.Process(pid['pid'])
            result['alive'] = (process.uids().real == os.getuid() and abs(process.create_time() - pid['create_time']) < .01
                and process.status() != 'zombie')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return result


def background(run):
    import psutil
    run = own_run(run)
    if status(run).get('alive'):
        raise ValueError('This perfusion experiment already has a live process')
    with (run / 'run.log').open('a') as log:
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--run-dir', str(run)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    save(run / 'pid.json', {'pid': process.pid, 'uid': os.getuid(), 'create_time': psutil.Process(process.pid).create_time(),
        'started_at': now(), 'script': str(Path(__file__).relative_to(ROOT))})
    return {'pid': process.pid, 'log': str((run / 'run.log').relative_to(ROOT))}


def cleanup(run):
    run = own_run(run)
    with (run / '.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, _ = load_batch(run)
        items = []
        for identity in sorted(protocol['computations']):
            for formula in FORMULAS:
                folder = inside(run / 'work' / identity / formula, ROOT)
                if not folder.is_relative_to(run):
                    raise ValueError('Cleanup cannot leave this perfusion directory')
                if (folder / 'mapping_locked.json').exists():
                    items.append(engine().cleanup_unselected(output_dir=folder, project_root=ROOT))
        return {'fits_checked': len(items), 'removed_count': sum(item['removed_count'] for item in items)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'report', 'pause', 'resume', 'cleanup'))
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    run = own_run(args.run_dir)
    if args.command == 'prepare':
        result = prepare(run, args.workers)
    elif args.command == 'status':
        result = status(run)
    elif args.command == 'run':
        result = background(run) if args.background else execute(run)
    elif args.command == 'pause':
        save(run / 'PAUSE', {'requested_at': now(), 'reason': 'Pause requested; no new native submissions'})
        result = {'status': 'pause_requested'}
    elif args.command == 'resume':
        if status(run).get('alive'):
            raise ValueError('Wait for the existing runner and native calls to stop')
        with (run / '.run.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            load_batch(run)
            (run / 'PAUSE').unlink(missing_ok=True)
        result = background(run) if args.background else execute(run)
    elif args.command == 'cleanup':
        result = cleanup(run)
    else:
        with (run / '.run.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = {'status': 'reported', 'units': len(report(run)['units'])}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
