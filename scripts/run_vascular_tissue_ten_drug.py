#!/usr/bin/env python3
"""Apply the frozen vascular–tissue exchange method uniformly to ten drugs.

This adapter changes cohort orchestration only. It calls the immutable transport
calibrator and scheduler; target observations remain sealed until all 49 new
predictions and all 40 formula fits have reached a terminal checkpoint.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from run_blueprint_learning import Controller, Paused, now
import run_blueprint_transport as scheduler

VERSION = 'vascular-tissue-ten-drug-1'
DEFAULT = ROOT / 'experiments/blueprint_design/vascular_tissue_ten_drug_20260930'
BASE = ROOT / 'experiments/blueprint_design/validation_20260929'
INPUT_RESPONSE = ROOT / 'experiments/blueprint_design/evidence_priority_20260929/run/input_response/protocol.json'
METHODS = ('flow_exchange', 'two_compartment', 'two_compartment_prior')
SAMPLING = ('full', 'five')
FORMULAS = ('PKSimStandard', 'RodgersRowland')
CONFIG = {**scheduler.DEFAULT_CONFIG, 'version': VERSION,
    'baseline_methods': list(METHODS[1:]), 'source_variants': ['two_compartment'],
    'scope': 'Uniform frozen vascular–tissue exchange evaluation on ten previously inspected development drugs; not blind validation.'}
TRAINING_FIELDS = ('schema_version', 'compound_id', 'compound_name', 'study_id', 'role', 'matrix',
    'time_unit', 'unit', 'molecular_weight_g_mol', 'human', 'dosing', 'observations')
api = scheduler.create_runner(config=CONFIG)
uid = api.unit_id


def own_run(path):
    raw = Path(path) if Path(path).is_absolute() else ROOT / path
    run = inside(raw, ROOT)
    if raw.is_symlink() or run.parent != DEFAULT.parent or not run.name.startswith('vascular_tissue_ten_drug_'):
        raise ValueError('Use a dedicated immediate vascular_tissue_ten_drug_ experiment directory')
    return run


def visible_training(document):
    """Exact observed subset, with full-curve metadata and estimates excluded."""
    return {key: deepcopy(document[key]) for key in TRAINING_FIELDS if key in document}


def validate_matrix(protocol, state, locked=False):
    if protocol.get('version') != VERSION or state.get('version') != VERSION or protocol.get('runner_config') != CONFIG:
        raise ValueError('Frozen ten-drug protocol configuration changed')
    if any(protocol.get(k) != list(v) for k, v in [('methods', METHODS), ('sampling', SAMPLING), ('formulas', FORMULAS)]):
        raise ValueError('Condition matrix changed')
    cases = protocol['cases']; ids = [c['case_id'] for c in cases]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError('Exactly ten unique original cases required')
    source = api.checked(protocol['source_protocol'])
    if cases != source['cases']:
        raise ValueError('Clinical cases, source requests or NCA policies changed')
    all_ids = {uid(cid, s, m) for cid in ids for s in SAMPLING for m in METHODS}
    new_ids = {uid(cid, s, METHODS[0]) for cid in ids for s in SAMPLING}
    fits = {api.fit_id(i, f) for i in new_ids for f in FORMULAS}
    if (len(protocol['units']) != 60 or {u['id'] for u in protocol['units']} != all_ids
            or set(protocol['computations']) != new_ids or not set(state['fits']) <= fits
            or not all_ids - new_ids <= set(state['computations']) <= all_ids):
        raise ValueError('Incomplete or unexpected comparison matrix')
    for case in cases:
        for sampling in SAMPLING:
            config = protocol['computations'][uid(case['case_id'], sampling, METHODS[0])]
            if (config['request'] != case['requests']['evidence']['development']
                    or config['target_request'] != case['requests']['evidence']['target']
                    or config['optimizer_budget'] != api.OPTIMIZER_BUDGET):
                raise ValueError('Frozen request or optimizer budget changed')
            training = api.checked(config['training_source'])
            if training != visible_training(api.checked(case['sampling_sources'][sampling])):
                raise ValueError('Development observations differ from the original full/five subset')
            if sampling == 'five' and len(training['observations']) != 5:
                raise ValueError('Five-point isolation failed')
    if locked:
        if (set(state['fits']) != fits or set(state['computations']) != all_ids
                or any(v['status'] not in ('predicted', 'failed') for v in state['computations'].values())
                or state['status'] not in ('predictions_locked', 'complete')
                or not state.get('all_attempts_locked_before_scoring') or not state.get('monitor_stopped_before_scoring')):
            raise ValueError('All prediction attempts must lock before target scoring')
        selection = api.checked(state['formula_selection'])
        if set(selection['choices']) != new_ids or selection['basis'] != 'development_objective_only':
            raise ValueError('Development-only formula selection is not locked')


def prepare(run, workers=14):
    run = own_run(run)
    if run.exists():
        raise FileExistsError('Preserve existing batch; use run/status to resume')
    if type(workers) is not int or not 1 <= workers <= 14:
        raise ValueError('Use one to fourteen native workers')
    source, complete = api.read(BASE / 'protocol.json'), api.read(BASE / 'summary.json')
    if complete['status'] not in ('complete', 'reported', 'predictions_locked') or not complete.get('all_attempts_locked_before_scoring'):
        raise ValueError('Original predictions are not locked')
    if complete['protocol']['sha256'] != sha256(BASE / 'protocol.json'):
        raise ValueError('Original protocol hash changed')
    api.verify(source['source_pins'])
    pins = dict(source['source_pins']); cases = deepcopy(source['cases'])
    contract = api.engine_contract(); pins.update(contract['code_pins'])
    units, computations, reused = [], {}, {}
    run.mkdir(parents=True)
    for case in cases:
        for sampling in SAMPLING:
            raw = api.checked(case['sampling_sources'][sampling])
            training = visible_training(raw)
            request = api.checked(case['requests']['evidence']['development'])
            if training['human'] != request['human'] or training['dosing'] != request['dosing']:
                raise ValueError('Development request and observation conditions differ')
            location = run / 'inputs' / case['case_id'] / (sampling + '.json')
            location.parent.mkdir(parents=True, exist_ok=True)
            save(location, training)
            identity = uid(case['case_id'], sampling, METHODS[0])
            computations[identity] = {'case_id': case['case_id'], 'sampling': sampling, 'method': METHODS[0],
                'request': deepcopy(case['requests']['evidence']['development']),
                'target_request': deepcopy(case['requests']['evidence']['target']),
                'training_source': {**api.ref(location), 'role': 'calibration_data_reconstruction'},
                'optimizer_budget': deepcopy(api.OPTIMIZER_BUDGET)}
            for method in METHODS:
                key = uid(case['case_id'], sampling, method)
                units.append({'id': key, 'case_id': case['case_id'], 'sampling': sampling, 'method': method})
                if method == METHODS[0]:
                    continue
                result = deepcopy(complete['units'][key]); target = api.checked(result['request'])
                original = api.checked(case['requests']['evidence']['target'])
                if any(target[k] != original[k] for k in ('human', 'dosing')):
                    raise ValueError('Comparator target conditions changed')
                result.update(reused_from=api.ref(BASE / 'summary.json'), original_resources=deepcopy(result['resources']),
                    resources={'actual_new_native_calls': 0, 'total_native_calls': 0, 'linear_solver_calls': 0,
                        'wall_seconds': 0, 'cache_reused': True})
                reused[key] = result
    scenarios = api.read(INPUT_RESPONSE)
    if len(scenarios['cases']) != 9 or scenarios['compound'] != 'Treprostinil':
        raise ValueError('Original nine Treprostinil input requests required')
    auxiliary = {'development': list(computations), 'input_response': [
        {k: deepcopy(c[k]) for k in ('case_id', 'dose_multiplier', 'condition', 'request')} for c in scenarios['cases']],
        'input_response_source': api.ref(INPUT_RESPONSE), 'input_response_notes': scenarios['notes']}
    for value in (cases, computations, reused, auxiliary):
        api.pin_refs(value, pins)
    for path in (Path(__file__), ROOT / 'scripts/run_blueprint_transport.py',
            ROOT / 'scripts/run_blueprint_learning.py', ROOT / 'scripts/report_blueprint_validation.py',
            BASE / 'protocol.json', BASE / 'summary.json', INPUT_RESPONSE):
        pins[str(path.relative_to(ROOT))] = sha256(path)
    protocol = {'version': VERSION, 'runner_config': CONFIG, 'created_at': now(), 'cases': cases,
        'methods': list(METHODS), 'sampling': list(SAMPLING), 'formulas': list(FORMULAS), 'units': units,
        'computations': computations, 'auxiliary': auxiliary, 'engine': contract,
        'source_protocol': api.ref(BASE / 'protocol.json'), 'source_summary': api.ref(BASE / 'summary.json'),
        'source_pins': pins, 'workers': workers, 'native_threads': 2, 'native_timeout_seconds': 180,
        'optimizer_budget': deepcopy(api.OPTIMIZER_BUDGET), 'monitor_seconds': 1200,
        'directory_limit_bytes': 10 * 1024**3, 'scope': CONFIG['scope'],
        'fairness': ['No old fit supplies initialization; sixteen fixed starts per formula, max 400 optimizer evaluations each.',
            'All development inputs and full/five subsets are identical to the original ten-drug validation.',
            'The unchanged frozen two-compartment controls reuse their original development-only fits and predictions.',
            'All 20 target, 20 development and nine hypothetical predictions lock before clinical target scoring.',
            'The five-point label concerns curve observations only; Moxifloxacin CL comes from the same development study.',
            'This previously inspected cohort is development evaluation, not new blinded testing.',
            'Nine Treprostinil scenarios have no corresponding observations and are not validation samples.',
            'Missing original clinical endpoints remain missing; no clinical reference is inferred from a prediction.',
            'The effective reversible exchange hypothesis does not validate organ concentrations or measured permeability.']}
    api.verify(pins)
    save(run / 'protocol.json', protocol)
    save(run / 'state.json', {'version': VERSION, 'status': 'prepared', 'protocol': api.ref(run / 'protocol.json'),
        'created_at': now(), 'fits': {}, 'computations': reused, 'monitor': {}})
    return {'drugs': 10, 'new_formula_fits': 40, 'target_predictions': 20,
        'development_predictions': 20, 'hypothetical_predictions': 9, 'reused_comparators': 40}


def auxiliary_predictions(run, protocol, state):
    """Generate and seal development/what-if curves without decoding outcomes."""
    receipt = run / 'auxiliary_locked.json'
    if receipt.exists():
        result = api.read(receipt); api.pin_refs(result, {})
        scenario_ids = {s['case_id'] for s in protocol['auxiliary']['input_response']}
        if (result.get('version') != VERSION or set(result['development']) != set(protocol['computations'])
                or set(result['input_response']) != scenario_ids
                or not result.get('all_29_attempts_locked_before_scoring')
                or not result.get('monitor_stopped_before_scoring')
                or any(r['status'] not in ('predicted', 'failed')
                    for group in ('development', 'input_response') for r in result[group].values())):
            raise ValueError('Auxiliary prediction matrix changed')
        return result
    controller = Controller(run, protocol, state)
    controller.check(force=True)
    monitor = threading.Thread(target=controller.watch, daemon=True); monitor.start()
    choices = api.checked(state['formula_selection'])['choices']
    jobs = [('development', identity, api.checked(config['request']), identity, {})
        for identity, config in protocol['computations'].items()]
    trepro = next(c['case_id'] for c in protocol['cases'] if c['drug'] == 'Treprostinil')
    for scenario in protocol['auxiliary']['input_response']:
        jobs.append(('input_response', scenario['case_id'], deepcopy(scenario['request']),
            uid(trepro, 'full', METHODS[0]), {k: scenario[k] for k in ('dose_multiplier', 'condition')}))
    result = {'version': VERSION, 'development': {}, 'input_response': {},
        'target_outcomes_used': False, 'parameters_refitted': False}

    def one(kind, identity, request, fitted_identity, metadata):
        controller.check(); choice = choices[fitted_identity]
        if choice is None:
            return kind, identity, {**metadata, 'status': 'failed', 'failure': 'No eligible development mapping'}
        request['partition_method'] = choice['formula']
        folder = run / 'auxiliary' / kind / identity
        def incomplete(path):
            return path.exists() and not (path / 'completion.json').exists() and any(
                p.name != '.calibration.lock' for p in path.iterdir())
        if incomplete(folder):
            folder = folder.with_name(identity + '_retry_1')
            if incomplete(folder):
                raise Paused('Two incomplete auxiliary attempts retained; inspect before retry')
        predicted = api.engine().predict_mapping(request=request, mapping_path=ROOT / choice['mapping']['path'],
            output_dir=folder, project_root=ROOT, before_trial=controller.check)
        cost = api.read(folder / 'result.json')
        row = {**metadata, 'status': predicted['status'], 'mapping': choice['mapping'],
            'request': api.ref(folder / 'request.json'), 'result': api.ref(folder / 'result.json'),
            'resources': {k: cost.get(k) for k in ('actual_new_calls', 'actual_native_solver_calls',
                'linear_solver_calls', 'elapsed_seconds')}}
        if folder.name.endswith('_retry_1'):
            row['resources']['prior_incomplete_attempt_costs_unknown'] = True
        if predicted['status'] == 'predicted':
            row.update(prediction=api.ref(folder / 'prediction_locked.json'), endpoints=api.ref(folder / 'endpoints.json'))
        else:
            row['failure'] = predicted
        return kind, identity, row
    try:
        with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
            for future in as_completed([pool.submit(one, *j) for j in jobs]):
                kind, identity, row = future.result(); result[kind][identity] = row
        controller.closed.set(); monitor.join(timeout=30)
        if monitor.is_alive():
            raise Paused('Auxiliary monitor still running')
        api.verify(protocol['source_pins']); api.pin_refs(result, {})
        result.update(locked_at=now(), all_29_attempts_locked_before_scoring=True, monitor_stopped_before_scoring=True)
        save(receipt, result)
        return result
    finally:
        controller.closed.set(); monitor.join(timeout=30)


original_statistics = api.statistics
def statistics(records, cases):
    result = original_statistics(records, cases)
    result['interpretation'] = 'Previously inspected ten-drug development cohort; study-cluster bootstrap with 2000 draws, not confirmatory significance.'
    return result


def report(run):
    from report_blueprint_validation import CURVE, UNITS, Store, _case_reference, _score_prediction, _unit_row, positive
    run = own_run(run); protocol, state = api.load_batch(run); validate_matrix(protocol, state, locked=True)
    if state['status'] == 'complete':
        result = api.checked(state['results']); api.verify(result['source_pins']); api.pin_refs(result['auxiliary'], {})
        return result
    auxiliary = auxiliary_predictions(run, protocol, state)
    store = Store(ROOT)
    store.ref(state['protocol'])
    for name, digest in protocol['source_pins'].items():
        store.checksum(name, digest)
    generated = {}; api.pin_refs([state['fits'], state['computations'], state['formula_selection'], auxiliary], generated)
    for name, digest in generated.items():
        store.checksum(name, digest)
    store.ref(api.ref(run / 'auxiliary_locked.json'))
    # First decoding of clinical target observations is after BOTH complete locks.
    references = {c['case_id']: _case_reference(c, store) for c in protocol['cases']}
    cases = {c['case_id']: c for c in protocol['cases']}; records, rows = [], []
    for unit in protocol['units']:
        case = cases[unit['case_id']]; clinical, curve, endpoints = references[unit['case_id']]
        computation = state['computations'][unit['id']]
        record = {'unit_id': unit['id'], 'case_id': unit['case_id'], 'drug': case['drug'], 'study_id': case['study_id'],
            'cluster_id': case['cluster_id'], 'sampling': unit['sampling'], 'method': unit['method'],
            'attempt_status': computation['status'], 'scoring_status': 'failed', 'failure': computation.get('failure', ''),
            'resources': deepcopy(computation.get('resources', {})), 'original_resources': deepcopy(computation.get('original_resources', {})),
            'prediction_source': computation.get('prediction', {}), 'request_source': computation.get('request', {}),
            'clinical_references': deepcopy(endpoints), 'observed_curve': deepcopy(curve),
            'source_approximations': deepcopy(clinical.get('approximations', []))}
        if computation['status'] == 'predicted':
            try:
                scored, _, _ = _score_prediction(case, clinical, curve, endpoints, computation, store)
                record.update(scored)
                record['scoring_status'] = 'scored' if positive(record.get('curve_metrics', {}).get('gmfe')) else 'partial'
                if unit['method'] == METHODS[0]:
                    folder = ROOT / computation['prediction']['path']
                    record['dense_endpoints_source'] = api.ref(folder.parent / 'endpoints.json')
                    store.ref(record['dense_endpoints_source'])
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                record.update(failure=str(exc), failure_category=getattr(exc, 'category', 'clinical_scoring_error'))
        records.append(record)
        rows.extend(_unit_row(record, endpoint, {'curve': curve, 'endpoints': endpoints}) for endpoint in (CURVE, *UNITS))
    result = {'version': VERSION, 'scope': protocol['scope'], 'engine': protocol['engine'], 'units': records,
        'statistics': statistics(records, protocol['cases']), 'fits': state['fits'], 'formula_selection': state['formula_selection'],
        'all_attempts_locked_before_scoring': True, 'auxiliary': auxiliary,
        'auxiliary_source': api.ref(run / 'auxiliary_locked.json'), 'source_pins': dict(store.pins),
        'limitations': protocol['fairness'], 'endpoint_rows': rows}
    auxiliary_costs = [row.get('resources', {}) for name in ('development', 'input_response')
        for row in auxiliary[name].values()]
    result['auxiliary_resources'] = {'scope': 'Additional development/what-if predictions; not included in comparison-unit cost rows',
        'actual_native_calls_recorded': sum(r.get('actual_new_calls', 0) or 0 for r in auxiliary_costs),
        'linear_solver_calls_recorded': sum(r.get('linear_solver_calls', 0) or 0 for r in auxiliary_costs),
        'sum_wall_seconds': sum(r.get('elapsed_seconds', 0) or 0 for r in auxiliary_costs),
        'all_attempt_costs_exact': not any(r.get('prior_incomplete_attempt_costs_unknown') for r in auxiliary_costs)}
    excluded = {'Nemonoxacin', 'Dexmedetomidine', 'Midazolam'}
    result['source_sensitivity'] = statistics([r for r in records if r['drug'] not in excluded],
        [c for c in protocol['cases'] if c['drug'] not in excluded])
    result['comparison_table'] = api.comparison_table(records, result['statistics'])
    store.unchanged()
    save(run / 'results.json', result)
    with (run / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['comparison_table'][0])); writer.writeheader(); writer.writerows(result['comparison_table'])
    return result


def background(run):
    import psutil
    run = own_run(run)
    if api.status(run).get('alive'):
        raise ValueError('This run already has a live process')
    with (run / 'run.log').open('a') as log:
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--run-dir', str(run)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    save(run / 'pid.json', {'pid': process.pid, 'uid': os.getuid(), 'create_time': psutil.Process(process.pid).create_time(),
        'started_at': now(), 'script': str(Path(__file__).relative_to(ROOT))})
    return {'pid': process.pid, 'log': str((run / 'run.log').relative_to(ROOT))}


# Instance-local adapters; no frozen source file or imported module global is changed.
api.own_run = own_run
api.prepare = prepare
api.validate_matrix = validate_matrix
api.statistics = statistics
api.report = report
api.background = background


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'report', 'cleanup', 'pause', 'resume'))
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args(); run = own_run(args.run_dir)
    if args.command == 'prepare': result = prepare(run, args.workers)
    elif args.command == 'status': result = api.status(run)
    elif args.command == 'run': result = background(run) if args.background else api.execute(run)
    elif args.command == 'pause':
        save(run / 'PAUSE', {'requested_at': now()}); result = {'status': 'pause_requested'}
    elif args.command == 'resume':
        if api.status(run).get('alive'): raise ValueError('Existing process still active')
        (run / 'PAUSE').unlink(missing_ok=True)
        result = background(run) if args.background else api.execute(run)
    elif args.command == 'cleanup': result = api.cleanup(run)
    else:
        with (run / '.run.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = {'status': 'reported', 'units': len(report(run)['units'])}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
