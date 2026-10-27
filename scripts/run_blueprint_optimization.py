#!/usr/bin/env python3
"""Bounded optimizer development pilot; all target predictions precede scoring."""
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
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from run_blueprint_learning import Controller, Paused, now

VERSION = 'blueprint-optimizer-pilot-1'
GROUP = ROOT / 'experiments/blueprint_design/optimization_20260929'
DEFAULT = GROUP / 'pilot'
BASE = ROOT / 'experiments/blueprint_design/evidence_policy_20260929'
PILOT_DRUGS = ('Remdesivir', 'Edaravone', 'Moxifloxacin', 'Cefepime')
VARIANTS = ('trust_region', 'trust_region_sparse', 'trust_region_robust', 'pattern')
BASELINES = ('frozen_policy', 'two_compartment', 'two_compartment_prior')
METHODS = VARIANTS + BASELINES
SAMPLING = ('full', 'five')
FORMULAS = ('PKSimStandard', 'RodgersRowland')


def read(path):
    return json.loads(inside(path, ROOT, exists=True).read_text())


def ref(path):
    path = inside(path, ROOT, exists=True)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def checked(source):
    path = inside(source['path'], ROOT, exists=True)
    if sha256(path) != source['sha256']:
        raise ValueError('Frozen source changed: ' + str(path))
    return read(path)


def verify(pins):
    for path, expected in pins.items():
        if sha256(inside(path, ROOT, exists=True)) != expected:
            raise ValueError('Frozen source/code changed: ' + path)


def own_run(path):
    raw = Path(path) if Path(path).is_absolute() else ROOT / path
    if raw.is_symlink() or GROUP.is_symlink():
        raise ValueError('Optimization runs cannot be symbolic links')
    run = inside(raw, ROOT)
    if run.parent != GROUP.resolve():
        raise ValueError('Use an immediate child of the optimization experiment group')
    return run


def unit_id(case_id, sampling, method):
    return '__'.join((case_id, sampling, method))


def fit_id(unit, formula):
    return unit + '__' + formula


def target_signature(value):
    """The validated optional terminal fit has the same None/omitted default."""
    result = deepcopy(value)
    if result.get('terminal_fit') is None:
        result.pop('terminal_fit', None)
    return result


def _pin_refs(value, pins):
    """Hash references without decoding clinical target documents."""
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            path = inside(value['path'], ROOT, exists=True)
            if sha256(path) != value['sha256']:
                raise ValueError('Frozen embedded source changed: ' + str(path))
            pins[str(path.relative_to(ROOT))] = value['sha256']
        for child in value.values():
            _pin_refs(child, pins)
    elif isinstance(value, list):
        for child in value:
            _pin_refs(child, pins)


def prepare(run, workers=14, all_ten=False):
    """Reuse exact old visible inputs; never read target outcomes or fit warm starts."""
    from pbpk_kb.blueprint_policy.adaptive_calibration import objective_config
    run = own_run(run)
    if run.exists():
        raise FileExistsError('Preserve this batch; resume it or choose another child directory')
    if not 1 <= workers <= 14:
        raise ValueError('Use one to fourteen native workers')
    source = read(BASE / 'protocol.json')
    completed = read(BASE / 'state.json')
    if source.get('version') != 'blueprint-evidence-policy-1' or completed.get('status') != 'complete':
        raise ValueError('Baselines must come from the completed frozen evidence-policy experiment')
    verify(source['source_pins'])
    cases = [deepcopy(c) for c in source['cases'] if all_ten or c['drug'] in PILOT_DRUGS]
    if len(cases) != (10 if all_ten else len(PILOT_DRUGS)):
        raise ValueError('The requested frozen drug cohort is incomplete')
    available = {(u['case_id'], u['sampling'], u['method']): u for u in source['units'] if u['scenario'] == 'available'}
    units, specifications, reused = [], {}, {}
    pins = dict(source['source_pins'])
    for case in cases:
        for sampling in SAMPLING:
            old_unit = available[case['case_id'], sampling, 'evidence_policy']
            native = source['computations'][old_unit['computation_id']]
            request = checked(native['request'])
            target = checked(native['target_request'])
            training = checked(native['training_source'])
            if sampling == 'five' and len(training['observations']) != 5:
                raise ValueError('Five-point input isolation differs from the frozen source')
            if any(key in training for key in ('raw_observations', 'full_curve', 'fit', 'selection')):
                raise ValueError('Training input contains forbidden full-curve/fit metadata')
            if training['human'] != request['human'] or training['dosing'] != request['dosing']:
                raise ValueError('Development curve human/dosing differs from the construction request')
            for method in METHODS:
                identity = unit_id(case['case_id'], sampling, method)
                unit = {'id': identity, 'case_id': case['case_id'], 'sampling': sampling, 'method': method,
                    'kind': 'native' if method in VARIANTS else 'frozen_baseline'}
                units.append(unit)
                if method in VARIANTS:
                    specifications[identity] = {'request': deepcopy(native['request']),
                        'target_request': deepcopy(native['target_request']), 'training_source': deepcopy(native['training_source']),
                        'case_id': case['case_id'], 'sampling': sampling, 'method': method,
                        'objective': objective_config(method, len(training['observations']))}
                else:
                    old_method = 'evidence_policy' if method == 'frozen_policy' else method
                    existing_unit = available[case['case_id'], sampling, old_method]
                    existing_config = source['computations'][existing_unit['computation_id']]
                    if existing_config['training_source'] != native['training_source']:
                        raise ValueError('A reused baseline used a different development subset')
                    other_request = checked(existing_config['request'])
                    for key in ('human', 'dosing'):
                        if other_request[key] != request[key]:
                            raise ValueError('A reused baseline used different development conditions')
                    if method == 'two_compartment_prior':
                        cl_parameters = {'total_clearance', 'hepatic_clearance', 'renal_clearance', 'elimination_scope'}
                        def cl_inputs(document):
                            return sorted([(e['parameter'], e['value'], e['unit']) for e in document.get('evidence', [])
                                if e['parameter'] in cl_parameters], key=lambda e: e[0])
                        if cl_inputs(other_request) != cl_inputs(request):
                            raise ValueError('The reused two-compartment prior differs from PBPK clearance evidence')
                    result = deepcopy(completed['computations'][existing_unit['computation_id']])
                    if result.get('status') not in ('predicted', 'failed'):
                        raise ValueError('A reused baseline is unfinished')
                    if result['status'] == 'predicted':
                        actual_target = checked(result['request'])
                        for key in ('human', 'dosing', 'target'):
                            left, right = actual_target[key], target[key]
                            if key == 'target':
                                left, right = target_signature(left), target_signature(right)
                            if left != right:
                                raise ValueError('A reused baseline has a different target request')
                    result['frozen_computation_id'] = existing_unit['computation_id']
                    result['reused_from'] = ref(BASE / 'state.json')
                    result['original_resources'] = deepcopy(result.get('resources', {}))
                    result['resources'] = {'actual_new_native_calls': 0, 'wall_seconds': 0,
                        'cached_native_calls': result['original_resources'].get('total_native_calls', 0),
                        'cache_reused': True, 'resource_scope': 'Frozen baseline reuse; original work retained separately'}
                    reused[identity] = result
    for value in (cases, specifications, reused):
        _pin_refs(value, pins)
    code = [Path(__file__), ROOT / 'scripts/run_blueprint_learning.py', ROOT / 'scripts/report_blueprint_validation.py',
        ROOT / 'src/pbpk_kb/blueprint_policy/adaptive_calibration.py',
        ROOT / 'src/pbpk_kb/blueprint_policy/adaptive_optimizer.py', BASE / 'protocol.json', BASE / 'state.json']
    for path in code:
        pins[str(path.relative_to(ROOT))] = sha256(path)
    protocol = {'version': VERSION, 'created_at': now(), 'cases': cases, 'methods': list(METHODS),
        'sampling': list(SAMPLING), 'formulas': list(FORMULAS), 'units': units, 'computations': specifications,
        'source_pins': pins, 'workers': workers, 'native_threads': 2, 'native_timeout_seconds': 180,
        'native_evaluations_per_formula': 24, 'monitor_seconds': 1200, 'directory_limit_bytes': 10 * 1024**3,
        'scope': 'Previously viewed development drugs; optimizer pilot, not blinded validation',
        'fairness': ['Every method uses the exact same frozen full/five development subset.',
            'Frozen policy includes local screening; new pattern excludes screening, so total changes cannot be attributed solely to TRF.',
            'TRF versus new pattern isolates search with the same quadratic objective and native budget.',
            'Sparse and robust objectives are separate declared engineering hypotheses; their fit objectives are not comparable across variants.',
            'Formulas are selected only within each variant using its development objective.',
            'All target predictions lock before target outcomes are decoded; no fit warm starts or outcome-guided method selection.'],
        'bootstrap': {'seed': 20260929, 'resamples': 2000},
        'retention': 'Keep final predictions, source pins, scores, selected fits and failure diagnostics; clean only successful unselected native executions.'}
    run.mkdir(parents=True)
    save(run / 'protocol.json', protocol)
    save(run / 'state.json', {'version': VERSION, 'status': 'prepared', 'protocol': ref(run / 'protocol.json'),
        'created_at': now(), 'fits': {}, 'computations': reused, 'monitor': {}})
    return {'drugs': len(cases), 'new_formula_fits': len(specifications) * len(FORMULAS),
        'new_predictions': len(specifications), 'reused_predictions': len(reused), 'comparison_units': len(units)}


def _validate_locked(protocol, state):
    if protocol.get('version') != VERSION or state.get('version') != VERSION:
        raise ValueError('Wrong optimization protocol version')
    for field, declared in (('methods', METHODS), ('sampling', SAMPLING), ('formulas', FORMULAS)):
        if protocol.get(field) != list(declared):
            raise ValueError('Changed or duplicated experiment conditions: ' + field)
    case_ids = [c['case_id'] for c in protocol['cases']]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError('Drug cases must be nonempty and unique')
    expected = {unit_id(c['case_id'], s, m) for c in protocol['cases'] for s in SAMPLING for m in METHODS}
    if len(protocol['units']) != len(expected) or {u['id'] for u in protocol['units']} != expected:
        raise ValueError('Incomplete or duplicated comparison matrix')
    if set(state['computations']) != expected or any(v.get('status') not in ('predicted', 'failed') for v in state['computations'].values()):
        raise ValueError('All prediction attempts must be terminal before scoring')
    if (state.get('status') not in ('predictions_locked', 'complete')
            or not state.get('all_attempts_locked_before_scoring') or not state.get('monitor_stopped_before_scoring')):
        raise ValueError('Prediction lock and stopped monitor are required before target outcomes')
    fits = {fit_id(unit, formula) for unit in protocol['computations'] for formula in FORMULAS}
    if set(state['fits']) != fits:
        raise ValueError('All formula fits must be locked before formula selection and scoring')


def choose_formula(mappings):
    if len(mappings) != len(FORMULAS) or {m['construction_partition_method'] for m in mappings} != set(FORMULAS):
        raise ValueError('Both predeclared formulas must finish before selection')
    eligible = [m for m in mappings if m.get('eligible')]
    return min(eligible, key=lambda m: (m['decision']['objective'], m['construction_partition_method'])) if eligible else None


def execute(run):
    from pbpk_kb.blueprint_policy.adaptive_calibration import fit_mapping, predict_mapping, cleanup_unselected
    from pbpk_kb.blueprints.evidence_preserving_calibration import _infrastructure_problem
    run = own_run(run)
    with (run / '.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, state = read(run / 'protocol.json'), read(run / 'state.json')
        if protocol['version'] != VERSION or state['version'] != VERSION:
            raise ValueError('Wrong batch version')
        if state['status'] == 'complete':
            return {'status': 'complete', 'reused': True}
        controller = Controller(run, protocol, state)
        controller.check(force=True)
        monitor = threading.Thread(target=controller.watch, daemon=True)
        monitor.start()
        controller.persist(status='running', started_at=state.get('started_at', now()))

        def fit_one(identity, config, formula):
            controller.check()
            key = fit_id(identity, formula)
            folder = run / 'work' / identity / formula
            if key in state['fits']:
                checked(state['fits'][key]['mapping'])
                return
            request = checked(config['request'])
            request['partition_method'] = formula
            mapping = fit_mapping(reference_request=request, training_source=config['training_source'], output_dir=folder,
                project_root=ROOT, variant=config['method'], max_evaluations=24, before_trial=controller.check)
            cleaned = cleanup_unselected(output_dir=folder, project_root=ROOT)
            with controller.lock:
                state['fits'][key] = {'mapping': mapping['mapping'], 'status': mapping['status'], 'eligible': mapping['eligible'],
                    'native_calls': mapping['total_native_calls'], 'cleanup': cleaned}
                controller.persist()

        def predict_one(identity, config):
            controller.check()
            if identity in state['computations']:
                _pin_refs(state['computations'][identity], {})
                return
            mappings = [checked(state['fits'][fit_id(identity, formula)]['mapping']) for formula in FORMULAS]
            chosen = choose_formula(mappings)
            resources = {'total_native_calls': sum(m['total_native_calls'] for m in mappings),
                'actual_new_native_calls': sum(m['total_native_calls'] for m in mappings),
                'wall_seconds': sum(m.get('elapsed_seconds', 0) for m in mappings),
                'failed_native_trials': sum(m['failed_evaluation_count'] for m in mappings), 'cache_reused': False}
            if chosen is None:
                result = {'status': 'failed', 'failure': 'No eligible formula fit', 'resources': resources}
            else:
                formula = chosen['construction_partition_method']
                mapping_ref = state['fits'][fit_id(identity, formula)]['mapping']
                request = checked(config['target_request'])
                request['partition_method'] = formula
                folder = run / 'work' / identity / 'target'
                incomplete = lambda p: p.exists() and not (p / 'completion.json').exists() and any(x.name != '.calibration.lock' for x in p.iterdir())
                if incomplete(folder):
                    folder = folder.with_name('target_retry_1')
                    if incomplete(folder):
                        raise Paused('Two incomplete target attempts retained; inspect before retry')
                result_native = predict_mapping(request=request, mapping_path=ROOT / mapping_ref['path'],
                    output_dir=folder, project_root=ROOT)
                if _infrastructure_problem(result_native):
                    raise Paused('Target native infrastructure failure')
                resources['total_native_calls'] += 1 + (folder.name == 'target_retry_1')
                resources['actual_new_native_calls'] = resources['total_native_calls']
                resources['wall_seconds'] += result_native.get('elapsed_seconds', 0)
                result = {'status': 'predicted' if result_native.get('status') == 'predicted' else 'failed',
                    'native_status': result_native.get('status'), 'mapping': mapping_ref,
                    'request': ref(folder / 'request.json'), 'selected_formula': formula,
                    'selection_basis': 'Within-variant development objective only', 'resources': resources}
                if result['status'] == 'predicted':
                    result['prediction'] = ref(folder / 'prediction_locked.json')
                else:
                    result['failure'] = result_native
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
                                state.setdefault('interruptions', []).append({'unit': futures[future], 'time': now(),
                                    'error': type(exc).__name__ + ': ' + str(exc)})
                                controller.pause(type(exc).__name__ + ': ' + str(exc))
            controller.check(force=True)

        try:
            items = sorted(protocol['computations'].items())
            phase([(identity, config, formula) for identity, config in items for formula in FORMULAS], fit_one)
            controller.persist(status='predicting_targets', formulas_locked_before_target_prediction=True)
            phase(items, predict_one)
            # No thread may mutate state while Store validates immutable sources.
            controller.closed.set()
            monitor.join(timeout=30)
            if monitor.is_alive():
                raise Paused('Monitor did not stop; scoring deferred to avoid a source race')
            controller.persist(status='predictions_locked', all_attempts_locked_before_scoring=True,
                monitor_stopped_before_scoring=True, locked_at=now())
            _validate_locked(protocol, state)
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
    groups = {}
    for sampling in SAMPLING:
        groups[sampling] = {}
        for method in METHODS:
            selected = [lookup[cid, sampling, method] for cid in ids]
            scores = {cid: v for cid, v in values(sampling, method).items() if positive(v)}
            curve = clustered_geomean(scores, {cid: clusters[cid] for cid in scores})
            curve.update(attempted=len(ids), failed=sum(r['attempt_status'] == 'failed' for r in selected),
                scored=len(scores), coverage=len(scores) / len(ids), within_twofold=sum(x <= 2 for x in scores.values()))
            endpoints = {}
            for endpoint in UNITS:
                errors = {r['case_id']: r.get('endpoints', {}).get(endpoint, {}).get('error') for r in selected}
                errors = {cid: value for cid, value in errors.items() if value is not None}
                item = {'scored': len(errors), 'reference_available': sum(r['clinical_references'][endpoint]['value'] is not None for r in selected),
                    'weak_scored': sum(r.get('endpoints', {}).get(endpoint, {}).get('error') is not None and
                        r['endpoints'][endpoint].get('weak_tail', False) for r in selected)}
                if endpoint == 'tmax':
                    item['mean_absolute_error_h'] = math.fsum(errors.values()) / len(errors) if errors else None
                else:
                    item['fold_error'] = clustered_geomean(errors, {cid: clusters[cid] for cid in errors})
                    item['within_twofold'] = sum(x <= 2 for x in errors.values())
                endpoints[endpoint] = item
            groups[sampling][method] = {'curve': curve, 'endpoints': endpoints}
    paired = {s: {m: {base: paired_ratio(values(s, m), values(s, base), clusters, expected_ids=ids)
        for base in ('pattern', *BASELINES) if base != m} for m in VARIANTS} for s in SAMPLING}
    degradation = {m: paired_ratio(values('five', m), values('full', m), clusters, expected_ids=ids) for m in METHODS}
    return {'by_sampling': groups, 'paired_curve_gmfe_ratios': paired, 'five_to_full_curve_gmfe_ratios': degradation,
        'interpretation': 'Study-cluster paired bootstrap, 2000 resamples; very small development pilot, no confirmatory significance claim.'}


def report(run):
    from report_blueprint_validation import CURVE, UNITS, Store, _case_reference, _score_prediction, _unit_row, positive
    run = own_run(run)
    store = Store(ROOT)
    protocol, state = store.read(run / 'protocol.json'), store.read(run / 'state.json')
    _validate_locked(protocol, state)
    store.ref(state['protocol'])
    for name, digest in protocol['source_pins'].items():
        store.checksum(name, digest)
    generated_pins = {}
    _pin_refs(state['fits'], generated_pins)
    _pin_refs(state['computations'], generated_pins)
    for name, digest in generated_pins.items():
        store.checksum(name, digest)
    if state['status'] == 'complete':
        result = store.ref(state['results'])
        for name, digest in result['source_pins'].items():
            store.checksum(name, digest)
        store.unchanged()
        return result
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
            'resources': deepcopy(computation.get('resources', {})), 'prediction_source': computation.get('prediction', {}),
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
    result = {'version': VERSION, 'scope': protocol['scope'], 'all_attempts_locked_before_scoring': True,
        'units': records, 'statistics': statistics(records, protocol['cases']), 'fits': deepcopy(state['fits']),
        'source_pins': {name: value for name, value in store.pins.items() if store.path(name) != run / 'state.json'},
        'limitations': protocol['fairness'] + ['Missing clinical values are not imputed.',
            'All drugs and target studies were previously viewed; this run develops methods, not a new blind evaluation.',
            'No aggregate winner is selected across incompatible endpoint dimensions or objective variants.']}
    store.unchanged()
    with tempfile.TemporaryDirectory(prefix='.optimizer-report-', dir=run) as temporary:
        temporary = Path(temporary)
        (temporary / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        with (temporary / 'results.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
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
    result = {'status': state['status'], 'drugs': len(protocol['cases']), 'comparison_units': len(protocol['units']),
        'fits_completed': len(state['fits']), 'fits_total': len(protocol['computations']) * len(FORMULAS),
        'new_predictions_completed': sum(key in state['computations'] for key in protocol['computations']),
        'new_predictions_total': len(protocol['computations']),
        'frozen_baselines_reused': len(state['computations']) - sum(key in state['computations'] for key in protocol['computations']),
        'native_fit_calls_recorded': sum(item.get('total_native_calls', 0) for item in progress),
        'monitor': state.get('monitor'), 'pause_reason': state.get('pause_reason')}
    if (run / 'pid.json').exists():
        import psutil
        pid = read(run / 'pid.json')
        result.update(pid=pid['pid'], alive=False)
        try:
            process = psutil.Process(pid['pid'])
            result['alive'] = process.uids().real == os.getuid() and abs(process.create_time() - pid['create_time']) < .01 and process.status() != 'zombie'
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return result


def background(run):
    import psutil
    run = own_run(run)
    if status(run).get('alive'):
        raise ValueError('This experiment already has its own live process')
    with (run / 'run.log').open('a') as log:
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--run-dir', str(run)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    save(run / 'pid.json', {'pid': process.pid, 'uid': os.getuid(), 'create_time': psutil.Process(process.pid).create_time(),
        'started_at': now(), 'script': str(Path(__file__).relative_to(ROOT))})
    return {'pid': process.pid, 'log': str((run / 'run.log').relative_to(ROOT))}


def cleanup(run):
    from pbpk_kb.blueprint_policy.adaptive_calibration import cleanup_unselected
    run = own_run(run)
    if status(run).get('alive'):
        raise ValueError('The active runner already cleans completed fits')
    items = [cleanup_unselected(output_dir=path.parent, project_root=ROOT)
        for path in sorted((run / 'work').glob('*/*/mapping_locked.json'))]
    return {'fits_checked': len(items), 'removed_count': sum(item['removed_count'] for item in items)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'report', 'pause', 'resume', 'cleanup'))
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--all-ten', action='store_true')
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    run = own_run(args.run_dir)
    if args.command == 'prepare':
        result = prepare(run, args.workers, args.all_ten)
    elif args.command == 'status':
        result = status(run)
    elif args.command == 'run':
        result = background(run) if args.background else execute(run)
    elif args.command == 'pause':
        save(run / 'PAUSE', {'requested_at': now(), 'reason': 'Pause requested; no new native submissions'})
        result = {'status': 'pause_requested'}
    elif args.command == 'resume':
        if status(run).get('alive'):
            raise ValueError('Wait for existing native calls and the current runner to stop')
        (run / 'PAUSE').unlink(missing_ok=True)
        state = read(run / 'state.json')
        state.pop('pause_reason', None)
        state['status'] = 'prepared'
        save(run / 'state.json', state)
        result = background(run) if args.background else execute(run)
    elif args.command == 'cleanup':
        result = cleanup(run)
    else:
        with (run / '.run.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report_result = report(run)
            result = {'status': 'reported', 'units': len(report_result['units'])}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
