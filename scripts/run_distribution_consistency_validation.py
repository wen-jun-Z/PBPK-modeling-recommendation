#!/usr/bin/env python3
"""Evaluate one predeclared distribution fallback without training or fitting.

All source-qualified cases and input conditions remain present. Only a failure
caused exclusively by the immutable vascular Vss floor can trigger one Schmitt
native execution. Successful predictions and other failures remain unchanged.
Clinical observations are opened only after every prediction/failure is locked.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_qualified_binding_validation as source_runner
from pbpk_kb.blueprints.source_workflow import save, sha256

base = source_runner.base
value, retained, clone = base.value, base.retained, base.clone
VERSION = 'distribution-consistency-validation-1'
CANDIDATE = 'distribution_consistency'
MODULE = 'pbpk_kb.blueprint_policy.distribution_consistency'
SOURCE = source_runner.DEFAULT
DEFAULT = ROOT/'experiments/blueprint_design/distribution_consistency_20261007'
CONDITIONS = base.CONDITIONS
FLOOR_REASON = 'vss_target_at_or_below_immutable_vascular_observer_floor'


def own(path):
    path = value.inside(path, ROOT)
    if path != DEFAULT:
        raise ValueError('Use only distribution_consistency_20261007')
    return path


def floor_failure(entry):
    return importlib.import_module(MODULE).eligible_capacity_failure(entry)


def source_entry(task):
    """Verify complete source inputs; never load the source clinical scores."""
    original = task['source_task']
    if source_runner.exact_input_digest(original) != task['source_exact_input_digest']:
        raise ValueError('Frozen source request, sidecars or core differs')
    if (task['request'] != original['request'] or task['sidecars'] != original['sidecars']
            or retained.key_for(original['request'], original['sidecars'], original['core']) != original['key']):
        raise ValueError('The candidate may not change parameter inputs or source identity')
    outcome = value.old.checked(task['source_prediction_receipt'])
    entry = outcome['entry']
    if (outcome.get('key') != original['key'] or not outcome.get('locked_at')
            or entry.get('status') not in {'predicted', 'failed'}
            or entry.get('request') != task['request']
            or entry.get('target_observations_used') is not False):
        raise ValueError('Source outcome is not an exact-input, locked prediction/failure')
    if not task.get('construction_blocked') and entry.get('request_sha256') != value.model._digest(task['request']):
        raise ValueError('Source solver request digest differs')
    if entry.get('observations_used', False) is not False:
        raise ValueError('Source design used observations')
    value.old.collect_refs(outcome, {})
    return deepcopy(entry)


def candidate_task(original, receipt, design_version, design_pin, design_dependencies=None):
    core = deepcopy(original['core'])
    core['base_design_version'] = core['design_version']
    core['design_version'] = design_version
    core['distribution_consistency'] = design_pin
    core['code_pins'].update({design_pin['path']: design_pin['sha256']})
    core['code_pins'].update(design_dependencies or {})
    key = retained.key_for(original['request'], original['sidecars'], core)
    task = {'key': key, 'case_id': original['case_id'], 'request': deepcopy(original['request']),
        'sidecars': deepcopy(original['sidecars']), 'core': core, 'candidate': CANDIDATE,
        'construction_blocked': deepcopy(original.get('construction_blocked')),
        'source_task': {k: deepcopy(original[k]) for k in ('key', 'request', 'sidecars', 'core')},
        'source_exact_input_digest': source_runner.exact_input_digest(original),
        'source_prediction_receipt': deepcopy(receipt), 'reuse_value_receipt': None}
    entry = source_entry(task)
    task['fallback_eligible'] = (not bool(task['construction_blocked']) and floor_failure(entry)
        and not importlib.import_module(MODULE).protected_blood_partition(task['request']))
    task['source_status'] = entry['status']
    return task


def prepare(run=DEFAULT, workers=14):
    run = own(run)
    if (run/'protocol.json').exists():
        raise FileExistsError('Protocol already frozen; resume with run')
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 14:
        raise ValueError('workers must be 1..14')
    module = importlib.import_module(MODULE)
    state = value.old.read(SOURCE/'state.json')
    source = value.old.checked(state['protocol']); value.verify(source)
    if state['status'] != 'complete' or state.get('all_predictions_locked_before_scoring') is not True:
        raise ValueError('The completed, prediction-locked source-qualified baseline is required')
    if set(state['outcomes']) != set(source['tasks']):
        raise ValueError('Source predictions are incomplete')
    pins, code = deepcopy(source['source_pins']), deepcopy(source['code_pins'])
    for receipt in (state['protocol'], state['results']):
        base.merge_pins(pins, {receipt['path']: receipt['sha256']})  # Hash only: no reference values.
    design_pin = value.old.ref(Path(module.__file__))
    for path in (Path(__file__), Path(module.__file__), Path(source_runner.__file__), Path(base.__file__)):
        base.merge_pins(code, {str(path.resolve().relative_to(ROOT)): sha256(path)})
    # Optional explicit dependencies supplied by the new module, never inferred
    # from unrelated files or a repository-wide scan.
    base.merge_pins(code, getattr(module, 'CODE_PINS', {}))
    tasks, units, baselines = {}, [], {}
    for unit in source['units']:
        original = source['tasks'][unit['key']]
        task = candidate_task(original, state['outcomes'][unit['key']], module.VERSION, design_pin,
                              getattr(module, 'CODE_PINS', {}))
        key = task['key']
        if key in tasks and source_runner.exact_input_digest(tasks[key]) != source_runner.exact_input_digest(task):
            raise ValueError('Semantic key collision between different complete requests')
        if key not in tasks:
            outcome = value.old.checked(task['source_prediction_receipt'])
            value.old.collect_refs(outcome, pins)
            pins[task['source_prediction_receipt']['path']] = task['source_prediction_receipt']['sha256']
            if not task['construction_blocked']:
                value.model._check_request(task['request'])
                for rows in task['sidecars'].values():
                    value.model._check_sidecars(task['request'], rows, ROOT)
            tasks[key] = task
        units.append({**deepcopy(unit), 'key': key, 'source_key': unit['key']})
        if unit['condition'] == 'all_ml':
            baselines[unit['case_id']] = {'fingerprint': key}
    matrix = {(c['case_id'], condition) for c in source['cases'] for condition in CONDITIONS}
    if len(units) != len(matrix) or {(u['case_id'], u['condition']) for u in units} != matrix:
        raise ValueError('Retain every original clinical case and input condition')
    cores = list(tasks.values())
    if not cores or any(t['core'] != cores[0]['core'] for t in cores):
        raise ValueError('Every case must use the same frozen prior and algorithm core')
    eligible = [t for t in tasks.values() if t['fallback_eligible']]
    protocol = deepcopy(source)
    protocol.update(version=VERSION, candidate=CANDIDATE, created_at=value.old.now(),
        tasks=tasks, units=units, baselines=baselines, core=cores[0]['core'],
        design_version=module.VERSION, source_qualified_protocol=state['protocol'],
        source_qualified_results=state['results'], source_pins=pins, code_pins=code,
        workers=workers, native_threads=2, native_timeout_seconds=180,
        directory_limit_bytes=10*1024**3, directory_budget_scope=str(DEFAULT.relative_to(ROOT)),
        monitor_seconds=1200, publication_snapshot=value.old.paper_pins(), paper_guard={},
        cache_only=False, no_parameter_training=True, no_clinical_curve_fitting=True,
        new_training_calls=0, logical_comparisons=len(units), unique_predictions=len(tasks),
        new_unique_predictions=len(eligible), expected_new_native_calls=len(eligible),
        fallback_eligible_cases=sorted({t['case_id'] for t in eligible}),
        source_predictions_retained=len(tasks)-len(eligible),
        clinical_cases_dropped=0, new_independent_clinical_samples=0,
        fallback_rule='Only exclusively vascular-floor-failed PS searches try one Schmitt native distribution; 33 PS calls maximum; no clinical score used for the switch',
        limitations=[*source['limitations'],
            'Distribution fallback is a development intervention proposed after reviewing construction failures, not blinded validation',
            'Successful source predictions and non-floor failures are preserved; no per-drug clinical-score-based formula selection',
            'Schmitt RBC partition and hematocrit determine a reduced-model blood/plasma consistency correction; it is not a measured clinical ratio or a native transient prediction',
            'Current predictor fitting excludes all forty target identities/scaffolds, but inherited recipe selection historically used PK labels from Ethinylestradiol, Montelukast, Paroxetine and Pitavastatin; this is development validation, not a fully independent model-selection test',
            'Coverage gains and common-case accuracy must be reported separately; newly recovered cases change the all-returned GMFE denominator',
            'The original twelve-case evidence group remains fixed, including rejected source inputs; failures and missing scalar references remain in the report'])
    value.verify(protocol, publication=True)
    run.mkdir(parents=True, exist_ok=True); save(run/'protocol.json', protocol)
    save(run/'state.json', {'version': VERSION, 'status': 'prepared', 'created_at': value.old.now(),
        'protocol': value.old.ref(run/'protocol.json'), 'outcomes': {}, 'all_predictions_locked_before_scoring': False})
    return {'status': 'prepared', 'clinical_cases': len(source['cases']), 'logical_units': len(units),
        'unique_predictions': len(tasks), 'native_fallbacks_expected': len(eligible),
        'fallback_cases': protocol['fallback_eligible_cases'], 'source_predictions_retained': len(tasks)-len(eligible),
        'construction_blocked_unique': sum(bool(t['construction_blocked']) for t in tasks.values()),
        'evidence_group': len(protocol['changed_case_ids']), 'new_training_calls': 0}


def validate_completion(completion, task):
    if (completion.get('key') != task['key']
            or completion.get('candidate') != CANDIDATE
            or completion.get('exact_input_digest') != source_runner.exact_input_digest(task)
            or completion.get('source_prediction_receipt') != task['source_prediction_receipt']):
        raise ValueError('Recovery receipt belongs to different complete inputs or source outcome')
    entry = completion['entry']
    if (entry.get('request') != task['request'] or entry.get('target_observations_used') is not False
            or entry.get('status') not in {'predicted', 'failed'}):
        raise ValueError('Recovery entry lacks a terminal, observation-isolated request')
    value.old.collect_refs(completion, {})
    return deepcopy(entry)


def infrastructure_failure(entry):
    """A stopped project-owned native process is not a scientific failure."""
    if entry.get('reason') not in {'native_execution_failed', 'native_source_failed'}:
        return None
    process = value.old.checked(entry['execution_ref']).get('process', {})
    if (process.get('stop_reason') or process.get('error') or process.get('status') == 'blocked'
            or process.get('returncode') not in {None, 0}):
        return {k: process.get(k) for k in ('status', 'stop_reason', 'returncode', 'error')}
    return None


def invoke_design(task, folder, original):
    """Bind the invocation before a module writes files; recover complete work."""
    module = importlib.import_module(MODULE)
    if module.VERSION != task['core']['design_version']:
        raise ValueError('Distribution implementation version differs from the frozen core')
    binding = {'key': task['key'], 'candidate': CANDIDATE,
        'exact_input_digest': source_runner.exact_input_digest(task),
        'source_prediction_receipt': task['source_prediction_receipt']}
    invocation = folder.parent/(folder.name+'.invocation.json')
    if invocation.exists():
        prior = value.old.read(invocation)
        if {k: prior.get(k) for k in binding} != binding:
            raise ValueError('Native invocation receipt differs from current inputs')
    else:
        if folder.exists():
            raise ValueError('Existing native folder lacks its pre-execution input receipt')
        prior = {**binding, 'started_at': value.old.now(), 'interrupted_attempts': []}
        save(invocation, prior)
    result_path = folder/'result.json'
    entry = None
    if result_path.exists():
        result = value.old.read(result_path)
        if (result.get('version') != module.VERSION
                or result.get('request_sha256') != value.model._digest(task['request'])
                or result.get('request') != task['request']
                or result.get('target_observations_used') is not False
                or result.get('status') not in {'predicted', 'failed'}):
            raise ValueError('Module recovery result differs from its bound invocation')
        value.old.collect_refs(result, {})
        detail = infrastructure_failure(result)
        if detail is None:
            entry = {**result, 'result_ref': value.old.ref(result_path)}
        else:
            prior['interrupted_attempts'].append({'time': value.old.now(),
                'reason': 'Completed process receipt describes an infrastructure failure; retained before explicit resume',
                'execution_failure_detail': detail, 'completed_native_calls': result.get('native_solver_calls')})
    if entry is None:
        if folder.exists():
            value.old.preserve_interrupted(folder)
            prior['interrupted_attempts'].append({'time': value.old.now(),
                'reason': 'No complete module result; original work preserved before infrastructure retry',
                'native_calls_before_interruption': 'unknown_not_counted_as_a_completed_scientific_attempt'})
            save(invocation, prior)
        entry = module.run_design(deepcopy(task['request']), deepcopy(task['sidecars']['vss']),
            deepcopy(task['sidecars']['half']), folder, ROOT, source_entry=original)
    if prior['interrupted_attempts']:
        entry = {**entry, 'interrupted_attempts': deepcopy(prior['interrupted_attempts'])}
    return entry


def solve(task, run, monitor):
    key = task['key']
    with monitor.lock:
        if key in monitor.state['outcomes']:
            saved = value.old.checked(monitor.state['outcomes'][key])
            if saved['key'] != key or saved['entry']['request'] != task['request']:
                raise value.old.Pause('Locked result no longer matches its request')
            return key
    monitor.check(); folder = run/'models'/key[:20]; started = time.monotonic()
    try:
        original = source_entry(task)
        completion_path = folder/'completion.json'
        if completion_path.exists():
            entry = validate_completion(value.old.read(completion_path), task)
        else:
            if task['construction_blocked']:
                entry = deepcopy(original)
                entry.update(native_solver_calls=0, linear_trajectory_calls=0,
                    distribution_audit={'status': 'source_failure_retained', 'reason': 'construction_blocked',
                        'source_prediction_receipt': task['source_prediction_receipt']})
            else:
                entry = invoke_design(task, folder, original)
                if entry.get('request_sha256') != value.model._digest(task['request']):
                    raise ValueError('Design entry must preserve the original request digest')
            entry = deepcopy(entry)
            entry.update(request=deepcopy(task['request']), target_observations_used=False, condition='no_clinical_curve')
            if entry['status'] not in {'predicted', 'failed'}:
                raise ValueError('Every scientific result must be a prediction or explicit failure')
            if entry.get('observations_used', False) is not False:
                raise ValueError('Distribution design used clinical observations')
            detail = infrastructure_failure(entry)
            if detail is not None:
                raise value.old.Pause('Native infrastructure failure (not a scientific prediction): '+str(detail))
            if not task['fallback_eligible']:
                if (entry['status'] != original['status'] or entry.get('curve') != original.get('curve')
                        or entry.get('endpoints') != original.get('endpoints')
                        or entry.get('native_solver_calls') != 0 or entry.get('linear_trajectory_calls') != 0):
                    raise ValueError('Successful predictions and non-floor failures must be exactly retained')
            if entry['status'] == 'predicted':
                if entry['endpoints'].get('observations_used') is not False:
                    raise ValueError('Predicted endpoint used observations')
                entry['endpoints'].update(comparison_observations_used=False, endpoint_estimator_uses_observations=False)
                value._native_fields(entry)
            else:
                entry.setdefault('failure', entry.get('reason', 'Prediction failed'))
            if (entry.get('native_solver_calls') not in {0, 1}
                    or not isinstance(entry.get('linear_trajectory_calls'), int)
                    or not 0 <= entry['linear_trajectory_calls'] <= 33):
                raise ValueError('Design execution exceeded the frozen call budget')
            folder.mkdir(parents=True, exist_ok=True)
            save(completion_path, {'key': key, 'candidate': CANDIDATE,
                'exact_input_digest': source_runner.exact_input_digest(task),
                'source_prediction_receipt': task['source_prediction_receipt'], 'entry': entry})
        item = {'key': key, 'case_id': task['case_id'], 'entry': entry,
            'reused': not task['fallback_eligible'], 'source_prediction_receipt': task['source_prediction_receipt'],
            'completion_ref': value.old.ref(completion_path), 'new_native_calls': entry.get('native_solver_calls', 0),
            'new_linear_trajectory_calls': entry.get('linear_trajectory_calls', 0),
            'elapsed_seconds': time.monotonic()-started, 'locked_at': value.old.now()}
        path = run/'models'/f'{key[:20]}.json'; save(path, item)
    except Exception as error:
        with monitor.lock:
            count = monitor.state.get('consecutive_infrastructure_errors', 0)+1
            monitor.update(consecutive_infrastructure_errors=count, pause_reason=type(error).__name__+': '+str(error))
            if count >= 3:
                monitor.halted.set()
        save(run/'models'/f'{key[:20]}.infrastructure.json', {'key': key, 'candidate': CANDIDATE,
            'reason': type(error).__name__+': '+str(error), 'scientific_failure': False, 'time': value.old.now()})
        raise value.old.Pause('Execution exception retained for review: '+str(error)) from error
    with monitor.lock:
        monitor.state['outcomes'][key] = value.old.ref(path)
        monitor.update(completed_unique=len(monitor.state['outcomes']), consecutive_infrastructure_errors=0)
    return key


def _case_rows(data, condition, cases):
    ids = {c['case_id'] for c in cases}
    rows = [r for r in data['units'] if r['condition'] == condition and r['case_id'] in ids]
    if len(rows) != len(ids) or {r['case_id'] for r in rows} != ids:
        raise ValueError('Comparison must retain every requested case, including failures')
    return {r['case_id']: r for r in rows}


def comparison_statistics(result, previous, protocol):
    groups = {'all': protocol['cases'],
        'old_20': [c for c in protocol['cases'] if c['validation_stratum'] == 'old_20'],
        'new_compounds': [c for c in protocol['cases'] if c['validation_stratum'] == 'new_compounds'],
        'original_evidence_eligible': [c for c in protocol['cases'] if c['case_id'] in protocol['changed_case_ids']]}
    comparisons = {}
    for name, cases in groups.items():
        comparisons[name] = {}
        for condition in CONDITIONS:
            after, before = _case_rows(result, condition, cases), _case_rows(previous, condition, cases)
            a = {k: r['curve_metrics'].get('gmfe') for k, r in after.items()}
            b = {k: r['curve_metrics'].get('gmfe') for k, r in before.items()}
            common = [c for c in cases if value.scores.positive(a[c['case_id']]) and value.scores.positive(b[c['case_id']])]
            newly = [c for c in cases if value.scores.positive(a[c['case_id']]) and not value.scores.positive(b[c['case_id']])]
            lost = [c['case_id'] for c in cases if not value.scores.positive(a[c['case_id']]) and value.scores.positive(b[c['case_id']])]
            endpoints = {}
            for family in value.evaluation.REFERENCE_FAMILIES:
                selected = [c for c in cases if after[c['case_id']]['reference_family'] == family]
                if not selected:
                    continue
                endpoints[family] = {}
                for endpoint in value.UNITS:
                    ea = {c['case_id']: after[c['case_id']]['endpoints'][endpoint] for c in selected}
                    eb = {c['case_id']: before[c['case_id']]['endpoints'][endpoint] for c in selected}
                    if any(ea[k]['reference'] != eb[k]['reference'] for k in ea):
                        raise ValueError('Candidate and source scalar references differ')
                    pairs = [k for k in ea if ea[k]['error'] is not None and eb[k]['error'] is not None]
                    stats = {'attempted': len(selected), 'reference_available': sum(ea[k]['reference'] is not None for k in ea),
                        'candidate_scored': sum(ea[k]['error'] is not None for k in ea),
                        'source_scored': sum(eb[k]['error'] is not None for k in eb), 'complete_pairs': len(pairs)}
                    if endpoint == 'tmax':
                        diffs = {k: ea[k]['error']-eb[k]['error'] for k in pairs}
                        stats.update(error_measure='absolute_error_hours', per_case_error_difference_h=diffs,
                            mean_paired_error_difference_h=math.fsum(diffs.values())/len(diffs) if diffs else None)
                    else:
                        stats['candidate_over_source_error'] = base.paired(
                            {k: v['error'] for k, v in ea.items()}, {k: v['error'] for k, v in eb.items()}, selected)
                    endpoints[family][endpoint] = stats
            comparisons[name][condition] = {'requested': len(cases),
                'candidate_returned': sum(r['attempt_status'] == 'predicted' for r in after.values()),
                'source_returned': sum(r['attempt_status'] == 'predicted' for r in before.values()),
                'candidate_scored': sum(value.scores.positive(x) for x in a.values()),
                'source_scored': sum(value.scores.positive(x) for x in b.values()),
                'common_candidate_curve': base.drug_equal({c['case_id']: a[c['case_id']] for c in common}, common),
                'common_source_curve': base.drug_equal({c['case_id']: b[c['case_id']] for c in common}, common),
                'common_curve_ratio': base.paired(a, b, cases), 'newly_scored_case_ids': [c['case_id'] for c in newly],
                'newly_scored_curve': base.drug_equal({c['case_id']: a[c['case_id']] for c in newly}, newly),
                'lost_scored_case_ids': lost, 'endpoints': endpoints}
    return {'candidate_over_source_qualified': comparisons,
        'original_evidence_group_case_ids': protocol['changed_case_ids'],
        'denominator_reduced_for_benefit_claim': False, 'clinical_safety_established': False,
        'new_independent_clinical_samples': 0,
        'interpretation': 'Common-case accuracy and newly returned cases are separated; coverage recovery alone is not proven clinical accuracy'}


def report(run=DEFAULT):
    run = own(run)
    info = clone(base.report, own=own, VERSION=VERSION)(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = value.old.read(run/'state.json'); protocol = value.old.checked(state['protocol'])
        result = value.old.checked(state['results'])
        previous = value.old.checked(protocol['source_qualified_results'])
        audits = {k: value.old.checked(r)['entry'].get('distribution_audit', {}) for k, r in state['outcomes'].items()}
        result.update(candidate=CANDIDATE, design_version=protocol['design_version'],
            source_qualified_results=protocol['source_qualified_results'], distribution_audits=audits,
            distribution_consistency=comparison_statistics(result, previous, protocol),
            no_parameter_training=True, new_training_calls=0, new_independent_clinical_samples=0)
        value.verify(protocol); save(run/'results.json', result)
        state['results'] = value.old.ref(run/'results.json'); save(run/'state.json', state)
        lines = ['# Distribution-consistency development validation', '',
            'Fixed source-qualified cases, inputs and parameter-model weights; zero training and zero clinical calibration. Only exclusive vascular-floor failures may try Schmitt distribution. Successful results and other failures are retained.', '',
            '| Input condition | Source scored / requested | Candidate scored / requested | Source common-case GMFE | Candidate common-case GMFE | Newly scored | Lost |',
            '|---|---:|---:|---:|---:|---:|---:|']
        def fmt(x): return '—' if x is None else f'{x:.4f}'
        for condition, stats in result['distribution_consistency']['candidate_over_source_qualified']['all'].items():
            lines.append(f"| {condition} | {stats['source_scored']}/{stats['requested']} | {stats['candidate_scored']}/{stats['requested']} | {fmt(stats['common_source_curve']['estimate'])} | {fmt(stats['common_candidate_curve']['estimate'])} | {len(stats['newly_scored_case_ids'])} | {len(stats['lost_scored_case_ids'])} |")
        lines += ['', 'All eleven outputs, strict <2/<3/<5 thresholds, missing references, original-versus-added strata, failures and original evidence-group denominators remain in results.json/results.csv. The paired study-cluster bootstrap uses 2,000 draws and seed 20261007; reference families are kept separate.',
            'The module derives reduced-model blood/plasma consistency from Schmitt RBC partition and hematocrit. This is neither a measured clinical B/P value nor a native transient prediction. The prior and independent evidence are unchanged.',
            'Recovered predictions are reported separately from common-case accuracy: additional difficult cases can change the all-returned GMFE. No increase in compounds, patients or independent clinical experiments is claimed. This development intervention does not establish clinical safety and does not update the manuscript.', '']
        (run/'README.md').write_text('\n'.join(lines))
    return {**info, 'candidate': CANDIDATE, 'new_training_calls': 0}


def execute(run=DEFAULT):
    return clone(retained.execute, own=own, solve=solve, Monitor=retained.Monitor, report=report)(own(run))


def status(run=DEFAULT):
    return clone(base.status, own=own)(own(run))


def cleanup(run=DEFAULT):
    return clone(base.cleanup, own=own)(own(run))


def background(run=DEFAULT):
    run = own(run)
    with (run/'run.log').open('ab', buffering=0) as log:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--run-dir', str(run)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    save(run/'pid.json', {'pid': process.pid, 'project': str(ROOT), 'created_at': value.old.now()})
    return {'status': 'started', 'pid': process.pid, 'log': str(run/'run.log')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'status', 'report', 'cleanup'))
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.run_dir, args.workers)
    elif args.command == 'run': result = background(args.run_dir) if args.background else execute(args.run_dir)
    else: result = globals()[args.command](args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__': main()
