#!/usr/bin/env python3
"""Evaluate joint partition design with frozen priors and zero native executions.

All eligible source-qualified requests receive the same new design algorithm.
A predeclared conservative policy uses it only for source capacity failures.
Both policies are scored only after every new prediction/failure is locked.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_distribution_consistency_validation as infra
from pbpk_kb.blueprints.source_workflow import save, sha256

base, value, retained, clone = infra.base, infra.value, infra.retained, infra.clone
source_runner = infra.source_runner
VERSION = 'joint-partition-validation-1'
CANDIDATE = 'joint_partition'
MODULE = 'pbpk_kb.blueprint_policy.joint_partition_runtime'
SOURCE = source_runner.DEFAULT
DEFAULT = ROOT/'experiments/blueprint_design/joint_partition_20261007'
CONDITIONS = base.CONDITIONS
PRIMARY_CONDITIONS = ('available', 'missing_clearance')


def own(path):
    path = value.inside(path, ROOT)
    if path != DEFAULT: raise ValueError('Use only joint_partition_20261007')
    return path


def candidate_task(original, receipt, module):
    core = deepcopy(original['core'])
    pin = value.old.ref(Path(module.__file__))
    core.update(base_design_version=core['design_version'], design_version=module.VERSION, joint_partition=pin)
    core['code_pins'].update({pin['path']: pin['sha256'], **module.CODE_PINS})
    task = {'key': retained.key_for(original['request'], original['sidecars'], core),
        'case_id': original['case_id'], 'request': deepcopy(original['request']),
        'sidecars': deepcopy(original['sidecars']), 'core': core, 'candidate': CANDIDATE,
        'construction_blocked': deepcopy(original.get('construction_blocked')),
        'source_task': {k: deepcopy(original[k]) for k in ('key', 'request', 'sidecars', 'core')},
        'source_exact_input_digest': source_runner.exact_input_digest(original),
        'source_prediction_receipt': deepcopy(receipt), 'reuse_value_receipt': None}
    entry = infra.source_entry(task)
    task['joint_design_required'] = not bool(task['construction_blocked'])
    task['conservative_use_joint'] = task['joint_design_required'] and infra.floor_failure(entry)
    task['source_status'] = entry['status']
    if task['joint_design_required']:
        # Reuse only the exact source PK-Sim readback, never a formula from the
        # preceding Schmitt candidate or another drug/request.
        native = value.old.checked(entry['native_source_ref'])
        if (native.get('request_sha256') != value.model._digest(task['request'])
                or native.get('partition_method') != 'PKSimStandard'
                or native.get('status') != 'predicted'
                or native.get('execution_ref') != entry.get('execution_ref')
                or native.get('adapter_sha256') != sha256(Path(value.model.__file__))):
            raise ValueError('Joint design requires the exact frozen Standard native readback')
        value.old.checked(entry['physiology_ref'])
    return task


def prepare(run=DEFAULT, workers=14):
    run = own(run)
    if (run/'protocol.json').exists(): raise FileExistsError('Protocol frozen; resume with run')
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 14:
        raise ValueError('workers must be 1..14')
    module = importlib.import_module(MODULE)
    state = value.old.read(SOURCE/'state.json'); source = value.old.checked(state['protocol']); value.verify(source)
    if (state['status'] != 'complete' or state.get('all_predictions_locked_before_scoring') is not True
            or set(state['outcomes']) != set(source['tasks'])):
        raise ValueError('Source-qualified baseline must be complete and prediction-locked')
    pins, code = deepcopy(source['source_pins']), deepcopy(source['code_pins'])
    for receipt in (state['protocol'], state['results']):
        base.merge_pins(pins, {receipt['path']: receipt['sha256']})  # Do not parse old clinical scores before locking.
    for path in (Path(__file__), Path(infra.__file__), Path(module.__file__)):
        base.merge_pins(code, {str(path.resolve().relative_to(ROOT)): sha256(path)})
    base.merge_pins(code, module.CODE_PINS)
    tasks, units, baselines, old_to_new = {}, [], {}, {}
    for oldkey, original in source['tasks'].items():
        task = candidate_task(original, state['outcomes'][oldkey], module); key = task['key']
        if key in tasks: raise ValueError('Different original tasks unexpectedly share a candidate key')
        outcome = value.old.checked(task['source_prediction_receipt'])
        value.old.collect_refs(outcome, pins)
        pins[task['source_prediction_receipt']['path']] = task['source_prediction_receipt']['sha256']
        if task['joint_design_required']:
            value.model._check_request(task['request'])
            for records in task['sidecars'].values(): value.model._check_sidecars(task['request'], records, ROOT)
        tasks[key] = task; old_to_new[oldkey] = key
    for unit in source['units']:
        key = old_to_new[unit['key']]
        units.append({**deepcopy(unit), 'key': key, 'source_key': unit['key'],
            'conservative_policy': 'joint' if tasks[key]['conservative_use_joint'] else 'source'})
        if unit['condition'] == 'all_ml': baselines[unit['case_id']] = {'fingerprint': key}
    matrix = {(c['case_id'], condition) for c in source['cases'] for condition in CONDITIONS}
    if len(units) != len(matrix) or {(u['case_id'], u['condition']) for u in units} != matrix:
        raise ValueError('Every source case and all five conditions must remain')
    core = next(iter(tasks.values()))['core']
    if any(t['core'] != core for t in tasks.values()): raise ValueError('Unequal scientific core or prior weights')
    nnew = sum(t['joint_design_required'] for t in tasks.values())
    protocol = deepcopy(source)
    protocol.update(version=VERSION, candidate=CANDIDATE, created_at=value.old.now(), tasks=tasks, units=units,
        baselines=baselines, core=core, design_version=module.VERSION,
        source_qualified_protocol=state['protocol'], source_qualified_results=state['results'],
        source_pins=pins, code_pins=code, workers=workers, native_threads=2, native_timeout_seconds=180,
        directory_limit_bytes=10*1024**3, directory_budget_scope=str(DEFAULT.relative_to(ROOT)), monitor_seconds=1200,
        publication_snapshot=value.old.paper_pins(), paper_guard={}, cache_only=False,
        no_parameter_training=True, no_clinical_curve_fitting=True, new_training_calls=0,
        logical_comparisons=len(units), unique_predictions=len(tasks), new_unique_predictions=nnew,
        expected_new_native_calls=0, maximum_new_linear_trajectory_calls=nnew*33,
        cpu_threads_per_worker=2, uses_gpu_training=False, clinical_cases_dropped=0,
        new_independent_clinical_samples=0, source_construction_failures_retained=len(tasks)-nnew,
        conservative_policy={'name': 'conservative_fallback', 'rule': 'Use joint design only for the frozen source exclusive vascular-capacity failure; otherwise source',
            'selection_frozen_before_scoring': True, 'source_curve_errors_used_for_branch': False,
            'additional_native_calls': 0, 'additional_independent_experiments': 0},
        adoption_gate={'coverage_conditions': list(CONDITIONS), 'primary_conditions': list(PRIMARY_CONDITIONS),
            'scored_coverage_must_not_decrease': True, 'common_case_gmfe_must_not_increase': True,
            'meaningful_threshold': .05, 'meaningful_benefit_count_must_exceed_harm_in_each_primary_condition': True,
            'confidence_intervals_descriptive_not_significance_gate': True,
            'failed_gate_preserves_current_research_version': True},
        limitations=[*source['limitations'],
            'One uniformly applied joint-capacity algorithm; frozen parameter weights and zero clinical calibration',
            'All exact Standard native readbacks and physiology are reused; no new native execution or GPU training',
            'The conservative policy is chosen only from frozen source failure diagnostics, not clinical errors',
            'Current fitting excludes forty target structures/scaffolds; historical recipe selection used PK labels from Ethinylestradiol, Montelukast, Paroxetine and Pitavastatin, so this is development validation',
            'Common-case accuracy, newly scored cases and coverage changes are reported separately; missing references and failures remain',
            'Original twelve-drug evidence group is retained; no manuscript or production-default update by this runner'])
    value.verify(protocol, publication=True)
    run.mkdir(parents=True, exist_ok=True); save(run/'protocol.json', protocol)
    save(run/'state.json', {'version': VERSION, 'status': 'prepared', 'created_at': value.old.now(),
        'protocol': value.old.ref(run/'protocol.json'), 'outcomes': {}, 'all_predictions_locked_before_scoring': False})
    return {'status': 'prepared', 'clinical_cases': len(source['cases']), 'logical_units': len(units),
        'unique_predictions': len(tasks), 'new_designs': nnew, 'new_native_calls': 0, 'new_training_calls': 0,
        'conservative_joint_cases': sorted({t['case_id'] for t in tasks.values() if t['conservative_use_joint']})}


def invoke_design(task, folder, original):
    return clone(infra.invoke_design, MODULE=MODULE, CANDIDATE=CANDIDATE)(task, folder, original)


def solve(task, run, monitor):
    key = task['key']
    with monitor.lock:
        if key in monitor.state['outcomes']:
            saved = value.old.checked(monitor.state['outcomes'][key])
            if saved['key'] != key or saved['entry']['request'] != task['request']:
                raise value.old.Pause('Locked result does not match its complete request')
            return key
    monitor.check(); folder = run/'models'/key[:20]; started = time.monotonic()
    try:
        original = infra.source_entry(task); completion = folder/'completion.json'
        if completion.exists():
            entry = clone(infra.validate_completion, CANDIDATE=CANDIDATE)(value.old.read(completion), task)
        else:
            if not task['joint_design_required']:
                entry = deepcopy(original)
                entry.update(native_solver_calls=0, linear_trajectory_calls=0,
                    joint_partition_audit={'reason': 'construction_blocked', 'source_prediction_receipt': task['source_prediction_receipt']})
            else:
                entry = invoke_design(task, folder, original)
                if entry.get('request_sha256') != value.model._digest(task['request']):
                    raise ValueError('Joint runtime changed the scientific input digest')
            entry = deepcopy(entry)
            if entry.get('observations_used', False) is not False or entry.get('target_observations_used') is not False:
                raise ValueError('Joint design used clinical observations')
            if entry['status'] not in {'predicted', 'failed'}: raise ValueError('Nonterminal joint outcome')
            failure = infra.infrastructure_failure(entry)
            if failure is not None: raise value.old.Pause('Infrastructure failure, not scientific failure: '+str(failure))
            entry.update(request=deepcopy(task['request']), target_observations_used=False, condition='no_clinical_curve')
            if entry['status'] == 'predicted':
                if entry['endpoints'].get('observations_used') is not False: raise ValueError('Endpoint estimator used observations')
                entry['endpoints'].update(comparison_observations_used=False, endpoint_estimator_uses_observations=False)
                value._native_fields(entry)
            else: entry.setdefault('failure', entry.get('reason', 'Prediction failed'))
            if (entry.get('native_solver_calls') != 0 or not isinstance(entry.get('linear_trajectory_calls'), int)
                    or not 0 <= entry['linear_trajectory_calls'] <= 33):
                raise ValueError('Joint design requires zero native calls and at most 33 linear trajectories')
            folder.mkdir(parents=True, exist_ok=True)
            save(completion, {'key': key, 'candidate': CANDIDATE,
                'exact_input_digest': source_runner.exact_input_digest(task),
                'source_prediction_receipt': task['source_prediction_receipt'], 'entry': entry})
        item = {'key': key, 'case_id': task['case_id'], 'entry': entry, 'reused': not task['joint_design_required'],
            'source_prediction_receipt': task['source_prediction_receipt'], 'completion_ref': value.old.ref(completion),
            'new_native_calls': entry['native_solver_calls'], 'new_linear_trajectory_calls': entry['linear_trajectory_calls'],
            'elapsed_seconds': time.monotonic()-started, 'locked_at': value.old.now()}
        path = run/'models'/f'{key[:20]}.json'; save(path, item)
    except Exception as error:
        with monitor.lock:
            count = monitor.state.get('consecutive_infrastructure_errors', 0)+1
            monitor.update(consecutive_infrastructure_errors=count, pause_reason=type(error).__name__+': '+str(error))
            if count >= 3: monitor.halted.set()
        save(run/'models'/f'{key[:20]}.infrastructure.json', {'key': key, 'candidate': CANDIDATE,
            'reason': type(error).__name__+': '+str(error), 'scientific_failure': False, 'time': value.old.now()})
        raise value.old.Pause('Execution exception retained for review: '+str(error)) from error
    with monitor.lock:
        monitor.state['outcomes'][key] = value.old.ref(path)
        monitor.update(completed_unique=len(monitor.state['outcomes']), consecutive_infrastructure_errors=0)
    return key


def benefit_statistics(ratios, cases, resamples=base.RESAMPLES, seed=base.SEED):
    """Drug-equal 5% changes; bootstrap whole study clusters with multiplicity."""
    aggregate = base.drug_equal(ratios, cases, resamples=resamples, seed=seed)
    changes = {cid: ('benefit' if ratio <= .95+1e-12 else 'harm' if ratio >= 1.05-1e-12 else 'within_5pct')
        for cid, ratio in aggregate['per_drug'].items()}
    nrequested = len({c['compound_id'] for c in cases}); ncommon = len(changes)
    counts = {label: sum(x == label for x in changes.values()) for label in ('benefit', 'harm', 'within_5pct')}
    results = {'per_drug_ratio': aggregate['per_drug'], 'per_drug_change': changes, **counts,
        'requested_drugs': nrequested, 'common_scored_drugs': ncommon, 'unpaired_drugs': nrequested-ncommon,
        'threshold': .05, 'rate_denominator': 'requested and common-scored drugs reported separately',
        'bootstrap': {'unit': 'study_cluster', 'resamples': resamples, 'seed': seed, 'multiplicity_retained': True}}
    for population, selected in (('requested', cases), ('common_scored', [c for c in cases if c['compound_id'] in changes])):
        counts_per_drug = defaultdict(int)
        for case in selected: counts_per_drug[case['compound_id']] += 1
        clusters = defaultdict(list)
        for case in selected: clusters[case['cluster_id']].append(case)
        names = sorted(clusters); denominator = len(counts_per_drug)
        rates = {}
        for label in ('benefit', 'harm'):
            ci = None
            if len(names) > 1:
                numerator = np.array([math.fsum((changes.get(c['compound_id']) == label)/counts_per_drug[c['compound_id']]
                    for c in clusters[name]) for name in names])
                weights = np.array([math.fsum(1/counts_per_drug[c['compound_id']] for c in clusters[name]) for name in names])
                draws = np.random.default_rng(seed).integers(0, len(names), size=(resamples, len(names)))
                distribution = numerator[draws].sum(axis=1)/weights[draws].sum(axis=1)
                ci = [float(x) for x in np.quantile(distribution, [.025, .975])]
            rates[label] = {'count': counts[label], 'denominator': denominator,
                'fraction': counts[label]/denominator if denominator else None, 'ci95': ci}
        results[population+'_rates'] = rates
    return results


def comparison_statistics(result, previous, protocol):
    compared = infra.comparison_statistics(result, previous, protocol)
    for name, rows in compared['candidate_over_source_qualified'].items():
        if name == 'all': cases = protocol['cases']
        elif name == 'original_evidence_eligible': cases = [c for c in protocol['cases'] if c['case_id'] in protocol['changed_case_ids']]
        else: cases = [c for c in protocol['cases'] if c['validation_stratum'] == name]
        for stats in rows.values():
            stats['meaningful_changes'] = benefit_statistics(stats['common_curve_ratio']['per_case_ratio'], cases)
    return compared


def adoption_gate(comparison):
    all_stats = comparison['candidate_over_source_qualified']['all']; checks = {}
    for condition in CONDITIONS:
        row = all_stats[condition]
        checks[condition+'_coverage'] = row['candidate_scored'] >= row['source_scored']
    for condition in PRIMARY_CONDITIONS:
        row = all_stats[condition]; ratio = row['common_curve_ratio']['estimate']
        checks[condition+'_common_gmfe'] = ratio is not None and ratio <= 1+1e-12
        changes = row['meaningful_changes']
        checks[condition+'_benefit_exceeds_harm'] = changes['benefit'] > changes['harm']
    return {'passed': all(checks.values()), 'checks': checks, 'automatic_version_update': False,
        'interpretation': 'Development adoption gate; passing is not clinical qualification, superiority proof or a manuscript update'}


def conservative_rows(protocol, state):
    """Score the precommitted policy using locked outputs, without new designs."""
    if (state.get('all_predictions_locked_before_scoring') is not True
            or set(state['outcomes']) != set(protocol['tasks'])):
        raise ValueError('Lock all joint predictions before conservative policy scoring')
    choices = {}
    for key, task in protocol['tasks'].items():
        if task['conservative_use_joint']:
            receipt = state['outcomes'][key]; outcome = value.old.checked(receipt)
            if outcome['key'] != key or not outcome.get('locked_at'): raise ValueError('Joint outcome is not locked')
            entry = outcome['entry']; fingerprint = key
        else:
            receipt = task['source_prediction_receipt']; entry = infra.source_entry(task)
            fingerprint = task['source_task']['key']
        if entry['request'] != task['request'] or entry.get('target_observations_used') is not False:
            raise ValueError('Conservative choice changed scientific inputs')
        choices[key] = (receipt, entry, fingerprint)
    store = base.ClinicalStore(ROOT, protocol['cases']); rows = []
    for case in protocol['cases']:
        clinical, curve, truth = value.evaluation.clinical_truth(case, store)
        units = [u for u in protocol['units'] if u['case_id'] == case['case_id']]
        nominal = next(u['key'] for u in units if u['condition'] == 'available')
        all_ml = next(u['key'] for u in units if u['condition'] == 'all_ml')
        nominal_entry, nominal_fp = choices[nominal][1:]
        all_ml_fp = choices[all_ml][2]; scored = {}
        for unit in units:
            key = unit['key']; receipt, entry, fingerprint = choices[key]
            if fingerprint not in scored:
                scored[fingerprint] = value.score_one(case, receipt, entry, store, clinical, curve, truth)
            row = deepcopy(scored[fingerprint])
            row.update(condition=unit['condition'], method=unit['condition'], policy='conservative_fallback',
                policy_source=unit['conservative_policy'], validation_stratum=case['validation_stratum'], fingerprint=fingerprint,
                prediction_reused=True, prediction_receipt=receipt, source_hash_verified=True,
                historical_development_used=case.get('historical_development_used', case['validation_stratum']=='old_20'),
                qualification_flags=deepcopy(case.get('qualification_flags', [])),
                changed_from_nominal=fingerprint != nominal_fp, input_changed_from_all_ml=fingerprint != all_ml_fp,
                prediction_drift=value.drift(case, nominal_entry, entry))
            if entry['status'] == 'failed': row['failure_root_causes'] = entry.get('design_audit', {}).get('reasons', [])
            rows.append(row)
    store.unchanged(); return rows


def report(run=DEFAULT):
    run = own(run)
    info = clone(base.report, own=own, VERSION=VERSION)(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = value.old.read(run/'state.json'); protocol = value.old.checked(state['protocol']); value.verify(protocol)
        result = value.old.checked(state['results']); source = value.old.checked(protocol['source_qualified_results'])
        comparison = comparison_statistics(result, source, protocol); decision = adoption_gate(comparison)
        fallback = {'units': conservative_rows(protocol, state)}
        summaries, pairs, missingness = base.summarize(fallback['units'], protocol)
        fallback.update(summaries=summaries, available_over_all_ml=pairs, missingness=missingness,
            comparison=comparison_statistics(fallback, source, protocol), additional_native_calls=0,
            additional_linear_trajectory_calls=0, additional_independent_experiments=0,
            branch_rule=protocol['conservative_policy'])
        result.update(candidate=CANDIDATE, design_version=protocol['design_version'],
            source_qualified_results=protocol['source_qualified_results'], joint_partition=comparison,
            conservative_policy=fallback, adoption_decision=decision, new_training_calls=0,
            no_parameter_training=True, new_independent_clinical_samples=0,
            joint_partition_audits={k: value.old.checked(ref)['entry'].get('joint_partition_audit', {})
                for k, ref in state['outcomes'].items()})
        csv_rows = []
        for policy, dataset in (('source', source), ('joint', result), ('conservative_fallback', fallback)):
            for row in dataset['units']:
                item = deepcopy(row); item['method'] = policy+'/'+item['condition']; csv_rows.append(item)
        labels = {row['method']: row['method'].replace('/', ' / ') for row in csv_rows}
        clone(value.evaluation.write_csv, LABELS=labels)(run/'results.csv', csv_rows)
        value.verify(protocol); save(run/'results.json', result)
        state.update(results=value.old.ref(run/'results.json'), result_table=value.old.ref(run/'results.csv'), adoption_decision=decision)
        save(run/'state.json', state)
        lines = ['# Joint-partition development validation', '',
            'All source-qualified cases and five conditions use frozen parameter weights. Every nonblocked request receives the same joint-capacity design; all native readbacks are reused. This is CPU linear-trajectory evaluation, with zero training, zero native runs and zero clinical calibration.', '',
            '| Policy / condition | Scored / requested | Common source GMFE | Common candidate GMFE | At least 5% better / worse | Newly scored / lost |',
            '|---|---:|---:|---:|---:|---:|']
        def fmt(x): return '—' if x is None else f'{x:.4f}'
        for policy, compared in (('joint', comparison), ('conservative', fallback['comparison'])):
            for condition, row in compared['candidate_over_source_qualified']['all'].items():
                change = row['meaningful_changes']
                lines.append(f"| {policy} / {condition} | {row['candidate_scored']}/{row['requested']} | {fmt(row['common_source_curve']['estimate'])} | {fmt(row['common_candidate_curve']['estimate'])} | {change['benefit']} / {change['harm']} | {len(row['newly_scored_case_ids'])} / {len(row['lost_scored_case_ids'])} |")
        lines += ['', 'Main-method adoption gate passed: '+str(decision['passed'])+'. The current research version is not modified by this runner.',
            'The gate requires scored coverage not to decrease in any condition; available and missing-clearance common-case GMFE must not increase, and each must have more drugs with at least 5% benefit than harm. Confidence intervals describe uncertainty and are not an added significance gate.',
            'The conservative policy was assigned from source capacity-failure diagnostics before clinical scoring. It is assembled from locked predictions, adds no simulation or independent sample, and does not select by prediction error.',
            'All eleven outputs and missing references remain in results.csv for source, joint and conservative policies. Results include strict FE <2/<3/<5 coverage, old/additional strata, fixed original evidence group, endpoint tradeoffs, missing-input stability and paired study-cluster 2,000-draw intervals (seed 20261007). Newly returned cases and common-case accuracy are separate.',
            'Historical recipe selection used labels from four additional drugs despite current forty-target training exclusion. This remains development validation, not a fully independent model-selection test. No clinical safety, GPU training, new patient or paper update is claimed.', '']
        (run/'README.md').write_text('\n'.join(lines))
    return {**info, 'candidate': CANDIDATE, 'adoption_decision': decision, 'new_training_calls': 0}


def execute(run=DEFAULT):
    # Also enforce the limit for foreground invocations after NumPy is loaded.
    with threadpool_limits(limits=2):
        return clone(retained.execute, own=own, solve=solve, Monitor=retained.Monitor, report=report)(own(run))


def status(run=DEFAULT): return clone(base.status, own=own)(own(run))
def cleanup(run=DEFAULT): return clone(base.cleanup, own=own)(own(run))


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
    parser.add_argument('--run-dir', type=Path, default=DEFAULT); parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--background', action='store_true'); args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.run_dir, args.workers)
    elif args.command == 'run': result = background(args.run_dir) if args.background else execute(args.run_dir)
    else: result = globals()[args.command](args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__': main()
