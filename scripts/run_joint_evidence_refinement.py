#!/usr/bin/env python3
"""One frozen joint CL/fu conditioning candidate; no clinical fitting or training.

Reuse the selected complementary-CL blueprint, not either preceding candidate.
Protect independent inputs, rebuild changed native requests, and lock all 100
prediction/failure units before the existing clinical scorer opens observations.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_blueprint_evidence_refinement as separate
from pbpk_kb.blueprints.source_workflow import save, sha256

value = separate.value
retained = separate.retained
clone = separate.clone
VERSION = 'joint-evidence-refinement-1'
CANDIDATE = 'joint'
BATCH = separate.BATCH
DEFAULT = BATCH/'joint'
SOURCE = separate.SOURCE
CONDITIONS = separate.CONDITIONS
POLICY_MODULE = 'pbpk_kb.blueprint_learning.joint_evidence_prior'


def own(path):
    path = value.inside(path, ROOT)
    if path != DEFAULT:
        raise ValueError('Use only this batch joint/ directory')
    return path


def _checked_prior(path):
    prior, receipt = separate._checked_prior(path)
    module = importlib.import_module(POLICY_MODULE)
    if prior.get('version') != module.VERSION:
        raise ValueError('The joint prior version does not match its implementation')
    if prior.get('audit', {}).get('clinical_results_used_for_selection') is not False:
        raise ValueError('Joint coefficients must not use clinical target scores')
    if not prior.get('current_target_priors'):
        raise ValueError('Current complementary CL centers must be explicit')
    return prior, receipt


def _protect_rows(before, after, allowed):
    if not isinstance(after, list) or len(before) != len(after):
        raise ValueError('Do not add or drop construction evidence')
    mutable = {'value', 'unit', 'source', 'prediction_method', 'notes'}
    for old, new in zip(before, after):
        if old.get('basis') != 'predicted' or old.get('parameter') not in allowed:
            if new != old:
                raise ValueError('Independent evidence and other parameters must remain unchanged')
        else:
            if {k:v for k,v in old.items() if k not in mutable} != {k:v for k,v in new.items() if k not in mutable}:
                raise ValueError('Conditioning changed identity, basis or biological context')
            if not value.scores.positive(new.get('value')):
                raise ValueError('Conditioned inputs must be finite and positive')
            if old['parameter'] == 'fraction_unbound' and new['value'] > 1:
                raise ValueError('Fraction unbound must not exceed one')


def transformed_task(original, prior, receipt):
    """Enforce the numeric-input boundary independently of the learner module."""
    module = importlib.import_module(POLICY_MODULE)
    request, sidecars, audit = module.condition_inputs(
        deepcopy(original['request']), deepcopy(original['sidecars']), prior, receipt)
    before = original['request']
    if {k:v for k,v in request.items() if k!='evidence'} != {k:v for k,v in before.items() if k!='evidence'}:
        raise ValueError('Human, dosing, target, source boundary and ledger may not be changed by the prior')
    _protect_rows(before['evidence'], request.get('evidence'), {'total_clearance', 'fraction_unbound'})
    if set(sidecars) != {'vss', 'half'}:
        raise ValueError('Only the two declared sidecars are allowed')
    for name, parameter in (('vss', 'vss'), ('half', 'terminal_half_life')):
        _protect_rows(original['sidecars'][name], sidecars[name], {parameter})
    independent = any(r.get('parameter') in {'total_clearance', 'fraction_unbound'}
        and r.get('basis') in {'measured', 'derived'} for r in before['evidence'])
    if not independent and (request != before or sidecars != original['sidecars']):
        raise ValueError('Without independent CL/fu, every original input must be preserved')
    if request != before and any(r.get('parameter') in {'renal_clearance','hepatic_clearance'} for r in before['evidence']):
        old_cl = [r for r in before['evidence'] if r['parameter']=='total_clearance']
        new_cl = [r for r in request['evidence'] if r['parameter']=='total_clearance']
        if old_cl != new_cl:
            raise ValueError('Do not replace total CL while retaining an independent organ split')
    if audit.get('request_modified') != (request != before) or audit.get('sidecars_modified') != (sidecars != original['sidecars']):
        raise ValueError('Conditioning audit disagrees with actual input changes')
    return {**deepcopy(original), 'request': request, 'sidecars': sidecars}, audit


def reusable_native(native, request):
    return (native.get('request_sha256') == value.model._digest(request)
        and native.get('partition_method') == 'PKSimStandard'
        and native.get('adapter_sha256') == sha256(Path(value.model.__file__)))


def fresh_ledger(task, run, prior_receipt, pins):
    path = own(run)/'inputs'/(task['key'][:20]+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, {'version': VERSION, 'conditioning_prior': prior_receipt,
                'evidence': task['request']['evidence']})
    receipt = {**value.old.ref(path), 'location': 'Exact joint-conditioned construction inputs; no target observations'}
    pins[receipt['path']] = receipt['sha256']
    task['request']['evidence_ledger'] = receipt


def prepare(run, prior_path, workers=14):
    run = own(run)
    if (run/'protocol.json').exists():
        raise FileExistsError('Protocol is frozen; resume with run')
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1<=workers<=14:
        raise ValueError('workers must be 1..14')
    state = value.old.read(SOURCE/'state.json')
    source = value.old.checked(state['protocol']); value.verify(source)
    if state['status']!='complete' or not state['all_predictions_locked_before_scoring']:
        raise ValueError('A completed selected baseline is required')
    prior, receipt = _checked_prior(prior_path)
    expected = {c['compound_id'] for c in source['cases']}
    if set(prior['current_target_priors']) != expected:
        raise ValueError('Joint prior must cover exactly the frozen twenty compounds')
    pins = deepcopy(source['source_pins']); code = deepcopy(source['code_pins'])
    for ref in (state['protocol'], state['results'], receipt, source['cl_prior']):
        if sha256(value.inside(ref['path'], ROOT, exists=True)) != ref['sha256']:
            raise ValueError('Selected baseline or prior source changed')
        # In particular, baseline clinical results are pinned as bytes here;
        # their reference values are not parsed until every prediction is locked.
        pins[ref['path']] = ref['sha256']
    pins.update(prior['source_refs']); code.update(prior['code_pins'])
    for path in (Path(__file__), Path(separate.__file__), Path(importlib.import_module(POLICY_MODULE).__file__)):
        code[str(path.resolve().relative_to(ROOT))] = sha256(path)
    tasks, units, baselines, adjustments = {}, [], {}, {}
    for unit in source['units']:
        oldkey = unit['key']; original = source['tasks'][oldkey]
        changed, audit = transformed_task(original, prior, receipt)
        request, sidecars, core = changed['request'], changed['sidecars'], changed['core']
        if unit['condition'] in {'all_ml', 'missing_both'} and (request!=original['request'] or sidecars!=original['sidecars']):
            raise ValueError('All-predicted and both-missing conditions must be exact identities')
        key = retained.key_for(request, sidecars, core)
        if key not in tasks:
            selected_ref = state['outcomes'][oldkey]
            selected = value.old.checked(selected_ref)
            value.old.collect_refs(selected, pins); pins[selected_ref['path']] = selected_ref['sha256']
            native_ref = selected['entry'].get('native_source_ref')
            if native_ref is None:
                raise ValueError('Selected baseline lacks a native execution receipt')
            native = value.old.checked(native_ref)
            value.old.collect_refs(native, pins); pins[native_ref['path']] = native_ref['sha256']
            if not reusable_native(native, original['request']):
                raise ValueError('Selected native receipt does not match its original request')
            request_changed = request != original['request']
            reuse = selected_ref if key==oldkey and not request_changed else None
            task = {'key':key, 'case_id':unit['case_id'], 'request':request,
                'sidecars':sidecars, 'core':core, 'reuse_value_receipt':reuse,
                'selected_prediction_receipt':selected_ref, 'candidate':CANDIDATE,
                'request_modified':request_changed, 'native_source_ref':None}
            if request_changed:
                fresh_ledger(task, run, receipt, pins)
            if reusable_native(native, task['request']):
                task['native_source_ref'] = native_ref
            value.model._check_request(task['request'])
            for rows in sidecars.values():
                value.model._check_sidecars(task['request'], rows, ROOT)
            if request_changed:
                assembled = value.model.assemble_evidence_blueprint(task['request'], ROOT)
                if assembled['status']!='assembled':
                    raise ValueError('Native construction preflight failed: '+str(assembled.get('gaps')))
            if retained.key_for(task['request'], sidecars, core)!=key:
                raise ValueError('Evidence ledger unexpectedly changed scientific fingerprint')
            tasks[key] = task; adjustments[key] = audit
        units.append({'case_id':unit['case_id'], 'condition':unit['condition'], 'key':key,
                      'selected_key':oldkey, 'method_changed':key!=oldkey})
        if unit['condition']=='all_ml':
            baselines[unit['case_id']] = {'fingerprint':key}
    ids = {c['case_id'] for c in source['cases']}
    if len(units)!=100 or {(u['case_id'],u['condition']) for u in units}!={(cid,con) for cid in ids for con in CONDITIONS}:
        raise ValueError('Retain all twenty drugs and five conditions')
    if any(t['core']!=source['core'] for t in tasks.values()):
        raise ValueError('The original selected PBPK solver core must stay unchanged')
    new = sum(t['reuse_value_receipt'] is None for t in tasks.values())
    native = sum(t['reuse_value_receipt'] is None and t['native_source_ref'] is None for t in tasks.values())
    protocol = {'version':VERSION, 'candidate':CANDIDATE, 'created_at':value.old.now(),
        'cases':deepcopy(source['cases']), 'tasks':tasks, 'units':units, 'baselines':baselines,
        'source_protocol':state['protocol'], 'baseline_results':state['results'],
        'cl_prior':source['cl_prior'], 'conditioning_prior':receipt, 'core':deepcopy(source['core']),
        'changed_case_ids':source['changed_case_ids'], 'adjustments':adjustments,
        'source_pins':pins, 'code_pins':code, 'paper_guard':{}, 'publication_snapshot':value.old.paper_pins(),
        'workers':workers, 'native_threads':2, 'native_timeout_seconds':180,
        'directory_limit_bytes':10*1024**3, 'directory_budget_scope':str(BATCH.relative_to(ROOT)),
        'monitor_seconds':1200, 'logical_comparisons':100, 'new_unique_predictions':new,
        'expected_new_native_calls':native, 'selected_cl_frozen_in_every_request':False,
        'independent_inputs_preserved':True, 'all_predicted_inputs_exactly_reused':True,
        'candidates_combined':False,
        'publication_decision':{'absolute_gmfe_conditions':['available','missing_clearance'],
            'absolute_gmfe_and_scored_coverage_must_not_worsen':True, 'actual_missing_cl_population':6,
            'actual_missing_cl_absolute_gmfe_must_not_worsen':True, 'evidence_benefit_population':8,
            'meaningful_benefit_fraction':.05, 'meaningful_evidence_benefit_count_must_exceed_selected':True,
            'expected_selected_meaningful_benefit_count':3,
            'drift_and_missing_over_available_ratio':'diagnostic only; improving available errors can increase this ratio'},
        'limitations':['Previously viewed twenty-drug development cohort',
            'Only predicted CL/fu/Vss/half may be conditioned on independent CL/fu; no per-drug model choice',
            'Measured or derived inputs, organ clearance, physiology, dosing and selected PBPK core are unchanged',
            'Training-only conditional shifts are not patient-specific clinical estimates or calibrated confidence intervals',
            'All eleven outputs are reported; missing references remain unscored',
            'Relative missing/available error and prediction drift are diagnostic, not clinical safety criteria',
            'No publication or production-default update occurs in this runner']}
    value.verify(protocol, publication=True)
    run.mkdir(parents=True, exist_ok=True); save(run/'protocol.json', protocol)
    save(run/'state.json', {'version':VERSION, 'candidate':CANDIDATE, 'protocol':value.old.ref(run/'protocol.json'),
        'status':'prepared', 'created_at':value.old.now(), 'outcomes':{}, 'all_predictions_locked_before_scoring':False})
    return {'status':'prepared', 'candidate':CANDIDATE, 'unique_predictions':len(tasks),
        'new_design_predictions':new, 'reused_selected_predictions':len(tasks)-new,
        'new_native_calls_expected':native, 'native_receipts_reused_for_new_designs':new-native}


def _executor(task):
    # Candidate joint never activates the separate uncertain-half solver.
    return separate._executor(task)


def solve(task, run, monitor):
    if task['reuse_value_receipt'] is not None:
        return retained.solve(task, run, monitor)
    key = task['key']
    with monitor.lock:
        if key in monitor.state['outcomes']:
            value.old.checked(monitor.state['outcomes'][key]); return key
    monitor.check(); folder = run/'models'/key[:20]; started = time.monotonic()
    try:
        entry = None
        if folder.exists():
            if (folder/'result.json').is_file():
                entry = value.old.read(folder/'result.json')
                if (entry.get('experiment_fingerprint')!=key or entry.get('experiment_candidate')!=CANDIDATE
                        or entry.get('request_sha256')!=value.model._digest(task['request'])):
                    raise value.old.Pause('Recovery artifact belongs to different scientific inputs')
                value.old.collect_refs(entry, {}); entry = {**entry, 'result_ref':value.old.ref(folder/'result.json')}
            else:
                value.old.preserve_interrupted(folder)
        if entry is None:
            native = None
            if task['native_source_ref'] is not None:
                native = {**value.old.checked(task['native_source_ref']), 'result_ref':task['native_source_ref']}
                if not reusable_native(native, task['request']):
                    raise value.old.Pause('Native reuse requires the exact request digest')
            entry = _executor(task)(task['request'], task['sidecars']['vss'], task['sidecars']['half'], folder, ROOT, native_result=native)
        entry = deepcopy(entry); entry.update(request=task['request'], target_observations_used=False, condition='no_clinical_curve')
        if entry['status']=='predicted':
            if entry['endpoints'].get('observations_used') is not False:
                raise value.old.Pause('Prediction endpoint used observations')
            entry['endpoints'].update(comparison_observations_used=False, endpoint_estimator_uses_observations=False)
            value._native_fields(entry)
        else:
            entry['failure'] = entry.get('reason', 'Prediction failed')
    except Exception as error:
        with monitor.lock:
            count = monitor.state.get('consecutive_infrastructure_errors', 0)+1
            monitor.update(consecutive_infrastructure_errors=count, pause_reason=type(error).__name__+': '+str(error))
            if count>=3:
                monitor.halted.set()
        save(run/'models'/f'{key[:20]}.infrastructure.json', {'key':key, 'candidate':CANDIDATE,
            'reason':type(error).__name__+': '+str(error), 'scientific_failure':False, 'time':value.old.now()})
        raise value.old.Pause('Execution exception retained for review: '+str(error)) from error
    item = {'key':key, 'case_id':task['case_id'], 'entry':entry, 'reused':False,
        'native_reused_from':task['native_source_ref'], 'new_native_calls':entry.get('native_solver_calls',0),
        'new_linear_trajectory_calls':entry.get('linear_trajectory_calls',0),
        'elapsed_seconds':time.monotonic()-started, 'locked_at':value.old.now()}
    path = run/'models'/f'{key[:20]}.json'; save(path,item)
    with monitor.lock:
        monitor.state['outcomes'][key] = value.old.ref(path)
        monitor.update(completed_unique=len(monitor.state['outcomes']), consecutive_infrastructure_errors=0,
            completed_new=sum(not value.old.checked(r)['reused'] for r in monitor.state['outcomes'].values()))
    return key


def report(run):
    run = own(run)
    info = clone(retained.report, own=own, VERSION=VERSION)(run)
    state = value.old.read(run/'state.json'); protocol = value.old.checked(state['protocol'])
    result = value.old.checked(state['results']); selected = value.old.checked(protocol['baseline_results'])
    result.update(candidate=CANDIDATE, conditioning_prior=protocol['conditioning_prior'],
        selected_cl_frozen_in_every_request=False, independent_inputs_preserved=True, candidates_combined=False,
        evidence_refinement=separate.refinement_statistics(result, selected, protocol))
    save(run/'results.json',result); state['results'] = value.old.ref(run/'results.json')
    state['publication_decision'] = result['evidence_refinement']['publication_decision']; save(run/'state.json',state)
    lines = ['# Joint evidence conditioning on the selected PBPK blueprint', '',
        'Twenty previously viewed drugs, five conditions. This candidate conditions only predicted inputs on reviewed independent CL/fu. Independent evidence and the PBPK solver are unchanged. All-predicted and both-missing requests exactly reuse the selected version; no clinical curve is fitted.', '',
        '| Condition | Selected curve GMFE | Candidate curve GMFE | Improved / unchanged / worse | At least 5% better / worse | Scored pairs / requested |',
        '|---|---:|---:|---|---|---|']
    for con, counts in result['evidence_refinement']['counts_vs_selected'].items():
        before = separate._gm(separate._curves(selected,con).values())
        after = separate._gm(separate._curves(result,con).values())
        lines.append(f"| {con} | {before:.4f} | {after:.4f} | {counts['improved']} / {counts['unchanged']} / {counts['worsened']} | {counts['meaningful_improved_5pct']} / {counts['meaningful_worsened_5pct']} | {counts['complete_pairs']}/{counts['requested']} |")
    benefits = result['evidence_refinement']['evidence_benefit_actual_eight']
    lines += ['', f"Drugs with at least 5% evidence benefit: {benefits['selected']['meaningful_improved_5pct']}/8 selected baseline; {benefits['candidate']['meaningful_improved_5pct']}/8 candidate.",
        'Frozen publication decision passed: '+str(state['publication_decision']['passed'])+'. This development decision does not establish clinical safety.',
        'The absolute-error/coverage and meaningful-benefit thresholds are unchanged from candidates A/B. The old relative-error/drift gate remains diagnostic only. All eleven outputs, missing references, failures and endpoint tradeoffs are retained in results.json/results.csv.',
        'Native receipts are reused only when their complete request digest matches. Changed CL/fu requests have fresh evidence ledgers and native executions. A/B are not combined with this candidate. No per-drug candidate selection or paper update occurs.', '']
    (run/'README.md').write_text('\n'.join(lines))
    return {**info, 'candidate':CANDIDATE, 'publication_decision':state['publication_decision']}


def execute(run):
    return clone(retained.execute, own=own, solve=solve, Monitor=separate.Monitor, report=report)(own(run))


def status(run):
    return clone(separate.status, own=own)(own(run))


def cleanup(run):
    return clone(separate.cleanup, own=own)(own(run))


def background(run):
    run = own(run)
    with (run/'run.log').open('ab', buffering=0) as log:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--run-dir', str(run)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2'})
    save(run/'pid.json', {'pid':process.pid, 'project':str(ROOT), 'created_at':value.old.now()})
    return {'status':'started', 'pid':process.pid, 'log':str(run/'run.log')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run','status','report','cleanup'))
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--prior', type=Path); parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--background', action='store_true'); args = parser.parse_args()
    if args.command=='prepare':
        if args.prior is None:
            parser.error('prepare requires --prior')
        result = prepare(args.run_dir, args.prior, args.workers)
    elif args.command=='run':
        result = background(args.run_dir) if args.background else execute(args.run_dir)
    else:
        result = globals()[args.command](args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__=='__main__':
    main()
