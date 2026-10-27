#!/usr/bin/env python3
"""Evaluate two separate evidence refinements against the selected PBPK version.

No training occurs here. Candidate A conditions predicted sidecars on independent
CL; candidate B keeps all inputs and uses the training-derived half-life interval.
All predictions/failures are locked before the frozen clinical scorer is called.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
from functools import partial
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_blueprint_cl_refinement as retained
from pbpk_kb.blueprint_learning.clinical_shared_prior import _digest as training_digest
from pbpk_kb.blueprints.source_workflow import save, sha256

value = retained.value
VERSION = 'separate-evidence-refinements-1'
BATCH = ROOT/'experiments/blueprint_design/evidence_conditioning_20261005'
SOURCE = ROOT/'experiments/blueprint_design/blueprint_cl_refinement_complementary_20261004'
CONDITIONS = retained.CONDITIONS
CANDIDATES = ('conditional', 'soft_half')
POLICY_MODULE = 'pbpk_kb.blueprint_learning.evidence_conditioned_prior'
SOFT_MODULE = 'pbpk_kb.blueprint_policy.uncertain_exchange_design'


def clone(function, **replacements):
    """Isolate globals and preserve keyword defaults; never patch live modules."""
    result = FunctionType(function.__code__, {**function.__globals__, **replacements},
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return result


def own(path):
    path = value.inside(path, ROOT)
    if path not in {BATCH/name for name in CANDIDATES}:
        raise ValueError('Use only this batch conditional/ or soft_half/ experiment directories')
    return path


def _checked_prior(path):
    path = value.inside(path, ROOT, exists=True)
    receipt = value.old.ref(path); prior = value.old.checked(receipt)
    if prior.get('result_sha256') != training_digest({k:v for k,v in prior.items() if k!='result_sha256'}):
        raise ValueError('Conditioning prior checksum mismatch')
    for field in ('target_labels_used', 'target_observation_files_opened',
                  'training_structure_overlap', 'training_generic_scaffold_overlap'):
        if prior.get('audit', {}).get(field) is not False:
            raise ValueError('Training isolation certification missing: '+field)
    width = prior.get('half_log_uncertainty_q80')
    if isinstance(width, bool) or not isinstance(width, (int,float)) or not math.isfinite(width) or width<=0:
        raise ValueError('A positive finite natural-log half-life uncertainty is required')
    for name, digest in {**prior.get('source_refs',{}), **prior.get('code_pins',{})}.items():
        if sha256(value.inside(name, ROOT, exists=True)) != digest:
            raise ValueError('Conditioning training dependency changed: '+name)
    return prior, receipt


def transformed_task(original, candidate, prior, receipt):
    """Return changed sidecars/core, while retaining every request field exactly."""
    task = deepcopy(original)
    if candidate == 'conditional':
        policy = importlib.import_module(POLICY_MODULE)
        request_before = deepcopy(task['request'])
        sidecars, audit = policy.condition_sidecars(task['request'], task['sidecars'], prior, receipt)
        if task['request'] != request_before:
            raise ValueError('Conditioning must not alter CL, renal input, fu, physiology or request')
        if set(sidecars) != {'vss','half'}:
            raise ValueError('Only the two declared sidecars are allowed')
        for name in sidecars:
            if len(sidecars[name]) != len(original['sidecars'][name]):
                raise ValueError('Do not add or drop construction evidence')
            for before,after in zip(original['sidecars'][name],sidecars[name]):
                if before.get('basis') != 'predicted' and after != before:
                    raise ValueError('Measured/derived sidecars must remain unchanged')
        task['sidecars'] = sidecars
    elif candidate == 'soft_half':
        soft = importlib.import_module(SOFT_MODULE)
        name = str(Path(soft.__file__).resolve().relative_to(ROOT))
        task['core'] = {**task['core'], 'design_version': soft.VERSION,
            'code_pins': {**task['core']['code_pins'], name: sha256(Path(soft.__file__))},
            'half_log_tolerance': prior['half_log_uncertainty_q80'], 'uncertainty_prior': receipt}
        audit = {'applied': True, 'changed_parameters': [], 'reason': 'Soft half-life objective; request and sidecars unchanged',
                 'half_log_tolerance': prior['half_log_uncertainty_q80'], 'core_version': soft.VERSION}
    else:
        raise ValueError('Unknown candidate')
    if task['request'] != original['request']:
        raise ValueError('The selected CL and entire native request must remain frozen')
    return task, audit


def prepare(run, candidate, prior_path, workers=14):
    run = own(run)
    if run.name != candidate or candidate not in CANDIDATES:
        raise ValueError('Candidate must match its batch subdirectory')
    if (run/'protocol.json').exists():
        raise FileExistsError('Protocol is frozen; resume with run')
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=14:
        raise ValueError('workers must be 1..14')
    state = value.old.read(SOURCE/'state.json')
    source = value.old.checked(state['protocol']); value.verify(source)
    if state['status']!='complete' or not state['all_predictions_locked_before_scoring']:
        raise ValueError('A completed selected baseline is required')
    prior,receipt = _checked_prior(prior_path)
    pins = deepcopy(source['source_pins']); code = deepcopy(source['code_pins'])
    for ref in (state['protocol'],state['results'],receipt,source['cl_prior']):
        if sha256(value.inside(ref['path'],ROOT,exists=True)) != ref['sha256']:
            raise ValueError('Selected baseline source changed')
        pins[ref['path']] = ref['sha256']
    for name,digest in prior['source_refs'].items():pins[name]=digest
    for name,digest in prior['code_pins'].items():code[name]=digest
    code[str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
    policy = importlib.import_module(POLICY_MODULE)
    code[str(Path(policy.__file__).resolve().relative_to(ROOT))] = sha256(Path(policy.__file__))
    if candidate=='soft_half':
        soft = importlib.import_module(SOFT_MODULE)
        code[str(Path(soft.__file__).resolve().relative_to(ROOT))]=sha256(Path(soft.__file__))
    tasks,units,baselines,adjustments = {},[],{},{}
    native_reuse_count=0
    for unit in source['units']:
        source_key=unit['key'];original=source['tasks'][source_key]
        transformed,audit=transformed_task(original,candidate,prior,receipt)
        key=retained.key_for(transformed['request'],transformed['sidecars'],transformed['core'])
        if key not in tasks:
            source_receipt=state['outcomes'][source_key]
            item=value.old.checked(source_receipt);value.old.collect_refs(item,pins);pins[source_receipt['path']]=source_receipt['sha256']
            entry=item['entry'];native_ref=entry.get('native_source_ref')
            if native_ref is None:raise ValueError('Selected baseline lacks a native execution receipt')
            native=value.old.checked(native_ref);value.old.collect_refs(native,pins);pins[native_ref['path']]=native_ref['sha256']
            if (native.get('request_sha256')!=value.model._digest(transformed['request'])
                    or native.get('partition_method')!='PKSimStandard'
                    or native.get('adapter_sha256')!=sha256(Path(value.model.__file__))):
                raise ValueError('Native reuse requires the identical request, partition and adapter')
            tasks[key]={'key':key,'case_id':unit['case_id'],'request':transformed['request'],
                'sidecars':transformed['sidecars'],'core':transformed['core'],
                'reuse_value_receipt':source_receipt if key==source_key else None,
                'native_source_ref':native_ref,'selected_prediction_receipt':source_receipt,
                'candidate':candidate,'half_log_tolerance':prior['half_log_uncertainty_q80']}
            if key!=source_key:native_reuse_count+=1
            adjustments[key]=audit
            # Full sidecar validation occurs before any experiment is launched.
            value.model._check_sidecars(transformed['request'],transformed['sidecars']['vss'],ROOT)
            value.model._check_sidecars(transformed['request'],transformed['sidecars']['half'],ROOT)
        units.append({'case_id':unit['case_id'],'condition':unit['condition'],'key':key,
                      'selected_key':source_key,'method_changed':key!=source_key})
        if unit['condition']=='all_ml':baselines[unit['case_id']]={'fingerprint':key}
    ids={c['case_id'] for c in source['cases']}
    if len(units)!=100 or {(u['case_id'],u['condition']) for u in units}!={(cid,c) for cid in ids for c in CONDITIONS}:
        raise ValueError('Retain all twenty drugs and five conditions')
    actual_core=deepcopy(next(iter(tasks.values()))['core'])
    if any(task['core']!=actual_core for task in tasks.values()):
        raise ValueError('This candidate must use one declared solver core across all drugs')
    protocol={'version':VERSION,'candidate':candidate,'created_at':value.old.now(),'cases':deepcopy(source['cases']),
        'tasks':tasks,'units':units,'baselines':baselines,'source_protocol':state['protocol'],
        'baseline_results':state['results'],'cl_prior':source['cl_prior'],'conditioning_prior':receipt,
        'core':actual_core,'changed_case_ids':source['changed_case_ids'],'adjustments':adjustments,
        'source_pins':pins,'code_pins':code,'paper_guard':{},'publication_snapshot':value.old.paper_pins(),
        'workers':workers,'native_threads':2,'native_timeout_seconds':180,'directory_limit_bytes':10*1024**3,
        'directory_budget_scope':str(BATCH.relative_to(ROOT)),'monitor_seconds':1200,
        'logical_comparisons':100,'new_unique_predictions':native_reuse_count,'expected_new_native_calls':0,
        'selected_cl_frozen_in_every_request':True,'candidates_combined':False,
        'publication_decision':{'absolute_gmfe_conditions':['available','missing_clearance'],
            'absolute_gmfe_and_scored_coverage_must_not_worsen':True,'actual_missing_cl_population':6,
            'actual_missing_cl_absolute_gmfe_must_not_worsen':True,'evidence_benefit_population':8,
            'meaningful_benefit_fraction':.05,'meaningful_evidence_benefit_count_must_exceed_selected':True,
            'expected_selected_meaningful_benefit_count':3,
            'drift_and_missing_over_available_ratio':'diagnostic only; improving available errors can increase this ratio'},
        'limitations':['Previously viewed twenty-drug development cohort','No candidate-specific drug selection',
            'CL, renal clearance, fu, native physiology and dosing stay fixed at the selected version',
            'All prior endpoints remain reported, with missing clinical references unscored',
            'Relative missing/available error is diagnostic, not a hard adoption criterion for evidence conditioning',
            'No publication or production-default update occurs in this runner']}
    value.verify(protocol,publication=True)
    run.mkdir(parents=True,exist_ok=True);save(run/'protocol.json',protocol)
    save(run/'state.json',{'version':VERSION,'candidate':candidate,'protocol':value.old.ref(run/'protocol.json'),
        'status':'prepared','created_at':value.old.now(),'outcomes':{},'all_predictions_locked_before_scoring':False})
    return {'status':'prepared','candidate':candidate,'unique_predictions':len(tasks),'new_design_predictions':native_reuse_count,
            'reused_selected_predictions':len(tasks)-native_reuse_count,'new_native_calls_expected':0}


class Monitor(retained.Monitor):
    def check(self,force=False):
        super().check(force)
        tick=time.monotonic()
        if force or tick-getattr(self,'batch_last_size',0.)>=60:
            size=0
            for path in BATCH.rglob('*'):
                if path.is_symlink():raise value.old.Pause('Unexpected symlink in bounded experiment batch')
                try:
                    if path.is_file():size+=path.stat().st_size
                except FileNotFoundError:pass
            if size>self.protocol['directory_limit_bytes']:
                self.halted.set();raise value.old.Pause('Combined evidence-conditioning batch exceeds 10 GiB')
            self.batch_last_size=tick
            self.update(batch_bytes=size)


def _executor(task):
    replacements={}
    if task['candidate']=='soft_half':
        module=importlib.import_module(SOFT_MODULE)
        replacements['simulate_learned_design']=partial(module.simulate_uncertain_design,
                                                       half_log_tolerance=task['half_log_tolerance'])
    def complete(folder,root,result):
        return value.model._complete(folder,root,{**result,'experiment_fingerprint':task['key'],
                                                 'experiment_candidate':task['candidate']})
    replacements['_complete']=complete
    return clone(value.model.run_ours,**replacements)


def solve(task,run,monitor):
    if task['reuse_value_receipt'] is not None:
        return retained.solve(task,run,monitor)
    key=task['key']
    with monitor.lock:
        if key in monitor.state['outcomes']:
            value.old.checked(monitor.state['outcomes'][key]);return key
    monitor.check();folder=run/'models'/key[:20];started=time.monotonic()
    try:
        entry=None
        if folder.exists():
            if (folder/'result.json').is_file():
                entry=value.old.read(folder/'result.json')
                if (entry.get('experiment_fingerprint')!=key or entry.get('experiment_candidate')!=task['candidate']
                        or entry.get('request_sha256')!=value.model._digest(task['request'])):
                    raise value.old.Pause('Recovery artifact belongs to different scientific inputs')
                value.old.collect_refs(entry,{});entry={**entry,'result_ref':value.old.ref(folder/'result.json')}
            else:value.old.preserve_interrupted(folder)
        if entry is None:
            native={**value.old.checked(task['native_source_ref']),'result_ref':task['native_source_ref']}
            entry=_executor(task)(task['request'],task['sidecars']['vss'],task['sidecars']['half'],folder,ROOT,native_result=native)
        entry=deepcopy(entry);entry.update(request=task['request'],target_observations_used=False,condition='no_clinical_curve')
        if entry['status']=='predicted':
            if entry['endpoints'].get('observations_used') is not False:raise value.old.Pause('Prediction endpoint used observations')
            entry['endpoints'].update(comparison_observations_used=False,endpoint_estimator_uses_observations=False)
            value._native_fields(entry)
        else:entry['failure']=entry.get('reason','Prediction failed')
    except Exception as error:
        with monitor.lock:
            count=monitor.state.get('consecutive_infrastructure_errors',0)+1
            monitor.update(consecutive_infrastructure_errors=count,pause_reason=type(error).__name__+': '+str(error))
            if count>=3:monitor.halted.set()
        save(run/'models'/f'{key[:20]}.infrastructure.json',{'key':key,'candidate':task['candidate'],
            'reason':type(error).__name__+': '+str(error),'scientific_failure':False,'time':value.old.now()})
        raise value.old.Pause('Execution exception retained for review: '+str(error)) from error
    item={'key':key,'case_id':task['case_id'],'entry':entry,'reused':False,
        'native_reused_from':task['native_source_ref'],'new_native_calls':entry.get('native_solver_calls',0),
        'new_linear_trajectory_calls':entry.get('linear_trajectory_calls',0),'elapsed_seconds':time.monotonic()-started,
        'locked_at':value.old.now()}
    path=run/'models'/f'{key[:20]}.json';save(path,item)
    with monitor.lock:
        monitor.state['outcomes'][key]=value.old.ref(path)
        monitor.update(completed_unique=len(monitor.state['outcomes']),consecutive_infrastructure_errors=0,
            completed_new=sum(not value.old.checked(r)['reused'] for r in monitor.state['outcomes'].values()))
    return key


def _curves(result,condition,ids=None):
    return {r['case_id']:r['curve_metrics'].get('gmfe') for r in result['units']
            if r['condition']==condition and (ids is None or r['case_id'] in ids)}


def comparison_counts(candidate,selected,ids):
    records={};counts={'improved':0,'unchanged':0,'worsened':0,'meaningful_improved_5pct':0,
        'meaningful_worsened_5pct':0,'both_unscored':0,'recovered':0,'new_unscored':0}
    for cid in sorted(ids):
        after,before=candidate.get(cid),selected.get(cid)
        valid_after=value.scores.positive(after);valid_before=value.scores.positive(before)
        if not valid_before or not valid_after:
            status='both_unscored' if not valid_after and not valid_before else ('recovered' if valid_after else 'new_unscored')
            counts[status]+=1;records[cid]={'status':status,'ratio':None};continue
        ratio=after/before
        status='unchanged' if math.isclose(ratio,1.,rel_tol=1e-12,abs_tol=1e-12) else ('improved' if ratio<1 else 'worsened')
        counts[status]+=1
        meaningful_better=ratio<=.95+1e-12;meaningful_worse=ratio>=1.05-1e-12
        counts['meaningful_improved_5pct']+=int(meaningful_better);counts['meaningful_worsened_5pct']+=int(meaningful_worse)
        records[cid]={'status':status,'ratio':ratio,'meaningful_improved_5pct':meaningful_better,'meaningful_worsened_5pct':meaningful_worse}
    return {'requested':len(ids),'complete_pairs':counts['improved']+counts['unchanged']+counts['worsened'],**counts,'per_drug':records}


def _gm(values):
    valid=[v for v in values if value.scores.positive(v)]
    return math.exp(math.fsum(math.log(v) for v in valid)/len(valid)) if valid else None


def refinement_statistics(result,selected,protocol):
    ids={c['case_id'] for c in protocol['cases']};eight=set(protocol['changed_case_ids'])
    all_counts={con:comparison_counts(_curves(result,con),_curves(selected,con),ids) for con in CONDITIONS}
    selected_benefit=comparison_counts(_curves(selected,'available'),_curves(selected,'all_ml'),eight)
    candidate_benefit=comparison_counts(_curves(result,'available'),_curves(result,'all_ml'),eight)
    old_rows=[r for r in selected['units'] if r['condition']=='missing_clearance' and r['changed_from_nominal']]
    six={r['case_id'] for r in old_rows}
    new_six={r['case_id'] for r in result['units'] if r['condition']=='missing_clearance' and r['changed_from_nominal']}
    if len(eight)!=8 or len(six)!=6 or six!=new_six:raise ValueError('Declared evidence/missing-CL populations changed')
    if selected_benefit['meaningful_improved_5pct']!=protocol['publication_decision']['expected_selected_meaningful_benefit_count']:
        raise ValueError('Selected meaningful-benefit baseline differs from the frozen decision contract')
    checks={};absolute={}
    for con in ('available','missing_clearance'):
        after=_curves(result,con);before=_curves(selected,con)
        ga,gb=_gm(after.values()),_gm(before.values())
        checks[con+'_gmfe_not_worse']=ga is not None and gb is not None and ga<=gb*(1+1e-12)
        checks[con+'_coverage_not_lower']=sum(value.scores.positive(v) for v in after.values())>=sum(value.scores.positive(v) for v in before.values())
        absolute[con]={'candidate':ga,'selected':gb}
    after=_curves(result,'missing_clearance',six);before=_curves(selected,'missing_clearance',six)
    ga,gb=_gm(after.values()),_gm(before.values())
    checks['actual_six_missing_cl_gmfe_not_worse']=ga is not None and gb is not None and ga<=gb*(1+1e-12)
    checks['actual_six_missing_cl_coverage_not_lower']=sum(value.scores.positive(v) for v in after.values())>=sum(value.scores.positive(v) for v in before.values())
    checks['more_drugs_with_5pct_evidence_benefit']=candidate_benefit['meaningful_improved_5pct']>selected_benefit['meaningful_improved_5pct']
    old_missing=next(r for r in selected['missingness'] if r['scenario']=='missing_clearance')
    new_missing=next(r for r in result['missingness'] if r['scenario']=='missing_clearance')
    old_gate={'primary_absolute_error_and_coverage':all(checks[k] for k in checks if k.startswith(('available_','missing_clearance_'))),
        'drift_not_worse':new_missing['prediction_drift']['estimate']<=old_missing['prediction_drift']['estimate']*(1+1e-12),
        'missing_over_available_error_ratio_not_worse':new_missing['masked_over_nominal_clinical_gmfe']['estimate']<=old_missing['masked_over_nominal_clinical_gmfe']['estimate']*(1+1e-12)}
    return {'comparison_baseline':protocol['baseline_results'],'counts_vs_selected':all_counts,
        'evidence_benefit_actual_eight':{'selected':selected_benefit,'candidate':candidate_benefit},
        'absolute_curve_gmfe':absolute,'actual_six_missing_cl':{'case_ids':sorted(six),'candidate_gmfe':ga,'selected_gmfe':gb,
            'counts':comparison_counts(after,before,six)},'publication_decision':{'checks':checks,'passed':all(checks.values()),
            'meaningful_threshold_fraction':.05,'majority_of_eight_benefit':candidate_benefit['meaningful_improved_5pct']>4,
            'clinical_safety_established':False},
        'old_joint_gate_diagnostic':{'checks':old_gate,'passed':all(old_gate.values()),'used_for_adoption':False,
            'reason':'Improving available-input error can increase missing/available error, so that ratio is not a hard gate for candidate A'}}


def report(run):
    run=own(run)
    result_info=clone(retained.report,own=own,VERSION=VERSION)(run)
    state=value.old.read(run/'state.json');protocol=value.old.checked(state['protocol'])
    result=value.old.checked(state['results']);selected=value.old.checked(protocol['baseline_results'])
    result['evidence_refinement']=refinement_statistics(result,selected,protocol)
    result['candidate']=protocol['candidate'];result['conditioning_prior']=protocol['conditioning_prior']
    result['selected_cl_frozen_in_every_request']=True;result['candidates_combined']=False
    save(run/'results.json',result);state['results']=value.old.ref(run/'results.json')
    state['publication_decision']=result['evidence_refinement']['publication_decision'];save(run/'state.json',state)
    lines=['# Evidence refinement: '+protocol['candidate'],'',
        'Compared with the frozen selected complementary-CL blueprint. All twenty drugs and five input conditions are retained. No training or clinical-curve fitting occurs here.',
        '', '| Condition | Improved / unchanged / worse | At least 5% better / worse | Scored pairs / requested |',
        '|---|---|---|---|']
    for con,counts in result['evidence_refinement']['counts_vs_selected'].items():
        lines.append(f"| {con} | {counts['improved']} / {counts['unchanged']} / {counts['worsened']} | {counts['meaningful_improved_5pct']} / {counts['meaningful_worsened_5pct']} | {counts['complete_pairs']}/{counts['requested']} |")
    benefits=result['evidence_refinement']['evidence_benefit_actual_eight']
    lines+=['',f"Drugs with at least 5% evidence benefit: {benefits['selected']['meaningful_improved_5pct']}/8 selected baseline; {benefits['candidate']['meaningful_improved_5pct']}/8 candidate.",
        'Publication decision passed: '+str(state['publication_decision']['passed'])+'. This development decision does not establish clinical safety.',
        'The previous relative-error/drift joint gate is retained as a diagnostic only. Every endpoint, failure, missing reference and paired confidence interval remains in results.json/results.csv.',
        'Frozen native executions are reused only for identical native requests. The two candidates are separate experiments; no per-drug candidate selection is performed.','']
    (run/'README.md').write_text('\n'.join(lines))
    return {**result_info,'candidate':protocol['candidate'],'publication_decision':state['publication_decision']}


def execute(run):
    return clone(retained.execute,own=own,solve=solve,Monitor=Monitor,report=report)(own(run))


def status(run):
    output=clone(retained.status,own=own)(own(run));state=value.old.read(own(run)/'state.json')
    return {**output,'candidate':state['candidate'],'publication_decision':state.get('publication_decision')}


def background(run):
    run=own(run)
    with (run/'run.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run','--candidate',run.name,'--run-dir',str(run)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    save(run/'pid.json',{'pid':process.pid,'project':str(ROOT),'created_at':value.old.now()})
    return {'status':'started','pid':process.pid,'log':str(run/'run.log')}


def cleanup(run):
    run=own(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=value.old.read(run/'state.json');protocol=value.old.checked(state['protocol']);value.verify(protocol)
        if state['status']!='complete':raise ValueError('Cleanup requires a completed experiment')
        pinned={value.inside(p,ROOT) for p in protocol['source_pins']}
        deleted=[]
        for path in run.rglob('*.tmp'):
            if path.is_symlink():raise ValueError('Refuse symlink cleanup')
            if path.is_file() and path.resolve() not in pinned:
                path.unlink();deleted.append(str(path.relative_to(ROOT)))
        return {'deleted_temporary_files':deleted,'models_and_failure_diagnostics_retained':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','status','report','cleanup'))
    parser.add_argument('--candidate',choices=CANDIDATES,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--prior',type=Path);parser.add_argument('--workers',type=int,default=14)
    parser.add_argument('--background',action='store_true');args=parser.parse_args()
    run=own(args.run_dir)
    if run.name!=args.candidate:parser.error('candidate must match run directory')
    if args.command=='prepare':
        if args.prior is None:parser.error('prepare requires --prior')
        result=prepare(run,args.candidate,args.prior,args.workers)
    elif args.command=='run':result=background(run) if args.background else execute(run)
    else:result=globals()[args.command](run)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
