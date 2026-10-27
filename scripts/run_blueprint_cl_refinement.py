#!/usr/bin/env python3
"""Evaluate a training-only-selected CL prior on the unchanged PBPK blueprint.

All prior value-study inputs and all twenty drugs are retained. Only predicted
total CL may change. Independent parameters and Vss/half-life stay frozen.
Clinical observations cannot select the parameter learner or individual inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_blueprint_value as value
from pbpk_kb.blueprints.source_workflow import save, sha256

VERSION = 'clinical-blueprint-cl-refinement-2'
DEFAULT = ROOT/'experiments/blueprint_design/blueprint_cl_refinement_20261004'
SOURCE = value.DEFAULT
CONDITIONS = ('all_ml', *value.SCENARIOS)


def own(path):
    p = value.inside(path, ROOT)
    if p.parent != DEFAULT.parent or not p.name.startswith('blueprint_cl_refinement_'):
        raise ValueError('Use a project blueprint_cl_refinement_ directory')
    return p


def key_for(request, sidecars, core):
    return value.fingerprint(request, sidecars, core)


def replace_predicted_cl(request, prediction, receipt, version):
    """No independent CL, renal split, binding or other parameter can change."""
    result = deepcopy(request)
    rows = [r for r in result['evidence'] if r['parameter']=='total_clearance']
    if len(rows)!=1:
        raise ValueError('Exactly one total-clearance input is required')
    row = rows[0]
    if row['basis'] != 'predicted':
        return result, False
    if isinstance(prediction, bool) or not value.scores.positive(prediction):
        raise ValueError('CL prediction must be finite and positive')
    if any(r['parameter'] in {'hepatic_clearance','renal_clearance'} for r in result['evidence']):
        raise ValueError('Do not silently replace total CL while retaining an independent split')
    if row['unit'].lower()=='ml/min/kg' and float(row['value'])==float(prediction):
        return result,False  # An unchanged selected baseline needs no simulation.
    before = deepcopy(row)
    row.update(value=float(prediction), unit='ml/min/kg',
        source={**receipt,'location':'Frozen training-only-selected total CL prediction for '+request['compound_id']},
        prediction_method={'name':'scaffold_selected_clearance_prior','version':version,
            'training_overlap':False,'applicability':{'human_IV_parameter_model':True,
            'target_structure_and_scaffold_excluded':True},
            'selection':'Training-scaffold OOF; clinical target results never used'},
        notes='Only predicted total CL replaced; independent evidence and other parameter priors preserved')
    restored = deepcopy(result)
    restored['evidence'] = [before if r['parameter']=='total_clearance' else r for r in restored['evidence']]
    if restored != request:
        raise ValueError('Mutation outside predicted CL')
    return result, True


def prepare(run, prior_path, workers=14):
    run = own(run)
    if (run/'protocol.json').exists():
        raise FileExistsError('Frozen protocol exists; resume with run')
    if not 1<=workers<=14:
        raise ValueError('Workers must be 1..14')
    source_state = value.old.read(SOURCE/'state.json')
    old_protocol = value.old.checked(source_state['protocol'])
    value.verify(old_protocol)
    if source_state['status']!='complete' or not source_state['all_predictions_locked_before_scoring']:
        raise ValueError('Completed baseline required')
    prior_path = value.inside(prior_path, ROOT, exists=True)
    prior_receipt = value.old.ref(prior_path); prior = value.old.checked(prior_receipt)
    # This small public output contract is separate from model-library artifacts.
    if prior.get('unit')!='mL/min/kg' or not prior.get('selected_using_training_only'):
        raise ValueError('Need a training-only-selected CL model with explicit units')
    for field in ('target_labels_used','target_observation_files_opened','training_structure_overlap','training_generic_scaffold_overlap'):
        if prior['audit'].get(field) is not False:
            raise ValueError('Prior exclusion/selection audit missing: '+field)
    cases = deepcopy(old_protocol['cases'])
    if set(prior['predictions'])!={c['compound_id'] for c in cases}:
        raise ValueError('CL model must cover exactly the frozen twenty-drug cohort')
    pins = deepcopy(old_protocol['source_pins'])
    value.old.collect_refs(prior,pins)
    for name,digest in prior.get('source_refs',{}).items():
        source_path=value.inside(name,ROOT,exists=True)
        if sha256(source_path)!=digest:
            raise ValueError('CL training source changed: '+name)
        pins[str(source_path.relative_to(ROOT))]=digest
    for receipt in (source_state['protocol'],source_state['results'],prior_receipt):
        value.old.checked(receipt);pins[receipt['path']]=receipt['sha256']
    tasks, units, baselines = {},[],{}
    protected=changed=0
    for case in cases:
        cid=case['case_id']
        old_units = {u['scenario']:u['key'] for u in old_protocol['units'] if u['case_id']==cid}
        old_units['all_ml']=old_protocol['baselines'][cid]['fingerprint']
        for condition in CONDITIONS:
            old_key=old_units[condition]; old_task=old_protocol['tasks'][old_key]
            request,did_change=replace_predicted_cl(old_task['request'],prior['predictions'][case['compound_id']],prior_receipt,prior['version'])
            core=old_task['core'];sidecars=old_task['sidecars']
            key=key_for(request,sidecars,core)
            if key not in tasks:
                receipt=source_state['outcomes'][old_key]
                item=value.old.checked(receipt);value.old.collect_refs(item,pins)
                pins[receipt['path']]=receipt['sha256']
                tasks[key]={'key':key,'case_id':cid,'request':request,'sidecars':sidecars,'core':core,
                            'reuse_value_receipt':None if did_change else receipt}
            units.append({'case_id':cid,'condition':condition,'key':key,'old_key':old_key,
                          'predicted_cl_replaced':did_change})
            if condition=='all_ml':baselines[cid]={'fingerprint':key}
            changed+=int(did_change)
            protected+=int(next(r for r in old_task['request']['evidence'] if r['parameter']=='total_clearance')['basis']!='predicted')
    run.mkdir(exist_ok=True,parents=True)
    (run/'inputs').mkdir(exist_ok=True)
    for task in tasks.values():
        if task['reuse_value_receipt'] is None:
            ledger_path=run/'inputs'/(task['key'][:20]+'.json')
            save(ledger_path,{'version':VERSION,'cl_prior':prior_receipt,'evidence':task['request']['evidence']})
            ledger_ref={**value.old.ref(ledger_path),'location':'Exact construction evidence for this unique input'}
            pins[ledger_ref['path']]=ledger_ref['sha256']
            task['request']['evidence_ledger']=ledger_ref
            # Validate the real immutable builder and source schema before any
            # native call. Interface defects must never become clinical scores.
            assembled=value.model.assemble_evidence_blueprint(task['request'],ROOT)
            if assembled['status']!='assembled':
                raise ValueError('Native construction preflight failed: '+str(assembled.get('gaps')))
    code={**old_protocol['code_pins'],str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__))}
    for name,digest in prior.get('code_pins',{}).items():
        if sha256(ROOT/name)!=digest:raise ValueError('CL learner source changed')
        code[name]=digest
    protocol={'version':VERSION,'created_at':value.old.now(),'cases':cases,'tasks':tasks,'units':units,'baselines':baselines,
        'source_protocol':source_state['protocol'],'baseline_results':source_state['results'],'cl_prior':prior_receipt,
        'core':old_protocol['core'],'changed_case_ids':old_protocol['changed_case_ids'],
        'source_pins':pins,'code_pins':code,'paper_guard':{},'publication_snapshot':value.old.paper_pins(),
        'workers':workers,'native_threads':2,'native_timeout_seconds':180,'directory_limit_bytes':10*1024**3,'monitor_seconds':1200,
        'new_unique_predictions':sum(t['reuse_value_receipt'] is None for t in tasks.values()),
        'logical_comparisons':100,'protected_independent_cl_units':protected,'replaced_predicted_cl_units':changed,
        'selection_contract':'Freeze CL recipe using training-scaffold OOF before clinical scoring; no per-drug switch or target calibration',
        'decision_contract':'Report curve and endpoint errors, failures and missing-input drift. Do not promote on a parameter-only score or remove unfavorable cases.',
        'development_adoption_gate':'Available-input and missing-CL curve GMFE must both not increase, with no lower scored coverage; endpoint changes remain separately reported. Passing is not independent clinical validation.',
        'limitations':['Previously viewed twenty-drug development cohort','Only CL predictor changes; PBPK structure, independent parameters, fu/Vss/half predictions frozen',
                       'Training OOF selection is not external validation','No new publication until the candidate has demonstrated value']}
    value.old.verify(protocol)
    run.mkdir(exist_ok=True,parents=True);save(run/'protocol.json',protocol)
    save(run/'state.json',{'version':VERSION,'protocol':value.old.ref(run/'protocol.json'),'status':'prepared',
                         'created_at':value.old.now(),'outcomes':{},'all_predictions_locked_before_scoring':False})
    return {'status':'prepared','unique_predictions':len(tasks),'new_native_predictions':protocol['new_unique_predictions']}


def solve(task,run,monitor):
    if task['reuse_value_receipt'] is None:
        return value.solve_task({**task,'reuse':None},run,monitor)
    key=task['key']
    with monitor.lock:
        if key in monitor.state['outcomes']:
            value.old.checked(monitor.state['outcomes'][key]);return key
    monitor.check()
    original=value.old.checked(task['reuse_value_receipt']);entry=deepcopy(original['entry'])
    if key_for(entry['request'],task['sidecars'],task['core'])!=key:
        raise ValueError('Protected input reuse does not match')
    item={'key':key,'case_id':task['case_id'],'entry':entry,'reused':True,
          'reused_from':task['reuse_value_receipt'],'new_native_calls':0,'new_linear_trajectory_calls':0,
          'elapsed_seconds':0.,'locked_at':value.old.now()}
    path=run/'models'/f'{key[:20]}.json';save(path,item)
    with monitor.lock:
        monitor.state['outcomes'][key]=value.old.ref(path)
        monitor.update(completed_unique=len(monitor.state['outcomes']))
    return key


class Monitor(value.old.Monitor):
    def check(self, force=False):
        super().check(force)
        if force:
            for path,expected in self.protocol['publication_snapshot'].items():
                if sha256(ROOT/path)!=expected:
                    raise value.old.Pause('Paper changed before prediction locking: '+path)


def execute(run):
    run=own(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=value.old.read(run/'state.json');protocol=value.old.checked(state['protocol'])
        value.old.verify(protocol)
        if state['status'] not in {'complete','predictions_locked'}:
            (run/'models').mkdir(exist_ok=True)
            monitor=Monitor(run,protocol,state)
            thread=threading.Thread(target=monitor.watch,daemon=True);thread.start()
            monitor.update(status='running',pid=os.getpid(),started_at=state.get('started_at',value.old.now()))
            try:
                monitor.check(force=True)
                with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
                    futures=[pool.submit(solve,t,run,monitor) for t in protocol['tasks'].values()]
                    try:
                        for f in as_completed(futures):f.result()
                    except Exception:
                        monitor.halted.set()
                        for f in futures:f.cancel()
                        raise
                monitor.check(force=True)
                if set(state['outcomes'])!=set(protocol['tasks']):raise ValueError('Prediction coverage differs')
                monitor.update(status='predictions_locked',all_predictions_locked_before_scoring=True,predictions_locked_at=value.old.now())
            except Exception as exc:
                monitor.halted.set();monitor.update(status='paused',pause_reason=type(exc).__name__+': '+str(exc));raise
            finally:
                monitor.done.set();thread.join(timeout=20)
    return report(run)


def report(run):
    run=own(run)
    with (run/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=value.old.read(run/'state.json');protocol=value.old.checked(state['protocol']);value.old.verify(protocol)
        if not state.get('all_predictions_locked_before_scoring') or set(state['outcomes'])!=set(protocol['tasks']):
            raise ValueError('Lock every prediction/failure before opening target observations')
        loaded={}
        for key,receipt in state['outcomes'].items():
            item=value.old.checked(receipt);task=protocol['tasks'][key];entry=item['entry']
            if (item.get('key')!=key or not item.get('locked_at') or item['case_id']!=task['case_id'] or entry.get('target_observations_used') is not False
                or entry['status'] not in {'predicted','failed'} or key_for(entry['request'],task['sidecars'],task['core'])!=key):
                raise ValueError('Prediction identity, information or lock differs')
            value.old.collect_refs(item,{})
            if entry['status']=='predicted':value._native_fields(entry)
            loaded[key]=item
        store=value.scores.Store(ROOT);scored={};rows=[]
        for case in protocol['cases']:
            clinical,observed,truth=value.evaluation.clinical_truth(case,store)
            units=[u for u in protocol['units'] if u['case_id']==case['case_id']]
            nominal=next(u['key'] for u in units if u['condition']=='available')
            base=protocol['baselines'][case['case_id']]['fingerprint']
            for unit in units:
                key=unit['key'];item=loaded[key];entry=item['entry']
                if key not in scored:
                    scored[key]=value.score_one(case,state['outcomes'][key],entry,store,clinical,observed,truth)
                row=deepcopy(scored[key]);condition=unit['condition']
                row.update(condition=condition,method=condition,fingerprint=key,prediction_reused=item['reused'],
                           prediction_receipt=state['outcomes'][key],source_hash_verified=True,
                           changed_from_nominal=key!=nominal,input_changed_from_all_ml=key!=base,
                           prediction_drift=value.drift(case,loaded[nominal]['entry'],entry))
                root_causes=entry.get('design_audit',{}).get('reasons',[])
                if entry['status']=='failed' and root_causes:
                    row['failure_root_causes']=root_causes
                    row['failure']=str(row.get('failure'))+'; root causes: '+'; '.join(root_causes)
                rows.append(row)
        summaries,pairs,missing=value.summarize(rows,protocol)
        old_results=value.old.checked(protocol['baseline_results'])
        comparisons={};clusters={c['case_id']:c['cluster_id'] for c in protocol['cases']}
        for condition in CONDITIONS:
            new={r['case_id']:r['curve_metrics'].get('gmfe') for r in rows if r['condition']==condition}
            old={r['case_id']:r['curve_metrics'].get('gmfe') for r in old_results['units'] if r['condition']==condition}
            comparisons[condition]=value.scores.paired_ratio(new,old,clusters,expected_ids=set(clusters),resamples=2000,seed=20261004)
        endpoint_pairs={}
        for family in value.evaluation.REFERENCE_FAMILIES:
            endpoint_pairs[family]={}
            for ep in value.UNITS:
                if ep=='tmax':continue
                before=[r for r in old_results['units'] if r['condition']=='available' and r['reference_family']==family]
                after=[r for r in rows if r['condition']=='available' and r['reference_family']==family]
                eligible={r['case_id'] for r in before if r['endpoints'][ep]['reference'] is not None}
                oldvals={r['case_id']:r['endpoints'][ep]['error'] for r in before}
                newvals={r['case_id']:r['endpoints'][ep]['error'] for r in after}
                endpoint_pairs[family][ep]=value.scores.paired_ratio(newvals,oldvals,clusters,expected_ids=eligible,resamples=2000,seed=20261004)
        gate=[]
        for con in ('available','missing_clearance'):
            a=next(s for s in old_results['summaries'] if s['group']=='all_20' and s['condition']==con)
            b=next(s for s in summaries if s['group']=='all_20' and s['condition']==con)
            gate.append({'condition':con,'coverage_not_lower':b['scored']>=a['scored'],
                         'curve_gmfe_not_higher':bool(value.scores.positive(b['curve']['gmfe']) and
                             value.scores.positive(a['curve']['gmfe']) and b['curve']['gmfe']<=a['curve']['gmfe'])})
        store.unchanged();value.old.verify(protocol)
        result={'version':VERSION,'protocol':state['protocol'],'cl_prior':protocol['cl_prior'],
            'all_predictions_locked_before_scoring':True,'predictions_locked_at':state['predictions_locked_at'],
            'units':rows,'summaries':summaries,'available_over_all_ml':pairs,'missingness':missing,
            'candidate_over_previous':comparisons,'previous_results':protocol['baseline_results'],
            'available_endpoint_pairs':endpoint_pairs,'development_adoption_gate':gate,
            'primary_gate_passed':all(g['coverage_not_lower'] and g['curve_gmfe_not_higher'] for g in gate),
            'new_native_calls':sum(v['new_native_calls'] for v in loaded.values()),
            'new_linear_trajectory_calls':sum(v['new_linear_trajectory_calls'] for v in loaded.values()),
            'paper_updated':False,'limitations':protocol['limitations']}
        save(run/'results.json',result)
        allrows=[];labels={}
        for variant,items in [('previous',old_results['units']),('candidate',rows)]:
            for r in items:
                c=deepcopy(r);c['method']=variant+'_'+r['condition'];labels[c['method']]=c['method'];allrows.append(c)
        writer=FunctionType(value.evaluation.write_csv.__code__,{**value.evaluation.write_csv.__globals__,'LABELS':labels})
        writer(run/'results.csv',allrows)
        lines=['# CL-prior refinement on the frozen PBPK blueprint','',
            'Twenty previously viewed drugs; only the predicted total-clearance input changes. Independent CL, fu, Vss, half-life and the PBPK core stay fixed.','',
            '| Input condition | Previous curve GMFE | Candidate GMFE | Scored / requested | Candidate / previous (95% CI) |',
            '|---|---:|---:|---:|---|']
        for con in CONDITIONS:
            a=next(s for s in old_results['summaries'] if s['group']=='all_20' and s['condition']==con)
            b=next(s for s in summaries if s['group']=='all_20' and s['condition']==con);ratio=comparisons[con]
            ci=ratio['ci95']
            lines.append(f"| {con} | {a['curve']['gmfe']:.3f} | {b['curve']['gmfe']:.3f} | {b['scored']}/20 | {ratio['estimate']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
        lines+=['','Failures remain in denominators; curve errors are drug-equal. Clinical values did not select the prior. Training-model selection does not establish clinical improvement.',
            'Development adoption gate (no worse available-input and missing-CL curve errors, with no lower coverage): '+('passed' if result['primary_gate_passed'] else 'not passed')+'. Endpoint tradeoffs remain separately reported in results.json.',
            'The previous paper remains unchanged. All eleven outputs and available references are retained; unobserved endpoints are not assigned fabricated truth.','',
            '[All 2,200 endpoint/curve rows](results.csv) · [Paired outcomes and missingness](results.json)','',
            '```bash','runtime/app-env/bin/python -B scripts/run_blueprint_cl_refinement.py status','```','']
        (run/'README.md').write_text('\n'.join(lines))
        state.update(status='complete',completed_at=value.old.now(),results=value.old.ref(run/'results.json'),
                     result_table=value.old.ref(run/'results.csv'))
        save(run/'state.json',state)
        return {'status':'complete','candidate_over_previous':comparisons,'new_native_calls':result['new_native_calls']}


def status(run):
    run=own(run);state=value.old.read(run/'state.json');protocol=value.old.checked(state['protocol']);value.old.verify(protocol)
    for receipt in state['outcomes'].values():value.old.checked(receipt)
    return {k:state.get(k) for k in ('status','completed_unique','completed_new','pause_reason','monitor','results')}


def background(run):
    run=own(run)
    with (run/'run.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'run','--run-dir',str(run)],cwd=ROOT,
             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
             env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    save(run/'pid.json',{'pid':p.pid,'project':str(ROOT),'created_at':value.old.now()})
    return {'status':'started','pid':p.pid,'log':str(run/'run.log')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','status','report'))
    parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--workers',type=int,default=14)
    parser.add_argument('--background',action='store_true');args=parser.parse_args()
    if args.command=='prepare':
        if args.prior is None:parser.error('--prior is required for prepare')
        result=prepare(args.run_dir,args.prior,args.workers)
    elif args.command=='run':result=background(args.run_dir) if args.background else execute(args.run_dir)
    else:result=globals()[args.command](args.run_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
