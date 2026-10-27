#!/usr/bin/env python3
"""Bounded development-only CL-policy selection; no target-driven policy tuning."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import run_exposure_blueprint as base
from pbpk_kb.blueprint_policy import development_prior_selection as selector
from pbpk_kb.blueprints.development_curve_blueprint import _training_document

ROOT=base.ROOT
PRIOR=base.RUN
RUN=ROOT/'experiments/blueprint_design/development_prior_selection_20261008'
VERSION='development-cl-prior-selection-1'
METHODS=('fixed_prior','direct_relaxed')
BUDGET=4096
STARTS=16
read,save,ref,checked,verify=base.read,base.save,base.ref,base.checked,base.verify

def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():raise ValueError('Protocol already frozen')
    prior=read(PRIOR/'protocol.json');state=read(PRIOR/'state.json')
    if state['status']!='complete':raise ValueError('Previous final candidate pool must be complete')
    checked(state['protocol']);verify(prior['source_pins']);verify(state['output_pins'])
    pins={**prior['source_pins'],**state['output_pins']}
    for path in (Path(__file__),ROOT/'src/pbpk_kb/blueprint_policy/development_prior_selection.py',PRIOR/'protocol.json',PRIOR/'state.json',PRIOR/'results.json'):
        pins[str(path.relative_to(ROOT))]=base.sha256(path)
    jobs={f'{c["case_id"]}__{s}':{'case_id':c['case_id'],'sampling':s,'training_source':c['sampling_sources'][s]} for c in prior['cases'] for s in base.SAMPLINGS}
    candidates={key:unit for key,unit in state['units'].items() if unit['method'] in METHODS}
    protocol={'version':VERSION,'created_at':time.time(),'cases':prior['cases'],'jobs':jobs,'final_candidates':candidates,
        'physiology_cache':prior['physiology_cache'],'source_pins':pins,'score_reference':ref(PRIOR/'results.json'),
        'selection':'Five deterministic time-rank modulo folds; compare pooled held-out absolute log errors. Within each fold, formula chosen on training objective only. Tie <=1e-12 prefers fixed_prior.',
        'fold_budget_per_candidate_formula':BUDGET,'starts':STARTS,'folds':5,'expected_formula_fits':400,
        'final_candidate_reuse':'Exact prior independently fitted full/five candidate pool at 16384 calls/formula; no CV fold warm starts. No additional independent predictions claimed.',
        'scope':'Previously viewed ten-drug development cohort. Four-point fold training in the five-point condition can make selection unstable. No missing-parameter or zero-curve claim.',
        'interpretation_rule':'If selection does not improve both full/five means over always-relaxed, stop tuning this selector on these targets. Paired intervals crossing one do not support general superiority.',
        'no_threshold_search':True,'max_workers':8,'monitor_seconds':1200,'directory_limit_gib':10}
    save(RUN/'protocol.json',protocol)
    save(RUN/'state.json',{'version':VERSION,'status':'prepared','protocol':ref(RUN/'protocol.json'),'units':{}})
    return {'jobs':len(jobs),'formula_fits':400,'final_candidates_reused':len(candidates)}

class Guard:
    def __init__(self,protocol):self.protocol=protocol;self.last=time.monotonic()
    def __call__(self):
        if (RUN/'PAUSE').exists():raise RuntimeError('Batch paused')
        if time.monotonic()-self.last<1200:return
        verify(self.protocol['source_pins'])
        size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
        if size>10*1024**3:raise RuntimeError('Batch exceeds 10 GiB')
        self.last=time.monotonic()

def simulate(physiology,request,times,parameters):
    return base.engine._simulate(physiology,request,times,'flow_exchange',base.engine.PARAMETERS['flow_exchange'],parameters)

def execute(protocol,key,job):
    folder=RUN/'work'/key;folder.mkdir(parents=True,exist_ok=True)
    existing=folder/'result.json';digest=base.sha256(RUN/'protocol.json')
    if existing.exists():
        result=read(existing)
        if result['protocol_sha256']!=digest:raise ValueError('Resume protocol changed')
        verify(result['output_pins']);return result
    case=next(c for c in protocol['cases'] if c['case_id']==job['case_id'])
    request=base.normalized(checked(case['requests']['evidence']['development']))
    training=_training_document(job['training_source'],request,ROOT)
    if job['sampling']=='five' and len(training['observations'])!=5:raise ValueError('Five-point input changed')
    physiology={f:checked(protocol['physiology_cache'][f'{job["case_id"]}__development__{f}']) for f in base.FORMULAS}
    guard=Guard(protocol);guard();started=time.monotonic()
    selection=selector.select(physiology,request,training,
        {'fixed_prior':base.fit_cell,'direct_relaxed':base.relaxed_fit},simulate,
        budget=BUDGET,starts=STARTS,before_trial=guard)
    save(folder/'selection.json',selection)
    result={**job,'protocol_sha256':digest,'selection':ref(folder/'selection.json'),
        'chosen_method':selection['selected_method'],'status':'selected' if selection['selected_method'] else 'blocked',
        'locked_at':time.time(),'wall_seconds':time.monotonic()-started}
    if selection['selected_method']:
        candidate=protocol['final_candidates'][key+'__'+selection['selected_method']]
        if candidate['status']!='predicted':raise ValueError('Chosen final candidate has no frozen prediction')
        result['prediction']=candidate['prediction'];result['request']=candidate['request']
    result['output_pins']={};base.pins_in({k:v for k,v in result.items() if k!='output_pins'},result['output_pins'])
    save(existing,result);return result

def run(workers):
    protocol=read(RUN/'protocol.json');state=read(RUN/'state.json');checked(state['protocol']);verify(protocol['source_pins'])
    if state['status']=='complete':return status()
    state.update(status='running',pid=os.getpid(),started_at=state.get('started_at') or time.time());save(RUN/'state.json',state)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(execute,protocol,key,job):key for key,job in protocol['jobs'].items() if key not in state['units']}
        try:
            for task in as_completed(tasks):
                key=tasks[task];state['units'][key]=task.result();save(RUN/'state.json',state)
                print(json.dumps({'completed':len(state['units']),'total':20,'unit':key}),flush=True)
        except Exception as exc:
            save(RUN/'PAUSE',{'reason':str(exc)});state.update(status='paused',reason=str(exc));save(RUN/'state.json',state);raise
    verify(protocol['source_pins']);state.update(status='selections_locked',locked_at=time.time());save(RUN/'state.json',state)
    return report()

def report():
    from report_blueprint_validation import clustered_geomean,paired_ratio
    protocol=read(RUN/'protocol.json');state=read(RUN/'state.json');checked(state['protocol'])
    if state['status'] not in {'selections_locked','complete'} or set(state['units'])!=set(protocol['jobs']):raise ValueError('Lock every selection before target scoring')
    verify(protocol['source_pins'])
    for unit in state['units'].values():verify(unit['output_pins'])
    # First opening of target scores in this program. Predictions are reused,
    # not represented as new trials or unseen clinical studies.
    prior=checked(protocol['score_reference'])
    index={(r['case_id'],r['sampling'],r['method']):r for r in prior['records']}
    records=[];choices=[];costs={'fitting_calls':0,'fitting_failures':0,'validation_calls':0,'validation_failures':0,'formula_fits':0,'new_final_predictions':0,'reused_final_candidates':40}
    for key,unit in state['units'].items():
        selection=checked(unit['selection']);cid,s=unit['case_id'],unit['sampling'];chosen=unit['chosen_method']
        for name in ('fitting_calls','fitting_failures','validation_calls','validation_failures'):costs[name]+=selection[name]
        costs['formula_fits']+=sum(len(f['formula_fits']) for m in selection['methods'].values() for f in m['folds'])
        choices.append({'case_id':cid,'sampling':s,'chosen_method':chosen,'selection':unit['selection']})
        for method in ('adaptive',)+METHODS:
            selected=chosen if method=='adaptive' else method
            if selected:
                r=deepcopy(index[cid,s,selected]);r.update(method=method,selected_method=selected,prediction_reused=True);records.append(r)
            else:
                reference=index[cid,s,'fixed_prior']
                r={'case_id':cid,'drug':reference['drug'],'sampling':s,'method':method,'status':'blocked',
                    'curve_metrics':{'gmfe':None},'reason':'Neither policy has a complete valid CV score',
                    'endpoints':{ep:{'value':None,'reference':v['reference'],'unit':v['unit'],'error':None,
                        'error_metric':v.get('error_metric'),'weak_tail':None} for ep,v in reference['endpoints'].items()}}
                records.append(r)
    clusters={c['case_id']:c['cluster_id'] for c in protocol['cases']};summary={};comparisons={}
    for sampling in base.SAMPLINGS:
        scores={m:{r['case_id']:r['curve_metrics'].get('gmfe') for r in records if r['method']==m and r['sampling']==sampling} for m in ('adaptive',)+METHODS}
        summary[sampling]={}
        for method,values in scores.items():
            valid={k:v for k,v in values.items() if v is not None and math.isfinite(v) and v>0}
            summary[sampling][method]=clustered_geomean(valid,{k:clusters[k] for k in valid})|{'attempted':10,'failed_or_unscorable':10-len(valid),'within_twofold':sum(v<=2 for v in valid.values())}
        comparisons[sampling]={m:paired_ratio(scores['adaptive'],scores[m],clusters) for m in METHODS}
    results={'version':VERSION,'protocol':state['protocol'],'status':'complete','locked_at':state['locked_at'],'scored_at':time.time(),
        'records':records,'selections':choices,'summary':summary,'comparisons':comparisons,'additional_cv_resources':costs,
        'limitations':['Retrospective development comparison on already viewed drugs.','Within-study point CV does not guarantee cross-study transfer.','Final candidate trajectories reused; no new independent predictions.','No conclusion on missing-parameter stability, zero-curve performance, or clinical risk.']}
    save(RUN/'results.json',results)
    rows=[]
    for r in records:
        head={k:r[k] for k in ('drug','sampling','method','status')}
        rows.append(head|{'endpoint':'concentration_time_curve','prediction':None,'reference':None,'unit':None,'error':r['curve_metrics'].get('gmfe'),'metric':'GMFE','weak_tail':None})
        for ep,v in r['endpoints'].items():rows.append(head|{'endpoint':ep,'prediction':v['value'],'reference':v['reference'],'unit':v['unit'],'error':v['error'],'metric':v.get('error_metric'),'weak_tail':v.get('weak_tail')})
    with (RUN/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Automatic CL-constraint selection from development curves: ten-drug re-evaluation','','This tests the selection rule without changing the human PBPK equations. Frozen target scores are read only after all selections are locked; all data were previously inspected during development.','', '| Method | Full GMFE | Five-point GMFE |','|---|---:|---:|']
    for method in ('adaptive',)+METHODS:
        a,b=summary['full'][method]['estimate'],summary['five'][method]['estimate']
        lines.append(f'| {method} | {a if a is None else round(a,4)} | {b if b is None else round(b,4)} |')
    lines+=['','[All 11 endpoint types](results.csv) · [Selections, paired intervals, and failures](results.json).','',
        'fold = time-order index mod 5. Each fold/policy/formula gets 4096 physical fitting calls and 16 fixed starts. Formula selection uses training objectives only; policy selection uses pooled held-point mean absolute log error, retaining the fixed prior for differences <=1e-12. A policy is ineligible if any fold has no feasible solution. Leaving one of five points leaves four and may destabilize selection.',
        'Final candidates reuse the preceding 40 independent full/five-point fits and locked predictions. Five-point runs are not warm-started from CV folds or full curves. Additional CV costs are reported separately; equal compute to a single fixed policy is not claimed.',
        'The project already included two-fold weak/strong regularization CV. This batch only retests selection after truly removing the CL constraint. If it does not beat uniform relaxation, further tuning of folds, thresholds, or regularization grids on these targets stops rather than being presented as a new innovation.',
        '', 'Use `runtime/app-env/bin/python -B scripts/run_development_prior_selection.py status`; `report` recalculates the report.']
    (RUN/'README.md').write_text('\n'.join(lines)+'\n')
    state.update(status='complete',completed_at=time.time());save(RUN/'state.json',state)
    return {'status':'complete','units':20,'rows':len(rows),'summary':summary,'comparisons':comparisons}

def status():
    if not (RUN/'state.json').exists():return {'status':'not_prepared'}
    state=read(RUN/'state.json')
    return {'status':state['status'],'completed':len(state['units']),'total':20,'reason':state.get('reason')}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','status','report']);parser.add_argument('--workers',type=int,default=8);parser.add_argument('--background',action='store_true');args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('workers must be 1..8')
    RUN.mkdir(parents=True,exist_ok=True)
    if args.command=='status':print(json.dumps(status(),ensure_ascii=False));return
    if args.command=='run' and args.background:
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'}
        with (RUN/'run.log').open('ab') as log:process=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'run','--workers',str(args.workers)],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
        (RUN/'run.pid').write_text(str(process.pid)+'\n');print(json.dumps({'pid':process.pid}));return
    with (RUN/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run(args.workers) if args.command=='run' else globals()[args.command]()
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
