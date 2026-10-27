#!/usr/bin/env python3
"""Small clinical pilot for source-condition bundle selection, never target fitting."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pbpk_kb.blueprints.source_workflow import save, sha256
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
from pbpk_kb.blueprint_policy import transport_calibration as engine
from pbpk_kb.blueprint_policy.condition_matched_selection import select_bundles
from pbpk_kb.blueprints.development_curve_blueprint import _training_document
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
import numpy as np
from scipy.optimize import least_squares

RUN = ROOT/'experiments/blueprint_design/condition_matched_selection_20261007'
PAPER = ROOT/'experiments/blueprint_design/vascular_tissue_ten_drug_20260930'
VERSION = 'condition-matched-clinical-pilot-1'
METHODS = ('joint', 'independent', 'unconstrained_search')
SAMPLINGS = ('full', 'five')
FORMULAS = ('PKSimStandard', 'RodgersRowland')
BUDGET = 4096  # actual objective calls, including finite differences and failures
STARTS = 16    # total starts across bundles/formulas, not per candidate

def read(p):
    return json.loads(Path(p).read_text())

def ref(p):
    p=Path(p).resolve();p.relative_to(ROOT)
    return {'path':str(p.relative_to(ROOT)), 'sha256':sha256(p), 'location':'Frozen pilot input or output'}

def checked(r):
    p=(ROOT/r['path']).resolve();p.relative_to(ROOT)
    if sha256(p)!=r['sha256']: raise ValueError('Source changed: '+r['path'])
    return read(p)

def pins_in(value, pins):
    if isinstance(value, dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            p=(ROOT/value['path']).resolve();p.relative_to(ROOT)
            if sha256(p)!=value['sha256']: raise ValueError('Source changed: '+value['path'])
            pins[value['path']]=value['sha256']
        for v in value.values():pins_in(v,pins)
    elif isinstance(value,list):
        for v in value:pins_in(v,pins)

def verify(pins):
    for name,digest in pins.items():
        p=(ROOT/name).resolve();p.relative_to(ROOT)
        if sha256(p)!=digest:raise ValueError('Source changed: '+name)

def allocated(total, count):
    if count<1 or count>total:raise ValueError('Invalid budget allocation')
    return [total//count+(i<total%count) for i in range(count)]

def make_request(original, bundle, folder):
    request=deepcopy(original)
    selected={r['parameter']:r for r in bundle['records']}
    rows=[r for r in request['evidence'] if r['parameter'] not in {'total_clearance','renal_clearance','hepatic_clearance','fraction_unbound'}]
    for parameter,r in selected.items():
        if parameter not in {'total_clearance','renal_clearance','fraction_unbound'}:continue
        source={k:r['source'][k] for k in ('path','sha256','location')}
        context=deepcopy(r['context']);context['study_role']='development'
        context.update(species='human',matrix='plasma')
        if parameter.endswith('clearance'):context['route']='intravenous'
        rows.append({'parameter':parameter,'compound_id':request['compound_id'],'value':float(r['value']),
            'unit':'ml/min/kg' if parameter.endswith('clearance') else 'fraction','basis':r['basis'],
            'source':source,'context':context,'notes':'Selected candidate '+r['candidate_id']+'; '+str(r.get('normalization',{}))})
    scope=next(r for r in rows if r['parameter']=='elimination_scope')
    scope['value']='hepatic_and_renal' if 'renal_clearance' in selected else 'hepatic_only'
    ledger=folder/'evidence.json';save(ledger,{'evidence':rows})
    request.update(evidence=rows,evidence_ledger=ref(ledger))
    return EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json',exclude_none=True)

class Guard:
    def __init__(self,protocol):
        self.protocol=protocol;self.last=0.;self.lock=threading.Lock()
    def __call__(self,force=False):
        with self.lock:
            if (RUN/'PAUSE').exists():raise RuntimeError('Paused by project PAUSE file')
            if not force and time.monotonic()-self.last<1200:return
            verify(self.protocol['source_pins'])
            size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
            if size>10*1024**3:raise RuntimeError('Batch exceeded 10 GiB; new submission paused')
            self.last=time.monotonic()
            save(RUN/'monitor.json',{'checked_at':time.time(),'bytes':size,'pid':os.getpid(),'source_integrity':True})

class CallLimit(Exception):pass

def fit_cell(physiology,request,training,call_limit,start_count,guard):
    """Hard cap includes finite-difference, failed and final objective calls."""
    mw=next(float(r['value']) for r in request['evidence'] if r['parameter']=='molecular_weight')
    times=np.asarray([r['time_min']/60 for r in training['observations']],float)
    obs=np.asarray([r['concentration_umol_per_l']*mw/1000 for r in training['observations']],float)
    if np.any(obs<=0) or not np.isfinite(obs).all():raise ValueError('Invalid development concentrations')
    keys,lower,upper,starts=engine.parameter_starts('flow_exchange')
    costs=allocated(call_limit,start_count);calls=0;failures=0;best=None;trace=[]
    began=time.monotonic()
    for index,(x0,cap) in enumerate(zip(starts,costs)):
        used=0;local=None
        def residual(x):
            nonlocal used,calls,failures,local,best
            if used>=cap:raise CallLimit()
            guard();used+=1;calls+=1
            try:
                prediction=engine._simulate(physiology,request,times,'flow_exchange',keys,x)
                p=np.asarray(prediction['concentration_mg_l'])
                if p.shape!=obs.shape or np.any(p<=0) or not np.isfinite(p).all():raise ValueError('Invalid concentration shape or values')
                data=(np.log(p)-np.log(obs))/math.sqrt(len(obs))
                penalty=np.r_[math.sqrt(.1)*x[0]/math.log(2),.1*x[1]/math.log(10),.1*x[2]/math.log(100)]
                r=np.r_[data,penalty];score=float(r@r)
                row={'objective':score,'log_parameters':np.asarray(x).tolist(),
                    'development_ln_rmse':float(np.linalg.norm(data)),'start_index':index}
                if local is None or score<local['objective']:local=row
                if best is None or score<best['objective']:best=row
                return r
            except (ValueError,ArithmeticError,np.linalg.LinAlgError):
                failures+=1;return np.full(len(obs)+3,1e6)
        status='converged'
        try:
            fitted=least_squares(residual,x0,bounds=(lower,upper),max_nfev=400,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            if not fitted.success:status='optimizer_limit'
        except CallLimit:status='call_budget_exhausted'
        trace.append({'index':index,'status':status,'calls':used,'call_limit':cap,'best':local})
    return {'best':best,'starts':trace,'calls':calls,'failures':failures,'call_limit':call_limit,
        'wall_seconds':time.monotonic()-began,'no_warm_start':True}

def prepare():
    if (RUN/'protocol.json').exists():raise ValueError('Already prepared; use run/status')
    candidates=read(RUN/'candidates.json')
    paper=read(PAPER/'protocol.json');paper_state=read(PAPER/'state.json')
    inventory={x['case_id']:x for x in candidates['inventory'] if x.get('eligible_for_multisource_experiment')}
    cases=[];jobs={};selections={};pins=engine._code_pins(ROOT)
    pins_in(candidates,pins)
    for c in paper['cases']:
        cid=c['case_id']
        if cid not in inventory:continue
        entry=inventory[cid]
        if 'selector_context' not in entry:raise ValueError('Candidate context is not finalized')
        rows=[r for r in candidates['candidates'] if r['case_id']==cid]
        cases.append(deepcopy(c));selections[cid]={}
        pins_in(c,pins)
        for method in METHODS:
            selection=select_bundles(rows,entry['selector_context'],method=method,limit=8)
            selections[cid][method]=selection
            for sampling in SAMPLINGS:
                key=f'{cid}__{sampling}__{method}'
                configs=[]
                for bundle in selection['bundles']:
                    for formula in FORMULAS:configs.append({'bundle':bundle,'formula':formula})
                if configs:
                    budgets=allocated(BUDGET,len(configs));starts=allocated(STARTS,len(configs))
                    for row,calls,n in zip(configs,budgets,starts):row.update(call_limit=calls,start_count=n)
                jobs[key]={'case_id':cid,'sampling':sampling,'method':method,'configs':configs,
                    'training_source':deepcopy(c['sampling_sources'][sampling]),'selection_status':selection['status']}
    if not cases:raise ValueError('No verified multi-source cases; no synthetic clinical substitutes')
    for path in (Path(__file__),ROOT/'src/pbpk_kb/blueprint_policy/condition_matched_selection.py',RUN/'candidates.json',
                 PAPER/'protocol.json',PAPER/'state.json',ROOT/'scripts/report_blueprint_validation.py'):
        pins[str(path.relative_to(ROOT))]=sha256(path)
    controls={}
    for c in cases:
        for sampling in SAMPLINGS:
            for original,label in [('flow_exchange','frozen_paper_pbpk'),('two_compartment','two_compartment')]:
                old_id=f'{c["case_id"]}__{sampling}__{original}'
                u=deepcopy(paper_state['computations'][old_id]);pins_in(u,pins)
                controls[f'{c["case_id"]}__{sampling}__{label}']={**u,'method':label,'sampling':sampling,'case_id':c['case_id'],'reused':True}
    protocol={'version':VERSION,'created_at':time.time(),'cases':cases,'jobs':jobs,'selections':selections,'controls':controls,
        'source_pins':pins,'sampling':list(SAMPLINGS),'methods':list(METHODS),'limit_bundles':8,
        'total_objective_budget_per_method_drug_sampling':BUDGET,'total_starts_per_method_drug_sampling':STARTS,
        'formula_methods':list(FORMULAS),'equal_budget':'Same total permitted objective calls including failed and finite differences; actual usage also reported.',
        'selection_basis':'development penalized log concentration loss only',
        'scope':'Previously inspected development cohort; condition selection and clinical pilot, not blinded validation.',
        'original_paper_control_budget_differs':True,'target_used_for_selection':False,
        'monitor_seconds':1200,'directory_limit_gib':10,
        'pre_scoring_correction':{'reason':'Label total CL has unspecified population; excluded by all methods before target scoring.',
            'unscored_native_constructions_discarded':4,'target_predictions_or_scores_available':False}}
    verify(pins);save(RUN/'protocol.json',protocol)
    save(RUN/'state.json',{'version':VERSION,'status':'prepared','protocol':ref(RUN/'protocol.json'),'units':{},'created_at':time.time()})
    return {'drugs':len(cases),'jobs':len(jobs),'controls_reused':len(controls),'candidate_counts':{k:{m:len(v['bundles']) for m,v in d.items()} for k,d in selections.items()}}

def execute(protocol,job,key,guard):
    case=next(c for c in protocol['cases'] if c['case_id']==job['case_id'])
    folder=RUN/'work'/key;folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'result.json'
    if result_path.exists():
        old=read(result_path)
        if old.get('protocol_sha256')!=sha256(RUN/'protocol.json'):raise ValueError('Changed resumed protocol')
        verify(old.get('output_pins',{}));return old
    result={'case_id':job['case_id'],'sampling':job['sampling'],'method':job['method'],
        'protocol_sha256':sha256(RUN/'protocol.json'),'resources':{'objective_calls':0,'failed_objective_calls':0,'native_constructions':0},'candidate_scores':[]}
    start=time.monotonic()
    if not job['configs']:
        result.update(status='blocked',reason=job['selection_status']);save(result_path,result);return result
    dev=checked(case['requests']['evidence']['development']);target=checked(case['requests']['evidence']['target'])
    training=_training_document(job['training_source'],dev,ROOT)
    if job['sampling']=='five' and len(training['observations'])!=5:raise ValueError('Five-point isolation failed')
    best=None
    for index,config in enumerate(job['configs']):
        guard();cell=folder/f'candidate_{index:02d}';cell.mkdir(parents=True,exist_ok=True)
        request=make_request(dev,config['bundle'],cell);request['partition_method']=config['formula']
        fit_path=cell/'fit.json'
        if fit_path.exists():
            fitted=read(fit_path)
            if (fitted.get('config')!=config or fitted.get('request')!=request or
                fitted.get('calls',config['call_limit']+1)>config['call_limit'] or
                len(fitted.get('starts',[]))!=config['start_count']):
                raise ValueError('Resumed candidate fit/configuration mismatch')
        else:
            physiology=engine._physiology(request,cell,ROOT,guard)
            fitted=fit_cell(physiology,request,training,config['call_limit'],config['start_count'],guard)
            fitted.update(config=config,request=request);save(fit_path,fitted)
        result['resources']['objective_calls']+=fitted['calls'];result['resources']['failed_objective_calls']+=fitted['failures']
        result['resources']['native_constructions']+=1
        summary={'index':index,'bundle_id':config['bundle']['id'],'selected':config['bundle']['selected'],'formula':config['formula'],
            'best':fitted['best'],'calls':fitted['calls'],'call_limit':config['call_limit']}
        result['candidate_scores'].append(summary)
        if fitted['best'] is not None and (best is None or fitted['best']['objective']<best['score']):
            best={'index':index,'score':fitted['best']['objective'],'fitted':fitted,'bundle':config['bundle']}
    if best is None:result.update(status='failed',reason='No finite development fit')
    else:
        target_dir=folder/'target';target_dir.mkdir(parents=True,exist_ok=True)
        request=make_request(target,best['bundle'],target_dir);request['partition_method']=best['fitted']['config']['formula']
        save(target_dir/'request.json',request)
        physiology=engine._physiology(request,target_dir,ROOT,guard)
        times=engine.prediction_grid(request);keys=engine.PARAMETERS['flow_exchange']
        pred=engine._simulate(physiology,request,times,'flow_exchange',keys,best['fitted']['best']['log_parameters'])
        mw=next(float(r['value']) for r in request['evidence'] if r['parameter']=='molecular_weight')
        curve={'time_min':(times*60).tolist(),'concentration':(np.asarray(pred['concentration_mg_l'])*1000/mw).tolist(),
            'time_unit':'min','unit':'umol/l','observation':pred['observation']}
        endpoints=summarize_extended_prediction(curve,request)
        save(target_dir/'prediction_locked.json',curve);save(target_dir/'endpoints.json',endpoints)
        result.update(status='predicted',selected_candidate=best['index'],selected_bundle=best['bundle']['selected'],
            formula=request['partition_method'],parameters=best['fitted']['best'],request=ref(target_dir/'request.json'),
            prediction=ref(target_dir/'prediction_locked.json'),endpoints=ref(target_dir/'endpoints.json'),
            mass_balance=pred['mass_balance'],target_observations_used=False,training_source=job['training_source'])
        result['resources']['native_constructions']+=1
    result['resources']['wall_seconds']=time.monotonic()-start
    output_pins={};pins_in({k:v for k,v in result.items() if k not in ('candidate_scores',)},output_pins)
    result['output_pins']=output_pins;save(result_path,result);return result

def run(workers=4):
    protocol=read(RUN/'protocol.json');state=read(RUN/'state.json');checked(state['protocol'])
    if state['status']=='complete':return status()
    guard=Guard(protocol);guard(force=True)
    state.update(status='running',pid=os.getpid());save(RUN/'state.json',state)
    # Slots are independent. Workers never inspect other project processes/services.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(execute,protocol,j,key,guard):key for key,j in protocol['jobs'].items() if key not in state['units']}
        try:
            for future in as_completed(futures):
                key=futures[future];state['units'][key]=future.result();save(RUN/'state.json',state)
                print(json.dumps({'completed':len(state['units']),'total':len(protocol['jobs']),'unit':key},ensure_ascii=False),flush=True)
        except Exception as exc:
            save(RUN/'PAUSE',{'reason':str(exc),'at':time.time()});state.update(status='paused',reason=str(exc));save(RUN/'state.json',state);raise
    guard(force=True)
    state.update(status='predictions_locked',locked_at=time.time(),all_predictions_locked_before_scoring=True)
    state['output_pins']={}
    for unit in state['units'].values():pins_in(unit,state['output_pins'])
    save(RUN/'state.json',state)
    return report()

def report():
    from report_blueprint_validation import Store,_case_reference,_score_prediction,paired_ratio
    protocol=read(RUN/'protocol.json');state=read(RUN/'state.json');checked(state['protocol'])
    if state['status'] not in {'predictions_locked','complete'} or not state.get('all_predictions_locked_before_scoring'):
        raise ValueError('Lock all prediction attempts before accessing outcomes')
    if set(state['units'])!=set(protocol['jobs']) or any(u.get('status') not in {'predicted','failed','blocked'} for u in state['units'].values()):
        raise ValueError('All registered prediction attempts must have terminal records before scoring')
    verify(protocol['source_pins']);verify(state['output_pins'])
    store=Store(ROOT);scored=[];rows=[]
    combined={**state['units'],**protocol['controls']}
    for case in protocol['cases']:
        clinical,curve,endpoints=_case_reference(case,store)
        for key,u in combined.items():
            if not key.startswith(case['case_id']+'__'):continue
            row={'unit_id':key,'drug':case['drug'],'case_id':case['case_id'],'sampling':u['sampling'],'method':u['method'],
                'status':u['status'],'selected_bundle':u.get('selected_bundle'),'resources':u.get('resources',{})}
            if u['status']=='predicted':
                metrics,_,_=_score_prediction(case,clinical,curve,endpoints,u,store);row.update(metrics)
            else:row.update(reason=u.get('reason',''),curve_metrics={'gmfe':None},endpoints={})
            scored.append(row)
            rows.append({k:row[k] for k in ('drug','sampling','method','status')}|{'endpoint':'concentration_time_curve','prediction':None,'reference':None,'unit':curve['unit'],'error':row['curve_metrics'].get('gmfe'),'metric':'GMFE','weak_tail':None})
            for ep,reference in endpoints.items():
                e=row['endpoints'].get(ep,{})
                rows.append({k:row[k] for k in ('drug','sampling','method','status')}|{'endpoint':ep,'prediction':e.get('value'),'reference':reference['value'],'unit':e.get('unit',reference.get('unit')),'error':e.get('error'),'metric':e.get('error_metric'),'weak_tail':e.get('weak_tail')})
    clusters={c['case_id']:c['cluster_id'] for c in protocol['cases']};comparisons={}
    for sampling in SAMPLINGS:
        by={m:{r['case_id']:r['curve_metrics'].get('gmfe') for r in scored if r['method']==m and r['sampling']==sampling} for m in ('joint','unconstrained_search','frozen_paper_pbpk','two_compartment')}
        comparisons[sampling]={m:paired_ratio(by['joint'],v,clusters) for m,v in by.items() if m!='joint'}
    result={'version':VERSION,'status':'complete','locked_at':state['locked_at'],'scored_at':time.time(),
        'protocol':ref(RUN/'protocol.json'),'cases':len(protocol['cases']),'records':scored,'comparisons':comparisons,
        'selection':protocol['selections'],'limitations':['Previously viewed small development cohort; no general superiority claim.',
            'Every method has the same single-record applicability gates. Known liver impairment exclusions are not a new joint-selection benefit.',
            'Frozen PBPK and two-compartment use their original larger optimization budgets; only joint and unconstrained_search are budget-matched.',
            'Drug-level summaries do not establish individual CL/fu correlation; compatible sources can still give inaccurate clinical predictions.']}
    save(RUN/'results.json',result)
    with (RUN/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 研究条件约束的参数成组选择：小规模验证','',f'实际临床药物：{len(protocol["cases"])}。所有预测先锁定，再读取目标观测；未修改论文。','',
        '| Drug | Sampling | Method | Curve GMFE | Status |','|---|---|---|---:|---|']
    for r in scored:
        g=r['curve_metrics'].get('gmfe');lines.append(f'| {r["drug"]} | {r["sampling"]} | {r["method"]} | {g:.4f}'+' | '+r['status']+' |' if g else f'| {r["drug"]} | {r["sampling"]} | {r["method"]} | — | {r["status"]} |')
    lines+=['','全部11类输出与有参考项的误差见[唯一结果表](results.csv)，候选、失败及实际调用量见[机器结果](results.json)。',
        '', '相同单条适用门限用于全部方法；明确肝损害证据被共同排除，不算新方法收益。新方法和无约束搜索每药/采样允许4096次目标函数调用、16起点，包含有限差分与失败；实际提前收敛调用数不同。原论文PBPK/两室仅作准确率参照，预算更大。',
        '', '同臂群体CL/fu不代表个体参数相关性。这些病例已参与开发，样本不足以证明普遍优势或临床安全。缺临床参考与不可估尾部保留空值。',
        '', '命令：`runtime/app-env/bin/python -B scripts/run_condition_matched_selection.py status|report|cleanup`。']
    (RUN/'README.md').write_text('\n'.join(lines)+'\n')
    state.update(status='complete',reported_at=time.time());save(RUN/'state.json',state)
    return {'status':'complete','drugs':len(protocol['cases']),'units':len(scored),'table_rows':len(rows),'comparisons':comparisons}

def status():
    if not (RUN/'state.json').exists():return {'status':'not_prepared'}
    s=read(RUN/'state.json');p=read(RUN/'protocol.json')
    return {'status':s['status'],'completed':len(s['units']),'total':len(p['jobs']),'reason':s.get('reason')}

def cleanup():
    import shutil
    state=read(RUN/'state.json')
    if state['status']!='complete':raise ValueError('Only completed locked results can be cleaned')
    removed=0;size=0
    for key,u in state['units'].items():
        if u['status']!='predicted':continue
        for row in u['candidate_scores']:
            if row['index']==u['selected_candidate']:continue
            # Keep all candidate fit scores/requests; remove only successful native work.
            cell=RUN/'work'/key/f'candidate_{row["index"]:02d}'
            for name in ('base','physiology'):
                p=cell/name
                if p.exists():
                    size+=sum(f.stat().st_size for f in p.rglob('*') if f.is_file());shutil.rmtree(p);removed+=1
    return {'removed_successful_unselected_native_directories':removed,'bytes':size,'candidate_scores_retained':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','status','report','cleanup'])
    parser.add_argument('--background',action='store_true');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if not 1<=args.workers<=14:parser.error('workers must be 1..14')
    if args.command=='run' and args.background:
        with (RUN/'run.log').open('ab') as log:
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'}
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'run','--workers',str(args.workers)],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
        (RUN/'run.pid').write_text(str(proc.pid)+'\n');print(json.dumps({'pid':proc.pid}));return
    if args.command=='status':print(json.dumps(status(),ensure_ascii=False,indent=2));return
    with (RUN/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run(args.workers) if args.command=='run' else globals()[args.command]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
