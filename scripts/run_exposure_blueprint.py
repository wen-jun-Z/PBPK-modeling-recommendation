#!/usr/bin/env python3
"""Ten-drug exposure-constrained PBPK calibration; targets opened after locking."""
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
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pbpk_kb.blueprints.source_workflow import save,sha256
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
from pbpk_kb.blueprints.development_curve_blueprint import _training_document
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.blueprint_policy import transport_calibration as engine
from pbpk_kb.blueprint_policy.perfusion_export import export_physiology
from pbpk_kb.blueprint_policy import exposure_profile
from run_condition_matched_selection import ref,checked,read,pins_in,verify,fit_cell,allocated
import numpy as np
from scipy.optimize import least_squares

RUN=ROOT/'experiments/blueprint_design/exposure_profile_20261008'
PAPER=ROOT/'experiments/blueprint_design/vascular_tissue_ten_drug_20260930'
VERSION='exposure-constrained-blueprint-1'
METHODS=('fixed_prior','direct_relaxed','exposure_constrained')
SAMPLINGS=('full','five')
FORMULAS=('PKSimStandard','RodgersRowland')
BUDGET=16384
STARTS=16

def normalized(request):
    return EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json',exclude_none=True)

def cached_export(folder,request):
    base=read(folder/'base/completion.json')
    if normalized(base['request'])!=normalized(request):raise ValueError('Native base differs from frozen request')
    return export_physiology(execution_manifest_path=ROOT/base['execution_manifest']['path'],
        source_manifest_sha256=base['execution_manifest']['sha256'],output_dir=folder/'physiology',project_root=ROOT)

class Guard:
    def __init__(self,protocol):self.protocol=protocol;self.last=0.;self.lock=threading.Lock()
    def __call__(self,force=False):
        with self.lock:
            if (RUN/'PAUSE').exists():raise RuntimeError('Project batch paused')
            if not force and time.monotonic()-self.last<1200:return
            verify(self.protocol['source_pins'])
            n=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
            if n>10*1024**3:raise RuntimeError('Batch exceeds 10 GiB')
            self.last=time.monotonic();save(RUN/'monitor.json',{'time':time.time(),'bytes':n,'pid':os.getpid(),'source_integrity':True})

class BudgetExceeded(Exception):pass

class NativeBuildFailed(ValueError):
    """Known construction failure; source-integrity exceptions still propagate."""

def relaxed_fit(physiology,request,training,budget=BUDGET,starts=STARTS,before_trial=None):
    """Ablation: direct 3-axis fitting with exactly the exposure method's bounds."""
    mw=next(float(e['value']) for e in request['evidence'] if e['parameter']=='molecular_weight')
    times=np.asarray([r['time_min']/60 for r in training['observations']],float)
    values=np.asarray([r['concentration_umol_per_l']*mw/1000 for r in training['observations']],float)
    if times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0) or np.any(values<=0) or not np.isfinite(values).all():raise ValueError('Invalid development curve')
    cl=engine.clearance_inputs(request);renal=cl['renal_clearance_l_h'] or 0.;nonrenal=cl['total_clearance_l_h']-renal
    bounds=(1e-4,10*float(physiology['circulation']['cardiac_flow_l_h']))
    lower=np.log([bounds[0]/nonrenal,.1,.01]);upper=np.log([bounds[1]/nonrenal,10,100])
    outer=exposure_profile.initial_starts(starts)
    cl_initial=np.r_[np.clip(0.,lower[0],upper[0]),np.random.default_rng(20260930).uniform(lower[0],upper[0],starts-1)]
    initial=np.column_stack([cl_initial,outer])
    calls=0;failures=0;best=None;trace=[];begin=time.monotonic()
    for index,(x0,cap) in enumerate(zip(initial,allocated(budget,starts))):
        used=0;local=None
        def residual(x):
            nonlocal used,calls,failures,best,local
            if used>=cap:raise BudgetExceeded()
            if before_trial:before_trial()
            used+=1;calls+=1
            try:
                prediction=engine._simulate(physiology,request,times,'flow_exchange',engine.PARAMETERS['flow_exchange'],x)
                p=np.asarray(prediction['concentration_mg_l'],float)
                if p.shape!=values.shape or not np.isfinite(p).all() or np.any(p<=0):raise ValueError('Invalid prediction')
                data=(np.log(p)-np.log(values))/math.sqrt(len(values))
                residuals=np.r_[data,.1*x[1]/math.log(10),.1*x[2]/math.log(100)]
                objective=float(residuals@residuals)
                row={'objective':objective,'log_parameters':np.asarray(x).tolist(),'start_index':index,
                    'development_ln_rmse':float(np.linalg.norm(data)),
                    'profiled_nonrenal_clearance_l_h':nonrenal*math.exp(x[0]),
                    'observed_window_auc_mg_h_l':float(np.trapezoid(values,times)),
                    'predicted_window_auc_mg_h_l':float(np.trapezoid(p,times))}
                if local is None or objective<local['objective']:local=row
                if best is None or objective<best['objective']:best=row
                return residuals
            except (ValueError,ArithmeticError,np.linalg.LinAlgError):
                failures+=1;return np.full(len(values)+2,1e6)
        status='converged'
        try:
            fitted=least_squares(residual,x0,bounds=(lower,upper),max_nfev=400,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            if not fitted.success:status='optimizer_limit'
        except BudgetExceeded:status='call_budget_exhausted'
        trace.append({'index':index,'calls':used,'call_limit':cap,'status':status,'best':local})
    return {'best':best,'calls':calls,'failures':failures,'call_limit':budget,'starts':trace,
        'wall_seconds':time.monotonic()-begin,'nonrenal_clearance_bounds_l_h':list(bounds),'no_warm_start':True}

def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():raise ValueError('Prepared already; use run/status')
    paper=read(PAPER/'protocol.json');old=read(PAPER/'state.json');cases=deepcopy(paper['cases'])
    if len(cases)!=10:raise ValueError('Retain all original ten cases')
    caches={};pins=engine._code_pins(ROOT);jobs={};controls={}
    for case in cases:
        cid=case['case_id'];pins_in(case,pins)
        for formula in FORMULAS:
            request=checked(case['requests']['evidence']['development']);request['partition_method']=formula
            folder=ROOT/old['fits'][f'{cid}__full__flow_exchange__{formula}']['mapping']['path'];folder=folder.parent
            export=cached_export(folder,request)
            key=f'{cid}__development__{formula}';caches[key]=export['manifest'];pins_in(export['manifest'],pins)
            for d in (export.get('source_pins',{}),export.get('artifact_pins',{})):verify(d);pins.update(d)
            for sampling in SAMPLINGS:
                unit=old['computations'][f'{cid}__{sampling}__flow_exchange']
                if unit['selected_formula']!=formula:continue
                target=checked(case['requests']['evidence']['target']);target['partition_method']=formula
                target_folder=(ROOT/unit['request']['path']).parent
                exported=cached_export(target_folder,target)
                caches[f'{cid}__target__{formula}']=exported['manifest'];pins_in(exported['manifest'],pins)
                for d in (exported.get('source_pins',{}),exported.get('artifact_pins',{})):verify(d);pins.update(d)
        for sampling in SAMPLINGS:
            for method in METHODS:
                uid=f'{cid}__{sampling}__{method}'
                jobs[uid]={'case_id':cid,'method':method,'sampling':sampling,'training_source':case['sampling_sources'][sampling]}
            for source_method,label in [('two_compartment','two_compartment_reference'),('flow_exchange','paper_pbpk_reference')]:
                uid=f'{cid}__{sampling}__{label}';u=deepcopy(old['computations'][f'{cid}__{sampling}__{source_method}'])
                u.update(case_id=cid,sampling=sampling,method=label,reused=True);controls[uid]=u;pins_in(u,pins)
    for path in (Path(__file__),ROOT/'src/pbpk_kb/blueprint_policy/exposure_profile.py',ROOT/'scripts/run_condition_matched_selection.py',ROOT/'scripts/report_blueprint_validation.py',PAPER/'protocol.json',PAPER/'state.json'):
        pins[str(path.relative_to(ROOT))]=sha256(path)
    protocol={'version':VERSION,'created_at':time.time(),'cases':cases,'jobs':jobs,'controls':controls,'physiology_cache':caches,'source_pins':pins,
        'methods':list(METHODS),'sampling':list(SAMPLINGS),'formulas':list(FORMULAS),'objective_calls_per_formula':BUDGET,'starts_per_formula':STARTS,
        'equal_budget':'16384 actual simulations per formula, including root search, differences, failed numerical evaluations; 32768 maximum per method/drug/sampling.',
        'exposure_constraint':'Linear trapezoidal integral over exactly visible development points; no inserted zero and no terminal extrapolation. This is not observed total CL or AUC infinity.',
        'parameter_count':'Three data-estimated parameters in every model; exposure-constrained has two outer optimizer coordinates and one conditionally estimated CL.',
        'selection_basis':'Development objective only within each method; target outcomes never used for fitting or formula selection.',
        'native_reuse':'Only uncalibrated native physiology/partition exports are reused. No old fitted parameters or full-curve initialization enter five-point fitting.',
        'scope':'All ten previously inspected development drugs, not blind validation; no patient-specific safety claim.',
        'nonrenal_clearance_l_h_bounds':[1e-4,'10 * cardiac flow in L/h; numerical bracket, not validated organ clearance'],
        'original_references_have_different_budget':True,'monitor_seconds':1200,'directory_limit_gib':10,'development_only_preflight':{'drug':'Ertapenem','target_read':False,'new_native':0,'actual_simulations':21999,'reason':'Increased all methods to 16384 calls/formula before target scoring after development-only convergence check'}}
    verify(pins);save(RUN/'protocol.json',protocol);save(RUN/'state.json',{'status':'prepared','version':VERSION,'protocol':ref(RUN/'protocol.json'),'units':{},'started_at':None})
    return {'drugs':10,'new_units':len(jobs),'reused_reference_units':len(controls),'cached_native_settings':len(caches)}

_cache_lock=threading.Lock();_cache_locks={}
def physiology(protocol,case,phase,formula,request,guard):
    key=f'{case["case_id"]}__{phase}__{formula}'
    if key in protocol['physiology_cache']:return checked(protocol['physiology_cache'][key]),{'cached_native':True,'new_native':0}
    with _cache_lock:lock=_cache_locks.setdefault(key,threading.Lock())
    with lock:
        folder=RUN/'native_cache'/key
        existing=(folder/'physiology/perfusion_physiology.json').exists()
        try:
            result=engine._physiology(request,folder,ROOT,guard)
        except ValueError as exc:
            if str(exc) not in {'Incomplete native construction evidence','Bounded native export failed; inspect retained process log'}:raise
            raise NativeBuildFailed(str(exc)) from exc
        return result,{'cached_native':existing,'new_native':0 if existing else 1}

def execute(protocol,key,job,guard):
    case=next(c for c in protocol['cases'] if c['case_id']==job['case_id']);folder=RUN/'work'/key;folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'result.json';protocol_sha=sha256(RUN/'protocol.json')
    if result_path.exists():
        previous=read(result_path)
        if previous['protocol_sha256']!=protocol_sha:raise ValueError('Changed completed protocol')
        verify(previous['output_pins']);return previous
    dev=normalized(checked(case['requests']['evidence']['development']));target=normalized(checked(case['requests']['evidence']['target']))
    observed=_training_document(job['training_source'],dev,ROOT)
    if job['sampling']=='five' and len(observed['observations'])!=5:raise ValueError('Five-point isolation failure')
    if observed['human']!=dev['human'] or observed['dosing']!=dev['dosing']:raise ValueError('Development condition mismatch')
    unit={**job,'status':'fitting','protocol_sha256':protocol_sha,'formula_scores':[],'resources':{'objective_calls':0,'failed_calls':0,'new_native':0}}
    began=time.monotonic();best=None
    for formula in FORMULAS:
        guard();req=deepcopy(dev);req['partition_method']=formula
        exported,cost=physiology(protocol,case,'development',formula,req,guard);unit['resources']['new_native']+=cost['new_native']
        p=folder/(formula+'.json')
        if p.exists():
            fitted=read(p)
            if fitted.get('protocol_sha256')!=protocol_sha or fitted.get('request')!=req or fitted.get('training_source')!=job['training_source']:raise ValueError('Changed resumed fit input')
            if fitted['calls']>BUDGET or len(fitted['starts'])!=STARTS:raise ValueError('Invalid resumed budget')
        else:
            if job['method']=='fixed_prior':fitted=fit_cell(exported,req,observed,BUDGET,STARTS,guard)
            elif job['method']=='direct_relaxed':fitted=relaxed_fit(exported,req,observed,before_trial=guard)
            else:fitted=exposure_profile.fit(exported,req,observed,budget=BUDGET,starts=STARTS,before_trial=guard)
            fitted.update(protocol_sha256=protocol_sha,request=req,training_source=job['training_source']);save(p,fitted)
        unit['resources']['objective_calls']+=fitted['calls'];unit['resources']['failed_calls']+=fitted['failures']
        unit['formula_scores'].append({'formula':formula,'best':fitted['best'],'fit':ref(p),'calls':fitted['calls'],'failures':fitted['failures']})
        if fitted['best'] is not None and (best is None or fitted['best']['objective']<best['fit']['best']['objective']):best={'formula':formula,'fit':fitted}
    if best is None:unit.update(status='failed',reason='No eligible development solution')
    else:
        formula=best['formula'];target['partition_method']=formula;save(folder/'request.json',target)
        # Choice is persisted before any target prediction or observation access.
        save(folder/'selection_locked.json',{'formula':formula,'parameters':best['fit']['best'],'basis':'development_only','locked_at':time.time()})
        try:
            exported,cost=physiology(protocol,case,'target',formula,target,guard)
        except NativeBuildFailed as exc:
            unit.update(status='failed',reason=str(exc),failure_stage='native_target_construction')
            exported=None
        if exported is not None:
            unit['resources']['new_native']+=cost['new_native']
            times=engine.prediction_grid(target);params=best['fit']['best']['log_parameters']
        try:
            if exported is None:raise NativeBuildFailed(unit['reason'])
            prediction=engine._simulate(exported,target,times,'flow_exchange',engine.PARAMETERS['flow_exchange'],params)
            mw=next(float(e['value']) for e in target['evidence'] if e['parameter']=='molecular_weight')
            curve={'time_min':(times*60).tolist(),'concentration':(np.asarray(prediction['concentration_mg_l'])*1000/mw).tolist(),'time_unit':'min','unit':'umol/l','observation':prediction['observation']}
            endpoints=summarize_extended_prediction(curve,target);save(folder/'prediction_locked.json',curve);save(folder/'endpoints.json',endpoints)
            unit.update(status='predicted',request=ref(folder/'request.json'),prediction=ref(folder/'prediction_locked.json'),endpoints=ref(folder/'endpoints.json'),
                formula=formula,parameters=best['fit']['best'],mass_balance=prediction['mass_balance'],effective_total_clearance_l_h=prediction['effective_total_clearance_l_h'],
                target_observations_used=False,selection=ref(folder/'selection_locked.json'))
        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:unit.update(status='failed',reason=str(exc))
    unit['resources']['wall_seconds']=time.monotonic()-began;unit['output_pins']={};pins_in({k:v for k,v in unit.items() if k!='output_pins'},unit['output_pins']);save(result_path,unit);return unit

def run(workers):
    protocol=read(RUN/'protocol.json');state=read(RUN/'state.json');checked(state['protocol'])
    if state['status']=='complete':return status()
    guard=Guard(protocol);guard(force=True);state.update(status='running',pid=os.getpid(),started_at=state.get('started_at') or time.time());save(RUN/'state.json',state)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(execute,protocol,key,job,guard):key for key,job in protocol['jobs'].items() if key not in state['units']}
        try:
            for task in as_completed(tasks):
                key=tasks[task];state['units'][key]=task.result();save(RUN/'state.json',state);print(json.dumps({'done':len(state['units']),'total':len(protocol['jobs']),'unit':key}),flush=True)
        except Exception as exc:
            save(RUN/'PAUSE',{'reason':str(exc)});state.update(status='paused',reason=str(exc));save(RUN/'state.json',state);raise
    guard(force=True);state.update(status='predictions_locked',locked_at=time.time(),all_predictions_locked_before_scoring=True,output_pins={})
    for u in state['units'].values():state['output_pins'].update(u['output_pins'])
    save(RUN/'state.json',state);return report()

def report():
    from report_blueprint_validation import Store,_case_reference,_score_prediction,paired_ratio,clustered_geomean
    protocol=read(RUN/'protocol.json');state=read(RUN/'state.json');checked(state['protocol'])
    if state['status'] not in {'predictions_locked','complete'} or not state.get('all_predictions_locked_before_scoring'):raise ValueError('All predictions must lock before scoring')
    if set(state['units'])!=set(protocol['jobs']) or any(u['status'] not in {'predicted','failed'} for u in state['units'].values()):raise ValueError('Incomplete terminal attempt matrix')
    verify(protocol['source_pins']);verify(state['output_pins']);store=Store(ROOT);records=[];rows=[]
    all_units={**state['units'],**protocol['controls']}
    for case in protocol['cases']:
        clinical,curve,endpoints=_case_reference(case,store)
        for sampling in SAMPLINGS:
            for method in METHODS+('paper_pbpk_reference','two_compartment_reference'):
                uid=f'{case["case_id"]}__{sampling}__{method}';u=all_units[uid]
                rec={'case_id':case['case_id'],'drug':case['drug'],'sampling':sampling,'method':method,'status':u['status'],'resources':u.get('resources',{}),'parameters':u.get('parameters'),'formula':u.get('formula')}
                if u['status']=='predicted':metrics,_,_=_score_prediction(case,clinical,curve,endpoints,u,store);rec.update(metrics)
                else:rec.update(reason=u.get('reason',''),curve_metrics={'gmfe':None},endpoints={ep:{'value':None,'reference':obs['value'],'unit':obs.get('unit'),'error':None} for ep,obs in endpoints.items()})
                records.append(rec);head={k:rec[k] for k in ('drug','sampling','method','status')}
                rows.append(head|{'endpoint':'concentration_time_curve','prediction':None,'reference':None,'unit':curve['unit'],'error':rec['curve_metrics'].get('gmfe'),'metric':'GMFE','weak_tail':None})
                for ep,obs in endpoints.items():
                    pred=rec['endpoints'].get(ep,{});rows.append(head|{'endpoint':ep,'prediction':pred.get('value'),'reference':obs['value'],'unit':pred.get('unit',obs.get('unit')),'error':pred.get('error'),'metric':pred.get('error_metric'),'weak_tail':pred.get('weak_tail')})
    clusters={c['case_id']:c['cluster_id'] for c in protocol['cases']};summary={};comparison={}
    for sampling in SAMPLINGS:
        values={m:{r['case_id']:r['curve_metrics'].get('gmfe') for r in records if r['sampling']==sampling and r['method']==m} for m in METHODS+('paper_pbpk_reference','two_compartment_reference')}
        summary[sampling]={}
        for m,v in values.items():
            valid={k:x for k,x in v.items() if x is not None and math.isfinite(x) and x>0}
            summary[sampling][m]=clustered_geomean(valid,{k:clusters[k] for k in valid})|{'attempted':len(v),'failed_or_unscorable':len(v)-len(valid),'within_twofold':sum(x<=2 for x in valid.values())}
        comparison[sampling]={m:paired_ratio(values['exposure_constrained'],v,clusters) for m,v in values.items() if m!='exposure_constrained'}
    endpoint_summary={};resources={};sparse_change={};sensitivity={}
    for method in METHODS+('paper_pbpk_reference','two_compartment_reference'):
        selected=[r for r in records if r['method']==method]
        sparse_change[method]=paired_ratio(
            {r['case_id']:r['curve_metrics'].get('gmfe') for r in selected if r['sampling']=='five'},
            {r['case_id']:r['curve_metrics'].get('gmfe') for r in selected if r['sampling']=='full'},clusters)
        resources[method]={name:sum(r['resources'].get(name,0) for r in selected) for name in ('objective_calls','failed_calls','new_native','wall_seconds')}
        resources[method]['reused']=method not in METHODS
        if method in METHODS:
            starts=[s for u in state['units'].values() if u['method']==method for formula in u['formula_scores'] for s in checked(formula['fit'])['starts']]
            resources[method].update(starts=len(starts),budget_exhausted_starts=sum(s['status']=='call_budget_exhausted' for s in starts),converged_starts=sum(s['status']=='converged' for s in starts))
    for sampling in SAMPLINGS:
        endpoint_summary[sampling]={};sensitivity[sampling]={}
        subset={c['case_id']:clusters[c['case_id']] for c in protocol['cases'] if c['drug'].lower() not in {'nemonoxacin','dexmedetomidine','midazolam'}}
        for method in METHODS+('paper_pbpk_reference','two_compartment_reference'):
            selected=[r for r in records if r['method']==method and r['sampling']==sampling];stats={}
            for endpoint in selected[0]['endpoints'] if selected else []:
                entries=[r.get('endpoints',{}).get(endpoint,{}) for r in selected]
                errors=[p['error'] for p in entries if p.get('error') is not None]
                stat={'attempted':len(selected),'reference_available':sum(p.get('reference') is not None for p in entries),'predicted':sum(p.get('value') is not None for p in entries),'scored':len(errors),'weak_scored':sum(p.get('weak_tail',False) and p.get('error') is not None for p in entries)}
                if errors:
                    if endpoint=='tmax':stat['mean_absolute_error_h']=float(np.mean(errors))
                    else:stat.update(gmfe=math.exp(float(np.mean(np.log(errors)))),within_twofold=sum(v<=2 for v in errors))
                stats[endpoint]=stat
            endpoint_summary[sampling][method]=stats
            values={r['case_id']:r['curve_metrics'].get('gmfe') for r in selected if r['case_id'] in subset}
            ours={r['case_id']:r['curve_metrics'].get('gmfe') for r in records if r['sampling']==sampling and r['method']=='exposure_constrained' and r['case_id'] in subset}
            sensitivity[sampling][method]=paired_ratio(ours,values,subset)
    document={'version':VERSION,'status':'complete','protocol':ref(RUN/'protocol.json'),'locked_at':state['locked_at'],'scored_at':time.time(),'records':records,'summary':summary,'comparisons':comparison,
        'endpoint_summary':endpoint_summary,'resources':resources,'five_over_full_gmfe':sparse_change,'source_sensitivity_excluding_nemonoxacin_dexmedetomidine_midazolam':sensitivity,
        'limitations':['Previously viewed development cohort, not blind validation.','Window trapezoid matching is not observed systemic CL or AUC infinity.','Exposure-constrained estimates three parameters from data, with two outer optimizer coordinates.','No guarantee of improvement or clinical risk reliability; failed and weak-tail endpoints remain visible.']}
    save(RUN/'results.json',document)
    with (RUN/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 暴露约束分步校准：十药开发验证','','只用开发研究估计参数，全部目标预测锁定后评分。未替换论文。','', '| Method | Full GMFE | Five-point GMFE | Full coverage |','|---|---:|---:|---:|']
    for m in METHODS+('paper_pbpk_reference','two_compartment_reference'):
        a,b=summary['full'][m],summary['five'][m]
        av='—' if a['estimate'] is None else f"{a['estimate']:.4f}"
        bv='—' if b['estimate'] is None else f"{b['estimate']:.4f}"
        lines.append(f'| {m} | {av} | {bv} | {a["n_cases"]}/10 |')
    lines+=['','[逐药11类终点、真实参考与误差](results.csv) · [配对区间、失败及计算成本](results.json)。',
        '', '新方法按可见开发采样窗的梯形面积约束非肾CL，再优化K/PS。没有把局部面积当AUC∞，没有把得到的CL当独立观测；统计上仍估计三个参数。净肾构建输入固定。',
        '', 'fixed_prior与原法范围/惩罚相同；direct_relaxed只移除CL惩罚并放宽至共同数值区间；exposure_constrained加入面积等式。新三方法每公式16384次实际求解、16起点，内部求根、失败、有限差分均计数。旧论文与两室仅为冻结参照，预算不同。',
        '', '全部十药已经参与开发，不能据此声称新药无数据预测可靠或具有临床安全性。全部终点与失败保留，未按目标结果删药、改规则或逐药拼接最好方法。',
        '', '`runtime/app-env/bin/python -B scripts/run_exposure_blueprint.py status` 查看状态，`report`重算结果。']
    (RUN/'README.md').write_text('\n'.join(lines)+'\n');state.update(status='complete',completed_at=time.time());save(RUN/'state.json',state)
    return {'status':'complete','new_units':len(state['units']),'rows':len(rows),'summary':summary}

def status():
    if not (RUN/'state.json').exists():return {'status':'not_prepared'}
    state=read(RUN/'state.json');protocol=read(RUN/'protocol.json')
    return {'status':state['status'],'done':len(state['units']),'total':len(protocol['jobs']),'formula_fits':sum(len(list((RUN/'work').glob('*/'+f+'.json'))) for f in FORMULAS),'reason':state.get('reason')}

def cleanup():
    if read(RUN/'state.json')['status']!='complete':raise ValueError('Complete first')
    return {'removed':0,'per_evaluation_files_created':False,'compact_formula_fits_and_shared_native_cache_retained':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','status','report','cleanup']);p.add_argument('--background',action='store_true');p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if not 1<=args.workers<=14:p.error('workers 1..14')
    RUN.mkdir(parents=True,exist_ok=True)
    if args.command=='run' and args.background:
        with (RUN/'run.log').open('ab') as log:
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'}
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'run','--workers',str(args.workers)],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
        (RUN/'run.pid').write_text(str(process.pid)+'\n');print(json.dumps({'pid':process.pid}));return
    if args.command=='status':print(json.dumps(status(),ensure_ascii=False,indent=2));return
    with (RUN/'.runner.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);result=run(args.workers) if args.command=='run' else globals()[args.command]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
