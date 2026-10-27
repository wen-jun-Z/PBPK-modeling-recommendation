#!/usr/bin/env python3
"""Thirty-drug development repair: aligned labels, stable choices, feasible fitting."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import math
import os
from pathlib import Path
import time
import numpy as np
import run_sampling_expansion as previous
from blueprint_sampling_stability_features import rebuild_original_features
from pbpk_kb.blueprint_policy import stable_sampling_design as selector
from pbpk_kb.blueprint_policy import visible_partition_feasibility as repair
from report_blueprint_validation import clustered_geomean

ROOT=previous.ROOT
RUN=ROOT/'experiments/blueprint_design/sampling_stability_20261009'
OLD=previous.RUN
read,save,ref,checked,verify=previous.read,previous.save,previous.ref,previous.checked,previous.verify
gap,base,prior,old,data=previous.gap,previous.base,previous.prior,previous.old,previous.data
GOALS=previous.GOALS
METHODS=('previous_learned','aligned_only','stable','time','stable_repaired','time_repaired','no_addition','no_addition_repaired')
TOL=1e-10


class Guard:
    def __init__(self,p):self.p,self.last=p,0.
    def __call__(self,force=False):
        if (RUN/'PAUSE').exists():raise RuntimeError('Paused; inspect retained diagnosis')
        if force or time.monotonic()-self.last>=1200:
            verify(self.p['source_pins'])
            size=sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file() and not f.is_symlink())
            if size>10*1024**3:raise RuntimeError('Batch exceeds10GiB')
            self.last=time.monotonic()


def label_rows(cases,original_ids,features,records,goal):
    """A hard original-ten whitelist precedes reading any numeric target label."""
    ids=set(original_ids)
    if len(ids)!=10:raise ValueError('Exactly original ten may train')
    eligible=[r for r in records if r['case_id'] in ids]
    raw={(r['design_key'],r['candidate']):r for r in eligible}
    if len(raw)!=len(eligible):raise ValueError('Duplicate original training record')
    case_by={c['case_id']:c for c in cases};out=dict(X=[],y=[],design_ids=[],drug_ids=[],cluster_ids=[])
    for key,f in sorted(features.items()):
        cid=f['case_id']
        if cid not in ids:continue
        detail=f['goals'][goal]
        if detail.get('status')!='selected':continue
        baseline=raw[key,'baseline']['endpoints'][goal]['error']
        if baseline is None:continue
        labels=[raw[key,key+f'__point_{i:02d}']['endpoints'][goal]['error'] for i in f['candidate_ids']]
        if len(detail['X'])!=len(labels):raise ValueError('Misaligned training menu')
        if any(v is None for v in labels):continue
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in [baseline,*labels]):
            raise ValueError('Training errors must be finite and positive')
        for x,value in zip(detail['X'],labels):
            out['X'].append(x);out['y'].append(math.log(baseline/value))
            out['design_ids'].append(key);out['drug_ids'].append(cid);out['cluster_ids'].append(case_by[cid]['cluster_id'])
    return out


def prepare():
    if (RUN/'protocol.json').exists():return status()
    state=read(OLD/'state.json');p=checked(state['protocol'])
    if state['status']!='complete':raise ValueError('Prior30-drug batch must be complete')
    verify(p['source_pins']); original_state=read(previous.OLD/'state.json')
    features_path=RUN/'features.json'
    if features_path.exists():
        doc=read(features_path)
        if doc['source_protocol']!=state['protocol']:raise ValueError('Changed feature source')
        features=doc['designs']
    else:
        features=rebuild_original_features(p,original_state)
        for key,entry in state['designs'].items():
            d=checked(entry['features']);cid=p['designs'][key]['case_id'];c=gap.get_case(p,cid)
            features[key]=d|dict(case_id=cid,cluster_id=c['cluster_id'],times=d['candidate_times_h'])
        save(features_path,dict(source_protocol=state['protocol'],designs=features))
    records=checked(state['results'])['records']
    original_ids=set(p['original_case_ids']);bundles={};choices={};jobs={};designs={}
    original_choices=read(previous.TRAIN/'decisions.json')['choices']
    for goal in ('cmax','terminal_half_life'):
        rows=label_rows(p['cases'],original_ids,features,records,goal)
        if set(rows['drug_ids'])!=original_ids:raise ValueError('Incomplete original-ten aligned labels')
        bundles[goal]={}
        for exclude in [None,*sorted(set(rows['cluster_ids']))]:
            bundles[goal][exclude or '__all__']=selector.train(**rows,exclude_cluster=exclude)
    for cid in [c['case_id'] for c in p['cases']]:
        case=gap.get_case(p,cid)
        for regime in gap.REGIMES:
            key=cid+'__'+regime;owner=original_state if cid in original_ids else state
            d=checked(owner['designs'][key]['result']);f=features[key]
            designs[key]=dict(case_id=cid,regime=regime,source=owner['designs'][key]['result'])
            choices[key]={}
            for goal in GOALS:
                ti=d['choices']['time_farthest'][goal]
                old_index=(original_choices[key][goal]['index'] if cid in original_ids else
                           checked(owner['designs'][key]['features'])['goals'][goal]['choices']['learned_gain'])
                if goal not in bundles or f['goals'][goal].get('status')!='selected':
                    choice=dict(selected_index=ti,proposed_index=ti,time_index=ti,adopted=False,
                                reason='No matched training labels or model-only endpoint support',margins=[])
                else:
                    bundle=bundles[goal][case['cluster_id'] if cid in original_ids else '__all__']
                    choice=selector.choose(bundle,f['goals'][goal]['X'],d['candidate_times_h'],ti)
                chosen=dict(previous_learned=old_index,aligned_only=choice['proposed_index'],stable=choice['selected_index'],
                    time=ti,stable_repaired=choice['selected_index'],time_repaired=ti,no_addition=None,no_addition_repaired=None)
                sources={}
                for method,index in chosen.items():
                    if method.startswith('no_addition'):
                        candidate='baseline';source=d['baseline']
                    elif index is None:
                        sources[method]=None;continue
                    else:
                        candidate=key+f'__point_{d["candidate_ids"][index]:02d}'
                        source=owner['units'][candidate]['result']
                    uid=key+'__baseline' if candidate=='baseline' else candidate
                    sources[method]=dict(candidate=candidate,source=source,unit_id=uid,
                        time_h=d['candidate_times_h'][index] if index is not None else None)
                    if method.endswith('_repaired'):
                        jobs[uid]=dict(case_id=cid,design_key=key,source=source,candidate=candidate)
                choices[key][goal]=dict(decision=choice,sources=sources)
    save(RUN/'models.json',bundles)
    save(RUN/'choices.json',dict(choices=choices,locked_at=time.time(),new_twenty_labels_used=False,
        only_visible_before_acquisition=True,predictor_repair_after_fixed_choices=True))
    pins=deepcopy(p['source_pins'])
    for f in (Path(__file__),ROOT/'scripts/blueprint_sampling_stability_features.py',
        ROOT/'src/pbpk_kb/blueprint_policy/stable_sampling_design.py',
        ROOT/'src/pbpk_kb/blueprint_policy/visible_partition_feasibility.py',
        OLD/'protocol.json',OLD/'state.json',OLD/'results.json',RUN/'models.json',RUN/'choices.json',features_path):
        pins[str(f.relative_to(ROOT))]=base.sha256(f)
    new=dict(version='matched-label-jackknife-visible-mass-repair-1',created_at=time.time(),
        source_protocol=state['protocol'],source_state=ref(OLD/'state.json'),source_pins=pins,
        cases=p['cases'],original_case_ids=p['original_case_ids'],jobs=jobs,designs=designs,
        choices=ref(RUN/'choices.json'),models=ref(RUN/'models.json'),features=ref(features_path),
        numeric_tie_relative_tolerance=TOL,bootstrap=dict(seed=20261009,resamples=2000),
        primary='Added20 late_four matched sampled Cmax; compare stable_repaired to time_repaired and previous_learned separately.',
        fit='Samefour-axis objective,16starts,16384calls/formula; only proven visible mass infeasibility expands unmeasured RBC floor to include K_RBC1. Notargetobservations or extra freeparameter.',
        design='Only original10 common matched errors train8feature alpha1ridge; nested whole-study exclusion for original10; everystudy-jackknife must preferfullproposal over time. Missing commonAUCinfinitylabels=>time fallback.',
        decomposition='Aligned-only and stable choices retain old predictor; stable_repaired and time_repaired receive identical general fitterrepair. Choices/features fixed before acquisition fitting; do notattribute predictorrepair gain to selector.',
        limitations=['All30 outcomes were viewed previously; this is another development iteration, not independent blindvalidation.',
            'Studyagreement is a stabilitycheck, not guaranteed improvement or calibratedconfidence.',
            'K_RBC1 is an engineeringneutralreference, not measured biologicaltruth.',
            'Corrected fitting doesnotrefit acquisitionprofile ensembles; fixedpointchoices isolate downstreampredictorrepair.',
            'AUCinfinity/CL/Vz/MRT/Vss commonclinicalreferences remainmissing. No11endpointvalidationclaim.'],
        monitor_seconds=1200,directory_limit_gib=10)
    verify(pins);save(RUN/'protocol.json',new)
    save(RUN/'state.json',dict(status='prepared',protocol=ref(RUN/'protocol.json'),units={}))
    return status()


def fit_job(p,key,job):
    guard=Guard(p);guard(force=True);start=time.monotonic()
    cached=checked(job['source']);training=checked(cached['training'])
    source=checked(p['source_protocol']);case=gap.get_case(source,job['case_id']);audits={};physical={}
    for formula in base.FORMULAS:
        req=gap.request(case,'development',formula);phys=prior.get_phys(source,case,'development',formula,req)
        audits[formula]=repair.feasibility(phys,req,training);physical[formula]=(req,phys)
    if not any(a['repair_required'] for a in audits.values()):
        return dict(result=job['source'],reused=True,feasibility=audits,calls=0,wall_seconds=time.monotonic()-start)
    folder=RUN/'repaired'/key;path=folder/'result.json'
    if path.exists():
        value=read(path)
        if value['protocol']!=ref(RUN/'protocol.json') or value['original_source']!=job['source']:raise ValueError('Changed repair resume')
        verify(value['output_pins']);return dict(result=ref(path),reused=False,feasibility=audits,calls=value['resources']['calls'])
    fits=[];calls=0
    for formula in base.FORMULAS:
        req,phys=physical[formula]
        if audits[formula]['repair_required']:
            f=repair.fit(lambda x,t:old.simulate('blood_partition',phys,req,t,x),req,training,phys,
                         budget=16384,starts=16,before_trial=guard)
            calls+=f['calls']
        else:f=checked(next(f['fit'] for f in cached['fits'] if f['formula']==formula))
        fits.append(dict(formula=formula,fit=f))
    save(folder/'fits.json',fits)
    eligible=[f for f in fits if f['fit']['best'] is not None]
    result=dict(protocol=ref(RUN/'protocol.json'),original_source=job['source'],training=cached['training'],
        status='failed',fits=ref(folder/'fits.json'),feasibility=audits,resources=dict(calls=calls),output_pins={})
    if eligible:
        selected=min(eligible,key=lambda f:f['fit']['best']['objective']);formula=selected['formula'];best=selected['fit']['best']
        target=gap.request(case,'target',formula);phys=prior.get_phys(source,case,'target',formula,target)
        grid=base.engine.prediction_grid(target)
        save(folder/'selection.json',dict(formula=formula,parameters=best,locked_at=time.time(),target_values_used=False))
        prediction=old.simulate('blood_partition',phys,target,grid,best['log_parameters'])
        save(folder/'prediction.json',prior.curve_document(prediction,grid,target));save(folder/'request.json',target)
        result.update(status='predicted',formula=formula,parameters=best,prediction=ref(folder/'prediction.json'),
            request=ref(folder/'request.json'),selection=ref(folder/'selection.json'),mass_balance=prediction['mass_balance'])
    else:result['reason']='No feasible repaired formula'
    result['resources']['wall_seconds']=time.monotonic()-start
    base.pins_in({k:v for k,v in result.items() if k!='output_pins'},result['output_pins']);guard(force=True);save(path,result)
    return dict(result=ref(path),reused=False,feasibility=audits,calls=calls,wall_seconds=time.monotonic()-start)


def run(workers):
    s=read(RUN/'state.json');p=checked(s['protocol']);Guard(p)(force=True)
    if s['status']=='complete':return status()
    s.update(status='fitting',pid=os.getpid(),started_at=s.get('started_at',time.time()));save(RUN/'state.json',s)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(fit_job,p,k,j):k for k,j in p['jobs'].items() if k not in s['units']}
            for future in as_completed(pending):
                key=pending[future];s['units'][key]=future.result();save(RUN/'state.json',s)
                print(f'Checked/refit {len(s["units"])}/{len(p["jobs"])}',flush=True)
        s.update(status='predictions_locked',predictions_locked_at=time.time());save(RUN/'state.json',s)
        return report()
    except BaseException as exc:
        s.update(status='paused',reason=str(exc));save(RUN/'state.json',s);save(RUN/'PAUSE',dict(reason=str(exc)));raise


def counts(ratios):
    return dict(improved=sum(v<1-TOL for v in ratios.values()),worse=sum(v>1+TOL for v in ratios.values()),
                tied=sum(abs(v-1)<=TOL for v in ratios.values()))


def report():
    s=read(RUN/'state.json');p=checked(s['protocol']);Guard(p)(force=True)
    if s['status'] not in ('predictions_locked','complete') or set(s['units'])!=set(p['jobs']):raise ValueError('Predictions must lock before scoring')
    lock=checked(p['choices']);old_records=checked(read(OLD/'state.json')['results'])['records']
    lookup={(r['design_key'],r['candidate']):r for r in old_records};repaired={}
    for key,unit in s['units'].items():
        if unit['reused']:continue
        prediction=checked(unit['result']);case=gap.get_case(p,p['jobs'][key]['case_id'])
        if prediction['status']!='predicted':raise ValueError('Failed repairedprediction retained; inspect result')
        repaired[key]=data.score_case(case,prediction)
    rows=[]
    for key,goals in lock['choices'].items():
        cid=p['designs'][key]['case_id'];case=gap.get_case(p,cid)
        for goal,item in goals.items():
            for method,choice in item['sources'].items():
                record=None
                if choice is not None:
                    record=lookup[key,choice['candidate']]
                    if method.endswith('_repaired') and choice['unit_id'] in repaired:record=repaired[choice['unit_id']]
                for ep in ('curve',*data.UNITS):
                    e={} if record is None or ep=='curve' else record['endpoints'][ep]
                    error=(record['curve_metrics']['gmfe'] if record else None) if ep=='curve' else e.get('error')
                    rows.append(dict(case_id=cid,drug=case['drug'],cluster_id=case['cluster_id'],
                        cohort='original_ten' if cid in p['original_case_ids'] else 'added_twenty',regime=p['designs'][key]['regime'],
                        requested_goal=goal,method=method,endpoint=ep,prediction=e.get('value'),reference=e.get('reference'),
                        error=error,unit=e.get('unit'),weak_tail=e.get('weak_tail'),
                        selected_time_h=choice['time_h'] if choice else None,reference_kind=e.get('reference_kind','observed_curve' if ep=='curve' else 'unavailable')))
    with (RUN/'endpoints.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={};pairs={};cluster={c['case_id']:c['cluster_id'] for c in p['cases']}
    for scope in ('original_ten','added_twenty','all_thirty'):
        subset=[r for r in rows if (scope=='all_thirty' or r['cohort']==scope) and r['regime']=='late_four' and r['requested_goal']=='cmax']
        summary[scope]={};values={}
        for method in METHODS:
            summary[scope][method]={}
            for ep in ('cmax','curve'):
                v={r['case_id']:r['error'] for r in subset if r['method']==method and r['endpoint']==ep and r['error'] is not None}
                values[method,ep]=v;summary[scope][method][ep]=clustered_geomean(v,{k:cluster[k] for k in v},**p['bootstrap'])
        pairs[scope]={}
        for numerator,denominator in (('stable_repaired','time_repaired'),('stable_repaired','previous_learned'),
            ('stable','time'),('aligned_only','time'),('time_repaired','time'),('stable_repaired','no_addition_repaired')):
            name=numerator+'/'+denominator;pairs[scope][name]={}
            for ep in ('cmax','curve'):
                a,b=values[numerator,ep],values[denominator,ep];ratios={k:a[k]/b[k] for k in a.keys()&b.keys()}
                pairs[scope][name][ep]=dict(**clustered_geomean(ratios,{k:cluster[k] for k in ratios},**p['bootstrap']),
                    **counts(ratios),per_case_ratio=ratios)
    result=dict(protocol=s['protocol'],summary=summary,comparisons=pairs,scored_at=time.time(),
        prediction_lock=s['predictions_locked_at'],endpoint_rows=len(rows),endpoints=ref(RUN/'endpoints.csv'),repaired_scores=repaired,
        counts=dict(drugs=len(p['cases']),layouts=len(p['designs']),checked_units=len(s['units']),refitted_units=len(repaired)),limitations=p['limitations'])
    save(RUN/'results.json',result);s.update(status='complete',results=ref(RUN/'results.json'),completed_at=time.time());save(RUN/'state.json',s)
    lines=['# 30药选点稳定性与拟合范围修复','',
        '本轮沿用30药，作为已查看资料上的开发迭代。统一原10药的特征和标签口径；新增20药的结果不训练评分器。原10药按研究整组留出，再在训练内做留一研究稳定性检查。',
        '只在全部留一研究模型都支持时偏离时间规则；质量守恒证明可见数据不可能被旧RBC范围拟合时，扩展未实测RBC下界。选点先固定、再修复所选数据的拟合，分别报告两项改动收益。',
        '', '| 范围 | 方法 | Cmax GMFE | 曲线GMFE |','|---|---|---:|---:|']
    for scope in ('added_twenty','all_thirty'):
        for method in METHODS:
            v=summary[scope][method];lines.append(f"| {scope} | {method} | {v['cmax']['estimate']:.4f} | {v['curve']['estimate']:.4f} |")
    lines+=['','[全部11类输出](endpoints.csv) · [结果与逐药配对](results.json) · [冻结协议](protocol.json)',
        '', '原有论文和本轮之前的结果保留。AUC∞/CL/Vz/MRT/Vss统一临床参考仍缺失，不能声称11类均已验证。浮点比较采用1e-10相对容差，避免把数值噪声算成退步。']
    (RUN/'README.md').write_text('\n'.join(lines)+'\n');return status()


def status():
    if not (RUN/'state.json').exists():return dict(status='preparing')
    s=read(RUN/'state.json');p=checked(s['protocol'])
    return dict(status=s['status'],drugs=len(p['cases']),units_complete=len(s['units']),units_total=len(p['jobs']),
                refitted=sum(not u['reused'] for u in s['units'].values()),reason=s.get('reason'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','run','all','status','report']);ap.add_argument('--workers',type=int,default=14);args=ap.parse_args()
    if not 1<=args.workers<=14:ap.error('workers1..14')
    RUN.mkdir(parents=True,exist_ok=True)
    if args.command=='status':answer=status()
    else:
        with (RUN/'batch.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.command=='all':prepare();answer=run(args.workers)
            elif args.command=='run':answer=run(args.workers)
            else:answer=globals()[args.command]()
    import json
    print(json.dumps(answer,ensure_ascii=False,indent=2))
