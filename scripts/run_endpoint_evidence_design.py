#!/usr/bin/env python3
"""One-observation, request-directed PBPK evidence acquisition benchmark.

Decisions see five concentrations and candidate times only. All actions lock
before six-point fits; all target predictions lock before clinical scoring.
All possible single acquisitions are evaluated for exact random expectation
and an explicitly undeployable oracle, not fed back into the action policy.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import math
import os
from pathlib import Path
import time

import numpy as np
import run_selective_blood_partition as prior
from blueprint_evidence_acquisition_data import split_development, reveal_one
from pbpk_kb.blueprint_policy import endpoint_evidence_design as design

old=prior.old; base=prior.base; ROOT=prior.ROOT
RUN=ROOT/'experiments/blueprint_design/endpoint_evidence_design_20261008'
VERSION='request-directed-one-observation-pbpk-1'
GOALS=('cmax','auc_inf','terminal_half_life')
SCORING={'cmax':'cmax','auc_inf':'auc_inf','terminal_half_life':'terminal_half_life'}
POLICIES=('no_addition','request_directed','maximum_variance','time_farthest','uniform_random','oracle')
read,save,ref,checked,verify=old.read,old.save,old.ref,old.checked,old.verify


class Guard:
    def __init__(self,p): self.p=p;self.last=0.
    def __call__(self,force=False):
        if (RUN/'PAUSE').exists():raise RuntimeError('Acquisition batch paused')
        if not force and time.monotonic()-self.last<1200:return
        verify(self.p['source_pins'])
        size=sum(x.stat().st_size for x in RUN.rglob('*') if x.is_file() and not x.is_symlink())
        if size>10*1024**3:raise RuntimeError('Batch exceeds 10 GiB')
        self.last=time.monotonic()


def request(case,phase,formula):
    req=base.normalized(checked(case['requests']['evidence'][phase]));req['partition_method']=formula
    return req


def validate_cached_profile(cached,case,formula,entry):
    """Reuse only the identical isolated five-point, same-formula profile."""
    expected=dict(kind='clinical',case_id=case['case_id'],sampling='five',formula=formula)
    if (cached.get('job')!=expected or cached.get('status')!='locked'
            or cached.get('protocol_sha256')!=entry['protocol_sha256']):
        raise ValueError('Cached profile identity or protocol differs')
    verify(cached['output_pins'])
    if prior.baseline_formula(cached['candidates']['three'],case,'five')!=formula:
        raise ValueError('Cached profile formula differs from development source')
    prof=cached['profile']
    if prof.get('status')=='protected':
        raise ValueError('Measured blood partition is protected; this profile acquisition pilot is unsupported')
    if (prof.get('visible_points')!=5 or prof.get('max_calls')!=32768
            or prof.get('requested_grid_points')!=25
            or prof.get('log_tolerance')!=math.log(1.1)
            or prof.get('target_observations_used') is not False
            or prof.get('external_warm_start_used') is not False):
        raise ValueError('Cached profile five-point isolation or numerical rule differs')
    return prof


def validate_decision(p,case,data):
    visible=checked(case['visible_design'])['design_only']
    if (data.get('case_id')!=case['case_id']
            or data.get('protocol_sha256')!=base.sha256(RUN/'protocol.json')
            or data.get('visible_source')!=case['visible_design']
            or data.get('candidate_ids')!=visible['candidate_ids']
            or data.get('candidate_times_h')!=visible['candidate_times_h']
            or data.get('target_values_used') is not False
            or data.get('hidden_development_values_used') is not False):
        raise ValueError('Changed decision resume or data isolation')
    n=len(visible['candidate_ids'])
    choices=list(data['goal_choices'].values())+list(data['generic_choices'].values())
    if (set(data['goal_choices'])!=set(GOALS)
            or set(data['generic_choices'])!={'maximum_variance','time_farthest'}
            or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<n for i in choices)):
        raise ValueError('Invalid locked acquisition index')


def validate_unit(p,key,result,lock):
    if (key not in p['jobs'] or result.get('job')!=p['jobs'][key]
            or result.get('protocol_sha256')!=base.sha256(RUN/'protocol.json')
            or result.get('decisions_lock')!=lock
            or result.get('status') not in ('predicted','failed')):
        raise ValueError('Changed fit resume identity or global lock')
    verify(result['output_pins'])


def expansion_checks(primary,pairs,units):
    """Missing or failed comparisons cannot pass the prespecified gate."""
    def positive(value):return value is not None and math.isfinite(value) and value>0
    curve=primary['request_directed']['curve']['estimate']
    return dict(
        all_ten_references=all(primary[m]['endpoint']['n_cases']==10 for m in POLICIES),
        one_percent_better_than_all_generic=all(positive(pairs[m]['estimate']) and pairs[m]['estimate']<=.99 for m in ('time_farthest','maximum_variance','uniform_random')),
        six_cmax_wins=pairs['time_farthest']['ratio_below_one']>=6,
        curve_not_harmed=all(positive(curve) and positive(primary[m]['curve']['estimate'])
            and primary['request_directed']['curve']['n_cases']==10 and primary[m]['curve']['n_cases']==10
            and curve<=1.01*primary[m]['curve']['estimate'] for m in ('time_farthest','no_addition')),
        no_failed_acquisitions=bool(units) and all(u['status']=='predicted' for u in units.values()))


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():raise ValueError('Frozen protocol exists')
    pp=read(prior.RUN/'protocol.json');ps=read(prior.RUN/'state.json')
    bp=read(old.RUN/'protocol.json');bs=read(old.RUN/'state.json')
    checked(ps['protocol']);checked(bs['protocol']);verify(pp['source_pins']);verify(bs['output_pins'])
    if ps['status']!='complete' or bs['status']!='complete':raise ValueError('Require frozen completed prior runs')
    pins=deepcopy(pp['source_pins']);pins.update(bs['output_pins'])
    prepared=[];jobs={};profiles={};baselines={}
    for c in pp['cases']:
        cid=c['case_id'];req=request(c,'development','PKSimStandard')
        full=base._training_document(c['sampling_sources']['full'],req,ROOT)
        five=base._training_document(c['sampling_sources']['five'],req,ROOT)
        split=split_development(full,five,req)
        folder=RUN/'inputs'/cid;folder.mkdir(parents=True,exist_ok=True)
        save(folder/'visible_design.json',split)
        cc=deepcopy(c);cc['visible_design']=ref(folder/'visible_design.json');prepared.append(cc)
        base.pins_in(cc['visible_design'],pins)
        for ix,t in zip(split['design_only']['candidate_ids'],split['design_only']['candidate_times_h']):
            jobs[f'{cid}__point_{ix:02d}']=dict(case_id=cid,candidate_id=ix,time_h=t)
        punit=checked(ps['units'][f'clinical__{cid}__five']['result'])
        profiles[cid]=dict(source=ps['units'][f'clinical__{cid}__five']['result'],formula=punit['job']['formula'],protocol_sha256=ps['protocol']['sha256'])
        validate_cached_profile(punit,c,profiles[cid]['formula'],profiles[cid])
        pins.update(punit['output_pins'])
        base.pins_in(profiles[cid],pins)
        baselines[cid]=bs['units'][f'{cid}__five__blood_partition'];base.pins_in(baselines[cid],pins)
    paths=[Path(__file__),ROOT/'scripts/blueprint_evidence_acquisition_data.py',
        ROOT/'src/pbpk_kb/blueprint_policy/endpoint_evidence_design.py',
        prior.RUN/'protocol.json',prior.RUN/'state.json',old.RUN/'state.json']
    for x in paths:pins[str(x.relative_to(ROOT))]=base.sha256(x)
    p=dict(version=VERSION,created_at=time.time(),cases=prepared,jobs=jobs,profiles=profiles,
        baselines=baselines,source_pins=pins,physiology_cache=pp['physiology_cache'],
        goals=list(GOALS),primary='Clinical Cmax fold error across ten drugs; other endpoint goals separately, never substitute a favorable secondary result.',
        method='Uniform finite collection of data-compatible RBC profiles in both partition formulas; log-endpoint/candidate-concentration conditional-variance reduction; no posterior or clinical confidence claim.',
        ensemble='Use converged profile points within global minimum data MSE+ln(1.1)^2. Reuse exact five-point prior profile for one formula; compute the other formula identically. Existing five-point regularized predictor is not replaced.',
        log_noise_sd=.1,noise_sensitivity=[.05,.1,.2],profile_budget=32768,
        fit_budget_per_formula=16384,starts=16,extra_observations=1,
        data_isolation='Data steward validates complete development inputs, but action API only receives candidate times, five visible observations and model predictions. Reveal exactly one concentration per fit after all policies lock. Clinical target values read only after all predictions lock.',
        controls='Time farthest on log-time; maximum log-concentration variance; exact uniform-random expected log error over all one-point actions. Oracle is post-outcome diagnostic, never a deployable policy.',
        expansion_gate='Cmax requested action GMFE ratio <=0.99 versus each of time, maximum-variance and exact random policies; >=6/10 directional Cmax wins versus time; no added failures/missing predictions; target-curve GMFE no more than 1.01x time and no-addition. Engineering expansion decision, not significance or clinical acceptance. Otherwise retain ten-drug development scope.',
        committee_endpoint_estimators='AUCinf=dose/effective plasma CL; Cmax and terminal half-life from fixed model-only extended-curve summary. Final clinical scoring retains published matched-NCA convention; estimator differences are disclosed.',
        references=['https://doi.org/10.1137/24M1649344','https://doi.org/10.1007/s11538-024-01294-0','https://doi.org/10.1038/s41540-023-00325-1'],
        limitations=['Ten repeatedly viewed development drugs, not blinded clinical validation.',
            'One added concentration within an existing development study; not acquiring a whole independent study or proof of missing CL recovery.',
            'Finite profiles incompletely explore parameter/structure uncertainty; assumed assay log SD is an engineering sensitivity setting.',
            'No laboratory experiment conducted; retrospective hidden-point intervention benchmark.'],
        preflight=dict(target_request_only_simulations=176,clinical_target_values_read=False),
        monitor_seconds=1200,directory_limit_gib=10)
    verify(pins);save(RUN/'protocol.json',p)
    save(RUN/'state.json',dict(status='prepared',protocol=ref(RUN/'protocol.json'),decisions={},units={}))
    return dict(status='prepared',cases=len(prepared),unique_acquisitions=len(jobs))


def committee(p,case):
    guard=Guard(p);guard(force=True);cid=case['case_id'];folder=RUN/'decisions'/cid;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'locked.json'
    if path.exists():
        data=read(path)
        validate_decision(p,case,data)
        return dict(result=ref(path))
    visible=checked(case['visible_design']);training=visible['visible_training'];times=np.asarray(visible['design_only']['candidate_times_h'])
    cached=checked(p['profiles'][cid]['source']);profiles={};profile_calls=0
    for formula in base.FORMULAS:
        req=request(case,'development',formula);phys=prior.get_phys(p,case,'development',formula,req)
        if formula==p['profiles'][cid]['formula']:
            prof=validate_cached_profile(cached,case,formula,p['profiles'][cid]);reused=True
        else:
            tt,yy=old.optimizer._visible(req,training)
            prof=prior.analyze(phys,req,tt,yy,dict(p,rule=dict(minimum_points=8,log_tolerance=math.log(1.1),grid_points=25,max_calls=p['profile_budget'])),guard)
            profile_calls+=prof['calls'];reused=False
        if prof.get('status')=='protected':raise ValueError('Measured blood partition is protected; acquisition profile unsupported')
        profiles[formula]=dict(profile=prof,reused=reused)
    valid=[r['mse'] for f in profiles.values() for r in f['profile'].get('profile',[]) if r['successful']]
    if not valid:raise ValueError('No feasible profile collection')
    threshold=min(valid)+math.log(1.1)**2
    members=[];concentrations=[];goal_values={g:[] for g in GOALS};failures=[];simulation_calls=0
    for formula,item in profiles.items():
        dev=request(case,'development',formula);target=request(case,'target',formula)
        dphys=prior.get_phys(p,case,'development',formula,dev);tphys=prior.get_phys(p,case,'target',formula,target)
        grid=base.engine.prediction_grid(target)
        for i,row in enumerate(item['profile']['profile']):
            if not row['successful'] or row['mse']>threshold:continue
            x=row['best']['log_parameters']
            try:
                guard();simulation_calls+=1
                dev_pred=old.simulate('blood_partition',dphys,dev,times,x)['concentration_mg_l']
                guard();simulation_calls+=1
                pred=old.simulate('blood_partition',tphys,target,grid,x)
                doc=prior.curve_document(pred,grid,target)
                ep=base.summarize_extended_prediction(doc,target)['endpoints']
                values=dict(cmax=ep['cmax'].get('value_mg_per_l'),
                    auc_inf=target['dosing']['dose_mg']/pred['effective_total_clearance_l_h'],
                    terminal_half_life=ep['terminal_half_life'].get('value_h'))
                if not np.isfinite(dev_pred).all() or np.any(np.asarray(dev_pred)<=0):raise ValueError('Invalid candidate concentration')
                members.append(dict(formula=formula,profile_index=i,log_parameters=x,mse=row['mse']))
                concentrations.append(dev_pred)
                for g in GOALS:goal_values[g].append(values[g])
            except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                failures.append(dict(formula=formula,index=i,reason=str(exc)))
    if len(members)<2:raise ValueError('Insufficient model collection for acquisition')
    cc=np.asarray(concentrations);dd=visible['design_only'];choices={};details={}
    # Generic controls always use the same full model collection, regardless of goal.
    generic=design.choose(times,dd['observed_times_h'],cc,{'constant':np.ones(len(cc))},log_noise_sd=.1)
    for g in GOALS:
        good=np.array([v is not None and math.isfinite(v) and v>0 for v in goal_values[g]])
        if good.sum()<2:
            details[g]=dict(status='insufficient_goal_members',available=int(good.sum()),main=None,sensitivity={})
            choices[g]=generic['policies']['time_farthest']['selected_index'];continue
        vals=np.array([v for v,yes in zip(goal_values[g],good) if yes]);sensitivity={}
        for sigma in p['noise_sensitivity']:
            sensitivity[str(sigma)]=design.choose(times,dd['observed_times_h'],cc[good],{g:vals},log_noise_sd=sigma)
        main=sensitivity[str(p['log_noise_sd'])]
        choices[g]=main['policies']['goal_directed'][g]['selected_index']
        details[g]=dict(status='selected',available=int(good.sum()),main=main,sensitivity=sensitivity)
    out=dict(case_id=cid,protocol_sha256=base.sha256(RUN/'protocol.json'),locked_at=time.time(),
        visible_source=case['visible_design'],candidate_times_h=times.tolist(),candidate_ids=dd['candidate_ids'],
        profiles=profiles,ensemble_threshold=threshold,members=members,concentrations_mg_l=cc.tolist(),
        goal_values=goal_values,goal_choices=choices,details=details,
        generic_choices={name:generic['policies'][field]['selected_index'] for name,field in [('maximum_variance','max_variance'),('time_farthest','time_farthest')]},
        resources=dict(new_profile_calls=profile_calls,committee_simulations=simulation_calls),
        member_failures=failures,target_values_used=False,hidden_development_values_used=False)
    validate_decision(p,case,out);guard(force=True);save(path,out);return dict(result=ref(path))


def execute(p,key,job,lock):
    guard=Guard(p);guard(force=True);start=time.monotonic();folder=RUN/'work'/key;folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'result.json';protocol_sha=base.sha256(RUN/'protocol.json')
    if p['jobs'].get(key)!=job:raise ValueError('Acquisition differs from frozen job')
    receipt=checked(lock)
    if (receipt.get('protocol')!=ref(RUN/'protocol.json')
            or set(receipt.get('decisions',{}))!={c['case_id'] for c in p['cases']}):
        raise ValueError('All policies must lock before revealing any concentration')
    if result_path.exists():
        result=read(result_path)
        validate_unit(p,key,result,lock);return dict(result=ref(result_path),status=result['status'])
    case=next(c for c in p['cases'] if c['case_id']==job['case_id'])
    dev=request(case,'development','PKSimStandard')
    decision=checked(receipt['decisions'][case['case_id']]['result']);validate_decision(p,case,decision)
    visible=checked(case['visible_design']);menu=visible['design_only']
    if (job['candidate_id'],job['time_h']) not in list(zip(menu['candidate_ids'],menu['candidate_times_h'])):
        raise ValueError('Acquisition is absent from frozen time-only menu')
    full=base._training_document(case['sampling_sources']['full'],dev,ROOT)
    five=visible['visible_training'];training=reveal_one(full,five,job['time_h'])
    if len(training['observations'])!=6:raise ValueError('Acquisition did not yield exactly six observations')
    if (folder/'training.json').exists() and read(folder/'training.json')!=training:raise ValueError('Changed resumed training data')
    save(folder/'training.json',training);fits=[];calls=0;failed_calls=0
    for formula in base.FORMULAS:
        req=request(case,'development',formula);phys=prior.get_phys(p,case,'development',formula,req)
        phys_source=p['physiology_cache'][f'{case["case_id"]}__development__{formula}']
        fitpath=folder/(formula+'.json')
        if fitpath.exists():
            fit=read(fitpath)
            if (fit.get('protocol_sha256')!=protocol_sha or fit.get('request')!=req
                    or fit.get('training_source')!=ref(folder/'training.json')
                    or fit.get('development_physiology')!=phys_source
                    or fit.get('variant')!='blood_partition' or fit.get('call_limit')!=p['fit_budget_per_formula']
                    or len(fit.get('starts',[]))!=p['starts'] or fit.get('visible_points')!=6
                    or fit.get('no_warm_start') is not True or fit.get('target_observations_used') is not False):raise ValueError('Changed resumed fit')
        else:
            fit=old.optimizer.fit(lambda x,t:old.simulate('blood_partition',phys,req,t,x),req,training,phys,
                variant='blood_partition',budget=p['fit_budget_per_formula'],starts=p['starts'],before_trial=guard)
            fit.update(protocol_sha256=protocol_sha,request=req,training_source=ref(folder/'training.json'),development_physiology=phys_source)
            save(fitpath,fit)
        calls+=fit['calls'];failed_calls+=fit['failures'];fits.append(dict(formula=formula,fit=ref(fitpath),best=fit['best']))
    valid=[f for f in fits if f['best'] is not None]
    result=dict(protocol_sha256=protocol_sha,job=job,decisions_lock=lock,fits=fits,status='failed',
        training=ref(folder/'training.json'),resources=dict(objective_calls=calls,failed_calls=failed_calls,prediction_calls=0))
    if valid:
        best=min(valid,key=lambda f:f['best']['objective']);formula=best['formula'];target=request(case,'target',formula)
        save(folder/'selection.json',dict(formula=formula,parameters=best['best'],locked_at=time.time(),target_values_used=False))
        phys=prior.get_phys(p,case,'target',formula,target);times=base.engine.prediction_grid(target)
        guard();result['resources']['prediction_calls']=1
        try:
            pred=old.simulate('blood_partition',phys,target,times,best['best']['log_parameters'])
            doc=prior.curve_document(pred,times,target)
            save(folder/'prediction.json',doc);save(folder/'request.json',target)
            result.update(status='predicted',formula=formula,parameters=best['best'],prediction=ref(folder/'prediction.json'),
                request=ref(folder/'request.json'),selection=ref(folder/'selection.json'),mass_balance=pred['mass_balance'])
        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:result['reason']=str(exc)
    else:result['reason']='No successful formula fit'
    result['resources']['wall_seconds']=time.monotonic()-start;result['output_pins']={}
    base.pins_in({k:v for k,v in result.items() if k!='output_pins'},result['output_pins'])
    guard(force=True);save(result_path,result);return dict(result=ref(result_path),status=result['status'])


def run(workers):
    state=read(RUN/'state.json');p=checked(state['protocol']);Guard(p)(force=True)
    if state['status']=='complete':return status()
    state.update(status='designing',pid=os.getpid(),started_at=state.get('started_at',time.time()));save(RUN/'state.json',state)
    cases={c['case_id']:c for c in p['cases']}
    for cid,entry in state['decisions'].items():
        if cid not in cases:raise ValueError('Unknown resumed decision')
        validate_decision(p,cases[cid],checked(entry['result']))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tasks={pool.submit(committee,p,c):c['case_id'] for c in p['cases'] if c['case_id'] not in state['decisions']}
            for f in as_completed(tasks):
                state['decisions'][tasks[f]]=f.result();save(RUN/'state.json',state)
                print('decisions',len(state['decisions']),len(p['cases']),flush=True)
        if set(state['decisions'])!={c['case_id'] for c in p['cases']}:raise ValueError('Incomplete action lock')
        if not (RUN/'decisions_locked.json').exists():save(RUN/'decisions_locked.json',dict(protocol=state['protocol'],decisions=state['decisions'],locked_at=time.time()))
        lock=ref(RUN/'decisions_locked.json');receipt=checked(lock)
        if receipt['protocol']!=state['protocol'] or receipt['decisions']!=state['decisions']:raise ValueError('Changed global action lock')
        state.update(status='fitting',decisions_lock=lock);save(RUN/'state.json',state)
        for key,entry in state['units'].items():validate_unit(p,key,checked(entry['result']),lock)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tasks={pool.submit(execute,p,k,j,lock):k for k,j in p['jobs'].items() if k not in state['units']}
            for f in as_completed(tasks):
                state['units'][tasks[f]]=f.result();save(RUN/'state.json',state)
                print('acquisitions',len(state['units']),len(p['jobs']),flush=True)
        Guard(p)(force=True);state.update(status='predictions_locked',locked_at=time.time());save(RUN/'state.json',state)
        return report()
    except Exception as exc:
        save(RUN/'PAUSE',dict(reason=str(exc)));state.update(status='paused',reason=str(exc));save(RUN/'state.json',state);raise


def report():
    from report_blueprint_validation import Store,_case_reference,_score_prediction,clustered_geomean,paired_ratio
    state=read(RUN/'state.json');p=checked(state['protocol'])
    if state['status'] not in ('predictions_locked','complete') or set(state['units'])!=set(p['jobs']):raise ValueError('All predictions must lock')
    verify(p['source_pins']);store=Store(ROOT);all_scores={};records=[];rows=[];goals=[]
    if set(state['decisions'])!={c['case_id'] for c in p['cases']}:raise ValueError('All decisions must lock')
    receipt=checked(state['decisions_lock'])
    if receipt['decisions']!=state['decisions'] or receipt['protocol']!=state['protocol']:raise ValueError('Changed global action lock')
    units={k:checked(v['result']) for k,v in state['units'].items()};decisions={k:checked(v['result']) for k,v in state['decisions'].items()}
    for key,u in units.items():validate_unit(p,key,u,state['decisions_lock'])
    for c in p['cases']:validate_decision(p,c,decisions[c['case_id']])
    for c in p['cases']:
        cid=c['case_id'];clinical,curve,endpoints=_case_reference(c,store);scored={}
        for key,u in [('baseline',p['baselines'][cid])]+[(k,u) for k,u in units.items() if u['job']['case_id']==cid]:
            rec=dict(case_id=cid,drug=c['drug'],candidate=key,status=u['status'])
            if u['status']=='predicted':
                m,_,_=_score_prediction(c,clinical,curve,endpoints,u,store);m.pop('curve',None);rec.update(m)
            else:rec.update(reason=u.get('reason'),curve_metrics={'gmfe':None},endpoints={e:{'value':None,'reference':r['value'],'error':None,'unit':r['unit']} for e,r in endpoints.items()})
            scored[key]=rec;records.append(rec)
            head={k:rec[k] for k in ('drug','candidate','status')}
            rows.append(head|dict(endpoint='curve',prediction=None,reference=None,error=rec['curve_metrics'].get('gmfe'),unit=curve['unit'],metric='GMFE'))
            for ep,v in rec['endpoints'].items():rows.append(head|dict(endpoint=ep,prediction=v.get('value'),reference=v.get('reference'),error=v.get('error'),unit=v.get('unit'),metric=v.get('error_metric')))
        all_scores[cid]=scored;d=decisions[cid]
        keys=[f'{cid}__point_{ix:02d}' for ix in d['candidate_ids']]
        for goal in GOALS:
            endpoint=SCORING[goal];base_record=scored['baseline'];candidate_errors=[scored[k]['endpoints'][endpoint].get('error') for k in keys]
            complete=all(v is not None and v>0 and math.isfinite(v) for v in candidate_errors)
            curves=[scored[k]['curve_metrics'].get('gmfe') for k in keys]
            row=dict(case_id=cid,drug=c['drug'],goal=goal,reference=base_record['endpoints'][endpoint].get('reference'),
                candidate_count=len(keys),policies={},complete_random_endpoint=complete,
                request_changes_action=len(set(d['goal_choices'].values()))>1)
            for policy in POLICIES:
                selected=None
                if policy=='no_addition':r=base_record
                elif policy=='request_directed':
                    if d['details'][goal]['status']!='selected':
                        row['policies'][policy]=dict(error=None,curve_gmfe=None,reason='Insufficient supported endpoint committee; no request-directed action')
                        continue
                    selected=d['goal_choices'][goal];r=scored[keys[selected]]
                elif policy in d['generic_choices']:selected=d['generic_choices'][policy];r=scored[keys[selected]]
                elif policy=='oracle':
                    if not complete:row['policies'][policy]=dict(error=None,curve_gmfe=None,reason='Incomplete candidate references');continue
                    selected=int(np.argmin(candidate_errors));r=scored[keys[selected]]
                else:
                    row['policies'][policy]=dict(error=math.exp(float(np.mean(np.log(candidate_errors)))) if complete else None,
                        curve_gmfe=math.exp(float(np.mean(np.log(curves)))) if all(v is not None and math.isfinite(v) and v>0 for v in curves) else None,
                        semantics='Exact expected log error of a single uniformly selected observation; no averaged prediction')
                    continue
                er=r['endpoints'][endpoint]
                row['policies'][policy]=dict(error=er.get('error'),prediction=er.get('value'),weak_tail=er.get('weak_tail'),
                    curve_gmfe=r['curve_metrics'].get('gmfe'),selected_index=selected,
                    selected_time_h=None if selected is None else d['candidate_times_h'][selected])
            goals.append(row)
    clusters={c['case_id']:c['cluster_id'] for c in p['cases']};summary={};comparison={}
    for goal in GOALS:
        chosen=[r for r in goals if r['goal']==goal];summary[goal]={};comparison[goal]={}
        score={m:{r['case_id']:r['policies'][m].get('error') for r in chosen} for m in POLICIES}
        for policy in POLICIES:
            v={k:x for k,x in score[policy].items() if x is not None and math.isfinite(x) and x>0}
            cr={r['case_id']:r['policies'][policy].get('curve_gmfe') for r in chosen};cr={k:x for k,x in cr.items() if x is not None and math.isfinite(x) and x>0}
            summary[goal][policy]=dict(endpoint=clustered_geomean(v,{k:clusters[k] for k in v}),curve=clustered_geomean(cr,{k:clusters[k] for k in cr}),
                missing_endpoint=len(p['cases'])-len(v),endpoint_over_twofold=sum(x>2 for x in v.values()))
            if policy!='request_directed':comparison[goal][policy]=paired_ratio(score['request_directed'],score[policy],clusters)
    primary=summary['cmax'];pairs=comparison['cmax']
    gatechecks=expansion_checks(primary,pairs,units)
    result=dict(version=VERSION,scored_at=time.time(),global_prediction_lock=state['locked_at'],summary=summary,comparisons=comparison,
        expansion_checks=gatechecks,expand_to_twenty=all(gatechecks.values()),records=records,goal_records=goals,
        resource=dict(new_fit_calls=sum(u['resources']['objective_calls'] for u in units.values()),
            new_profile_calls=sum(d['resources']['new_profile_calls'] for d in decisions.values()),
            committee_simulations=sum(d['resources']['committee_simulations'] for d in decisions.values()),
            target_predictions=sum(u['resources']['prediction_calls'] for u in units.values()),
            elapsed_seconds=time.time()-state['started_at']),limitations=p['limitations'])
    save(RUN/'results.json',result)
    with (RUN/'endpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Request-directed PBPK evidence acquisition','','Same five visible development samples + one hidden concentration per action. All ten previously viewed development drugs; no clinical-risk guarantee.','',
        '| Requested endpoint | No addition | Request-directed | Max variance | Time rule | Exact random | Oracle (undeployable) |','|---|---:|---:|---:|---:|---:|---:|']
    for goal in GOALS:
        values=[summary[goal][m]['endpoint']['estimate'] for m in POLICIES]
        lines.append('| '+goal+' | '+' | '.join('NA' if v is None else f'{v:.4f}' for v in values)+' |')
    lines+=['',f'Expand to twenty: **{result["expand_to_twenty"]}**. Checks: `{gatechecks}`.',
        '', 'All 11 endpoint categories and missing references are retained in `endpoints.csv`. Random values are expected log errors, not a combined predictor. Oracle uses target outcomes and cannot be deployed.',
        '', 'Policies use model-only finite ensembles and assumed measurement noise, not a calibrated posterior. Per-goal noise sensitivity, candidate choices and all profiles are stored in each decision receipt.',
        '', '```bash','runtime/app-env/bin/python -B scripts/run_endpoint_evidence_design.py status','```']
    (RUN/'README.md').write_text('\n'.join(lines)+'\n')
    state.update(status='complete',completed_at=time.time(),results=ref(RUN/'results.json'));save(RUN/'state.json',state)
    return dict(status='complete',summary=summary,expand_to_twenty=result['expand_to_twenty'])


def status():
    s=read(RUN/'state.json');p=checked(s['protocol'])
    return dict(status=s['status'],decisions=len(s['decisions']),cases=len(p['cases']),acquisitions=len(s['units']),total=len(p['jobs']),pid=s.get('pid'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run','status','report'));parser.add_argument('--workers',type=int,default=8)
    a=parser.parse_args()
    if not 1<=a.workers<=14:parser.error('workers must be 1..14')
    print({'prepare':prepare,'run':lambda:run(a.workers),'status':status,'report':report}[a.command](),flush=True)
