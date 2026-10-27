#!/usr/bin/env python3
"""Development-only engineering profile gate; clinical replay + known-truth pilot.

The profile band is a fixed engineering tolerance, NOT a confidence interval.
Clinical candidates keep the baseline-selected formula and frozen regularized
fits. Synthetic comparisons use the same data-only profile fits for all gates.
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
import run_blood_partition_blueprint as old
from pbpk_kb.blueprint_policy import blood_partition_calibration as blood
from pbpk_kb.blueprint_policy import selective_blood_partition as gate
from blueprint_selective_synthetic import build_case

base = old.base
ROOT = old.ROOT
RUN = ROOT / 'experiments/blueprint_design/selective_partition_20261008'
VERSION = 'selective-predicted-partition-engineering-pilot-1'
METHODS = ('always_three', 'always_four', 'necessity_only', 'selective')
SCENARIOS = ('correct_RBC', 'wrong_RBC', 'nuisance_error', 'wrong_exchange')
read, save, ref, checked, verify = old.read, old.save, old.ref, old.checked, old.verify


def baseline_formula(control, case, sampling):
    """Replay the frozen development selection, never a target error ranking."""
    if (control.get('status') != 'predicted' or control.get('method') != 'direct_relaxed'
            or control.get('case_id') != case['case_id'] or control.get('sampling') != sampling
            or control.get('training_source') != case['sampling_sources'][sampling]
            or control.get('target_observations_used') is not False):
        raise ValueError('Frozen three-axis control identity or development isolation differs')
    eligible = [r for r in control.get('formula_scores', []) if r.get('best') is not None]
    if not eligible or any(not math.isfinite(r['best']['objective']) for r in eligible):
        raise ValueError('No finite development formula score')
    chosen = min(eligible, key=lambda r: r['best']['objective'])
    selection = checked(control['selection'])
    if (chosen['formula'] not in base.FORMULAS or control.get('formula') != chosen['formula']
            or selection.get('formula') != chosen['formula']
            or selection.get('basis') != 'development_only'
            or selection.get('parameters') != control.get('parameters')
            or chosen['best'] != control.get('parameters')):
        raise ValueError('Baseline formula was not the frozen development-only selection')
    return chosen['formula']


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise ValueError('Frozen protocol already exists')
    prior = read(old.RUN/'protocol.json')
    state = read(old.RUN/'state.json')
    checked(state['protocol']); verify(prior['source_pins']); verify(state['output_pins'])
    if state['status'] != 'complete': raise ValueError('Require completed frozen candidates')
    pins = deepcopy(prior['source_pins'])
    pins.update(state['output_pins'])
    caches = deepcopy(prior['physiology_cache'])
    for p in (old.RUN/'native_cache').glob('*/physiology/perfusion_physiology.json'):
        caches[p.parent.parent.name] = ref(p)
    for r in caches.values():
        phys = checked(r); base.pins_in(r, pins)
        pins.update(phys['source_pins']); pins.update(phys['artifact_pins'])
        base.pins_in(ref((ROOT/r['path']).parent.parent/'base/completion.json'), pins)
    controls = {}; jobs = {}
    for c in prior['cases']:
        cid = c['case_id']
        for sampling in base.SAMPLINGS:
            control = prior['controls'][f'{cid}__{sampling}__direct_relaxed']
            formula = baseline_formula(control, c, sampling)
            candidate = state['units'][f'{cid}__{sampling}__blood_partition']
            if (candidate.get('case_id') != cid or candidate.get('sampling') != sampling
                    or candidate.get('method') != 'blood_partition'
                    or candidate.get('training_source') != c['sampling_sources'][sampling]):
                raise ValueError('Frozen four-axis unit identity differs')
            candidates = [f['fit'] for f in candidate['formula_scores'] if f['formula']==formula]
            if len(candidates) != 1: raise ValueError('Require exactly one same-formula four-axis fit')
            selected_fit = candidates[0]
            key = f'clinical__{cid}__{sampling}'
            controls[key] = dict(three=control, four_fit=selected_fit)
            base.pins_in(controls[key], pins)
            jobs[key] = dict(kind='clinical',case_id=cid,sampling=sampling,formula=formula)
        for scenario in SCENARIOS:
            for sigma in (.05,.15):
                for seed in (1001,1002):
                    for sampling in base.SAMPLINGS:
                        key = f'synthetic__{cid}__{scenario}__{sigma}__{seed}__{sampling}'
                        jobs[key] = dict(kind='synthetic',case_id=cid,sampling=sampling,
                            formula='PKSimStandard',scenario=scenario,log_sigma=sigma,seed=seed)
    for p in (Path(__file__), ROOT/'scripts/blueprint_selective_synthetic.py',
              ROOT/'src/pbpk_kb/blueprint_policy/selective_blood_partition.py',
              old.RUN/'protocol.json', old.RUN/'state.json'):
        pins[str(p.relative_to(ROOT))] = base.sha256(p)
    protocol = dict(version=VERSION,created_at=time.time(),cases=prior['cases'],jobs=jobs,
        controls=controls,physiology_cache=caches,source_pins=pins,
        rule=dict(minimum_points=8,log_tolerance=math.log(1.1),grid_points=25,max_calls=32768,
            rationale='Fixed 10% log-concentration engineering tolerance; at least two points per candidate parameter. Neither a significance threshold nor a calibrated confidence interval.'),
        numerical_preflight=dict(development_only=True,target_observations_read=False,
            profile_calls=13874,synthetic_generator_calls=320,synthetic_unique_cases=160,
            synthetic_generator_failures=0,formal_budget_separate=True,
            tested_budgets=[8192,32768],uniform_formal_budget=32768,
            final_profiles={'Treprostinil':dict(successful_points=26,total_points=26,calls=3407),
                            'Cefepime':dict(successful_points=25,total_points=25,calls=3751)},
            basis='Budget raised for every task after an 8192-call profile contained an unconverged point; no target-error threshold tuning.'),
        mechanism='Only native-predicted RBC partition eligible; measured evidence is protected. Select four axes only if the data-only profile excludes scale=1, is connected and bounded, and has enough observations.',
        formula='Clinical baseline3 development-selected formula is fixed for profile and four-axis candidate. Synthetic formula fixed PKSimStandard.',
        clinical='20 decisions on ten previously viewed drugs; frozen regularized candidate fits reused. All predictions lock before outcome scoring. No new blind validation.',
        synthetic='320 visible-data jobs: ten templates x four truth states x two log-noise levels x two seeds x full/five. Shared noisy curve with subset sampling; no actual clinical fit used as truth. Data-only fits shared between four decision policies.',
        synthetic_limits='Same-generator cases are engineering recovery checks. Wrong-exchange cases check out-of-family attribution. Neither adds clinical patients or compounds.',
        adoption='Do not replace the current point predictor unless decisions reduce unnecessary RBC changes without destroying useful recovery; report all clinical error changes, both sampling regimes, all failures. No target-based threshold revision in this batch.',
        references=['https://doi.org/10.1371/journal.pcbi.1011515','https://doi.org/10.1098/rsta.2024.0219'],
        monitor_seconds=1200,directory_limit_gib=10)
    verify(pins); save(RUN/'protocol.json',protocol)
    save(RUN/'state.json',dict(version=VERSION,status='prepared',protocol=ref(RUN/'protocol.json'),units={}))
    return {'status':'prepared','clinical_jobs':20,'synthetic_jobs':320}


class Guard:
    def __init__(self, protocol): self.protocol=protocol;self.last=0.
    def __call__(self, force=False):
        if (RUN/'PAUSE').exists(): raise RuntimeError('Project experiment paused')
        if not force and time.monotonic()-self.last<1200:return
        verify(self.protocol['source_pins'])
        size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
        if size>10*1024**3:raise RuntimeError('Experiment exceeds 10 GiB')
        self.last=time.monotonic()


def get_phys(protocol, case, phase, formula, request):
    r=protocol['physiology_cache'][f'{case["case_id"]}__{phase}__{formula}']
    receipt=read((ROOT/r['path']).parent.parent/'base/completion.json')
    if base.normalized(receipt['request'])!=base.normalized(request):
        raise ValueError('Native cache request mismatch')
    return checked(r)


def arrays(case, sampling, req):
    if sampling not in base.SAMPLINGS: raise ValueError('Unknown visible sampling regime')
    training=base._training_document(case['sampling_sources'][sampling],req,ROOT)
    if sampling == 'five' and len(training.get('observations', [])) != 5:
        raise ValueError('Five-point isolation failed')
    return old.optimizer._visible(req,training)


def analyze(phys,req,times,obs,protocol,guard):
    if protocol['rule']['minimum_points'] != 8:
        raise ValueError('Profile implementation requires the frozen eight-point engineering rule')
    # This pure check also rejects unknown/conflicting blood-partition evidence.
    policy=blood._evidence_policy(phys,req,0.)
    blood._native_source(phys)
    if policy['explicit_measured_records']:
        return dict(status='protected',calls=0,failures=0,
            decision={'accepted':False,'selected_model':'direct_relaxed','reasons':['measured_evidence_protected']},
            necessity_only_decision={'accepted':False,'selected_model':'direct_relaxed','reasons':['measured_evidence_protected']})
    lower,upper,_,_,_=old.optimizer._initialization(req,phys,'blood_partition',2)
    return gate.analyze(lambda x,t: blood.simulate(phys,req,t,x),times,obs,lower,upper,
        log_tolerance=protocol['rule']['log_tolerance'],grid_points=protocol['rule']['grid_points'],
        max_calls=protocol['rule']['max_calls'],before_trial=guard)


def curve_document(prediction,times,req):
    mw=next(float(e['value']) for e in req['evidence'] if e['parameter']=='molecular_weight')
    return dict(time_min=(np.asarray(times)*60).tolist(),
        concentration=(np.asarray(prediction['concentration_mg_l'])*1000/mw).tolist(),
        time_unit='min',unit='umol/l',observation=prediction.get('observation','plasma'))


def clinical(protocol,key,job,case,folder,guard):
    control=protocol['controls'][key]['three']
    if baseline_formula(control,case,job['sampling']) != job['formula']:
        raise ValueError('Job formula differs from the baseline development selection')
    req=base.normalized(checked(case['requests']['evidence']['development']));req['partition_method']=job['formula']
    phys=get_phys(protocol,case,'development',job['formula'],req)
    times,obs=arrays(case,job['sampling'],req)
    candidate=checked(protocol['controls'][key]['four_fit'])
    if candidate['request']!=req or candidate['training_source']!=case['sampling_sources'][job['sampling']]:
        raise ValueError('Frozen four-axis candidate differs from visible data/formula')
    expected_phys=protocol['physiology_cache'][f'{case["case_id"]}__development__{job["formula"]}']
    if (candidate.get('version') != old.optimizer.VERSION or candidate.get('variant') != 'blood_partition'
            or candidate.get('development_physiology') != expected_phys
            or candidate.get('call_limit') != old.BUDGET
            or not 0 < candidate.get('calls', 0) <= old.BUDGET
            or len(candidate.get('starts', [])) != old.STARTS
            or candidate.get('no_warm_start') is not True
            or candidate.get('target_observations_used') is not False
            or candidate.get('visible_points') != len(times)):
        raise ValueError('Frozen four-axis candidate provenance, budget or visible-data contract differs')
    if candidate.get('best') is None or candidate.get('status') != 'fitted':
        raise ValueError('Frozen candidate has no eligible fit')
    x=np.asarray(candidate['best']['log_parameters'],float)
    lower,upper,_,_,_=old.optimizer._initialization(req,phys,'blood_partition',2)
    if x.shape != (4,) or not np.isfinite(x).all() or np.any(x<lower) or np.any(x>upper):
        raise ValueError('Frozen four-axis parameters are invalid or outside the declared bounds')
    result=analyze(phys,req,times,obs,protocol,guard)
    protected=result['status']=='protected'
    selected={m:('four' if (m=='always_four' or m=='selective' and result['decision']['accepted']
                  or m=='necessity_only' and result['necessity_only_decision']['accepted']) and not protected else 'three') for m in METHODS}
    save(folder/'decision.json',dict(profile=result,choices=selected,locked_at=time.time(),
        formula=job['formula'],four_fit=protocol['controls'][key]['four_fit'],target_observations_used=False))
    if protected:
        return dict(profile=result,choices=selected,decision=ref(folder/'decision.json'),
            candidates=dict(three=control,four=control),prediction_calls=0)
    target=base.normalized(checked(case['requests']['evidence']['target']));target['partition_method']=job['formula']
    target_phys=get_phys(protocol,case,'target',job['formula'],target)
    grid=base.engine.prediction_grid(target)
    guard();prediction=blood.simulate(target_phys,target,grid,x)
    doc=curve_document(prediction,grid,target)
    save(folder/'prediction.json',doc);save(folder/'request.json',target)
    four=dict(status='predicted',prediction=ref(folder/'prediction.json'),request=ref(folder/'request.json'),
              mass_balance=prediction['mass_balance'],parameters=candidate['best'])
    return dict(profile=result,choices=selected,decision=ref(folder/'decision.json'),
        candidates=dict(three=protocol['controls'][key]['three'],four=four),prediction_calls=1)


def synthetic(protocol,key,job,case,folder,guard):
    req=base.normalized(checked(case['requests']['evidence']['development']));req['partition_method']='PKSimStandard'
    phys=get_phys(protocol,case,'development','PKSimStandard',req)
    full,_=arrays(case,'full',req);five,_=arrays(case,'five',req)
    indices=[int(np.flatnonzero(np.isclose(full,t,rtol=0,atol=1e-12))[0]) for t in five]
    generated=build_case(case['case_id'],phys,req,full.tolist(),indices,
        job['scenario'],job['log_sigma'],job['seed'])
    visible=generated['calibration'][job['sampling']]
    result=analyze(phys,visible['request'],visible['times_h'],visible['observations_mg_l'],protocol,guard)
    save(folder/'decision.json',dict(profile=result,locked_at=time.time(),target_observations_used=False))
    # Synthetic truth/transfer is consumed for scoring only after the decision is saved.
    return dict(profile=result,decision=ref(folder/'decision.json'),
        generated=generated,prediction_calls=0)


def execute(protocol,key,job):
    guard=Guard(protocol);guard(force=True);started=time.monotonic()
    folder=RUN/'work'/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        result=read(folder/'result.json')
        if result['protocol_sha256']!=base.sha256(RUN/'protocol.json') or result['job']!=job:
            raise ValueError('Resume inputs changed')
        verify(result['output_pins'])
        if result.get('status')!='locked':raise ValueError('Resumed unit is not locked')
        return dict(result=ref(folder/'result.json'),job=job,status='locked',wall_seconds=result['wall_seconds'])
    case=next(c for c in protocol['cases'] if c['case_id']==job['case_id'])
    content=(clinical if job['kind']=='clinical' else synthetic)(protocol,key,job,case,folder,guard)
    result=dict(job=job,protocol_sha256=base.sha256(RUN/'protocol.json'),status='locked',**content,
        wall_seconds=time.monotonic()-started,output_pins={})
    base.pins_in(content,result['output_pins']);guard(force=True)
    save(folder/'result.json',result)
    return dict(result=ref(folder/'result.json'),job=job,status='locked',wall_seconds=result['wall_seconds'])


def run(workers):
    state=read(RUN/'state.json');protocol=checked(state['protocol']);Guard(protocol)(force=True)
    if state['status']=='complete':return status()
    state.update(status='running',pid=os.getpid(),started_at=state.get('started_at',time.time()))
    save(RUN/'state.json',state)
    # Verify every resumed unit before skipping it.
    for k,u in state['units'].items():
        r=checked(u['result']);verify(r['output_pins'])
        if (k not in protocol['jobs'] or u.get('job')!=protocol['jobs'][k]
                or r.get('job')!=protocol['jobs'][k] or r.get('status')!='locked'
                or r.get('protocol_sha256')!=base.sha256(RUN/'protocol.json')):
            raise ValueError('Resumed unit identity or protocol differs')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(execute,protocol,k,j):k for k,j in protocol['jobs'].items() if k not in state['units']}
        try:
            for f in as_completed(pending):
                state['units'][pending[f]]=f.result();save(RUN/'state.json',state)
                print(f'{len(state["units"])}/{len(protocol["jobs"])} {pending[f]}',flush=True)
        except Exception as exc:
            save(RUN/'PAUSE',dict(reason=str(exc)))
            for f in pending:f.cancel()
            state.update(status='paused',reason=str(exc));save(RUN/'state.json',state);raise
    Guard(protocol)(force=True)
    state.update(status='predictions_locked',locked_at=time.time());save(RUN/'state.json',state)
    return report()


def profile_candidates(p):
    valid=[]
    for row in p.get('profile',[]):
        best=row.get('best')
        if not row.get('successful') or best is None:continue
        x=np.asarray(best.get('log_parameters'),float)
        if (x.shape!=(4,) or not np.isfinite(x).all() or not math.isfinite(best['mse'])
                or best['mse']<0 or best['mse']!=row.get('mse')
                or abs(x[3]-row['log_rbc_scale'])>1e-12):
            raise ValueError('Profile candidate violates its fixed-coordinate or data-loss contract')
        valid.append(dict(mse=best['mse'],log_parameters=x.tolist()))
    null=next((r for r in valid if abs(r['log_parameters'][3])<=1e-12),None)
    if null is None:raise ValueError('No successful null candidate')
    best=min(valid,key=lambda r:r['mse'])
    return dict(three=null['log_parameters'],four=best['log_parameters'])


def summarize_synthetic(subset):
    return dict(n=len(subset),activated=sum(r['decision']['accepted'] for r in subset),
        necessity_activated=sum(r['necessity_only']['accepted'] for r in subset),
        failed_candidates=sum(r['candidate_failure'] is not None for r in subset),
        profile_status_counts={s:sum(r['status']==s for r in subset) for s in sorted({r['status'] for r in subset})},
        transfer_gmfe={m:math.exp(float(np.mean(np.log([r['errors'][m] for r in subset if m in r['errors']]))))
            if any(m in r['errors'] for r in subset) else None for m in METHODS},
        transfer_scored_counts={m:sum(m in r['errors'] for r in subset) for m in METHODS})


def report():
    from report_blueprint_validation import Store,_case_reference,_score_prediction,clustered_geomean,paired_ratio
    state=read(RUN/'state.json');protocol=checked(state['protocol'])
    if state['status'] not in ('predictions_locked','complete') or set(state['units'])!=set(protocol['jobs']):
        raise ValueError('All decisions/predictions must lock before scoring')
    verify(protocol['source_pins']);store=Store(ROOT);records=[];rows=[];synth=[]
    calls=failures=0
    units={k:checked(u['result']) for k,u in state['units'].items()}
    for k,u in units.items():
        verify(u['output_pins'])
        if (u.get('status')!='locked' or u.get('job')!=protocol['jobs'][k]
                or u.get('protocol_sha256')!=base.sha256(RUN/'protocol.json')):
            raise ValueError('Scoring requires the exact locked job and protocol')
        calls+=u['profile']['calls'];failures+=u['profile']['failures']
    for case in protocol['cases']:
        clinical_ref,curve,endpoints=_case_reference(case,store)
        for sampling in base.SAMPLINGS:
            u=units[f'clinical__{case["case_id"]}__{sampling}'];metrics={}
            for name,candidate in u['candidates'].items():
                m,_,_=_score_prediction(case,clinical_ref,curve,endpoints,candidate,store);m.pop('curve',None);metrics[name]=m
            for method in METHODS:
                rec=dict(case_id=case['case_id'],drug=case['drug'],sampling=sampling,method=method,
                    chosen=u['choices'][method],**metrics[u['choices'][method]])
                records.append(rec)
                head={k:rec[k] for k in ('drug','sampling','method','chosen')}
                rows.append(head|dict(endpoint='curve',prediction=None,reference=None,error=rec['curve_metrics']['gmfe'],unit=curve['unit'],metric='GMFE'))
                for ep,r in rec['endpoints'].items():
                    rows.append(head|dict(endpoint=ep,prediction=r.get('value'),reference=r.get('reference'),error=r.get('error'),unit=r['unit'],metric=r.get('error_metric')))
    for key,u in units.items():
        j=u['job']
        if j['kind']!='synthetic':continue
        g=u['generated'];p=u['profile'];transfer=g['transfer']
        case=next(c for c in protocol['cases'] if c['case_id']==j['case_id'])
        req=g['calibration'][j['sampling']]['request']
        phys=get_phys(protocol,case,'development','PKSimStandard',req)
        row=dict(**j,decision=p['decision'],necessity_only=p['necessity_only_decision'],status=p['status'],calls=p['calls'],errors={},candidate_failure=None)
        try:
            xs=profile_candidates(p)
            truth=np.asarray(transfer['observations_mg_l']);times=np.asarray(transfer['times_h'])
            mask=np.isfinite(truth)&(truth>max(truth)*1e-8)
            if not mask.any():raise ValueError('No positive synthetic transfer truth')
            errors={}
            for name,x in xs.items():
                pred=np.asarray(blood.simulate(phys,transfer['request'],times,x)['concentration_mg_l'])
                if not np.isfinite(pred[mask]).all() or np.any(pred[mask]<=0):raise ValueError('Invalid synthetic prediction')
                errors[name]=math.exp(float(np.mean(np.abs(np.log(pred[mask])-np.log(truth[mask])))))
            for method in METHODS:
                choice='four' if (method=='always_four' or method=='selective' and p['decision']['accepted'] or method=='necessity_only' and p['necessity_only_decision']['accepted']) else 'three'
                row['errors'][method]=errors[choice]
            row['evaluated_points']=int(mask.sum());row['conditional_truth_mask']=True
        except (ValueError,KeyError,ArithmeticError,np.linalg.LinAlgError) as exc:row['candidate_failure']=str(exc)
        synth.append(row)
    clusters={c['case_id']:c['cluster_id'] for c in protocol['cases']};summary={};comparisons={}
    for sampling in base.SAMPLINGS:
        scores={m:{r['case_id']:r['curve_metrics']['gmfe'] for r in records if r['sampling']==sampling and r['method']==m} for m in METHODS}
        summary[sampling]={m:clustered_geomean(v,clusters) for m,v in scores.items()}
        comparisons[sampling]={m:paired_ratio(scores['selective'],scores[m],clusters) for m in METHODS if m!='selective'}
    synthetic_summary={}
    for sampling in base.SAMPLINGS:
        synthetic_summary[sampling]={}
        for scenario in SCENARIOS:
            subset=[r for r in synth if r['sampling']==sampling and r['scenario']==scenario]
            synthetic_summary[sampling][scenario]=summarize_synthetic(subset)
            synthetic_summary[sampling][scenario]['by_log_sigma']={str(sigma):summarize_synthetic(
                [r for r in subset if r['log_sigma']==sigma]) for sigma in sorted({r['log_sigma'] for r in subset})}
    result=dict(version=VERSION,scored_at=time.time(),global_lock_at=state['locked_at'],
        clinical_summary=summary,clinical_comparisons=comparisons,clinical_records=records,
        synthetic_summary=synthetic_summary,synthetic_records=synth,
        resources=dict(profile_calls=calls,failed_calls=failures,
            clinical_predictions=sum(u['prediction_calls'] for u in units.values() if u['job']['kind']=='clinical'),
            synthetic_generator_calls=sum(u['generated']['provenance']['generator_calls'] for u in units.values() if u['job']['kind']=='synthetic'),
            synthetic_transfer_predictions=sum(2 for r in synth if r['candidate_failure'] is None),elapsed_seconds=time.time()-state['started_at']),
        limitations=['Engineering tolerance, not a confidence interval or clinical risk guarantee.',
            'Only previously viewed ten clinical drugs; no independent cohort.',
            'Synthetic repeated tasks are not independent clinical samples; errors conditional on clean truth >peak*1e-8.',
            'Synthetic is a curve/decision recovery pilot; eleven clinical endpoint categories are reported, missing references stay missing.',
            'Clinical candidates are regularized frozen fits; synthetic candidates are unregularized profile solutions. Compare policies only within each arm.'])
    save(RUN/'results.json',result)
    with (RUN/'clinical_endpoints.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Selective blood-partition pilot','',
          'This is an engineering-decision pilot, not evidence of top-tier novelty or clinical-risk validation. Thresholds, grids, and failure rules were frozen before scoring; all ten drugs remain.',
          '', '## Clinical target curve GMFE','', '| Sampling | Always 3 | Always 4, same formula | Necessity only | Selective |','|---|---:|---:|---:|---:|']
    for sampling in base.SAMPLINGS:text.append('| '+sampling+' | '+' | '.join(f'{summary[sampling][m]["estimate"]:.4f}' for m in METHODS)+' |')
    text+=['','The five-point primary rule retains three parameters. Four parameters do not statistically establish true RBC binding. Full curves must also pass the engineering profile-band check.','',
           '## Synthetic mechanism decisions','','| Sampling | Truth | Log-noise SD | Tasks | Selective activated | Necessity activated | Profile status |','|---|---|---:|---:|---:|---:|---|']
    for sampling,ss in synthetic_summary.items():
        for scenario,s in ss.items():
            for sigma,part in s['by_log_sigma'].items():
                text.append(f'| {sampling} | {scenario} | {sigma} | {part["n"]} | {part["activated"]} | {part["necessity_activated"]} | {part["profile_status_counts"]} |')
    text+=['','All 11 clinical endpoint types are in clinical_endpoints.csv, retaining individual predictions, references, and errors. Synthetic results and every failure are in results.json.',
           '','The manuscript PDF and preceding best predictor are unchanged. Synthetic recovery checks are not additional clinical evidence.',
           '', '```bash','runtime/app-env/bin/python -B scripts/run_selective_blood_partition.py status','```']
    (RUN/'README.md').write_text('\n'.join(text)+'\n')
    state.update(status='complete',completed_at=time.time(),results=ref(RUN/'results.json'));save(RUN/'state.json',state)
    return dict(status='complete',clinical=summary,synthetic=synthetic_summary)


def status():
    state=read(RUN/'state.json');protocol=checked(state['protocol'])
    return {k:state.get(k) for k in ('status','pid','started_at','completed_at')}|dict(done=len(state['units']),total=len(protocol['jobs']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run','status','report'));p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if not 1<=a.workers<=14:p.error('workers must be 1..14')
    print({'prepare':prepare,'run':lambda:run(a.workers),'status':status,'report':report}[a.command](),flush=True)
