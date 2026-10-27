#!/usr/bin/env python3
"""Versioned reviewer analyses; frozen source experiments are strictly read-only.

The strict NCA layer is an additional sensitivity definition, not a correction
of the deliberately descriptive historical sampled-NCA protocol. Profiles are
finite data-compatible sets, not posterior draws or confidence intervals.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import run_sampling_stability as frozen
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.blueprint_policy import multiaxis_evidence_profile as profiles
from pbpk_kb.blueprint_policy import visible_partition_feasibility as repair
from summarize_additional_clinical_baselines import prediction_in_hours

RUN = ROOT / 'experiments/blueprint_design/reviewer_revision_20261009/diagnostics'
SOURCE = frozen.RUN
VERSION = 'reviewer-diagnostics-and-rf-ablation-1'
QUALITY = dict(version='strict-nca-sensitivity-1', min_sampled_points=3,
    min_dense_points=4, min_adjusted_r_squared=.98, min_span_half_lives=1.,
    max_auc_extrapolated_fraction=.2, max_aumc_extrapolated_fraction=.2,
    interpretation='Additional strict sensitivity; original sampled NCA values and flags retained unchanged')
AXES = ('clearance_scale', 'partition_scale', 'bidirectional_exchange_scale', 'blood_partition_scale')
read, checked = frozen.read, frozen.checked


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT): raise ValueError('Project boundary')
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p))


def save(path, value):
    p = Path(path).resolve()
    if not p.is_relative_to(RUN): raise ValueError('Reviewer diagnostic output boundary')
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def compact(value):
    """Retain audit scalars and window endpoints without duplicating dense arrays."""
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()
                if k not in ('fit_times_h', 'sampled_time_h', 'sampled_concentration_ng_ml', 'terminal_candidates', 'model_full_exposure_nca')}
    if isinstance(value, list): return [compact(v) for v in value]
    return value


def strict_quality(diagnostics, *, dense=False):
    fit = diagnostics.get('terminal_fit') if dense else diagnostics.get('selected_tail')
    fit = fit or {}
    n = fit.get('fit_points', fit.get('n', 0))
    r2 = fit.get('adjusted_r_squared')
    span = fit.get('span_half_lives', diagnostics.get('terminal_span_in_half_lives'))
    reasons = []
    if n < QUALITY['min_dense_points' if dense else 'min_sampled_points']: reasons.append('insufficient_terminal_points')
    if r2 is None or r2 < QUALITY['min_adjusted_r_squared']: reasons.append('adjusted_r_squared_below_0.98_or_missing')
    if span is None or span < QUALITY['min_span_half_lives']: reasons.append('span_below_one_half_life_or_missing')
    full = fit if dense else diagnostics.get('model_full_exposure_nca', diagnostics)
    auc = full.get('auc_extrapolated_fraction')
    aumc = full.get('aumc_extrapolated_fraction')
    infinite_reasons = reasons + ([] if auc is not None and auc <= .2 else ['auc_extrapolation_above_20pct_or_missing_initial_exposure'])
    moment_reasons = infinite_reasons + ([] if aumc is not None and aumc <= .2 else ['aumc_extrapolation_above_20pct_or_missing'])
    return dict(policy=QUALITY['version'], terminal_half_life_qualified=not reasons,
        terminal_reasons=reasons, infinity_qualified=not infinite_reasons, infinity_reasons=infinite_reasons,
        moment_qualified=not moment_reasons, moment_reasons=moment_reasons,
        n=n, adjusted_r_squared=r2, span_half_lives=span,
        auc_extrapolated_fraction=auc, aumc_extrapolated_fraction=aumc)


def dense_summary(native, request):
    mw = next(e['value'] for e in request['evidence'] if e['parameter'] == 'molecular_weight')
    t, y, _ = prediction_in_hours(native, mw)
    clean = dict(time_min=[float(x*60) for x in t], concentration=[float(x/mw) for x in y], time_unit='min', unit='umol/l')
    return summarize_extended_prediction(clean, request), np.asarray(t), np.asarray(y)


def source_bundle():
    s = read(SOURCE / 'state.json'); p = checked(s['protocol'])
    if s['status'] != 'complete': raise ValueError('Frozen source must be complete')
    ch = checked(p['choices'])['choices']
    return s, p, checked(p['source_protocol']), ch


def load_fits(unit):
    if isinstance(unit['fits'], dict): return checked(unit['fits'])
    return [dict(formula=f['formula'], fit=checked(f['fit'])) for f in unit['fits']]


def summarize_fit(unit):
    rows = []
    for item in load_fits(unit):
        f = item['fit']; lo=np.asarray(f['parameter_lower']); hi=np.asarray(f['parameter_upper'])
        b=f['best']; x=np.asarray(b['log_parameters']) if b else None
        position=(x-lo)/(hi-lo) if x is not None else None
        cl=f['clearance_input']; nr=cl['total_clearance_l_h']-(cl['renal_clearance_l_h'] or 0.)
        rows.append(dict(formula=item['formula'], selected=item['formula']==unit['formula'],
            best=b, bounds_log=dict(lower=lo.tolist(), upper=hi.tolist()),
            bounds_physical_multiplier=dict(lower=np.exp(lo).tolist(), upper=np.exp(hi).tolist()),
            normalized_boundary_distance=None if position is None else np.minimum(position,1-position).tolist(),
            near_boundary_1pct=None if position is None else (np.minimum(position,1-position)<=.01).tolist(),
            initial_clearance=cl, final_total_clearance_l_h=None if b is None else
                (cl['renal_clearance_l_h'] or 0.)+nr*b['parameters']['clearance_scale'],
            starts=f['starts'], calls=f['calls'], failures=f['failures']))
    return rows


def rf_data():
    """Recover predictions with saved forests and frozen row identities; no training."""
    import joblib
    from pbpk_kb.blueprints import pk_parameter_learning as learning
    wheel=ROOT/'artifacts/pk_predictors/pksmart/pksmart-3.0.1-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as z: payload=z.read(learning.CSV_MEMBER)
    raw=list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))
    batches={}
    for name, folder in [('original_ten', ROOT/'experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug/run'),
                         ('added_twenty',ROOT/'experiments/blueprint_design/sampling_expansion_20261009/inputs/priors')]:
        split=read(folder/'parameter_split.json')
        if hashlib.sha256(payload).hexdigest()!=split['source']['csv_sha256']:raise ValueError('Parameter source differs')
        rows={r['index']:r for r in split['rows']}; med={}
        for ep in learning.ENDPOINTS:
            vals=[learning._label(raw[i][ep],ep)[0] for i in split['train_indices']]
            vals=[v for v in vals if v is not None]
            med[ep]=float(10**np.median(vals))
        if name=='original_ten':
            document=read(folder/'summary.json');metrics=document['parameter_metrics']; predictions=document['parameter_test_predictions']
        else:
            document=read(folder/'predictions.json');metrics=document['held_out_parameter_metrics']
            models=joblib.load(folder/'parameter_models.joblib');predictions=[]
            inds=split['test_indices'];features=learning._features([rows[i]['identity_smiles'] for i in inds])
            for ep,model in models.items():
                model.n_jobs=1; yp=model.predict(features)
                for i,estimate in zip(inds,yp):
                    truth=learning._label(raw[i][ep],ep)[0]
                    if truth is not None:predictions.append(dict(row_index=i,endpoint=ep,identity_smiles=rows[i]['identity_smiles'],
                        unit=learning.ENDPOINTS[ep]['unit'],observed=float(10**truth),random_forest=float(10**estimate),
                        training_median=med[ep],fold_error=float(10**abs(estimate-truth))))
        for ep in learning.ENDPOINTS:
            rr=[r for r in predictions if r['endpoint']==ep]
            if rr:
                gm=math.exp(sum(math.log(r['fold_error']) for r in rr)/len(rr))
                if not math.isclose(gm,metrics[ep]['random_forest']['gmfe'],rel_tol=1e-10):raise ValueError('RF heldout reconstruction differs')
                if any(not math.isclose(r['training_median'],med[ep],rel_tol=1e-12) for r in rr):raise ValueError('Median differs')
        batches[name]=dict(split=ref(folder/'parameter_split.json'),train_count=split['train_count'],
            test_count=split['test_count'],scaffold_overlap=split['scaffold_overlap'],
            medians=med,metrics=metrics,predictions=predictions,trained_again=False)
    return dict(version=VERSION,source=ref(wheel),batches=batches)


def prepare():
    if (RUN/'protocol.json').exists():return status()
    s,p,basep,ch=source_bundle(); frozen.verify(p['source_pins'])
    rf=rf_data();save(RUN/'rf_heldout.json',rf)
    tasks={}; caseids=[]
    for c in p['cases']:
        cid=c['case_id'];caseids.append(cid);orig=cid in p['original_case_ids']
        choices=ch[cid+'__late_four']['cmax']['sources']; seen={}
        for method in ('time_repaired','stable_repaired'):
            uid=choices[method]['unit_id'];u=s['units'][uid]['result']
            if uid in seen:tasks[seen[uid]]['methods'].append(method);continue
            key=cid+'__'+method;seen[uid]=key
            tasks[key]=dict(case_id=cid,kind='profile',methods=[method],source=u,selected_time_h=choices[method]['time_h'])
    protocol=dict(version=VERSION,created_at=time.time(),source=ref(SOURCE/'protocol.json'),source_state=ref(SOURCE/'state.json'),
        source_pins=deepcopy(p['source_pins']),cases=p['cases'],original_case_ids=p['original_case_ids'],
        source_protocol=p['source_protocol'],profile_tasks=tasks,rf_heldout=ref(RUN/'rf_heldout.json'),quality=QUALITY,
        profile=dict(grid_points=7,max_calls_per_formula=65536,tolerance_log=math.log(1.1),
            retention='Visible log-concentration MSE <= best converged profile MSE + log(1.1)^2, both formulas; no posterior claim'),
        fit=dict(starts=16,budget_per_formula=16384),workers_maximum=6,monitor_seconds=1200,directory_limit_gib=10,
        same_target_observations='Only post-lock score; profile acceptance uses visible development values only',
        retrospective=True,strict_quality_does_not_replace_original_flags=True)
    protocol['source_pins'][str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    save(RUN/'protocol.json',protocol);save(RUN/'state.json',dict(status='prepared',profile_done={},rf_done={},native_done={}))
    return status()


class Guard:
    def __init__(self,p):self.p,self.last=p,0.
    def __call__(self,force=False):
        if (RUN/'PAUSE').exists():raise RuntimeError('Review diagnostics paused')
        if force or time.monotonic()-self.last>=1200:
            frozen.verify(self.p['source_pins'])
            if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())>10*1024**3:raise RuntimeError('Diagnostics exceeds 10 GiB')
            self.last=time.monotonic()


def profile_task(key, job):
    p=read(RUN/'protocol.json');guard=Guard(p);guard(force=True)
    path=RUN/'profiles'/key/'result.json'
    if path.exists():return ref(path)
    start=time.monotonic();basep=checked(p['source_protocol']);case=frozen.gap.get_case(basep,job['case_id'])
    unit=checked(job['source']);training=checked(unit['training']);all_profiles={};phys={};requests={}
    for formula in frozen.base.FORMULAS:
        req=frozen.gap.request(case,'development',formula)
        ph=frozen.prior.get_phys(basep,case,'development',formula,req)
        audit=repair.feasibility(ph,req,training)
        lo,hi,_,_,_=repair.initialization(req,ph,'blood_partition',16,audit)
        ts,obs=frozen.old.optimizer._visible(req,training)
        fp=path.parent/(formula+'.json')
        if fp.exists():pr=read(fp)
        else:
            pr=profiles.analyze(lambda x,t:frozen.old.simulate('blood_partition',ph,req,t,x),ts,obs,lo,hi,
                grid_points=7,max_calls=65536,before_trial=guard);save(fp,pr)
        all_profiles[formula]=pr;requests[formula]=req;phys[formula]=ph
    successes=[r['mse'] for f in all_profiles.values() for r in f['profile'] if r['successful']]
    if not successes:raise ValueError('No converged five-point profile')
    threshold=min(successes)+math.log(1.1)**2;members=[];curves=[];grid=None
    for formula,pr in all_profiles.items():
        target=frozen.gap.request(case,'target',formula)
        tph=frozen.prior.get_phys(basep,case,'target',formula,target)
        grid=frozen.base.engine.prediction_grid(target)
        contract=checked(case['target_contract'])
        for idx,row in frozen.gap.profile_members(pr,threshold):
            guard();x=row['best']['log_parameters'];pred=frozen.old.simulate('blood_partition',tph,target,grid,x)
            native=frozen.prior.curve_document(pred,grid,target)
            matched=frozen.data.prediction_nca(contract,native,target)
            dense,_,_=dense_summary(native,target)
            members.append(dict(formula=formula,index=idx,mse=row['mse'],parameters=np.exp(x).tolist(),
                log_parameters=x,matched_endpoints={ep:e['value'] for ep,e in matched['endpoints'].items()},
                dense_endpoints={ep:compact(e) for ep,e in dense['endpoints'].items() if ep!='plasma_concentration_time'},
                matched_quality=strict_quality(matched['diagnostics']),dense_quality=strict_quality(dense['diagnostics'],dense=True)))
            curves.append(pred['concentration_mg_l'])
    if not members:raise ValueError('No retained five-point profile')
    matrix=np.asarray(curves);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.parent/'target_curves.npz',time_h=grid,concentrations_mg_l=matrix,
        minimum_mg_l=matrix.min(axis=0),maximum_mg_l=matrix.max(axis=0),median_mg_l=np.median(matrix,axis=0))
    pars=np.asarray([r['parameters'] for r in members])
    out=dict(job=job,version=VERSION,visible_points=len(training['observations']),source_training=unit['training'],
        original_best_parameters=unit['parameters'],fits=summarize_fit(unit),threshold=threshold,
        members=members,parameter_min=pars.min(axis=0).tolist(),parameter_max=pars.max(axis=0).tolist(),
        curves=ref(path.parent/'target_curves.npz'),profiles={k:ref(path.parent/(k+'.json')) for k in all_profiles},
        calls=sum(v['calls'] for v in all_profiles.values()),wall_seconds=time.monotonic()-start,
        target_values_used=False,confidence_region=False)
    save(path,out);return ref(path)


def status():
    p=read(RUN/'protocol.json');s=read(RUN/'state.json')
    return dict(status=s['status'],profiles_done=len(s['profile_done']),profiles_total=len(p['profile_tasks']),
        rf_done=len(s['rf_done']),native_done=len(s['native_done']),reason=s.get('reason'))


def run_profiles(workers):
    if not 1<=workers<=6:raise ValueError('Workers must be 1..6')
    p=read(RUN/'protocol.json');Guard(p)(force=True);s=read(RUN/'state.json')
    s.update(status='profiles_running',pid=os.getpid());save(RUN/'state.json',s)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(profile_task,k,j):k for k,j in p['profile_tasks'].items() if k not in s['profile_done']}
        for f in as_completed(fs):
            key=fs[f];s['profile_done'][key]=f.result();save(RUN/'state.json',s)
            print(json.dumps(dict(kind='five_point_profile',done=len(s['profile_done']),total=len(p['profile_tasks']),key=key)),flush=True)
    s.update(status='profiles_locked',profiles_locked_at=time.time());save(RUN/'state.json',s)
    return status()


def diagnostics():
    s,p,basep,ch=source_bundle();curves=[];parameter_rows=[];special={};strict=[]
    for c in p['cases']:
        cid=c['case_id'];seen=set()
        for method in ('time_repaired','stable_repaired'):
            choice=ch[cid+'__late_four']['cmax']['sources'][method];u=s['units'][choice['unit_id']]['result']
            if u['path'] in seen:continue
            seen.add(u['path']);unit=checked(u);req=checked(unit['request']);native=checked(unit['prediction'])
            score=frozen.data.score_case(c,unit);dense,t,y=dense_summary(native,req)
            d=dense['diagnostics'];fit=d.get('terminal_fit') or {};lo=fit.get('requested_fit_start_h');hi=fit.get('requested_fit_end_h')
            tail=(t>=lo)&(t<=hi) if lo is not None else np.zeros(len(t),dtype=bool)
            contract=checked(c['target_contract']);cutoff=contract['cutoff_ng_ml']
            row=dict(case_id=cid,drug=c['drug'],method=method,source=u,
                dose=req['dosing'],clinical_contract=contract,original_endpoints=score['endpoints'],
                matched=compact(score['nca_diagnostics']),observed=compact(score['reference_nca_diagnostics']),
                dense=compact(d),dense_endpoints={k:compact(v) for k,v in dense['endpoints'].items() if k!='plasma_concentration_time'},
                dense_extension=dense['extension'],strict_matched=strict_quality(score['nca_diagnostics']),
                strict_observed=strict_quality(score['reference_nca_diagnostics']),strict_dense=strict_quality(d,dense=True),
                dense_tail_concentration_ng_ml=[float(y[tail][0]),float(y[tail][-1])] if tail.any() else None,
                dense_tail_below_clinical_lloq_fraction=float(np.mean(y[tail]<cutoff)) if cutoff>0 and tail.any() else None)
            curves.append(row);parameter_rows.append(dict(case_id=cid,drug=c['drug'],method=method,source=u,fits=summarize_fit(unit)))
            q=row['strict_matched']['terminal_half_life_qualified'] and row['strict_observed']['terminal_half_life_qualified']
            strict.append(dict(case_id=cid,drug=c['drug'],method=method,eligible=q,
                original_half_life=score['endpoints']['terminal_half_life'],
                predicted_reasons=row['strict_matched']['terminal_reasons'],reference_reasons=row['strict_observed']['terminal_reasons']))
            if c['drug'] in ('Remdesivir','Sufentanil'):
                special[c['drug']+'__'+method]=dict(time_h=t.tolist(),prediction_ng_ml=y.tolist(),
                    observed_time_h=score['reference_nca_diagnostics']['sampled_time_h'],
                    observed_concentration_ng_ml=score['reference_nca_diagnostics']['sampled_concentration_ng_ml'],
                    dense_tail_window=d['terminal_window_h'],clinical_lloq_ng_ml=cutoff,
                    observed_tail=score['reference_nca_diagnostics']['selected_tail'],matched_tail=score['nca_diagnostics']['selected_tail'])
    records=read(ROOT/'experiments/blueprint_design/sampling_expansion_20261009/results.json')['records']
    audit=[]
    for r in records:
        if r['case_id'] not in p['original_case_ids']:continue
        pq=strict_quality(r['nca_diagnostics']);oq=strict_quality(r['reference_nca_diagnostics'])
        audit.append(dict(case_id=r['case_id'],design_key=r['design_key'],candidate=r['candidate'],
            original_value=r['endpoints']['terminal_half_life'],strict_prediction=pq,strict_reference=oq,
            label_eligible=pq['terminal_half_life_qualified'] and oq['terminal_half_life_qualified']))
    save(RUN/'nca_quality.json',dict(version=VERSION,quality=QUALITY,cases=curves,strict_half_life=strict,
        original_half_life_training_label_audit=audit,parameters=parameter_rows,
        frozen_values_modified=False,new_quality_is_sensitivity_not_historical_protocol=True))
    save(RUN/'tail_case_curves.json',dict(version=VERSION,cases=special))
    return dict(cases=len(curves),strict_half_life_pairs=sum(r['eligible'] for r in strict),
        historical_training_labels=len(audit),qualified_training_labels=sum(r['label_eligible'] for r in audit))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','profiles','diagnostics','status'))
    ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    out=prepare() if a.command=='prepare' else run_profiles(a.workers) if a.command=='profiles' else diagnostics() if a.command=='diagnostics' else status()
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
