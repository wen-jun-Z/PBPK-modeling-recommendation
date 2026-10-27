#!/usr/bin/env python3
"""Additional same-five-point PK controls; all predictions lock before scoring."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.optimize import least_squares
import blueprint_reviewer_comparators as old
from report_sampling_expansion import bootstrap, json_native

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/blueprint_design/model_controls_extension_20261011'
SOURCE = old.OUT
METHODS = ('three_compartment', 'one_compartment_cl_prior', 'two_compartment_fixed_cl')
VERSION = 'same-five-point-additional-pk-controls-1'
STARTS, MAX_NFEV, SEED = old.STARTS, old.MAX_NFEV, old.SEED
SOURCES = {
    'official_three_compartment_code': 'https://raw.githubusercontent.com/metrumresearchgroup/mrgsolve/main/inst/models/pk3cmt.cpp',
    'official_three_compartment_release': 'https://mrgsolve.org/blog/posts/2026-new-2-0-1.html',
    'scipy_least_squares': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html',
}


def read(path):
    return old.read(path)


def checked(reference):
    return old.checked(reference)


def ref(path):
    return old.ref(path)


def save(path, document):
    p = old.inside(path)
    if not p.is_relative_to(OUT):
        raise ValueError('Extension outputs must remain in their own project directory')
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(json_native(document), ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(p)


def three_compartment_concentration(parameters, time_h, *, dose_mg, infusion_h, start_h=0.):
    """Exact linear mammillary three-compartment IV concentration.

    Parameters are CL,Vc,Q1,Vp1,Q2,Vp2 in L/h or L. For amounts A,
    A'=M A + IV input. D^(-1/2) M D^(1/2), D=diag(Vc,Vp1,Vp2),
    is symmetric. Its three nonpositive modes give a stable exponential
    convolution, equivalent to mrgsolve pk3cmt with VMAX=0 and no oral input.
    Q2=0 is accepted only to independently check the two-compartment limit.
    """
    p, time_h = np.asarray(parameters, float), np.asarray(time_h, float)
    if (p.shape != (6,) or not np.isfinite(p).all() or min(p[[0,1,3,5]]) <= 0
            or min(p[[2,4]]) < 0 or not np.isfinite(time_h).all()
            or time_h.ndim != 1 or not math.isfinite(dose_mg) or dose_mg <= 0
            or not math.isfinite(infusion_h) or infusion_h < 0
            or not math.isfinite(start_h) or start_h < 0):
        raise ValueError('Valid positive volumes/CL, nonnegative Q, and finite IV inputs required')
    cl,vc,q1,v1,q2,v2 = p
    symmetric = np.array([[-(cl+q1+q2)/vc, q1/math.sqrt(vc*v1), q2/math.sqrt(vc*v2)],
                          [q1/math.sqrt(vc*v1), -q1/v1, 0.],
                          [q2/math.sqrt(vc*v2), 0., -q2/v2]])
    rates, vectors = np.linalg.eigh(symmetric)
    weights = vectors[0]**2
    active_modes = weights > 1e-30
    rates, weights = rates[active_modes], weights[active_modes]
    if np.any(rates > 1e-12):
        raise ArithmeticError('Unstable three-compartment generator')
    rates = np.minimum(rates, 0.)
    elapsed = np.maximum(time_h-start_h, 0.)
    if infusion_h == 0:
        values = dose_mg/vc * (np.exp(elapsed[:,None]*rates[None,:]) @ weights)
        return np.where(time_h >= start_h, values, 0.)
    during, post = np.minimum(elapsed,infusion_h), np.maximum(elapsed-infusion_h,0.)
    integrals = np.empty((len(elapsed),len(rates)))
    nonzero = rates < -1e-15
    integrals[:,nonzero] = np.expm1(during[:,None]*rates[None,nonzero])/rates[None,nonzero]
    integrals[:,~nonzero] = during[:,None]
    values = dose_mg/infusion_h/vc * ((integrals*np.exp(post[:,None]*rates[None,:])) @ weights)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ArithmeticError('Invalid three-compartment concentration')
    return values


def initial_three(training, case_id):
    base, _, _, init = old.initial_parameters(training,'two_compartment',case_id)
    weight = float(training['human']['weight_kg'])
    lower = weight*np.array([.0001,.001,.0001,.001,.0001,.001])
    upper = weight*np.array([20.,30.,50.,100.,50.,100.])
    starts=[]
    for i,row in enumerate(base):
        cl,vc,q,vp=row
        if i < 4:
            # Fixed, different fast/slow splittings; avoid initializing both
            # peripheral rates identically and trapping a degenerate two-mode fit.
            frac_q,frac_v=((.8,.3),(.5,.2),(.2,.5),(.9,.6))[i]
            starts.append([cl,vc,q*frac_q,vp*frac_v,q*(1-frac_q),vp*(1-frac_v)])
        else:
            rng=np.random.default_rng(init['seed']+i)
            starts.append([cl,vc,q*rng.uniform(.15,.9),vp*rng.uniform(.1,.8),
                           q*rng.uniform(.1,.8),vp*rng.uniform(.2,.9)])
    return np.clip(starts,lower*1.001,upper/1.001),lower,upper,init


def fit_visible(training, model, case_id, prior, checkpoint=None):
    if model not in METHODS:
        raise ValueError('Unregistered additional control')
    t,y=old.training_arrays(training)
    if len(t)!=5:
        raise ValueError('The extension requires the exact five-point subset')
    if model=='one_compartment_cl_prior':
        result=old.fit_visible(training,'one_compartment',case_id,prior,checkpoint)
        result.update(model=model,free_parameters=2,prior_L_h=prior,prior_policy='soft_same_source_CL')
        return result
    if model=='three_compartment':
        starts,lower,upper,initialization=initial_three(training,case_id)
        forward=three_compartment_concentration
        expand=lambda x:np.exp(x)
        parameters=('CL','Vc','Q1','Vp1','Q2','Vp2')
    else:
        if prior is None or not math.isfinite(prior) or prior<=0:
            raise ValueError('Fixed CL requires an explicitly qualified positive same-source value')
        original,lo,hi,initialization=old.initial_parameters(training,'two_compartment',case_id)
        starts,lower,upper=original[:,1:],lo[1:],hi[1:]
        forward=old.concentration
        expand=lambda x:np.r_[prior,np.exp(x)]
        parameters=('Vc','Q','Vp')
    kwargs,logs=old.dosing(training),np.log(y)
    attempts,candidates=[],[]
    for i,start in enumerate(starts):
        calls=0
        def residual(x):
            nonlocal calls
            calls+=1
            predicted=forward(expand(x),t,**kwargs)
            if not np.isfinite(predicted).all() or np.any(predicted<=0):
                raise ArithmeticError('Nonpositive/nonfinite concentration on visible fit times')
            return (np.log(predicted)-logs)/math.sqrt(len(t))
        try:
            optimized=least_squares(residual,np.log(start),bounds=(np.log(lower),np.log(upper)),
                max_nfev=MAX_NFEV,ftol=1e-10,xtol=1e-10,gtol=1e-8)
            objective=float(optimized.fun@optimized.fun)
            if not math.isfinite(objective):
                raise ArithmeticError('Nonfinite fit objective')
            row=dict(start=i,objective=objective,parameters=expand(optimized.x).tolist(),
                initial=expand(np.log(start)).tolist(),converged=bool(optimized.success),
                active_bounds=optimized.active_mask.tolist(),nfev=int(optimized.nfev),
                residual_calls=calls,message=str(optimized.message),
                final_visible_jacobian_singular_values=np.linalg.svd(optimized.jac,compute_uv=False).tolist())
            candidates.append(row)
        except (ValueError,FloatingPointError,ArithmeticError,np.linalg.LinAlgError) as exc:
            row=dict(start=i,failed=True,error=str(exc),residual_calls=calls)
        attempts.append(row)
        if checkpoint:
            checkpoint(attempts)
    result=dict(model=model,attempts=attempts,initialization=initialization,
        bounds=dict(lower=lower.tolist(),upper=upper.tolist()),free_parameter_names=parameters,
        free_parameters=len(lower),residual_calls=sum(a['residual_calls'] for a in attempts),
        objective='mean(log(P/O)^2)',prior_L_h=prior if model.endswith('fixed_cl') else None,
        prior_policy='fixed_same_source_CL' if model.endswith('fixed_cl') else 'no_parameter_prior')
    if not candidates:
        return result|dict(status='failed',reason='No finite multistart solution')
    selected=min(candidates,key=lambda a:(a['objective'],a['start']))
    pred=forward(selected['parameters'],t,**kwargs)
    return result|dict(status='fitted',selected=selected,subset_prediction_mg_L=pred.tolist(),
                       subset_log_mse=float(np.mean(np.log(pred/y)**2)))


def transferred(parameters, source_weight, target_weight, model, policy):
    if model!='three_compartment':
        return old.transfer_parameters(parameters,source_weight,target_weight,policy)
    if min(source_weight,target_weight)<=0 or not all(math.isfinite(v) for v in (source_weight,target_weight)):
        raise ValueError('Positive transfer weights required')
    powers=np.ones(6)
    if policy=='allometric_sensitivity':
        powers[[0,2,4]]=.75
    elif policy!='linear_weight':
        raise ValueError('Unknown transfer policy')
    return np.asarray(parameters)*(target_weight/source_weight)**powers


def prepare():
    if (OUT/'protocol.json').exists():
        guard(checked(ref(OUT/'protocol.json')))
        return status()
    if read(SOURCE/'state.json')['status']!='complete':
        raise ValueError('Need completed, frozen original comparator data')
    source=read(SOURCE/'protocol.json')
    old.verify(source['source_pins'])
    pins=dict(source['source_pins'])
    for path in (Path(__file__),ROOT/'tests/test_blueprints/test_extended_model_controls.py',
                 SOURCE/'protocol.json',SOURCE/'state.json',SOURCE/'endpoints.csv',
                 ROOT/'scripts/blueprint_sampling_expansion_data.py'):
        pins[str(path.relative_to(ROOT))]=old.sha(path)
    cases=deepcopy(source['cases'])
    jobs={}
    for cid,case in cases.items():
        cells=[c for c in source['cells'] if c['case_id']==cid and c['info']=='time_five']
        subsets={json.dumps(c['training_source'],sort_keys=True) for c in cells}
        if len(subsets)!=1:
            raise ValueError('Original time-five controls must use identical subset references')
        cell=cells[0]
        training=checked(cell['training_source'])
        if len(old.training_arrays(training)[0])!=5:
            raise ValueError('Exactly five frozen observations required')
        dev=checked(case['development_request'])
        forbidden={r['path'] for r in dev.get('forbidden_sources',[])}|{case['reference_source']['path']}
        prior=old.clearance_prior(dev,forbidden)
        if prior!=case['clearance_prior']:
            raise ValueError('Revalidated same-source clearance differs from the old experiment')
        bases={row.get('basis','unknown') for row in prior.get('evidence',[])}
        case['prior_group']='RF_or_other_predicted' if 'predicted' in bases else 'independent_measured_or_derived'
        case['prior_bases']=sorted(bases)
        for method in METHODS:
            key=cid+'__'+method
            jobs[key]=dict(case_id=cid,model=method,training_source=cell['training_source'],
                development_request=case['development_request'],target_request=case['target_request'],
                target_contract=case['target_contract'],clearance_prior=prior,
                selected_time_h=cell['selected_time_h'],prior_group=case['prior_group'])
    if len(cases)!=30 or len(jobs)!=90:
        raise ValueError('Expected all 30 frozen drugs and 90 planned jobs')
    protocol=dict(version=VERSION,source=ref(SOURCE/'protocol.json'),source_pins=pins,cases=cases,jobs=jobs,
        methods=METHODS,starts=STARTS,max_nfev_per_start=MAX_NFEV,seed=SEED,
        optimizer='SciPy bounded log-parameter TRF, 16 starts, max_nfev=400 per start; ftol=xtol=1e-10,gtol=1e-8',
        cost='Actual residual calls include finite differences; max_nfev is NOT a hard forward-call limit. Different dimensions require different counts.',
        information='Identical frozen time-rule five development points for all seven methods including PBPK. No full development curve or target outcome for fitting.',
        clinical_information='PBPK additionally uses anatomy/drug evidence; CL controls use only exactly shared qualified CL. Not identical full evidence input.',
        sources=SOURCES,main_transfer='linear_weight',sensitivity_transfer='allometric_sensitivity',
        free_parameters=dict(three_compartment=6,one_compartment_cl_prior=2,two_compartment_fixed_cl=3),
        model_interpretation=dict(three_compartment='Classical linear three-compartment IV equations; equivalent numerical implementation of the public mrgsolve structure, not an execution or full study reproduction of mrgsolve.',
          one_compartment_cl_prior='Same one-compartment equations with existing 0.1*(ln(CL/CL_prior)/ln2)^2 engineering penalty.',
          two_compartment_fixed_cl='Same two-compartment equations, qualified construction CL held fixed; only Vc,Q,Vp fitted. Sensitivity control, not proof CL is true.'),
        limitations=['Five points cannot generally identify all six three-compartment parameters; no parameter-mechanism claim.',
          'Converged optimization is not an identifiability certificate; boundaries and all starts retained.',
          'CL may be RF predicted or independently sourced; fixed CL is not patient-specific truth.',
          'Known retrospective 30-drug development cohort, not new blind validation.',
          'Curve-derived NCA references are not original study scalar means; missing references stay null.',
          'All source, population, overlap and tail limitations of the frozen experiment remain.'],
        bootstrap=dict(seed=SEED,resamples=2000),prediction_before_target_scoring=True,
        maximum_workers=8,threads_per_worker=1,monitor_seconds=1200,directory_limit_gib=10,created_at=time.time())
    old.verify(pins)
    save(OUT/'protocol.json',protocol)
    save(OUT/'state.json',dict(status='prepared',protocol=ref(OUT/'protocol.json'),completed={}))
    return status()


def fit_job(key,job,protocol_ref):
    start=time.monotonic()
    path=OUT/'fits'/(key+'.json')
    if path.exists():
        cached=read(path)
        if cached['protocol']!=protocol_ref or cached['job']!=job:
            raise ValueError('Resumed job differs from the frozen protocol')
        if cached['status'] in ('predicted','failed'):
            return ref(path)
    training=checked(job['training_source'])
    dev,target,contract=(checked(job[k]) for k in ('development_request','target_request','target_contract'))
    prior=job['clearance_prior'].get('value_L_h')
    document=dict(protocol=protocol_ref,job=job,target_observations_used=False)
    try:
        if job['model']!='three_compartment' and not job['clearance_prior']['available']:
            raise ValueError('Same-source CL unavailable: '+job['clearance_prior'].get('reason','unknown'))
        fit=fit_visible(training,job['model'],job['case_id'],prior,
            checkpoint=lambda attempts:save(path,document|dict(status='fitting',partial_attempts=attempts)))
        document.update(fit=fit,parameter_locked_at=time.time(),status=fit['status'])
        if fit['status']=='fitted':
            from pbpk_kb.blueprint_policy.perfusion_calibration import prediction_grid
            grid=np.unique(np.r_[prediction_grid(target),contract['sampling_time_h']])
            forward=three_compartment_concentration if job['model']=='three_compartment' else old.concentration
            predictions={}
            for policy in ('linear_weight','allometric_sensitivity'):
                parameters=transferred(fit['selected']['parameters'],dev['human']['weight_kg'],
                                       target['human']['weight_kg'],job['model'],policy)
                curve=forward(parameters,grid,**old.dosing(target))
                if not np.isfinite(curve).all() or np.any(curve<0):
                    raise ArithmeticError('Nonfinite/negative frozen target prediction')
                predictions[policy]=dict(parameters=parameters.tolist(),prediction=dict(time_min=(grid*60).tolist(),
                    concentration=(curve*1000/training['molecular_weight_g_mol']).tolist(),time_unit='min',unit='umol/l'))
            document.update(status='predicted',predictions=predictions,prediction_locked_at=time.time())
        else:
            document['prediction_locked_at']=time.time()
    except Exception as exc:
        document.update(status='failed',prediction_locked_at=time.time(),reason=repr(exc),
                        failure_traceback=traceback.format_exc(),fit=document.get('fit',{}))
    document['wall_seconds']=time.monotonic()-start
    save(path,document)
    return ref(path)


def guard(protocol):
    if (OUT/'PAUSE').exists():
        raise RuntimeError('Extension paused by project PAUSE file')
    old.verify(protocol['source_pins'])
    if sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file() and not f.is_symlink())>10*1024**3:
        raise RuntimeError('Extension outputs exceed 10 GiB')


def run(workers=8):
    state=read(OUT/'state.json');protocol=checked(state['protocol']);guard(protocol)
    if state['status']=='complete':
        return status()
    state.update(status='fitting',pid=os.getpid(),started_at=state.get('started_at',time.time()))
    save(OUT/'state.json',state)
    try:
        last=time.monotonic()
        with ProcessPoolExecutor(max_workers=min(8,max(1,workers))) as pool:
            futures={pool.submit(fit_job,k,j,state['protocol']):k for k,j in protocol['jobs'].items() if k not in state['completed']}
            for future in as_completed(futures):
                key=futures[future]
                state['completed'][key]=future.result()
                save(OUT/'state.json',state)
                print(f'Completed {len(state["completed"])}/{len(protocol["jobs"])}',flush=True)
                if time.monotonic()-last>=1200:
                    guard(protocol);last=time.monotonic()
        guard(protocol)
        if set(state['completed'])!=set(protocol['jobs']):
            raise RuntimeError('Not every planned prediction/failure has completed')
        state.update(status='predictions_locked',predictions_locked_at=time.time())
        save(OUT/'state.json',state)
        return report()
    except BaseException as exc:
        state.update(status='paused',reason=repr(exc));save(OUT/'state.json',state);raise


def write_csv(path,rows):
    p=old.inside(path)
    if not p.is_relative_to(OUT):
        raise ValueError('CSV outside extension directory')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def aggregates(rows,cases):
    groups={
        'all_thirty':set(cases),
        'original_ten':{k for k,c in cases.items() if c['cohort']=='original_ten'},
        'added_twenty':{k for k,c in cases.items() if c['cohort']=='added_twenty'},
        'predicted_CL':{k for k,c in cases.items() if c['prior_group']=='RF_or_other_predicted'},
        'independent_CL':{k for k,c in cases.items() if c['prior_group']=='independent_measured_or_derived'},
    }
    output={}
    for name,ids in groups.items():
        groups_out={}
        for method in ('pbpk',*old.MODELS,*METHODS):
            metrics={}
            for endpoint in old.ENDPOINTS:
                rr=[r for r in rows if r['case_id'] in ids and r['method']==method
                    and r['transfer']=='linear_weight' and r['endpoint']==endpoint]
                values={r['case_id']:float(r['error']) for r in rr if r['error'] not in (None,'')}
                if endpoint=='tmax':
                    result=dict(estimate=float(np.mean(list(values.values()))) if values else None,
                                ci95=None,n_cases=len(values),metric='mean_absolute_error_h')
                else:
                    result=bootstrap(values,{k:cases[k]['cluster_id'] for k in values},seed=SEED,resamples=2000)
                    result.update(metric='drug_equal_geometric_mean_fold_error',within_two=sum(v<=2 for v in values.values()))
                result.update(enrolled=len(ids),n_scored=len(values),missing_reference_or_prediction=len(ids)-len(values))
                metrics[endpoint]=result
            groups_out[method]=metrics
        comparisons={}
        for method in (*old.MODELS,*METHODS):
            comparisons['pbpk/'+method]={}
            for endpoint in ('curve','cmax','auc','c_last','terminal_half_life'):
                def paired_values(m):
                    return {r['case_id']:float(r['error']) for r in rows if r['case_id'] in ids and r['method']==m
                        and r['transfer']=='linear_weight' and r['endpoint']==endpoint and r['error'] not in (None,'')}
                comparisons['pbpk/'+method][endpoint]=old.pair_stats(paired_values('pbpk'),paired_values(method),cases)
        output[name]={'time_five':dict(methods=groups_out,comparisons=comparisons)}
    return output


def report():
    state=read(OUT/'state.json');protocol=checked(state['protocol'])
    if state['status'] not in ('predictions_locked','complete') or set(state['completed'])!=set(protocol['jobs']):
        raise ValueError('Every one of 90 predictions/failures must lock before target outcomes are decoded')
    guard(protocol)
    from blueprint_sampling_expansion_data import score_case
    rows,diagnostics,failures=[],[],[]
    for key in sorted(protocol['jobs']):
        job=protocol['jobs'][key];case=protocol['cases'][job['case_id']]
        doc=checked(state['completed'][key])
        if doc['job']!=job or doc['protocol']!=state['protocol'] or doc['target_observations_used'] or doc['prediction_locked_at']>state['predictions_locked_at']:
            raise ValueError('Locked prediction provenance or target isolation failed')
        if doc['status'] not in ('predicted','failed'):
            raise ValueError('Incomplete prediction artifact')
        if doc['status']=='failed':
            failures.append(dict(job=key,reason=doc.get('reason',doc['fit'].get('reason'))))
        fit=doc['fit'];selected=fit.get('selected',{})
        diagnostics.append(dict(case_id=job['case_id'],drug=case['drug'],method=job['model'],status=doc['status'],
            prior_group=case['prior_group'],free_parameters=fit.get('free_parameters'),
            wall_seconds=doc['wall_seconds'],residual_calls=fit.get('residual_calls'),
            selected_converged=selected.get('converged'),selected_active_bounds=selected.get('active_bounds'),
            converged_starts=sum(bool(a.get('converged')) for a in fit.get('attempts',[])),
            parameters=selected.get('parameters'),development_log_mse=fit.get('subset_log_mse')))
        target=checked(job['target_request'])
        for transfer in ('linear_weight','allometric_sensitivity'):
            score=(score_case(case['source_case'],dict(request=target,prediction=doc['predictions'][transfer]['prediction']))
                   if doc['status']=='predicted' else None)
            for endpoint in old.ENDPOINTS:
                item={} if score is None or endpoint=='curve' else score['endpoints'][endpoint]
                rows.append(dict(case_id=job['case_id'],drug=case['drug'],cohort=case['cohort'],cluster_id=case['cluster_id'],
                    info='time_five',method=job['model'],transfer=transfer,visible_points=5,selected_time_h=job['selected_time_h'],
                    endpoint=endpoint,prediction=item.get('value'),reference=item.get('reference'),
                    error=(score['curve_metrics']['gmfe'] if score else None) if endpoint=='curve' else item.get('error'),
                    unit=item.get('unit'),weak_tail=item.get('weak_tail'),reference_weak_tail=item.get('reference_weak_tail'),
                    status=doc['status'],job=key,prior_group=case['prior_group'],
                    reference_kind=item.get('reference_kind'),reference_is_original_reported_scalar=item.get('reference_is_original_reported_scalar')))
    with (SOURCE/'endpoints.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['info']!='time_five':
                continue
            for name in ('prediction','reference','error','selected_time_h'):
                row[name]=float(row[name]) if row[name] else None
            row['prior_group']=protocol['cases'][row['case_id']]['prior_group']
            rows.append(row)
    write_csv(OUT/'endpoints.csv',rows)
    write_csv(OUT/'diagnostics.csv',diagnostics)
    comparisons={}
    for method in (*old.MODELS,*METHODS):
        comparisons[method]={}
        for endpoint in ('curve','cmax','auc','c_last','terminal_half_life'):
            def values(m):
                return {r['case_id']:float(r['error']) for r in rows if r['method']==m and r['endpoint']==endpoint
                        and r['transfer']=='linear_weight' and r['error'] not in (None,'')}
            comparisons[method][endpoint]=old.pair_stats(values('pbpk'),values(method),protocol['cases'])
    result=dict(status='complete',protocol=state['protocol'],prediction_lock=state['predictions_locked_at'],
        scored_at=time.time(),summary=aggregates(rows,protocol['cases']),paired_pbpk_over_control=comparisons,
        failures=failures,diagnostics=diagnostics,methods=['pbpk',*old.MODELS,*METHODS],
        new_fits=len(protocol['jobs']),drugs=len(protocol['cases']),source_urls=SOURCES,
        endpoints=ref(OUT/'endpoints.csv'),limitations=protocol['limitations'])
    save(OUT/'results.json',result)
    state.update(status='complete',completed_at=time.time(),results=ref(OUT/'results.json'))
    save(OUT/'state.json',state)
    return status()


def status():
    if not (OUT/'state.json').exists():
        return dict(status='not_prepared')
    state=read(OUT/'state.json');p=checked(state['protocol'])
    return dict(status=state['status'],completed=len(state['completed']),total=len(p['jobs']),output=str(OUT),reason=state.get('reason'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','status','report'))
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.action=='status':
        result=status()
    else:
        with (OUT/'batch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=prepare() if args.action=='prepare' else run(args.workers) if args.action=='run' else report()
    print(json.dumps(json_native(result),indent=2))


if __name__=='__main__':
    main()
