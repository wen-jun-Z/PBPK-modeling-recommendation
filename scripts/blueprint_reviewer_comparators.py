#!/usr/bin/env python3
"""Same-observation classical PK controls and frozen-result reviewer sensitivity.

prepare/run never decode a target observation. Each fit sees only its exact four
or five development rows. report opens target outcomes only after every planned
prediction (or documented fitting failure) has been locked. Old code is immutable.
"""
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

import numpy as np
from scipy.optimize import least_squares
import run_classical_pk_comparator as analytic
from report_sampling_expansion import bootstrap, json_native

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'experiments/blueprint_design/sampling_stability_20261009'
OUT = ROOT / 'experiments/blueprint_design/reviewer_revision_20261009/comparators'
VERSION = 'same-visible-observations-classical-controls-1'
SEED, STARTS, MAX_NFEV = 20261009, 16, 400
INFO = {'four': 'no_addition_repaired', 'time_five': 'time_repaired', 'learned_five': 'stable_repaired'}
MODELS = ('one_compartment', 'two_compartment', 'two_compartment_cl_prior')
ENDPOINTS = ('curve', 'auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life', 'auc_inf', 'clearance', 'vz', 'mrt', 'vss')
TOL = 1e-10


def inside(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Only this project may be accessed')
    return p


def read(path):
    return json.loads(inside(path).read_text())


def sha(path):
    return hashlib.sha256(inside(path).read_bytes()).hexdigest()


def ref(path):
    p = inside(path)
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p))


def checked(reference):
    if sha(reference['path']) != reference['sha256']:
        raise ValueError('Source checksum changed: '+reference['path'])
    return read(reference['path'])


def save(path, document):
    p = inside(path)
    if not p.is_relative_to(OUT):
        raise ValueError('Comparator outputs must stay in their directory')
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(json_native(document), ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(p)


def pins_in(value, pins):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            name = str(inside(value['path']).relative_to(ROOT))
            if name in pins and pins[name] != value['sha256']:
                raise ValueError('Conflicting source checksum')
            pins[name] = value['sha256']
        for v in value.values():
            pins_in(v, pins)
    elif isinstance(value, list):
        for v in value:
            pins_in(v, pins)


def verify(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed: '+path)


def dosing(document):
    d = document['dosing']
    result = dict(dose_mg=float(d['dose_mg']), infusion_h=float(d['infusion_minutes'])/60,
                  start_h=float(d.get('start_h', 0.)))
    if d['route'] != 'intravenous' or not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Finite IV dose is required')
    if result['dose_mg'] <= 0 or min(result['infusion_h'], result['start_h']) < 0:
        raise ValueError('Invalid IV bolus/infusion')
    return result


def concentration(parameters, time_h, *, dose_mg, infusion_h, start_h=0.):
    """Exact one/two-compartment central concentration, including true bolus."""
    p, t = np.asarray(parameters, float), np.asarray(time_h, float)-start_h
    if len(p) not in (2, 4) or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Positive CL,V or CL,Vc,Q,Vp required')
    if not np.isfinite(t).all() or infusion_h < 0 or dose_mg <= 0 or start_h < 0:
        raise ValueError('Invalid concentration inputs')
    active = t >= 0
    elapsed = np.maximum(t, 0.)
    if len(p) == 4:
        if infusion_h > 0:
            return analytic.concentration_mg_l(p, time_h, dose_mg=dose_mg,
                                                infusion_h=infusion_h, start_h=start_h)
        alpha, beta, a, b = analytic.macro_parameters(p)
        return np.where(active, dose_mg/p[1]*(a*np.exp(-alpha*elapsed)+b*np.exp(-beta*elapsed)), 0.)
    k = p[0]/p[1]
    if infusion_h == 0:
        return np.where(active, dose_mg/p[1]*np.exp(-k*elapsed), 0.)
    during = np.minimum(elapsed, infusion_h)
    return dose_mg/infusion_h/p[0]*(-np.expm1(-k*during))*np.exp(-k*np.maximum(elapsed-infusion_h, 0.))


def transfer_parameters(parameters, source_weight, target_weight, policy='linear_weight'):
    if not all(math.isfinite(x) and x > 0 for x in (source_weight, target_weight)):
        raise ValueError('Positive source and target weights required')
    p = np.asarray(parameters, float)
    powers = np.ones(len(p))
    if policy == 'allometric_sensitivity':
        powers[0] = .75
        if len(p) == 4:
            powers[2] = .75
    elif policy != 'linear_weight':
        raise ValueError('Unregistered transfer policy')
    return p*(target_weight/source_weight)**powers


def clearance_prior(request, forbidden):
    """Share the exact construction input, never a target-derived CL label."""
    e = {r['parameter']: r for r in request['evidence']}
    if 'total_clearance' in e:
        rows = [e['total_clearance']]
    elif {'hepatic_clearance', 'renal_clearance'} <= e.keys() and e.get('elimination_scope', {}).get('value') == 'hepatic_and_renal':
        rows = [e['hepatic_clearance'], e['renal_clearance']]
    else:
        return dict(available=False, reason='No explicitly scoped construction CL')
    for r in rows:
        if (r['unit'] != 'ml/min/kg' or r.get('compound_id') != request['compound_id']
            or any(r.get('context',{}).get(k)!=v for k,v in dict(species='human', matrix='plasma', route='intravenous').items())
            or not math.isfinite(float(r['value'])) or float(r['value']) < 0
            or r.get('source', {}).get('path') in forbidden):
            return dict(available=False, reason='Invalid, mismatched, or forbidden CL input')
        checked(r['source'])
    value = sum(float(r['value']) for r in rows)*float(request['human']['weight_kg'])*.06
    if value <= 0:
        return dict(available=False, reason='Nonpositive CL input')
    return dict(available=True, value_L_h=value, evidence=rows,
        conversion='ml/min/kg * development body weight kg * 0.06',
        role='Same independent or RF construction input as PBPK; not necessarily clinically measured',
        penalty='0.1 * (ln(CL/CL_prior)/ln(2))^2', weight=.1)


def training_arrays(training):
    if training.get('role') != 'calibration_data_reconstruction' or training['time_unit'] != 'min' or training['unit'] != 'umol/l':
        raise ValueError('Explicit development subset with min/umol/l is required')
    if len(training['observations']) not in (4, 5):
        raise ValueError('This frozen comparison only permits four/five visible points')
    t = np.array([r['time_min']/60 for r in training['observations']], float)
    c = np.array([r['concentration_umol_per_l']*training['molecular_weight_g_mol']/1000 for r in training['observations']], float)
    if not np.isfinite(t).all() or not np.isfinite(c).all() or (c <= 0).any() or not (np.diff(t) > 0).all() or min(t) <= dosing(training)['start_h']:
        raise ValueError('Finite, positive, ordered post-dose subset required')
    return t, c


def initial_parameters(training, model, case_id):
    """No full curve, stored fit, or target outcomes are arguments to this function."""
    times, observed = training_arrays(training)
    kw, weight = dosing(training), float(training['human']['weight_kg'])
    area = float(np.trapezoid(np.r_[0., observed], np.r_[kw['start_h'], times]))
    slope = float(np.polyfit(times[-3:], np.log(observed[-3:]), 1)[0])
    if slope < 0:
        area += observed[-1]/-slope
    cl, volume = kw['dose_mg']/area, kw['dose_mg']/max(observed)
    seed = SEED + int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    if model == 'one_compartment':
        lower, upper = weight*np.array([.0001, .001]), weight*np.array([20., 100.])
        starts = [[cl, volume], [cl*.5, volume*.5], [cl*2, volume*2], weight*np.array([.2, .3])]
        center = np.array([cl, volume])
    else:
        lower, upper = weight*np.array([.0001, .001, .0001, .001]), weight*np.array([20., 30., 50., 100.])
        starts = [[cl, volume*.15, cl, volume], [cl*.5, volume*.5, cl*2, volume*2],
                  [cl*2, volume*.2, cl*.5, volume*5], weight*np.array([.2, .08, .5, .3])]
        center = np.array([cl, volume*.25, cl, volume])
    starts.extend(center*np.exp(rng.uniform(np.log(.08), np.log(12.), size=(STARTS-4, len(center)))))
    return np.clip(starts, lower*1.001, upper/1.001), lower, upper, dict(
        seed=seed, observed_times_h=times.tolist(), observations=len(times),
        rough_subset_auc_mg_h_L=area, rough_subset_log_slope=slope,
        full_development_read=False, target_observations_used=False, old_fit_used=False)


def fit_visible(training, model, case_id, prior=None, checkpoint=None):
    times, observed = training_arrays(training)
    starts, lower, upper, init = initial_parameters(training, model, case_id)
    kw, logs = dosing(training), np.log(observed)
    attempts, candidates = [], []
    for i, start in enumerate(starts):
        calls = 0
        def residual(x):
            nonlocal calls
            calls += 1
            p = concentration(np.exp(x), times, **kw)
            vector = (np.log(np.maximum(p, np.finfo(float).tiny))-logs)/math.sqrt(len(times))
            if prior is not None:
                vector = np.r_[vector, math.sqrt(.1)*(x[0]-math.log(prior))/math.log(2)]
            return vector
        try:
            result = least_squares(residual, np.log(start), bounds=(np.log(lower), np.log(upper)),
                max_nfev=MAX_NFEV, ftol=1e-10, xtol=1e-10, gtol=1e-8)
            objective = float(result.fun@result.fun)
            p = np.exp(result.x)
            if not math.isfinite(objective) or not np.isfinite(p).all():
                raise ArithmeticError('Nonfinite fitted parameters')
            a = dict(start=i, objective=objective, parameters=p.tolist(), initial=start.tolist(),
                converged=bool(result.success), active_bounds=result.active_mask.tolist(), nfev=result.nfev,
                residual_calls=calls, message=result.message)
            candidates.append(a)
        except (ValueError, FloatingPointError, ArithmeticError) as exc:
            a = dict(start=i, failed=True, error=str(exc), residual_calls=calls)
        attempts.append(a)
        if checkpoint:
            checkpoint(attempts)
    fit = dict(attempts=attempts, initialization=init, bounds=dict(lower=lower.tolist(), upper=upper.tolist()),
        residual_calls=sum(a['residual_calls'] for a in attempts), model=model,
        objective='mean(log(P/O)^2)' + (' + 0.1*(ln(CL/CL_prior)/ln(2))^2' if prior else ''))
    if not candidates:
        return fit | dict(status='failed', reason='No finite multistart solution')
    selected = min(candidates, key=lambda a:(a['objective'], a['start']))
    pred = concentration(selected['parameters'], times, **kw)
    return fit | dict(status='fitted', selected=selected, subset_prediction_mg_L=pred.tolist(),
        subset_log_mse=float(np.mean(np.log(pred/observed)**2)))


def prepare():
    if (OUT/'protocol.json').exists():
        verify(checked(ref(OUT/'protocol.json'))['source_pins'])
        return status()
    source_state = read(OLD/'state.json')
    if source_state['status'] != 'complete':
        raise ValueError('Only completed frozen source is allowed')
    old = checked(source_state['protocol'])
    verify(old['source_pins'])
    choices = checked(old['choices'])['choices']
    pins = dict(old['source_pins'])
    for p in (Path(__file__), Path(analytic.__file__), ROOT/'scripts/blueprint_sampling_expansion_data.py',
              ROOT/'scripts/report_sampling_expansion.py', ROOT/'scripts/blueprint_gap_design_data.py',
              ROOT/'src/pbpk_kb/blueprints/clinical_prediction_nca.py',
              ROOT/'src/pbpk_kb/blueprint_policy/perfusion_calibration.py', OLD/'state.json', OLD/'protocol.json', OLD/'endpoints.csv'):
        pins[str(p.relative_to(ROOT))] = sha(p)
    cases, jobs, cells = {}, {}, []
    originals = set(old['original_case_ids'])
    for c in old['cases']:
        cid = c['case_id']
        dev, target = (checked(c['requests']['evidence'][phase]) for phase in ('development', 'target'))
        # Metadata contract already excludes all concentration and scalar reference values.
        contract = checked(c['target_contract'])
        forbidden = {r['path'] for r in dev.get('forbidden_sources', [])} | {c['reference_source']['path']}
        prior = clearance_prior(dev, forbidden)
        pins_in(c, pins); pins_in(dev, pins); pins_in(target, pins)
        cases[cid] = dict(case_id=cid, drug=c['drug'], cohort='original_ten' if cid in originals else 'added_twenty',
            cluster_id=c['cluster_id'], target_contract=c['target_contract'], target_request=c['requests']['evidence']['target'],
            development_request=c['requests']['evidence']['development'], reference_source=c['reference_source'],
            approximations=c.get('approximations', []), sensitivity_flags=c.get('sensitivity_flags', []),
            cross_study_pair=c.get('cross_study_pair', True), source_case=c, clearance_prior=prior,
            demographic_differences={k:[dev['human'].get(k),target['human'].get(k)] for k in dev['human'] if dev['human'].get(k)!=target['human'].get(k)})
        for info, method in INFO.items():
            selection = choices[cid+'__late_four']['cmax']['sources'][method]
            artifact = checked(selection['source'])
            training_ref = artifact['training']; training = checked(training_ref)
            t, y = training_arrays(training)
            if training['human'] != dev['human'] or training['dosing'] != dev['dosing']:
                raise ValueError('Subset/development conditions differ')
            if len(t) != (4 if info == 'four' else 5):
                raise ValueError('Visible point count differs')
            base = cases[cid].get('four_rows')
            if info == 'four':
                cases[cid]['four_rows'] = training['observations']
            elif any(r not in training['observations'] for r in base):
                raise ValueError('Five point subset must preserve the exact four visible rows')
            pins_in(training_ref, pins)
            subset_id = hashlib.sha256(json.dumps(training['observations'], sort_keys=True).encode()).hexdigest()[:12]
            for model in MODELS:
                key = f'{cid}__{subset_id}__{model}'
                cell = dict(case_id=cid, info=info, method=model, job=key, visible_points=len(t),
                    selected_time_h=selection['time_h'], training_source=training_ref)
                if model.endswith('_cl_prior') and not prior['available']:
                    cell.update(job=None, unavailable_reason=prior['reason'])
                else:
                    jobs[key] = dict(case_id=cid, model=model, training_source=training_ref,
                        # No case outcomes/full development source reaches a worker.
                        target_contract=c['target_contract'], development_request=c['requests']['evidence']['development'],
                        target_request=c['requests']['evidence']['target'], clearance_prior=prior)
                cells.append(cell)
    protocol = dict(version=VERSION, source=ref(OLD/'protocol.json'), source_state=ref(OLD/'state.json'),
        source_pins=pins, cases=cases, jobs=jobs, cells=cells, info=INFO, starts=STARTS, max_nfev=MAX_NFEV,
        seed=SEED, created_at=time.time(), purpose='Same 30 drugs, late-four layout, Cmax request; model controls use IDENTICAL visible concentrations and fifth-point choices',
        main_transfer='CL,Vc,Q,Vp multiplied by target/development weight: common per-kg CL convention of the frozen PBPK. This does not reproduce PBPK age/sex physiological organs.',
        transfer_sensitivity='Same fitted parameters; CL,Q weight exponent 0.75 and volumes 1.0; no refitting or target-value access.',
        primary='Time-rule fifth point; compare one/two-compartment vs repaired PBPK at equal information. Four-point baseline reported separately.',
        supplemental='Frozen learned fifth point; only different subsets require additional fitting.',
        prior='Two-compartment + same valid construction CL, separately labelled engineering regularization. Current PBPK has no CL prior penalty. RF prior is not clinical observation.',
        model_information='Classical models use concentration/IV dosing/body weight; PBPK additionally uses native human anatomy and independent/RF drug evidence. Not identical mechanism/evidence input.',
        bootstrap=dict(seed=SEED, resamples=2000), prediction_before_target_scoring=True,
        maximum_workers=4, threads_per_worker=1, monitor_seconds=1200, directory_limit_gib=10,
        limitations=['Retrospective, all outcomes historically viewed; no new blinded validation.',
                    'Four/five observations do not identify unique physiological parameters.',
                    'Same-study different dose arms are preserved and flagged, not called external studies.',
                    'No guarantee equal optimization FLOPs: same start/evaluation caps only; actual calls/times recorded.',
                    'All 11 output categories retained; missing observed references remain null.'])
    verify(pins); save(OUT/'protocol.json', protocol)
    save(OUT/'state.json', dict(status='prepared', protocol=ref(OUT/'protocol.json'), completed={}))
    return status()


def fit_job(key, job, protocol_ref):
    started = time.monotonic(); dest = OUT/'fits'/(key+'.json')
    if dest.exists():
        old = read(dest)
        if old['protocol'] != protocol_ref:
            raise ValueError('Resume protocol differs')
        if old['status'] in ('predicted', 'failed'):
            return ref(dest)
    training = checked(job['training_source'])
    dev, target, contract = (checked(job[k]) for k in ('development_request','target_request','target_contract'))
    prior = job['clearance_prior']['value_L_h'] if job['model'].endswith('_cl_prior') else None
    fit = fit_visible(training, job['model'], job['case_id'], prior,
        checkpoint=lambda attempts:save(dest, dict(status='fitting', protocol=protocol_ref, job=job,
            completed_starts=len(attempts), partial_attempts=attempts)))
    document = dict(protocol=protocol_ref, job=job, fit=fit, target_observations_used=False,
                    parameter_locked_at=time.time(), status=fit['status'])
    if fit['status']=='fitted':
        # A dense display grid plus target protocol times ensures exact concentration matching.
        from pbpk_kb.blueprint_policy.perfusion_calibration import prediction_grid
        grid = np.unique(np.r_[prediction_grid(target), contract['sampling_time_h']])
        predictions={}
        for policy in ('linear_weight','allometric_sensitivity'):
            parameters=transfer_parameters(fit['selected']['parameters'], dev['human']['weight_kg'],target['human']['weight_kg'],policy)
            curve=concentration(parameters,grid,**dosing(target))
            if not np.isfinite(curve).all() or (curve<0).any():
                raise ValueError('Nonfinite/negative target prediction')
            predictions[policy]=dict(parameters=parameters.tolist(), prediction=dict(time_min=(grid*60).tolist(),
                concentration=(curve*1000/training['molecular_weight_g_mol']).tolist(),time_unit='min',unit='umol/l'))
        document.update(status='predicted',predictions=predictions, prediction_locked_at=time.time())
    document['wall_seconds']=time.monotonic()-started
    save(dest,document)
    return ref(dest)


def guard(p):
    if (OUT/'PAUSE').exists():
        raise RuntimeError('Comparator batch paused')
    verify(p['source_pins'])
    size=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file() and not f.is_symlink())
    if size>10*1024**3:
        raise RuntimeError('Comparator outputs exceeded10GiB')


def run(workers=4):
    state=read(OUT/'state.json'); p=checked(state['protocol']);guard(p)
    if state['status']=='complete':return status()
    state.update(status='fitting',pid=os.getpid(),started_at=state.get('started_at',time.time()));save(OUT/'state.json',state)
    try:
        last=time.monotonic()
        with ProcessPoolExecutor(max_workers=min(4,max(1,workers))) as pool:
            futures={pool.submit(fit_job,k,j,state['protocol']):k for k,j in p['jobs'].items() if k not in state['completed']}
            for f in as_completed(futures):
                key=futures[f];state['completed'][key]=f.result();save(OUT/'state.json',state)
                print(f'Completed {len(state["completed"])}/{len(p["jobs"])}',flush=True)
                if time.monotonic()-last>=1200:
                    guard(p);last=time.monotonic()
        guard(p)
        state.update(status='predictions_locked',predictions_locked_at=time.time());save(OUT/'state.json',state)
        return report()
    except BaseException as exc:
        state.update(status='paused',reason=repr(exc));save(OUT/'state.json',state);raise


def csv_save(path, rows):
    p=inside(path)
    if not p.is_relative_to(OUT):raise ValueError('CSV outside comparator directory')
    with p.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def source_rows():
    with (OLD/'endpoints.csv').open(newline='') as h:
        return [r for r in csv.DictReader(h) if r['regime']=='late_four' and r['requested_goal']=='cmax' and r['method'] in INFO.values()]


def pair_stats(a,b,cases):
    ids=sorted(a.keys()&b.keys())
    ratios={k:a[k]/b[k] for k in ids}
    out=bootstrap(ratios,{k:cases[k]['cluster_id'] for k in ids},seed=SEED,resamples=2000)
    out.update(improved=sum(v<1-TOL for v in ratios.values()),worse=sum(v>1+TOL for v in ratios.values()),
               tied=sum(abs(v-1)<=TOL for v in ratios.values()),per_case_ratio=ratios)
    return out


def summary(rows,p):
    cases=p['cases']; result={}
    for cohort in ('all_thirty','added_twenty','original_ten'):
        ids={k for k,c in cases.items() if cohort=='all_thirty' or c['cohort']==cohort}
        group={}
        for info in INFO:
            per={}
            for model in ('pbpk',*MODELS):
                per[model]={}
                for ep in ('cmax','curve','auc','c_last','terminal_half_life'):
                    rr=[r for r in rows if r['case_id'] in ids and r['info']==info and r['method']==model and r['transfer']=='linear_weight' and r['endpoint']==ep]
                    values={r['case_id']:float(r['error']) for r in rr if r['error'] is not None}
                    stat=bootstrap(values,{k:cases[k]['cluster_id'] for k in values},seed=SEED,resamples=2000)
                    stat.update(enrolled=len(ids),within_two=sum(v<=2 for v in values.values()),failed_or_missing=len(ids)-len(values))
                    per[model][ep]=stat
            comparisons={}
            for model in MODELS:
                comparisons['pbpk/'+model]={}
                for ep in ('cmax','curve'):
                    def vals(m):return {r['case_id']:float(r['error']) for r in rows if r['case_id'] in ids and r['info']==info and r['method']==m and r['transfer']=='linear_weight' and r['endpoint']==ep and r['error'] is not None}
                    comparisons['pbpk/'+model][ep]=pair_stats(vals('pbpk'),vals(model),cases)
            group[info]=dict(methods=per,comparisons=comparisons)
        result[cohort]=group
    return result


def frozen_sensitivity(p, reference_tail_flags=None):
    """Result-level deletions do NOT retrain the selection jackknife models."""
    rr=source_rows(); cases=p['cases']; records=[]; out={}
    values={(r['case_id'],r['method'],r['endpoint']):float(r['error']) for r in rr if r['error']!=''}
    for cohort in ('all_thirty','added_twenty'):
        ids={k for k,c in cases.items() if cohort=='all_thirty' or c['cohort']=='added_twenty'}
        groups={'all':ids,'without_Montelukast':{k for k in ids if cases[k]['drug']!='Montelukast'},
                'cross_study_only':{k for k in ids if cases[k]['cross_study_pair']}}
        # Flag definitions derive solely from frozen metadata, never prediction error.
        flagsets={
          'without_source_approximations':('serum','arterial','microdose','patent','registry_identifier'),
          'without_potential_overlap':('overlap','within_study'),
          'without_population_or_aggregation_transfer':('surgical','oncology','myocardial','individual','group_mean','coadministration','male_to_female'),
        }
        for name,tokens in flagsets.items():
            groups[name]={k for k in ids if not any(token in flag.lower() for flag in cases[k]['sensitivity_flags'] for token in tokens)}
        # Common observed-curve tail quality, not prediction error.
        if reference_tail_flags is not None:
            groups['without_weak_reference_tail']={k for k in ids if not reference_tail_flags.get(k, True)}
        groups.update({'without_study:'+cluster:{k for k in ids if cases[k]['cluster_id']!=cluster}
                       for cluster in sorted({cases[k]['cluster_id'] for k in ids})})
        out[cohort]={}
        for label,keep in groups.items():
            bucket={}
            for endpoint in ('cmax','curve','auc','c_last','terminal_half_life'):
                methods={m:{k:values[k,m,endpoint] for k in keep if (k,m,endpoint) in values} for m in INFO.values()}
                b={m:bootstrap(v,{k:cases[k]['cluster_id'] for k in v},seed=SEED,resamples=2000) for m,v in methods.items()}
                for comparator in ('no_addition_repaired','time_repaired'):
                    pair=pair_stats(methods['stable_repaired'],methods[comparator],cases)
                    b['stable_repaired/'+comparator]=pair
                    records.append(dict(cohort=cohort,subset=label,endpoint=endpoint,comparison='stable_repaired/'+comparator,
                        n=pair['n_cases'],clusters=pair['n_clusters'],gmfe_ratio=pair['estimate'],
                        ci_lower=pair['ci95'][0] if pair['ci95'] else None,ci_upper=pair['ci95'][1] if pair['ci95'] else None,
                        improved=pair['improved'],worse=pair['worse'],tied=pair['tied'],excluded=';'.join(sorted(cases[k]['drug'] for k in ids-keep))))
                bucket[endpoint]=b
            out[cohort][label]=bucket
    csv_save(OUT/'sensitivity.csv',records)
    return dict(groups=out,definition='Frozen prediction leave-study-out, no retraining; drug-equal GMFE and paired study-cluster bootstrap2000.',
        flag_rules=flagsets,source_metadata_flags={k:c['sensitivity_flags'] for k,c in cases.items()},
        caution='No inference that unflagged sources are verified; flags reflect original metadata completeness.')


def report():
    state=read(OUT/'state.json');p=checked(state['protocol']);guard(p)
    if state['status'] not in ('predictions_locked','complete') or set(state['completed'])!=set(p['jobs']):
        raise ValueError('Every prediction/failure must lock before target outcomes are decoded')
    from blueprint_sampling_expansion_data import score_case
    rows=[];scores={};failures=[];reference_tail_flags={}
    for key,reference in state['completed'].items():
        doc=checked(reference);job=p['jobs'][key];case=p['cases'][job['case_id']]
        if doc['status']!='predicted':
            failures.append(dict(job=key,reason=doc['fit'].get('reason')));continue
        if doc['prediction_locked_at']>state['predictions_locked_at'] or doc['target_observations_used']:
            raise ValueError('Prediction lock or outcome isolation failed')
        target=checked(job['target_request'])
        scores[key]={}
        for transfer,variant in doc['predictions'].items():
            scores[key][transfer]=score_case(case['source_case'],dict(request=target,prediction=variant['prediction']))
            reference_tail_flags[job['case_id']]=scores[key][transfer]['endpoints']['terminal_half_life']['reference_weak_tail']
    for cell in p['cells']:
        case=p['cases'][cell['case_id']]
        for transfer in ('linear_weight','allometric_sensitivity'):
            score=scores.get(cell['job'],{}).get(transfer)
            for endpoint in ENDPOINTS:
                e={} if score is None or endpoint=='curve' else score['endpoints'][endpoint]
                rows.append(dict(case_id=cell['case_id'],drug=case['drug'],cohort=case['cohort'],cluster_id=case['cluster_id'],
                    info=cell['info'],method=cell['method'],transfer=transfer,visible_points=cell['visible_points'],
                    selected_time_h=cell['selected_time_h'],endpoint=endpoint,prediction=e.get('value'),reference=e.get('reference'),
                    error=(score['curve_metrics']['gmfe'] if score else None) if endpoint=='curve' else e.get('error'),
                    unit=e.get('unit'),weak_tail=e.get('weak_tail'),reference_weak_tail=e.get('reference_weak_tail'),
                    status='predicted' if score else 'failed_or_unavailable',job=cell['job']))
    # Only completed, frozen PBPK results are brought into the equal-information table.
    for r in source_rows():
        rows.append(dict(case_id=r['case_id'],drug=r['drug'],cohort=r['cohort'],cluster_id=r['cluster_id'],
            info=next(k for k,v in INFO.items() if v==r['method']),method='pbpk',transfer='linear_weight',
            visible_points=4 if r['method']=='no_addition_repaired' else 5,
            selected_time_h=float(r['selected_time_h']) if r['selected_time_h'] else None,endpoint=r['endpoint'],
            prediction=float(r['prediction']) if r['prediction'] else None,reference=float(r['reference']) if r['reference'] else None,
            error=float(r['error']) if r['error'] else None,unit=r['unit'],weak_tail=r['weak_tail'] or None,
            reference_weak_tail=reference_tail_flags.get(r['case_id']) if r['endpoint']=='terminal_half_life' else None,status='frozen_predicted',job='frozen:'+r['method']))
    csv_save(OUT/'endpoints.csv',rows)
    sens=frozen_sensitivity(p,reference_tail_flags)
    save(OUT/'sensitivity.json',sens)
    output=dict(protocol=state['protocol'],status='complete',predictions_locked_at=state['predictions_locked_at'],scored_at=time.time(),
        endpoints=ref(OUT/'endpoints.csv'),sensitivity=ref(OUT/'sensitivity.json'),summary=summary(rows,p),failures=failures,
        unique_fits=len(p['jobs']),comparison_cells=len(p['cells']),counts=dict(drugs=len(p['cases']),clinical_source_clusters=len({c['cluster_id'] for c in p['cases'].values()})),
        diagnostics={key:dict(wall_seconds=checked(reference)['wall_seconds'],fit_calls=checked(reference)['fit']['residual_calls']) for key,reference in state['completed'].items()},
        limitations=p['limitations'])
    save(OUT/'results.json',output)
    state.update(status='complete',results=ref(OUT/'results.json'),completed_at=time.time());save(OUT/'state.json',state)
    return status()


def status():
    if not (OUT/'state.json').exists():return dict(status='not_prepared')
    s=read(OUT/'state.json');p=checked(s['protocol'])
    return dict(status=s['status'],completed=len(s['completed']),total=len(p['jobs']),cells=len(p['cells']),output=str(OUT),reason=s.get('reason'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('prepare','run','status','report'));ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.action=='status':result=status()
    else:
        with (OUT/'batch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=prepare() if a.action=='prepare' else run(a.workers) if a.action=='run' else report()
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
