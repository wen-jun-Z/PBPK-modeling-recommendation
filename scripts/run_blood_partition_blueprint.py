#!/usr/bin/env python3
"""Frozen ten-drug test of calibrating predicted RBC partition, with a 4D control."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import run_exposure_blueprint as base
from pbpk_kb.blueprint_policy import adaptive_exchange_model, blood_partition_calibration
from pbpk_kb.blueprint_policy import distribution_calibration_optimizer as optimizer

ROOT = base.ROOT
RUN = ROOT / 'experiments/blueprint_design/blood_partition_20261008'
VERSION = 'predicted-blood-partition-pilot-1'
METHODS = ('blood_partition', 'exchange_shape')
ALL_METHODS = METHODS + ('direct_relaxed', 'two_compartment_reference')
BUDGET = 16384
STARTS = 16
read, save, ref, checked, verify = base.read, base.save, base.ref, base.checked, base.verify


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'protocol.json').exists():
        raise ValueError('Protocol already frozen; do not overwrite')
    prior = read(base.RUN / 'protocol.json')
    prior_state = read(base.RUN / 'state.json')
    checked(prior_state['protocol'])
    verify(prior['source_pins'])
    verify(prior_state['output_pins'])
    if prior_state['status'] != 'complete' or len(prior['cases']) != 10:
        raise ValueError('Require the completed ten-drug development baseline')
    pins = deepcopy(prior['source_pins'])
    caches = deepcopy(prior['physiology_cache'])
    # Read only already exported, uncalibrated physiology. No old fit initializes a new fit.
    for path in (base.RUN / 'native_cache').glob('*/physiology/perfusion_physiology.json'):
        caches[path.parent.parent.name] = ref(path)
    for reference in caches.values():
        physiology = checked(reference)
        base.pins_in(reference, pins)
        receipt=(ROOT/reference['path']).parent.parent/'base/completion.json'
        if not receipt.exists():raise ValueError('Cached physiology lacks its native request receipt')
        base.pins_in(ref(receipt),pins)
        for kind in ('source_pins', 'artifact_pins'):
            verify(physiology.get(kind, {}))
            pins.update(physiology.get(kind, {}))
    controls, jobs = {}, {}
    for case in prior['cases']:
        cid = case['case_id']
        for sampling in base.SAMPLINGS:
            for method in METHODS:
                key = f'{cid}__{sampling}__{method}'
                jobs[key] = dict(case_id=cid, sampling=sampling, method=method,
                                 training_source=case['sampling_sources'][sampling])
            for method in ALL_METHODS[2:]:
                key = f'{cid}__{sampling}__{method}'
                source = prior_state['units'] if method == 'direct_relaxed' else prior['controls']
                controls[key] = deepcopy(source[key])
                controls[key]['reused'] = True
                base.pins_in(controls[key], pins)
    paths = [Path(__file__), base.RUN / 'protocol.json', base.RUN / 'state.json',
             ROOT / 'scripts/run_exposure_blueprint.py']
    paths += [ROOT / 'src/pbpk_kb/blueprint_policy' / name for name in (
        'blood_partition_calibration.py', 'distribution_calibration_optimizer.py',
        'adaptive_exchange_model.py', 'robust_exchange_model.py')]
    for path in paths:
        pins[str(path.relative_to(ROOT))] = base.sha256(path)
    protocol = dict(version=VERSION, created_at=time.time(), cases=prior['cases'], jobs=jobs,
        controls=controls, physiology_cache=caches, source_pins=pins,
        methods=list(METHODS), comparison_methods=list(ALL_METHODS),
        formulas=list(base.FORMULAS), sampling=list(base.SAMPLINGS),
        hypothesis='Native-predicted RBC partition can create a vascular capacity floor that tissue-only scaling cannot correct. Calibrate it only when no measured blood-partition evidence is present.',
        mechanism='RBC Kp multiplied by exp(x4); update B/P, all vascular capacities and blood/plasma observation consistently; preserve real volumes, blood flows, fu and renal input.',
        bounds={'RBC_partition_scale': [.05, 20.], 'exchange_shape_alpha': [-1., 1.]},
        parameter_counts={'blood_partition': 4, 'exchange_shape': 4, 'direct_relaxed': 3},
        extra_axis_penalty='0.01*(log(RBCscale)/log(10))^2 or 0.01*alpha^2; fixed engineering regularization, not calibrated biological prior',
        objective_calls_per_formula=BUDGET, starts_per_formula=STARTS,
        initialization='Same first three-axis independent starts as direct_relaxed; fourth axis independent fixed seed; no development or target fitted warm start',
        selection='Within-method formula chosen only by visible development penalized log loss; no target-driven drug-specific method choice',
        reuse='20 direct_relaxed and 20 two-compartment frozen predictions; neither counts as new independent evidence',
        evaluation='All ten previously viewed development drugs, full and five points separately; all predictions lock before target outcomes are opened',
        endpoints=11, primary='Drug-equally-weighted target curve GMFE, paired by study cluster',
        uncertainty='2000 paired study-cluster bootstrap draws, seed 20260929; Ertapenem and Ceftriaxone share a cluster',
        acceptance='Adopt only if both full/five cohort GMFE decrease versus direct_relaxed, at least 6/10 drugs improve in each, and no increased failure count. Superiority of the RBC mechanism additionally needs evidence versus the 4-parameter control; passing adoption is development evidence, not publication/clinical proof.',
        stress_test='For each selected development formula, hold fitted values fixed and perturb only predicted RBC Kp by 0.5x/2x. Report target-trajectory drift on request grid, without reading target observations. This is sensitivity, not missing-CL or clinical uncertainty.',
        preflight={'target_observations_read': False, 'uses_old_development_start_for_diagnosis_only': True,
                   'total_simulations': 572, 'development_jacobians_and_center': 140,
                   'Treprostinil_gradient': 6, 'Treprostinil_fitting_calls': 234,
                   'Remdesivir_fitting_calls': 190, 'old_development_metric_replays': 2,
                   'native_component_compatibility_checks_no_propagation':60,
                   'diagnostic_calls_separate_from_formal_budget': True},
        monitor_seconds=1200, directory_limit_gib=10,
        limitations=['A drug-specific development curve is required; no new zero-curve performance claim.',
                     'RBC fitted from plasma alone is an effective parameter, not identified cellular binding.',
                     'Five observations for four parameters may overfit; boundary and Jacobian diagnostics retained.',
                     'Earlier source ambiguities and native runtime-version mismatch remain unchanged; no clinical-risk claim.'],
        method_reference='https://docs.open-systems-pharmacology.org/working-with-pk-sim/pk-sim-documentation/pk-sim-compounds-definition-and-work-flow')
    verify(pins)
    save(RUN / 'protocol.json', protocol)
    save(RUN / 'state.json', dict(status='prepared', version=VERSION,
                                protocol=ref(RUN / 'protocol.json'), units={}, started_at=None))
    return dict(status='prepared', new_units=len(jobs), reused_controls=len(controls))


class Guard:
    def __init__(self, protocol):
        self.protocol = protocol
        self.last = 0.

    def __call__(self, force=False):
        if (RUN / 'PAUSE').exists():
            raise RuntimeError('Batch paused')
        if not force and time.monotonic() - self.last < 1200:
            return
        try:
            verify(self.protocol['source_pins'])
        except ValueError as exc:
            raise RuntimeError('Source-integrity guard stopped the batch: '+str(exc)) from exc
        size = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and not p.is_symlink())
        if size > 10 * 1024**3:
            raise RuntimeError('Batch exceeds 10 GiB')
        self.last = time.monotonic()


def physiology(protocol, case, phase, formula, request, guard):
    key = f'{case["case_id"]}__{phase}__{formula}'
    if key in protocol['physiology_cache']:
        reference=protocol['physiology_cache'][key]
        receipt=read((ROOT/reference['path']).parent.parent/'base/completion.json')
        if base.normalized(receipt['request'])!=base.normalized(request):
            raise ValueError('Cached native request differs from this case/condition/formula')
        return checked(reference), reference, 0
    folder = RUN / 'native_cache' / key
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.export.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = folder / 'physiology/perfusion_physiology.json'
        existed = path.exists()
        try:
            exported = base.engine._physiology(request, folder, ROOT, guard)
        except ValueError as exc:
            if str(exc) not in {'Incomplete native construction evidence','Bounded native export failed; inspect retained process log'}:raise
            raise base.NativeBuildFailed(str(exc)) from exc
        return exported, ref(path), 0 if existed else 1


def simulate(method, physiology, request, times, parameters):
    if method == 'blood_partition':
        return blood_partition_calibration.simulate(physiology, request, times, parameters)
    if method == 'direct_relaxed':
        return base.engine._simulate(physiology, request, times, 'flow_exchange',
                                     base.engine.PARAMETERS['flow_exchange'], parameters)
    cl = base.engine.clearance_inputs(request)
    return adaptive_exchange_model.simulate(physiology, times_h=times,
        dose_mg=request['dosing']['dose_mg'], infusion_duration_h=request['dosing']['infusion_minutes']/60.,
        total_clearance_l_h=cl['total_clearance_l_h'], renal_clearance_l_h=cl['renal_clearance_l_h'],
        family='adaptive_exchange', log_parameters=dict(zip(
            adaptive_exchange_model.PARAMETERS['adaptive_exchange'], parameters)), observation='auto')


def diagnostic(method, phys, req, training, parameters, guard):
    times = np.asarray([r['time_min']/60. for r in training['observations']])
    x = np.asarray(parameters, float)
    calls=0
    try:
        guard();calls+=1
        center = np.asarray(simulate(method, phys, req, times, x)['concentration_mg_l'])
        columns = []
        for i in range(len(x)):
            z = x.copy(); z[i] += 1e-4
            guard();calls+=1
            p = np.asarray(simulate(method, phys, req, times, z)['concentration_mg_l'])
            columns.append((np.log(p)-np.log(center))/1e-4)
        s = np.linalg.svd(np.array(columns).T, compute_uv=False)
        return dict(status='completed',calls=calls, data_jacobian_singular_values=s.tolist(),
                    condition_number=float(s[0]/s[-1]) if s[-1] > 0 else None,
                    scope='Local finite-difference diagnostic, not global identifiability or clinical uncertainty')
    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
        return dict(status='failed',calls=calls,reason=str(exc))


def partition_sensitivity(method, phys, req, parameters, guard):
    """Two predeclared input perturbations; no clinical outcomes or refitting."""
    grid = base.engine.prediction_grid(req)
    times = np.geomspace(float(grid[grid > 0][0]), float(grid[-1]), 61)
    calls = 0

    def evaluate(factor):
        nonlocal calls
        guard()
        calls += 1
        if method in ('blood_partition','direct_relaxed'):
            x = list(parameters) if method=='blood_partition' else list(parameters)+[0.]
            x[3] += math.log(factor)
            result = blood_partition_calibration.simulate(phys,req,times,x)
        else:
            derived = blood_partition_calibration.derive_physiology(phys,req,math.log(factor))
            result = simulate(method,derived,req,times,parameters)
        return np.asarray(result['concentration_mg_l'],float)

    try:
        center = evaluate(1.)
        mask = np.isfinite(center) & (center > float(np.max(center))*1e-8)
        if not mask.any(): raise ValueError('No usable positive sensitivity-grid concentration')
        scenarios=[]
        for factor in (.5,2.):
            changed=evaluate(factor)
            if not np.isfinite(changed[mask]).all() or np.any(changed[mask]<=0):
                raise ValueError('Invalid sensitivity prediction')
            errors=np.abs(np.log(changed[mask])-np.log(center[mask]))
            scenarios.append(dict(rbc_input_factor=factor,trajectory_gmfe=math.exp(float(np.mean(errors))),
                                  maximum_point_fold=math.exp(float(np.max(errors)))))
        return dict(status='completed',calls=calls,points=int(mask.sum()),scenarios=scenarios,
                    worst_trajectory_gmfe=max(s['trajectory_gmfe'] for s in scenarios),
                    interpretation='Conditional RBC sensitivity, not clinical error, missing-CL stability or a confidence interval')
    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
        return dict(status='failed',calls=calls,reason=str(exc))


def execute(protocol, key, job):
    guard = Guard(protocol); guard(force=True)
    folder = RUN / 'work' / key; folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'result.json'
    protocol_sha = base.sha256(RUN / 'protocol.json')
    if output.exists():
        previous = read(output)
        if previous['protocol_sha256'] != protocol_sha or any(previous.get(k)!=v for k,v in job.items()):
            raise ValueError('Changed resumed protocol')
        verify(previous['output_pins'])
        return previous
    case = next(c for c in protocol['cases'] if c['case_id'] == job['case_id'])
    dev = base.normalized(checked(case['requests']['evidence']['development']))
    target = base.normalized(checked(case['requests']['evidence']['target']))
    training = base._training_document(job['training_source'], dev, ROOT)
    if training['human'] != dev['human'] or training['dosing'] != dev['dosing']:
        raise ValueError('Development condition mismatch')
    if job['sampling'] == 'five' and len(training['observations']) != 5:
        raise ValueError('Five-point isolation failed')
    unit = dict(**job, protocol_sha256=protocol_sha, status='fitting', formula_scores=[],
                resources=dict(objective_calls=0, failed_calls=0, new_native=0, diagnostic_calls=0))
    start = time.monotonic(); best = None
    for formula in base.FORMULAS:
        req = deepcopy(dev); req['partition_method'] = formula
        phys, phys_ref, cost = physiology(protocol, case, 'development', formula, req, guard)
        path = folder / (formula+'.json')
        if path.exists():
            fitted = read(path)
            if (fitted['protocol_sha256'] != protocol_sha or fitted['request'] != req
                    or fitted['training_source'] != job['training_source']
                    or fitted.get('development_physiology')!=phys_ref
                    or fitted.get('variant')!=job['method']
                    or fitted.get('call_limit')!=BUDGET or fitted.get('calls',BUDGET+1)>BUDGET
                    or len(fitted.get('starts',[]))!=STARTS or not fitted.get('no_warm_start')):
                raise ValueError('Changed resumed formula inputs')
        else:
            predict = lambda x, t: simulate(job['method'], phys, req, t, x)
            fitted = optimizer.fit(predict, req, training, phys, variant=job['method'],
                                   budget=BUDGET, starts=STARTS, before_trial=guard)
            fitted.update(protocol_sha256=protocol_sha, request=req, training_source=job['training_source'],
                          development_physiology=phys_ref)
            save(path, fitted)
        unit['resources']['objective_calls'] += fitted['calls']
        unit['resources']['failed_calls'] += fitted['failures']
        unit['resources']['new_native'] += cost
        row = dict(formula=formula, fit=ref(path), best=fitted['best'], calls=fitted['calls'], failures=fitted['failures'])
        unit['formula_scores'].append(row)
        if fitted['best'] is not None and (best is None or fitted['best']['objective'] < best['best']['objective']):
            best = row
    if best is None:
        unit.update(status='failed', reason='No eligible development solution')
    else:
        formula = best['formula']; fitted = checked(best['fit']); x = best['best']['log_parameters']
        target['partition_method'] = formula
        save(folder / 'request.json', target)
        save(folder / 'selection_locked.json', dict(formula=formula, parameters=best['best'],
              basis='development_only', locked_at=time.time(), parameter_count=4))
        try:
            phys, phys_ref, cost = physiology(protocol, case, 'target', formula, target, guard)
        except base.NativeBuildFailed as exc:
            unit.update(status='failed',reason=str(exc),failure_stage='native_target_construction')
            phys=None
        if phys is not None:
            unit['resources']['new_native'] += cost
            unit['target_physiology'] = phys_ref
        times = base.engine.prediction_grid(target)
        try:
            if phys is None:raise base.NativeBuildFailed(unit['reason'])
            unit['resources']['prediction_calls'] = 1
            prediction = simulate(job['method'], phys, target, times, x)
            mw = next(float(e['value']) for e in target['evidence'] if e['parameter'] == 'molecular_weight')
            curve = dict(time_min=(times*60).tolist(), concentration=(np.asarray(prediction['concentration_mg_l'])*1000/mw).tolist(),
                         time_unit='min', unit='umol/l', observation=prediction['observation'])
            save(folder / 'prediction_locked.json', curve)
            save(folder / 'endpoints.json', base.summarize_extended_prediction(curve, target))
            unit.update(status='predicted', prediction=ref(folder / 'prediction_locked.json'),
                endpoints=ref(folder / 'endpoints.json'), request=ref(folder / 'request.json'),
                selection=ref(folder / 'selection_locked.json'), formula=formula, parameters=best['best'],
                mass_balance=prediction['mass_balance'], effective_total_clearance_l_h=prediction['effective_total_clearance_l_h'],
                blood_partition_mapping=prediction.get('blood_partition_audit'), target_observations_used=False,
                parameters_fitted_to_development_only=True)
            development_phys = checked(fitted['development_physiology'])
            d = diagnostic(job['method'], development_phys, fitted['request'], training, x, guard)
            unit['development_diagnostic'] = d
            unit['resources']['diagnostic_calls'] = d['calls']
            unit['partition_sensitivity'] = partition_sensitivity(job['method'],phys,target,x,guard)
            unit['resources']['sensitivity_calls'] = unit['partition_sensitivity']['calls']
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            unit.update(status='failed', reason=str(exc), failure_stage=unit.get('failure_stage','target_prediction'))
    unit['resources']['wall_seconds'] = time.monotonic()-start
    unit['output_pins'] = {}
    base.pins_in({k:v for k,v in unit.items() if k!='output_pins'}, unit['output_pins'])
    guard(force=True)
    save(output, unit)
    return unit


def run(workers):
    state = read(RUN / 'state.json'); protocol = checked(state['protocol'])
    if state['status'] == 'complete': return status()
    Guard(protocol)(force=True)
    state.update(status='running', pid=os.getpid(), started_at=state.get('started_at') or time.time())
    save(RUN / 'state.json', state)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, protocol, k, j): k for k,j in protocol['jobs'].items() if k not in state['units']}
        try:
            for future in as_completed(futures):
                key = futures[future]; state['units'][key] = future.result()
                save(RUN / 'state.json', state)
                print(dict(done=len(state['units']), total=len(protocol['jobs']), unit=key), flush=True)
        except Exception as exc:
            save(RUN / 'PAUSE', dict(reason=str(exc)))
            for future in futures: future.cancel()
            state.update(status='paused', reason=str(exc)); save(RUN / 'state.json', state)
            raise
    Guard(protocol)(force=True)
    # Reused 3D baseline gets exactly the same outcome-free perturbation audit.
    sensitivity_controls={}
    guard=Guard(protocol)
    for key, unit in protocol['controls'].items():
        if unit['method']!='direct_relaxed' or unit['status']!='predicted':continue
        case=next(c for c in protocol['cases'] if c['case_id']==unit['case_id'])
        req=checked(unit['request'])
        phys,_,_=physiology(protocol,case,'target',unit['formula'],req,guard)
        sensitivity_controls[key]=partition_sensitivity('direct_relaxed',phys,req,unit['parameters']['log_parameters'],guard)
    state['sensitivity_controls']=sensitivity_controls
    state.update(status='predictions_locked', locked_at=time.time(), output_pins={})
    for unit in state['units'].values(): state['output_pins'].update(unit['output_pins'])
    save(RUN / 'state.json', state)
    return report()


def report():
    from report_blueprint_validation import Store, _case_reference, _score_prediction, clustered_geomean, paired_ratio
    state = read(RUN / 'state.json'); protocol = checked(state['protocol'])
    if state['status'] not in {'predictions_locked', 'complete'} or set(state['units']) != set(protocol['jobs']):
        raise ValueError('All prediction attempts must lock before clinical scoring')
    if any(u['status'] not in {'predicted','failed'} for u in state['units'].values()):
        raise ValueError('Nonterminal prediction attempt')
    verify(protocol['source_pins']); verify(state['output_pins'])
    all_units = {**state['units'], **protocol['controls']}; store = Store(ROOT)
    records, rows = [], []
    for case in protocol['cases']:
        clinical, curve, endpoints = _case_reference(case, store)
        for sampling in base.SAMPLINGS:
            for method in ALL_METHODS:
                key = f'{case["case_id"]}__{sampling}__{method}'; unit = all_units[key]
                rec = dict(case_id=case['case_id'], drug=case['drug'], sampling=sampling, method=method,
                    status=unit['status'], resources=unit.get('resources',{}), formula=unit.get('formula'),
                    parameters=unit.get('parameters'), reused=method not in METHODS,
                    development_diagnostic=unit.get('development_diagnostic'), blood_partition_mapping=unit.get('blood_partition_mapping'))
                rec['partition_sensitivity']=unit.get('partition_sensitivity',state.get('sensitivity_controls',{}).get(key))
                if unit['status'] == 'predicted':
                    metrics, _, _ = _score_prediction(case, clinical, curve, endpoints, unit, store)
                    metrics.pop('curve', None); rec.update(metrics)
                    rec['prediction_source'] = unit['prediction']
                else:
                    rec.update(reason=unit.get('reason'), curve_metrics={'gmfe':None}, endpoints={
                        ep:dict(value=None,reference=o['value'],unit=o['unit'],error=None) for ep,o in endpoints.items()})
                records.append(rec)
                head = {k:rec[k] for k in ('drug','sampling','method','status')}
                rows.append(head | dict(endpoint='concentration_time_curve', prediction=None, reference=None,
                    unit=curve['unit'], error=rec['curve_metrics'].get('gmfe'), metric='GMFE', weak_tail=None))
                for endpoint, obs in endpoints.items():
                    p = rec['endpoints'][endpoint]
                    rows.append(head | dict(endpoint=endpoint, prediction=p.get('value'), reference=obs['value'],
                        unit=p.get('unit',obs['unit']), error=p.get('error'), metric=p.get('error_metric'), weak_tail=p.get('weak_tail')))
    clusters = {c['case_id']:c['cluster_id'] for c in protocol['cases']}
    summary, comparisons, endpoints_summary, sensitivity = {}, {}, {}, {}
    for sampling in base.SAMPLINGS:
        scores = {m:{r['case_id']:r['curve_metrics'].get('gmfe') for r in records if r['method']==m and r['sampling']==sampling} for m in ALL_METHODS}
        summary[sampling], comparisons[sampling], endpoints_summary[sampling] = {}, {}, {}
        for method, values in scores.items():
            valid = {k:v for k,v in values.items() if v is not None and math.isfinite(v) and v>0}
            summary[sampling][method] = clustered_geomean(valid, {k:clusters[k] for k in valid}) | dict(
                attempted=10, failed_or_unscorable=10-len(valid), within_twofold=sum(v<=2 for v in valid.values()))
            if method != 'blood_partition': comparisons[sampling][method] = paired_ratio(scores['blood_partition'], values, clusters)
            stats = {}
            selected = [r for r in records if r['method']==method and r['sampling']==sampling]
            for ep in selected[0]['endpoints']:
                entries = [r['endpoints'][ep] for r in selected]; errors = [e['error'] for e in entries if e.get('error') is not None]
                stats[ep] = dict(reference_available=sum(e.get('reference') is not None for e in entries),
                    predicted=sum(e.get('value') is not None for e in entries), scored=len(errors), weak_scored=sum(e.get('weak_tail',False) and e.get('error') is not None for e in entries))
                if errors:
                    stats[ep].update({'mean_absolute_error_h':float(np.mean(errors))} if ep=='tmax' else
                                    dict(gmfe=math.exp(float(np.mean(np.log(errors)))), within_twofold=sum(x<=2 for x in errors)))
            endpoints_summary[sampling][method] = stats
        subset = {c['case_id']:c['cluster_id'] for c in protocol['cases'] if c['drug'].lower() not in {'nemonoxacin','dexmedetomidine','midazolam'}}
        sensitivity[sampling] = paired_ratio(scores['blood_partition'],scores['direct_relaxed'],subset)
    adoption = all(summary[s]['blood_partition']['estimate'] is not None
                   and summary[s]['direct_relaxed']['estimate'] is not None
                   and summary[s]['blood_partition']['estimate'] < summary[s]['direct_relaxed']['estimate']
                   and comparisons[s]['direct_relaxed']['ratio_below_one'] >= 6
                   and summary[s]['blood_partition']['failed_or_unscorable'] <= summary[s]['direct_relaxed']['failed_or_unscorable']
                   for s in base.SAMPLINGS)
    resources = {m:{k:sum(r['resources'].get(k,0) for r in records if r['method']==m) for k in
                   ('objective_calls','failed_calls','diagnostic_calls','prediction_calls','sensitivity_calls','new_native','wall_seconds')} for m in ALL_METHODS}
    resources['direct_relaxed']['new_sensitivity_calls']=sum(v['calls'] for v in state.get('sensitivity_controls',{}).values())
    result = dict(version=VERSION,status='complete',protocol=state['protocol'],locked_at=state['locked_at'],scored_at=time.time(),
        records=records,summary=summary,comparisons=comparisons,endpoint_summary=endpoints_summary,
        source_sensitivity_excluding_nemonoxacin_dexmedetomidine_midazolam=sensitivity,resources=resources,
        development_adoption_rule_passed=adoption,limitations=protocol['limitations'],preflight=protocol['preflight'])
    save(RUN/'results.json',result)
    with (RUN/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Blood-cell partition calibration: ten-drug structural evaluation','',
        'All ten drugs are previously inspected development data. Full and five-point curves are fitted separately; scoring starts only after every prediction is locked. The manuscript is not replaced.','',
        '| Method | Full GMFE | Five-point GMFE |','|---|---:|---:|']
    for m in ALL_METHODS:
        values=[summary[s][m]['estimate'] for s in base.SAMPLINGS]
        a,b=['—' if value is None else f'{value:.4f}' for value in values]
        lines.append(f'| {m} | {a} | {b} |')
    lines += ['',f'Predefined development adoption criterion: **{"Pass" if adoption else "Fail"}**. Both full/five-point mean errors must improve, at least 6/10 drugs must improve in each, and failures must not increase. This is not a clinical-validation or publication-acceptance criterion.','',
        '| Condition | RBC / direct GMFE ratio | 95% study-cluster interval | Improved / worsened |','|---|---:|---|---:|']
    for s in base.SAMPLINGS:
        c=comparisons[s]['direct_relaxed']
        value='—' if c['estimate'] is None else f'{c["estimate"]:.4f}'
        interval='—' if c['ci95'] is None else f'[{c["ci95"][0]:.4f}, {c["ci95"][1]:.4f}]'
        lines.append(f'| {s} | {value} | {interval} | {c["ratio_below_one"]} / {c["ratio_above_one"]} |')
    lines += ['','[Predictions, real references, and errors for all 11 endpoint types](results.csv) · [Statistics, failures, and diagnostics](results.json).','',
        'Only native-formula-predicted RBC partition is opened to development calibration. B/P and vascular capacity change together; human blood volume, flow, measured fu, and net renal inputs remain fixed. Values are recorded as derived model parameters, not native readbacks or measurements.',
        'blood_partition and exchange_shape both fit four parameters; the latter controls additional degrees of freedom. direct_relaxed is the frozen three-parameter baseline. Each new method uses 16 fixed starts and at most 16384 physical calls per formula, without warm starts from old fits.',
        'Forty new method-drug-sampling units reuse 40 baseline/two-compartment references; these are not 80 independent drugs. All 880 endpoint rows remain, with absent references left empty.',
        'Fitting four parameters to five points risks overfitting. Fitted RBC partition is only an effective parameter; this does not validate stability without CL, prediction without a development curve, or clinical-risk reliability.','',
        'Inspect with `runtime/app-env/bin/python -B scripts/run_blood_partition_blueprint.py status`; `report` recalculates scores.']
    (RUN/'README.md').write_text('\n'.join(lines)+'\n')
    state.update(status='complete',completed_at=time.time());save(RUN/'state.json',state)
    return dict(status='complete',new_units=len(state['units']),rows=len(rows),summary=summary,development_adoption_rule_passed=adoption)


def status():
    if not (RUN/'state.json').exists(): return dict(status='not_prepared')
    state=read(RUN/'state.json')
    return dict(status=state['status'],completed=len(state['units']),total=40,reason=state.get('reason'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','status','report','cleanup'])
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--background',action='store_true');args=parser.parse_args()
    if not 1<=args.workers<=8: parser.error('workers must be 1..8')
    RUN.mkdir(parents=True,exist_ok=True)
    if args.command=='status': print(status());return
    if args.command=='cleanup':
        if read(RUN/'state.json')['status']!='complete':raise ValueError('Complete first')
        print(dict(removed=0,reason='No per-evaluation files; compact fits and final trajectories retained'));return
    if args.command=='run' and args.background:
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'}
        with (RUN/'run.log').open('ab') as log:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'run','--workers',str(args.workers)],
                cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
        (RUN/'run.pid').write_text(str(process.pid)+'\n');print(dict(pid=process.pid));return
    with (RUN/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run(args.workers) if args.command=='run' else globals()[args.command]()
    print(result)


if __name__=='__main__': main()
