#!/usr/bin/env python3
"""Published inference algorithms on the frozen thirty-drug PBPK benchmark.

The existing physiological model, drug evidence, two partition formulae and
visible-data repair are shared. Only the parameter inference algorithm changes.
No old fitted parameter enters a prior or an initializer. All target predictions
lock before report() is permitted to decode any clinical target outcome.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

for _thread_option in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_option]='1'

import numpy as np
import blueprint_reviewer_comparators as frozen
import run_sampling_stability as current
from report_sampling_expansion import bootstrap, json_native
from pbpk_kb.blueprint_policy import distribution_calibration_optimizer as optimizer
from pbpk_kb.blueprint_policy import visible_partition_feasibility as repair
from pbpk_kb.blueprint_policy import stable_log_pbpk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/blueprint_design/published_inference_controls_20261011'
METHODS = ('bayesian_dram', 'pss_bayesian_dram', 'snpe_c')
FORMULAS = ('PKSimStandard', 'RodgersRowland')
SEED = 20261011
VERSION = 'published-pbpk-inference-same-visible-thirty-2-stable-log'
read, checked, ref = frozen.read, frozen.checked, frozen.ref


def save(path, value):
    path = frozen.inside(path)
    if not path.is_relative_to(OUT):
        raise ValueError('Output outside this experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(json_native(value), ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    temporary.replace(path)


def csv_save(path, rows):
    path = frozen.inside(path)
    if not path.is_relative_to(OUT):
        raise ValueError('CSV output outside this experiment')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def build_adapter(case, training_ref, physiology_cache):
    """Return visible-only inputs. Case outcomes and old fitted parameters omitted."""
    training = checked(training_ref)
    output = dict(training_source=training_ref, formulas={})
    for formula in FORMULAS:
        req = current.gap.request(case, 'development', formula)
        target = current.gap.request(case, 'target', formula)
        development_phys = physiology_cache[f'{case["case_id"]}__development__{formula}']
        target_phys = physiology_cache[f'{case["case_id"]}__target__{formula}']
        phys = checked(development_phys)
        # Reuse the same receipt/request validation as the current predictor.
        current.prior.get_phys(dict(physiology_cache=physiology_cache), case, 'development', formula, req)
        current.prior.get_phys(dict(physiology_cache=physiology_cache), case, 'target', formula, target)
        times, observed = optimizer._visible(req, training)
        if len(times) != 5:
            raise ValueError('Exactly the five frozen visible observations required')
        feasibility = repair.feasibility(phys, req, training)
        if feasibility['protected_measured_partition']:
            raise ValueError('Measured partition needs a validated fixed-RBC engine')
        lower, upper, guesses, absolute, clearances = repair.initialization(req, phys, 'blood_partition', 16, feasibility)
        output['formulas'][formula] = dict(development_request=req, target_request=target,
            development_physiology=development_phys, target_physiology=target_phys,
            times_h=times.tolist(), observed_mg_l=observed.tolist(), initial=guesses[0].tolist(),
            lower=lower.tolist(), upper=upper.tolist(), feasibility=feasibility,
            absolute_nonrenal_clearance_bounds=absolute, clearance_inputs=clearances)
    return output


def predictor(formula_input):
    """Development-only callable. This closure never captures a target request."""
    phys = checked(formula_input['development_physiology'])
    req = deepcopy(formula_input['development_request'])
    times = np.asarray(formula_input['times_h'], float)
    def predict(log_parameters):
        return stable_log_pbpk.predict_log(phys,req,times,np.asarray(log_parameters,float))
    return predict


def development_objective(log_parameters, prediction_log, observed_log):
    """Same formula-selection objective as the frozen four-axis blueprint."""
    x = np.asarray(log_parameters, float)
    mse = float(np.mean((np.asarray(prediction_log)-observed_log)**2))
    penalty = .01*((x[1]/math.log(10))**2+(x[2]/math.log(100))**2+(x[3]/math.log(10))**2)
    return mse+penalty


def verify_adapter(case, adapter, source_ref):
    """Replay old parameters ONLY for identity verification, never training."""
    unit = checked(source_ref)
    formula = unit['formula']; inp = adapter['formulas'][formula]
    x = np.asarray(unit['parameters']['log_parameters'])
    observed = np.log(np.asarray(inp['observed_mg_l']))
    predicted = predictor(inp)(x)
    objective = development_objective(x, predicted, observed)
    if not math.isclose(objective, unit['parameters']['objective'], rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError('Replayed development objective differs from frozen blueprint')
    old_curve = checked(unit['prediction'])
    target = inp['target_request']; phys = checked(inp['target_physiology'])
    times = np.asarray(old_curve['time_min'])/60
    result = current.old.simulate('blood_partition', phys, target, times, x)
    curve = current.prior.curve_document(result, times, target)
    a, b = np.asarray(curve['concentration']), np.asarray(old_curve['concentration'])
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    return dict(case_id=case['case_id'], formula=formula, source=source_ref,
        points=len(a), maximum_absolute_difference=float(np.max(np.abs(a-b))),
        relative_development_objective_difference=abs(objective-unit['parameters']['objective'])/max(objective,1e-12),
        old_parameters_used_for_initialization=False)


def algorithm_libraries(pins):
    """Pin installed algorithm code inside the project's isolated dependencies."""
    descriptions=[]
    for folder,name,subdir in (
        ('runtime/published-inference-python','pymcmcstat','pymcmcstat'),
        ('runtime/published-inference-python','arviz','arviz/stats'),
        ('runtime/sbi-python','sbi','sbi'),
        ('runtime/sbi-python','nflows','nflows')):
        location=ROOT/folder
        matches=[d for d in importlib.metadata.distributions(path=[str(location)])
                 if d.metadata.get('Name','').lower().replace('_','-')==name.replace('_','-')]
        if len(matches)!=1:
            raise ValueError('Missing/ambiguous project-local inference dependency: '+name)
        files=sorted((location/subdir).rglob('*.py'))
        if not files:raise ValueError('Missing project-local inference source: '+name)
        for path in files:
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError('Algorithm dependency resolves outside project')
            pins[str(path.relative_to(ROOT))]=frozen.sha(path)
        descriptions.append(dict(package=name,version=matches[0].version,root=folder+'/'+subdir,pinned_python_files=len(files)))
    return descriptions


def prepare():
    if (OUT/'protocol.json').exists():
        guard(checked(read(OUT/'state.json')['protocol']))
        return status()
    from pbpk_kb.blueprint_policy import published_inference as engine
    from pbpk_kb.blueprint_policy import published_neural_inference as neural
    comparator_state = read(frozen.OUT/'state.json')
    if comparator_state['status'] != 'complete':
        raise ValueError('Frozen comparator batch must be complete')
    comparator = checked(comparator_state['protocol'])
    frozen.verify(comparator['source_pins'])
    current_state = read(current.RUN/'state.json')
    current_protocol = checked(current_state['protocol'])
    source = checked(current_protocol['source_protocol'])
    choices = checked(current_protocol['choices'])['choices']
    pins = dict(comparator['source_pins'])
    cases, jobs, audits = {}, {}, []
    for cid, case in comparator['cases'].items():
        cell = next(c for c in comparator['cells'] if c['case_id']==cid and c['info']=='time_five')
        adapter = build_adapter(case['source_case'], cell['training_source'], source['physiology_cache'])
        selection = choices[cid+'__late_four']['cmax']['sources']['time_repaired']
        source_ref = current_state['units'][selection['unit_id']]['result']
        audits.append(verify_adapter(case['source_case'], adapter, source_ref))
        cases[cid] = deepcopy(case)
        cases[cid].update(selected_time_h=cell['selected_time_h'], frozen_pbpk_prediction=source_ref)
        for method in METHODS:
            jobs[cid+'__'+method] = dict(case_id=cid, method=method, adapter=adapter,
                selected_time_h=cell['selected_time_h'])
        frozen.pins_in(adapter, pins); frozen.pins_in(source_ref, pins)
    for path in (Path(__file__), Path(engine.__file__), Path(neural.__file__), Path(stable_log_pbpk.__file__), frozen.OUT/'protocol.json', frozen.OUT/'state.json',
                 frozen.OUT/'endpoints.csv', current.RUN/'state.json'):
        pins[str(path.relative_to(ROOT))] = frozen.sha(path)
    source_manifest=ref(ROOT/'artifacts/published_blueprint_controls/source_manifest.json')
    source_audit=ref(ROOT/'artifacts/published_blueprint_controls/source_audit.json')
    frozen.pins_in(source_manifest,pins);frozen.pins_in(source_audit,pins)
    frozen.pins_in(checked(source_manifest),pins)
    libraries=algorithm_libraries(pins)
    amendment=ref(OUT/'revisions/numerical_underflow/amendment.json')
    frozen.pins_in(amendment,pins)
    protocol = dict(version=VERSION, created_at=time.time(), cases=cases, jobs=jobs, source_pins=pins,
        methods=list(METHODS), formulas=list(FORMULAS), engine_config={
            'bayesian_dram':engine.DEFAULT_CONFIG,'pss_bayesian_dram':engine.DEFAULT_CONFIG,'snpe_c':neural.DEFAULT_CONFIG},
        seed=SEED, bootstrap=dict(seed=frozen.SEED, resamples=2000), adapter_identity=audits,
        published_source_manifest=source_manifest,published_source_audit=source_audit,algorithm_libraries=libraries,
        numerical_amendment=amendment,numerical_policy='All three inference methods share exact log-concentration propagation of the same physiological matrix, with scaled exponential fallback for underflow/cancellation. No clipped concentration floor, parameter filtering, or target-based changes.',
        input_policy='Exactly the same five visible development concentrations and evidence/human physiology as the frozen blueprint; no full-curve warm start.',
        inference_policy='Published inference algorithms adapted to the fixed four-axis physiological simulator. Not a reproduction of another paper\'s complete drug-specific PBPK model.',
        prior_policy='Initializer is nominal unit multipliers clipped to legal bounds. Log-CL has a bounded uniform prior; the other three axes have zero-centered Gaussian densities truncated to the shared legal bounds (the truncated mean need not be zero). Neither prior nor initializer uses previous fitted clinical parameters.',
        repair_policy='The same visible-only mass bound and neutral-RBC lower-bound extension as the frozen PBPK, applied before every inference method. Repair is shared preprocessing, not an external paper contribution.',
        formula_selection='Two formulae per method; select minimum frozen penalized visible-development objective of the returned point parameters. Target values never select formula.',
        point_prediction='Plug-in point parameters returned by inference engine; full posterior retained. Not a posterior-predictive average.',
        endpoints=list(frozen.ENDPOINTS), target_observation_values_used=False,
        maximum_workers=14, maximum_neural_workers=8, threads_per_worker=1, monitor_seconds=1200, directory_limit_gib=10,
        resource_probe=dict(kind='Same 4096-simulation, 256-batch, 64-hidden, five-transform synthetic two-round SNPE-C memory probe; no clinical targets',
            peak_gpu_allocated_bytes=35097088,peak_gpu_reserved_bytes=41943040,approximate_context_library_bytes=368377856,
            device_free_before_probe_bytes=7906525184,neural_worker_budget_bytes=536870912,
            minimum_remaining_device_headroom_bytes=1073741824,neural_workers=8,cpu_workers_when_neural_active=6),
        maximum_job_wall_seconds=7200,wall_limit_policy='A two-hour per-drug inference limit is an infrastructure pause, never a scientific prediction failure or a target-based exclusion.',
        limitations=['Retrospective development benchmark: all targets historically viewed, not a new blinded test.',
            'Four effective calibration parameters and five points cannot establish unique physiological identification.',
            'Posterior diagnostic failures remain visible and are not removed based on target error.',
            'Compute is reported explicitly, not claimed matched to frozen least squares.',
            'Missing clinical endpoints remain null; all 11 output categories retained.'])
    frozen.verify(pins)
    save(OUT/'protocol.json', protocol)
    save(OUT/'state.json', dict(status='prepared', protocol=ref(OUT/'protocol.json'), completed={}))
    return status()


def guard(protocol):
    if (OUT/'PAUSE').exists():
        raise RuntimeError('Experiment paused; inspect project-local PAUSE')
    frozen.verify(protocol['source_pins'])
    size = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())
    if size > protocol['directory_limit_gib']*1024**3:
        raise RuntimeError('Experiment output exceeds configured 10 GiB bound')


def fit_job(key, job, protocol_ref, engine_config):
    if job['method']=='snpe_c':
        from pbpk_kb.blueprint_policy import published_neural_inference as engine
    else:
        from pbpk_kb.blueprint_policy import published_inference as engine
    protocol=checked(protocol_ref);guard(protocol)
    folder = OUT/'fits'/key
    path = folder/'result.json'
    if path.exists():
        result = read(path)
        if result['protocol'] != protocol_ref or result['job'] != key:
            raise ValueError('Changed resumed fit identity')
        for item in result.get('output_refs', []):
            if frozen.sha(item['path']) != item['sha256']:
                raise ValueError('Changed resumed output')
        return ref(path)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); fits = []; output_refs = []
    for formula, inp in job['adapter']['formulas'].items():
        guard(protocol)
        fit_path = folder/(formula+'.json')
        if fit_path.exists():
            fit = read(fit_path)
            if fit['protocol'] != protocol_ref:
                raise ValueError('Changed partial formula protocol')
        else:
            raw_visible = predictor(inp)
            last_check=[time.monotonic()]
            def visible(x):
                if time.monotonic()-started>protocol['maximum_job_wall_seconds']:
                    raise RuntimeError('Per-job two-hour wall limit exceeded; pause and preserve checkpoints')
                if time.monotonic()-last_check[0]>=protocol['monitor_seconds']:
                    guard(protocol);last_check[0]=time.monotonic()
                return raw_visible(x)
            observed = np.log(np.asarray(inp['observed_mg_l']))
            seed = SEED+int(hashlib.sha256((key+'__'+formula).encode()).hexdigest()[:7],16)
            try:
                result = engine.fit(visible, observed, np.asarray(inp['initial']), np.asarray(inp['lower']),
                    np.asarray(inp['upper']), method=job['method'], seed=seed, **engine_config)
                samples = np.asarray(result.pop('posterior_samples'))
                samples_path = folder/(formula+'_posterior.npz')
                np.savez_compressed(samples_path, log_parameters=samples)
                x = np.asarray(result['point_parameters'])
                predicted = visible(x)
                fit = dict(status='fitted', formula=formula, inference=result, parameters=x.tolist(),
                    development_log_mse=float(np.mean((predicted-observed)**2)),
                    development_objective=development_objective(x,predicted,observed), posterior=ref(samples_path))
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                fit = dict(status='failed', formula=formula, reason=type(exc).__name__+': '+str(exc))
            fit.update(protocol=protocol_ref, method=job['method'], locked_at=time.time(),
                target_observations_used=False, visible_points=5)
            save(fit_path, fit)
        output_refs.append(ref(fit_path))
        if fit.get('posterior'): output_refs.append(fit['posterior'])
        fits.append(fit)
    eligible = [f for f in fits if f['status']=='fitted']
    result = dict(job=key, case_id=job['case_id'], method=job['method'], protocol=protocol_ref,
        status='failed', fits=[ref(folder/(f+'.json')) for f in FORMULAS],
        target_observations_used=False, full_development_observations_used=False,
        previous_fitted_parameters_used=False, visible_points=5)
    if eligible:
        chosen = min(eligible,key=lambda x:x['development_objective'])
        formula = chosen['formula']; inp = job['adapter']['formulas'][formula]
        target = inp['target_request']; phys = checked(inp['target_physiology'])
        grid = current.base.engine.prediction_grid(target)
        result.update(formula=formula, parameters=chosen['parameters'],inference_diagnostics=chosen['inference']['diagnostics'])
        try:
            predicted = current.old.simulate('blood_partition',phys,target,grid,chosen['parameters'])
            curve = current.prior.curve_document(predicted,grid,target)
            save(folder/'prediction.json',curve); save(folder/'target_request.json',target)
            output_refs += [ref(folder/'prediction.json'),ref(folder/'target_request.json')]
            result.update(status='predicted',prediction=ref(folder/'prediction.json'), request=ref(folder/'target_request.json'),
                mass_balance=predicted['mass_balance'])
        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
            result.update(status='prediction_failed',reason=type(exc).__name__+': '+str(exc))
    result.update(wall_seconds=time.monotonic()-started,prediction_locked_at=time.time(),output_refs=output_refs)
    save(path,result)
    return ref(path)


def run(workers=8, limit=None, methods=None):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(OUT/'state.json'); p=checked(state['protocol']); guard(p)
        if state['status']=='complete': return status()
        jobs=[(k,j) for k,j in p['jobs'].items() if k not in state['completed'] and (methods is None or j['method'] in methods)]
        if limit is not None: jobs=jobs[:limit]
        state.update(status='running',pid=os.getpid(),started_at=state.get('started_at',time.time()))
        save(OUT/'state.json',state); last=time.monotonic()
        active_neural_workers=p['maximum_neural_workers'] if any(j['method']=='snpe_c' for _,j in jobs) else 0
        with ProcessPoolExecutor(max_workers=min(workers,p['maximum_workers']-active_neural_workers)) as pool, ProcessPoolExecutor(max_workers=p['maximum_neural_workers']) as neural_pool:
            pending={(neural_pool if j['method']=='snpe_c' else pool).submit(fit_job,k,j,state['protocol'],p['engine_config'][j['method']]):k for k,j in jobs}
            for future in as_completed(pending):
                key=pending[future]
                try: state['completed'][key]=future.result()
                except Exception as exc:
                    state.update(status='paused',infrastructure_error=type(exc).__name__+': '+str(exc),failed_job=key)
                    save(OUT/'state.json',state)
                    for future in pending: future.cancel()
                    raise
                save(OUT/'state.json',state)
                print(json.dumps(dict(completed=len(state['completed']),total=len(p['jobs']),job=key)),flush=True)
                if time.monotonic()-last>=p['monitor_seconds']:
                    guard(p); last=time.monotonic()
        if set(state['completed'])==set(p['jobs']):
            guard(p); state.update(status='predictions_locked',predictions_locked_at=time.time())
        else: state['status']='partial'
        save(OUT/'state.json',state)
    return status()


def aggregates(rows,cases):
    output={}
    for cohort in ('all_thirty','original_ten','added_twenty'):
        ids={k for k,c in cases.items() if cohort=='all_thirty' or c['cohort']==cohort}
        groups={}; comparisons={}
        for method in ('pbpk','two_compartment',*METHODS):
            groups[method]={}
            for endpoint in frozen.ENDPOINTS:
                selected=[r for r in rows if r['case_id'] in ids and r['method']==method and r['endpoint']==endpoint]
                values={r['case_id']:float(r['error']) for r in selected if r['error'] is not None}
                if endpoint=='tmax':
                    stat=dict(estimate=float(np.mean(list(values.values()))) if values else None,
                        metric='mean_absolute_error_h',n_cases=len(values),ci95=None)
                else:
                    stat=bootstrap(values,{k:cases[k]['cluster_id'] for k in values},seed=frozen.SEED,resamples=2000)
                    stat['metric']='GMFE';stat['within_two']=sum(x<=2 for x in values.values())
                stat.update(enrolled=len(ids),failed_or_missing=len(ids)-len(values));groups[method][endpoint]=stat
            if method!='pbpk':
                comparisons['pbpk/'+method]={}
                for endpoint in ('curve','cmax','auc','c_last','terminal_half_life'):
                    def values(m): return {r['case_id']:float(r['error']) for r in rows if r['case_id'] in ids and r['method']==m and r['endpoint']==endpoint and r['error'] is not None}
                    comparisons['pbpk/'+method][endpoint]=frozen.pair_stats(values('pbpk'),values(method),cases)
        output[cohort]={'time_five':dict(methods=groups,comparisons=comparisons)}
    return output


def report():
    state=read(OUT/'state.json');p=checked(state['protocol']);guard(p)
    if state['status'] not in ('predictions_locked','complete') or set(state['completed'])!=set(p['jobs']):
        raise ValueError('All target predictions or documented failures must lock before target scoring')
    from blueprint_sampling_expansion_data import score_case
    rows=[];diagnostics=[];failures=[]
    for key, reference in state['completed'].items():
        document=checked(reference);job=p['jobs'][key];case=p['cases'][job['case_id']]
        if document['prediction_locked_at']>state['predictions_locked_at'] or document['target_observations_used']:
            raise ValueError('Invalid prediction-before-target lock')
        score=score_case(case['source_case'],dict(request=document['request'],prediction=document['prediction'])) if document['status']=='predicted' else None
        if score is None:failures.append(dict(job=key,status=document['status'],reason=document.get('reason'),fits=document['fits']))
        for endpoint in frozen.ENDPOINTS:
            e={} if score is None or endpoint=='curve' else score['endpoints'][endpoint]
            rows.append(dict(case_id=job['case_id'],drug=case['drug'],cohort=case['cohort'],cluster_id=case['cluster_id'],
                info='time_five',method=job['method'],transfer='linear_weight',visible_points=5,
                selected_time_h=job['selected_time_h'],endpoint=endpoint,prediction=e.get('value'),reference=e.get('reference'),
                error=(score['curve_metrics']['gmfe'] if score else None) if endpoint=='curve' else e.get('error'),
                unit=e.get('unit'),weak_tail=e.get('weak_tail'),reference_weak_tail=e.get('reference_weak_tail'),status=document['status'],job=key))
        for fitref in document['fits']:
            fit=checked(fitref);diag=fit.get('inference',{}).get('diagnostics',{})
            diagnostics.append(dict(case_id=job['case_id'],drug=case['drug'],method=job['method'],formula=fit['formula'],
                selected=fit['formula']==document.get('formula'),status=fit['status'],
                development_log_mse=fit.get('development_log_mse'),wall_seconds=document['wall_seconds'],
                simulator_calls=fit.get('inference',{}).get('simulator_calls',diag.get('simulator_calls')),
                formula_elapsed_seconds=diag.get('elapsed_seconds',fit.get('inference',{}).get('wall_seconds')),converged=diag.get('converged'),
                inference_status=diag.get('status'),
                diagnostics=json.dumps(diag,separators=(',',':')),reason=fit.get('reason')))
    with (frozen.OUT/'endpoints.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['info']!='time_five' or row['transfer']!='linear_weight' or row['method'] not in ('pbpk','two_compartment'):continue
            for name in ('prediction','reference','error','selected_time_h'):row[name]=float(row[name]) if row[name] else None
            rows.append(row)
    csv_save(OUT/'endpoints.csv',rows);csv_save(OUT/'diagnostics.csv',diagnostics)
    result=dict(status='complete',protocol=state['protocol'],predictions_locked_at=state['predictions_locked_at'],
        scored_at=time.time(),summary=aggregates(rows,p['cases']),failures=failures,
        new_fits=len(p['jobs']),formula_fits=len(diagnostics),drugs=len(p['cases']),
        methods=['pbpk','two_compartment',*METHODS],endpoints=ref(OUT/'endpoints.csv'),
        diagnostics=ref(OUT/'diagnostics.csv'),limitations=p['limitations'])
    save(OUT/'results.json',result)
    state.update(status='complete',results=ref(OUT/'results.json'),completed_at=time.time());save(OUT/'state.json',state)
    return status()


def status():
    if not (OUT/'state.json').exists():return dict(status='not_prepared')
    state=read(OUT/'state.json');p=checked(state['protocol'])
    return dict(status=state['status'],completed=len(state['completed']),total=len(p['jobs']),
        formula_fits=sum((OUT/'fits'/k/f'{formula}.json').exists() for k in p['jobs'] for formula in FORMULAS),
        started_at=state.get('started_at'),results=state.get('results'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','status','report'))
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--limit',type=int)
    parser.add_argument('--methods',nargs='+',choices=METHODS)
    args=parser.parse_args()
    value=run(args.workers,args.limit,args.methods) if args.command=='run' else globals()[args.command]()
    print(json.dumps(json_native(value),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
