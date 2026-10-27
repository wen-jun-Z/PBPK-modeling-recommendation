#!/usr/bin/env python3
"""Controlled CL gap-filler comparison in the frozen thirty-drug PBPK workflow.

Only molecular CL estimates change. All five development points, non-CL inputs,
physiology, absolute search bounds and visible-only repair rules are shared.
All clinical predictions lock before any target observation is decoded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
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
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import blueprint_reviewer_diagnostics as common
from pbpk_kb.blueprint_policy import visible_partition_feasibility as repair
from pbpk_kb.blueprint_policy.perfusion_calibration import clearance_inputs
from report_blueprint_validation import clustered_geomean

RUN = ROOT / 'experiments/blueprint_design/rf_gap_comparison_20261012'
VERSION = 'common-cl-gap-pbpk-1'
METHODS = ('rf', 'tabpfn', 'complementary')
ENDPOINTS = ('curve', 'auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life',
             'auc_inf', 'clearance', 'vz', 'mrt', 'vss')
frozen = common.frozen
read, checked, ref = common.read, common.checked, common.ref


def save(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(RUN):
        raise ValueError('Output outside this experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def eligible_cl(request):
    rows = [e for e in request['evidence'] if e['parameter'] == 'total_clearance']
    if len(rows) != 1:
        raise ValueError('Exactly one selected total CL required')
    e = rows[0]
    method = (e.get('prediction_method') or {}).get('name', '').lower().replace(' ', '_')
    return e['basis'] == 'predicted' and 'random_forest' in method


def replace_cl(request, value, method, source):
    """Protect independent evidence; change a declared RF-derived CL only."""
    result = deepcopy(request)
    if not eligible_cl(request):
        return result, []
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('CL prediction must be finite and positive')
    e = next(e for e in result['evidence'] if e['parameter'] == 'total_clearance')
    if e['unit'].lower() != 'ml/min/kg':
        raise ValueError('Predictor CL unit must be ml/min/kg; no implicit conversion')
    old = e['value']
    e.update(value=float(value), source={**source, 'location': request['compound_id']},
             prediction_method=dict(name=method, version=VERSION,
                 applicability='Frozen common scaffold split; no target clinical observations'),
             notes='Controlled CL-only gap replacement; independent evidence protected.')
    changes = [dict(parameter='total_clearance', previous=old, replacement=float(value), unit=e['unit'])]
    return result, changes


def prepare():
    if (RUN / 'protocol.json').exists():
        return status()
    _, source, _, choices = common.source_bundle()
    cases = source['cases']
    if len(cases) != 30:
        raise ValueError('The frozen thirty-drug cohort is required')
    pins = deepcopy(source['source_pins'])
    inputs = checked(ref(RUN / 'clinical_inputs.json'))
    if {c['case_id'] for c in cases} != {c['case_id'] for c in inputs['cases']}:
        raise ValueError('Molecular and clinical cohorts differ')
    jobs = {}
    for case in cases:
        cid = case['case_id']
        old = choices[cid + '__late_four']['cmax']['sources']['time_repaired']['unit_id']
        state = read(common.SOURCE / 'state.json')
        origin = state['units'][old]['result']
        unit = checked(origin)
        training = checked(unit['training'])
        if len(training['observations']) != 5:
            raise ValueError('Require the same five visible development points')
        req = frozen.gap.request(case, 'development', 'PKSimStandard')
        target = frozen.gap.request(case, 'target', 'PKSimStandard')
        if eligible_cl(req) != eligible_cl(target):
            raise ValueError('Development and target evidence policies differ')
        jobs[cid] = dict(case_id=cid, compound_id=case['compound_id'], drug=case['drug'],
                        source=origin, training=unit['training'], replaceable=eligible_cl(req))
        for r in (origin, unit['training']):
            pins[r['path']] = r['sha256']
    for path in (Path(__file__), Path(common.__file__), RUN / 'clinical_inputs.json'):
        pins[str(path.relative_to(ROOT))] = common.sha(path)
    if sum(j['replaceable'] for j in jobs.values()) != 24:
        raise ValueError('Expected 24 replaceable and six protected clinical inputs')
    protocol = dict(version=VERSION, created_at=time.time(), cases=cases, jobs=jobs,
        source_protocol=source['source_protocol'], source_pins=pins, methods=list(METHODS),
        parameter_split=inputs['source_split'], fitting=dict(starts=16, calls_per_formula=16384,
            formulas=list(frozen.base.FORMULAS), formula_timeout_seconds=180),
        monitor_seconds=1200, directory_limit_bytes=10*1024**3, max_workers=6,
        clinical_protocol='Frozen same five points, PBPK, other inputs, repair and absolute CL bounds; no warm start.',
        predictor_protocol='Common latest human CL split; all 30 identities/scaffolds excluded. RF is retrained for this comparison.',
        analysis=dict(seed=20261012, cluster_resamples=2000, scopes=['all_thirty', 'changed_twenty_four'],
            scalar_outputs=list(ENDPOINTS[1:]), primary='curve', no_composite_score=True,
            ties_relative_tolerance=1e-6, no_effect_based_exclusions=True),
        limitations=['Retrospective development cohort, not new blinded evidence.',
            'CL is freely recalibrated with fixed absolute bounds; the initial estimate can be absorbed by calibration.',
            'Complementary predictor has additional animal-label information; this is a practical pipeline comparison, not an architecture-only test.',
            'This common-CL-input experiment does not overwrite the original heterogeneous-RF Table 13 cohort.'])
    save(RUN / 'protocol.json', protocol)
    save(RUN / 'state.json', dict(status='prepared', jobs_done={}))
    return status()


class Guard:
    def __init__(self, protocol, *, formula_deadline=None):
        self.protocol, self.last, self.formula_deadline = protocol, time.monotonic(), formula_deadline

    def __call__(self, force=False):
        if (RUN / 'PAUSE').exists():
            raise RuntimeError('Experiment paused; no new tasks may be submitted')
        if self.formula_deadline is not None and time.monotonic() > self.formula_deadline:
            raise TimeoutError('Predeclared 180-second formula budget exhausted')
        if force or time.monotonic()-self.last >= self.protocol['monitor_seconds']:
            frozen.verify(self.protocol['source_pins'])
            if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()) > self.protocol['directory_limit_bytes']:
                raise RuntimeError('Experiment exceeds 10 GiB; paused')
            self.last = time.monotonic()


def fit_task(key, job, prior_ref):
    p = read(RUN / 'protocol.json')
    prior = checked(prior_ref)
    if not job['replaceable']:
        checked(job['source'])
        return dict(result=job['source'], reused=True, reason='Qualified independent CL retained')
    directory = RUN / 'fits' / key
    destination = directory / 'result.json'
    if destination.exists():
        result = read(destination)
        if result['prior'] != prior_ref:
            raise ValueError('Cached prediction belongs to different molecular prior')
        frozen.verify(result['output_pins'])
        return dict(result=ref(destination), reused=False)
    begin = time.monotonic()
    method = key.split('__', 1)[0]
    value = prior['predictions'][job['compound_id']]
    case = frozen.gap.get_case(checked(p['source_protocol']), job['case_id'])
    basep = checked(p['source_protocol'])
    training = checked(job['training'])
    fits, original_ledgers = [], {}
    for formula in frozen.base.FORMULAS:
        original = frozen.gap.request(case, 'development', formula)
        request, changes = replace_cl(original, value, method, prior_ref)
        physiology = frozen.prior.get_phys(basep, case, 'development', formula, original)
        original_ledgers['development_' + formula] = original['evidence_ledger']
        guard = Guard(p, formula_deadline=time.monotonic()+p['fitting']['formula_timeout_seconds'])
        try:
            fit = repair.fit(lambda x,t: frozen.old.simulate('blood_partition', physiology, request, t, x),
                request, training, physiology, budget=p['fitting']['calls_per_formula'],
                starts=p['fitting']['starts'], before_trial=guard)
            cl = clearance_inputs(request)
            best = fit['best']
            final_cl = (cl['renal_clearance_l_h'] or 0.) + (
                cl['total_clearance_l_h']-(cl['renal_clearance_l_h'] or 0.))*best['parameters']['clearance_scale'] if best else None
            fits.append(dict(formula=formula, fit=fit, best=best,
                             initial_cl_l_h=cl['total_clearance_l_h'], final_cl_l_h=final_cl))
        except TimeoutError as exc:
            fits.append(dict(formula=formula, best=None, status='timeout', reason=str(exc)))
    good = [f for f in fits if f['best'] is not None]
    result = dict(version=VERSION, case_id=job['case_id'], method=method, status='failed', prior=prior_ref,
        training=job['training'], changes=changes, fits=fits, target_observations_used=False,
        resources=dict(wall_seconds=time.monotonic()-begin,
            calls=sum(f.get('fit', {}).get('calls', 0) for f in fits),
            failures=sum(f.get('fit', {}).get('failures', 0) for f in fits)), output_pins={})
    if good:
        chosen = min(good, key=lambda f: f['best']['objective'])
        formula, best = chosen['formula'], chosen['best']
        original = frozen.gap.request(case, 'target', formula)
        request, _ = replace_cl(original, value, method, prior_ref)
        physiology = frozen.prior.get_phys(basep, case, 'target', formula, original)
        ledger = dict(version=VERSION, method=method, prior=prior_ref, changes=changes,
            original_development_ledgers=original_ledgers, original_target_ledger=original['evidence_ledger'],
            independent_evidence_protected=True, only_predicted_cl_changes=True)
        save(directory / 'input_ledger.json', ledger)
        request['evidence_ledger'] = {**ref(directory/'input_ledger.json'), 'location': 'changes'}
        # Formula and parameters lock before evaluating target trajectory; no target values used.
        result['selection'] = dict(formula=formula, parameters=best, locked_at=time.time(), target_values_used=False)
        grid = frozen.base.engine.prediction_grid(request)
        prediction = frozen.old.simulate('blood_partition', physiology, request, grid, best['log_parameters'])
        save(directory/'prediction.json', frozen.prior.curve_document(prediction, grid, request))
        save(directory/'request.json', request)
        result.update(status='predicted', formula=formula, parameters=best,
            initial_cl_l_h=chosen['initial_cl_l_h'], final_cl_l_h=chosen['final_cl_l_h'],
            request=ref(directory/'request.json'), prediction=ref(directory/'prediction.json'),
            mass_balance=prediction['mass_balance'])
        for path in (directory/'input_ledger.json', directory/'request.json', directory/'prediction.json'):
            result['output_pins'][str(path.relative_to(ROOT))] = common.sha(path)
    else:
        result['reason'] = 'No feasible completed formula; failed trials retained'
    result['prediction_locked_at'] = time.time()
    save(destination, result)
    return dict(result=ref(destination), reused=False)


def run(workers=6):
    prepare()
    p = read(RUN/'protocol.json')
    if not 1 <= workers <= p['max_workers']:
        raise ValueError('Use between one and six project workers')
    with (RUN/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(RUN/'state.json')
        if state['status'] == 'complete':
            return status()
        guard = Guard(p); guard(force=True)
        priors = {method: ref(RUN/'priors'/f'{method}.json') for method in METHODS}
        # All three predictors must be frozen before this clinical comparison begins.
        for method, prior_ref in priors.items():
            prior = checked(prior_ref)
            if set(prior['predictions']) != {j['compound_id'] for j in p['jobs'].values()}:
                raise ValueError('Prior must cover the exact same thirty compounds')
        if state.get('priors') and state['priors'] != priors:
            raise ValueError('Molecular predictions changed after clinical launch')
        state.update(status='running', priors=priors, pid=os.getpid(), started_at=state.get('started_at', time.time()))
        save(RUN/'state.json', state)
        pending = [(method+'__'+cid, job, priors[method]) for method in METHODS
                   for cid,job in p['jobs'].items() if method+'__'+cid not in state['jobs_done']]
        consecutive_errors = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            active = {}
            while pending or active:
                guard()
                while pending and len(active) < workers:
                    key, job, prior_ref = pending.pop(0)
                    active[pool.submit(fit_task,key,job,prior_ref)] = key
                done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    key = active.pop(future)
                    try:
                        state['jobs_done'][key] = future.result()
                        consecutive_errors = 0
                    except Exception as exc:
                        consecutive_errors += 1
                        state.setdefault('infrastructure_errors', {})[key] = dict(error=repr(exc), traceback=traceback.format_exc())
                        state.update(status='paused', reason='Infrastructure failure; fix before resuming', updated_at=time.time())
                        save(RUN/'state.json', state)
                        raise
                    state.update(updated_at=time.time(), active_jobs=list(active.values()))
                    save(RUN/'state.json', state)
                    print(json.dumps(dict(done=len(state['jobs_done']), total=90, key=key)), flush=True)
        state.update(status='predictions_locked', predictions_locked_at=time.time(), active_jobs=[])
        save(RUN/'state.json', state)
        return report()


def require_prediction_locks(protocol, state):
    expected = {method+'__'+cid for method in METHODS for cid in protocol['jobs']}
    if state.get('status') not in ('predictions_locked','complete') or set(state['jobs_done']) != expected:
        raise ValueError('All ninety prediction attempts must lock before target scoring')
    for record in state['jobs_done'].values():
        unit = checked(record['result'])
        if not record['reused']:
            frozen.verify(unit['output_pins'])
    return True


def metric(score, endpoint):
    if score is None:
        return None
    return score['curve_metrics']['gmfe'] if endpoint=='curve' else score['endpoints'][endpoint]['error']


def report():
    p, state = read(RUN/'protocol.json'), read(RUN/'state.json')
    require_prediction_locks(p, state)
    if state['status']=='complete':
        checked(state['results'])
        return status()
    records, score_cache = [], {}
    for method in METHODS:
        for case in p['cases']:
            key = method+'__'+case['case_id']; record = state['jobs_done'][key]
            unit = checked(record['result'])
            path = record['result']['path']
            if path not in score_cache:
                score_cache[path] = frozen.data.score_case(case, unit) if unit['status']=='predicted' else None
            score = score_cache[path]
            records.append(dict(case_id=case['case_id'], drug=case['drug'], cluster_id=case['cluster_id'],
                method=method, replaceable=p['jobs'][case['case_id']]['replaceable'], result=record['result'],
                reused=record['reused'], status=unit['status'], score=common.compact(score),
                initial_cl_l_h=unit.get('initial_cl_l_h'), final_cl_l_h=unit.get('final_cl_l_h'),
                formula=unit.get('formula'), resources=unit.get('resources')))
    summary = {}
    for scope in p['analysis']['scopes']:
        rr = [r for r in records if scope=='all_thirty' or r['replaceable']]
        summary[scope] = {}
        for method in METHODS:
            group = [r for r in rr if r['method']==method]
            metrics = {}
            for endpoint in ENDPOINTS:
                values = {r['case_id']:metric(r['score'],endpoint) for r in group}
                values = {k:v for k,v in values.items() if v is not None and math.isfinite(v)}
                if endpoint=='tmax':
                    metrics[endpoint]=dict(n=len(values), mae_h=sum(values.values())/len(values) if values else None)
                else:
                    clusters = {r['case_id']:r['cluster_id'] for r in group if r['case_id'] in values}
                    stats = clustered_geomean(values, clusters, seed=p['analysis']['seed'], resamples=2000) if values else dict(n_cases=0,estimate=None,ci95=None)
                    metrics[endpoint]=dict(**stats, within_2fold=sum(v<=2 for v in values.values()))
            summary[scope][method]=dict(enrolled=len(group),predicted=sum(r['status']=='predicted' for r in group),metrics=metrics)
        for method in METHODS[1:]:
            paired = {}
            for endpoint in ENDPOINTS:
                if endpoint=='tmax':
                    continue
                rf = {r['case_id']:r for r in rr if r['method']=='rf'}
                values, clusters = {}, {}
                for row in rr:
                    if row['method']!=method:
                        continue
                    a,b=metric(row['score'],endpoint),metric(rf[row['case_id']]['score'],endpoint)
                    if a is not None and b is not None and a>0 and b>0:
                        values[row['case_id']]=a/b;clusters[row['case_id']]=row['cluster_id']
                stats = clustered_geomean(values,clusters,seed=p['analysis']['seed'],resamples=2000) if values else dict(n_cases=0,estimate=None,ci95=None)
                paired[endpoint]=dict(**stats, improved=sum(v<1-1e-6 for v in values.values()),
                    worsened=sum(v>1+1e-6 for v in values.values()), tied=sum(abs(v-1)<=1e-6 for v in values.values()))
            summary[scope][method]['paired_to_rf']=paired
    result = dict(version=VERSION, protocol=ref(RUN/'protocol.json'), priors=state['priors'],
        predictions_locked_at=state['predictions_locked_at'], scored_at=time.time(),
        records=records, summary=summary, independent_evidence_protected=True)
    save(RUN/'results.json', result)
    with (RUN/'endpoints.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['case_id','drug','method','endpoint','prediction','reference','unit','error','error_metric','status','replaceable'])
        writer.writeheader()
        for r in records:
            for endpoint in ENDPOINTS:
                data=(r['score'] or {}).get('endpoints',{}).get(endpoint,{})
                writer.writerow(dict(case_id=r['case_id'],drug=r['drug'],method=r['method'],endpoint=endpoint,
                    prediction=data.get('value'),reference=data.get('reference'),unit=data.get('unit'),
                    error=metric(r['score'],endpoint),error_metric='curve_GMFE' if endpoint=='curve' else data.get('error_metric'),
                    status=r['status'],replaceable=r['replaceable']))
    state.update(status='complete',results=ref(RUN/'results.json'),finished_at=time.time())
    save(RUN/'state.json',state)
    return status()


def status():
    state=read(RUN/'state.json') if (RUN/'state.json').exists() else dict(status='not_prepared',jobs_done={})
    return dict(status=state['status'],completed=len(state['jobs_done']),total=90,
                active=state.get('active_jobs',[]),pid=state.get('pid'),
                output_mib=round(sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())/1024**2,2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','status','report'])
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--background',action='store_true')
    args=parser.parse_args()
    if args.background:
        if args.command!='run':raise ValueError('Background applies to run only')
        prepare()
        with (RUN/'run.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run','--workers',str(args.workers)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (RUN/'run.pid').write_text(str(proc.pid)+'\n')
        output=dict(pid=proc.pid,log=str((RUN/'run.log').relative_to(ROOT)))
    else:
        output=prepare() if args.command=='prepare' else run(args.workers) if args.command=='run' else report() if args.command=='report' else status()
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':
    main()
