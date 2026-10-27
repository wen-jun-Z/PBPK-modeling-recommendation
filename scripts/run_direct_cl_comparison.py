#!/usr/bin/env python3
"""Frozen molecular CL -> uncalibrated PBPK, then separate calibrated context.

The direct arm never decodes development concentrations and never selects a
formula from observations. All four log multipliers are zero. The two formulas
are separate prespecified strata, not per-drug candidates selected by accuracy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import run_cl_gap_comparison as previous
from report_blueprint_validation import clustered_geomean, paired_ratio

RUN = previous.RUN / 'direct_prediction'
VERSION = 'direct-cl-without-clinical-calibration-1'
METHODS, ENDPOINTS = previous.METHODS, previous.ENDPOINTS
FORMULAS = ('PKSimStandard', 'RodgersRowland')
read, checked, ref = previous.read, previous.checked, previous.ref


def save(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(RUN):
        raise ValueError('Output outside direct prediction experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')) + '\n')
    temporary.replace(path)


def direct_predict(original_request, physiology, cl_value, method, prior_ref):
    """Pure solve: no fitted parameters, observation access, repair or selection."""
    if method not in METHODS or original_request['partition_method'] not in FORMULAS:
        raise ValueError('Unknown frozen predictor or formula')
    request, changes = previous.replace_cl(original_request, cl_value, method, prior_ref)
    logs = np.zeros(4, dtype=float)
    grid = previous.frozen.base.engine.prediction_grid(request)
    cl = previous.clearance_inputs(request)
    native = previous.frozen.old.simulate('blood_partition', physiology, request, grid, logs)
    effective = native['effective_total_clearance_l_h']
    if not math.isclose(effective, cl['total_clearance_l_h'], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Direct model did not preserve the requested CL')
    return dict(status='predicted', request=request,
                prediction=previous.frozen.prior.curve_document(native, grid, request),
                changes=changes, cl_input=cl, effective_total_clearance_l_h=effective,
                log_parameters=logs.tolist(), mass_balance=native['mass_balance'],
                calibration_performed=False, repair_performed=False,
                development_observations_used=False, target_observations_used=False)


def prepare():
    if (RUN / 'protocol.json').exists():
        return status()
    old = read(previous.RUN / 'protocol.json')
    base = checked(old['source_protocol'])
    pins = deepcopy(old['source_pins'])
    cases, jobs, originals, physiology_refs = [], {}, {}, {}
    for case in old['cases']:
        cid = case['case_id']
        # Scoring sources stay opaque until the full prediction set is locked.
        allowed = ('case_id', 'compound_id', 'drug', 'cluster_id', 'study_id',
                   'reference_source', 'target_contract', 'approximations', 'nca_policy')
        cases.append({k: deepcopy(case[k]) for k in allowed if k in case})
        jobs[cid] = {k: old['jobs'][cid][k] for k in ('case_id', 'compound_id', 'drug', 'replaceable')}
        for formula in FORMULAS:
            key = formula + '__' + cid
            request = previous.frozen.gap.request(case, 'target', formula)
            if previous.eligible_cl(request) != jobs[cid]['replaceable']:
                raise ValueError('CL protection policy differs')
            # Checks original request and pristine native export, never a fitted export.
            phys = previous.frozen.prior.get_phys(base, case, 'target', formula, request)
            if phys.get('observations_used') is not False or phys.get('solver_calls') != 0:
                raise ValueError('Direct prediction requires uncalibrated physiology')
            originals[key] = request
            physiology_refs[key] = base['physiology_cache'][cid + '__target__' + formula]
    priors = {m: ref(previous.RUN / 'priors' / (m + '.json')) for m in METHODS}
    for source in [*priors.values(), ref(Path(__file__)), ref(previous.RUN/'protocol.json'),
                   ref(previous.RUN/'results.json')]:
        pins[source['path']] = source['sha256']
    protocol = dict(version=VERSION, created_at=time.time(), source_pins=pins,
        source_protocol=old['source_protocol'], cases=cases, jobs=jobs, priors=priors,
        originals=originals, physiology_refs=physiology_refs, methods=list(METHODS), formulas=list(FORMULAS),
        calibrated_results=ref(previous.RUN/'results.json'),
        primary_formula='PKSimStandard', sensitivity_formula='RodgersRowland',
        log_parameters=[0., 0., 0., 0.], calibration=False, observation_driven_repair=False,
        direct_development_points=0, calibrated_development_points=5,
        analysis=dict(seed=20261012, cluster_resamples=2000, ties_relative_tolerance=1e-6,
                      scopes=['changed_twenty_four', 'all_thirty'], primary='curve'),
        monitor_seconds=1200, directory_limit_bytes=10*1024**3, max_workers=6,
        limitations=['Previously viewed drugs; retrospective development comparison.',
            'Only CL predictor changes. Other evidence and frozen RF fu remain shared.',
            'Six qualified independent CL inputs are protected; direct outputs are shared across predictors.',
            'Formulae are reported separately; no per-drug or aggregate outcome selection.',
            'Calibrated context also changes partition, exchange, RBC scaling, formula selection and repair. Its gain is not attributable to CL calibration alone.',
            'Common clinical references cover six classes; absent references remain missing.',
            'Source hashes may cover observation files, but direct prediction never decodes their concentration values.'])
    previous.frozen.verify(pins)
    save(RUN/'protocol.json', protocol)
    save(RUN/'state.json', dict(status='prepared', completed=0, total=180))
    return status()


def task(key, protocol):
    formula, method, cid = key.split('__', 2)
    original = protocol['originals'][formula+'__'+cid]
    physiology = checked(protocol['physiology_refs'][formula+'__'+cid])
    prior_ref = protocol['priors'][method]
    prior = checked(prior_ref)
    begun = time.monotonic()
    try:
        result = direct_predict(original, physiology,
            prior['predictions'][protocol['jobs'][cid]['compound_id']], method, prior_ref)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        result = dict(status='failed', reason=f'{type(exc).__name__}: {exc}',
                      calibration_performed=False, repair_performed=False)
    return dict(**result, case_id=cid, method=method, formula=formula,
                replaceable=protocol['jobs'][cid]['replaceable'], prior=prior_ref,
                physiology=protocol['physiology_refs'][formula+'__'+cid],
                wall_seconds=time.monotonic()-begun, locked_at=time.time())


def expected_keys(protocol):
    return {f+'__'+m+'__'+cid for f in FORMULAS for m in METHODS for cid in protocol['jobs']}


def require_prediction_locks(protocol, predictions):
    if predictions.get('protocol') != ref(RUN/'protocol.json'):
        raise ValueError('Prediction protocol differs')
    if set(predictions.get('units', {})) != expected_keys(protocol):
        raise ValueError('All 180 attempts must finish before scoring')
    stamp = predictions.get('all_predictions_locked_at')
    if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
        raise ValueError('Predictions are not globally locked')
    for key, unit in predictions['units'].items():
        formula, method, cid = key.split('__', 2)
        if (unit['formula'], unit['method'], unit['case_id']) != (formula, method, cid):
            raise ValueError('Prediction key and identity differ')
        if unit['replaceable'] != protocol['jobs'][cid]['replaceable'] or unit['prior'] != protocol['priors'][method]:
            raise ValueError('Prediction input provenance differs')
        if (unit['status'] not in ('predicted', 'failed') or not math.isfinite(unit['locked_at'])
                or unit['locked_at'] > stamp):
            raise ValueError('Attempt is not locked')
        if unit['status'] == 'predicted':
            if unit['log_parameters'] != [0., 0., 0., 0.] or unit['calibration_performed'] or unit['repair_performed']:
                raise ValueError('Direct trajectory was calibrated or repaired')
            cl=previous.clearance_inputs(unit['request'])
            if not math.isclose(unit['effective_total_clearance_l_h'],cl['total_clearance_l_h'],rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('Direct CL readback differs from request')
    return True


def run(workers=6):
    prepare()
    if not 1 <= workers <= 6:
        raise ValueError('Use one to six workers')
    p = read(RUN/'protocol.json')
    previous.frozen.verify(p['source_pins'])
    with (RUN/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(RUN/'state.json')
        if state['status'] == 'complete':
            return status()
        document = (read(RUN/'predictions.json') if (RUN/'predictions.json').exists()
                    else dict(protocol=ref(RUN/'protocol.json'), units={}))
        if document['protocol'] != ref(RUN/'protocol.json'):
            raise ValueError('Resume protocol changed')
        if 'all_predictions_locked_at' in document:
            return report()
        start = time.monotonic()
        keys = sorted(expected_keys(p))
        # Only RF executes protected cases, which have identical inputs for every predictor.
        pending = [k for k in keys if k not in document['units']
                   and (k.split('__',2)[1]=='rf' or p['jobs'][k.split('__',2)[2]]['replaceable'])]
        state.update(status='predicting', pid=os.getpid(), completed=len(document['units']),
                     started_at=state.get('started_at', time.time()))
        save(RUN/'state.json',state)
        last_guard=time.monotonic(); since_save=0
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                active={}
                while pending or active:
                    if (RUN/'PAUSE').exists():
                        raise RuntimeError('Paused by project control')
                    if time.monotonic()-last_guard>=p['monitor_seconds']:
                        previous.frozen.verify(p['source_pins'])
                        if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())>p['directory_limit_bytes']:
                            raise RuntimeError('Direct batch exceeds 10 GiB')
                        last_guard=time.monotonic()
                    while pending and len(active)<workers:
                        key=pending.pop(0);active[pool.submit(task,key,p)]=key
                    done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                    for future in done:
                        key=active.pop(future); document['units'][key]=future.result();since_save+=1
                    if since_save>=24 or not pending and not active:
                        save(RUN/'predictions.json',document)
                        state['completed']=len(document['units']);save(RUN/'state.json',state)
                        print(f"Direct solves: {len(document['units'])}/156",flush=True);since_save=0
            for f in FORMULAS:
                for cid,job in p['jobs'].items():
                    if job['replaceable']:continue
                    source_key=f+'__rf__'+cid
                    for m in METHODS[1:]:
                        unit=deepcopy(document['units'][source_key]);unit.update(method=m,
                            prior=p['priors'][m],reused_from=source_key)
                        document['units'][f+'__'+m+'__'+cid]=unit
            previous.frozen.verify(p['source_pins'])
            document['all_predictions_locked_at']=time.time()
            document['resources']=dict(unique_prediction_attempts=156,logical_reuses=24,
                successful_unique_predictions=sum(u['status']=='predicted' and not u.get('reused_from') for u in document['units'].values()),
                wall_seconds=time.monotonic()-start,workers=workers,calibrations=0)
            require_prediction_locks(p,document)
            save(RUN/'predictions.json',document)
            state.update(status='predictions_locked',completed=180,
                         predictions=ref(RUN/'predictions.json'))
            save(RUN/'state.json',state)
        except BaseException as exc:
            save(RUN/'predictions.json',document)
            state.update(status='paused',reason=str(exc),completed=len(document['units']))
            save(RUN/'state.json',state)
            raise
        return report()


def summarize(records, enrolled):
    result={}
    for ep in ENDPOINTS:
        values={r['case_id']:previous.metric(r['score'],ep) for r in records}
        values={k:v for k,v in values.items() if v is not None and math.isfinite(v)}
        if ep=='tmax':
            stats=dict(n=len(values),mae_h=sum(values.values())/len(values) if values else None)
        else:
            clusters={r['case_id']:r['cluster_id'] for r in records if r['case_id'] in values}
            stats=clustered_geomean(values,clusters,seed=20261012,resamples=2000)
            stats['within_2fold']=sum(v<=2 for v in values.values())
        stats['unavailable']=enrolled-len(values)
        stats['weak_prediction_n']=sum(ep in (r['score'] or {}).get('weak_endpoints',[]) for r in records)
        stats['weak_reference_n']=sum(ep in (r['score'] or {}).get('reference_weak_endpoints',[]) for r in records)
        result[ep]=stats
    return dict(enrolled=enrolled,predicted=sum(r['status']=='predicted' for r in records),metrics=result)


def paired(numerator, denominator, ids):
    a={r['case_id']:r for r in numerator};b={r['case_id']:r for r in denominator}
    out={}
    for ep in ENDPOINTS:
        if ep=='tmax':continue
        av={cid:previous.metric(a[cid]['score'],ep) for cid in ids}
        bv={cid:previous.metric(b[cid]['score'],ep) for cid in ids}
        stats=paired_ratio(av,bv,{cid:a[cid]['cluster_id'] for cid in ids},
                           expected_ids=ids,resamples=2000,seed=20261012)
        ratios=stats['per_case_ratio'].values()
        stats.update(improved=sum(v<1-1e-6 for v in ratios),
                     worsened=sum(v>1+1e-6 for v in ratios),
                     tied=sum(abs(v-1)<=1e-6 for v in ratios))
        out[ep]=stats
    return out


def report():
    p=read(RUN/'protocol.json');state=read(RUN/'state.json')
    if state.get('status') not in ('predictions_locked','complete') or not state.get('predictions'):
        raise ValueError('State must pin all predictions before scoring')
    if state['predictions']['path'] != str((RUN/'predictions.json').relative_to(ROOT)):
        raise ValueError('Unexpected locked prediction bundle')
    document=checked(state['predictions'])
    require_prediction_locks(p,document)
    previous.frozen.verify(p['source_pins'])
    if state['status']=='complete':
        checked(state['results']);return status()
    # Observation-value decoding starts only here, after every trajectory is locked.
    cases={c['case_id']:c for c in p['cases']};records=[]
    for key,u in sorted(document['units'].items()):
        score=previous.frozen.data.score_case(cases[u['case_id']],u) if u['status']=='predicted' else None
        records.append(dict(case_id=u['case_id'],drug=cases[u['case_id']]['drug'],
            cluster_id=cases[u['case_id']]['cluster_id'],method=u['method'],formula=u['formula'],
            replaceable=u['replaceable'],status=u['status'],reason=u.get('reason'),
            score=previous.common.compact(score),prediction_key=key,reused_from=u.get('reused_from')))
    calibrated=checked(p['calibrated_results'])['records']
    summary={};calibration_comparisons={}
    for formula in FORMULAS:
        summary[formula]={};calibration_comparisons[formula]={}
        for scope in p['analysis']['scopes']:
            ids=sorted(cid for cid,j in p['jobs'].items() if scope=='all_thirty' or j['replaceable'])
            subset=[r for r in records if r['formula']==formula and r['case_id'] in ids]
            groups={m:[r for r in subset if r['method']==m] for m in METHODS}
            summary[formula][scope]={m:summarize(groups[m],len(ids)) for m in METHODS}
            for m in METHODS[1:]:
                summary[formula][scope][m]['paired_to_rf']=paired(groups[m],groups['rf'],ids)
            calibration_comparisons[formula][scope]={m:paired(
                [r for r in calibrated if r['method']==m and r['case_id'] in ids],groups[m],ids) for m in METHODS}
    result=dict(version=VERSION,protocol=ref(RUN/'protocol.json'),predictions=ref(RUN/'predictions.json'),
                predictions_locked_at=document['all_predictions_locked_at'],scored_at=time.time(),
                records=records,summary=summary,calibration_comparisons=calibration_comparisons,
                calibrated_results=p['calibrated_results'],
                interpretation='Separate information regimes, not an isolated causal effect of CL recalibration.')
    save(RUN/'results.json',result)
    with (RUN/'endpoints.csv').open('w',newline='') as handle:
        fields=['case_id','drug','method','formula','endpoint','prediction','reference','unit','error',
                'error_metric','status','replaceable','reason','reference_reason','weak_tail','reference_weak_tail']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in records:
            for ep in ENDPOINTS:
                v=(r['score'] or {}).get('endpoints',{}).get(ep,{})
                writer.writerow(dict(case_id=r['case_id'],drug=r['drug'],method=r['method'],formula=r['formula'],
                    endpoint=ep,prediction=v.get('value'),reference=v.get('reference'),unit=v.get('unit'),
                    error=previous.metric(r['score'],ep),error_metric='curve_GMFE' if ep=='curve' else v.get('error_metric'),
                    status=r['status'],replaceable=r['replaceable'],reason=r['reason'] or v.get('reason'),
                    reference_reason=v.get('reference_reason'),weak_tail=v.get('weak_tail'),
                    reference_weak_tail=v.get('reference_weak_tail')))
    state.update(status='complete',completed=180,finished_at=time.time(),results=ref(RUN/'results.json'))
    save(RUN/'state.json',state)
    return status()


def status():
    state=read(RUN/'state.json') if (RUN/'state.json').exists() else dict(status='not_prepared',completed=0,total=180)
    return {k:state[k] for k in ('status','completed','total','pid','reason') if k in state}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','run','status','report'])
    ap.add_argument('--workers',type=int,default=6);ap.add_argument('--background',action='store_true')
    args=ap.parse_args()
    if args.background:
        if args.command!='run':raise ValueError('Background applies only to run')
        prepare()
        with (RUN/'run.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run','--workers',str(args.workers)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        (RUN/'run.pid').write_text(str(proc.pid)+'\n');answer=dict(pid=proc.pid,log=str(RUN/'run.log'))
    else:
        answer=prepare() if args.command=='prepare' else run(args.workers) if args.command=='run' else report() if args.command=='report' else status()
    print(json.dumps(answer,ensure_ascii=False))


if __name__=='__main__':
    main()
