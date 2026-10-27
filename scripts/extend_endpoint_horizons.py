#!/usr/bin/env python3
"""One bounded extension of model time horizons; no new observations or parameter fit."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
import numpy as np
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint, run_evidence_blueprint
from pbpk_kb.blueprints.endpoint_prediction import predict_requested_endpoints
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BASELINE=Path('experiments/blueprint_design/multidrug_endpoints_20260923/evaluation')
DEFAULT_OUTPUT=Path('experiments/blueprint_design/multidrug_endpoints_20260923/extended_horizons')
IV_ENDPOINTS=['plasma_concentration_time','auc','cmax','tmax','c_last','terminal_half_life','auc_inf','clearance','vz','mrt','vss']
RTOL=1e-4
ATOL=1e-9

def load(path):return json.loads(path.read_text())
def now():return datetime.now(timezone.utc).isoformat()
def relative(path):return str(path.relative_to(ROOT))
def unchanged(pins):return all((ROOT/p).is_file() and sha256(ROOT/p)==digest for p,digest in pins.items())
def compact_endpoint(value):return {k:v for k,v in value.items() if k!='diagnostics'}
def prepare(baseline):
    protocol=load(baseline/'protocol.json')
    if not unchanged(protocol['pins']):raise ValueError('Baseline code or input pins have changed; do not silently compare different algorithms')
    summary=load(baseline/'summary.json')
    if summary['status']!='completed' or summary['successful_native_jobs']!=len(protocol['cases']):raise ValueError('A completed baseline is required')
    pins=dict(protocol['pins'])
    for file in (Path(__file__),baseline/'protocol.json',baseline/'summary.json',baseline/'predictions_lock.json'):
        pins[relative(file)]=sha256(file)
    locks=load(baseline/'predictions_lock.json')['predictions']
    cases=[]
    for row in protocol['cases']:
        cid=row['case_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',cid):raise ValueError('Unsafe case ID')
        casepath=baseline/cid
        old=EvidenceBlueprintRequest.model_validate(row['request']).model_dump(mode='json',exclude_none=True)
        recorded=EvidenceBlueprintRequest.model_validate(load(casepath/'request.json')).model_dump(mode='json',exclude_none=True)
        if old!=recorded or old['partition_method']!='PKSimStandard' or old['dosing']['route']!='intravenous':raise ValueError('Baseline request disagreement')
        if old['target']['endpoints']!=IV_ENDPOINTS:raise ValueError('Preserve the same eleven endpoint requests')
        if sha256(casepath/'prediction_locked.json')!=locks[cid]:raise ValueError('Baseline prediction lock mismatch')
        endpoint=load(casepath/'endpoints.json')
        lam=endpoint.get('terminal_diagnostics',{}).get('lambda_z_per_h')
        oldend=old['target']['window_end_h']
        valid_lambda=isinstance(lam,(float,int)) and not isinstance(lam,bool) and math.isfinite(lam) and lam>0
        suggested=max(2*oldend,old['dosing']['start_h']+10*math.log(2)/lam) if valid_lambda else max(2*oldend,48.)
        newend=float(math.ceil(min(168.,suggested)))
        if newend<=oldend:raise ValueError('This cohort must allow a strict extension within the 168h cap')
        request=deepcopy(old)
        request['target']['window_end_h']=newend
        request['target']['terminal_fit']['start_h']=2*newend/3
        request['target']['terminal_fit']['end_h']=newend
        request=EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json',exclude_none=True)
        oldcopy=deepcopy(old);newcopy=deepcopy(request)
        for q in (oldcopy,newcopy):
            q['target'].pop('window_end_h')
            q['target']['terminal_fit'].pop('start_h');q['target']['terminal_fit'].pop('end_h')
        if oldcopy!=newcopy:raise ValueError('Drug, dosing, physiology, algorithm, endpoints or quality gates changed')
        assembled=assemble_evidence_blueprint(request,ROOT)
        if assembled['status']!='assembled':raise ValueError('Extended request fails construction: '+cid)
        for name in ('request.json','result.json','endpoints.json','prediction_locked.json'):
            pins[relative(casepath/name)]=sha256(casepath/name)
        cases.append({'case_id':cid,'compound_id':row['compound_id'],'compound_name':row['compound_name'],
            'study_id':row['study_id'],'role':row['role'],'baseline_dir':relative(casepath),'baseline_request':old,
            'request':request,'old_window_end_h':oldend,'new_window_end_h':newend,
            'model_lambda_z_per_h_used_for_horizon':float(lam) if valid_lambda else None,
            'horizon_capped_at_168h':suggested>168.,'drug_dosing_physiology_and_gates_unchanged':True,
            'notes':row.get('notes',[])})
    return cases,pins,summary['endpoint_prediction_coverage']

def compare_original_window(case, output):
    oldpath=ROOT/case['baseline_dir'];newpath=output/case['case_id']
    oldcurve=load(oldpath/'prediction_locked.json');newcurve=load(newpath/'prediction_locked.json')
    for curve in (oldcurve,newcurve):
        if curve['time_unit']!='min' or curve['unit']!='umol/l':raise ValueError('Unexpected trajectory units')
    t=np.asarray(oldcurve['time_min'],dtype=float);a=np.asarray(oldcurve['concentration'],dtype=float)
    nt=np.asarray(newcurve['time_min'],dtype=float);nc=np.asarray(newcurve['concentration'],dtype=float)
    if t[0]<nt[0] or t[-1]>nt[-1]:raise ValueError('Old window outside extended trajectory')
    b=np.interp(t,nt,nc)
    absdiff=np.abs(a-b)
    eps=ATOL+RTOL*np.abs(a)
    passed=bool(np.all(absdiff<=eps))
    req=case['baseline_request'];mw=next(e['value'] for e in req['evidence'] if e['parameter']=='molecular_weight')
    context={**req['dosing'],'single_dose':True}
    target=deepcopy(req['target']);target['endpoints']=['auc','cmax','tmax','c_last'];target['primary_endpoint']='auc';target.pop('terminal_fit',None)
    original=load(oldpath/'endpoints.json')['endpoints']
    repeated=predict_requested_endpoints(newcurve,target,mw,context=context)['endpoints']
    full=load(newpath/'endpoints.json')['endpoints']
    details={'status':'passed' if passed else 'failed','comparison':'new trajectory interpolated at every original model output time; no new observed data',
        'original_window_h':[float(t[0]/60),float(t[-1]/60)],'compared_points':int(len(t)),
        'tolerance':{'relative':RTOL,'absolute_umol_l':ATOL},'max_absolute_difference_umol_l':float(absdiff.max()),
        'max_fraction_of_tolerance':float((absdiff/eps).max()),
        'baseline_original_window':{key:compact_endpoint(original[key]) for key in target['endpoints']},
        'extended_curve_clipped_to_original_window':repeated,
        'extended_full_window':{key:compact_endpoint(full[key]) for key in ('auc','cmax','tmax','c_last')},
        'interpretation':'Original-window and extended-full-window AUC are different quantities. Extension adds numerical model outputs, not observations or evidence of improved accuracy.'}
    save(newpath/'original_window_consistency.json',details)
    return details

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    baseline=inside(args.baseline,ROOT,exists=True);output=inside(args.output,ROOT)
    # Restrict this single-purpose supplement to its dedicated result folder.
    if output!=ROOT/DEFAULT_OUTPUT:raise ValueError('Use the dedicated extended_horizons output directory')
    cases,pins,baseline_coverage=prepare(baseline)
    if args.dry_run:
        print(json.dumps({'status':'validated_without_native_execution','cases':len(cases),'horizons':[{'case_id':c['case_id'],'old_h':c['old_window_end_h'],'new_h':c['new_window_end_h']} for c in cases]},indent=2));return 0
    if output.exists():raise FileExistsError('One extension only; existing results are never overwritten')
    output.mkdir(parents=True)
    for case in cases:
        path=output/'requests'/(case['case_id']+'.json');path.parent.mkdir(exist_ok=True)
        save(path,case['request']);pins[relative(path)]=sha256(path)
    protocol={'schema_version':'bounded-horizon-extension-1','started_at':now(),'baseline':relative(baseline),'pins':pins,
        'cases':cases,'planned_native_jobs':len(cases),'one_extension_per_case':True,'horizon_rule':'ceil(min(168,max(2*Told,dose_start+10*ln(2)/lambda_model))); if no finite positive lambda use ceil(min(168,max(2*Told,48)))',
        'lambda_source':'Already frozen model trajectory, never observed terminal concentrations; model-adaptive numerical output only',
        'terminal_window_rule':'[2*Tnew/3,Tnew], fixed before rerun; all original quality gates unchanged',
        'parameters_optimized':False,'mechanisms_changed':False,'new_observations_added':False,'accuracy_improvement_test':False,
        'independent_validation':False,'old_window_consistency_tolerance':{'relative':RTOL,'absolute_umol_l':ATOL},
        'resources':{'workers':args.workers,'cpu_threads_per_job':2,'native_timeout_seconds':240,'max_native_output_bytes_per_job':128*1024**2,'solver':'CPU_CVODES','gpu_used':False},
        'baseline_endpoint_prediction_coverage':baseline_coverage}
    save(output/'protocol.json',protocol)
    save(output/'runner.json',{'pid':os.getpid(),'started_at':now(),'project':str(ROOT),'purpose':'bounded endpoint horizon supplement'})
    started=time.monotonic();results=[]
    def progress(stage,status='running'):
        save(output/'progress.json',{'stage':stage,'status':status,'started_at':protocol['started_at'],'elapsed_seconds':round(time.monotonic()-started,2),'planned_native_jobs':len(cases),'completed_native_jobs':len(results),'successful_native_jobs':sum(r['result']['status']=='predicted' for r in results)})
    def run(case):
        row={k:v for k,v in case.items() if k not in ('request','baseline_request')}
        try:
            row['result']=run_evidence_blueprint(request=case['request'],output_dir=output/case['case_id'],project_root=ROOT,execute=True)
        except Exception as error:
            row['result']={'status':'failed','error':type(error).__name__+': '+str(error)}
        return row
    progress('native_prediction')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,case) for case in cases]
        for future in as_completed(futures):
            row=future.result();results.append(row);progress('native_prediction')
            print(json.dumps({'case':row['case_id'],'status':row['result']['status'],'completed':len(results),'total':len(cases)}),flush=True)
    results.sort(key=lambda r:r['case_id'])
    locks={r['case_id']:sha256(output/r['case_id']/'prediction_locked.json') for r in results if r['result']['status']=='predicted'}
    save(output/'predictions_lock.json',{'locked_at':now(),'predictions':locks,'locked_before_consistency_checks':True})
    progress('original_window_consistency')
    coverage={name:{'requested':len(cases),'completed':0,'not_estimable':0,'native_unavailable':0} for name in IV_ENDPOINTS}
    for row in results:
        for name,counts in coverage.items():
            status=row['result'].get('endpoint_statuses',{}).get(name)
            counts['completed' if status=='completed' else 'not_estimable' if status=='not_estimable' else 'native_unavailable']+=1
        if row['result']['status']=='predicted':
            try:
                case=next(c for c in cases if c['case_id']==row['case_id'])
                details=compare_original_window(case,output)
                row['original_window_consistency']={k:v for k,v in details.items() if k not in ('baseline_original_window','extended_curve_clipped_to_original_window','extended_full_window')}
                ep=load(output/row['case_id']/'endpoints.json')
                row['endpoint_values']={k:compact_endpoint(v) for k,v in ep['endpoints'].items() if k!='plasma_concentration_time'}
                row['terminal_diagnostics']={k:v for k,v in ep.get('terminal_diagnostics',{}).items() if k!='fit_times_h'}
            except Exception as error:row['original_window_consistency']={'status':'failed','error':type(error).__name__+': '+str(error)}
    stable=unchanged(pins)
    prediction_stable=all(sha256(output/c/'prediction_locked.json')==h for c,h in locks.items())
    success=sum(r['result']['status']=='predicted' for r in results)
    consistent=sum(r.get('original_window_consistency',{}).get('status')=='passed' for r in results)
    status='completed' if stable and prediction_stable and success==consistent==len(cases) else 'completed_with_failures'
    summary={'status':status,'elapsed_seconds':round(time.monotonic()-started,2),'attempted_cases':len(cases),'successful_native_jobs':success,'consistent_original_windows':consistent,'all_attempts_retained':True,'compound_count':len({r['compound_id'] for r in results}),
        'code_and_inputs_unchanged':stable,'predictions_unchanged':prediction_stable,'parameters_optimized':False,'accuracy_improvement_claim':False,'new_observations_added':False,'independent_validation':False,
        'endpoint_prediction_coverage_before':baseline_coverage,'endpoint_prediction_coverage_after':coverage,'results':results,
        'interpretation':'Endpoint computability supplement only. Longer simulation and declared model-tail extrapolation do not add clinical validation; full-window AUC must not be compared with old observed-window AUC as if the target were unchanged.'}
    save(output/'summary.json',summary);progress('completed',status)
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)
    return 0 if status=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
