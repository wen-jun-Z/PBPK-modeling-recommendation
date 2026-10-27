#!/usr/bin/env python3
"""Frozen ten-drug validation: prepare / run / status / report / cleanup.

Fits decode development observations only. Target outcomes are opened only by
report after every attempted prediction has been locked. All paths are local.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from run_complete_reference_pilot import read, verify
from run_renal_blueprint_pilot import embedded_pins

RUN = ROOT/'experiments/blueprint_design/validation_20260929'
CURRENT = ROOT/'experiments/blueprint_design/evidence_priority_20260929/run'
BASE = ROOT/'experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug/run'
FORMULAS = ('PKSimStandard', 'RodgersRowland')
METHODS = ('ours', 'fixed_pbpk', 'rf_only', 'two_compartment_prior', 'two_compartment')
SAMPLING = ('five', 'full')
VERSION = 'blueprint-validation-1'
LIMIT = 10*1024**3

def now(): return datetime.now(timezone.utc).isoformat()
def ref(path, location='Frozen validation input or output'):
    path=inside(path,ROOT,exists=True)
    return {'path':str(path.relative_to(ROOT)), 'sha256':sha256(path), 'location':location}
def uid(cid, sampling, method): return '__'.join((cid, sampling, method))
def checked(record):
    verify({record['path']:record['sha256']})
    return read(ROOT/record['path'])

def directory_bytes(directory):
    total=0
    for path in directory.rglob('*'):
        try:
            if path.is_file() and not path.is_symlink(): total+=path.stat().st_size
        except FileNotFoundError:
            pass  # Atomic writes and locked-fit cleanup can remove a just-listed file.
    return total

def select_five(document):
    """Time-only farthest log-time selection; ties <=1e-12 use earlier time."""
    times = [float(r['time_min']) for r in document['observations']]
    start = float(document['dosing'].get('start_h', 0))*60
    end = start+float(document['dosing']['infusion_minutes'])
    if len(times)<5 or any(not math.isfinite(t) or t<=start for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Five-point selection requires positive unique ordered postdose times')
    selected = {0, min(range(len(times)), key=lambda i:(abs(times[i]-end), times[i])), len(times)-1}
    logs = [math.log(t-start) for t in times]
    while len(selected)<5:
        options = [i for i in range(len(times)) if i not in selected]
        if sum(times[i]>end for i in selected)<3:
            options = [i for i in options if times[i]>end]
        if not options: raise ValueError('Cannot retain three postinfusion samples')
        distances = {i:min(abs(logs[i]-logs[j]) for j in selected) for i in options}
        maximum = max(distances.values())
        selected.add(min(i for i in options if maximum-distances[i]<=1e-12))
    if sum(times[i]>end for i in selected)<3: raise ValueError('Insufficient postinfusion samples')
    return sorted(selected)

def make_subset(document, source):
    indices = select_five(document)
    # Whitelist metadata: never carry raw_observations, full-curve moments or endpoints.
    keys = ('schema_version','compound_id','compound_name','study_id','study_label','role',
            'matrix','time_unit','unit','molecular_weight_g_mol','human','dosing')
    subset = {key:deepcopy(document[key]) for key in keys if key in document}
    subset['observations'] = [deepcopy(document['observations'][i]) for i in indices]
    subset['selection'] = {'rule':'time-only-five-1', 'original_indices':indices,
                           'original_source':source, 'concentrations_used_for_selection':False,
                           'full_curve_initialization_allowed':False}
    return subset

def jobs(protocol):
    result = []
    for case in protocol['cases']:
        for sampling in SAMPLING:
            for method in ('ours','fixed_pbpk','rf_only'):
                if method=='rf_only' and not case['effective_evidence_ablation']: continue
                for formula in (FORMULAS[:1] if method=='fixed_pbpk' else FORMULAS):
                    result.append((case,sampling,method,formula,48 if method=='fixed_pbpk' else 24))
    return result

def prepare(run, workers=14):
    from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
    if run.exists(): raise FileExistsError('Existing batch retained; use run/status to resume')
    original=read(CURRENT/'protocol.json')
    verify(original['source_pins'])
    protocol={'version':VERSION,'prepared_at':now(),'workers':workers,'native_threads':2,
      'native_timeout_seconds':180,'directory_limit_bytes':LIMIT,'monitor_seconds':1200,
      'bootstrap':{'seed':20260929,'resamples':2000,'unit':'study cluster','confidence':.95},
      'methods':list(METHODS),'sampling':list(SAMPLING),'cases':[],
      'selection':'Development-only objective; all target predictions locked before clinical scoring',
      'interpretation':'Previously inspected development cohort; not a blinded or independent new study',
      'clearance_policy':'only_nonrenal_when_net_renal_evidence','warm_start':False,
      'native_budgets':{'ours':[24,24],'fixed_pbpk':[48],'rf_only':[24,24]},
      'classical_budget':{'starts':16,'max_nfev_per_start':400,'prior_weight':.1,
        'objective':'mean squared log concentration error + 0.1*(ln(CL/CLprior)/ln(2))^2',
        'equal_compute_to_pbpk_claimed':False},
      'automation_scope':'From precurated structured requests; existing manual literature curation excluded',
      'source_pins':deepcopy(original['source_pins']),
      'renal_references':ref(ROOT/'experiments/blueprint_design/renal_pilot_20260929/references.json'),
      'limitations':['No organ-specific validation, LLM or new learner.','Clinical references may be missing; never imputed.',
        'Transfer pairs differ by study, not only dose/infusion.','Fixed renal input is not verified patient renal clearance.',
        'Six unchanged ablations reuse Ours; not additional evidence contributions.',
        'Two-compartment controls use body-weight allometry; full PBPK physiological covariates differ.'],
      'sensitivity_exclusions':['Nemonoxacin','Dexmedetomidine','Midazolam'],
      'engineering_checks':{}}
    for path in [CURRENT/'protocol.json',Path(__file__),ROOT/'scripts/blueprint_validation_classical.py',
                 ROOT/'scripts/run_classical_pk_comparator.py',*sorted((ROOT/'src/pbpk_kb/blueprints').glob('*.py'))]:
        protocol['source_pins'][str(path.relative_to(ROOT))]=sha256(path)
    run.mkdir(parents=True)
    for original_case in original['cases']:
        case=deepcopy(original_case); cid=case['case_id']; folder=run/'inputs'/cid; folder.mkdir(parents=True)
        full=checked(case['development_source']); sparse=make_subset(full,case['development_source'])
        save(folder/'five.json',sparse)
        case['sampling_sources']={'full':case['development_source'],
            'five':{**ref(folder/'five.json'),'role':'calibration_data_reconstruction'}}
        case['requests']={
            'evidence':{'development':ref(CURRENT/'inputs'/cid/'development.json'),
                        'target':ref(CURRENT/'inputs'/cid/'target.json')},
            'rf':{'development':ref(BASE/'inputs'/cid/'development_request.json'),
                  'target':ref(BASE/'inputs'/cid/'target_request.json')}}
        evidence=checked(case['requests']['evidence']['development']); rf=checked(case['requests']['rf']['development'])
        case['effective_evidence_ablation']=bool(case['updated_inputs'])
        if not case['effective_evidence_ablation']:
            if evidence!=rf or checked(case['requests']['evidence']['target'])!=checked(case['requests']['rf']['target']):
                raise ValueError('Alias requires exactly identical requests: '+cid)
        case['cluster_id']='Pletz2004' if case['drug'] in ('Ertapenem','Ceftriaxone') else case['study_id']
        case['transfer_group']=('primary' if case['drug'] in ('Treprostinil','Cefepime','Midazolam','Remdesivir')
                                else 'confounded' if case['drug']=='Dexmedetomidine' else 'unchanged')
        for kind, request_refs in case['requests'].items():
            for phase, record in request_refs.items():
                request=checked(record)
                construction=assemble_evidence_blueprint(request,ROOT)
                if construction['status']!='assembled': raise ValueError('Invalid structured request: '+cid+'/'+kind+'/'+phase)
                protocol['source_pins'].update(embedded_pins(request))
        protocol['source_pins'].update(embedded_pins(case))
        protocol['cases'].append(case)
    protocol['source_pins'].update(embedded_pins(protocol['renal_references']))
    planned=jobs(protocol)
    protocol['engineering_checks']={'five_point_subset_count':10,'same_requests_for_six_aliases':True,
        'request_assembly_checks':40,'target_outcomes_decoded':False,'warm_starts':0,
        'pbpk_fit_jobs':len(planned),'native_logical_trial_budget':sum(j[-1] for j in planned),
        'comparison_units':100,'unique_prediction_units':88,'analytic_units':40}
    if len(planned)!=76 or sum(j[-1] for j in planned)!=2304: raise ValueError('Unexpected experiment count')
    verify(protocol['source_pins']); save(run/'protocol.json',protocol)
    summary={'status':'prepared','prepared_at':now(),'protocol':ref(run/'protocol.json'),
       'fits':{},'units':{},'resources':{},'monitor':{},'all_attempts_locked_before_scoring':False}
    save(run/'summary.json',summary)
    return summary

class Halt(RuntimeError): pass

class Controller:
    def __init__(self,run,protocol,summary):
        self.run,self.protocol,self.summary=run,protocol,summary
        self.lock=threading.RLock(); self.paused=threading.Event(); self.closed=threading.Event()
        self.started=time.monotonic(); self.prior=summary.get('elapsed_seconds',0)
        self.last_size=0; self.last_audit=0; self.peak_tree_rss=0
    def persist(self,**values):
        with self.lock:
            self.summary.update(values,updated_at=now(),elapsed_seconds=round(self.prior+time.monotonic()-self.started,2))
            save(self.run/'summary.json',self.summary)
    def pause(self,reason):
        self.paused.set(); self.persist(status='paused',pause_reason=reason)
    def before(self):
        if self.paused.is_set(): raise Halt(self.summary.get('pause_reason','Paused'))
        self.audit()
    def audit(self,force=False):
        with self.lock:
            t=time.monotonic()
            if force or t-self.last_size>=60:
                size=directory_bytes(self.run)
                self.last_size=t
                self.summary['monitor'].update(bytes=size,checked_at=now(),directory_growth_bytes=size-self.summary['monitor'].get('bytes',0))
                if size>LIMIT: self.pause('Batch directory exceeds 10 GiB'); raise Halt('Disk threshold')
            if force or t-self.last_audit>=1200:
                try: verify(self.protocol['source_pins']); checked(self.summary['protocol'])
                except Exception as exc:
                    self.pause('Frozen source changed: '+str(exc)); raise Halt(str(exc)) from exc
                self.last_audit=t; self.summary['monitor']['source_hashes_checked_at']=now()
            self.persist()
    def watch(self):
        while not self.closed.wait(10):
            try:
                # Inspect this process and descendants only; never enumerate other services.
                import psutil
                process=psutil.Process(os.getpid()); own=[process]+process.children(recursive=True)
                rss=0; count=0
                for child in own:
                    try:
                        if child.uids().real==os.getuid(): rss+=child.memory_info().rss;count+=1
                    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                self.peak_tree_rss=max(self.peak_tree_rss,rss)
                with self.lock: self.summary['monitor'].update(own_process_count=count,own_tree_rss_bytes=rss,peak_own_tree_rss_bytes=self.peak_tree_rss)
                self.audit()
            except Halt: pass
            except Exception as exc: self.pause('Monitor failure: '+str(exc))

def cleanup_fit(folder):
    """Only unselected successful native executions of a locked new-policy fit."""
    path=folder/'mapping_locked.json'
    if not path.exists(): return 0
    mapping=read(path); verify(mapping['trial_artifact_pins'])
    retained={str((ROOT/p).resolve()) for p in mapping['trial_artifact_pins']}
    removed=0
    for execution in folder.glob('attempt_*/execution_*'):
        receipt=execution/'completion.json'
        if not receipt.exists(): continue
        completion=read(receipt)
        if completion.get('infrastructure_failure'): continue
        result=checked(completion['result'])
        if result.get('status')!='predicted': continue
        if any(p.startswith(str(execution.resolve())+os.sep) for p in retained): continue
        shutil.rmtree(execution); removed+=1
        if not any(execution.parent.iterdir()): execution.parent.rmdir()
    return removed

def execute(run,workers=14,auto_report=False):
    from pbpk_kb.blueprints.evidence_preserving_calibration import fit_mapping,predict_mapping,InfrastructurePause,_infrastructure_problem
    from blueprint_validation_classical import fit_predict
    protocol=read(run/'protocol.json'); summary=read(run/'summary.json')
    checked(summary['protocol']); verify(protocol['source_pins'])
    if summary['status'] in ('predictions_locked','reported'):
        if auto_report: return reporting(run)
        return summary
    with (run/'.run.lock').open('a+') as lease:
        try: fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This batch already has an active runner')
        import psutil
        save(run/'process.json',{'pid':os.getpid(),'created':psutil.Process().create_time(),'project':str(ROOT),'run':str(run)})
        controller=Controller(run,protocol,summary); controller.audit(force=True)
        monitor=threading.Thread(target=controller.watch,daemon=True);monitor.start()
        controller.persist(status='fitting_development',pause_reason=None)
        def fit_job(job):
            case,sampling,method,formula,budget=job; cid=case['case_id']; key=uid(cid,sampling,method)+'__'+formula
            controller.before(); start=time.monotonic()
            request=checked(case['requests']['rf' if method=='rf_only' else 'evidence']['development'])
            request['partition_method']=formula
            folder=run/'fits'/cid/sampling/method/formula
            with controller.lock: summary['fits'][key]={'status':'running','budget':budget}
            controller.persist()
            try:
                fit=fit_mapping(reference_request=request,training_source=case['sampling_sources'][sampling],
                    output_dir=folder,project_root=ROOT,max_evaluations=budget,before_trial=controller.before)
                record={'status':'completed','mapping':fit['mapping'],'eligible':fit['eligible'],
                    'budget':budget,'decision':fit.get('decision'),'elapsed_seconds':fit.get('elapsed_seconds',time.monotonic()-start),
                    'actual_native_calls':fit.get('total_native_calls',budget),'evaluations':fit['evaluation_count'],
                    'failed_evaluations':fit['failed_evaluation_count']}
                cleanup_fit(folder)
            except (InfrastructurePause,Halt,OSError) as exc:
                controller.pause(str(exc)); record={'status':'paused','reason':str(exc),'budget':budget}
            except Exception as exc:
                # Unexpected implementation/provenance failures must not become model rankings.
                controller.pause(type(exc).__name__+': '+str(exc));record={'status':'paused','reason':str(exc),'budget':budget}
            with controller.lock: summary['fits'][key]=record
            controller.persist();return record
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures=[executor.submit(fit_job,job) for job in jobs(protocol)]
                for future in as_completed(futures):
                    try: future.result()
                    except Halt: pass
            if controller.paused.is_set(): return summary
            controller.audit(force=True);controller.persist(status='predicting')
            def predict_unit(case,sampling,method):
                controller.before();cid=case['case_id'];key=uid(cid,sampling,method);start=time.monotonic()
                with controller.lock: previous=summary['units'].get(key)
                if previous and previous['status'] in ('predicted','failed'):
                    for field in ('prediction','request','mapping','fit','renal_prediction'):
                        if previous.get(field): checked(previous[field])
                    return previous
                kind='rf' if method=='rf_only' else 'evidence'
                target_ref=case['requests'][kind]['target']
                unit={'case_id':cid,'sampling':sampling,'method':method,'request':target_ref,
                      'resources':{'manual_interventions':0,'cache_reused':False,'peak_memory':None}}
                folder=run/'predictions'/cid/sampling/method
                if method.startswith('two_compartment'):
                    result=fit_predict(case=case,request=checked(case['requests']['evidence']['development']),
                        training_source=case['sampling_sources'][sampling],output_dir=folder,project_root=ROOT,
                        with_cl_prior=method=='two_compartment_prior')
                    unit.update({k:result[k] for k in ('status','prediction','request','fit')})
                    unit['resources'].update(result['resources'],actual_new_native_calls=0,
                        analytic_residual_calls=result['resources'].get('analytic_cpu_residual_calls',0))
                else:
                    fits=[summary['fits'][key+'__'+formula] for formula in (FORMULAS[:1] if method=='fixed_pbpk' else FORMULAS)]
                    valid=[f for f in fits if f.get('eligible') and f.get('decision')]
                    unit['resources'].update(actual_new_native_calls=sum(f.get('actual_native_calls',0) for f in fits),
                        failed_native_trials=sum(f.get('failed_evaluations',0) for f in fits),cached_native_calls=0)
                    if not valid:
                        unit.update(status='failed',failure='No eligible development mapping')
                    else:
                        chosen=min(valid,key=lambda f:f['decision']['objective']); mapping=checked(chosen['mapping'])
                        request=checked(target_ref);request['partition_method']=mapping['construction_partition_method']
                        # Reuse a fully completed target after parent crash; an incomplete directory stays as diagnosis.
                        if (folder/'result.json').exists():
                            result=read(folder/'result.json')
                            construction=read(folder/'construction.json')
                            if construction['mapping'] != {k:v for k,v in chosen['mapping'].items() if k in ('path','sha256')}:
                                raise Halt('Completed prediction mapping differs from selected mapping')
                            from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
                            normalized=EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json',exclude_none=True)
                            if read(folder/'request.json')!=normalized: raise Halt('Completed target request changed')
                            if result.get('status')=='predicted' and sha256(folder/'prediction_locked.json')!=result.get('prediction_sha256'):
                                raise Halt('Completed target prediction checksum changed')
                        else:
                            if folder.exists():
                                abandoned=folder.with_name(folder.name+'_interrupted')
                                if abandoned.exists(): raise Halt('Repeated interrupted target; manual inspection needed')
                                folder.rename(abandoned)
                            result=predict_mapping(request=request,mapping_path=ROOT/chosen['mapping']['path'],output_dir=folder,
                                project_root=ROOT,renal_outputs=bool(case.get('renal_enabled') and method!='rf_only'))
                        if _infrastructure_problem(result): raise Halt('Target infrastructure failure; diagnostics retained')
                        unit.update(mapping=chosen['mapping'],request=ref(folder/'request.json'))
                        if result.get('status')=='predicted':
                            unit.update(status='predicted',prediction=ref(folder/'prediction_locked.json'))
                            if result.get('renal_prediction'): unit['renal_prediction']=result['renal_prediction']
                        else: unit.update(status='failed',failure=result)
                        unit['resources']['actual_new_native_calls']+=1
                    unit['resources']['wall_seconds']=sum(f.get('elapsed_seconds',0) for f in fits)+time.monotonic()-start
                unit['elapsed_seconds']=unit['resources'].get('wall_seconds',time.monotonic()-start)
                with controller.lock: summary['units'][key]=unit
                controller.persist();return unit
            units=[(c,s,m) for c in protocol['cases'] for s in SAMPLING for m in METHODS
                   if m!='rf_only' or c['effective_evidence_ablation']]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures=[executor.submit(predict_unit,*item) for item in units]
                for future in as_completed(futures):
                    try: future.result()
                    except Exception as exc: controller.pause(type(exc).__name__+': '+str(exc))
            if controller.paused.is_set(): return summary
            for case in protocol['cases']:
                if case['effective_evidence_ablation']: continue
                for sampling in SAMPLING:
                    source=uid(case['case_id'],sampling,'ours');key=uid(case['case_id'],sampling,'rf_only')
                    unit=deepcopy(summary['units'][source]);unit.update(method='rf_only',cached_from=source)
                    unit['resources']={'wall_seconds':0.,'actual_new_native_calls':0,'cached_native_calls':
                        summary['units'][source]['resources'].get('actual_new_native_calls',0),
                        'manual_interventions':0,'cache_reused':True,'peak_memory':None}
                    unit['elapsed_seconds']=0.;summary['units'][key]=unit
            controller.audit(force=True)
            if len(summary['units'])!=100: raise RuntimeError('Incomplete comparison matrix')
            for unit in summary['units'].values():
                for field in ('prediction','request','mapping','fit','renal_prediction'):
                    if unit.get(field): checked(unit[field])
            usages=[resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)]
            summary['resources']={'process_tree_cpu_seconds_this_invocation':sum(r.ru_utime+r.ru_stime for r in usages),
                'peak_process_tree_rss_bytes':controller.peak_tree_rss,'workers':workers,'native_threads':2,
                'manual_interventions_after_frozen_request':0,'literature_curation_automated':False,
                'per_unit_peak_memory_available':False}
            controller.persist(status='predictions_locked',all_attempts_locked_before_scoring=True,predictions_locked_at=now())
        except Exception as exc:
            controller.pause(type(exc).__name__+': '+str(exc));raise
        finally:
            controller.closed.set();monitor.join(timeout=2)
    if auto_report and summary['status']=='predictions_locked': reporting(run)
    return read(run/'summary.json')

def reporting(run):
    summary=read(run/'summary.json');checked(summary['protocol'])
    if not summary.get('all_attempts_locked_before_scoring') or len(summary['units'])!=100:
        raise RuntimeError('All 100 prediction attempts must be locked before reading target observations')
    from report_blueprint_validation import report
    result=report(run,project_root=ROOT)
    if result.get('status')=='integrity_failed': raise RuntimeError('Report integrity checks failed; inspect results.json')
    # Do not mutate prediction summary after the report locks it.
    return result

def cleanup(run):
    protocol=read(run/'protocol.json');verify(protocol['source_pins'])
    count=sum(cleanup_fit(p.parent) for p in (run/'fits').glob('*/*/*/*/mapping_locked.json'))
    return {'removed_unselected_successful_executions':count,'raw_data_and_paper_preserved':True}

def status(run):
    s=read(run/'summary.json');p=read(run/'protocol.json');counts={}
    for fit in s['fits'].values():counts[fit['status']]=counts.get(fit['status'],0)+1
    trials=sum(read(path).get('native_evaluations',0) for path in (run/'fits').glob('*/*/*/*/progress.json'))
    units={}
    for unit in s['units'].values():units[unit['status']]=units.get(unit['status'],0)+1
    active=False
    if (run/'process.json').exists():
        import psutil
        proc=read(run/'process.json')
        try:
            obj=psutil.Process(proc['pid']);active=obj.uids().real==os.getuid() and obj.create_time()==proc['created'] and obj.is_running() and obj.status()!=psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    return {'status':s['status'],'runner_alive':active,'fits':counts,'planned_fits':76,
       'completed_or_started_trial_points':trials,'planned_trial_budget':2304,'prediction_units':units,
       'expected_prediction_units':100,'elapsed_seconds':s.get('elapsed_seconds'),
       'monitor':s.get('monitor'),'pause_reason':s.get('pause_reason'),
       'results_available':(run/'results.json').exists(),'log':str(run/'run.log')}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','status','report','cleanup'])
    parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--workers',type=int,default=14)
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--auto-report',action='store_true')
    args=parser.parse_args();run=inside(args.run_dir,ROOT)
    if not 1<=args.workers<=14: parser.error('workers must be 1..14')
    if args.command=='prepare': result=prepare(run,args.workers)
    elif args.command=='status': result=status(run)
    elif args.command=='report': result=reporting(run)
    elif args.command=='cleanup': result=cleanup(run)
    elif args.background:
        if not (run/'protocol.json').exists(): raise RuntimeError('Run prepare first')
        if status(run)['runner_alive']: raise RuntimeError('Batch already running')
        with (run/'run.log').open('ab') as log:
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'run','--run-dir',str(run),'--workers',str(args.workers)]
            if args.auto_report: command+=['--auto-report']
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result={'started_pid':process.pid,'run':str(run)}
    else: result=execute(run,args.workers,args.auto_report)
    print(json.dumps(result if args.command not in ('prepare','run') or args.background else {'status':result.get('status')},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
