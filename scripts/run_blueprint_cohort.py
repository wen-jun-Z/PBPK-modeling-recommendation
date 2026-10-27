"""Run a frozen multi-drug method benchmark with isolated, bounded child queues.

All drug predictions are locked before any final-study observations are decoded.
Source-model reconstruction and independently qualified tests remain separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile

from pbpk_kb.blueprints.experiment_runtime import file_sha, save, validate_native_parallelism

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['M0','M1','M2','M3','M4','M5','M6','A1','A2','A3','E0']
SEEDS = [17,43,101]


def inside(value, *, exists=True):
    path=(ROOT/value).resolve(strict=exists)
    if path==ROOT or not path.is_relative_to(ROOT):
        raise ValueError('Cohort resources must remain inside this project')
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def method_jobs():
    return [{'method':method,'seed':seed} for method in METHODS
            for seed in ([17] if method in {'M0','M1','E0'} else SEEDS)]


def load_manifest(path):
    manifest=read_json(inside(path));cases=manifest['cases']
    if not isinstance(cases,list) or not cases:
        raise ValueError('A cohort needs explicit drug cases')
    if manifest.get('methods',METHODS)!=METHODS or manifest.get('seeds',SEEDS)!=SEEDS:
        raise ValueError('This benchmark uses all registered methods and seeds')
    seen=set();drugs=set()
    for case in cases:
        identifier=case.get('case_id','')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,160}',identifier) or identifier in seen:
            raise ValueError('Unique filesystem-safe case IDs are required')
        seen.add(identifier)
        if not isinstance(case.get('compound_id'),str) or not case['compound_id']:
            raise ValueError('Every case needs a compound identity')
        if case.get('tier') not in {'development','reconstruction','bottom_up_external','main_holdout'}:
            raise ValueError('Every case needs an explicit scientific tier')
        if case.get('status') not in {'ready','blocked'}:
            raise ValueError('Eligibility must be ready or blocked before queue launch')
        if case['status']=='ready':
            bundle=inside(case['bundle_path'])
            if file_sha(bundle)!=case['bundle_sha256']:
                raise ValueError('Frozen input bundle changed')
            payload=read_json(bundle)
            if payload['case']['case_id']!=identifier or payload['case']['compound_id']!=case['compound_id']:
                raise ValueError('Case identity does not match the frozen bundle')
            if payload['case'].get('tier')!=case['tier'] or not payload.get('external_scenarios'):
                raise ValueError('A method benchmark requires a matching tier and external study designs')
            case['expected_scenario_ids']=[s['scenario_id'] for s in payload['external_scenarios']]
            if all(s.get('study_group_id') for s in payload['external_scenarios']):
                case['expected_study_group_ids']=sorted({s['study_group_id'] for s in payload['external_scenarios']})
            drugs.add(case['compound_id'])
        elif not case.get('blocked_reason'):
            raise ValueError('Blocked eligibility requires a concrete reason')
    if len(drugs)<2:
        raise ValueError('A multi-drug queue needs at least two executable compounds')
    return manifest


def planned_jobs(cases):
    # Pre-execution eligibility exclusions stay in the frozen screening/queue
    # denominator. Only eligible drug cases enter method performance analysis.
    return [dict(job,case_id=c['case_id'],compound_id=c['compound_id'],tier=c['tier'],
                 eligibility_status=c['status'],reason=c.get('blocked_reason'),
                 **{key:c[key] for key in ('expected_scenario_ids','expected_study_group_ids') if key in c})
            for c in cases if c['status']=='ready' for job in method_jobs()]


def process_state(pid):
    """Read only a process already identified as our launched task/descendant."""
    try:
        directory=Path('/proc')/str(pid)
        if directory.stat().st_uid!=os.getuid():return None
        fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
        return {'ticks':fields[19],'state':fields[0]}
    except (FileNotFoundError,ProcessLookupError):return None


def owned_descendants(pid):
    found=[]
    def visit(current):
        state=process_state(current)
        if state is None:return
        found.append((current,state['ticks']))
        path=Path('/proc')/str(current)/'task'/str(current)/'children'
        try:children=[int(x) for x in path.read_text().split()]
        except FileNotFoundError:children=[]
        for child in children:visit(child)
    visit(pid)
    return found


def stop_child(process):
    if process.poll() is not None:return
    owned=owned_descendants(process.pid)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid,ticks in reversed(owned):
            state=process_state(pid)
            if state and state['ticks']==ticks and state['state']!='Z':
                try:os.kill(pid,sig)
                except ProcessLookupError:pass
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:pass
        if not any((state:=process_state(pid)) and state['ticks']==ticks and state['state']!='Z'
                   for pid,ticks in owned):break


def disk_bytes(directory):
    total=0
    for path in directory.rglob('*'):
        if path.is_symlink():raise ValueError('Benchmark outputs cannot contain symlinks')
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:pass
    return total


def single_runner():
    spec=importlib.util.spec_from_file_location('blueprint_single_runner',ROOT/'scripts/run_blueprint_experiments.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4,
                        help='Concurrent drug queues (1..8), within the combined CPU affinity budget')
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--scenario-workers',type=int,default=3)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--max-output-gib',type=int,default=24)
    parser.add_argument('--case-timeout-hours',type=float,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    parallelism=validate_native_parallelism(cpu_threads=args.cpu_threads,
        scenario_workers=args.scenario_workers,drug_workers=args.workers)
    if not 1<=args.max_output_gib<=64 or not 0.1<=args.case_timeout_hours<=24:
        raise ValueError('Unsupported disk/time bound')
    manifest=load_manifest(args.input_manifest);cases=manifest['cases'];by_id={c['case_id']:c for c in cases}
    output=inside(args.output,exists=False);started=time.monotonic()
    code_paths=sorted({'scripts/run_blueprint_cohort.py','scripts/run_blueprint_experiments.py',
       *(str(p.relative_to(ROOT)) for p in (ROOT/'src/pbpk_kb/blueprints').glob('*.py')),
       'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
       'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R'})
    controls={'workers':args.workers,'cpu_threads_per_native_process':args.cpu_threads,
              'parallel_scenarios':args.scenario_workers,'device':args.device,
              'maximum_output_bytes':args.max_output_gib*1024**3,
              'maximum_seconds_per_case':args.case_timeout_hours*3600}
    if output.exists():
        if not args.resume:raise FileExistsError('Use --resume for an existing frozen cohort')
        config=read_json(output/'configuration.json')
        if config['input_manifest_sha256']!=file_sha(inside(args.input_manifest)) or config['controls']!=controls:
            raise ValueError('Cannot resume with changed frozen inputs or controls')
        for p,h in config['code_sha256'].items():
            if file_sha(ROOT/p)!=h:raise ValueError('Cannot resume with changed frozen executable code')
        registration=read_json(output/'runner.json');state=process_state(registration['pid'])
        if state and state['ticks']==registration['start_ticks'] and state['state']!='Z':
            raise RuntimeError('This cohort already has a live registered supervisor')
        queue=read_json(output/'queue.json')['cases']
        for row in queue:
            if row['status']=='launching':
                raise RuntimeError('An interrupted launch needs its project child identity checked before resubmission')
            if row['status']=='pending' and (output/'cases'/row['case_id']).exists():
                raise RuntimeError('An unregistered prior attempt exists; refusing a duplicate drug queue')
            if row['status'] in {'running','interrupted'}:
                child=process_state(row.get('pid',0))
                if child and child['ticks']==row.get('start_ticks') and child['state']!='Z':
                    raise RuntimeError('A previous project child remains alive; do not duplicate it')
                row['status']='pending'
    else:
        if args.resume:raise FileNotFoundError('No frozen cohort exists to resume')
        output.mkdir(parents=True)
        config={'schema_version':'blueprint-cohort-1','scope':manifest.get('scope','public_model_method_benchmark'),
                'frozen_at_utc':datetime.now(UTC).isoformat(),
                'input_manifest_path':str(inside(args.input_manifest).relative_to(ROOT)),
                'input_manifest_sha256':file_sha(inside(args.input_manifest)),
                'controls':controls,'parallelism':parallelism,
                'jobs':planned_jobs(cases),'methods':METHODS,'seeds':SEEDS,
                'native_solver':'CPU_CVODES','gpu_role':'curve_interpolation_and_scoring',
                'failure_policy':'Isolate failed drugs; pause after three consecutive identical execution errors',
                'scoring_policy':'Freeze all cohort predictions before decoding final-study outcomes',
                'strict_independent_main_completed':False,
                'code_sha256':{p:file_sha(ROOT/p) for p in code_paths}}
        save(output/'configuration.json',config);save(output/'cohort.json',manifest)
        with zipfile.ZipFile(output/'runner_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for p in code_paths:archive.write(ROOT/p,arcname=p)
        queue=[{'case_id':c['case_id'],'compound_id':c['compound_id'],'tier':c['tier'],
                'status':'pending' if c['status']=='ready' else 'blocked',
                'reason':c.get('blocked_reason'),'attempt':0} for c in cases]
    save(output/'runner.json',{'pid':os.getpid(),'start_ticks':process_state(os.getpid())['ticks'],
                             'started_at_utc':datetime.now(UTC).isoformat()},replace=True)
    from pbpk_kb.blueprints.cohort_results import summarize_cohort_results
    single=single_runner();running={};halt=None;last_failure=None;failure_streak=0;stop_requested=False

    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)

    def publish(stage):
        data={}
        for row in queue:
            item={'queue_status':row['status'],'queue_error':row.get('reason')}
            if row.get('output'):
                directory=inside(row['output'],exists=False)
                for key,name in [('results','results.json'),('completion','completion.json'),('external_results','external_results.json')]:
                    if row['status']=='running' and key!='results':continue
                    p=directory/name
                    if p.is_file():item[key]=read_json(p)
                p=directory/'progress.json'
                if p.is_file():row['current_progress']=read_json(p)
            data[row['case_id']]=item
        summary=summarize_cohort_results(planned_jobs=config['jobs'],case_results=data)
        save(output/'summary.json',summary,replace=True)
        save(output/'queue.json',{'cases':queue},replace=True)
        progress={'status':'running' if stage!='finished' else ('paused' if halt else 'completed'),
                  'stage':stage,'elapsed_seconds':time.monotonic()-started,
                  'screened_cases':len(cases),'eligible_cases':sum(c['status']=='ready' for c in cases),
                  'source_screening_summary':manifest.get('summary',{}),
                  'case_status_counts':dict(Counter(row['status'] for row in queue)),
                  'active_cases':[row['case_id'] for row in queue if row['status']=='running'],
                  'planned_method_jobs':len(config['jobs']),
                  'eligible_method_jobs':sum(c['status']=='ready' for c in cases)*len(method_jobs()),
                  'method_submission_status_counts':dict(Counter(r.get('status')
                      for value in data.values() for r in value.get('results',{}).get('methods',[]))),
                  'evaluation_status_counts':summary['status_counts'],
                  'strict_independent_main_completed':False,'halt_reason':halt}
        save(output/'progress.json',progress,replace=True)
        print(json.dumps(progress,ensure_ascii=False),flush=True)

    try:
        publish('submitting_drug_queues')
        while any(row['status']=='pending' for row in queue) or running:
            if stop_requested:halt='Stop requested for this project cohort'
            if disk_bytes(output)>controls['maximum_output_bytes']:halt='Cohort output byte limit exceeded'
            if halt:break
            for row in queue:
                if len(running)>=args.workers:break
                if row['status']!='pending':continue
                for p,h in config['code_sha256'].items():
                    if file_sha(ROOT/p)!=h:raise RuntimeError('Frozen executable changed; no further jobs may be submitted')
                case=by_id[row['case_id']]
                if file_sha(inside(case['bundle_path']))!=case['bundle_sha256']:
                    raise RuntimeError('A frozen drug bundle changed')
                row['attempt']+=1
                directory=output/'cases'/row['case_id']/('attempt_%03d'%row['attempt'])
                directory.parent.mkdir(parents=True,exist_ok=True)
                log=directory.parent/('attempt_%03d.log'%row['attempt'])
                row.update(status='launching',output=str(directory.relative_to(ROOT)),log=str(log.relative_to(ROOT)))
                save(output/'queue.json',{'cases':queue},replace=True)
                command=[sys.executable,str(ROOT/'scripts/run_blueprint_experiments.py'),
                         '--case-file',str(inside(case['bundle_path'])),'--output',str(directory),
                         '--device',args.device,'--cpu-threads',str(args.cpu_threads),
                         '--scenario-workers',str(args.scenario_workers),'--defer-external-scoring']
                env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                with log.open('xb') as stream:
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                running[row['case_id']]={'process':process,'started':time.monotonic(),'row':row}
                identity=process_state(process.pid)
                row.update(status='running',pid=process.pid,start_ticks=identity['ticks'] if identity else None,
                           output=str(directory.relative_to(ROOT)),log=str(log.relative_to(ROOT)),
                           started_at_utc=datetime.now(UTC).isoformat())
                save(output/'queue.json',{'cases':queue},replace=True)
            for key,active in list(running.items()):
                process=active['process'];row=active['row']
                if time.monotonic()-active['started']>controls['maximum_seconds_per_case']:
                    stop_child(process);row['reason']='Native drug queue exceeded the frozen wall-time limit'
                if process.poll() is None:continue
                directory=inside(row['output'],exists=False)
                completion=read_json(directory/'completion.json') if (directory/'completion.json').is_file() else None
                if process.returncode==0 and completion and (directory/'external_prediction_locks.json').is_file():
                    row.update(status='predictions_locked',returncode=0);last_failure=None;failure_streak=0
                else:
                    progress=read_json(directory/'progress.json') if (directory/'progress.json').is_file() else {}
                    row.update(status='failed',returncode=process.returncode,
                               reason=row.get('reason') or progress.get('error') or 'Child exited without locked predictions; see project log')
                    signature=re.sub(r'/[^\s]+','<path>',row['reason'])
                    failure_streak=failure_streak+1 if signature==last_failure else 1;last_failure=signature
                    if failure_streak>=3:halt='Three consecutive identical execution failures: '+signature
                row['ended_at_utc']=datetime.now(UTC).isoformat();del running[key]
            publish('native_prediction_queues')
            if running:time.sleep(10)
        if halt:
            for active in running.values():
                stop_child(active['process']);active['row'].update(status='interrupted',reason=halt)
        else:
            # This barrier is persisted before opening any final-study values.
            barrier_path=output/'cohort_prediction_barrier.json'
            lock_hashes={r['case_id']:file_sha(inside(r['output'])/'external_prediction_locks.json')
                         for r in queue if r.get('output') and
                         (inside(r['output'],exists=False)/'external_prediction_locks.json').is_file()}
            barrier={
                'locked_at_utc':datetime.now(UTC).isoformat(),
                'states':[{k:r.get(k) for k in ['case_id','status','output','reason']} for r in queue],
                'prediction_lock_sha256':lock_hashes,'input_manifest_sha256':config['input_manifest_sha256'],
                'all_eligible_drug_queues_terminal':True}
            if barrier_path.exists():
                original=read_json(barrier_path)
                if (original.get('prediction_lock_sha256')!=lock_hashes
                        or original.get('input_manifest_sha256')!=config['input_manifest_sha256']):
                    raise ValueError('Cannot replace the original pre-evaluation prediction barrier')
            else:save(barrier_path,barrier)
            publish('external_scoring')
            for row in queue:
                if row['status']!='predictions_locked':continue
                directory=inside(row['output']);case,external=single.load_case_bundle(by_id[row['case_id']]['bundle_path'])
                if (directory/'external_results.json').is_file():
                    row['status']='evaluated';continue
                locks=read_json(directory/'external_prediction_locks.json')['locks']
                records=[read_json(directory/'methods'/('%s_seed_%d'%(lock['method'],lock['seed']))/'external_predictions.json') for lock in locks]
                if [record['locked_selection'] for record in records]!=locks:
                    raise ValueError('A prediction lock changed before cohort evaluation')
                try:
                    single.score_external(case,external,records,directory,lambda **kwargs:None,device=args.device)
                    row['status']='evaluated'
                except (ValueError,RuntimeError,OSError) as exc:
                    row.update(status='evaluation_failed',reason=type(exc).__name__+': '+str(exc))
                publish('external_scoring')
        publish('finished')
        save(output/'completion.json',{'status':'paused' if halt else 'queue_finished',
            'elapsed_seconds':time.monotonic()-started,'case_status_counts':dict(Counter(r['status'] for r in queue)),
            'all_eligible_queues_terminal':not halt,'strict_independent_main_completed':False,
            'performance_complete':'See summary.json; report generation does not imply all metrics are valid',
            'halt_reason':halt},replace=True)
        return 1 if halt else 0
    except BaseException as exc:
        halt=type(exc).__name__+': '+str(exc)
        for active in running.values():
            stop_child(active['process']);active['row'].update(status='interrupted',reason=halt)
        publish('finished')
        raise


if __name__=='__main__':
    raise SystemExit(main())
