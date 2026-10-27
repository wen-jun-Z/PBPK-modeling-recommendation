"""Run audited development cases, budget comparisons and dimension controls.

This finite queue does not upgrade historical reconstructions to independent
clinical validation. All external predictions are locked across every block
before external observations are decoded. No task is silently retried.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
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
from run_blueprint_cohort import ROOT, inside, process_state, stop_child, disk_bytes
import run_blueprint_experiments as single

METHODS = ['M0','M1','M2','M3','M4','M5','M6','A1','A2','A3','E0']
SEEDS = [17,43,101]


def read(path):
    return json.loads(Path(path).read_text())


def block_definitions():
    return [
        {'block_id':'main_5000','methods':METHODS,'budget':5000,'matched_count':None},
        {'block_id':'budget_500','methods':['M3','M4','M5','M6'],'budget':500,'matched_count':None},
        {'block_id':'budget_2000','methods':['M3','M4','M5','M6'],'budget':2000,'matched_count':None},
        {'block_id':'matched_1','methods':['M5','M6'],'budget':5000,'matched_count':1},
    ]


def make_plan(cases):
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Cases must be nonempty with unique IDs')
    tasks=[]
    if any(not isinstance(c.get('case_id'),str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,160}',c['case_id']) for c in cases):
        raise ValueError('Filesystem-safe case IDs are required')
    for block in block_definitions():
        for case in cases:
            if case['tier'] not in {'development','reconstruction','bottom_up_external'}:
                raise ValueError('This runner does not admit unqualified independent-main tasks')
            if block['matched_count'] is not None and case['parameter_count'] < block['matched_count']:
                raise ValueError('The predeclared matched dimension is unavailable')
            jobs=[{'method':m,'seed':s} for m in block['methods']
                  for s in ([17] if m in {'M0','M1','E0'} else SEEDS)]
            tasks.append({**case,**block,'task_id':block['block_id']+'__'+case['case_id'],
                          'jobs':jobs,'status':'pending'})
    return tasks


def collect_results(tasks):
    """One record per prespecified job, including unexecuted/failed jobs.

    Budgets, matched dimensions, compounds and seeds remain separate. These
    correlated runs are not independent clinical samples or a pooled effect.
    """
    rows=[]
    for task in tasks:
        directory=inside(task['output'],exists=False) if task.get('output') else None
        method_rows = read(directory/'results.json')['methods'] if directory and (directory/'results.json').is_file() else []
        results={(r['method'],r['seed']):r for r in method_rows}
        evaluations = read(directory/'external_results.json')['evaluations'] if directory and (directory/'external_results.json').is_file() else []
        external={(r['method'],r['seed']):r['report'] for r in evaluations}
        for job in task['jobs']:
            key=(job['method'],job['seed'])
            result=results.get(key)
            rows.append({**{k:task[k] for k in ('task_id','block_id','case_id','compound_id','tier','budget','matched_count')},
                **job,'task_status':task['status'],'method_status':result['status'] if result else 'not_completed',
                'reason':result.get('blocked_reason') if result else task.get('reason'),
                'result':result,'external_evaluation':external.get(key)})
    return {'rows':rows,'planned_jobs':len(rows),
        'method_status_counts':dict(Counter(r['method_status'] for r in rows)),
        'independent_drug_count':0,'main_benchmark_completed':False,
        'aggregation_policy':'No pooling across budgets, parameter dimensions, seeds or scientific tiers'}


def validate_inputs(manifest_path, proposals):
    manifest=read(inside(manifest_path))
    cases=[]
    from pbpk_kb.blueprints.experiment_runtime import candidate_space, validate_case
    from pbpk_kb.blueprints.qwen_proposals import load_proposer
    for row in manifest['cases']:
        path=inside(row['bundle_path'])
        if file_sha(path)!=row['bundle_sha256']:
            raise ValueError('Frozen case bundle changed')
        case,external=single.load_case_bundle(path)
        validate_case(case)
        if (row['case_id']!=case['case_id'] or row['compound_id']!=case['compound_id']
                or row['tier']!=case.get('tier','development')):
            raise ValueError('Frozen case identity mismatch')
        candidates=candidate_space(case,root=ROOT)
        load_proposer(proposals,case,candidates,project_root=ROOT)
        cases.append({**row,'tier':case.get('tier','development'),
                      'parameter_count':len(case['parameter_search']['parameters']),
                      'external_scenario_count':len(external)})
    return manifest,cases


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--qwen-proposals',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--scenario-workers',type=int,default=3)
    parser.add_argument('--max-output-gib',type=int,default=16)
    parser.add_argument('--task-timeout-hours',type=float,default=4)
    args=parser.parse_args(argv)
    parallel=validate_native_parallelism(cpu_threads=args.cpu_threads,
         scenario_workers=args.scenario_workers,drug_workers=args.workers)
    if not 1<=args.max_output_gib<=32 or not .1<=args.task_timeout_hours<=8:
        raise ValueError('Unsupported output or wall-time limit')
    output=inside(args.output,exists=False)
    if output.exists():raise FileExistsError('Use a new output; previous attempts are preserved')
    proposals=inside(args.qwen_proposals)
    manifest,cases=validate_inputs(args.manifest,proposals)
    tasks=make_plan(cases)
    output.mkdir(parents=True)
    code=sorted({'scripts/run_blueprint_reviewed_suite.py','scripts/run_blueprint_cohort.py',
        'scripts/run_blueprint_experiments.py',
        *(str(p.relative_to(ROOT)) for p in (ROOT/'src/pbpk_kb/blueprints').glob('*.py')),
        'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
        'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R'})
    config={'version':'blueprint-reviewed-development-suite-v1','frozen_at':datetime.now(UTC).isoformat(),
        'input_manifest':str(inside(args.manifest).relative_to(ROOT)),
        'input_manifest_sha256':file_sha(inside(args.manifest)),
        'qwen_proposals':str(proposals.relative_to(ROOT)),
        'qwen_manifest_sha256':file_sha(proposals/'manifest.json'),
        'parallelism':parallel,'workers':args.workers,'cpu_threads':args.cpu_threads,
        'scenario_workers':args.scenario_workers,'tasks':tasks,'blocks':block_definitions(),
        'planned_method_jobs':sum(len(t['jobs']) for t in tasks),
        'native_solver':'CPU_CVODES','gpu_role':'batched_offline_Qwen_and_curve_scoring',
        'max_output_bytes':args.max_output_gib*1024**3,
        'max_task_seconds':args.task_timeout_hours*3600,
        'scientific_scope':'audited_historical_development_and_reconstruction',
        'strict_main_eligible_drugs':0,'main_benchmark_completed':False,
        'planned_paper_design_runs':7032,'paper_protocol_frozen':False,
        'unlaunched_protocol_blocks':['independent_main_60_drugs','480_request_variants','120_mechanism_tasks'],
        'code_sha256':{p:file_sha(ROOT/p) for p in code}}
    save(output/'configuration.json',config)
    with zipfile.ZipFile(output/'runner_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in code:archive.write(ROOT/p,arcname=p)
    save(output/'runner.json',{'pid':os.getpid(),'start_ticks':process_state(os.getpid())['ticks'],
                             'started_at':datetime.now(UTC).isoformat()})
    start=time.monotonic();active={};halt=None;stopping=False;failure_streak=0;previous_error=None

    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)

    def publish(stage):
        method_counts=Counter()
        for task in tasks:
            if task.get('output'):
                directory=inside(task['output'],exists=False)
                if (directory/'results.json').is_file():
                    method_counts.update(r['status'] for r in read(directory/'results.json')['methods'])
                if (directory/'progress.json').is_file():
                    p=read(directory/'progress.json')
                    task['progress']={k:p.get(k) for k in ('status','stage','method','seed','completed_jobs','planned_jobs','error')}
        save(output/'queue.json',{'tasks':tasks},replace=True)
        progress={'status':'stopped' if halt else 'completed' if stage=='finished' else 'running',
            'stage':stage,'elapsed_seconds':time.monotonic()-start,'active_tasks':list(active),
            'task_status_counts':dict(Counter(t['status'] for t in tasks)),
            'method_status_counts':dict(method_counts),'planned_method_jobs':config['planned_method_jobs'],
            'strict_main_eligible_drugs':0,'main_benchmark_completed':False,'halt_reason':halt}
        save(output/'progress.json',progress,replace=True)
        print(json.dumps(progress,ensure_ascii=False),flush=True)

    try:
        publish('submitting')
        while active or any(t['status']=='pending' for t in tasks):
            if stopping:halt='User stop requested for this project suite'
            if disk_bytes(output)>config['max_output_bytes']:halt='Suite disk cap exceeded'
            if time.monotonic()-start>24*3600:halt='Suite wall-time cap exceeded'
            if halt:break
            for task in tasks:
                if len(active)>=args.workers:break
                if task['status']!='pending':continue
                for p,sha in config['code_sha256'].items():
                    if file_sha(ROOT/p)!=sha:raise RuntimeError('Executable changed; no new job submitted')
                if file_sha(proposals/'manifest.json')!=config['qwen_manifest_sha256']:
                    raise RuntimeError('Frozen Qwen manifest changed')
                if file_sha(inside(task['bundle_path']))!=task['bundle_sha256']:
                    raise RuntimeError('Frozen case changed')
                directory=output/'tasks'/task['task_id'];directory.parent.mkdir(exist_ok=True)
                log=directory.with_suffix('.log')
                task.update(status='launching',output=str(directory.relative_to(ROOT)),log=str(log.relative_to(ROOT)))
                save(output/'queue.json',{'tasks':tasks},replace=True)
                command=[sys.executable,str(ROOT/'scripts/run_blueprint_experiments.py'),
                    '--case-file',str(inside(task['bundle_path'])),'--output',str(directory),
                    '--qwen-proposals',str(proposals),'--device','cuda',
                    '--cpu-threads',str(args.cpu_threads),'--scenario-workers',str(args.scenario_workers),
                    '--max-solver-calls',str(task['budget']),'--methods',*task['methods'],'--defer-external-scoring']
                if task['matched_count'] is not None:
                    command.extend(['--exact-free-parameter-count',str(task['matched_count'])])
                env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
                with log.open('xb') as stream:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state=process_state(child.pid)
                task.update(status='running',pid=child.pid,start_ticks=state['ticks'] if state else None,
                            started_at=datetime.now(UTC).isoformat())
                active[task['task_id']]={'process':child,'task':task,'started':time.monotonic()}
            for name,value in list(active.items()):
                child,task=value['process'],value['task']
                if time.monotonic()-value['started']>config['max_task_seconds']:
                    stop_child(child);task['reason']='Task wall-time cap exceeded'
                log=inside(task['log'])
                if log.stat().st_size>20*1024**2:
                    stop_child(child);task['reason']='Task log cap exceeded'
                if child.poll() is None:continue
                directory=inside(task['output'],exists=False)
                completed=(directory/'completion.json').is_file()
                locked=(directory/'external_prediction_locks.json').is_file()
                if child.returncode==0 and completed and (locked or not task['external_scenario_count']):
                    task['status']='predictions_locked' if task['external_scenario_count'] else 'development_completed'
                    previous_error=None;failure_streak=0
                else:
                    p=read(directory/'progress.json') if (directory/'progress.json').is_file() else {}
                    task.update(status='failed',reason=task.get('reason') or p.get('error') or 'Missing completed submission')
                    signature=task['reason'].split(':',1)[0]
                    failure_streak=failure_streak+1 if signature==previous_error else 1;previous_error=signature
                    if failure_streak>=3:halt='Three consecutive equivalent execution errors'
                task.update(returncode=child.returncode,ended_at=datetime.now(UTC).isoformat())
                del active[name]
            publish('native_queues')
            if active:time.sleep(10)
        if halt:
            for value in active.values():
                stop_child(value['process']);value['task'].update(status='interrupted',reason=halt)
            active.clear()
        else:
            locks={t['task_id']:file_sha(inside(t['output'])/'external_prediction_locks.json')
                   for t in tasks if t['status']=='predictions_locked'}
            save(output/'prediction_barrier.json',{'locked_at':datetime.now(UTC).isoformat(),
                 'all_tasks_terminal':True,'lock_sha256':locks,'task_states':[{k:t.get(k) for k in ('task_id','status','reason')} for t in tasks]})
            publish('external_scoring')
            for task in tasks:
                if stopping:
                    halt='User stop requested during external scoring'
                    break
                if task['status']!='predictions_locked':continue
                directory=inside(task['output']);case,external=single.load_case_bundle(task['bundle_path'])
                lockfile=directory/'external_prediction_locks.json'
                if file_sha(lockfile)!=locks[task['task_id']]:raise ValueError('Locked predictions changed')
                locked=read(lockfile)['locks']
                records=[read(directory/'methods'/('%s_seed_%d'%(r['method'],r['seed']))/'external_predictions.json') for r in locked]
                if [r['locked_selection'] for r in records]!=locked:raise ValueError('Prediction identity mismatch')
                try:
                    single.score_external(case,external,records,directory,lambda **kw:None,device='cuda')
                    task['status']='evaluated'
                except (ValueError,RuntimeError,OSError) as exc:
                    task.update(status='evaluation_failed',reason=type(exc).__name__+': '+str(exc))
                publish('external_scoring')
        publish('finished')
        save(output/'results.json',collect_results(tasks))
        save(output/'completion.json',{'status':'stopped' if halt else 'finished',
             'task_status_counts':dict(Counter(t['status'] for t in tasks)),
             'elapsed_seconds':time.monotonic()-start,'main_benchmark_completed':False,
             'scientific_accepted':False,'halt_reason':halt})
        return 1 if halt else 0
    except BaseException as exc:
        halt=type(exc).__name__+': '+str(exc)
        for value in active.values():
            stop_child(value['process']);value['task'].update(status='interrupted',reason=halt)
        active.clear();publish('finished');raise


if __name__=='__main__':
    raise SystemExit(main())
