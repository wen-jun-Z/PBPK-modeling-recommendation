"""Run one frozen project suite after a registered predecessor releases workers.

Only registered project PIDs and their task/TID descendants are inspected.
Predecessor processes are observed, never signaled by this launcher.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time

from watch_blueprint_suite import ProcessScope, directory_bytes, inside, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
POLL_SECONDS = 5
MAX_WAIT_SECONDS = MAX_RUN_SECONDS = 6*3600
MAX_OUTPUT_BYTES = 16*1024**3
MAX_LOG_BYTES = 20*1024**2
SUCCESSFUL_TASK_STATES = {'development_completed','evaluated'}


def project_path(value):
    return inside(value,root=ROOT)


def checksum(path):
    path=project_path(path)
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid():
        raise ValueError('Frozen resource must be an owned regular project file')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def identity(value):
    if (not isinstance(value,dict) or type(value.get('pid')) is not int or value['pid']<=0
        or not str(value.get('start_ticks','')).isdecimal()):
        raise ValueError('Registered process requires explicit PID/start ticks')
    return {'pid':value['pid'],'start_ticks':str(value['start_ticks'])}


def parallelism(workers,scenario_workers,cpu_threads,other_reserved_cpus,*,available=None):
    if (type(workers) is not int or not 1<=workers<=4 or type(scenario_workers) is not int or
        not 1<=scenario_workers<=4 or type(cpu_threads) is not int or cpu_threads not in {1,2,4} or
        type(other_reserved_cpus) is not int or other_reserved_cpus<0):
        raise ValueError('Invalid bounded worker/thread reservation')
    available=len(os.sched_getaffinity(0)) if available is None else available
    own=workers*scenario_workers*cpu_threads
    if own>48 or own+other_reserved_cpus>available:
        raise ValueError('Declared concurrent CPU reservations exceed this process affinity')
    return {'workers':workers,'scenario_workers':scenario_workers,'cpu_threads':cpu_threads,
        'native_cpu_upper_bound':own,'other_project_cpus_reserved':other_reserved_cpus,
        'declared_combined_cpu_upper_bound':own+other_reserved_cpus,'available_affinity_cpus':available,
        'scope':'Explicit project scheduling reservations; no global utilization or other-user process scan'}


class RegisteredSuite:
    """Completion means authenticated normal terminal state AND no live workers."""
    def __init__(self,directory,scope,*,expected_supervisor=None):
        self.directory,self.scope=project_path(directory),scope
        self.config=read_json(self.directory/'configuration.json')
        self.root=identity(read_json(self.directory/'runner.json'))
        if expected_supervisor is not None and self.root!=identity(expected_supervisor):
            raise ValueError('Experiment registered a different supervisor process instance')
        if self.config.get('version')!='blueprint-reviewed-development-suite-v1':
            raise ValueError('Only registered reviewed development suites can be prerequisites')
        tasks=self.config['tasks'];self.allowed={t['task_id'] for t in tasks}
        if not tasks or len(self.allowed)!=len(tasks):raise ValueError('Frozen suite has duplicate or missing tasks')
        if any(not isinstance(v,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}',v) for v in self.allowed):
            raise ValueError('Unsafe frozen task identifier')
        self.frozen={name:checksum(self.directory/name) for name in ('configuration.json','runner.json')}
        self.known={}
        self.last_progress={}

    def sample(self,*,_terminal_retry=True):
        for name,value in self.frozen.items():
            if checksum(self.directory/name)!=value:raise ValueError('Registered suite identity/configuration changed')
        roots=[self.root,*self.known.values()];tasks=[]
        queue=self.directory/'queue.json'
        if queue.exists():
            tasks=read_json(queue)['tasks']
            if len(tasks)!=len(self.allowed) or {t['task_id'] for t in tasks}!=self.allowed:
                raise ValueError('Queue differs from its frozen task registration')
            for task in tasks:
                if task.get('output') and project_path(task['output'])!=self.directory/'tasks'/task['task_id']:
                    raise ValueError('Registered task output leaves its exact suite task directory')
                if task.get('pid') is not None:
                    if task.get('start_ticks') is None:
                        raise ValueError('A registered task PID has no reusable process identity')
                    roots.append(identity(task))
        # The independently registered watcher may have observed an orphan
        # before this follower started. Its supervisor identity must match.
        monitor=self.directory/'monitor.json'
        if monitor.exists():
            record=read_json(monitor)
            if record.get('supervisor')!=self.root:
                raise ValueError('Prior monitor belongs to another supervisor')
            tracked=record.get('tracked_processes',[])
            if len(tracked)>4096:raise ValueError('Project monitor process cap exceeded')
            roots.extend(identity(p) for p in tracked)
        self.known=self.scope.discover(roots)
        progress=read_json(self.directory/'progress.json') if (self.directory/'progress.json').exists() else {}
        self.last_progress=progress
        if progress.get('halt_reason') or progress.get('status') in {'stopped','failed'}:
            raise RuntimeError('Registered suite reports an abnormal stop')
        if any(t.get('status') in {'failed','interrupted','evaluation_failed'} for t in tasks):
            raise RuntimeError('Registered suite contains a failed/interrupted task')
        completion_path=self.directory/'completion.json'
        completed=read_json(completion_path) if completion_path.exists() else None
        normal=False
        if completed is not None:
            if completed.get('status')!='finished' or completed.get('halt_reason'):
                raise RuntimeError('Registered suite completion is not successful')
            normal=(progress.get('status')=='completed' and progress.get('stage')=='finished'
                and not progress.get('active_tasks') and bool(tasks)
                and all(t.get('status') in SUCCESSFUL_TASK_STATES for t in tasks))
            counts=dict(Counter(t.get('status') for t in tasks))
            if not normal or completed.get('task_status_counts')!=counts:
                # Queue/progress can be read just before a normal terminal
                # completion is atomically published. Retry one whole snapshot.
                if _terminal_retry:return self.sample(_terminal_retry=False)
                raise RuntimeError('Completion/progress/queue records disagree or tasks remain incomplete')
        alive=self.scope.state(self.root['pid'],self.root['start_ticks']) is not None
        if not alive and not normal:
            # The supervisor writes progress, then completion, then exits. It
            # may finish between our completion read and process-state read.
            if _terminal_retry:return self.sample(_terminal_retry=False)
            raise RuntimeError('Supervisor exited without a verified complete suite')
        return normal and not self.known


def freeze_resources(stage,manifest,qwen):
    proposal=read_json(qwen/'progress.json')
    if proposal.get('state')!='completed' or proposal.get('failures',0):
        raise ValueError('All source-bound Qwen proposals must already be complete')
    plan=read_json(manifest);cases=plan['cases']
    if not cases or len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Unique nonempty frozen cases required')
    qmanifest=read_json(qwen/'manifest.json')
    if (qmanifest.get('version')!='pbpk-local-qwen-proposals-v2' or qmanifest.get('final_status')!='completed'
        or set(qmanifest.get('requests',{}))!={c['case_id'] for c in cases}
        or qmanifest.get('seeds')!=[17,43,101]):
        raise ValueError('Qwen manifest is incomplete or belongs to a different case/seed scope')
    resources=[Path(__file__).resolve(),ROOT/'scripts/watch_blueprint_suite.py',manifest,qwen/'manifest.json',qwen/'progress.json',
        *(stage/'src').rglob('*.py'),
        *(stage/'scripts'/n for n in ('run_blueprint_reviewed_suite.py','run_blueprint_experiments.py','run_blueprint_cohort.py')),
        ROOT/'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',ROOT/'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R']
    for row in cases:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',row['case_id']):raise ValueError('Unsafe case identifier')
        bundle=project_path(row['bundle_path'])
        if checksum(bundle)!=row['bundle_sha256']:raise ValueError('Manifest case checksum mismatch')
        resources.append(bundle)
        for seed in (17,43,101):
            path=qwen/'cases'/row['case_id']/f'seed_{seed}.json'
            # Freeze the actual responses as well as the top-level manifest.
            record=read_json(path)
            if record.get('case_id')!=row['case_id'] or record.get('seed')!=seed or record.get('status')!='completed':
                raise ValueError('Missing or unsuccessful Qwen case/seed record')
            if (checksum(path)!=qmanifest.get('record_sha256',{}).get(str(path.relative_to(qwen)))
                or record.get('request_sha256')!=qmanifest['requests'][row['case_id']]
                or not qmanifest.get('weights_manifest_sha256')
                or record.get('weights_manifest_sha256')!=qmanifest['weights_manifest_sha256']):
                raise ValueError('Qwen record is not bound to its frozen manifest/request/weights')
            resources.append(path)
    if not (stage/'src/pbpk_kb/blueprints/design_runtime.py').is_file():raise ValueError('Missing isolated runtime')
    return {str(project_path(p).relative_to(ROOT)):checksum(p) for p in resources}


def verify_frozen(frozen):
    for path,sha in frozen.items():
        if checksum(ROOT/path)!=sha:raise RuntimeError('Frozen input changed: '+path)


def stop_owned(scope,known,*,registered=None,sleep=time.sleep):
    """Stop only this follower's child tree, never prerequisite identities."""
    errors=[]
    for sig in (signal.SIGTERM,signal.SIGKILL):
        try:
            known.update(scope.discover(known.values()))
            if registered is not None:
                try:registered.sample()
                finally:known.update(registered.known)
        except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
        for process in reversed(list(known.values())):
            try:scope.send(process,sig)
            except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
        if known:sleep(2)
    remaining=scope.discover(known.values())
    return {'remaining_tracked_processes':list(remaining.values()),'cleanup_errors':errors}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('after-suite','stage','manifest','qwen-proposals','output','name'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--scenario-workers',type=int,default=3)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--other-reserved-cpus',type=int,default=0)
    parser.add_argument('--preflight-only',action='store_true',help='Read and validate scope/frozen inputs without registering or starting a process')
    args=parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_]{1,80}',args.name):raise ValueError('A simple project registration name is required')
    scheduling=parallelism(args.workers,args.scenario_workers,args.cpu_threads,args.other_reserved_cpus)
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):raise RuntimeError('Safe termination requires Linux pidfd support')
    after,stage,manifest,qwen,output=map(project_path,(args.after_suite,args.stage,args.manifest,args.qwen_proposals,args.output))
    if output.is_relative_to(after) or after.is_relative_to(output):raise ValueError('Follow-up and prerequisite outputs must be disjoint')
    deployment=project_path(ROOT/'runtime/deployment')
    registration=deployment/('blueprint_followup_'+args.name+'.json')
    status_path=deployment/('blueprint_followup_'+args.name+'_status.json')
    log=deployment/('blueprint_followup_'+args.name+'.log')
    if any(p.exists() or p.is_symlink() for p in (registration,status_path,log,output)):
        raise ValueError('Do not overwrite or duplicate a registered follow-up')
    scope=ProcessScope();own=scope.state(os.getpid())
    if own is None:raise RuntimeError('Could not authenticate the project launcher')
    prerequisite=RegisteredSuite(after,scope)
    if prerequisite.root['pid']==own['pid']:raise ValueError('Follower cannot await itself')
    frozen=freeze_resources(stage,manifest,qwen)
    if args.preflight_only:
        ready=prerequisite.sample()
        print(json.dumps({'status':'preflight_passed','prerequisite_ready':ready,
            'prerequisite_live_processes':len(prerequisite.known),'frozen_files':len(frozen),
            'scheduling':scheduling,'started_processes':0},ensure_ascii=False,indent=2))
        return
    deployment.mkdir(parents=True,exist_ok=True)
    command=[str(ROOT/'runtime/app-env/bin/python'),str(stage/'scripts/run_blueprint_reviewed_suite.py'),
        '--manifest',str(manifest),'--qwen-proposals',str(qwen),'--output',str(output),
        '--workers',str(args.workers),'--scenario-workers',str(args.scenario_workers),'--cpu-threads',str(args.cpu_threads),
        '--max-output-gib','16','--task-timeout-hours','4']
    state={'status':'waiting_for_previous_suite','pid':own['pid'],'start_ticks':own['start_ticks'],
        'registered_at':datetime.now(timezone.utc).isoformat(),'after_suite':str(after.relative_to(ROOT)),
        'prerequisite_supervisor':prerequisite.root,'prerequisite_configuration_sha256':prerequisite.frozen['configuration.json'],
        'output':str(output.relative_to(ROOT)),'command':command,'source_sha256':frozen,'scheduling':scheduling,
        'automatic_retries':0,'max_wait_seconds':MAX_WAIT_SECONDS,'max_run_seconds':MAX_RUN_SECONDS,
        'max_output_bytes':MAX_OUTPUT_BYTES,'max_log_bytes':MAX_LOG_BYTES,'poll_seconds':POLL_SECONDS,
        'log':str(log.relative_to(ROOT)),'no_scientific_result_selection':True,
        'process_scope':'Registered owned PID/start ticks and all task/TID children; predecessor is never signaled',
        'monitor_limitations':'Unobserved detach-and-exit between polls cannot be recovered without cgroups; SIGKILL of this follower bypasses Python cleanup.'}
    with registration.open('x') as stream:json.dump(state,stream,ensure_ascii=False,indent=2)
    write_json(status_path,state)
    stopping=False
    def request_stop(*_):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
    child,known,registered=None,{},None
    try:
        start=time.monotonic()
        while not prerequisite.sample():
            if stopping or time.monotonic()-start>MAX_WAIT_SECONDS:raise RuntimeError('Prerequisite wait stopped/expired')
            state.update(wait_elapsed_seconds=time.monotonic()-start,prerequisite_live_processes=len(prerequisite.known))
            write_json(status_path,state);time.sleep(POLL_SECONDS)
        if stopping:raise RuntimeError('Stop requested before launch')
        verify_frozen(frozen)
        # Recheck immediately before dispatch, including known orphans.
        if not prerequisite.sample():raise RuntimeError('Prerequisite acquired live workers before launch')
        parallelism(args.workers,args.scenario_workers,args.cpu_threads,args.other_reserved_cpus)
        if output.exists():raise ValueError('Output was created during the wait; refuse duplicate work')
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with log.open('xb') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        child_identity=scope.state(child.pid)
        if child_identity is None or child_identity['ppid']!=own['pid']:
            raise RuntimeError('Child exited before authenticated parent/identity registration')
        known={child.pid:child_identity}
        state.update(status='running',child=identity(child_identity),started_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path,state);start=time.monotonic()
        while True:
            known.update(scope.discover(known.values()))
            if registered is None and (output/'configuration.json').exists() and (output/'runner.json').exists():
                registered=RegisteredSuite(output,scope,expected_supervisor=child_identity)
            ready=False
            if registered is not None:
                ready=registered.sample();known.update(registered.known)
            used=directory_bytes(output)
            if stopping or time.monotonic()-start>MAX_RUN_SECONDS or used>MAX_OUTPUT_BYTES or log.stat().st_size>MAX_LOG_BYTES:
                raise RuntimeError('Follow-up stop/time/output/log limit reached')
            verify_frozen(frozen)
            returncode=child.poll()
            if returncode is not None:
                if returncode!=0 or registered is None:raise RuntimeError('Follow-up supervisor exited without a successful registered suite')
                if not ready:raise RuntimeError('Follow-up exited but left incomplete records or live project descendants')
                break
            state.update(elapsed_seconds=time.monotonic()-start,output_bytes=used,tracked_child_processes=len(scope.discover(known.values())))
            write_json(status_path,state);time.sleep(POLL_SECONDS)
        state.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),output_bytes=directory_bytes(output))
        write_json(status_path,state)
    except BaseException as exc:
        cleanup=stop_owned(scope,known,registered=registered)
        if child is not None:
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:cleanup['child_wait_timed_out']=True
        state.update(status='stopped',reason=type(exc).__name__+': '+str(exc),cleanup=cleanup)
        write_json(status_path,state)
        raise


if __name__=='__main__':main()
