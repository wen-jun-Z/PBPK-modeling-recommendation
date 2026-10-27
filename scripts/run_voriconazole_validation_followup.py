#!/usr/bin/env python3
"""Wait for the registered Vori methods, then lock predictions and seal scores.

Methods are observed, never stopped. Only children launched here can be signaled.
All waits are bounded; source changes or failures stop without automatic retries.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_blueprint_followup import RegisteredSuite,identity,stop_owned
from watch_blueprint_suite import ProcessScope,inside,read_json,write_json,directory_bytes
import hashlib

POLL_SECONDS=5
MAX_WAIT_SECONDS=12*3600
NATIVE_DISPATCH_SECONDS=4*3600
LOCK_FINALIZATION_GRACE_SECONDS=120
MAX_PREDICTION_SECONDS=NATIVE_DISPATCH_SECONDS+LOCK_FINALIZATION_GRACE_SECONDS
MAX_SCORING_SECONDS=1800
PREDICTION_BYTES=2*1024**3
SCORING_BYTES=512*1024**2
LOG_BYTES=20*1024**2
STAGE_REL='runtime/blueprint_voriconazole_main_stage_20260920'
INPUT_REL='experiments/blueprint_design/development_inputs/voriconazole_mainmodel_20260920_v2/input_package.json'
SUITE_REL='experiments/blueprint_design/voriconazole_source_suite_20260920'
PROTOCOL_REL='experiments/blueprint_design/external_validation/concentration_curve_audit_20260920/protocols'


def local(value):return inside(value,root=ROOT)


def checksum(path):
    with local(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_frozen(frozen):
    for path,value in frozen.items():
        if checksum(path)!=value:raise RuntimeError('Frozen pipeline resource changed: '+path)


class AwaitMethodsFollower:
    """Authenticate a not-yet-created suite through its immutable follower."""
    def __init__(self,registration,expected_suite,scope):
        self.path=local(registration);self.status_path=self.path.with_name(self.path.stem+'_status.json')
        self.suite_path=local(expected_suite);self.scope=scope
        self.registration=None;self.registration_sha=None;self.child=None;self.suite=None;self.last_status='awaiting_registration'

    def sample(self,*,_retry=True):
        if not self.path.exists():
            if self.registration is not None:raise RuntimeError('Registered methods follower disappeared')
            return False
        if self.registration is None:
            registration=read_json(self.path);identity(registration)
            expected=['--manifest',str(local(INPUT_REL).parent/'suite_manifest.json')]
            command=registration.get('command',[])
            if (local(registration.get('output',''))!=self.suite_path
                or len(command)<4 or command[0]!=str(ROOT/'runtime/app-env/bin/python')
                or command[1]!=str(ROOT/STAGE_REL/'scripts/run_blueprint_reviewed_suite.py')
                or command[2:4]!=expected):
                raise ValueError('Follower does not register this exact Vori stage/manifest/output')
            self.registration,self.registration_sha=registration,checksum(self.path)
        elif checksum(self.path)!=self.registration_sha:
            raise RuntimeError('Methods follower registration changed')
        registration=self.registration
        verify_frozen(registration['source_sha256'])
        status=read_json(self.status_path) if self.status_path.exists() else None
        if status is not None:
            if (identity(status)!=identity(registration) or status.get('command')!=registration['command']
                or status.get('output')!=registration['output']
                or status.get('source_sha256')!=registration['source_sha256']):
                raise ValueError('Follower status is not bound to the registered process and inputs')
            self.last_status=status.get('status')
            if self.last_status=='stopped':raise RuntimeError('Methods follower stopped; external predictions will not run')
            if self.last_status not in {'waiting_for_previous_suite','running','completed'}:
                raise ValueError('Unknown methods follower state')
            if status.get('child') is not None:
                child=identity(status['child'])
                if self.child is not None and self.child!=child:raise RuntimeError('Methods child process identity changed')
                self.child=child
            if self.child is not None and self.suite is None and (self.suite_path/'configuration.json').exists() and (self.suite_path/'runner.json').exists():
                self.suite=RegisteredSuite(self.suite_path,self.scope,expected_supervisor=self.child)
            ready=self.suite.sample() if self.suite is not None else False
            if self.last_status=='completed':
                if self.suite is None:raise RuntimeError('Completed follower has no authenticated method suite')
                return ready
        alive=self.scope.state(registration['pid'],registration['start_ticks']) is not None
        if not alive:
            # Completion may have been atomically published after the status read.
            if _retry:return self.sample(_retry=False)
            raise RuntimeError('Methods follower died without successful completion')
        return False


def frozen_sources():
    stage=ROOT/STAGE_REL
    names=['run_voriconazole_validation_followup.py','predict_voriconazole_methods_external.py',
        'score_voriconazole_methods_curves.py','score_locked_independent_curves.py',
        'predict_furosemide_external_study.py','run_blueprint_followup.py','watch_blueprint_suite.py']
    resources=[*(ROOT/'scripts'/name for name in names),local(INPUT_REL),local(INPUT_REL).parent/'suite_manifest.json',
        *(local(PROTOCOL_REL)/f'voriconazole_li2023_reference_iv_{dose}mgkg_curves.json' for dose in (4,6)),
        *(stage/'src').rglob('*.py'),*(stage/'scripts').glob('*.py'),stage/'source_registration.json',
        ROOT/'src/pbpk_kb/blueprints/design_scoring.py',ROOT/'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
        ROOT/'src/pbpk_kb/model_runtime/R/blueprint_parameter_batch.R']
    return {str(local(path).relative_to(ROOT)):checksum(path) for path in resources}


def run_child_step(*,name,command,output,timeout,byte_limit,scope,owner,state,status_path,frozen,stopping,log_path):
    """Retain orphan identities until exit, including failures during startup."""
    verify_frozen(frozen)
    if output.exists() or log_path.exists():raise ValueError('Never overwrite a prior prediction/scoring attempt')
    known={};child=None;started=time.monotonic()
    row={'name':name,'status':'starting','output':str(output.relative_to(ROOT)),
        'log':str(log_path.relative_to(ROOT)),'command':command,'timeout_seconds':timeout,'max_output_bytes':byte_limit}
    state['steps'].append(row);state.update(status='running',stage=name);write_json(status_path,state)
    try:
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with log_path.open('xb') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        child_identity=scope.state(child.pid)
        if child_identity is None or child_identity['ppid']!=owner['pid']:
            raise RuntimeError('Prediction/scoring child exited before parent/identity authentication')
        known[child.pid]=child_identity;row.update(status='running',child=identity(child_identity))
        while True:
            known.update(scope.discover(known.values()))
            used=directory_bytes(output)
            if stopping() or time.monotonic()-started>timeout or used>byte_limit or log_path.stat().st_size>LOG_BYTES:
                raise RuntimeError('Prediction/scoring stop, timeout, output or log limit reached')
            verify_frozen(frozen)
            returncode=child.poll()
            if returncode is not None:
                if returncode!=0:raise RuntimeError(name+' exited unsuccessfully')
                if scope.discover(known.values()):raise RuntimeError(name+' exited with live owned descendants')
                verify_frozen(frozen)
                break
            row.update(elapsed_seconds=time.monotonic()-started,output_bytes=used,
                tracked_processes=[identity(p) for p in scope.discover(known.values()).values()])
            write_json(status_path,state);time.sleep(POLL_SECONDS)
        row.update(status='completed',output_bytes=directory_bytes(output),elapsed_seconds=time.monotonic()-started,returncode=0)
        write_json(status_path,state)
    except BaseException:
        row.update(status='failed',cleanup=stop_owned(scope,known))
        if child is not None:
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:row['child_wait_timed_out']=True
        write_json(status_path,state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follower-registration',default='runtime/deployment/blueprint_followup_voriconazole_20260920.json')
    parser.add_argument('--output',default='experiments/blueprint_design/external_validation/voriconazole_li2023_main_methods_20260920')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    base=local(args.output);scope=ProcessScope();owner=scope.state(os.getpid())
    if owner is None:raise RuntimeError('Cannot authenticate this project pipeline')
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):raise RuntimeError('Linux pidfd support is required')
    watcher=AwaitMethodsFollower(local(args.follower_registration),local(SUITE_REL),scope)
    frozen=frozen_sources()
    if args.preflight_only:
        ready=watcher.sample()
        print(json.dumps({'status':'preflight_passed','methods_ready':ready,'frozen_files':len(frozen),
            'methods_state':watcher.last_status,'started_processes':0,'outcomes_read':False},indent=2));return
    deployment=local('runtime/deployment');deployment.mkdir(parents=True,exist_ok=True)
    registration=deployment/'blueprint_voriconazole_validation_followup.json'
    status_path=deployment/'blueprint_voriconazole_validation_followup_status.json'
    logs={name:deployment/('blueprint_voriconazole_'+name+'_20260920.log') for name in ('predictions','scoring')}
    if any(p.exists() or p.is_symlink() for p in (base,registration,status_path,*logs.values())):
        raise ValueError('Use the existing registered pipeline; do not overwrite results or retry automatically')
    state={'version':'pbpk-voriconazole-external-followup-v1','status':'waiting_for_methods',**identity(owner),
        'registered_at':datetime.now(timezone.utc).isoformat(),'follower_registration':str(watcher.path.relative_to(ROOT)),
        'expected_methods_suite':SUITE_REL,'output':str(base.relative_to(ROOT)),'frozen_input_sha256':frozen,
        'max_wait_seconds':MAX_WAIT_SECONDS,'prediction_seconds':MAX_PREDICTION_SECONDS,'prediction_max_bytes':PREDICTION_BYTES,
        'prediction_native_dispatch_seconds':NATIVE_DISPATCH_SECONDS,'prediction_lock_finalization_grace_seconds':LOCK_FINALIZATION_GRACE_SECONDS,
        'scoring_seconds':MAX_SCORING_SECONDS,'scoring_max_bytes':SCORING_BYTES,'poll_seconds':POLL_SECONDS,'steps':[],
        'automatic_retries':0,'methods_processes_never_signaled':True,'sealed_outcomes_read_by_development':False,
        'monitor_limitations':'SIGKILL bypasses Python cleanup; an unobserved detach-and-exit between polls cannot be recovered without cgroups.'}
    with registration.open('x') as stream:json.dump(state,stream,indent=2)
    write_json(status_path,state)
    stopped=False
    def request_stop(*_):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
    try:
        start=time.monotonic()
        while not watcher.sample():
            if stopped or time.monotonic()-start>MAX_WAIT_SECONDS:raise RuntimeError('Methods wait stopped or expired')
            verify_frozen(frozen)
            state.update(wait_elapsed_seconds=time.monotonic()-start,methods_state=watcher.last_status,
                methods_follower_sha256=watcher.registration_sha,methods_child=watcher.child)
            write_json(status_path,state);time.sleep(POLL_SECONDS)
        if stopped:raise RuntimeError('Stop requested before independent predictions')
        if not watcher.sample():raise RuntimeError('Methods acquired live descendants before predictions')
        state.update(methods_follower_sha256=watcher.registration_sha,methods_child=watcher.child)
        python=str(ROOT/'runtime/app-env/bin/python');predictions=base/'predictions';scores=base/'scores'
        command=[python,str(ROOT/'scripts/predict_voriconazole_methods_external.py'),'--suite',str(local(SUITE_REL)),
            '--input-package',str(local(INPUT_REL)),'--protocol-dir',str(local(PROTOCOL_REL)),
            '--output',str(predictions),'--wait-hours','0']
        run_child_step(name='predictions',command=command,output=predictions,timeout=MAX_PREDICTION_SECONDS,
            byte_limit=PREDICTION_BYTES,scope=scope,owner=owner,state=state,status_path=status_path,frozen=frozen,
            stopping=lambda:stopped,log_path=logs['predictions'])
        if stopped:raise RuntimeError('Stop requested before sealed scoring')
        # Calling the validator cannot read any sealed outcome, even on error.
        from score_voriconazole_methods_curves import validate_locked_inputs
        lock_path=predictions/'prediction_lock.json';validate_locked_inputs(lock_path)
        frozen[str(lock_path.relative_to(ROOT))]=checksum(lock_path)
        state['prediction_lock_sha256']=checksum(lock_path)
        command=[python,str(ROOT/'scripts/score_voriconazole_methods_curves.py'),'--lock',str(lock_path),'--output',str(scores)]
        run_child_step(name='scoring',command=command,output=scores,timeout=MAX_SCORING_SECONDS,
            byte_limit=SCORING_BYTES,scope=scope,owner=owner,state=state,status_path=status_path,frozen=frozen,
            stopping=lambda:stopped,log_path=logs['scoring'])
        receipt=read_json(scores/'receipt.json')
        if (receipt.get('status')!='scored_and_sealed' or receipt.get('method_jobs_expected')!=57
            or receipt.get('method_dose_rows_expected')!=114 or receipt.get('independent_drugs')!=1
            or receipt.get('independent_studies')!=1 or receipt.get('metrics_exposed_to_method_development') is not False
            or receipt.get('lock_sha256')!=checksum(lock_path)):
            raise RuntimeError('Scorer did not return the full frozen denominator and sealed receipt')
        state.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),receipt=receipt)
        write_json(status_path,state)
    except BaseException as exc:
        state.update(status='stopped',reason=type(exc).__name__+': '+str(exc));write_json(status_path,state)
        raise


if __name__=='__main__':main()
