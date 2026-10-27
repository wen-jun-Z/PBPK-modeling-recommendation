#!/usr/bin/env python3
"""Run the frozen Furosemide methods, lock new-study predictions, then score.

Only the launched project process trees are monitored or terminated. Failures
stop the pipeline; there is no automatic fitting retry or result-dependent edit.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from pbpk_kb.blueprints.experiment_runtime import save, file_sha
from watch_blueprint_suite import ProcessScope, directory_bytes

DEPLOYMENT = ROOT/'runtime/deployment'
STATUS = DEPLOYMENT/'blueprint_furosemide_validation_pipeline_status.json'
SUITE = ROOT/'experiments/blueprint_design/furosemide_mechanism_suite_20260920'
BASE = ROOT/'experiments/blueprint_design/external_validation/furosemide_2021'
PREDICTIONS = BASE/'predictions_all_blocks'
DESIGN = ROOT/'experiments/blueprint_design/independent_cohort_20260920/active_protocols_v3/prediction_metadata/furosemide_pmc8416862_trial2_baseline.json'
INPUT = ROOT/'experiments/blueprint_design/development_inputs/mechanisms_20260920/furosemide/input_package.json'
PROPOSALS = ROOT/'experiments/blueprint_design/qwen_furosemide_mechanism_proposals_20260920'
MANIFEST = INPUT.parent/'suite_manifest.json'
PYTHON = ROOT/'runtime/app-env/bin/python'


def main():
    if STATUS.exists() or SUITE.exists() or PREDICTIONS.exists():
        raise ValueError('Use the existing registered pipeline; do not overwrite runs')
    proposal_status = json.loads((PROPOSALS/'progress.json').read_text())
    if proposal_status.get('state')!='completed' or proposal_status.get('completed')!=3:
        raise ValueError('All three frozen local Qwen proposals must finish first')
    programs = ['run_blueprint_reviewed_suite.py','predict_furosemide_external_study.py','score_blueprint_locked_endpoints.py']
    hashes = {name:file_sha(ROOT/'scripts'/name) for name in programs}
    sources = {str(p.relative_to(ROOT)):file_sha(p) for p in (DESIGN,INPUT,MANIFEST,PROPOSALS/'manifest.json')}
    state={'status':'starting','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),
        'steps':[],'program_sha256':hashes,'source_sha256':sources,'automatic_retries':0,
        'native_parallelism':{'workers':2,'scenario_workers':2,'cpu_threads':2,'maximum_cores':8},
        'external_observations_available_to_optimizer':False}
    save(STATUS,state)
    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,request_stop)
    scope=ProcessScope()
    def terminate(known):
        for sig in (signal.SIGTERM,signal.SIGKILL):
            for identity in reversed(list(known.values())):scope.send(identity,sig)
            time.sleep(2)
    steps=[
        ('methods',[str(PYTHON),str(ROOT/'scripts'/programs[0]),'--manifest',str(MANIFEST),
            '--qwen-proposals',str(PROPOSALS),'--output',str(SUITE),'--workers','2','--cpu-threads','2',
            '--scenario-workers','2','--max-output-gib','16','--task-timeout-hours','4'],SUITE,16*1024**3,8*3600),
        ('predictions',[str(PYTHON),str(ROOT/'scripts'/programs[1]),'--suite',str(SUITE),
            '--design',str(DESIGN),'--input-package',str(INPUT),'--output',str(PREDICTIONS)],PREDICTIONS,2*1024**3,4*3600),
        ('evaluation',[str(PYTHON),str(ROOT/'scripts'/programs[2]),'--lock',str(PREDICTIONS/'prediction_lock.json'),
            '--design',str(DESIGN),'--output',str(BASE/'scores_all_blocks')],BASE,3*1024**3,600),
    ]
    try:
        for name,command,output,byte_limit,timeout in steps:
            if stopping:raise RuntimeError('Stop requested')
            if any(file_sha(ROOT/'scripts'/p)!=sha for p,sha in hashes.items()):
                raise RuntimeError('Pipeline implementation changed after registration')
            if any(file_sha(ROOT/p)!=sha for p,sha in sources.items()):
                raise RuntimeError('Pipeline input changed after registration')
            log=DEPLOYMENT/('blueprint_furosemide_'+name+'_20260920.log')
            with log.open('xb') as stream:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            identity=scope.state(child.pid)
            if identity is None:raise RuntimeError('Child exited before project registration')
            known={child.pid:identity};start=time.monotonic()
            row={'name':name,'status':'running','pid':child.pid,'start_ticks':identity['start_ticks'],
                'log':str(log.relative_to(ROOT)),'output':str(output.relative_to(ROOT)),
                'started_at':datetime.now(timezone.utc).isoformat(),'command':command}
            state['steps'].append(row);state.update(status='running',stage=name)
            save(STATUS,state,replace=True)
            try:
                while child.poll() is None:
                    known.update(scope.discover(known.values()))
                    used=directory_bytes(output)
                    if stopping or time.monotonic()-start>timeout or used>byte_limit or log.stat().st_size>20*1024**2:
                        raise RuntimeError('Project stop/time/output/log safety limit reached')
                    row.update(elapsed_seconds=time.monotonic()-start,output_bytes=used)
                    save(STATUS,state,replace=True)
                    time.sleep(10)
                if child.returncode!=0:raise RuntimeError(f'{name} exited with code {child.returncode}')
            except BaseException:
                terminate(known)
                child.wait(timeout=10)
                raise
            row.update(status='completed',returncode=child.returncode,elapsed_seconds=time.monotonic()-start)
            save(STATUS,state,replace=True)
        state.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat())
        save(STATUS,state,replace=True)
    except BaseException as exc:
        if state['steps'] and state['steps'][-1]['status']=='running':state['steps'][-1]['status']='failed'
        state.update(status='stopped',reason=type(exc).__name__+': '+str(exc))
        save(STATUS,state,replace=True)
        raise


if __name__=='__main__':
    main()
