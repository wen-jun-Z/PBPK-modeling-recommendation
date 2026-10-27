#!/usr/bin/env python3
"""Prepare/train the final fixed complementary CL combination."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pbpk_kb.blueprint_learning import clearance_complementary as learner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','run','status'))
    p.add_argument('--output',default=learner.OUTPUT)
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--background',action='store_true');a=p.parse_args()
    output=learner.shared._inside(a.output,ROOT)
    if a.command=='status':
        state=json.loads((output/'training_state.json').read_text()) if (output/'training_state.json').exists() else {'stage':'not_started'}
        print(json.dumps({k:state.get(k) for k in ('stage','selected','elapsed_seconds','updated_at')},indent=2));return
    if a.command=='prepare':
        state=learner.prepare(ROOT,output,a.jobs,a.device)
        print(json.dumps({'stage':state['stage'],'input_fingerprint':state['input_fingerprint'],'config':state['config']},indent=2));return
    if a.background:
        output.mkdir(parents=True,exist_ok=True)
        with (output/'training.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run','--output',str(output),'--jobs',str(a.jobs),'--device',a.device],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':str(a.jobs),'OPENBLAS_NUM_THREADS':str(a.jobs)})
        (output/'training.pid').write_text(str(process.pid)+'\n')
        print(json.dumps({'pid':process.pid,'log':str(output/'training.log')}));return
    result=learner.fit_prior(ROOT,output,a.jobs,a.device,progress=lambda x:print(json.dumps(x),flush=True))
    print(json.dumps({'selected':result['selection']['selected'],'result':str(output/'predictions.json')}))


if __name__=='__main__':main()
