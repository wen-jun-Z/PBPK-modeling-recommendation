#!/usr/bin/env python3
"""Train three fixed fu OOF models and joint evidence residual coefficients."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pbpk_kb.blueprint_learning import joint_evidence_prior as learner


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("run","status"))
    p.add_argument("--output",default=learner.OUTPUT);p.add_argument("--jobs",type=int,default=4)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto");p.add_argument("--background",action="store_true")
    a=p.parse_args();output=learner.shared._inside(a.output,ROOT)
    if a.command=="status":
        path=output/"training_state.json";s=json.loads(path.read_text())if path.exists()else{"stage":"not_started"}
        print(json.dumps({k:s.get(k)for k in("stage","current_fit","completed_fits","planned_fits","updated_at","elapsed_seconds")},indent=2));return
    if a.background:
        output.mkdir(parents=True,exist_ok=True)
        with(output/"training.log").open("ab",buffering=0)as log:
            proc=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"run","--output",str(output),"--jobs",str(a.jobs),"--device",a.device],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":str(a.jobs),"OPENBLAS_NUM_THREADS":str(a.jobs)})
        (output/"training.pid").write_text(str(proc.pid)+"\n");print(json.dumps({"pid":proc.pid,"log":str(output/"training.log")}));return
    result=learner.train(ROOT,output,a.jobs,a.device,progress=lambda event:print(json.dumps(event),flush=True))
    print(json.dumps({"result":str(output/"predictions.json"),"version":result["version"],"new_fu_fits":result["audit"]["new_fu_fits"]}))


if __name__=="__main__":main()
