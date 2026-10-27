#!/usr/bin/env python3
"""Audit and train the fixed selected PBPK parameter recipe for an expanded cohort."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pbpk_kb.blueprint_learning import expanded_selected_prior as prior


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","run","status"))
    p.add_argument("--targets",default=prior.TARGETS);p.add_argument("--output",default=prior.OUTPUT)
    p.add_argument("--jobs",type=int,default=8);p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    p.add_argument("--background",action="store_true");a=p.parse_args()
    output=prior.shared._inside(a.output,ROOT)
    if a.command=="status":
        path=output/"training_state.json";state=json.loads(path.read_text()) if path.exists() else {"stage":"not_started"}
        print(json.dumps({k:state.get(k) for k in ("stage","mode","pid","completed_fits","planned_fits","shared_progress",
             "current_fit","updated_at","elapsed_seconds","failure_reason")},indent=2));return
    if a.command=="prepare":
        state=prior.prepare(ROOT,a.targets,output,a.jobs,a.device)
        print(json.dumps({"stage":state["stage"],"mode":state["config"]["mode"],"targets":len(state["config"]["targets"]),
            "planned_fits":state["config"]["planned_fits"],"input_fingerprint":state["input_fingerprint"]},indent=2));return
    if a.background:
        # Preparation validates all cohort and pinned sources before detaching.
        prior.prepare(ROOT,a.targets,output,a.jobs,a.device)
        env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1",
             "OMP_NUM_THREADS":str(a.jobs),"OPENBLAS_NUM_THREADS":str(a.jobs),"MKL_NUM_THREADS":str(a.jobs)}
        with (output/"training.log").open("ab",buffering=0) as log:
            proc=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"run","--targets",a.targets,
                "--output",str(output),"--jobs",str(a.jobs),"--device",a.device],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (output/"training.pid").write_text(str(proc.pid)+"\n")
        print(json.dumps({"pid":proc.pid,"log":str(output/"training.log")}));return
    result=prior.train(ROOT,a.targets,output,a.jobs,a.device,progress=lambda event:print(json.dumps(event),flush=True))
    print(json.dumps({"version":result["version"],"mode":result["mode"],"target_count":result["audit"]["target_count"],
                      "result":str(output/"predictions.json")}))


if __name__=="__main__":main()
