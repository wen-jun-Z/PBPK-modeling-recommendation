#!/usr/bin/env python3
"""Prepare/run/status a project-local nested fu-conditioned CL candidate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "runtime/cache/fu_conditioned/cuda")
from pbpk_kb.blueprint_learning import fu_conditioned_clearance as prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--targets", default=prior.TARGETS)
    parser.add_argument("--source-prior", default=prior.SOURCE_PRIOR)
    parser.add_argument("--output", default=prior.OUTPUT)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--device", choices=("cuda","cpu","auto"), default="cuda")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(prior.status(ROOT,args.output),indent=2)); return
    params = (ROOT,args.targets,args.source_prior,args.output,args.jobs,args.device)
    if args.command == "prepare":
        state = prior.prepare(*params)
        print(json.dumps({"stage":state["stage"],"input_fingerprint":state["input_fingerprint"],
            "eligible_structures":len(state["config"]["eligible_indices"]),"planned_unique_fit_tasks_max":26,
            "retry_policy":"Extra attempts after interruption/failure are counted separately"},indent=2)); return
    if args.background:
        prior.prepare(*params)
        output = prior.shared._inside(args.output,ROOT)
        env = {**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1",
            "OMP_NUM_THREADS":str(args.jobs),"OPENBLAS_NUM_THREADS":str(args.jobs),"MKL_NUM_THREADS":str(args.jobs)}
        command = [sys.executable,"-B",str(Path(__file__).resolve()),"run","--targets",args.targets,
            "--source-prior",args.source_prior,"--output",args.output,"--jobs",str(args.jobs),"--device",args.device]
        with (output/"training.log").open("ab",buffering=0) as log:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (output/"training.pid").write_text(str(process.pid)+"\n")
        print(json.dumps({"pid":process.pid,"log":str(output/"training.log")})); return
    result = prior.train(*params,progress=lambda event:print(json.dumps(event),flush=True))
    print(json.dumps({"version":result["version"],"selection":result["selection"],
        "result":args.output+"/predictions.json"}))


if __name__ == "__main__":
    main()
