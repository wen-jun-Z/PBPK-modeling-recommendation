#!/usr/bin/env python3
"""Train the fixed evidence-conditioned prior or inspect project progress."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.blueprint_learning import evidence_conditioned_prior as learner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--output", default=learner.OUTPUT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args(); output = learner.shared._inside(args.output, ROOT)
    if args.command == "status":
        path = output / "training_state.json"
        state = json.loads(path.read_text()) if path.exists() else {"stage":"not_started"}
        print(json.dumps({k:state.get(k) for k in ("stage", "current_fit", "completed_fits", "planned_fits", "updated_at", "elapsed_seconds", "coefficients", "half_log_uncertainty_q80")}, indent=2)); return
    if args.background:
        output.mkdir(parents=True, exist_ok=True)
        with (output / "training.log").open("ab", buffering=0) as log:
            proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "run", "--output", str(output), "--jobs", str(args.jobs), "--device", args.device],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE":"1", "PYTHONUNBUFFERED":"1", "OMP_NUM_THREADS":str(args.jobs), "OPENBLAS_NUM_THREADS":str(args.jobs)})
        (output / "training.pid").write_text(str(proc.pid)+"\n")
        print(json.dumps({"pid":proc.pid, "log":str(output / "training.log")})); return
    result = learner.train(ROOT, output, args.jobs, args.device, progress=lambda event:print(json.dumps(event), flush=True))
    print(json.dumps({"result":str(output / "predictions.json"), "coefficients":result["coefficients"], "half_log_uncertainty_q80":result["half_log_uncertainty_q80"]}))


if __name__ == "__main__": main()
