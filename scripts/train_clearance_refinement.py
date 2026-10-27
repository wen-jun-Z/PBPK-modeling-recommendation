#!/usr/bin/env python3
"""Train/resume the CL-only refinement without opening clinical observations."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.blueprint_learning import clearance_refinement as cl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--output", default=cl.OUTPUT)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    output = cl.shared._inside(args.output, ROOT)
    if args.command == "status":
        p = output / "training_state.json"
        state = json.loads(p.read_text()) if p.exists() else {"stage": "not_started"}
        print(json.dumps({k: v for k,v in state.items() if k not in {"fits", "config"}}, indent=2))
        return
    if args.background:
        output.mkdir(parents=True, exist_ok=True)
        log = output / "training.log"
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "run", "--output", str(output),
                   "--jobs", str(args.jobs), "--device", args.device]
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(args.jobs), MKL_NUM_THREADS=str(args.jobs))
        with log.open("ab") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=stream, start_new_session=True)
        (output / "training.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "log": str(log), "state": str(output / "training_state.json")}))
        return
    result = cl.fit_clearance_refinement(ROOT, output, args.jobs, args.device,
                                         progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"selected": result["selection"]["selected"], "predictions": len(result["predictions"]),
                      "result": str(output / "predictions.json")}, indent=2))


if __name__ == "__main__":
    main()
