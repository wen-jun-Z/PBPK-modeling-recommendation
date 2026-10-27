#!/usr/bin/env python3
"""Prepare/run/status the isolated, fixed TabPFNv2 CL candidate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Do this before importing any ML library. Explicit local weights plus offline
# mode prevent fallback downloads or use of a user's external model caches.
_CACHE = ROOT / "runtime/cache/foundation"
for _name, _directory in {
    "HF_HOME": "hf", "TABPFN_MODEL_CACHE_DIR": "tabpfn", "TORCH_HOME": "torch",
    "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda",
}.items():
    os.environ[_name] = str(_CACHE / _directory)
for _name in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TABPFN_DISABLE_TELEMETRY"):
    os.environ[_name] = "1"
# Dedicated project overlay must precede numpy/sklearn/project ML imports.
sys.path.insert(0, str(ROOT / "runtime/foundation-python"))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
from pbpk_kb.blueprint_learning import foundation_clearance_prior as prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--targets", default=prior.TARGETS)
    parser.add_argument("--source-prior", default=prior.SOURCE_PRIOR)
    parser.add_argument("--output", default=prior.OUTPUT)
    parser.add_argument("--checkpoint", default=prior.CHECKPOINT)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(prior.status(ROOT, args.output), indent=2)); return
    params = (ROOT, args.targets, args.source_prior, args.output, args.checkpoint, args.jobs)
    if args.command == "prepare":
        state = prior.prepare(*params)
        split = state["config"]["split"]
        print(json.dumps({"stage": state["stage"], "eligible_CL_labels": len(split["eligible_indices"]),
            "final_context_count": len(split["final_context_indices"]),
            "calibration_count": len(split["calibration_indices"]),
            "planned_context_fits": 4, "input_fingerprint": state["input_fingerprint"]}, indent=2)); return
    if args.background:
        prior.prepare(*params)
        output = prior.shared._inside(args.output, ROOT)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(args.jobs), "OPENBLAS_NUM_THREADS": str(args.jobs), "MKL_NUM_THREADS": str(args.jobs)}
        command = [sys.executable, "-B", str(Path(__file__).absolute()), "run", "--targets", args.targets,
            "--source-prior", args.source_prior, "--output", args.output, "--checkpoint", args.checkpoint,
            "--jobs", str(args.jobs)]
        with (output / "training.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (output / "training.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "log": str(output / "training.log")})); return
    result = prior.train(*params, progress=lambda event: print(json.dumps(event), flush=True))
    print(json.dumps({"version": result["version"], "result": args.output + "/predictions.json",
        "targets": result["audit"]["target_count"], "oof": result["audit"]["oof_metrics"]}))


if __name__ == "__main__":
    main()
