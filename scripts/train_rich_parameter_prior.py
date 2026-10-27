#!/usr/bin/env python3
"""Background/resumable richer human PK parameter training, project scoped."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.blueprint_learning import rich_parameter_prior as prior


def _active(status):
    pid = status.get("pid")
    if not isinstance(pid, int) or pid == os.getpid():
        return False
    path = Path("/proc") / str(pid)
    try:
        if path.stat().st_uid != os.getuid():
            return False
        command = (path / "cmdline").read_bytes().split(b"\0")
        return any(str(Path(__file__).resolve()).encode() == arg for arg in command)
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "resume", "status"))
    parser.add_argument("--output", default=prior.OUTPUT)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    output = prior.base_prior._inside(args.output, ROOT, exists=False)
    state = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
    if args.command == "status":
        print(json.dumps({**{k:v for k,v in state.items() if k != "events"}, "process_alive": _active(state)}, indent=2))
        return
    if _active(state):
        raise SystemExit("This project's prior trainer is already running")
    if args.background:
        output.mkdir(parents=True, exist_ok=True)
        log = output.parent / "training.log"
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "resume", "--output", str(output),
                   "--jobs", str(args.jobs), "--device", args.device]
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(args.jobs), MKL_NUM_THREADS=str(args.jobs))
        with log.open("ab") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=stream, start_new_session=True)
        # Record only this project-owned child; never enumerate other processes.
        prior._atomic_json(output / "status.json", {**state, "version": prior.VERSION, "state": "starting",
             "pid": process.pid, "updated_at": prior._now(), "log": str(log.relative_to(ROOT))})
        print(json.dumps({"pid": process.pid, "status": str(output / "status.json"), "log": str(log)}))
        return
    try:
        prior.fit_rich_prior(ROOT, output_dir=output, n_jobs=args.jobs, device=args.device)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        prior._status(output, "failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
