#!/usr/bin/env python3
"""Resume the frozen RF ablation without rewriting completed scientific output.

The original experiment script is source-hash bound. This thin entry point adds
idempotent completed-run handling while preserving that script and its receipts.
"""
from __future__ import annotations

import argparse
import json

import blueprint_reviewer_rf_ablation as experiment


def resume(workers: int = 6) -> dict:
    if not 1 <= workers <= 6:
        raise ValueError("Maximum six processes")
    protocol = experiment.read(experiment.OUT / "protocol.json")
    state = experiment.read(experiment.OUT / "state.json")
    experiment.frozen.verify(protocol["source_pins"])
    if state["status"] == "complete":
        experiment.checked(state["results"])
        for record in state["native_done"].values():
            experiment.frozen.verify(experiment.checked(record)["artifact_pins"])
        for record in state["fits_done"].values():
            unit = experiment.checked(record["result"])
            if not record["reused"]:
                experiment.frozen.verify(unit["output_pins"])
        return {
            **experiment.status(),
            "action": "verified_existing_completed_run",
            "new_computations": 0,
            "results": state["results"],
        }
    if state["status"] == "predictions_locked":
        return experiment.report()
    return experiment.run(workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    print(json.dumps(resume(arguments.workers), ensure_ascii=False))
