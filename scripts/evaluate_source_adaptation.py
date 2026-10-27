#!/usr/bin/env python3
"""Small, fixed source-adaptation evaluation: implement, freeze, execute, score.

The official target is an engineering reference, not an independently trained
competitor. Historical model-development observations are never called blind.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from pbpk_kb.blueprints.design_runtime import execute_snapshot
from pbpk_kb.blueprints.heldout_evaluation import evaluate_locked_predictions, lock_predictions
from pbpk_kb.blueprints.source_workflow import inside, read_curve, save, sha256
from watch_blueprint_suite import ProcessScope

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("adapted", "official_reference")


def now():
    return datetime.now(timezone.utc).isoformat()


def compare_curves(left, right):
    """Require the same complete native grid; never hide off-grid peaks."""
    if not left or not right:
        return {"accepted": False, "reason": "missing_prediction"}
    a = dict(zip(left["time_min"], left["concentration"]))
    b = dict(zip(right["time_min"], right["concentration"]))
    if left["time_min"] != right["time_min"]:
        return {"accepted": False, "reason": "different_native_time_grids", "same_time_grid": False}
    common = sorted(a.keys() & b.keys())
    covers_window = min(a) == min(b) and max(a) == max(b)
    if len(common) < 10 or not covers_window:
        return {"accepted": False, "reason": "insufficient_common_native_time_grid",
                "common_points": len(common), "same_window": covers_window}
    peak = max(abs(b[t]) for t in common)
    delta = max(abs(a[t] - b[t]) for t in common)
    threshold = 1e-10 + 1e-6 * peak
    return {"accepted": delta <= threshold, "common_points": len(common),
            "same_time_grid": list(a) == list(b), "same_window": covers_window,
            "maximum_absolute_difference_umol_per_l": delta,
            "maximum_difference_over_reference_peak": delta / peak if peak else None,
            "absolute_tolerance_umol_per_l": 1e-10, "relative_to_peak_tolerance": 1e-6}


def execute_job(case, arm, snapshot, output, args):
    from pbpk_kb.blueprints.source_adaptation_audit import audit_execution
    directory = output / case["case_id"] / arm
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {"case_id": case["case_id"], "arm": arm, "status": "failed", "curve": None,
              "native_accepted": False, "readback_accepted": False}
    try:
        execution = execute_snapshot(snapshot=snapshot, output_dir=directory / "native",
                                     project_root=ROOT, timeout_seconds=args.timeout,
                                     cpu_threads=2, max_output_bytes=128 * 1024**2)
        result["execution_manifest"] = execution["manifest_path"]
        result["native_accepted"] = execution.get("engineering_accepted") is True
        if result["native_accepted"]:
            audit = audit_execution(snapshot=snapshot, execution_manifest=execution,
                                    output_dir=directory / "readback", project_root=ROOT,
                                    **case["native_expectations"][arm])
            save(directory / "audit.json", audit)
            result["readback_accepted"] = audit.get("accepted") is True
            result["curve"] = read_curve(directory / "native", execution, case["target_output"])
            result["status"] = "passed" if result["readback_accepted"] else "readback_failed"
        else:
            result["reason"] = execution.get("error") or execution.get("stop_reason") or execution.get("checks")
    except Exception as error:
        result["reason"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    save(directory / "result.json", result)
    return result


def run(args):
    from source_adaptation_cases import build_case_manifest, build_reference_snapshot, load_observations
    from pbpk_kb.blueprints.source_adaptation import load_source_model, adapt_source_model
    output = inside(args.output, ROOT)
    if output.exists() or not 1 <= args.workers <= 4 or not 60 <= args.timeout <= 600:
        raise ValueError("Use a new project output, 1..4 workers, and 60..600 second native timeout")
    output.mkdir(parents=True)
    started = time.monotonic()
    runner = ProcessScope().state(os.getpid())
    save(output / "runner.json", {**runner, "output": str(output.relative_to(ROOT)), "started_at": now()})
    save(ROOT / "runtime/deployment/source_adaptation_pilot.json",
         {"output": str(output.relative_to(ROOT)), "registered_at": now()})
    progress = {"status": "running", "stage": "implementation_freeze", "completed_native_jobs": 0,
                "planned_native_jobs": 0, "successful_native_jobs": 0, "jobs": {}, "started_at": now()}

    def update(**values):
        progress.update(values, elapsed_seconds=round(time.monotonic() - started, 2))
        save(output / "progress.json", progress)

    update()
    try:
        manifest = build_case_manifest(ROOT)
        cases = manifest["cases"]
        if not 1 <= len(cases) <= 8 or len({x["case_id"] for x in cases}) != len(cases):
            raise ValueError("This pilot requires at most eight unique registered cases")
        code = ["scripts/evaluate_source_adaptation.py", "scripts/source_adaptation_cases.py",
                "src/pbpk_kb/blueprints/source_adaptation.py", "src/pbpk_kb/blueprints/source_workflow.py",
                "src/pbpk_kb/blueprints/source_adaptation_audit.py",
                "src/pbpk_kb/blueprints/design_runtime.py", "src/pbpk_kb/blueprints/heldout_evaluation.py",
                "src/pbpk_kb/blueprints/design_scoring.py", "src/pbpk_kb/blueprints/multidrug_cases.py",
                "src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R",
                "src/pbpk_kb/model_runtime/R/source_adaptation_audit.R"]
        freeze = {"frozen_at": now(), "cases": manifest,
                  "code": [{"path": p, "sha256": sha256(ROOT / p)} for p in code],
                  "protocol": {"arms": list(ARMS), "seed": 0, "optimization": False,
                               "fidelity_absolute_tolerance_umol_l": 1e-10,
                               "fidelity_relative_to_peak_tolerance": 1e-6,
                               "independent_blinded_evaluation": False,
                               "native_solver": "CPU_CVODES", "workers": args.workers,
                               "per_job_timeout_seconds": args.timeout,
                               "max_output_bytes_per_native_job": 128 * 1024**2}}
        save(output / "protocol.json", freeze)
        snapshots, blueprints, results = {}, {}, {}
        # Finish every adaptation before any numerical execution starts.
        for case in cases:
            source_ref = case["source"]
            source = load_source_model(ROOT / source_ref["path"], source_ref["sha256"],
                                       source_ref["simulation_name"], ROOT)
            blueprint = adapt_source_model(source, case["request"], ROOT)
            blueprints[case["case_id"]] = blueprint
            if blueprint["status"] == "adapted":
                snapshots[(case["case_id"], "adapted")] = blueprint["snapshot"]
            else:
                results[(case["case_id"], "adapted")] = {"case_id": case["case_id"], "arm": "adapted",
                    "status": "blocked", "curve": None, "native_accepted": False, "readback_accepted": False,
                    "reason": blueprint.get("reasons")}
            snapshots[(case["case_id"], "official_reference")] = build_reference_snapshot(case, ROOT)
        save(output / "blueprints.json", blueprints)
        update(stage="native_execution_and_readback", planned_native_jobs=len(snapshots),
               cases=len(cases), adaptations_completed=len(cases),
               adaptations_accepted=sum(x["status"] == "adapted" for x in blueprints.values()))
        by_id = {case["case_id"]: case for case in cases}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = {pool.submit(execute_job, by_id[cid], arm, snapshot, output, args): (cid, arm)
                     for (cid, arm), snapshot in snapshots.items()}
            for future in as_completed(tasks):
                key = tasks[future]
                result = future.result()
                results[key] = result
                progress["jobs"]["/".join(key)] = {k: v for k, v in result.items() if k != "curve"}
                update(completed_native_jobs=progress["completed_native_jobs"] + 1,
                       successful_native_jobs=progress["successful_native_jobs"] + int(result["status"] == "passed"))
                print(f"{progress['completed_native_jobs']}/{len(snapshots)} {'/'.join(key)} {result['status']}", flush=True)
        predictions = {"/".join(key): value for key, value in results.items()}
        save(output / "predictions_locked.json", {"locked_at": now(), "jobs": predictions})
        prediction_lock_sha = sha256(output / "predictions_locked.json")
        # Only now decode historical observed concentrations. They were source
        # model development data; the lock does not make them independent.
        update(stage="historical_observation_scoring")
        observations = []
        for case in cases:
            for scenario in load_observations(case, ROOT):
                observations.append({**scenario, "case_id": case["case_id"]})
        save(output / "historical_observations.json", observations)
        evaluations = {}
        for arm in ARMS:
            curves = {s["scenario_id"]: results[(s["case_id"], arm)]["curve"]
                      if results[(s["case_id"], arm)]["status"] == "passed" else None for s in observations}
            if observations:
                lock = {"method": arm, "seed": 0, "candidate_id": "fixed_source_adaptation",
                        "status": "completed", "predictions_sha256": lock_predictions(curves)}
                evaluations[arm] = evaluate_locked_predictions(observations, curves, lock, device="cpu")
                save(output / f"evaluation_{arm}.json", evaluations[arm])
        rows = []
        for case in cases:
            cid = case["case_id"]
            adapted, reference = (results[(cid, arm)] for arm in ARMS)
            comparison = compare_curves(adapted["curve"], reference["curve"])
            rows.append({"case_id": cid, "compound_name": case["compound_name"],
                         "source_simulation": case["source"]["simulation_name"],
                         "target_simulation": case["engineering_reference"]["simulation_name"],
                         "adaptation_status": blueprints[cid]["status"],
                         "native_accepted": adapted["native_accepted"],
                         "readback_accepted": adapted["readback_accepted"],
                         "reference_readback_accepted": reference["readback_accepted"],
                         "fidelity": comparison,
                         "engineering_accepted": adapted["status"] == reference["status"] == "passed" and comparison["accepted"]})
        unchanged = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["code"])
        summary = {"completed_at": now(), "case_count": len(cases), "native_attempts": len(snapshots),
                   "engineering_passed": sum(row["engineering_accepted"] for row in rows),
                   "code_unchanged_during_execution": unchanged, "cases": rows,
                   "prediction_lock_sha256": prediction_lock_sha, "historical_scenario_count": len(observations),
                   "independent_blinded_test": False, "new_drug_generalization_established": False,
                   "predictive_superiority_established": False, "elapsed_seconds": round(time.monotonic() - started, 2),
                   "limitations": ["Two known drugs; historical source-development observations.",
                                   "Official target simulations are engineering references, not blind competitors.",
                                   "No de novo mechanism, disease physiology, or automatic parameter calibration tested.",
                                   "Linux native engine 12.3.137 vs source definition 12.3.173; cross-version equivalence unverified."]}
        save(output / "summary.json", summary)
        write_report(output, summary, evaluations)
        update(status="completed" if unchanged else "invalid_code_changed", stage="completed",
               engineering_passed=summary["engineering_passed"], historical_scenarios=len(observations))
        return summary
    except Exception as error:
        update(status="failed", stage="failed", reason=f"{type(error).__name__}: {error}")
        raise


def write_report(output, summary, evaluations):
    lines = ['# Complete-source PBPK model adaptation: pilot evaluation', "",
             f"Eight scenarios planned; actually {summary['case_count']} scenarios and {summary['native_attempts']} native executions. "
             f"Engineering checks passed for {summary['engineering_passed']}/{summary['case_count']}. ", "",
             'The method retains complete human and drug mechanisms, changes only requested settings, and does not refit parameters. '
             'Corresponding official simulations are engineering references. Historical observations support development regression, not independent blinded validation or superiority over the original model.', "",
             '| Scenario | Native execution | Request readback | Matches official target curve |', "|---|---|---|---|"]
    for row in summary["cases"]:
        yn = lambda value: 'Pass' if value else 'Fail'
        lines.append(f"| {row['case_id']} | {yn(row['native_accepted'])} | {yn(row['readback_accepted'])} | {yn(row['fidelity']['accepted'])} |")
    lines += ["", 'Numerical agreement requires identical complete native time grids and maximum difference <= 1e-10 micromol/L + official peak x 1e-6.', "",
              '| Historical observations | Curve log-RMSE | Geometric mean fold error | AUC fold error | Sampled Cmax fold error |', "|---|---|---|---|---|"]
    for arm, evaluation in evaluations.items():
        for row in evaluation["rows"]:
            if arm != "adapted":
                continue
            fmt = lambda x: f"{x:.3f}" if x is not None else 'Not scorable'
            folds = [math.exp(row["endpoints"][key]["absolute_log_error"])
                     if row["endpoints"][key] else None for key in ("observed_window_auc", "sampled_cmax")]
            lines.append(f"| {row['scenario_id']} | {fmt(row['log_rmse'])} | {fmt(row['gmfe'])} | {fmt(folds[0])} | {fmt(folds[1])} |")
    lines += ["", 'Fold error closer to 1 is better; different metrics are not added together. Rifampicin Nitti/Acocella data retain the source serum-to-plasma mapping assumption; strict matrix equivalence is not established.', "",
              'Historical curve scores are in evaluation_adapted.json and evaluation_official_reference.json. '
              'AUC integrates only the observation window and Cmax is the peak at observed sampling times, not AUC infinity or the continuous peak.', "",
              'protocol.json, blueprints.json, per-case native PKML/curves, and readback results are retained, including all failures.', "",
              'Not yet validated: de novo new-drug construction, disease populations, automatic addition of mechanisms, parameter calibration, or independent external predictive superiority.', "",
              'No LLM is called. OSP native ODE simulations run in parallel on CPU; GPU solving is not claimed.']
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, ensure_ascii=False, indent=2))
    return 0 if summary["engineering_passed"] == summary["case_count"] and summary["code_unchanged_during_execution"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
