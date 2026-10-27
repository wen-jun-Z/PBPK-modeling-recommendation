#!/usr/bin/env python3
"""Bounded long-tail numerical supplement to an already completed ten-drug run.

Append native output intervals only. Drug, physiology, dose and mechanisms
remain byte-equivalent as decoded JSON. No clinical outcomes select horizons.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

from pbpk_kb.blueprints.design_runtime import execute_snapshot
from pbpk_kb.blueprints.evidence_native_audit import audit_evidence_execution
from pbpk_kb.blueprints.extended_prediction import MODEL_TAIL_POLICY, IV_ENDPOINTS, summarize_extended_prediction
from pbpk_kb.blueprints.source_adaptation_audit import audit_execution, derive_administration_expectations
from pbpk_kb.blueprints.source_workflow import inside, read_curve, save, sha256


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "experiments/blueprint_design/ten_drug_endpoints_20260923"
HORIZONS = (336., 672., 1344.)
RTOL, ATOL, TIME_ATOL_MIN = 1e-4, 1e-9, 1e-8


def load(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return str(path.relative_to(ROOT))


def slim(value):
    if isinstance(value, dict):
        return {k: slim(v) for k, v in value.items()
                if k not in {"fit_times_h", "concentration_umol_per_l", "concentration_mg_per_l", "time_h"}}
    return [slim(v) for v in value] if isinstance(value, list) else value


def append_output_interval(snapshot, end_h):
    """Append one interval; assert every original model and output field remains."""
    if not math.isfinite(end_h) or end_h not in HORIZONS:
        raise ValueError("Use one of the three frozen supplement horizons")
    if len(snapshot.get("Simulations", [])) != 1:
        raise ValueError("Exactly one native simulation is required")
    output = deepcopy(snapshot)
    schema = output["Simulations"][0]["OutputSchema"]
    previous = 0.
    for interval in schema:
        parameters = {p["Name"]: p for p in interval["Parameters"]}
        if any(parameters[key].get("Unit") != "h" for key in ("Start time", "End time")):
            raise ValueError("Unreviewed native time units")
        start, end = parameters["Start time"]["Value"], parameters["End time"]["Value"]
        if start != previous or end <= start:
            raise ValueError("Original native intervals must be continuous and increasing")
        previous = end
    if end_h <= previous:
        raise ValueError("The appended interval must strictly extend the locked horizon")
    appended = deepcopy(schema[-1])
    for parameter in appended["Parameters"]:
        if parameter["Name"] == "Start time":
            parameter["Value"] = previous
        elif parameter["Name"] == "End time":
            parameter["Value"] = end_h
    schema.append(appended)
    restored = deepcopy(output)
    restored["Simulations"][0]["OutputSchema"].pop()
    if restored != snapshot:
        raise ValueError("Appending the horizon changed the original native model")
    return output


def compare_original_curve(original, extended):
    """Verify original native samples exist and concentrations remain unchanged."""
    for curve in (original, extended):
        if curve.get("time_unit") != "min" or curve.get("unit") != "umol/l":
            raise ValueError("Unexpected native trajectory units")
        times, values = curve["time_min"], curve["concentration"]
        if len(times) < 2 or len(times) != len(values) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Invalid native trajectory shape")
        if not all(math.isfinite(value) and value >= 0 for value in times + values):
            raise ValueError("Native trajectory contains invalid values")
    t, a = np.asarray(original["time_min"]), np.asarray(original["concentration"])
    nt, nc = np.asarray(extended["time_min"]), np.asarray(extended["concentration"])
    indices = np.searchsorted(nt, t)
    right = np.clip(indices, 0, len(nt) - 1)
    left = np.clip(indices - 1, 0, len(nt) - 1)
    nearest = np.where(np.abs(nt[right] - t) < np.abs(nt[left] - t), right, left)
    time_difference = np.abs(nt[nearest] - t)
    grids_match = bool(np.all(time_difference <= TIME_ATOL_MIN) and len(set(nearest.tolist())) == len(t))
    difference = np.abs(nc[nearest] - a)
    tolerance = ATOL + RTOL * np.abs(a)
    return {"status": "passed" if grids_match and bool(np.all(difference <= tolerance)) else "failed",
            "original_grid_retained": grids_match, "compared_points": len(t),
            "original_window_h": [float(t[0] / 60), float(t[-1] / 60)],
            "max_time_difference_min": float(time_difference.max()),
            "max_absolute_difference_umol_l": float(difference.max()),
            "max_fraction_of_tolerance": float((difference / tolerance).max()),
            "relative_tolerance": RTOL, "absolute_tolerance_umol_l": ATOL,
            "time_tolerance_min": TIME_ATOL_MIN, "comparison_interpolation_used": False}


def prepare(baseline):
    protocol = load(baseline / "protocol.json")
    summary = load(baseline / "summary.json")
    if summary.get("status") not in {"completed", "completed_with_failures"}:
        raise ValueError("Finish the main batch before supplementing")
    if len(summary["results"]) != len(protocol["cases"]):
        raise ValueError("Main batch lacks final results for some cases")
    pins = dict(protocol["pins"])
    if not all((ROOT / path).is_file() and sha256(ROOT / path) == digest for path, digest in pins.items()):
        raise ValueError("Main batch code or inputs changed")
    locks = load(baseline / "predictions_lock.json")["predictions"]
    requests = {case["case_id"]: case["request"] for case in protocol["cases"]}
    cases = []
    files = [Path(__file__), baseline / "protocol.json", baseline / "summary.json", baseline / "predictions_lock.json"]
    for row in summary["results"]:
        if row.get("status") != "predicted":
            continue
        missing = [ep for ep in IV_ENDPOINTS[5:] if row.get("endpoint_values", {}).get(ep, {}).get("status") != "completed"]
        if not missing:
            continue
        cid = row["case_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", cid):
            raise ValueError("Unsafe case ID")
        curve_path = inside(row["prediction_path"], ROOT, exists=True)
        if sha256(curve_path) != row["prediction_sha256"] or locks[cid] != row["prediction_sha256"]:
            raise ValueError("Original prediction lock mismatch")
        snapshot_path = curve_path.parent / "snapshot.json"
        snapshot = load(snapshot_path)
        native_source = curve_path.parent / "native/snapshot.json"
        if load(native_source) != snapshot:
            raise ValueError("Construction and executed native snapshots disagree")
        if row["simulation_end_h"] > 168.:
            raise ValueError("Main-batch baseline is outside the original horizon cap")
        cases.append({"case_id": cid, "request": requests[cid], "original_result": row,
                      "original_prediction_path": rel(curve_path), "snapshot_path": rel(snapshot_path),
                      "original_missing_endpoints": missing})
        files.extend((curve_path, snapshot_path, native_source, baseline / "cases" / cid / "case_result.json"))
    pins.update({rel(path): sha256(path) for path in files})
    return cases, pins


def run_case(case, output, quota_bytes):
    directory = output / "cases" / case["case_id"]
    directory.mkdir(parents=True)
    row = deepcopy(case["original_result"])
    for key in ("metrics", "scoring_status", "scoring_error"):
        row.pop(key, None)
    row["baseline_prediction_path"] = case["original_prediction_path"]
    row["horizon_supplement"] = {"status": "incomplete", "attempts": []}
    snapshot = load(ROOT / case["snapshot_path"])
    original_curve = load(ROOT / case["original_prediction_path"])
    try:
        for index, horizon in enumerate(HORIZONS, 1):
            if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > quota_bytes:
                raise RuntimeError("Supplement soft disk quota reached; no additional execution")
            snapshot = append_output_interval(snapshot, horizon)
            attempt = directory / f"attempt_{index}_{int(horizon)}h"
            attempt.mkdir()
            save(attempt / "snapshot.json", snapshot)
            receipt = {"path": rel(attempt), "horizon_h": horizon, "status": "started"}
            row["horizon_supplement"]["attempts"].append(receipt)
            execution = execute_snapshot(snapshot=snapshot, output_dir=attempt / "native", project_root=ROOT,
                                         timeout_seconds=360, cpu_threads=2, max_output_bytes=128 * 1024**2)
            if not execution.get("engineering_accepted"):
                receipt["status"] = "native_failed"
                raise RuntimeError("Native numerical extension failed")
            audit = audit_execution(snapshot=snapshot, execution_manifest=execution, output_dir=attempt / "readback",
                                    project_root=ROOT, **derive_administration_expectations(snapshot))
            save(attempt / "readback_result.json", audit)
            if not audit.get("accepted"):
                receipt["status"] = "readback_failed"
                raise RuntimeError("Extended native input readback failed")
            drug_audit = audit_evidence_execution(snapshot, execution["manifest_path"], attempt / "native_parameter_audit", ROOT)
            save(attempt / "parameter_audit_result.json", drug_audit)
            if not drug_audit.get("accepted"):
                receipt["status"] = "parameter_audit_failed"
                raise RuntimeError("Extended native evidence audit failed")
            target = snapshot["Simulations"][0]["OutputSelections"]
            if len(target) != 1:
                raise ValueError("Exactly one original plasma output is required")
            curve = read_curve(attempt / "native", load(Path(execution["manifest_path"])), target[0])
            curve_path = attempt / "prediction_locked.json"
            save(curve_path, curve)
            receipt["prediction_sha256"] = sha256(curve_path)
            consistency = compare_original_curve(original_curve, curve)
            save(attempt / "original_window_consistency.json", consistency)
            if consistency["status"] != "passed":
                receipt["status"] = "original_window_inconsistent"
                raise RuntimeError("Original output grid or concentrations changed")
            endpoints = summarize_extended_prediction(curve, case["request"])
            save(attempt / "endpoints.json", slim(endpoints))
            receipt.update(status="predicted", endpoint_counts=endpoints["endpoint_counts"],
                           original_window_consistency=consistency)
            curve_csv = attempt / "predicted_curve.csv"
            mw = next(e["value"] for e in case["request"]["evidence"] if e["parameter"] == "molecular_weight")
            with curve_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("time_h", "concentration_umol_l", "concentration_mg_l"))
                writer.writerows((t / 60, c, c * mw / 1000) for t, c in zip(curve["time_min"], curve["concentration"]))
            row.update(status="predicted", prediction_path=rel(curve_path), prediction_sha256=sha256(curve_path),
                       simulation_end_h=horizon, endpoint_values=slim(endpoints["endpoints"]), curve_csv=rel(curve_csv),
                       endpoint_diagnostics=slim({k: v for k, v in endpoints.items() if k != "endpoints"}),
                       attempts=deepcopy(case["original_result"].get("attempts", [])) + deepcopy(row["horizon_supplement"]["attempts"]))
            if endpoints["endpoint_counts"]["not_estimable"] == 0:
                row["horizon_supplement"]["status"] = "completed"
                break
            if endpoints["diagnostics"]["dropped_numeric_suffix_points"]:
                row["horizon_supplement"]["stop_reason"] = "Fixed numeric-floor prefix exhausted; further horizon cannot supply a later reliable tail"
                break
    except Exception as error:
        row["horizon_supplement"].update(status="failed", error=type(error).__name__ + ": " + str(error))
    row["attempts"] = (deepcopy(case["original_result"].get("attempts", []))
                       + deepcopy(row["horizon_supplement"]["attempts"]))
    save(directory / "case_result.json", row)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 7))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    baseline, output = BATCH / "evaluation", BATCH / "horizon_supplement"
    cases, pins = prepare(baseline)
    if args.prepare_only:
        print(json.dumps({"cases": len(cases), "horizons_h": HORIZONS, "maximum_native_jobs": 3 * len(cases)}, indent=2))
        return 0
    if output.exists():
        raise FileExistsError("Preserve existing supplement; this bounded batch does not overwrite or resume")
    output.mkdir(parents=True)
    protocol = {"started_at": now(), "cases": cases, "pins": pins, "horizons_h": HORIZONS,
        "maximum_attempts_per_case": 3, "maximum_horizon_h": 1344., "model_tail_policy": MODEL_TAIL_POLICY,
        "cap_exception": "The frozen module's 168h cap governs its suggestion only; this external numerical supplement appends native intervals up to 1344h. All endpoint equations, floor and quality gates are unchanged.",
        "parameters_optimized": False, "observations_used": False,
        "original_output_intervals_preserved": True, "max_output_bytes_per_native_job": 128 * 1024**2,
        "soft_total_output_quota_bytes": 4 * 1024**3, "quota_reserved_per_worker_bytes": 256 * 1024**2,
        "workers": args.workers, "cpu_threads_per_job": 2, "timeout_seconds_per_native_job": 360,
        "original_curve_tolerances": {"relative": RTOL, "absolute_umol_l": ATOL, "time_min": TIME_ATOL_MIN}}
    save(output / "protocol.json", protocol)
    stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    save(output / "runner.json", {"pid": os.getpid(), "start_ticks": int(stat[19]), "started_at": now()})
    start, rows = time.monotonic(), []
    def progress(status):
        save(output / "progress.json", {"status": status, "stage": "numerical_horizon_supplement",
            "planned_cases": len(cases), "completed_cases": len(rows),
            "planned_native_jobs": len(cases), "completed_native_jobs": len(rows),
            "planned_jobs_are_cases_with_bounded_numerical_extensions": True,
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "completed_endpoints_cases": sum(r["horizon_supplement"]["status"] == "completed" for r in rows)})
    progress("running")
    quota = protocol["soft_total_output_quota_bytes"] - args.workers * protocol["quota_reserved_per_worker_bytes"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_case, case, output, quota) for case in cases]
        for future in as_completed(jobs):
            row = future.result()
            rows.append(row)
            progress("running")
            print(json.dumps({"case": row["case_id"], "status": row["horizon_supplement"]["status"], "completed": len(rows), "total": len(cases)}), flush=True)
    rows.sort(key=lambda row: row["case_id"])
    unchanged = all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in pins.items())
    predictions_unchanged = all(sha256(ROOT / r["prediction_path"]) == r["prediction_sha256"] for r in rows)
    completed = sum(row["horizon_supplement"]["status"] == "completed" for row in rows)
    status = "completed" if unchanged and predictions_unchanged and completed == len(cases) else "completed_with_unavailable_endpoints_or_failures"
    save(output / "predictions_lock.json", {"locked_at": now(), "predictions": {r["case_id"]: r["prediction_sha256"] for r in rows}, "clinical_scoring_performed": False})
    summary = {"status": status, "case_count": len(cases), "completed_endpoint_cases": completed,
        "new_native_attempts": sum(len(r["horizon_supplement"]["attempts"]) for r in rows),
        "source_and_code_pins_unchanged": unchanged, "predictions_unchanged": predictions_unchanged,
        "parameters_optimized": False, "clinical_scoring_performed": False,
        "merge_instruction": "Replace only matching case IDs in the original results with these metadata rows, then derive a new report. Never overwrite main batch results.", "results": rows}
    save(output / "summary.json", summary)
    progress(status)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
