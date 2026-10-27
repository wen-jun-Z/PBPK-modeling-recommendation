#!/usr/bin/env python3
"""Freeze the existing PBPK construction rules and evaluate multiple drugs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, run_evidence_blueprint
from pbpk_kb.blueprints.evidence_evaluation import aggregate_cases, observed_rows, score_case
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT = Path(__file__).resolve().parents[1]
IV_ENDPOINTS = ["plasma_concentration_time", "auc", "cmax", "tmax", "c_last",
                "terminal_half_life", "auc_inf", "clearance", "vz", "mrt", "vss"]


def load(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare_cases(inputs, root):
    cases, files = [], list(inputs.rglob("*"))
    files = [p for p in files if p.is_file() and "__pycache__" not in p.parts]
    ids = set()
    for cohort_path in sorted(inputs.glob("*/cohort.json")):
        cohort = load(cohort_path)
        for item in cohort["cases"]:
            row = {**item, "compound_id": cohort["compound_id"], "compound_name": cohort["compound_name"]}
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", row["case_id"]) or row["case_id"] in ids:
                raise ValueError("Case IDs must be unique safe directory names")
            ids.add(row["case_id"])
            if row["role"] not in {"calibration_data_reconstruction", "separate_study_development_comparison"}:
                raise ValueError("Historical data must carry an explicit development role")
            request_path = inside(row["request_path"], root, exists=True)
            observations_path = inside(row["observations_path"], root, exists=True)
            request = deepcopy(load(request_path))
            if request["compound_id"] != row["compound_id"] or request["dosing"]["route"] != "intravenous":
                raise ValueError("This protocol is for identified single-dose IV cases")
            times, _ = observed_rows(load(observations_path))
            # Only the fixed sampling schedule defines terminal fitting; values
            # never select the tail window or any construction parameter.
            target = {"endpoints": IV_ENDPOINTS.copy(), "primary_endpoint": "plasma_concentration_time",
                      "window_start_h": request["dosing"].get("start_h", 0.), "window_end_h": times[-1]/60}
            if len(times) >= 4:
                target["terminal_fit"] = {"start_h": times[-4]/60, "end_h": times[-1]/60,
                    "min_points": 4, "min_adjusted_r2": .98, "min_span_half_lives": 1.,
                    "max_auc_extrapolated_fraction": .2, "max_aumc_extrapolated_fraction": .2}
            request["target"] = target
            # Keep the previous construction baseline, with no fitted update.
            if request.get("partition_method", "PKSimStandard") != "PKSimStandard":
                raise ValueError("Baseline partition method must be fixed before evaluation")
            request = EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
            row["request"] = request
            row["schedule_only_used_before_prediction"] = True
            cases.append(row)
            files.extend((request_path, observations_path, inside(request["evidence_ledger"]["path"], root, exists=True)))
            files.extend(inside(e["source"]["path"], root, exists=True) for e in request["evidence"])
    if len({c["compound_id"] for c in cases}) < 2:
        raise ValueError("At least two drug cohorts are required")
    # Reject accidental study-role overlap; several arms from the same study
    # cannot be counted as separate calibration and comparison studies.
    roles = {}
    for case in cases:
        key = (case["compound_id"], case["study_id"])
        if key in roles and roles[key] != case["role"]:
            raise ValueError("A study appears in incompatible evaluation roles")
        roles[key] = case["role"]
    for drug in {c["compound_id"] for c in cases}:
        drug_cases = [c for c in cases if c["compound_id"] == drug]
        evidence_ids = {json.dumps(c["request"]["evidence"], sort_keys=True) for c in drug_cases}
        if len(evidence_ids) != 1:
            raise ValueError("Drug evidence must remain identical across this drug's studies")
        forbidden = []
        for case in drug_cases:
            if case["role"] == "separate_study_development_comparison":
                path = inside(case["observations_path"], root, exists=True)
                forbidden.append({"path": str(path.relative_to(root)), "sha256": sha256(path),
                                  "location": "comparison observations; prohibited as construction evidence"})
        for case in drug_cases:
            refs = {r["path"]: r for r in case["request"].get("forbidden_sources", [])}
            refs.update({r["path"]: r for r in forbidden})
            case["request"]["forbidden_sources"] = list(refs.values())
    return cases, files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args(argv)
    inputs, output = inside(args.inputs, ROOT, exists=True), inside(args.output, ROOT)
    if output.exists():
        raise FileExistsError("Previous results are never overwritten")
    cases, files = prepare_cases(inputs, ROOT)
    files.extend([Path(__file__), ROOT / "scripts/design_evidence_pbpk.py"])
    for name in ("evidence_blueprint", "endpoint_prediction", "terminal_pk", "evidence_evaluation", "human_model_snapshot",
                 "human_model", "design_runtime", "source_workflow", "source_adaptation_audit", "evidence_native_audit"):
        path = ROOT / "src/pbpk_kb/blueprints" / (name + ".py")
        if path.exists():
            files.append(path)
    files.extend(ROOT / "src/pbpk_kb/model_runtime/R" / name for name in
                 ("blueprint_snapshot_run.R", "source_adaptation_audit.R", "evidence_native_audit.R"))
    pins = {str(p.resolve().relative_to(ROOT)): sha256(p) for p in files}
    output.mkdir(parents=True)
    started = time.monotonic()
    protocol = {"started_at": now(), "pins": pins, "cases": cases, "construction": "frozen_evidence_blueprint_1.0_rules",
        "endpoint_reporting_version": "expanded_1.1", "partition_method": "PKSimStandard",
        "independent_validation": False, "parameter_optimization_performed": False,
        "sampling_policy": "predeclared last four sample times; no terminal window search",
        "quality_thresholds_are_project_conventions": True,
        "endpoint_policy": "all endpoints use one native trajectory; sparse comparisons use identical observation times",
        "sparse_auc_inf_cl_vss_mrt_validation": False,
        "resources": {"workers": args.workers, "cpu_threads_per_native_job": 2, "solver": "CPU_CVODES", "gpu_used": False}}
    save(output / "protocol.json", protocol)
    # Register only this known project process, without inspecting other PIDs.
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    save(output / "runner.json", {"pid": os.getpid(), "start_ticks": int(fields[19]), "started_at": now()})
    results = []

    def progress(stage, status="running"):
        save(output / "progress.json", {"mode": "evidence_blueprint", "cohort_mode": "multidrug_endpoints",
            "stage": stage, "status": status, "started_at": protocol["started_at"], "elapsed_seconds": round(time.monotonic()-started, 2),
            "compound_count": len({c["compound_id"] for c in cases}),
            "study_count": len({(c["compound_id"],c["study_id"]) for c in cases}),
            "planned_native_jobs": len(cases), "completed_native_jobs": len(results),
            "successful_native_jobs": sum(r["result"]["status"] == "predicted" for r in results)})

    progress("native_prediction")
    save(ROOT / "runtime/deployment/source_adaptation_pilot.json", {"output": str(output.relative_to(ROOT)), "registered_at": now()})

    def run(case):
        row = {k: v for k, v in case.items() if k != "request"}
        try:
            row["result"] = run_evidence_blueprint(request=case["request"], output_dir=output/case["case_id"], project_root=ROOT, execute=True)
        except Exception as error:
            row["result"] = {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                             "native_executed": (output/case["case_id"]/"native/execution_manifest.json").is_file()}
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(run, case) for case in cases]
        for future in as_completed(jobs):
            row = future.result()
            results.append(row)
            progress("native_prediction")
            print(json.dumps({"case": row["case_id"], "status": row["result"]["status"], "completed": len(results), "total": len(cases)}), flush=True)
    results.sort(key=lambda r: r["case_id"])
    prediction_pins = {}
    for row in results:
        if row["result"]["status"] == "predicted":
            try:
                prediction_pins[row["case_id"]] = sha256(output/row["case_id"]/"prediction_locked.json")
            except OSError as error:
                row["result"].update(status="prediction_lock_failed", error=str(error))
    save(output / "predictions_lock.json", {"locked_at": now(), "locked_before_scoring": True, "predictions": prediction_pins})
    progress("scoring")
    for row in results:
        if row["result"]["status"] != "predicted":
            row["scoring_status"] = "native_prediction_unavailable"
            continue
        try:
            case = next(c for c in cases if c["case_id"] == row["case_id"])
            path = output/row["case_id"]/"prediction_locked.json"
            observed_path = inside(row["observations_path"], ROOT, exists=True)
            if sha256(path) != prediction_pins[row["case_id"]] or sha256(observed_path) != pins[str(observed_path.relative_to(ROOT))]:
                raise ValueError("Frozen prediction or observation changed")
            row["metrics"] = score_case(load(path), load(observed_path), case["request"])
            row["scoring_status"] = "completed"
        except Exception as error:
            row.update(scoring_status="failed", scoring_error=type(error).__name__ + ": " + str(error))
    unchanged = all((ROOT/p).is_file() and sha256(ROOT/p) == h for p,h in pins.items())
    predictions_unchanged = all((output/c/"prediction_locked.json").is_file()
                               and sha256(output/c/"prediction_locked.json") == digest
                               for c,digest in prediction_pins.items())
    endpoint_coverage = {name: {"requested": len(cases), "completed": 0, "not_estimable": 0, "native_unavailable": 0} for name in IV_ENDPOINTS}
    for row in results:
        for name, counts in endpoint_coverage.items():
            status = row["result"].get("endpoint_statuses", {}).get(name)
            key = "completed" if status == "completed" else "not_estimable" if status == "not_estimable" else "native_unavailable"
            counts[key] += 1
    success = sum(r["result"]["status"] == "predicted" for r in results)
    scored = sum(r.get("scoring_status") == "completed" for r in results)
    status = "completed" if unchanged and predictions_unchanged and success == scored == len(cases) else "completed_with_failures"
    summary = {"status": status, "elapsed_seconds": round(time.monotonic()-started, 2), "code_and_inputs_unchanged": unchanged,
        "attempted_cases": len(cases), "successful_native_jobs": success, "scored_cases": scored,
        "compound_count": len({c["compound_id"] for c in cases}), "study_count": len({(c["compound_id"],c["study_id"]) for c in cases}),
        "independent_validation": False, "parameter_optimization_performed": False,
        "predictions_unchanged_after_scoring": predictions_unchanged,
        "all_attempts_retained": True, "endpoint_prediction_coverage": endpoint_coverage,
        "stratified_metrics": aggregate_cases(results), "results": results}
    save(output / "summary.json", summary)
    progress("completed", status)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("results", "stratified_metrics")}, ensure_ascii=False, indent=2), flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
