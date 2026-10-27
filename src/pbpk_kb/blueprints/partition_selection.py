"""Two fixed tissue-partition candidates selected using development studies only.

The original evidence, drug parameters and simulation horizon stay unchanged.
This historical development comparison is not independent validation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time

from .evidence_blueprint import EvidenceBlueprintRequest, run_evidence_blueprint
from .evidence_evaluation import aggregate_cases, score_case
from .source_workflow import inside, save, sha256

VERSION = "partition-selection-1.0"
BASELINE = "PKSimStandard"
CANDIDATE = "RodgersRowland"
CALIBRATION = "calibration_data_reconstruction"
COMPARISON = "separate_study_development_comparison"
RULE = {"minimum_log_rmse_relative_reduction": .01,
        "maximum_auc_fold_relative_increase": .01,
        "maximum_sampled_cmax_fold_relative_increase": .01,
        "metric_source": "calibration study only; no comparison metrics",
        "no_calibration": "keep PKSimStandard",
        "failed_or_invalid_candidate": "keep PKSimStandard",
        "zero_baseline_error": "keep PKSimStandard"}


def _load(path):
    return json.loads(path.read_text())


def _now():
    return datetime.now(timezone.utc).isoformat()


def _metrics(metrics):
    values = {"log_rmse": metrics["log_rmse"], "auc_fold": metrics["auc"]["fold_error"],
              "sampled_cmax_fold": metrics["sampled_cmax"]["fold_error"]}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in values.values()) or values["log_rmse"] < 0 or any(values[k] < 1 for k in ("auc_fold", "sampled_cmax_fold")):
        raise ValueError("Selection metrics must be finite nonnegative RMSE and fold errors at least one")
    return values


def choose_partition(baseline_metrics=None, candidate_metrics=None):
    """Accept only the predeclared calibration metrics; no observations or role mixing."""
    choice = {"selected_partition_method": BASELINE, "rule": deepcopy(RULE),
              "comparison_used_for_selection": False}
    if baseline_metrics is None:
        return {**choice, "reason": "no_valid_calibration_baseline"}
    if candidate_metrics is None:
        return {**choice, "reason": "candidate_failed_or_unscored"}
    try:
        baseline, candidate = _metrics(baseline_metrics), _metrics(candidate_metrics)
    except (KeyError, TypeError, ValueError):
        return {**choice, "reason": "invalid_calibration_metrics"}
    if baseline["log_rmse"] == 0:
        return {**choice, "reason": "zero_baseline_error", "baseline": baseline, "candidate": candidate}
    # The tiny numerical tolerance only stabilizes exact threshold boundaries.
    below = lambda value, limit: value <= limit or math.isclose(value, limit, rel_tol=1e-12, abs_tol=0.)
    checks = {"log_rmse_reduction": below(candidate["log_rmse"], .99 * baseline["log_rmse"]),
              "auc_guard": below(candidate["auc_fold"], 1.01 * baseline["auc_fold"]),
              "sampled_cmax_guard": below(candidate["sampled_cmax_fold"], 1.01 * baseline["sampled_cmax_fold"])}
    accepted = all(checks.values())
    return {**choice, "selected_partition_method": CANDIDATE if accepted else BASELINE,
            "reason": "calibration_rule_passed" if accepted else "calibration_rule_rejected",
            "baseline": baseline, "candidate": candidate, "checks": checks,
            "log_rmse_relative_reduction": 1-candidate["log_rmse"]/baseline["log_rmse"]}


def candidate_request(request):
    if request.get("partition_method", BASELINE) != BASELINE:
        raise ValueError("The frozen baseline must use PKSimStandard")
    candidate = deepcopy(request)
    candidate["partition_method"] = CANDIDATE
    # Do not let schema normalization change the frozen values or target fields.
    EvidenceBlueprintRequest.model_validate(candidate)
    return candidate


def _verify_pins(pins, root):
    for relative, expected in pins.items():
        path = inside(relative, root, exists=True)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("Frozen file changed: " + relative)


def _prepare(baseline_dir, root):
    summary_path = baseline_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Finish the fixed baseline before partition selection")
    status = _load(summary_path)
    # Deliberately do not use summary.results or any of its comparison metrics.
    if (status.get("status") != "completed" or not status.get("code_and_inputs_unchanged")
            or not status.get("predictions_unchanged_after_scoring")):
        raise ValueError("A completed baseline with verified integrity is required")
    protocol_path, locks_path = baseline_dir / "protocol.json", baseline_dir / "predictions_lock.json"
    protocol, locks = _load(protocol_path), _load(locks_path)
    cases = protocol["cases"]
    pins = dict(protocol["pins"])
    if not locks.get("locked_before_scoring") or not cases:
        raise ValueError("Baseline predictions must be locked before scoring")
    ids, cal_counts, role_groups = set(), {}, {}
    for case in cases:
        identifier, drug, role = case["case_id"], case["compound_id"], case["role"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identifier) or identifier in ids:
            raise ValueError("Case IDs must be unique safe directory names")
        if role not in {CALIBRATION, COMPARISON} or case["request"]["compound_id"] != drug:
            raise ValueError("Case drug and development role must match the frozen request")
        ids.add(identifier)
        key = (drug, case["study_id"])
        if key in role_groups and role_groups[key] != role:
            raise ValueError("A study cannot supply calibration and comparison")
        role_groups[key] = role
        if role == CALIBRATION:
            cal_counts[drug] = cal_counts.get(drug, 0) + 1
        candidate_request(case["request"])
        observation = inside(case["observations_path"], root, exists=True)
        if str(observation.relative_to(root)) not in pins:
            raise ValueError("Observation source was not frozen by baseline")
        prediction = baseline_dir / identifier / "prediction_locked.json"
        expected = locks["predictions"].get(identifier)
        if expected is None or sha256(prediction) != expected:
            raise ValueError("Baseline prediction lock failed: " + identifier)
        pins[str(prediction.relative_to(root))] = expected
    if any(count > 1 for count in cal_counts.values()):
        raise ValueError("This bounded protocol requires at most one calibration case per drug")
    for path in (summary_path, protocol_path, locks_path):
        pins[str(path.relative_to(root))] = sha256(path)
    _verify_pins(pins, root)
    return cases, pins


def run_partition_selection(*, baseline_dir: Path, output_dir: Path,
                            project_root: Path, max_workers: int = 4):
    """Run calibration candidates, lock choices, then eligible comparison candidates."""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or not 1 <= max_workers <= 4:
        raise ValueError("max_workers must be an integer from one to four")
    root = project_root.resolve(strict=True)
    baseline = inside(baseline_dir, root, exists=True)
    output = inside(output_dir, root)
    if output.exists() or output == baseline or baseline in output.parents:
        raise FileExistsError("Use a new separate output directory; baseline is immutable")
    cases, pins = _prepare(baseline, root)
    calibration_drugs = {c["compound_id"] for c in cases if c["role"] == CALIBRATION}
    native_cases = [c for c in cases if c["compound_id"] in calibration_drugs]
    baseline_only_cases = [c for c in cases if c["compound_id"] not in calibration_drugs]
    for relative in ("src/pbpk_kb/blueprints/partition_selection.py", "scripts/evaluate_partition_selection.py"):
        path = root / relative
        if path.is_file():
            pins[relative] = sha256(path)
    output.mkdir(parents=True)
    started = time.monotonic()
    protocol = {"version": VERSION, "started_at": _now(), "baseline_dir": str(baseline.relative_to(root)),
                "pins": pins, "cases": cases, "selection_rule": RULE, "candidate_partition_method": CANDIDATE,
                "only_changed_request_field": "partition_method", "independent_validation": False,
                "eligible_comparison_candidates_retained_even_if_rejected": True,
                "planned_native_jobs": len(native_cases),
                "candidate_case_ids": [c["case_id"] for c in native_cases],
                "baseline_only_case_ids": [c["case_id"] for c in baseline_only_cases],
                "no_calibration_candidate_policy": "not_planned; baseline retained without RR execution",
                "baseline_predictions_reused": True,
                "resources": {"workers": max_workers, "cpu_threads_per_native_job": 2,
                              "solver": "CPU_CVODES", "gpu_used": False}}
    save(output / "protocol.json", protocol)
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    save(output / "runner.json", {"pid": os.getpid(), "start_ticks": int(fields[19]), "started_at": _now()})
    results, scored_baselines, prediction_pins = {}, {}, {}
    choices, lock_hash = {}, None

    def progress(stage, status="running"):
        save(output / "progress.json", {"mode": "partition_selection", "stage": stage, "status": status,
             "started_at": protocol["started_at"], "elapsed_seconds": round(time.monotonic()-started, 2),
             "planned_native_jobs": len(native_cases), "completed_native_jobs": len(results),
             "successful_native_jobs": sum(r["result"].get("status") == "predicted" for r in results.values()),
             "choice_locked": lock_hash is not None})

    def run(case):
        identifier = case["case_id"]
        row = {k: v for k, v in case.items() if k != "request"}
        try:
            row["result"] = run_evidence_blueprint(request=candidate_request(case["request"]),
                output_dir=output / "candidate" / identifier, project_root=root, execute=True)
        except Exception as error:
            row["result"] = {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                 "native_executed": (output / "candidate" / identifier / "native/execution_manifest.json").is_file()}
        return row

    def execute_phase(phase_cases, stage):
        _verify_pins(pins, root)
        progress(stage)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            jobs = [pool.submit(run, case) for case in phase_cases]
            for future in as_completed(jobs):
                row = future.result()
                results[row["case_id"]] = row
                progress(stage)
        for case in phase_cases:
            row = results[case["case_id"]]
            if row["result"].get("status") == "predicted":
                path = output / "candidate" / case["case_id"] / "prediction_locked.json"
                try:
                    prediction_pins[case["case_id"]] = sha256(path)
                except OSError as error:
                    row["result"].update(status="prediction_lock_failed", error=str(error))
        save(output / (stage + "_prediction_lock.json"), {"locked_before_scoring": True,
             "locked_at": _now(), "predictions": {c["case_id"]: prediction_pins[c["case_id"]]
                 for c in phase_cases if c["case_id"] in prediction_pins}})
        _verify_pins(pins, root)
        for case in phase_cases:
            identifier = case["case_id"]
            # Comparison observations are first parsed only after choice_locked.
            if case["role"] == COMPARISON:
                if lock_hash is None or sha256(output / "choice_locked.json") != lock_hash:
                    raise ValueError("Choices must be immutable before comparison scoring")
            observed = _load(inside(case["observations_path"], root, exists=True))
            base_row = {k: v for k, v in case.items() if k != "request"}
            try:
                base_row["metrics"] = score_case(_load(baseline / identifier / "prediction_locked.json"), observed, case["request"])
                base_row["scoring_status"] = "completed"
                base_row["result"] = {"status": "predicted", "reused_baseline_prediction": True}
            except Exception as error:
                base_row.update(scoring_status="failed", scoring_error=str(error), result={"status": "score_failed"})
            scored_baselines[identifier] = base_row
            row = results[identifier]
            if row["result"].get("status") != "predicted":
                row["scoring_status"] = "native_prediction_unavailable"
                continue
            try:
                path = output / "candidate" / identifier / "prediction_locked.json"
                if sha256(path) != prediction_pins[identifier]:
                    raise ValueError("Candidate prediction changed after lock")
                row["metrics"] = score_case(_load(path), observed, candidate_request(case["request"]))
                row["scoring_status"] = "completed"
            except Exception as error:
                row.update(scoring_status="failed", scoring_error=type(error).__name__ + ": " + str(error))

    fatal = None
    try:
        calibrations = [c for c in cases if c["role"] == CALIBRATION]
        execute_phase(calibrations, "calibration_candidates")
        for drug in sorted({c["compound_id"] for c in cases}):
            calibration = next((c for c in calibrations if c["compound_id"] == drug), None)
            if calibration is None:
                choices[drug] = {**choose_partition(), "reason": "no_calibration_case", "calibration_case_id": None}
            else:
                identifier = calibration["case_id"]
                choices[drug] = {**choose_partition(scored_baselines[identifier].get("metrics"), results[identifier].get("metrics")),
                                 "calibration_case_id": identifier}
        _verify_pins(pins, root)
        save(output / "choice_locked.json", {"locked_at": _now(), "locked_before_comparison_candidates": True,
             "selection_rule": RULE, "choices": choices, "independent_validation": False,
             "comparison_metrics_used_for_selection": False})
        lock_hash = sha256(output / "choice_locked.json")
        execute_phase([c for c in native_cases if c["role"] == COMPARISON], "comparison_candidates")
        # No-calibration drugs keep their original predictions; no RR run is planned.
        for case in baseline_only_cases:
            if sha256(output / "choice_locked.json") != lock_hash:
                raise ValueError("Choice lock changed before baseline-only comparison scoring")
            identifier = case["case_id"]
            row = {k: v for k, v in case.items() if k != "request"}
            try:
                observed = _load(inside(case["observations_path"], root, exists=True))
                row["metrics"] = score_case(_load(baseline / identifier / "prediction_locked.json"), observed, case["request"])
                row.update(scoring_status="completed", result={"status": "predicted", "reused_baseline_prediction": True})
            except Exception as error:
                row.update(scoring_status="failed", scoring_error=str(error), result={"status": "score_failed"})
            scored_baselines[identifier] = row
        _verify_pins(pins, root)
        if sha256(output / "choice_locked.json") != lock_hash:
            raise ValueError("Choice lock changed")
        for identifier, digest in prediction_pins.items():
            if sha256(output / "candidate" / identifier / "prediction_locked.json") != digest:
                raise ValueError("Candidate prediction changed: " + identifier)
    except Exception as error:
        fatal = type(error).__name__ + ": " + str(error)

    selected_rows, candidate_rows, baseline_rows = [], [], []
    for case in cases:
        identifier = case["case_id"]
        unavailable = {**{k: v for k, v in case.items() if k != "request"},
                       "result": {"status": "not_run", "reason": fatal or "not_available"}, "scoring_status": "not_run"}
        candidate = results.get(identifier, unavailable)
        base = scored_baselines.get(identifier, unavailable)
        method = choices.get(case["compound_id"], {}).get("selected_partition_method", BASELINE)
        selected = deepcopy(candidate if method == CANDIDATE else base)
        selected["selected_partition_method"] = method
        selected["selection_based_on_comparison"] = False
        if case["compound_id"] in calibration_drugs:
            candidate_rows.append(candidate)
        baseline_rows.append(base)
        selected_rows.append(selected)
    successful = sum(r["result"].get("status") == "predicted" for r in candidate_rows)
    scored = sum(r.get("scoring_status") == "completed" for r in candidate_rows)
    status = "completed" if not fatal and successful == scored == len(native_cases) and len(scored_baselines) == len(cases) and all(r.get("scoring_status") == "completed" for r in baseline_rows) else "completed_with_failures"
    summary = {"status": status, "fatal_error": fatal, "elapsed_seconds": round(time.monotonic()-started, 2),
         "attempted_cases": len(native_cases), "planned_native_jobs": len(native_cases),
         "selected_case_count": len(cases), "baseline_only_cases": [{"case_id":c["case_id"],"status":"not_planned_no_calibration"} for c in baseline_only_cases],
         "native_attempts_completed": len(results), "successful_native_jobs": successful,
         "scored_cases": scored, "all_attempts_retained": True, "independent_validation": False,
         "choice_lock_sha256": lock_hash, "code_and_inputs_unchanged": fatal is None,
         "parameter_values_optimized": False, "selected_only_on_calibration": True,
         "choices": choices, "candidate_results": candidate_rows, "baseline_results": baseline_rows,
         "selected_results": selected_rows, "stratified_metrics": {
             "baseline": aggregate_cases(baseline_rows), "all_RodgersRowland": aggregate_cases(candidate_rows),
             "selected_by_calibration": aggregate_cases(selected_rows)},
         "limitations": ["Historical development data; no independent validation or reliable improvement claim.",
             "No tuning of evidence, clearance, binding or dose; only a choice between two fixed partition formulas.",
             "No-calibration drugs retain baseline without candidate execution. Rejected eligible candidates remain visible; comparison errors never change selection.",
             "Selected candidate failures in another study remain failures; no retrospective fallback based on that study."]}
    save(output / "summary.json", summary)
    progress("completed", status)
    return summary
