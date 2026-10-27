"""Fit one native mechanism input, then select once on disjoint studies.

Clinical CL is never installed as intrinsic clearance. The complete native
model determines the concentrations. This small historical-development stage
does not qualify a model for clinical use or establish independent validation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

from .design_scoring import score_calibration_curves
from .parameter_batch import _inside, _number, _quantity_path, _sha, _text, run_parameter_batch
from .source_calibration import fit_bounded_source_parameters

VERSION = "single-mechanism-study-selection-v1"
_CASE_FIELDS = {"case_id", "study_group_id", "role", "pkml_path", "target_output", "observations"}
_SPEC_FIELDS = {"parameter_id", "path", "unit", "source_value", "lower_bound",
                "upper_bound", "evidence_reference", "scaling"}


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _case(case, role, root, device):
    fields = _CASE_FIELDS | ({"sampling_type"} if role == "selection" else set())
    if not isinstance(case, dict) or set(case) != fields or case.get("role") != role:
        raise ValueError(f"Every {role} case requires the documented fields and its explicit role")
    if role == "selection" and case["sampling_type"] != "concentration_time_profile":
        raise ValueError("AUC/Cmax selection accepts concentration-time profiles, not trough sequences")
    item = dict(case)
    for key in ("case_id", "study_group_id"):
        item[key] = _text(item[key], key)
    path = _inside(item["pkml_path"], root, must_exist=True)
    if not path.is_file() or path.suffix.lower() != ".pkml":
        raise ValueError("Case PKML must be a complete project file")
    item["pkml_path"] = str(path)
    item["target_output"] = _quantity_path(item["target_output"], "target output")
    observations = item["observations"]
    score_calibration_curves(runs=[], observations=observations, device=device)
    if len(observations) < 2:
        raise ValueError("Mechanism selection requires a curve with at least two points")
    times = [point["time_min"] for point in observations]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Observation times must be strictly increasing for observed-window AUC")
    for point in observations:
        if str(point.get("unit", "")).replace("µ", "u").replace("μ", "u") != "umol/l":
            raise ValueError("Observations must explicitly use umol/l")
    # Detach caller-owned observations before starting any native work.
    item["observations"] = json.loads(json.dumps(observations, allow_nan=False))
    return item


def _receipt(reference, case, spec, root):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Every case requires a readback receipt path and SHA256")
    path = _inside(root / reference["path"], root, must_exist=True)
    if not path.is_file() or _sha(path) != reference["sha256"]:
        raise ValueError("Native source readback receipt checksum mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    pkml_sha = _sha(Path(case["pkml_path"]))
    if receipt.get("accepted") is not True or receipt.get("pkml_sha256") != pkml_sha:
        raise ValueError("Native source readback does not accept this exact PKML")
    matches = [r for r in receipt.get("parameter_checks", []) if r.get("path") == spec["path"]]
    if not matches or any(r.get("passed") is not True or r.get("unit") != spec["unit"]
                          or _number(r.get("value")) != spec["source_value"] for r in matches):
        raise ValueError("Native source value was not explicitly verified by the readback receipt")
    for row in matches:
        actual, expected = _number(row.get("native_value")), _number(row.get("expected_native_value"))
        if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError("Readback receipt native value differs from its converted expectation")
    return {"path": str(path.relative_to(root)), "sha256": reference["sha256"], "pkml_sha256": pkml_sha}


def _metrics(run, case, device):
    row = score_calibration_curves(runs=[run], observations=case["observations"], device=device)["rows"][0]
    result = {"case_id": case["case_id"], "study_group_id": case["study_group_id"], **row,
              "auc_abs_log_error": None, "cmax_abs_log_error": None}
    if not row["valid"]:
        return result
    times = [p["time_min"] for p in case["observations"]]
    observed = [p["concentration_umol_per_l"] for p in case["observations"]]
    predicted = row["predictions_umol_per_l"]
    def auc(values):
        return sum((b - a) * (x + y) / 2 for a, b, x, y in
                   zip(times, times[1:], values, values[1:]))
    observed_auc, predicted_auc = auc(observed), auc(predicted)
    if not all(math.isfinite(v) and v > 0 for v in (observed_auc, predicted_auc)):
        result.update(valid=False, error="invalid_observed_window_auc")
        return result
    result.update(auc_abs_log_error=abs(math.log(predicted_auc) - math.log(observed_auc)),
                  cmax_abs_log_error=abs(math.log(max(predicted)) - math.log(max(observed))),
                  observed_window_auc={"observed": observed_auc, "predicted": predicted_auc,
                                       "unit": "umol*min/l", "start_min": times[0], "end_min": times[-1]},
                  sampled_cmax={"observed": max(observed), "predicted": max(predicted), "unit": "umol/l"})
    return result


def _aggregate(rows):
    keys = ("log_mse", "auc_abs_log_error", "cmax_abs_log_error")
    if not rows or any(not r["valid"] for r in rows):
        return {"valid": False, **{k: None for k in keys}, "per_study": {}, "rows": rows}
    studies = {}
    for row in rows:
        studies.setdefault(row["study_group_id"], []).append(row)
    per_study = {s: {k: sum(r[k] for r in group) / len(group) for k in keys}
                 for s, group in studies.items()}
    return {"valid": True, **{k: sum(r[k] for r in per_study.values()) / len(per_study) for k in keys},
            "per_study": per_study, "rows": rows}


def fit_select_source_mechanism(calibration_cases, selection_cases, parameter_specs, output_dir,
                                project_root, *, source_readback_receipts,
                                disclosed_context_assumptions, max_rounds=2,
                                device="cpu", tolerance=1e-6):
    """Calibrate one parameter and accept only a separately assessed candidate.

    Cases have the six fields used by ``fit_bounded_source_parameters``; their
    roles must be calibration/selection. Selection cases additionally require
    sampling_type='concentration_time_profile' (trough sequences are rejected).
    Receipts map case_id to an accepted
    ``source_adaptation_audit`` JSON path and SHA256 with source-value checks.
    The same fitted input must subsequently be applied to every drug scenario.
    Selection cannot tune it. AUC is observed-window AUC and Cmax is the peak at
    observation times, never extrapolated total exposure or a continuous peak.
    """
    root = Path(project_root).resolve(strict=True)
    destination = _inside(output_dir, root, must_exist=False)
    if destination.exists():
        raise FileExistsError("Mechanism calibration output already exists")
    if type(max_rounds) is not int or max_rounds not in {1, 2}:
        raise ValueError("Mechanism calibration allows one or two bounded rounds")
    tolerance = _number(tolerance)
    if not 0 <= tolerance <= 1e-6:
        raise ValueError("Endpoint tolerance is numerical and may not exceed 1e-6")
    if not isinstance(parameter_specs, list) or len(parameter_specs) != 1:
        raise ValueError("This stage allows exactly one mechanism parameter")
    spec = dict(parameter_specs[0])
    if set(spec) != _SPEC_FIELDS:
        raise ValueError("Parameter specification requires exactly the documented fields")
    for key in ("parameter_id", "unit", "evidence_reference"):
        spec[key] = _text(spec[key], key)
    spec["path"] = _quantity_path(spec["path"], "parameter path")
    for key in ("source_value", "lower_bound", "upper_bound"):
        spec[key] = _number(spec[key])
    low, source, high = (spec[k] for k in ("lower_bound", "source_value", "upper_bound"))
    if spec["scaling"] != "log" or not 0 < low <= source <= high or low == high:
        raise ValueError("Use one positive mechanism input with bounded log scaling")
    if low < source / 2 or high > source * 2:
        raise ValueError("This development stage limits search to source/2 .. source*2")
    if not isinstance(disclosed_context_assumptions, list) or not disclosed_context_assumptions:
        raise ValueError("Document context assumptions explicitly before fitting")
    assumptions = [_text(x, "context assumption", 8192) for x in disclosed_context_assumptions]
    if not isinstance(calibration_cases, list) or not calibration_cases or not isinstance(selection_cases, list) or not selection_cases:
        raise ValueError("Nonempty calibration and selection cases are required")
    training = [_case(c, "calibration", root, device) for c in calibration_cases]
    selection = [_case(c, "selection", root, device) for c in selection_cases]
    cases = training + selection
    ids = [c["case_id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Case identifiers must be unique across both roles")
    if {c["study_group_id"] for c in training} & {c["study_group_id"] for c in selection}:
        raise ValueError("Calibration and selection studies must be disjoint")
    if not isinstance(source_readback_receipts, dict) or set(source_readback_receipts) != set(ids):
        raise ValueError("Provide one source readback receipt for every case")
    receipts = {c["case_id"]: _receipt(source_readback_receipts[c["case_id"]], c, spec, root) for c in cases}
    request = {"version": VERSION, "calibration_cases": training, "selection_cases": selection,
               "parameter_specs": [spec], "source_readback_receipts": receipts,
               "disclosed_context_assumptions": assumptions, "max_rounds": max_rounds,
               "device": device, "endpoint_numerical_tolerance": tolerance,
               "selection_rule": "strictly lower study-equal log-MSE; neither study-equal AUC nor sampled-Cmax absolute log-error worsens beyond numerical tolerance",
               "source_scope": "historical development; no claim of independent validation"}
    destination.mkdir(parents=True)
    _write(destination / "mechanism_request.json", request)
    fit = fit_bounded_source_parameters(training, [spec], destination / "calibration", root,
                                        max_rounds=max_rounds, device=device)
    candidate = fit.get("selected_vector")
    files = {"request": str(destination / "mechanism_request.json"),
             "calibration_result": str(destination / "calibration/calibration_result.json")}
    result = {"version": VERSION, "status": "failed", "scientific_status": "historical_development_not_independent_validation",
              "source_values": [source], "candidate_values": candidate, "retained_values": [source],
              "parameter_specs": [spec], "disclosed_context_assumptions": assumptions,
              "decision": {"candidate_promoted": False, "reasons": ["calibration_failed"]},
              "metrics": {}, "files": files, "selection_batches": [],
              "source_files_overwritten": False, "parameters_installed": False}
    if fit.get("status") == "completed":
        if not isinstance(candidate, list) or len(candidate) != 1 or not low <= _number(candidate[0]) <= high:
            raise ValueError("Calibration candidate violates the frozen parameter contract")
        def run_case(pair):
            index, case = pair
            try:
                if _sha(Path(case["pkml_path"])) != receipts[case["case_id"]]["pkml_sha256"]:
                    raise ValueError("Selection PKML changed after source readback")
                return run_parameter_batch(pkml_path=Path(case["pkml_path"]),
                    parameter_specs=[{"path": spec["path"], "unit": spec["unit"]}],
                    values=[[source], candidate], target_output=case["target_output"],
                    output_dir=destination / "selection" / f"case_{index:03d}", project_root=root, cpu_threads=2)
            except (OSError, ValueError, RuntimeError) as exc:
                return {"status": "failed", "runs": [], "error": type(exc).__name__ + ": " + str(exc)}
        with ThreadPoolExecutor(max_workers=min(4, len(selection))) as executor:
            outcomes = list(executor.map(run_case, enumerate(selection)))
        # Persist both candidates before computing any selection scores.
        prediction_path = destination / "selection_predictions_locked.json"
        _write(prediction_path, {"source_values": [source], "candidate_values": candidate,
            "cases": [{"case_id": c["case_id"], "outcome": o} for c, o in zip(selection, outcomes)]})
        files.update(selection_predictions=str(prediction_path), selection_predictions_sha256=_sha(prediction_path))
        scores = {"source": [], "candidate": []}
        for case, outcome in zip(selection, outcomes):
            result["selection_batches"].append({"case_id": case["case_id"], "status": outcome["status"],
                                               "manifest_path": outcome.get("manifest_path"), "error": outcome.get("error")})
            runs = {r["vector_index"]: r for r in outcome.get("runs", [])}
            if outcome.get("status") != "completed" or set(runs) != {0, 1}:
                for arm in scores:
                    scores[arm].append({"case_id": case["case_id"], "study_group_id": case["study_group_id"],
                                        "valid": False, "error": outcome.get("error", "selection_batch_failed")})
                continue
            for index, arm in enumerate(scores):
                scores[arm].append(_metrics(runs[index], case, device))
        metrics = {arm: _aggregate(rows) for arm, rows in scores.items()}
        a, b = metrics["source"], metrics["candidate"]
        reasons = []
        if not a["valid"] or not b["valid"]:
            reasons.append("selection_failed")
        else:
            if candidate == [source] or not b["log_mse"] < a["log_mse"]:
                reasons.append("selection_curve_not_improved")
            for key in ("auc_abs_log_error", "cmax_abs_log_error"):
                if b[key] > a[key] + tolerance:
                    reasons.append("selection_" + key + "_worsened")
        promote = not reasons
        result.update(status="completed" if a["valid"] and b["valid"] else "failed", metrics=metrics,
            retained_values=candidate if promote else [source],
            decision={"candidate_promoted": promote, "reasons": reasons or ["selection_passed"],
                      "endpoint_numerical_tolerance": tolerance,
                      "selection_adjusted_candidate": False,
                      "evaluation_data_used": False})
    files["result"] = str(destination / "mechanism_result.json")
    _write(destination / "mechanism_result.json", result)
    return result
