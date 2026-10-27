"""Post-lock evaluation of aggregated human-IV PK labels, never curve validation.

The purpose policy and fixed automatic PBPK comparator share one nominal
prediction. Only their acceptance/ranking decisions differ. Clinical labels are
opened after every registered compound has a terminal, hash-locked outcome.
Bootstrap units are compounds because these labels contain no study identifiers.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

VERSION = "purpose-hundred-evaluation-1"
CODE_FILES = ["src/pbpk_kb/blueprint_policy/hundred_evaluation.py"]
SCOPE = "aggregated_human_IV_PK_parameters"
BOOTSTRAP_REPEATS = 2000
SEED = 20260930
COVERAGES = (.2, .4, .6, .8, 1.)
ENDPOINT_UNITS = {
    "plasma_concentration_time": "mg/l", "auc": "mg*h/l", "cmax": "mg/l",
    "tmax": "h", "c_last": "mg/l", "terminal_half_life": "h",
    "auc_inf": "mg*h/l", "clearance": "l/h", "vz": "l", "mrt": "h", "vss": "l",
}
REFERENCE_ENDPOINTS = frozenset({"clearance", "vss", "terminal_half_life", "mrt"})
POLICY_STATUSES = frozenset({"within_declared_limits", "sensitive", "unassessed", "not_assessed"})


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                            allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _finite(value, *, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (not positive or value > 0))


def _unit(value):
    return str(value).strip().lower().replace(" ", "")


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _ci(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def _checked_evaluation(reference, root):
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("Each case needs a frozen evaluation path and SHA256")
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evaluation must remain inside this project")
    data = path.read_bytes()
    if sha256(data).hexdigest() != reference.get("sha256"):
        raise ValueError("Evaluation source SHA256 changed: " + reference["path"])
    return json.loads(data)


def _complete_barrier(protocol, outcomes):
    """Validate *all* predictions before the first reference file is opened."""
    cases = protocol.get("cases")
    count = protocol.get("expected_compounds", 100)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("expected_compounds must be a positive integer")
    if (not isinstance(cases, list) or not isinstance(outcomes, list)
            or len(cases) != count or len(outcomes) != count):
        raise ValueError("All registered compounds must be locked before scoring")
    ids, identities, compounds = set(), set(), set()
    for case in cases:
        for key, seen in (("case_id", ids), ("compound_id", compounds), ("identity_smiles", identities)):
            value = case.get(key)
            if not isinstance(value, str) or not value or value in seen:
                raise ValueError("Cases need distinct case, compound and chemical identities")
            seen.add(value)
        if case.get("evaluation_scope") != SCOPE:
            raise ValueError("This scorer supports aggregated human-IV parameters only")
    indexed = {}
    for row in outcomes:
        cid = row.get("case_id")
        if cid not in ids or cid in indexed or row.get("status") != "locked":
            raise ValueError("All registered compounds must be uniquely locked before scoring")
        if row.get("construction_status") not in {"predicted", "failed", "blocked"}:
            raise ValueError("A locked outcome needs a terminal construction status")
        predictions = row.get("predictions")
        if not isinstance(predictions, list):
            raise ValueError("Missing locked predictions")
        sids = [p.get("scenario_id") for p in predictions]
        if len(set(sids)) != len(sids) or any(not isinstance(s, str) for s in sids):
            raise ValueError("Prediction scenario identifiers must be unique")
        if any(p.get("status") not in {"predicted", "failed"} for p in predictions):
            raise ValueError("All prediction scenarios must be terminal")
        inline = row.get("inline_prediction_sha256")
        actual = {p["scenario_id"]: _digest(p) for p in predictions}
        if inline != actual:
            raise ValueError("Locked inline prediction SHA256 changed")
        baseline = next((p for p in predictions if p["scenario_id"] == "baseline"), None)
        if row.get("baseline_scenario_id") != "baseline":
            raise ValueError("The prespecified nominal scenario is baseline")
        fixed = row.get("fixedauto_baseline", {})
        if (fixed.get("scenario_id") != "baseline" or fixed.get("same_nominal_prediction") is not True
                or fixed.get("prediction_sha256") != (_digest(baseline) if baseline else None)):
            raise ValueError("Fixed automatic PBPK must use the identical nominal prediction")
        successful = baseline is not None and baseline["status"] == "predicted"
        if successful != (row["construction_status"] == "predicted"):
            raise ValueError("Construction status disagrees with the nominal prediction")
        indexed[cid] = row
    for case in cases:
        row = indexed[case["case_id"]]
        if any(row.get(key) != case.get(key) for key in ("compound_id", "identity_smiles", "evaluation_scope")):
            raise ValueError("Outcome identity differs from the frozen case")
    return cases, indexed


def _reference_values(evaluation, case):
    if (evaluation.get("compound_id") != case["compound_id"]
            or evaluation.get("identity_smiles") != case["identity_smiles"]):
        raise ValueError("Evaluation chemical identity does not match the frozen case")
    if evaluation.get("evaluation_scope") != SCOPE or evaluation.get("clinical_curve") is not None:
        raise ValueError("Aggregated labels cannot be presented as clinical curve validation")
    weight = evaluation.get("weight_kg")
    if not _finite(weight, positive=True):
        raise ValueError("Aggregated reference conversion needs an explicit positive weight")
    values = evaluation.get("endpoints")
    if not isinstance(values, dict):
        raise ValueError("Evaluation endpoints must be a dictionary")
    output = {}
    for name, item in values.items():
        if not isinstance(item, dict):
            raise ValueError("Every reference endpoint needs value and unit fields")
        value = item.get("value")
        if value is None:
            continue
        if name not in REFERENCE_ENDPOINTS:
            raise ValueError("Unexpected reference endpoint for the aggregated PK dataset")
        if not _finite(value, positive=True) or _unit(item.get("unit")) != ENDPOINT_UNITS[name]:
            raise ValueError("Invalid reference value or canonical unit: " + name)
        output[name] = float(value)
    return output


def _prediction_value(baseline, name):
    if baseline is None or baseline.get("status") != "predicted":
        return None, "nominal_prediction_failed"
    item = baseline.get("endpoints", {}).get(name)
    if not isinstance(item, dict) or item.get("value") is None:
        return None, "endpoint_not_estimable"
    if item.get("status", "completed") not in {"completed", "predicted"}:
        return None, "endpoint_not_estimable"
    value = item.get("value")
    if _unit(item.get("unit")) != ENDPOINT_UNITS[name]:
        return None, "noncanonical_prediction_unit"
    if not _finite(value) or value < 0 or (name in REFERENCE_ENDPOINTS and value == 0):
        return None, "invalid_prediction_value"
    return float(value), None


def _curve_info(baseline):
    if baseline is None or baseline.get("status") != "predicted":
        return False, 0, None
    curve = baseline.get("curve", {})
    times, values = curve.get("time_h"), curve.get("concentration_mg_l")
    if (not isinstance(times, list) or not isinstance(values, list) or len(times) < 2
            or len(times) != len(values) or any(not _finite(t) for t in times)
            or any(not _finite(v) or v < 0 for v in values)
            or any(right <= left for left, right in zip(times, times[1:]))):
        return False, 0, None
    return True, len(times), _digest(curve)


def _records(case, outcome, evaluation):
    refs = _reference_values(evaluation, case)
    baseline = next((p for p in outcome["predictions"] if p["scenario_id"] == "baseline"), None)
    assessments = outcome.get("assessment", {}).get("endpoint_assessments", {})
    curve_available, curve_n, curve_sha = _curve_info(baseline)
    input_score = outcome.get("input_only_score")
    if input_score is not None and (not _finite(input_score) or input_score < 1):
        raise ValueError("input_only_score must be a finite range ratio >= 1, or null")
    for name, unit in ENDPOINT_UNITS.items():
        value, reason = _prediction_value(baseline, name)
        available = value is not None
        if name == "plasma_concentration_time":
            available, reason = curve_available, None if curve_available else "curve_not_available"
        ref = refs.get(name)
        log_error = abs(math.log(value) - math.log(ref)) if value is not None and ref is not None else None
        fold = math.exp(log_error) if log_error is not None and log_error <= math.log(float.fromhex('0x1.fffffffffffffp+1023')) else None
        assessment = assessments.get(name, {})
        policy = assessment.get("status", "unassessed")
        if policy not in POLICY_STATUSES:
            raise ValueError("Unexpected endpoint policy status")
        sensitivity = assessment.get("max_change")
        unbounded = assessment.get("unbounded_change", False)
        if not isinstance(unbounded, bool):
            raise ValueError("unbounded_change must be a boolean")
        if sensitivity is not None and (not _finite(sensitivity) or sensitivity < 0):
            raise ValueError("Sensitivity max_change must be finite and nonnegative, or null")
        yield {
            "case_id": case["case_id"], "compound_id": case["compound_id"],
            "identity_smiles": case["identity_smiles"], "drug_name": case.get("drug_name"),
            "cohort_role": case.get("cohort_role"), "evaluation_scope": SCOPE,
            "construction_status": outcome["construction_status"], "endpoint": name, "unit": unit,
            "prediction_value": value, "prediction_available": available,
            "prediction_missing_reason": reason, "reference_value": ref,
            "scorable": log_error is not None, "fold_error": fold,
            "fold_error_overflow": log_error is not None and fold is None,
            "absolute_ln_error": log_error,
            "within_twofold": log_error <= math.log(2.) + 1e-12 if log_error is not None else None,
            "policy_status": policy, "policy_accepted": policy == "within_declared_limits",
            "policy_rejected": policy == "sensitive",
            "sensitivity_score": sensitivity, "sensitivity_unbounded": bool(unbounded),
            "input_only_score": input_score,
            "fixed_auto_same_nominal_prediction": True,
            "fixed_auto_accepted": available,
            "curve_samples": curve_n if name == "plasma_concentration_time" else None,
            "curve_sha256": curve_sha if name == "plasma_concentration_time" else None,
            "reference_weight_kg": evaluation["weight_kg"],
            "reference_source": evaluation.get("source", evaluation.get("sources")),
            "reference_artifact": case["evaluation"],
            "reference_is_study_matched": False,
        }


def _nominal_metrics(paired):
    if not paired:
        return {"gmfe": None, "gmfe_ci95": None, "mean_absolute_ln_error": None,
                "within_twofold_n": 0, "within_twofold_rate": None, "within_twofold_ci95": None}
    errors = np.array([r["absolute_ln_error"] for r in paired])
    within = np.array([r["within_twofold"] for r in paired], dtype=float)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(len(paired), size=(BOOTSTRAP_REPEATS, len(paired)))
    # Log errors avoid loss of precision and overflow when forming raw ratios.
    mean = float(errors.mean())
    log_ci = _ci(errors[samples].mean(axis=1))
    def exp_or_none(x):
        return math.exp(x) if x < 709.78 else None
    return {"gmfe": exp_or_none(mean), "gmfe_ci95": [exp_or_none(x) for x in log_ci],
            "mean_absolute_ln_error": mean, "mean_absolute_ln_error_ci95": log_ci,
            "within_twofold_n": int(within.sum()), "within_twofold_rate": float(within.mean()),
            "within_twofold_ci95": _ci(within[samples].mean(axis=1))}


def _endpoint_metrics(rows, total):
    paired = [r for r in rows if r["scorable"]]
    accepted = [r for r in rows if r["policy_accepted"]]
    accepted_paired = [r for r in paired if r["policy_accepted"]]
    rejected_paired = [r for r in paired if r["policy_rejected"]]
    unassessed_paired = [r for r in paired if not r["policy_accepted"] and not r["policy_rejected"]]
    false_accepts = sum(not r["within_twofold"] for r in accepted_paired)
    false_rejects = sum(r["within_twofold"] for r in rejected_paired)
    good = sum(r["within_twofold"] for r in paired)
    reference_count = sum(r["reference_value"] is not None for r in rows)
    result = {
        "total_compounds": total, "prediction_available_n": sum(r["prediction_available"] for r in rows),
        "reference_available_n": reference_count, "scorable_n": len(paired),
        "missing_reference_n": total - reference_count,
        "reference_available_prediction_missing_n": reference_count - len(paired),
        "prediction_unavailable_n": total - sum(r["prediction_available"] for r in rows),
        "prediction_missing_reason_counts": dict(Counter(r["prediction_missing_reason"] for r in rows
                                                         if not r["prediction_available"])),
        "policy_status_counts": dict(Counter(r["policy_status"] for r in rows)),
        "accepted_n": len(accepted), "accepted_coverage_all_compounds": len(accepted) / total,
        "accepted_with_prediction_n": sum(r["prediction_available"] for r in accepted),
        "accepted_scorable_n": len(accepted_paired),
        "accepted_scorable_coverage": _rate(len(accepted_paired), len(paired)),
        "accepted_without_reference_n": sum(r["reference_value"] is None for r in accepted),
        "false_acceptance_n": false_accepts,
        "false_acceptance_rate_among_accepted_scorable": _rate(false_accepts, len(accepted_paired)),
        "explicitly_rejected_scorable_n": len(rejected_paired),
        "false_rejection_n": false_rejects,
        "false_rejection_rate_among_explicitly_rejected_scorable": _rate(false_rejects, len(rejected_paired)),
        "false_rejection_rate_among_within_twofold": _rate(false_rejects, good),
        "unassessed_scorable_n": len(unassessed_paired),
        "unassessed_within_twofold_n": sum(r["within_twofold"] for r in unassessed_paired),
        "decision_error_definition": "Twofold empirical error proxy; not clinical appropriateness or patient risk",
        "nominal": _nominal_metrics(paired), "policy_accepted_nominal": _nominal_metrics(accepted_paired),
    }
    result["fixed_auto_baseline"] = {
        "same_nominal_predictions": True, "nominal": dict(result["nominal"]),
        "accepted_n": result["prediction_available_n"],
        "accepted_coverage_all_compounds": result["prediction_available_n"] / total,
        "accepted_scorable_n": len(paired),
        "false_acceptance_n": len(paired) - result["nominal"]["within_twofold_n"],
        "false_acceptance_rate_among_accepted_scorable": 1. - result["nominal"]["within_twofold_rate"] if paired else None,
        "explicit_rejections": 0,
        "interpretation": "Same viable nominal PBPK construction with no endpoint sensitivity gate; no predictive accuracy contrast",
    }
    return result


def _risk_coverage(rows, total):
    paired = sorted((r for r in rows if r["scorable"]), key=lambda r: r["case_id"])
    n = len(paired)
    result = {"common_scorable_pool_n": n, "excluded_from_numeric_risk_n": total - n,
              "registered_compounds": total, "coverage_denominator": "same scorable compound pool for both rankings",
              "missing_or_unbounded_scores_rank_last": True,
              "tie_breaker": "ascending SHA256(case_id), fixed before labels are read",
              "risk_interpretation": "Empirical aggregated PK-label error, not clinical risk", "points": []}
    if not n:
        return result
    tie = [sha256(r["case_id"].encode()).hexdigest() for r in paired]
    errors = np.array([r["absolute_ln_error"] for r in paired])
    bad = np.array([not r["within_twofold"] for r in paired], dtype=float)
    scores = {
        "endpoint_sensitivity": [r["sensitivity_score"] if not r["sensitivity_unbounded"]
                                 and r["policy_status"] in {"within_declared_limits", "sensitive"}
                                 else None for r in paired],
        "input_only": [r["input_only_score"] for r in paired],
    }
    rng = np.random.default_rng(SEED)
    samples = rng.integers(n, size=(BOOTSTRAP_REPEATS, n))
    orders, sample_orders = {}, {}
    for method, values in scores.items():
        order = sorted(range(n), key=lambda i: (math.inf if values[i] is None else values[i], tie[i]))
        orders[method] = np.asarray(order)
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        sample_orders[method] = np.take_along_axis(samples, np.argsort(rank[samples], axis=1, kind="stable"), axis=1)
    result["unknown_rank_score_n"] = {key: sum(v is None for v in values) for key, values in scores.items()}
    for coverage in COVERAGES:
        selected_n = math.ceil(coverage * n)
        point = {"requested_coverage": coverage, "selected_n": selected_n,
                 "actual_coverage_scorable_pool": selected_n / n,
                 "actual_coverage_registered_compounds": selected_n / total, "methods": {}}
        draws = {}
        for method in scores:
            selected = orders[method][:selected_n]
            resampled = sample_orders[method][:, :selected_n]
            error_draw = errors[resampled].mean(axis=1)
            bad_draw = bad[resampled].mean(axis=1)
            draws[method] = (error_draw, bad_draw)
            point["methods"][method] = {
                "selected_case_ids": [paired[i]["case_id"] for i in selected],
                "mean_absolute_ln_error": float(errors[selected].mean()),
                "mean_absolute_ln_error_ci95": _ci(error_draw),
                "over_twofold_n": int(bad[selected].sum()), "over_twofold_rate": float(bad[selected].mean()),
                "over_twofold_rate_ci95": _ci(bad_draw),
            }
        left, right = point["methods"]["endpoint_sensitivity"], point["methods"]["input_only"]
        point["paired_difference_sensitivity_minus_input_only"] = {
            "mean_absolute_ln_error": left["mean_absolute_ln_error"] - right["mean_absolute_ln_error"],
            "mean_absolute_ln_error_ci95": _ci(draws["endpoint_sensitivity"][0] - draws["input_only"][0]),
            "over_twofold_rate": left["over_twofold_rate"] - right["over_twofold_rate"],
            "over_twofold_rate_ci95": _ci(draws["endpoint_sensitivity"][1] - draws["input_only"][1]),
        }
        result["points"].append(point)
    return result


def score_hundred(protocol: dict, outcomes: list[dict], project_root: Path) -> dict:
    """Evaluate all locked compounds; return JSON-safe rows and separate metrics.

    ``expected_compounds`` defaults to 100 and can be explicitly reduced for
    synthetic tests. The runner separately verifies full source/code/artifact
    pins; this function independently checks the complete lock barrier, inline
    prediction hashes, identical fixed baseline and held-out evaluation hashes.
    """
    root = Path(project_root).resolve()
    cases, indexed = _complete_barrier(protocol, outcomes)
    records = []
    for case in sorted(cases, key=lambda c: c["case_id"]):
        evaluation = _checked_evaluation(case["evaluation"], root)
        records.extend(_records(case, indexed[case["case_id"]], evaluation))
    count = len(cases)
    by_endpoint = {name: [r for r in records if r["endpoint"] == name] for name in ENDPOINT_UNITS}
    status = Counter(row["construction_status"] for row in outcomes)
    result = {
        "version": VERSION, "evaluation_scope": SCOPE, "total_compounds": count,
        "all_predictions_locked_before_scoring": True,
        "clinical_curve_validation_n": 0, "same_nominal_predictions_for_both_methods": True,
        "construction": {
            "total": count, "predicted": status["predicted"], "blocked": status["blocked"],
            "failed": status["failed"], "success_rate": status["predicted"] / count,
            "cases_with_failed_scenarios": sum(any(p["status"] == "failed" for p in r["predictions"]) for r in outcomes),
            "failed_scenarios": sum(p["status"] == "failed" for r in outcomes for p in r["predictions"]),
            "failures": [{"case_id": r["case_id"], "construction_status": r["construction_status"],
                          "failures": r.get("failures", [])} for r in outcomes
                         if r["construction_status"] != "predicted" or r.get("failures")],
        },
        "bootstrap": {"unit": "compound", "repeats": BOOTSTRAP_REPEATS, "seed": SEED,
                      "confidence_level": .95, "method": "paired percentile resampling with replacement",
                      "risk_rankings_recomputed_within_each_draw": True,
                      "study_clustered": False, "small_sample_intervals_are_descriptive": True},
        "endpoint_metrics": {name: _endpoint_metrics(rows, count) for name, rows in by_endpoint.items()},
        "risk_coverage": {name: _risk_coverage(rows, count) for name, rows in by_endpoint.items()},
        "limitations": [
            "PKSmart labels aggregate human-IV parameter reports; no matched dose, population, study or clinical time series is provided.",
            "Absolute CL and Vss labels use the explicit reference weight; this is not a documented matched clinical participant.",
            "Only clearance, Vss, terminal half-life and MRT can be scored here; absent clinical endpoints remain null.",
            "Passing an input-sensitivity limit does not establish clinical reliability; stress ranges are not clinical confidence intervals.",
            "Nominal predictions are identical for purpose-aware and fixed automatic PBPK; only decision coverage and error ordering are compared.",
            "False acceptance/rejection uses a descriptive twofold error proxy and is not a clinical appropriateness label.",
            "Missing labels or nonestimable predictions cannot count as prediction success; all registered compounds remain in the reported denominators.",
            "No aggregate score combines different endpoints, and no unsuccessful drug or method is removed.",
        ],
        "records": records,
    }
    # Reject any accidental NaN/Infinity before the runner commits its report.
    json.dumps(result, allow_nan=False)
    return result
