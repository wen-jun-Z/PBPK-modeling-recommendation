"""Endpoint-specific CL uncertainty decisions; no clinical labels at design time.

The parameter interval is NOT a clinical concentration prediction interval.
Independent measured CL is preserved; its uncertainty is unknown, not zero.
All point predictions are retained regardless of these research-only decisions.
"""
from __future__ import annotations

from copy import deepcopy
import math

VERSION = "clearance-uncertainty-policy-1"
FOLD_TOLERANCE = 2.0  # Prespecified engineering tolerance, not clinical acceptance.
SUPPORTED = ("plasma_concentration_time", "auc", "cmax", "tmax", "c_last",
             "terminal_half_life", "auc_inf", "clearance", "vz", "mrt", "vss")


def _positive(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError("A finite positive value is required")
    return float(value)


def decide(request: dict, interval: dict | None, *, reviewed_legacy_source=None,
           construction_blocked=None) -> dict:
    """Assess the CL contribution only, before any clinical output is read.

    The caller verifies receipts and exact-input equality for frozen, previously
    source-qualified evidence. Its raw record is retained without relabelling
    older context fields. A receipt alone does not qualify a different record.
    """
    if construction_blocked:
        return {"version": VERSION, "compound_id": request["compound_id"],
            "observations_used": False, "clinical_coverage_guaranteed": False,
            "point_prediction_modified": False, "prediction_values_retained": True,
            "action": "construction_blocked", "reason": deepcopy(construction_blocked),
            "risk_score": None, "interval": None, "endpoint_decisions": {},
            "new_clearance_prior_applied": False}
    records = [r for r in request["evidence"] if r["parameter"] == "total_clearance"]
    if len(records) != 1:
        raise ValueError("This pilot requires exactly one systemic total-CL input")
    record = records[0]
    context = record.get("context", {})
    legacy = False
    if reviewed_legacy_source is not None:
        if (record["basis"] not in {"measured", "derived"}
                or reviewed_legacy_source.get("record") != record):
            raise ValueError("Legacy qualification must match the preserved evidence record")
        for name in ("source_qualified_protocol", "source_outcome"):
            receipt = reviewed_legacy_source.get(name, {})
            digest = receipt.get("sha256", "")
            if (not receipt.get("path") or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Verified legacy source receipts are required")
        legacy = True
    independent = (context.get("independent_of_target") is True
        and context.get("study_role") in {"independent_parameters", "independent_measurement"})
    if (record.get("compound_id") != request["compound_id"] or record.get("unit") != "ml/min/kg"
            or not (independent or legacy)
            or context.get("study_role") in {"target", "test", "validation"}
            or context.get("species") != "human" or context.get("matrix") != "plasma"
            or context.get("route") not in {"iv", "intravenous"}):
        raise ValueError("Qualified independent human plasma IV clearance is required")
    value = _positive(record["value"])
    endpoints = request.get("target", {}).get("endpoints", list(SUPPORTED))
    if any(name not in SUPPORTED for name in endpoints):
        raise ValueError("Unsupported endpoint in this pilot")
    result = {"version": VERSION, "compound_id": request["compound_id"],
        "observations_used": False, "clinical_coverage_guaranteed": False,
        "point_prediction_modified": False, "source": deepcopy(record.get("source")),
        "cl_basis": record["basis"], "risk_score": None, "decision_threshold_fold": FOLD_TOLERANCE,
        "interval": None, "endpoint_decisions": {}, "prediction_values_retained": True}
    if record["basis"] in {"measured", "derived"}:
        result.update(action="preserve_independent_clearance", reason="Independent evidence uncertainty has not been quantified")
        if legacy:
            result["reviewed_legacy_source"] = deepcopy(reviewed_legacy_source)
        result["endpoint_decisions"] = {name: "uncertainty_unassessed" for name in endpoints}
        return result
    if record["basis"] != "predicted" or not isinstance(interval, dict):
        raise ValueError("Missing CL requires the matching predictor's calibrated interval")
    lo, median, hi = (_positive(interval[k]) for k in ("lower", "median", "upper"))
    if (interval.get("unit") != "mL/min/kg" or not lo <= median <= hi
            or not math.isclose(value, median, rel_tol=1e-9, abs_tol=1e-12)
            or interval.get("calibration") != "split_CQR"
            or interval.get("clinical_coverage_guaranteed") is not False):
        raise ValueError("Interval differs from the frozen CL input or method")
    max_fold = max(median / lo, hi / median)
    broad = max_fold > FOLD_TOLERANCE
    result.update(interval=deepcopy(interval), risk_score=math.log(hi / lo),
        maximum_cl_fold_from_point=max_fold,
        action="request_clearance_evidence" if broad else "research_parameter_range_within_tolerance",
        reason="CL uncertainty exceeds the prespecified research tolerance" if broad else "Only CL parameter uncertainty meets the prespecified tolerance")
    result["endpoint_decisions"] = {name: (result["action"] if name in {"clearance", "auc_inf"}
                                           else "uncertainty_unassessed") for name in endpoints}
    # This identity concerns the systemic single-IV model, not finite-window
    # observed AUC, NCA extrapolation uncertainty or a full clinical interval.
    if "auc_inf" in endpoints:
        if request["dosing"]["route"] != "intravenous":
            raise ValueError("AUC identity is restricted to single IV parent-drug disposition")
        dose = _positive(request["dosing"]["dose_mg"])
        factor = .06 * _positive(request["human"]["weight_kg"])
        result["systemic_auc_inf_cl_only_range"] = {"lower": dose/(hi*factor), "upper": dose/(lo*factor),
            "unit": "mg.h/L", "basis": "Dose/CL identity with other inputs fixed",
            "used_as_predicted_NCA_endpoint": False, "clinical_prediction_interval": False}
    return result


def _selective_curve(rows):
    """All forecasts remain; describe errors after ranking by CL width only.

    Fractional inclusion of score ties avoids choosing compounds by identifier.
    The fixed fractions are diagnostic, not fitted deployment cutoffs.
    """
    eligible = [r for r in rows if r["score"] is not None]
    scored = [r for r in eligible if r["gmfe"] is not None]
    # The selected set depends on frozen input scores, never on whether a
    # prediction succeeded or later obtained an observation-based score.
    ordered = sorted(eligible, key=lambda r: r["score"])
    points = []
    for fraction in (.25, .5, .75, 1.):
        if not ordered:
            points.append({"fraction_of_eligible_attempts": fraction, "selected_attempt_weight": 0,
                "selected_scored_weight": 0, "selected_failed_or_unscored_weight": 0, "gmfe": None,
                "coverage_of_eligible_attempts": None, "scored_coverage_of_eligible_attempts": None})
            continue
        n = max(1, math.ceil(fraction * len(ordered)))
        cutoff = ordered[n-1]["score"]
        strict = [r for r in ordered if r["score"] < cutoff]
        tied = [r for r in ordered if r["score"] == cutoff]
        w = (n-len(strict))/len(tied)
        valid_strict = [r for r in strict if r["gmfe"] is not None]
        valid_tied = [r for r in tied if r["gmfe"] is not None]
        scored_weight = len(valid_strict) + w*len(valid_tied)
        total = sum(math.log(r["gmfe"]) for r in valid_strict) + w*sum(math.log(r["gmfe"]) for r in valid_tied)
        points.append({"fraction_of_eligible_attempts": fraction, "selected_attempt_weight": n,
            "selected_scored_weight": scored_weight, "selected_failed_or_unscored_weight": n-scored_weight,
            "gmfe": math.exp(total/scored_weight) if scored_weight else None,
            "coverage_of_eligible_attempts": n/len(eligible),
            "scored_coverage_of_eligible_attempts": scored_weight/len(eligible), "tie_weight": w,
            "cutoff_log_cl_interval_width": cutoff})
    full = math.exp(sum(math.log(r["gmfe"]) for r in scored)/len(scored)) if scored else None
    # Average pairwise concordance; higher width should indicate larger error.
    positives = [r for r in scored if r["gmfe"] >= 2.]
    negatives = [r for r in scored if r["gmfe"] < 2.]
    auc = (sum(float(a["score"] > b["score"]) + .5*float(a["score"] == b["score"])
               for a in positives for b in negatives)/(len(positives)*len(negatives))) if positives and negatives else None
    return {"eligible_predicted_cl_attempts": len(eligible), "scored": len(scored),
        "unscored_or_failed": len(eligible)-len(scored), "full_scored_gmfe": full,
        "risk_coverage": points, "high_error_definition": "curve GMFE >= 2",
        "high_error_auroc": auc, "high_error_count": len(positives),
        "scope": "Diagnostic ranking by CL uncertainty only; not calibrated clinical risk"}


def summarize(units: list[dict], decisions: dict[str, dict]) -> dict:
    """Called only by the runner after every forecast is locked and scored.

    Decisions are keyed by each unit's scientific fingerprint. This function
    never selects/refits a model or deletes a forecast based on its score.
    """
    by_condition = {}
    for unit in units:
        decision = decisions[unit["fingerprint"]]
        score = decision["risk_score"]
        by_condition.setdefault(unit["condition"], []).append({"case_id": unit["case_id"],
            "score": score, "gmfe": (unit.get("curve_metrics") or {}).get("gmfe"),
            "action": decision["action"]})
    summaries = {}
    for condition, rows in by_condition.items():
        actions = {name: sum(r["action"] == name for r in rows) for name in sorted({r["action"] for r in rows})}
        summaries[condition] = {**_selective_curve(rows), "attempted": len(rows), "actions": actions,
            "measured_cl_uncertainty_not_evaluated": sum(r["action"] == "preserve_independent_clearance" for r in rows),
            "construction_blocked": sum(r["action"] == "construction_blocked" for r in rows)}
    return {"version": VERSION, "conditions": summaries, "retention_fractions_prespecified": True,
        "cutoffs_depend_only_on_frozen_input_scores": True,
        "conditional_accuracy_does_not_replace_all_drug_results": True,
        "limitations": ["CL intervals do not cover Vss, fu, half-life, structural or study-population uncertainty",
            "Scaffold distribution shift precludes claiming exchangeable-data coverage for clinical outcomes",
            "Independent measured CL uncertainty is unknown, not zero",
            "No drug or forecast was excluded from the main evaluation by this diagnostic policy"]}
