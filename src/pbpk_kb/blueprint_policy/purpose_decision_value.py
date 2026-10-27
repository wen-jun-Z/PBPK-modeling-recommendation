"""Select one complete PBPK candidate using purpose-specific calibration references.

This module performs no I/O, parameter fitting, simulation or clinical-label
lookup.  Reference distributions come from the held-out reference-calibration
compounds, separate from the parameter predictor's fitting compounds and all
target compounds. The caller verifies source hashes and these exclusions. Their empirical
large-error frequency is an engineering review score, not a calibrated clinical
risk probability.  Conditional fu stability is deliberately not used as a
substitute for a complete predictive-error distribution.
"""
from __future__ import annotations

from copy import deepcopy
import math
from types import FunctionType

from . import evidence_decision as legacy
from .purpose_design import ENDPOINT_UNITS

VERSION = "purpose-decision-value-1"
METHODS = ("fixed_native", "fixed_capacity", "global_decision", "local_risk_only", "purpose_decision")
CANDIDATES = ("parameter_only", "combined")
QUALIFIED_PREDICTORS = frozenset({"random_forest", "RF", "extra_trees", "ExtraTrees"})
TIE_TOLERANCE = 1e-8
REVIEW_THRESHOLD = 0.5


def _qualified_prediction(record):
    metadata = record.get("prediction_method") or {}
    applicability = metadata.get("applicability")
    return (record.get("basis") == "predicted"
            and metadata.get("name") in QUALIFIED_PREDICTORS
            and isinstance(metadata.get("version"), str) and bool(metadata["version"].strip())
            and metadata.get("training_overlap") is False
            and isinstance(applicability, dict)
            and applicability.get("human_IV_parameter_model") is True
            and applicability.get("target_structure_and_scaffold_excluded") is True)


# Clone the two functions into a private namespace.  This shares the frozen
# evidence rules without monkey-patching a global or falsely renaming ET to RF.
_namespace = dict(legacy.decide.__globals__)
_namespace["_rf_qualified"] = _qualified_prediction
_namespace["_record_issue"] = FunctionType(legacy._record_issue.__code__, _namespace,
    name="_qualified_record_issue", argdefs=legacy._record_issue.__defaults__)
_qualified_decide = FunctionType(legacy.decide.__code__, _namespace,
    name="_qualified_decide", argdefs=legacy.decide.__defaults__)


def validate_request(request: dict, fallback=None, purpose=None) -> dict:
    """Apply the shared evidence/capability gate, allowing qualified RF or ET.

    Vss remains an independently qualified construction sidecar; it is not
    inserted into the frozen evidence schema.  Candidate construction must have
    used ``capacity_guided_design`` to validate such sidecars.  This gate never
    interprets predicted endpoint values as construction evidence.
    """
    if not isinstance(request, dict):
        raise TypeError("request must be a dict")
    normalized = deepcopy(request)
    embedded_purpose = normalized.pop("purpose", None)
    if purpose is not None and embedded_purpose is not None and purpose != embedded_purpose:
        raise ValueError("Conflicting explicit and embedded purposes")
    selected_purpose = purpose if purpose is not None else embedded_purpose
    result = _qualified_decide(normalized, [] if fallback is None else fallback,
                              purpose=selected_purpose, method="evidence_policy")
    # Source content verification stays with the runner, but explicit target
    # exclusions can already be enforced without reading any file.
    exclusions = normalized.get("forbidden_sources", [])
    exclusions = exclusions if isinstance(exclusions, list) else []
    forbidden_paths = {row.get("path") for row in exclusions if isinstance(row, dict)} - {None}
    forbidden_hashes = {row.get("sha256") for row in exclusions if isinstance(row, dict)} - {None}
    records = normalized.get("evidence", [])
    records = (list(records) if isinstance(records, list) else []) + ([] if fallback is None else fallback)
    for row in records:
        source = row.get("source", {}) if isinstance(row, dict) else {}
        source = source if isinstance(source, dict) else {}
        if source.get("path") in forbidden_paths or source.get("sha256") in forbidden_hashes:
            reason = "forbidden_outcome_source:" + str(row.get("parameter"))
            if reason not in result["reasons"]:
                result["reasons"].append(reason)
            result["status"] = "blocked"
    result["policy_version"] = VERSION
    result["qualified_predictors"] = sorted(QUALIFIED_PREDICTORS)
    result["source_verification"] = "required_by_caller_not_performed_here"
    result["clinical_validation"] = False
    return result


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _curve_issue(candidate):
    curve = candidate.get("curve")
    if not isinstance(curve, dict):
        return "missing_complete_curve"
    times, values = curve.get("time_h"), curve.get("concentration_mg_l")
    if (not isinstance(times, list) or not isinstance(values, list) or len(times) < 2
            or len(times) != len(values) or any(not _finite(v) for v in times + values)
            or times[0] < 0 or any(b <= a for a, b in zip(times, times[1:]))
            or any(value < 0 for value in values)):
        return "invalid_complete_curve"
    return None


def _candidate(candidate, primary):
    result = {"feasible": False, "reasons": [], "prediction_value": None,
              "unit": ENDPOINT_UNITS[primary], "expected_log_error": None,
              "risk": None, "reference_count": 0}
    if not isinstance(candidate, dict) or candidate.get("status") != "predicted":
        result["reasons"].append("candidate_not_predicted")
        return result
    curve_issue = _curve_issue(candidate)
    if curve_issue:
        result["reasons"].append(curve_issue)
        return result
    if primary == "plasma_concentration_time":
        result["feasible"] = True
        result["reasons"].append("scalar_reference_distribution_cannot_score_curve")
        return result
    endpoints = candidate.get("endpoints")
    endpoint = endpoints.get(primary) if isinstance(endpoints, dict) else None
    if (not isinstance(endpoint, dict) or endpoint.get("status", "completed") not in {"completed", "predicted"}
            or not _finite(endpoint.get("value")) or endpoint["value"] < 0
            or (endpoint["value"] == 0 and primary != "tmax")):
        result["reasons"].append("primary_endpoint_not_estimable")
        return result
    if not isinstance(endpoint.get("unit"), str) or endpoint["unit"].lower() != ENDPOINT_UNITS[primary]:
        result["reasons"].append("primary_endpoint_noncanonical_unit")
        return result
    result.update(feasible=True, prediction_value=float(endpoint["value"]))
    if endpoint["value"] == 0:
        result["reasons"].append("zero_endpoint_has_no_log_error_score")
    return result


def _references(distributions, primary, field):
    if not isinstance(distributions, dict):
        raise TypeError("reference_distributions must be a dict")
    row = distributions.get(primary)
    if row is None:
        return [], None
    if not isinstance(row, dict):
        raise ValueError("Each endpoint reference distribution must be a dict")
    if row.get("role", "training_reference_distribution") != "training_reference_distribution":
        raise ValueError("Only training reference distributions may guide decisions")
    unit = row.get("unit")
    if not isinstance(unit, str) or unit.lower() != ENDPOINT_UNITS[primary]:
        raise ValueError("Noncanonical reference distribution unit")
    values = row.get(field, [])
    if not isinstance(values, list) or any(not _finite(value) for value in values):
        raise ValueError("Reference distribution must contain finite natural-log values")
    # The legacy role name "training_reference_distribution" denotes an
    # independent reference-calibration distribution, not the predictor's
    # fitting labels. Target exclusion and origin are verified by the runner.
    provenance = {key: deepcopy(value) for key, value in row.items()
                  if key not in {"local_log_reference_values", "global_log_reference_values"}}
    return list(values), provenance


def decide_value(request, candidates, reference_distributions, *, primary_endpoint, method):
    """Choose one whole candidate and score its declared primary endpoint.

    Fixed controls have the same legal fallback as decision methods.  Global
    and local decisions minimize mean absolute natural-log prediction error
    against the respective *reference-calibration* distributions. A tie
    within 1e-8 favors ``combined``.  All requested endpoints are delivered from
    the same selected model, including unavailable secondary endpoints; this
    function never merges endpoint values from different candidates.
    """
    if method not in METHODS:
        raise ValueError("Unknown purpose decision method")
    if not isinstance(candidates, dict) or set(candidates) - set(CANDIDATES):
        raise ValueError("Only parameter_only and combined candidates are supported")
    if not isinstance(primary_endpoint, str) or primary_endpoint not in ENDPOINT_UNITS:
        raise ValueError("Unknown primary endpoint")
    gate = validate_request(request)
    requested = gate["purpose"]["endpoints"]
    if primary_endpoint not in requested:
        raise ValueError("The primary endpoint must belong to the requested purpose")
    field = ("local_log_reference_values" if method in {"purpose_decision", "local_risk_only"}
             else "global_log_reference_values")
    reference, provenance = _references(reference_distributions, primary_endpoint, field)
    result = {
        "version": VERSION, "method": method, "status": "blocked", "action": "obtain_evidence",
        "selected_candidate": None, "primary_endpoint": primary_endpoint,
        "requested_endpoints": deepcopy(requested), "reasons": list(gate["reasons"]),
        "candidate_assessments": {}, "expected_log_error": None, "risk": None,
        "review_required": None, "review_threshold": REVIEW_THRESHOLD,
        "reference_kind": field, "distribution_provenance": provenance,
        "input_gate": gate, "whole_candidate_selection": True, "clinical_validation": False,
        "score_interpretation": "Empirical held-out reference-calibration error frequency; uncalibrated engineering review score",
        "conditional_fu_interpretation": "Conditional sensitivity is not total predictive uncertainty and is not this risk score",
        "limitations": ["no_target_outcome_read_or_used_by_this_module",
                        "predictor_fit_calibration_target_exclusion_and_source_integrity_checked_by_caller",
                        "selection_not_clinical_validation", "no_endpoint_mixing",
                        "review_threshold_not_a_clinical_safety_boundary"],
    }
    if gate["status"] != "ready":
        return result
    assessments = {name: _candidate(candidates.get(name), primary_endpoint) for name in CANDIDATES}
    for row in assessments.values():
        value = row["prediction_value"]
        if row["feasible"] and value is not None and value > 0 and reference:
            errors = [abs(math.log(value) - point) for point in reference]
            row.update(expected_log_error=sum(errors) / len(errors),
                       risk=sum(error > math.log(2.) for error in errors) / len(errors),
                       reference_count=len(errors))
    result["candidate_assessments"] = assessments
    feasible = [name for name in CANDIDATES if assessments[name]["feasible"]]
    if not feasible:
        result["reasons"].append("no_feasible_candidate_for_primary_endpoint")
        return result
    preference = ("parameter_only", "combined") if method == "fixed_native" else ("combined", "parameter_only")
    selected = next(name for name in preference if name in feasible)
    if method in {"global_decision", "purpose_decision"}:
        scored = [name for name in feasible if assessments[name]["expected_log_error"] is not None]
        if scored:
            best = min(assessments[name]["expected_log_error"] for name in scored)
            tied = [name for name in preference if name in scored and math.isclose(
                assessments[name]["expected_log_error"], best,
                rel_tol=TIE_TOLERANCE, abs_tol=TIE_TOLERANCE)]
            selected = tied[0]
    chosen = assessments[selected]
    result.update(selected_candidate=selected, expected_log_error=chosen["expected_log_error"], risk=chosen["risk"])
    if chosen["risk"] is None:
        result.update(status="unassessed", action="research_prediction_without_risk_assessment")
        result["reasons"].append("no_usable_primary_reference_distribution")
    else:
        review = chosen["risk"] >= REVIEW_THRESHOLD
        result.update(status="review_required" if review else "research_prediction",
                      action="review_evidence" if review else "retain_research_prediction",
                      review_required=review)
    if selected != preference[0] and method not in {"global_decision", "purpose_decision"}:
        result["reasons"].append("legal_candidate_fallback")
    return result
