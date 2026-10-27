"""Descriptive research checks for matched-reference PBPK predictions.

These project thresholds are not regulatory acceptance criteria or evidence of
clinical readiness. The caller must keep development and external validation
results separate. No reference data, fitting, or overall quality score is created.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real


RESEARCH_FOLD_LIMIT = 2.0
OPTIMIZATION_FOLD_LIMIT = 1.5


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        return None
    return value


def _exp(value):
    try:
        result = math.exp(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) and result > 0 else None


def _within(value, limit):
    return value <= limit or math.isclose(value, limit, rel_tol=1e-12, abs_tol=0.0)


def _assessment(value, metric, reason=None):
    return {
        "metric": metric,
        "value": value,
        "reason": reason,
        "research_screening": {
            "threshold": RESEARCH_FOLD_LIMIT,
            "status": "unassessable" if value is None else
                      "pass" if _within(value, RESEARCH_FOLD_LIMIT) else "fail",
        },
        "optimization_target": {
            "threshold": OPTIMIZATION_FOLD_LIMIT,
            "status": "unassessable" if value is None else
                      "pass" if _within(value, OPTIMIZATION_FOLD_LIMIT) else "fail",
        },
    }


def curve_diagnostics(observed, predicted):
    """Score pairs already aligned in time and concentration units.

    AFE = exp(mean(log(P/O))); values below/above one mean under/overprediction.
    GMFE = exp(mean(abs(log(P/O)))). Invalid pairs invalidate the whole case;
    they are never silently dropped or replaced by zero. No interpolation occurs.
    """
    result = {
        "status": "unassessable", "reason": "missing_or_invalid_curve_pairs",
        "point_count": None, "afe": None, "gmfe": None, "log_rmse": None,
        "within_twofold_count": None, "within_twofold_fraction": None,
    }
    if any(isinstance(v, (str, bytes, Mapping)) for v in (observed, predicted)):
        return result
    try:
        observed, predicted = list(observed), list(predicted)
    except TypeError:
        return result
    if not observed or len(observed) != len(predicted):
        return result
    pairs = [(_number(o, positive=True), _number(p, positive=True))
             for o, p in zip(observed, predicted)]
    if any(o is None or p is None for o, p in pairs):
        return result
    # Log differences avoid intermediate P/O overflow for extreme finite inputs.
    errors = [math.log(p) - math.log(o) for o, p in pairs]
    count = len(errors)
    afe = _exp(math.fsum(errors) / count)
    gmfe = _exp(math.fsum(abs(e) for e in errors) / count)
    if afe is None or gmfe is None:
        result["reason"] = "curve_error_outside_finite_numeric_range"
        return result
    covered = sum(_within(abs(e), math.log(RESEARCH_FOLD_LIMIT)) for e in errors)
    result.update(status="completed", reason=None, point_count=count, afe=afe,
                  gmfe=gmfe, log_rmse=math.sqrt(math.fsum(e * e for e in errors) / count),
                  within_twofold_count=covered, within_twofold_fraction=covered / count)
    return result


def _endpoint_comparison(data, observed_key, predicted_key):
    data = data if isinstance(data, Mapping) else {}
    observed = _number(data.get(observed_key), positive=True)
    predicted = _number(data.get(predicted_key), positive=True)
    reason = None
    value = None
    if observed is None or predicted is None:
        reason = "missing_or_invalid_positive_prediction_or_reference"
    else:
        value = _exp(abs(math.log(predicted) - math.log(observed)))
        if value is None:
            reason = "fold_error_outside_finite_numeric_range"
        if "fold_error" in data:
            reported = _number(data["fold_error"], positive=True)
            if (reported is None or reported < 1 or value is None
                    or not math.isclose(reported, value, rel_tol=1e-9, abs_tol=1e-12)):
                value, reason = None, "invalid_or_inconsistent_reported_fold_error"
    result = _assessment(value, "fold_error", reason)
    result.update(observed=observed, predicted=predicted, definition=data.get("definition"))
    return result


def assess_prediction_quality(metrics, *, tmax_tolerance_h=None):
    """Assess the complete output of ``evidence_evaluation.score_case``.

    Return separate ``curve`` and ``endpoints`` assessments. Status values are
    ``pass``, ``fail``, or ``unassessable``. Missing scalar references or curve
    pairs cannot pass. Tmax gets no threshold judgment unless the caller supplies
    a predeclared nonnegative tolerance in hours. Unsupported reference endpoints
    are explicit, rather than borrowed from predictions or clinical table means.
    """
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be the score_case mapping")
    if tmax_tolerance_h is not None:
        tmax_tolerance_h = _number(tmax_tolerance_h)
        if tmax_tolerance_h is None:
            raise ValueError("tmax_tolerance_h must be finite and nonnegative")

    diagnostics = curve_diagnostics(metrics.get("observed_umol_l"), metrics.get("predicted_umol_l"))
    curve_value = diagnostics["gmfe"]
    curve_reason = diagnostics["reason"]
    if curve_value is not None and "gmfe" in metrics:
        reported = _number(metrics["gmfe"], positive=True)
        if (reported is None or reported < 1
                or not math.isclose(reported, curve_value, rel_tol=1e-9, abs_tol=1e-12)):
            curve_value, curve_reason = None, "invalid_or_inconsistent_reported_gmfe"
    if curve_value is not None and "point_count" in metrics:
        count = metrics["point_count"]
        if isinstance(count, bool) or count != diagnostics["point_count"]:
            curve_value, curve_reason = None, "inconsistent_point_count"
    curve = _assessment(curve_value, "gmfe", curve_reason)
    curve["diagnostics"] = diagnostics
    curve["observed_window_h"] = metrics.get("observed_window_h")

    endpoints = {}
    for name, unit_suffix in (("auc", "umol_h_l"), ("sampled_cmax", "umol_l"), ("c_last", "umol_l")):
        endpoints[name] = _endpoint_comparison(metrics.get(name), "observed_" + unit_suffix,
                                               "predicted_" + unit_suffix)

    half = metrics.get("terminal_half_life")
    half = half if isinstance(half, Mapping) else {}
    half_o, half_p = half.get("observed"), half.get("predicted")
    if (half.get("status") == "completed" and isinstance(half_o, Mapping)
            and isinstance(half_p, Mapping)
            and all(v.get("status", "completed") == "completed" for v in (half_o, half_p))):
        comparison = dict(half, observed_h=half_o.get("value_h"), predicted_h=half_p.get("value_h"))
        endpoints["terminal_half_life"] = _endpoint_comparison(comparison, "observed_h", "predicted_h")
    else:
        endpoints["terminal_half_life"] = _assessment(None, "fold_error", "terminal_half_life_not_estimable")

    tmax = metrics.get("sampled_tmax")
    tmax = tmax if isinstance(tmax, Mapping) else {}
    observed, predicted = _number(tmax.get("observed_h")), _number(tmax.get("predicted_h"))
    error = abs(predicted - observed) if observed is not None and predicted is not None else None
    reason = "missing_or_invalid_prediction_or_reference" if error is None else None
    if error is not None and "absolute_error_h" in tmax:
        reported = _number(tmax["absolute_error_h"])
        if reported is None or not math.isclose(error, reported, rel_tol=1e-9, abs_tol=1e-12):
            error, reason = None, "invalid_or_inconsistent_reported_absolute_error"
    if error is not None and tmax_tolerance_h is None:
        reason = "no_predeclared_time_tolerance"
    status = ("unassessable" if error is None or tmax_tolerance_h is None else
              "pass" if _within(error, tmax_tolerance_h) else "fail")
    endpoints["sampled_tmax"] = {
        "metric": "absolute_error_h", "value": error, "observed": observed,
        "predicted": predicted, "definition": tmax.get("definition"), "reason": reason,
        "research_screening": {"threshold": tmax_tolerance_h, "status": status},
        "optimization_target": {"threshold": tmax_tolerance_h, "status": status},
    }
    for name in ("auc_inf", "clearance", "vz", "mrt", "vss"):
        endpoints[name] = _assessment(None, "fold_error", "qualified_reference_not_supplied_by_score_case")
    return {
        "criteria_scope": "project_research_thresholds_not_regulatory_acceptance",
        "clinical_readiness": "not_assessed",
        "validation_independence": "not_determined_by_accuracy_metrics",
        "curve": curve, "endpoints": endpoints,
    }
