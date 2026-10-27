"""Pure construction decisions for declared endpoints and finite input scenarios.

``plan_design(request, fallback_evidence, ranges, *, endpoint_limits, purpose)``
selects evidence and describes at most nine model runs. ``assess_design(plan,
predictions)`` compares their requested outputs. Neither function reads files,
runs a solver, fits parameters, or receives clinical outcome labels. The caller
must verify evidence/range sources and run the frozen physiological flow-exchange
model. In particular, a changed fu requires matching native physiology, rather
than a substitution of the global partition multiplier.

Ranges are supplied scenario assumptions, including when informed by training
validation; they are not confidence intervals. Passing a declared change limit
is conditional computational support, not clinical validation or safety.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import math
import re

from .evidence_decision import decide

VERSION = "purpose-design-1"
PARAMETERS = ("fraction_unbound", "total_clearance")
PARAMETER_UNITS = {"fraction_unbound": "fraction", "total_clearance": "ml/min/kg"}
AXES = ["clearance_scale", "partition_scale", "bidirectional_exchange_scale"]
ENDPOINT_UNITS = {
    "plasma_concentration_time": "mg/l", "auc": "mg*h/l", "cmax": "mg/l",
    "tmax": "h", "c_last": "mg/l", "terminal_half_life": "h",
    "auc_inf": "mg*h/l", "clearance": "l/h", "vz": "l", "mrt": "h", "vss": "l",
}
NUMERICAL_POLICY = {
    "concentration_floor_mg_l": 1e-12,
    "floor_role": "diagnostic_only_not_assay_LLOQ_or_error_bound",
    "positive_values_clipped_or_dropped": False,
    "zero_versus_positive_fold": "unbounded",
    "zero_versus_zero_fold": 1.,
    "grid_policy": "identical_samples_no_interpolation",
}


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(name + " must be a finite " + ("positive " if positive else "") + "number")
    return float(value)


def _source(source):
    if (not isinstance(source, dict) or set(source) != {"path", "sha256", "location"}
            or any(not isinstance(source.get(key), str) or not source[key].strip()
                   for key in ("path", "location"))
            or not isinstance(source.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None):
        raise ValueError("A range needs a source path, SHA256 and location; verification belongs to the caller")


def _limits(endpoints, limits):
    if not isinstance(limits, dict) or set(limits) != set(endpoints):
        raise ValueError("endpoint_limits must explicitly cover exactly the requested endpoints")
    result = {}
    for name in endpoints:
        limit = limits[name]
        if not isinstance(limit, dict) or set(limit) != {"metric", "max_change"}:
            raise ValueError("Each endpoint limit needs metric and max_change")
        metric = limit["metric"]
        maximum = _number(limit["max_change"], name + ".max_change", positive=True)
        if metric not in {"fold", "absolute"}:
            raise ValueError("Unknown endpoint change metric")
        if name == "tmax" and metric != "absolute":
            raise ValueError("tmax requires absolute changes in h, including a zero baseline")
        if metric == "fold" and maximum < 1:
            raise ValueError("Fold limits must be at least one")
        result[name] = {"metric": metric, "max_change": maximum}
    return result


def plan_design(request: dict, fallback_evidence: list, ranges: list, *,
                endpoint_limits: dict, purpose: dict | None = None) -> dict:
    """Select intact evidence, then enumerate lower/nominal/upper combinations.

    Malformed ranges/limits raise ValueError. Missing/conflicting construction
    inputs retain the underlying policy's blocked decision and have no runnable
    scenarios. Predicted CL/fu without a supplied range remain explicitly
    unassessed. Measured and derived values cannot receive stress ranges here.
    """
    decision = decide(request, fallback_evidence, purpose=purpose, method="evidence_policy")
    legacy = decision.pop("calibration")
    decision["calibration"] = {
        "model_version": "linear-finite-exchange-model-1", "variant": "flow_exchange",
        "active_parameters": list(AXES), "performed": False,
        "fraction_unbound_role": "retain_selected_construction_input",
        "protect_net_renal": legacy["protect_net_renal"],
        "clearance_scale_scope": legacy["clearance_scale_scope"],
        "parameter_status": "not_calibrated_by_this_policy",
    }
    endpoints = decision["purpose"]["endpoints"]
    limits = _limits(endpoints, endpoint_limits)
    selected = {row["parameter"]: row for row in decision["selected_evidence"]}
    nominal = {name: float(selected[name]["value"]) for name in PARAMETERS if name in selected}
    if not isinstance(ranges, list):
        raise ValueError("ranges must be a list")
    validated = {}
    renal = float(selected.get("renal_clearance", {}).get("value", 0.))
    for row in ranges:
        if not isinstance(row, dict) or set(row) != {"parameter", "lower", "upper", "unit", "basis", "source"}:
            raise ValueError("Each range needs parameter, lower, upper, unit, basis and source")
        parameter = row["parameter"]
        if not isinstance(parameter, str) or parameter not in PARAMETERS:
            raise ValueError("Only predicted total_clearance and fraction_unbound can have ranges")
        if parameter in validated:
            raise ValueError("Duplicate parameter range: " + parameter)
        if parameter not in selected or selected[parameter]["basis"] != "predicted":
            raise ValueError("Ranges cannot replace unselected, measured or derived evidence: " + parameter)
        if row["unit"] != PARAMETER_UNITS[parameter]:
            raise ValueError("Noncanonical range unit for " + parameter)
        if row["basis"] not in {"stress_assumption", "training_validation"}:
            raise ValueError("Range basis must declare stress_assumption or training_validation")
        _source(row["source"])
        lower = _number(row["lower"], parameter + ".lower", positive=True)
        upper = _number(row["upper"], parameter + ".upper", positive=True)
        if not lower < upper or not lower <= nominal[parameter] <= upper:
            raise ValueError("A nondegenerate range must include its selected nominal value")
        if parameter == "fraction_unbound" and upper > 1:
            raise ValueError("fraction_unbound must be in (0, 1]")
        if parameter == "total_clearance" and lower <= renal:
            raise ValueError("Total clearance must exceed the fixed renal input: positive nonrenal clearance is required")
        validated[parameter] = {**deepcopy(row), "lower": lower, "upper": upper,
                                "nominal": nominal[parameter]}
    missing = [name for name in PARAMETERS
               if name in selected and selected[name]["basis"] == "predicted" and name not in validated]
    reasons = list(decision["reasons"])
    scope = selected.get("elimination_scope", {}).get("value")
    total = nominal.get("total_clearance", renal + float(selected.get("hepatic_clearance", {}).get("value", 0.)))
    if scope == "renal_only" or (total > 0 and total <= renal):
        reasons.append("flow_exchange_requires_positive_nonrenal_clearance")
    status = "blocked" if reasons else "ready"
    selected_request = deepcopy(request)
    selected_request["evidence"] = deepcopy(decision["selected_evidence"])
    if status == "ready":
        selected_request["target"]["endpoints"] = list(endpoints)
        if selected_request["target"]["primary_endpoint"] not in endpoints:
            selected_request["target"]["primary_endpoint"] = endpoints[0]
    scenarios = []
    if status == "ready":
        scenarios.append({"scenario_id": "baseline", "values": deepcopy(nominal)})
        varied = sorted(validated)
        levels = [sorted({validated[name]["lower"], nominal[name], validated[name]["upper"]}) for name in varied]
        for combination in product(*levels):
            values = {**nominal, **dict(zip(varied, combination))}
            if values != nominal:
                scenarios.append({"scenario_id": f"scenario_{len(scenarios):03d}", "values": values})
    assert len(scenarios) <= 9
    return {
        "version": VERSION, "status": status,
        "action": "simulate_scenarios" if status == "ready" else "needs_evidence",
        "selected_request": selected_request, "selected_evidence": deepcopy(decision["selected_evidence"]),
        "evidence_decision": decision, "calibration": deepcopy(decision["calibration"]),
        "requested_endpoints": list(endpoints), "endpoint_limits": limits,
        "endpoint_units": {name: ENDPOINT_UNITS.get(name) for name in endpoints},
        "ranges": [validated[name] for name in sorted(validated)],
        "ranged_parameters": sorted(validated), "missing_ranges": missing,
        "range_coverage": "incomplete" if missing else "declared_ranges" if validated else "not_assessed",
        "range_interpretation": "finite input scenarios; not confidence intervals or probability coverage",
        "scenarios": scenarios, "reasons": list(dict.fromkeys(reasons)),
        "numerical_policy": deepcopy(NUMERICAL_POLICY),
        "source_verification": "required_by_caller_not_performed_here",
        "clinical_validation": False,
        "limitations": ["conditional_on_selected_human_physiology_and_exchange_structure",
                        "finite_scenarios_do_not_bound_all_intermediate_or_outside_values",
                        "changed_fu_requires_matching_native_physiology",
                        "no_target_outcomes_used", "no_clinical_safety_claim"],
    }


def _curve(prediction):
    curve = prediction.get("curve")
    if not isinstance(curve, dict):
        raise ValueError("missing_curve")
    times, values = curve.get("time_h"), curve.get("concentration_mg_l")
    if not isinstance(times, list) or not isinstance(values, list) or len(times) < 2 or len(times) != len(values):
        raise ValueError("invalid_curve_vectors")
    times = [_number(value, "curve.time_h") for value in times]
    values = [_number(value, "curve.concentration_mg_l") for value in values]
    if times[0] < 0 or any(right <= left for left, right in zip(times, times[1:])) or any(value < 0 for value in values):
        raise ValueError("invalid_curve_time_or_concentration")
    return times, values


def _change(baseline, other, metric):
    if metric == "absolute":
        return abs(other - baseline)
    if baseline == other:
        return 1.
    if baseline == 0 or other == 0:
        return math.inf
    return max(baseline / other, other / baseline)


def _unit(value):
    return value.lower() if isinstance(value, str) else None


def assess_design(plan: dict, predictions: list) -> dict:
    """Assess all planned predictions; never drop a failure to obtain stability.

    ``predictions`` contains scenario_id/status, canonical scalar endpoints and
    a curve with time_h/concentration_mg_l lists. A failed prediction may omit
    endpoints/curve. Missing scenarios are recorded as failures. Unknown or
    duplicate identifiers are rejected. Infinite changes use null plus an
    explicit unbounded flag so the result is strict JSON serializable.
    """
    if not isinstance(plan, dict) or plan.get("version") != VERSION:
        raise ValueError("Expected a purpose-design plan")
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    names = plan["requested_endpoints"]
    limits = _limits(names, plan["endpoint_limits"])
    scenarios = {row["scenario_id"]: row for row in plan["scenarios"]}
    rows = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("scenario_id"), str):
            raise ValueError("Every prediction needs a scenario_id")
        sid = row["scenario_id"]
        if sid not in scenarios or sid in rows:
            raise ValueError("Unknown or duplicate prediction scenario: " + sid)
        if row.get("status") not in {"predicted", "failed"}:
            raise ValueError("Prediction status must be predicted or failed")
        rows[sid] = row
    failures, curves = {}, {}
    for sid in scenarios:
        row = rows.get(sid)
        if row is None or row["status"] == "failed":
            failures[sid] = "missing_prediction" if row is None else str(row.get("reason", "prediction_failed"))
            continue
        try:
            curves[sid] = _curve(row)
        except ValueError as error:
            failures[sid] = str(error)
    baseline_times = curves.get("baseline", (None, None))[0]
    if baseline_times is not None:
        for sid, (times, _) in curves.items():
            if times != baseline_times:
                failures[sid] = "curve_time_grid_differs_from_baseline"
    assessments = {}
    numeric = {}
    for sid, (_, values) in curves.items():
        numeric[sid] = {"zero_samples": sum(value == 0 for value in values),
                        "positive_below_floor_samples": sum(0 < value < NUMERICAL_POLICY["concentration_floor_mg_l"] for value in values)}
    for name in names:
        issues = dict(failures)
        values = {}
        for sid in scenarios:
            if sid in issues:
                continue
            if name == "plasma_concentration_time":
                values[sid] = curves[sid][1]
                continue
            reported = rows[sid].get("endpoints")
            endpoint = reported.get(name) if isinstance(reported, dict) else None
            try:
                if not isinstance(endpoint, dict) or endpoint.get("value") is None:
                    raise ValueError("requested_endpoint_not_estimable")
                if endpoint.get("status", "completed") not in {"completed", "predicted"}:
                    raise ValueError("requested_endpoint_not_estimable")
                if _unit(endpoint.get("unit")) != ENDPOINT_UNITS.get(name):
                    raise ValueError("requested_endpoint_noncanonical_unit")
                number = _number(endpoint["value"], name)
                if number < 0:
                    raise ValueError("negative_requested_endpoint")
                values[sid] = number
            except ValueError as error:
                issues[sid] = str(error)
        metric = limits[name]["metric"]
        comparisons = []
        if "baseline" in values:
            for sid in scenarios:
                if sid not in values or sid == "baseline":
                    continue
                paired = zip(values["baseline"], values[sid]) if name == "plasma_concentration_time" else [(values["baseline"], values[sid])]
                change = max(_change(a, b, metric) for a, b in paired)
                comparisons.append({"scenario_id": sid, "max_change": change if math.isfinite(change) else None,
                                    "unbounded_change": not math.isfinite(change),
                                    "exceeds_limit": change > limits[name]["max_change"]})
        sensitive = any(row["exceeds_limit"] for row in comparisons)
        if issues or plan["status"] != "ready":
            status = "unassessed"
        elif sensitive:
            status = "sensitive"
        elif plan["missing_ranges"]:
            status = "unassessed"
        elif not plan["ranges"]:
            status = "not_assessed"
        else:
            status = "within_declared_limits"
        envelope = None
        if plan["ranges"] and not issues and values:
            if name == "plasma_concentration_time":
                envelope = {"time_h": list(baseline_times),
                            "lower": [min(point) for point in zip(*values.values())],
                            "upper": [max(point) for point in zip(*values.values())]}
            else:
                envelope = {"lower": min(values.values()), "upper": max(values.values())}
        unbounded = any(row["unbounded_change"] for row in comparisons)
        maximum = (None if unbounded or not comparisons else max(row["max_change"] for row in comparisons))
        assessments[name] = {
            "status": status, "metric": metric, "limit": limits[name]["max_change"],
            "unit": ENDPOINT_UNITS.get(name), "baseline": values.get("baseline"),
            "scenario_range": envelope, "max_change": maximum,
            "unbounded_change": unbounded, "observed_sensitivity": sensitive,
            "comparisons": comparisons, "unassessed_scenarios": issues,
            "missing_ranges": list(plan["missing_ranges"]),
            "range_interpretation": plan["range_interpretation"],
        }
    if plan["status"] != "ready" or failures or any(row["status"] in {"unassessed", "sensitive"} for row in assessments.values()):
        status, action = "needs_evidence", "obtain_evidence"
    elif not plan["ranges"]:
        status, action = "not_assessed", "retain_inputs_without_range_assessment"
    else:
        status, action = "supported_under_assumptions", "proceed_with_declared_assumptions"
    needed = set(plan["missing_ranges"])
    for assessment in assessments.values():
        for comparison in assessment["comparisons"]:
            if comparison["exceeds_limit"]:
                sid = comparison["scenario_id"]
                needed.update(name for name in plan["ranged_parameters"]
                              if scenarios[sid]["values"][name] != scenarios["baseline"]["values"][name])
    return {"version": VERSION, "status": status, "action": action,
            "requested_endpoints": list(names), "endpoint_assessments": assessments,
            "missing_ranges": list(plan["missing_ranges"]), "evidence_priorities": sorted(needed),
            "failed_scenarios": failures, "reasons": list(plan["reasons"]),
            "range_coverage": plan["range_coverage"], "range_interpretation": plan["range_interpretation"],
            "scenario_count": len(scenarios), "prediction_count": len(rows),
            "numerical_policy": deepcopy(NUMERICAL_POLICY), "curve_numeric_diagnostics": numeric,
            "clinical_validation": False, "limitations": list(plan["limitations"])}
