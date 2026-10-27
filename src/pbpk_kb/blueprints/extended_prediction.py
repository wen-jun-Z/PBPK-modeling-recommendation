"""Model-only terminal summaries, separate from observed-window evaluation.

This module never receives observations and never changes model parameters.
It uses one deterministic model prefix and one terminal window; it does not
search for a favourable regression. Original curves are neither edited nor
resampled. The numeric floor is a declared heuristic, not an assay LLOQ or a
proved bound on the native solver's error.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .endpoint_prediction import predict_requested_endpoints
from .terminal_pk import TERMINAL_ENDPOINTS


IV_ENDPOINTS = (
    "plasma_concentration_time", "auc", "cmax", "tmax", "c_last",
    "terminal_half_life", "auc_inf", "clearance", "vz", "mrt", "vss",
)
MODEL_TAIL_POLICY = {
    "version": "deterministic-model-prefix-1",
    "absolute_floor_umol_per_l": 1e-12,
    "relative_floor_of_peak": 1e-8,
    "terminal_window_fraction": 1 / 3,
    "min_points": 4,
    "min_adjusted_r2": .98,
    "min_span_half_lives": 1.,
    "max_auc_extrapolated_fraction": .2,
    "max_aumc_extrapolated_fraction": .2,
    "maximum_horizon_h": 168.,
    "suggested_half_lives": 12.,
}


def _slope(times, values):
    if len(times) < 3 or any(value <= 0 for value in values):
        return None
    mx = math.fsum(times) / len(times)
    logs = [math.log(value) for value in values]
    my = math.fsum(logs) / len(logs)
    xx = math.fsum((value - mx) ** 2 for value in times)
    if xx <= 0:
        return None
    slope = math.fsum((x - mx) * (y - my) for x, y in zip(times, logs)) / xx
    return slope if math.isfinite(slope) else None


def summarize_extended_prediction(curve: dict, request: dict) -> dict:
    """Return all 11 IV endpoints and model-only diagnostics from a locked curve.

    ``curve`` has the same schema as ``predict_requested_endpoints``. ``request``
    supplies target window, dosing and molecular-weight evidence. Its original
    finite window is used for the five direct outputs. The full model curve can
    extend beyond that window. Terminal endpoints use the dose-time-to-prefix
    window and the final third of the post-administration/post-peak interval.

    Quality failures have ``status='not_estimable'`` and ``value=None``. The
    return's ``extension`` suggests a bounded numerical rerun, never a parameter
    change. Oral requests are intentionally rejected: absolute CL/Vz/Vss are
    not available from an oral plasma trajectory with unknown bioavailability.
    """
    dosing = deepcopy(request["dosing"])
    if dosing.get("route") != "intravenous":
        raise ValueError("All eleven absolute endpoints require single-dose intravenous input")
    if dosing.get("single_dose", True) is not True:
        raise ValueError("Repeated-dose terminal summaries are unsupported")
    context = {**dosing, "single_dose": True}
    mw = next(row["value"] for row in request["evidence"] if row["parameter"] == "molecular_weight")
    target = deepcopy(request["target"])
    target["endpoints"] = list(IV_ENDPOINTS[:5])
    if target["primary_endpoint"] not in target["endpoints"]:
        target["primary_endpoint"] = "plasma_concentration_time"
    target.pop("terminal_fit", None)
    # This validates all curve samples, units, ordering and finite-window
    # coverage, even when the terminal calculation later uses a prefix.
    direct = predict_requested_endpoints(curve, target, mw, context=context)
    times = [value / 60 for value in curve["time_min"]]
    concentrations = curve["concentration"]
    start = float(dosing.get("start_h", 0.))
    administration_end = start + float(dosing.get("infusion_minutes", 0.)) / 60.
    postdose = [i for i, time in enumerate(times) if time >= start]
    if not postdose:
        raise ValueError("Model curve contains no postdose samples")
    peak_index = max(postdose, key=lambda index: concentrations[index])
    peak = concentrations[peak_index]
    floor = max(MODEL_TAIL_POLICY["absolute_floor_umol_per_l"],
                peak * MODEL_TAIL_POLICY["relative_floor_of_peak"])
    first_low = next((i for i in range(peak_index + 1, len(times))
                      if concentrations[i] < floor), None)
    rebound = first_low is not None and any(value >= floor for value in concentrations[first_low:])
    # Only a final below-floor suffix can be removed. A later rebound prevents
    # prefix truncation, so enterohepatic/re-distribution shapes are not hidden.
    end_index = first_low - 1 if first_low is not None and not rebound else len(times) - 1
    prefix_end = times[end_index]
    anchor = max(administration_end, times[peak_index])
    lower = anchor + (prefix_end - anchor) * (1 - MODEL_TAIL_POLICY["terminal_window_fraction"])
    original_fit = request["target"].get("terminal_fit") or {}
    gates = {}
    for key in ("min_points", "min_adjusted_r2", "min_span_half_lives"):
        gates[key] = max(MODEL_TAIL_POLICY[key], original_fit.get(key, MODEL_TAIL_POLICY[key]))
    for key in ("max_auc_extrapolated_fraction", "max_aumc_extrapolated_fraction"):
        gates[key] = min(MODEL_TAIL_POLICY[key], original_fit.get(key, MODEL_TAIL_POLICY[key]))
    diagnostics = {
        "policy": deepcopy(MODEL_TAIL_POLICY), "quality_gates": gates,
        "selection_uses_observations": False, "regression_candidate_count": 1,
        "numeric_floor_umol_per_l": floor, "peak_umol_per_l": peak,
        "original_curve_points": len(times), "original_curve_end_h": times[-1],
        "prefix_end_h": prefix_end, "prefix_points": end_index + 1,
        "dropped_numeric_suffix_points": len(times) - end_index - 1,
        "below_floor_rebound_detected": rebound,
        "terminal_window_h": [lower, prefix_end],
        "main_window_h": [target["window_start_h"], target["window_end_h"]],
        "model_curve_edited": False, "interpolated_regression_points": False,
        "floor_interpretation": "fixed numerical heuristic; not assay LLOQ or certified solver tolerance",
    }
    terminal_names = list(IV_ENDPOINTS[5:])
    prefix = {**curve, "time_min": curve["time_min"][:end_index + 1],
              "concentration": concentrations[:end_index + 1]}
    terminal_target = {"endpoints": terminal_names, "primary_endpoint": "terminal_half_life",
        "window_start_h": start, "window_end_h": prefix_end,
        "terminal_fit": {"start_h": lower, "end_h": prefix_end, **gates}}
    try:
        if prefix_end <= anchor or end_index < 1:
            raise ValueError("No usable post-administration model prefix")
        terminal = predict_requested_endpoints(prefix, terminal_target, mw, context=context)
        terminal_values = terminal["endpoints"]
        diagnostics["terminal_fit"] = terminal["terminal_diagnostics"]
    except ValueError as error:
        terminal_values = {name: {"status": "not_estimable", "reason": str(error)} for name in terminal_names}
        diagnostics["terminal_fit"] = None

    # A stability diagnostic only: do not select windows from these slopes.
    midpoint = (lower + prefix_end) / 2.
    halves = [[(t, c) for t, c in zip(times[:end_index + 1], concentrations[:end_index + 1])
               if lo <= t <= hi] for lo, hi in ((lower, midpoint), (midpoint, prefix_end))]
    slopes = [_slope([t for t, _ in pairs], [c for _, c in pairs]) for pairs in halves]
    relative_change = (abs(slopes[1] - slopes[0]) / max(abs(slopes[0]), abs(slopes[1]))
                       if all(s is not None and s < 0 for s in slopes) else None)
    diagnostics["adjacent_half_window_slopes_per_h"] = slopes
    diagnostics["adjacent_slope_relative_difference"] = relative_change
    endpoints = direct["endpoints"]
    endpoints.update(terminal_values)
    for name, endpoint in endpoints.items():
        endpoint.setdefault("status", "completed")
        if name in TERMINAL_ENDPOINTS:
            endpoint["model_prefix_end_h"] = prefix_end
            if endpoint["status"] == "not_estimable":
                endpoint["value"] = None
    unavailable = [name for name, endpoint in endpoints.items() if endpoint["status"] != "completed"]
    cap = MODEL_TAIL_POLICY["maximum_horizon_h"]
    should_extend = bool(unavailable and times[-1] < cap and end_index == len(times) - 1)
    half = endpoints["terminal_half_life"].get("value_h")
    next_horizon = None
    if should_extend:
        suggestion = max(2 * times[-1], administration_end + MODEL_TAIL_POLICY["suggested_half_lives"] * half
                         if half else 48.)
        next_horizon = float(min(cap, math.ceil(suggestion)))
    return {
        "status": "completed_with_unavailable_endpoints" if unavailable else "completed",
        "endpoints": endpoints, "endpoint_counts": {"requested": len(IV_ENDPOINTS),
            "completed": len(IV_ENDPOINTS) - len(unavailable), "not_estimable": len(unavailable)},
        "main_window": direct["window"], "diagnostics": diagnostics,
        "extension": {"recommended": should_extend, "suggested_end_h": next_horizon,
            "maximum_end_h": cap, "unavailable_endpoints": unavailable,
            "reason": "quality-gated endpoint incomplete on full positive model horizon" if should_extend
                else "complete, horizon cap reached, or numeric-floor prefix already exhausted"},
        "warnings": direct["warnings"] + (["Some terminal predictions remain unavailable; no quality gate was relaxed."] if unavailable else []),
        "observations_used": False, "parameters_fitted": False,
        "interpretation": "Model predictions only; clinical reference values and errors must be evaluated separately.",
    }
