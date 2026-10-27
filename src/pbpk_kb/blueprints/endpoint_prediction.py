"""Requested PK endpoints from one explicit model trajectory, without fitting.

AUC is limited to the requested covered interval. Cmax is the largest value
on the sampled trajectory (including interpolated window boundaries), not an
analytically established continuous-time maximum or a measured clinical value.
"""
from __future__ import annotations

from bisect import bisect_left
import math
from numbers import Real

from .terminal_pk import TERMINAL_ENDPOINTS, terminal_endpoints

SUPPORTED_ENDPOINTS = frozenset({"plasma_concentration_time", "auc", "cmax", "tmax", "c_last"}) | TERMINAL_ENDPOINTS
_TARGET_FIELDS = {"endpoints", "primary_endpoint", "window_start_h", "window_end_h"}


def _finite(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def predict_requested_endpoints(curve: dict, target: dict, molecular_weight: float, context: dict | None = None) -> dict:
    """Evaluate selected endpoints with boundary interpolation and no extrapolation.

    ``curve`` requires time_min, concentration, time_unit='min' and unit='umol/l'
    (micro-sign and Greek-mu spellings are also accepted). ``molecular_weight``
    is in g/mol. ``target`` requires endpoints, primary_endpoint,
    window_start_h and window_end_h, with optional predeclared terminal_fit.
    ``context`` declares single-dose route, dose and administration times for
    terminal NCA. Each selected endpoint is evaluated from
    the same clipped trajectory; the caller owns analyte/matrix binding and
    must pass the requested plasma output rather than another native observer.
    """
    mw = _finite(molecular_weight, "molecular_weight (g/mol)", positive=True)
    if (not isinstance(target, dict) or not _TARGET_FIELDS.issubset(target)
            or set(target) - (_TARGET_FIELDS | {"terminal_fit"})):
        raise ValueError("Target requires endpoints, primary_endpoint and both window boundaries in hours")
    selected = target["endpoints"]
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(x, str) or x not in SUPPORTED_ENDPOINTS for x in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Requested endpoints must be unique supported endpoint names")
    primary = target["primary_endpoint"]
    if not isinstance(primary, str) or primary not in selected:
        raise ValueError("The primary endpoint must be one of the selected endpoints")
    start = _finite(target["window_start_h"], "window_start_h")
    end = _finite(target["window_end_h"], "window_end_h")
    if end <= start:
        raise ValueError("The requested endpoint window must have positive duration")
    if not isinstance(curve, dict) or curve.get("time_unit") != "min":
        raise ValueError("Model curve must declare time_unit='min'")
    unit = curve.get("unit")
    if not isinstance(unit, str) or unit.replace("µ", "u").replace("μ", "u") != "umol/l":
        raise ValueError("Model curve must declare concentration in umol/l")
    times, values = curve.get("time_min"), curve.get("concentration")
    if (not isinstance(times, list) or not isinstance(values, list)
            or len(times) < 2 or len(times) != len(values)):
        raise ValueError("Model curve needs at least two matching time and concentration values")
    times = [_finite(x, "time_min") for x in times]
    values = [_finite(x, "concentration") for x in values]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Model curve times must be strictly increasing")
    start_min, end_min = start * 60., end * 60.
    if not math.isfinite(end_min) or start_min < times[0] or end_min > times[-1]:
        raise ValueError("Requested window must be fully covered by the curve; extrapolation is forbidden")

    def at(time):
        index = bisect_left(times, time)
        if index < len(times) and times[index] == time:
            return values[index]
        left, right = index - 1, index
        fraction = (time - times[left]) / (times[right] - times[left])
        return values[left] * (1. - fraction) + values[right] * fraction

    window_min = [start_min, *[t for t in times if start_min < t < end_min], end_min]
    window_values = [at(t) for t in window_min]
    window_h = [start, *[t / 60. for t in window_min[1:-1]], end]
    mass_values = [v * (mw / 1000.) for v in window_values]
    if any(not math.isfinite(v) for v in mass_values):
        raise ValueError("Mass-concentration conversion exceeds finite numeric range")
    endpoint_values = {}
    warnings = []
    if "plasma_concentration_time" in selected:
        endpoint_values["plasma_concentration_time"] = {
            "time_h": window_h, "concentration_umol_per_l": window_values,
            "concentration_mg_per_l": mass_values,
            "definition": "model plasma trajectory in the requested window, with linearly interpolated boundaries",
            "units": {"time_h": "h", "concentration_umol_per_l": "umol/l", "concentration_mg_per_l": "mg/l"},
        }
    if "auc" in selected:
        try:
            # Halving before addition avoids unnecessary overflow of x+y.
            molar_auc = math.fsum((b - a) * (x / 2. + y / 2.)
                                 for a, b, x, y in zip(window_h, window_h[1:], window_values, window_values[1:]))
        except OverflowError as error:
            raise ValueError("AUC exceeds finite numeric range") from error
        mass_auc = molar_auc * (mw / 1000.)
        if not math.isfinite(molar_auc) or not math.isfinite(mass_auc):
            raise ValueError("AUC exceeds finite numeric range")
        endpoint_values["auc"] = {
            "value_umol_h_per_l": molar_auc, "value_mg_h_per_l": mass_auc,
            "definition": "linear_trapezoid_in_requested_window; no tail extrapolation",
            "units": {"value_umol_h_per_l": "umol*h/l", "value_mg_h_per_l": "mg*h/l"},
        }
    if "cmax" in selected or "tmax" in selected:
        maximum = max(window_values)
        peak_index = window_values.index(maximum)
        gaps = [b - a for a, b in zip(times, times[1:]) if b > start_min and a < end_min]
        cmax = {
            "value_umol_per_l": maximum, "value_mg_per_l": mass_values[peak_index],
            "tmax_h": window_h[peak_index], "definition": "simulated_grid_max_in_window",
            "tmax_tie_rule": "earliest maximum among sampled points and interpolated window boundaries",
            "source_grid_min_interval_h": min(gaps) / 60.,
            "source_grid_max_interval_h": max(gaps) / 60.,
            "units": {"value_umol_per_l": "umol/l", "value_mg_per_l": "mg/l", "tmax_h": "h",
                      "source_grid_min_interval_h": "h", "source_grid_max_interval_h": "h"},
        }
        if "cmax" in selected:
            endpoint_values["cmax"] = cmax
        if "tmax" in selected:
            endpoint_values["tmax"] = {
                "status": "completed", "value_h": window_h[peak_index],
                "definition": "simulation time of earliest sampled maximum in the requested window",
                "units": {"value_h": "h"},
                "source_grid_min_interval_h": min(gaps) / 60.,
                "source_grid_max_interval_h": max(gaps) / 60.,
            }
        warnings.append("Cmax and Tmax depend on the simulation output grid and interpolated window boundaries; an intersample peak may be missed. This is not an analytic continuous-time maximum.")
    if "c_last" in selected:
        endpoint_values["c_last"] = {
            "status": "completed", "value_umol_per_l": window_values[-1],
            "value_mg_per_l": mass_values[-1], "time_h": end,
            "definition": "concentration at the requested window end, not a steady-state trough or assay LLOQ-based last concentration",
            "units": {"value_umol_per_l": "umol/l", "value_mg_per_l": "mg/l", "time_h": "h"},
        }
    terminal_values, terminal_diagnostics = terminal_endpoints(
        selected=selected, times_h=[t / 60. for t in times], concentrations=values,
        window_h=window_h, window_values=window_values, target=target,
        molecular_weight=mw, context=context)
    endpoint_values.update(terminal_values)
    unavailable = [name for name, result in endpoint_values.items() if result.get("status") == "not_estimable"]
    if unavailable:
        warnings.append("Requested endpoints not estimable under the declared evidence and quality gates: " + ", ".join(unavailable))
    extrapolated = any(name != "terminal_half_life" and result.get("status") == "completed"
                       for name, result in terminal_values.items())
    return {"status": "completed_with_unavailable_endpoints" if unavailable else "completed", "primary_endpoint": primary,
            "window": {"start_h": start, "end_h": end}, "endpoints": endpoint_values,
            "warnings": warnings, "source_curve_points": len(times), "window_curve_points": len(window_min),
            "extrapolation_performed": extrapolated, "parameters_fitted": False,
            "terminal_log_linear_regression_performed": bool(terminal_diagnostics and "lambda_z_per_h" in terminal_diagnostics),
            "terminal_diagnostics": terminal_diagnostics,
            "endpoint_counts": {"requested": len(selected), "completed": len(selected) - len(unavailable),
                                "not_estimable": len(unavailable)},
            "interpretation": "model predictions, not observed clinical values or validation of model accuracy"}
