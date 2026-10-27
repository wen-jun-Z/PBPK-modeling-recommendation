"""Conservative single-dose NCA summaries; these do not fit PBPK parameters.

Definitions follow Phoenix's NCA parameter formulas and computation rules:
https://onlinehelp.certara.com/phoenix/8.3/topics/ncaparamformulas.htm
https://onlinehelp.certara.com/phoenix/8.2/topics/ncacomprules.htm
The fit-quality thresholds are declared project gates, not universal standards.
"""
from __future__ import annotations

import math
from numbers import Real


TERMINAL_ENDPOINTS = frozenset({"terminal_half_life", "auc_inf", "clearance",
                              "apparent_clearance", "vz", "apparent_vz", "mrt", "vss"})
_FIT_FIELDS = {"start_h", "end_h", "min_points", "min_adjusted_r2",
               "min_span_half_lives", "max_auc_extrapolated_fraction",
               "max_aumc_extrapolated_fraction"}


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _unavailable(reason, diagnostics=None):
    return {"status": "not_estimable", "reason": reason, "diagnostics": diagnostics or {}}


def _trapz(times, values):
    return math.fsum((b - a) * (x / 2. + y / 2.)
                     for a, b, x, y in zip(times, times[1:], values, values[1:]))


def terminal_endpoints(*, selected, times_h, concentrations, window_h, window_values,
                       target, molecular_weight, context):
    """Return every requested NCA endpoint, including explicit unavailable results.

    Regression uses only original samples in a predeclared terminal window.
    It never searches for a favourable fit or inserts interpolated fit points.
    Extrapolated exposure requires a covered dose time, no residual drug, a
    single dose, a reliable declining tail, and small declared tail fractions.
    """
    wanted = [name for name in selected if name in TERMINAL_ENDPOINTS]
    if not wanted:
        return {}, None
    diagnostics = {"selection": "predeclared_time_window_original_samples_only",
                   "pbpk_parameters_fitted": False}
    try:
        if not isinstance(context, dict) or context.get("single_dose") is not True:
            raise ValueError("single_dose=true dosing context is required; repeated dosing is unsupported")
        route = context.get("route")
        if route not in {"intravenous", "oral"}:
            raise ValueError("route must be intravenous or oral")
        dose_time = _number(context.get("start_h"), "dosing start_h")
        infusion_h = _number(context.get("infusion_minutes", 0.), "infusion_minutes") / 60.
        if route == "oral" and infusion_h:
            raise ValueError("oral dosing cannot have an infusion duration")
        spec = target.get("terminal_fit")
        if not isinstance(spec, dict) or not {"start_h", "end_h"}.issubset(spec) or set(spec) - _FIT_FIELDS:
            raise ValueError("terminal_fit requires a predeclared start_h/end_h and supported quality gates")
        lower = _number(spec["start_h"], "terminal_fit start_h")
        upper = _number(spec["end_h"], "terminal_fit end_h")
        if lower < target["window_start_h"] or lower >= upper or upper != target["window_end_h"]:
            raise ValueError("terminal fit must lie within the endpoint window and end at its final time")
        if lower < dose_time or (infusion_h > 0 and lower <= dose_time + infusion_h):
            raise ValueError("terminal fit must begin after administration is complete")
        if route == "oral":
            absorption_end = _number(context.get("absorption_complete_h"), "independently justified absorption_complete_h")
            if absorption_end < dose_time or lower <= absorption_end:
                raise ValueError("oral terminal fit must start after independently justified absorption completion")
            diagnostics["oral_absorption_completion_is_declared_not_inferred_from_curve"] = True
        minimum = spec.get("min_points", 4)
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 3:
            raise ValueError("terminal_fit min_points must be an integer >=3")
        min_r2 = _number(spec.get("min_adjusted_r2", .98), "min_adjusted_r2")
        span_min = _number(spec.get("min_span_half_lives", 1.), "min_span_half_lives", positive=True)
        auc_limit = _number(spec.get("max_auc_extrapolated_fraction", .2), "max_auc_extrapolated_fraction", positive=True)
        aumc_limit = _number(spec.get("max_aumc_extrapolated_fraction", .2), "max_aumc_extrapolated_fraction", positive=True)
        if min_r2 > 1 or auc_limit >= 1 or aumc_limit >= 1:
            raise ValueError("quality fractions must be in their valid range; extrapolation limits must be <1")
        diagnostics["quality_gates"] = {"min_points": minimum, "min_adjusted_r2": min_r2,
            "min_span_half_lives": span_min, "max_auc_extrapolated_fraction": auc_limit,
            "max_aumc_extrapolated_fraction": aumc_limit}
        pairs = [(t, c) for t, c in zip(times_h, concentrations) if lower <= t <= upper]
        diagnostics.update({"requested_fit_start_h": lower, "requested_fit_end_h": upper,
                            "fit_points": len(pairs), "fit_times_h": [t for t, _ in pairs]})
        if len(pairs) < minimum:
            raise ValueError("insufficient original samples in the predeclared terminal window")
        if any(c <= 0 for _, c in pairs):
            raise ValueError("terminal fit contains a nonpositive concentration")
        if any(b >= a for (_, a), (_, b) in zip(pairs, pairs[1:])):
            raise ValueError("terminal concentrations must decrease strictly; rising or flat tails are not estimable")
        postdose = [(t, c) for t, c in zip(times_h, concentrations) if dose_time <= t <= upper]
        peak = max(c for _, c in postdose)
        peak_time = next(t for t, c in postdose if c == peak)
        if lower <= peak_time and peak_time > dose_time:
            raise ValueError("terminal fit includes or precedes the sampled postdose maximum")
        xs = [t for t, _ in pairs]
        ys = [math.log(c) for _, c in pairs]
        mean_x, mean_y = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
        ssx = math.fsum((x - mean_x) ** 2 for x in xs)
        ssy = math.fsum((y - mean_y) ** 2 for y in ys)
        slope = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / ssx
        if slope >= 0 or ssy <= 0:
            raise ValueError("no positive terminal decline rate can be estimated")
        rate = -slope
        residual = math.fsum((y - mean_y - slope * (x - mean_x)) ** 2 for x, y in zip(xs, ys))
        r2 = max(0., min(1., 1. - residual / ssy))
        adjusted = 1. - (1. - r2) * (len(xs) - 1) / (len(xs) - 2)
        half_life = math.log(2.) / rate
        span = (xs[-1] - xs[0]) / half_life
        if not all(math.isfinite(x) for x in (rate, half_life, r2, adjusted, span)):
            raise ValueError("terminal regression exceeds finite numeric range")
        diagnostics.update({"lambda_z_per_h": rate, "r_squared": r2,
                            "adjusted_r_squared": adjusted, "span_half_lives": span})
        if adjusted < min_r2:
            raise ValueError("terminal fit fails the declared adjusted-R-squared gate")
        if span < span_min:
            raise ValueError("terminal fit covers too few half-lives under the declared span gate")
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        reason = str(error) or "terminal fit exceeds finite numeric range"
        return {name: _unavailable(reason, diagnostics) for name in wanted}, diagnostics

    out = {}
    if "terminal_half_life" in wanted:
        out["terminal_half_life"] = {"status": "completed", "value_h": half_life,
            "lambda_z_per_h": rate, "definition": "ln(2)/lambda_z from predeclared log-linear terminal samples",
            "units": {"value_h": "h", "lambda_z_per_h": "1/h"}, "diagnostics": diagnostics.copy()}
    other = [name for name in wanted if name != "terminal_half_life"]
    if not other:
        return out, diagnostics
    # Absence of the dose time is never repaired by a synthetic C(0) or by
    # renaming a partial AUC as total AUC. Existing finite-window outputs remain.
    try:
        if window_h[0] != dose_time or dose_time not in times_h:
            raise ValueError("total exposure requires the endpoint window to start at an original dose-time sample")
        if route == "oral" or infusion_h > 0:
            if window_values[0] != 0:
                raise ValueError("single-dose oral/infusion total exposure requires zero predose concentration")
        if any(c != 0 for t, c in zip(times_h, concentrations) if t < dose_time):
            raise ValueError("nonzero predose concentration is incompatible with single-dose total exposure")
        elapsed = [t - dose_time for t in window_h]
        auc_last = _trapz(elapsed, window_values)
        aumc_last = _trapz(elapsed, [t * c for t, c in zip(elapsed, window_values)])
        tail = window_values[-1] / rate
        auc_inf = auc_last + tail
        aumc_tail = window_values[-1] * (elapsed[-1] / rate + 1. / rate ** 2)
        aumc_inf = aumc_last + aumc_tail
        if not all(math.isfinite(x) and x > 0 for x in (auc_inf, aumc_inf)):
            raise ValueError("infinite-time exposure is not finite and positive")
        auc_fraction, aumc_fraction = tail / auc_inf, aumc_tail / aumc_inf
        mass_auc = auc_inf * (molecular_weight / 1000.)
        if not math.isfinite(mass_auc) or mass_auc <= 0:
            raise ValueError("mass-concentration exposure exceeds finite numeric range")
        diagnostics.update({"auc_extrapolated_fraction": auc_fraction,
                            "aumc_extrapolated_fraction": aumc_fraction,
                            "tail_definition": "last_simulated_concentration/lambda_z",
                            "auc_integration": "linear_trapezoid_from_dose_time",
                            "aumc_integration": "linear_trapezoid_of_elapsed_time_times_concentration"})
        if auc_fraction > auc_limit:
            raise ValueError("AUC tail fraction exceeds the declared extrapolation gate")
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        out.update({name: _unavailable(str(error), diagnostics) for name in other})
        return out, diagnostics

    for name in other:
        try:
            common = {"status": "completed", "auc_extrapolated_fraction": auc_fraction,
                      "source": "single_model_trajectory_nca", "diagnostics": diagnostics.copy()}
            if name == "auc_inf":
                out[name] = {**common, "value_umol_h_per_l": auc_inf, "value_mg_h_per_l": mass_auc,
                    "definition": "AUC_dose_to_last + C_last/lambda_z; terminal extrapolation",
                    "units": {"value_umol_h_per_l": "umol*h/l", "value_mg_h_per_l": "mg*h/l"}}
                continue
            if name in {"clearance", "vz", "vss"} and route != "intravenous":
                raise ValueError(f"{name} requires intravenous dosing; absolute bioavailability is unknown after oral dosing")
            if name in {"apparent_clearance", "apparent_vz"} and route != "oral":
                raise ValueError(f"{name} is reserved for oral apparent CL/F or Vz/F")
            if name in {"mrt", "vss"}:
                if aumc_fraction > aumc_limit:
                    raise ValueError("AUMC tail fraction exceeds the declared extrapolation gate")
                mrt = aumc_inf / auc_inf - infusion_h / 2.
                if not math.isfinite(mrt) or mrt <= 0:
                    raise ValueError("infusion-corrected MRT must be finite and positive")
                if name == "mrt":
                    out[name] = {**common, "value_h": mrt, "units": {"value_h": "h"},
                        "infusion_correction_h": infusion_h / 2.,
                        "definition": "AUMCinf/AUCinf - infusion_duration/2",
                        "includes_absorption_time": route == "oral",
                        "aumc_extrapolated_fraction": aumc_fraction}
                    continue
            dose = _number(context.get("dose_mg"), "dose_mg", positive=True)
            clearance = dose / mass_auc
            if name in {"clearance", "apparent_clearance"}:
                value, key, unit = clearance, "value_l_per_h", "l/h"
                definition = "dose/AUCinf" if route == "intravenous" else "CL/F = oral dose/AUCinf; F unknown"
            elif name in {"vz", "apparent_vz"}:
                value, key, unit = clearance / rate, "value_l", "l"
                definition = "CL/lambda_z" if route == "intravenous" else "Vz/F = (CL/F)/lambda_z; F unknown"
            else:
                value, key, unit = clearance * mrt, "value_l", "l"
                definition = "CL * (AUMCinf/AUCinf - infusion_duration/2); IV only"
            if not math.isfinite(value) or value <= 0:
                raise ValueError("derived PK endpoint exceeds finite positive numeric range")
            out[name] = {**common, key: value, "units": {key: unit}, "definition": definition}
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            out[name] = _unavailable(str(error), diagnostics.copy())
    return out, diagnostics
