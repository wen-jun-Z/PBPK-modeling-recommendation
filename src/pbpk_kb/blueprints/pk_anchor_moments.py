"""Converged model integrals for PK anchor construction, not clinical NCA.

The caller supplies a native, single-dose IV plasma simulation. No clinical
observations are accepted or selected here. Tail regression bounds negligible
remaining model area; it does not establish or report a clinical half-life.
The returned Vss is a plasma-moment NCA equivalent, not anatomical body volume
or an independently identified distribution volume under peripheral elimination.

Equations: https://pmc.ncbi.nlm.nih.gov/articles/PMC11429322/
Vss limitation: https://pubmed.ncbi.nlm.nih.gov/15124219/
All convergence thresholds below are declared numerical project gates, not
regulatory standards or proof that an eventual clinical prediction is accurate.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real


MODEL_INTEGRAL_POLICY = {
    "version": "model-anchor-moments-1",
    "relative_floor_of_peak": 1e-8,
    "absolute_floor_umol_per_l": 1e-12,
    "tail_window_fraction": 1 / 3,
    "minimum_tail_points": 4,
    "minimum_adjacent_half_points": 3,
    "minimum_adjusted_r_squared": .98,
    "maximum_adjacent_slope_relative_difference": .10,
    "maximum_auc_remainder_fraction": .005,
    "maximum_aumc_remainder_fraction": .01,
    "maximum_final_half_auc_fraction": .005,
    "maximum_final_half_aumc_fraction": .01,
    "minimum_span_half_lives": None,
}


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(f"{name} must be finite and " + ("positive" if positive else "nonnegative"))
    return value


def _area(times, values):
    return math.fsum((right - left) * (before / 2 + after / 2)
                     for left, right, before, after in zip(times, times[1:], values, values[1:]))


def _regression(times, concentrations):
    if len(times) < 3 or any(value <= 0 for value in concentrations):
        raise ValueError("Insufficient positive original tail samples")
    if any(second >= first for first, second in zip(concentrations, concentrations[1:])):
        raise ValueError("Model tail is not strictly declining")
    logs = [math.log(value) for value in concentrations]
    mean_time = math.fsum(times) / len(times)
    mean_log = math.fsum(logs) / len(logs)
    xx = math.fsum((time - mean_time) ** 2 for time in times)
    yy = math.fsum((value - mean_log) ** 2 for value in logs)
    if xx <= 0 or yy <= 0:
        raise ValueError("Model tail lacks a measurable negative slope")
    slope = math.fsum((time - mean_time) * (value - mean_log)
                      for time, value in zip(times, logs)) / xx
    residual = math.fsum((value - mean_log - slope * (time - mean_time)) ** 2
                         for time, value in zip(times, logs))
    adjusted_r2 = 1 - residual / yy * (len(times) - 1) / (len(times) - 2)
    if slope >= 0 or not all(math.isfinite(value) for value in (slope, adjusted_r2)):
        raise ValueError("Model tail slope is not finite and negative")
    return slope, adjusted_r2


def _final_half_area(times, values):
    midpoint = times[-1] / 2
    index = next(i for i, time in enumerate(times) if time >= midpoint)
    if times[index] == midpoint:
        return _area(times[index:], values[index:])
    left = index - 1
    ratio = (midpoint - times[left]) / (times[index] - times[left])
    boundary = values[left] + ratio * (values[index] - values[left])
    return _area([midpoint, *times[index:]], [boundary, *values[index:]])


def model_integral_pk(curve: dict, request: dict, *, require_vss: bool = True) -> dict:
    """Estimate CL and NCA-equivalent Vss only after model-integral gates pass.

    ``curve`` uses native ``time_min``, ``concentration``, ``time_unit='min'``
    and ``unit='umol/l'``. ``request`` supplies the existing evidence-builder
    dosing and molecular-weight evidence. The builder's contract is one IV
    dose at t=0 without pre-existing drug, under linear time-invariant kinetics.
    The caller must establish native simulation provenance and this contract;
    this function cannot distinguish an unlabelled observed series by numbers.
    A declaration that the curve is observed is rejected explicitly.

    All returned areas include a small conditional exponential remainder.
    By default, estimates that fail any required CL/Vss gate remain unavailable.
    With ``require_vss=False``, independently converged AUC can qualify CL while
    an unconverged AUMC leaves Vss and MRT unavailable. The Vss/AUMC thresholds
    are unchanged, as are clinical NCA and its full-half-life span requirement.
    """
    diagnostics = {"policy": deepcopy(MODEL_INTEGRAL_POLICY), "observations_used": False,
        "method": "linear_trapezoid_model_integrals_with_small_exponential_remainder",
        "tail_selection": "one_fixed_final_third_of_post_administration_post_peak_prefix",
        "regression_candidate_count": 1, "half_life_reported": False,
        "vss_definition": "plasma_moment_nca_equivalent",
        "integration_grid_convergence_assessed": False,
        "solver_error_bound_proven": False,
        "assumptions": ["native model plasma trajectory supplied by caller",
            "single IV dose at zero with no pre-existing drug; linear time-invariant kinetics",
            "terminal log-linear continuation is used only to bound negligible remaining moments",
            "concentration floor is a numerical heuristic, not assay LLOQ or a certified solver bound",
            "moment-based Vss need not equal anatomical Vss with peripheral elimination"],
        "numerical_convergence_is_not_clinical_validation": True,
        "required_endpoints": ["clearance", "vss"] if require_vss else ["clearance"],
        "endpoint_gate_policy": "CL requires AUC gates; Vss and MRT additionally require AUMC gates"}
    try:
        if not isinstance(require_vss, bool):
            raise ValueError("require_vss must be boolean")
        if not isinstance(curve, dict) or not isinstance(request, dict):
            raise ValueError("A native model curve and evidence-builder request are required")
        if curve.get("source") in {"observed", "clinical", "clinical_observation"} or any(
                key in curve for key in ("observations", "observed_values", "reference_values")):
            raise ValueError("Observed or reference curves are not accepted")
        dosing = request["dosing"]
        if dosing.get("route") != "intravenous" or dosing.get("single_dose", True) is not True:
            raise ValueError("Model integral PK requires a single intravenous dose")
        if _number(dosing.get("start_h", 0), "start_h") != 0:
            raise ValueError("Model integral PK requires dosing at time zero")
        if request.get("linear_kinetics", True) is not True:
            raise ValueError("Linear time-invariant kinetics are required for PK anchors")
        infusion_h = _number(dosing.get("infusion_minutes", 0), "infusion_minutes") / 60
        dose = _number(dosing.get("dose_mg"), "dose_mg", positive=True)
        molecular_weights = [row["value"] for row in request["evidence"] if row["parameter"] == "molecular_weight"]
        if len(molecular_weights) != 1:
            raise ValueError("Exactly one molecular-weight evidence value is required")
        molecular_weight = _number(molecular_weights[0], "molecular_weight", positive=True)
        if curve.get("time_unit") != "min" or str(curve.get("unit")).replace("µ", "u").replace("μ", "u") != "umol/l":
            raise ValueError("Native units must be min and umol/l")
        raw_times, raw_values = curve["time_min"], curve["concentration"]
        if (not isinstance(raw_times, list) or not isinstance(raw_values, list)
                or len(raw_times) != len(raw_values) or len(raw_times) < 8):
            raise ValueError("At least eight matching original model samples are required")
        times = [_number(value, "time_min") / 60 for value in raw_times]
        molar = [_number(value, "concentration") for value in raw_values]
        if times[0] != 0 or any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("Model times must start at zero and increase strictly")
        if infusion_h and (molar[0] != 0 or not any(0 < time < infusion_h for time in times)):
            raise ValueError("An infusion needs zero predose concentration and original within-infusion samples")
        peak_index = max(range(len(molar)), key=molar.__getitem__)
        peak = _number(molar[peak_index], "peak concentration", positive=True)
        floor = max(MODEL_INTEGRAL_POLICY["absolute_floor_umol_per_l"],
                    peak * MODEL_INTEGRAL_POLICY["relative_floor_of_peak"])
        first_low = next((i for i in range(peak_index + 1, len(molar)) if molar[i] < floor), None)
        rebound = first_low is not None and any(value >= floor for value in molar[first_low:])
        diagnostics.update(original_curve_points=len(times), original_end_h=times[-1],
                           numeric_floor_umol_per_l=floor, below_floor_rebound_detected=rebound)
        if rebound:
            raise ValueError("A below-floor rebound prevents a trustworthy single model tail")
        last_index = first_low - 1 if first_low is not None else len(times) - 1
        prefix_t, prefix_c = times[:last_index + 1], molar[:last_index + 1]
        anchor = max(infusion_h, times[peak_index])
        if prefix_t[-1] <= anchor:
            raise ValueError("No usable post-administration model prefix")
        lower = anchor + (prefix_t[-1] - anchor) * (1 - MODEL_INTEGRAL_POLICY["tail_window_fraction"])
        tail = [(time, value) for time, value in zip(prefix_t, prefix_c) if time >= lower]
        diagnostics.update(prefix_end_h=prefix_t[-1], prefix_points=len(prefix_t),
                           dropped_numeric_suffix_points=len(times) - len(prefix_t),
                           tail_window_h=[lower, prefix_t[-1]], tail_points=len(tail))
        if len(tail) < MODEL_INTEGRAL_POLICY["minimum_tail_points"]:
            raise ValueError("Too few original points in the fixed model tail")
        slope, adjusted_r2 = _regression([row[0] for row in tail], [row[1] for row in tail])
        diagnostics.update(tail_slope_per_h=slope, adjusted_r_squared=adjusted_r2)
        if adjusted_r2 < MODEL_INTEGRAL_POLICY["minimum_adjusted_r_squared"]:
            raise ValueError("Model tail fails the adjusted-R-squared gate")
        midpoint = (lower + prefix_t[-1]) / 2
        halves = [[row for row in tail if lo <= row[0] <= hi]
                  for lo, hi in ((lower, midpoint), (midpoint, prefix_t[-1]))]
        if any(len(half) < MODEL_INTEGRAL_POLICY["minimum_adjacent_half_points"] for half in halves):
            raise ValueError("Too few original points to check adjacent tail slopes")
        slopes = [_regression([row[0] for row in half], [row[1] for row in half])[0] for half in halves]
        difference = abs(slopes[1] - slopes[0]) / max(abs(slopes[0]), abs(slopes[1]))
        diagnostics.update(adjacent_half_slopes_per_h=slopes, adjacent_slope_relative_difference=difference)
        if difference > MODEL_INTEGRAL_POLICY["maximum_adjacent_slope_relative_difference"]:
            raise ValueError("Adjacent model-tail slopes are insufficiently stable")
        mass = [value * molecular_weight / 1000 for value in prefix_c]
        auc = _area(prefix_t, mass)
        rate = -slope
        auc_tail = mass[-1] / rate
        auc_inf = auc + auc_tail
        if not math.isfinite(auc_inf) or auc_inf <= 0:
            raise ValueError("Model AUC is not finite and positive")
        suffix_t = times[last_index:]
        suffix_c = [value * molecular_weight / 1000 for value in molar[last_index:]]
        auc_fractions = {
            "auc_remainder_fraction": auc_tail / auc_inf,
            "final_half_auc_fraction": _final_half_area(prefix_t, mass) / auc_inf,
            "discarded_suffix_auc_fraction": _area(suffix_t, suffix_c) / auc_inf,
        }

        def check_fractions(fractions, moment_name):
            diagnostics.update(fractions)
            for name, fraction in fractions.items():
                limit_name = ("maximum_" + name if not name.startswith("discarded") else
                              "maximum_" + moment_name + "_remainder_fraction")
                if not math.isfinite(fraction) or fraction > MODEL_INTEGRAL_POLICY[limit_name]:
                    raise ValueError(f"{name} exceeds its declared model-integral convergence gate")

        check_fractions(auc_fractions, "auc")
        clearance = dose / auc_inf
        if not math.isfinite(clearance) or clearance <= 0:
            raise ValueError("Model CL is not finite and positive")
        diagnostics.update(auc_mg_h_l=auc_inf, auc_gates_passed=True, infusion_correction_h=infusion_h / 2,
                           integration_end_h=prefix_t[-1], unit_conversion="umol/l * molecular_weight_g_mol / 1000 = mg/l")
        vss = mrt = None
        vss_reason = None
        try:
            moment = [time * value for time, value in zip(prefix_t, mass)]
            aumc = _area(prefix_t, moment)
            aumc_tail = mass[-1] * (prefix_t[-1] / rate + 1 / rate ** 2)
            aumc_inf = aumc + aumc_tail
            if not math.isfinite(aumc_inf) or aumc_inf <= 0:
                raise ValueError("Model AUMC is not finite and positive")
            diagnostics["aumc_mg_h2_l"] = aumc_inf
            check_fractions({
                "aumc_remainder_fraction": aumc_tail / aumc_inf,
                "final_half_aumc_fraction": _final_half_area(prefix_t, moment) / aumc_inf,
                "discarded_suffix_aumc_fraction": _area(suffix_t, [time * value for time, value in zip(suffix_t, suffix_c)]) / aumc_inf,
            }, "aumc")
            candidate_mrt = aumc_inf / auc_inf - infusion_h / 2
            candidate_vss = clearance * candidate_mrt
            if not all(math.isfinite(value) and value > 0 for value in (candidate_mrt, candidate_vss)):
                raise ValueError("Infusion-corrected model MRT or moment volume is not finite and positive")
            mrt, vss = candidate_mrt, candidate_vss
            diagnostics["aumc_gates_passed"] = True
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            vss_reason = str(error)
            diagnostics.update(aumc_gates_passed=False, vss_unavailable_reason=vss_reason)
            if require_vss:
                raise ValueError(vss_reason) from error
        return {"status": "ok", "clearance_l_h": clearance, "vss_l": vss,
                "mrt_h": mrt, "diagnostics": diagnostics,
                "endpoint_statuses": {"clearance": "ok", "vss": "ok" if vss is not None else "not_estimable",
                                      "mrt": "ok" if mrt is not None else "not_estimable"},
                "vss_unavailable_reason": vss_reason}
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        return {"status": "not_estimable", "clearance_l_h": None, "vss_l": None,
                "reason": str(error), "diagnostics": diagnostics}
