"""Engineering feasibility profiles for an optional RBC distribution axis.

This is a fixed-tolerance, finite-grid diagnostic on visible development data.
It is neither a likelihood confidence interval nor a statistical identifiability
test. A bounded profile does not establish a biological mechanism or clinical
confidence. Target observations and fitted warm starts are not accepted.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import least_squares


VERSION = "selective-rbc-engineering-profile-1"
SEED = 20261008
MINIMUM_MAIN_OBSERVATIONS = 8


class ProfileBudgetExhausted(RuntimeError):
    """An assigned start or the complete profile exhausted its call budget."""


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _quotas(total, count):
    return [total // count + int(i < total % count) for i in range(count)]


def _bands(points, accepted):
    segments = []
    for index, yes in enumerate(accepted):
        if yes and (index == 0 or not accepted[index - 1]):
            segments.append([float(points[index]), float(points[index])])
        elif yes:
            segments[-1][1] = float(points[index])
    return segments


def _assessment(profile, visible_points, tolerance):
    """Describe finite-grid feasibility; unsupported diagnoses remain closed."""
    all_success = bool(profile) and all(row["successful"] for row in profile)
    successful = [row for row in profile if row["successful"]]
    minimum = min((row["mse"] for row in successful), default=None)
    threshold = None if minimum is None else minimum + tolerance ** 2
    null = next((row for row in profile if row["log_rbc_scale"] == 0.), None)
    null_mse = null["mse"] if null is not None and null["successful"] else None
    accepted = [row["successful"] and row["mse"] <= threshold for row in profile] if threshold is not None else [False] * len(profile)
    intervals = _bands([row["log_rbc_scale"] for row in profile], accepted)
    touches = bool(accepted) and (accepted[0] or accepted[-1])
    outside = null_mse is not None and threshold is not None and null_mse > threshold
    enough = visible_points >= MINIMUM_MAIN_OBSERVATIONS
    raw = all_success and outside and len(intervals) == 1 and not touches
    reasons = []
    if not enough:
        reasons.append("fewer_than_eight_visible_observations")
    if not all_success:
        reasons.append("one_or_more_profile_points_failed")
    if null_mse is None:
        reasons.append("null_profile_failed")
    elif not outside:
        reasons.append("null_within_engineering_tolerance")
    if len(intervals) != 1:
        reasons.append("accepted_band_not_one_connected_grid_component")
    if touches:
        reasons.append("accepted_band_touches_numerical_boundary")
    main = enough and raw
    necessity = enough and all_success and outside
    necessity_reasons = [r for r in reasons if r not in {
        "accepted_band_not_one_connected_grid_component", "accepted_band_touches_numerical_boundary"}]
    return {"minimum_profile_mse": minimum, "threshold": threshold, "null_mse": null_mse,
            "null_outside": outside, "all_profile_points_successful": all_success,
            "accepted_grid_mask": accepted, "accepted_intervals_log": intervals,
            "connected_components": len(intervals), "touches_boundary": touches,
            "raw_gate_without_observation_count": raw,
            "raw_five_point_gate": raw if visible_points == 5 else None,
            "decision": {"selected_model": "blood_partition" if main else "direct_relaxed", "accepted": main, "reasons": reasons},
            "necessity_only_decision": {"selected_model": "blood_partition" if necessity else "direct_relaxed",
                                        "accepted": necessity, "reasons": necessity_reasons}}


def decision_for(profile, *, protected=False):
    """Retain a protected/measured RBC input regardless of numerical proposals."""
    if not isinstance(protected, bool):
        raise ValueError("protected must be a boolean")
    if protected:
        return {"selected_model": "direct_relaxed", "accepted": False,
                "reasons": ["protected_blood_partition_input"], "protected": True}
    return deepcopy(profile["decision"]) | {"protected": False}


def analyze(predict, times, observed, lower4, upper4, *, log_tolerance=math.log(1.1),
            grid_points=25, max_calls=32768, before_trial=None):
    """Profile log RBC scale with CL/K/PS as data-only nuisance coordinates.

    ``times`` are hours and ``observed`` share the predictor's mg/L units. Bounds
    are four optimization coordinates (log CL multiplier, log K, log PS, log RBC
    scale), not physical values. Every attempted model call consumes budget,
    including finite differences and failed evaluations. No continuation or
    externally fitted parameters are used.
    """
    if not callable(predict) or (before_trial is not None and not callable(before_trial)):
        raise ValueError("Predictor and optional guard must be callable")
    grid_points = _integer(grid_points, "grid_points", 3)
    max_calls = _integer(max_calls, "max_calls")
    if (isinstance(log_tolerance, bool) or not isinstance(log_tolerance, Real)
            or not math.isfinite(log_tolerance) or log_tolerance <= 0):
        raise ValueError("log_tolerance must be finite and positive")
    tolerance = float(log_tolerance)
    t = np.asarray(times, dtype=float).copy()
    y = np.asarray(observed, dtype=float).copy()
    lower = np.asarray(lower4, dtype=float).copy()
    upper = np.asarray(upper4, dtype=float).copy()
    if (t.ndim != 1 or len(t) < 4 or y.shape != t.shape or not np.isfinite(t).all()
            or np.any(t < 0) or np.any(np.diff(t) <= 0)
            or not np.isfinite(y).all() or np.any(y <= 0)):
        raise ValueError("Require at least four ordered visible hours and positive finite mg/L observations")
    if (lower.shape != (4,) or upper.shape != (4,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower >= upper)
            or not lower[3] <= 0. <= upper[3]):
        raise ValueError("Four finite ordered log-coordinate bounds must include RBC null zero")
    zeros = np.clip(np.zeros(4), lower, upper)
    nuisance_starts = [zeros[:3], np.random.default_rng(SEED).uniform(lower[:3], upper[:3])]
    calls = failures = 0
    failure_reasons = Counter()

    def fit_at(initial, lo, hi, quota, fixed_rbc=None):
        nonlocal calls, failures
        used = local_failures = 0
        best = None
        if quota == 0:
            return {"successful": False, "status": "no_call_quota", "calls": 0, "failures": 0,
                    "call_limit": 0, "best": None, "initial_parameters": np.asarray(initial).tolist()}

        def residual(z):
            nonlocal used, local_failures, calls, failures, best
            if used >= quota or calls >= max_calls:
                raise ProfileBudgetExhausted()
            if before_trial is not None:
                before_trial()
            used += 1
            calls += 1
            x = np.asarray(z, dtype=float).copy() if fixed_rbc is None else np.r_[z, fixed_rbc]
            try:
                result = predict(x.copy(), t.copy())
                if not isinstance(result, Mapping) or "concentration_mg_l" not in result:
                    raise ValueError("Predictor must return concentration_mg_l")
                p = np.asarray(result["concentration_mg_l"], dtype=float)
                if p.shape != y.shape or not np.isfinite(p).all() or np.any(p <= 0):
                    raise ValueError("Predicted visible concentrations must be finite and positive")
                residuals = (np.log(p) - np.log(y)) / math.sqrt(len(y))
                mse = float(residuals @ residuals)
                if not math.isfinite(mse):
                    raise ValueError("Nonfinite development MSE")
                row = {"mse": mse, "log_parameters": x.tolist()}
                if best is None or mse < best["mse"]:
                    best = row
                return residuals
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                failures += 1
                local_failures += 1
                failure_reasons[type(exc).__name__ + ": " + str(exc)] += 1
                return np.full(len(y), 1e6)

        status = "converged"
        successful = False
        try:
            fitted = least_squares(residual, np.asarray(initial).copy(), bounds=(lo, hi),
                                   max_nfev=400, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            successful = bool(fitted.success) and best is not None
            if not fitted.success:
                status = "optimizer_limit"
        except ProfileBudgetExhausted:
            status = "call_budget_exhausted"
        if best is None:
            status = "no_feasible_evaluation"
        return {"successful": successful, "status": status, "calls": used, "failures": local_failures,
                "call_limit": quota, "best": best, "initial_parameters": np.asarray(initial).tolist()}

    # Reserve nearly all calls for complete profile coverage; this single
    # location fit contributes only an extra RBC grid coordinate, never a warm
    # start for the profile nuisance coordinates.
    full_quota = max_calls // (grid_points + 3)
    full_fit = fit_at(zeros, lower, upper, full_quota)
    grid = np.linspace(lower[3], upper[3], grid_points).tolist() + [0.]
    if full_fit["best"] is not None:
        grid.append(full_fit["best"]["log_parameters"][3])
    # Keep zero exact and coalesce only numerically identical grid coordinates.
    raw_grid = sorted(0. if abs(float(x)) <= 1e-14 else float(x) for x in grid)
    points = []
    for x in raw_grid:
        if not points or abs(x - points[-1]) > 1e-12:
            points.append(x)
        elif x == 0. or x == upper[3]:
            points[-1] = x
    point_quotas = _quotas(max_calls - calls, len(points))
    profile = []
    for log_rbc, quota in zip(points, point_quotas):
        runs = [fit_at(start, lower[:3], upper[:3], part, fixed_rbc=log_rbc)
                for start, part in zip(nuisance_starts, _quotas(quota, 2))]
        eligible = [run for run in runs if run["successful"]]
        best = min((run["best"] for run in eligible), key=lambda row: row["mse"], default=None)
        profile.append({"log_rbc_scale": log_rbc, "mse": None if best is None else best["mse"],
                        "successful": best is not None, "best": best, "starts": runs,
                        "calls": sum(run["calls"] for run in runs), "failures": sum(run["failures"] for run in runs),
                        "call_limit": quota})
    assessment = _assessment(profile, len(t), tolerance)
    return {"version": VERSION, "status": "profiled" if assessment["all_profile_points_successful"] else "incomplete",
            "visible_points": len(t), "profile": profile, "full_fit": full_fit,
            "calls": calls, "failures": failures, "max_calls": max_calls,
            "failure_reasons": dict(failure_reasons), "lower4": lower.tolist(), "upper4": upper.tolist(),
            "requested_grid_points": grid_points, "actual_grid_points": len(points),
            "log_tolerance": tolerance, "additive_mse_tolerance": tolerance ** 2,
            "units": {"times": "h", "observed": "mg/L", "parameters": "log multipliers"},
            "assumptions": ["Fixed 10% log-scale engineering tolerance by default, not a statistical confidence interval.",
                            "Finite grid and two nuisance starts cannot prove global identifiability.",
                            "No likelihood threshold, regularization, target observations, or external warm starts.",
                            "All visible observations reused only within this development profile.",
                            "Main gate requires eight observations; two per candidate parameter is an engineering rule.",
                            "Full four-axis fit supplies an extra grid location only; nuisance starts are always independent."],
            "target_observations_used": False, "external_warm_start_used": False, "neighbor_continuation_used": False,
            **assessment}
