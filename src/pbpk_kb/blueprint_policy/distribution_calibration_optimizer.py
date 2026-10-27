"""Bounded four-parameter development calibration for distribution hypotheses.

Only the explicit visible development observations enter the objective. The
injected predictor receives parameters and visible times, never observations.
The first three axes exactly retain the relaxed CL/K/PS initialization and
numerical bounds; a fourth RBC or exchange-shape axis is independently seeded.
These are regularized fits, not physiological confidence intervals.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from numbers import Integral, Real
import time

import numpy as np
from scipy.optimize import least_squares

from .exposure_profile import initial_starts as _shape_starts
from .perfusion_calibration import clearance_inputs


VERSION = "development-distribution-calibration-1"
VARIANTS = ("blood_partition", "exchange_shape")
BASE_SEED = 20260930
FOURTH_AXIS_SEED = 20261008


class CallBudgetExhausted(RuntimeError):
    """One start has spent its quota of actual predictor calls."""


def _positive(value, label, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError(label + " must be finite and " + ("nonnegative" if zero else "positive"))
    return float(value)


def _integer(value, label, *, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            or (maximum is not None and value > maximum)):
        raise ValueError(label + " must be a positive integer" + (f" no larger than {maximum}" if maximum else ""))
    return int(value)


def _visible(request, training):
    if not isinstance(request, dict) or not isinstance(training, dict):
        raise ValueError("Request and visible development data must be dictionaries")
    if training.get("compound_id") != request.get("compound_id") or not request.get("compound_id"):
        raise ValueError("Development compound must match the request")
    if any(training.get(k) != v for k, v in (("matrix", "plasma"), ("time_unit", "min"), ("unit", "umol/l"))):
        raise ValueError("Development observations must declare plasma, min and umol/l")
    if (training.get("human") != request.get("human")
            or training.get("dosing") != request.get("dosing")):
        raise ValueError("Development human and dosing conditions must match the request")
    dosing = request.get("dosing", {})
    if dosing.get("route") != "intravenous" or dosing.get("start_h", 0.) != 0.:
        raise ValueError("Calibration requires a single intravenous dose starting at zero")
    _positive(dosing.get("dose_mg"), "Dose")
    _positive(dosing.get("infusion_minutes"), "Infusion minutes", zero=True)
    _positive(request.get("human", {}).get("weight_kg"), "Body weight")
    evidence = request.get("evidence", [])
    mw_rows = [row for row in evidence if row.get("parameter") == "molecular_weight"]
    if len(mw_rows) != 1:
        raise ValueError("Exactly one molecular weight is required")
    mw = _positive(mw_rows[0].get("value"), "Molecular weight")
    rows = training.get("observations")
    if not isinstance(rows, list) or len(rows) < 4:
        raise ValueError("At least four explicit visible development points are required")
    if any(row.get("time_unit", "min") != "min" or row.get("unit", "umol/l") != "umol/l" for row in rows):
        raise ValueError("Per-row development units differ")
    times = np.asarray([_positive(row.get("time_min"), "Time", zero=True) / 60. for row in rows])
    observed = np.asarray([_positive(row.get("concentration_umol_per_l"), "Concentration") * mw / 1000. for row in rows])
    if np.any(np.diff(times) <= 0) or not np.isfinite(observed).all():
        raise ValueError("Visible times must be strictly increasing and converted concentrations finite")
    return times, observed


def _initialization(request, physiology, variant, starts):
    cl = clearance_inputs(request)
    renal = cl["renal_clearance_l_h"] or 0.
    nonrenal = cl["total_clearance_l_h"] - renal
    cardiac = _positive(physiology.get("circulation", {}).get("cardiac_flow_l_h"), "Cardiac flow")
    absolute = (1e-4, 10. * cardiac)
    if absolute[1] <= absolute[0] or not math.isfinite(absolute[1]):
        raise ValueError("Numerical nonrenal clearance bracket is empty or nonfinite")
    lower = np.log([absolute[0] / nonrenal, .1, .01])
    upper = np.log([absolute[1] / nonrenal, 10., 100.])
    # Exactly the current three-axis relaxed fitter, including the independent
    # seeds for CL and K/PS and the clipped zero CL first start.
    cl_initial = np.r_[np.clip(0., lower[0], upper[0]),
                       np.random.default_rng(BASE_SEED).uniform(lower[0], upper[0], starts - 1)]
    fourth_bounds = np.log([.05, 20.]) if variant == "blood_partition" else np.asarray([-1., 1.])
    fourth = np.r_[0., np.random.default_rng(FOURTH_AXIS_SEED).uniform(*fourth_bounds, size=15)][:starts]
    guesses = np.column_stack([cl_initial, _shape_starts(starts), fourth])
    return np.r_[lower, fourth_bounds[0]], np.r_[upper, fourth_bounds[1]], guesses, absolute, cl


def fit(predict, request, training, physiology, *, variant, budget=16384,
        starts=16, before_trial=None):
    """Fit CL/K/PS plus log RBC scale or exchange-shape alpha.

    ``predict(x, times_h)`` must return finite positive ``concentration_mg_l``.
    Every attempted predictor call, including finite differences and numerical
    failures, is charged to its fixed start quota. Guard exceptions propagate
    before an attempt is charged. Failed evaluations never become best fits.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown distribution calibration variant")
    if not callable(predict) or (before_trial is not None and not callable(before_trial)):
        raise ValueError("Predictor and optional before_trial must be callable")
    budget = _integer(budget, "budget")
    starts = _integer(starts, "starts", maximum=16)
    if budget < starts:
        raise ValueError("budget must permit every declared start")
    times, observed = _visible(request, training)
    lower, upper, guesses, nonrenal_bounds, cl = _initialization(request, physiology, variant, starts)
    quotas = [budget // starts + int(i < budget % starts) for i in range(starts)]
    calls = failures = 0
    best = None
    traces = []
    reasons = Counter()
    beginning = time.monotonic()
    for index, (x0, quota) in enumerate(zip(guesses, quotas)):
        used = local_failures = 0
        local_best = None

        def residual(x):
            nonlocal calls, failures, used, local_failures, local_best, best
            if used >= quota or calls >= budget:
                raise CallBudgetExhausted()
            if before_trial is not None:
                before_trial()
            used += 1
            calls += 1
            try:
                result = predict(np.asarray(x, dtype=float).copy(), times.copy())
                if not isinstance(result, Mapping) or "concentration_mg_l" not in result:
                    raise ValueError("Predictor must return concentration_mg_l")
                predicted = np.asarray(result["concentration_mg_l"], dtype=float)
                if (predicted.shape != observed.shape or not np.isfinite(predicted).all()
                        or np.any(predicted <= 0)):
                    raise ValueError("Predictor must return one positive finite concentration per visible time")
                log_errors = np.log(predicted) - np.log(observed)
                fourth_penalty = .1 * x[3] / math.log(10.) if variant == "blood_partition" else .1 * x[3]
                residuals = np.r_[log_errors / math.sqrt(len(observed)),
                                  .1 * x[1] / math.log(10.), .1 * x[2] / math.log(100.), fourth_penalty]
                objective = float(residuals @ residuals)
                if not math.isfinite(objective):
                    raise ValueError("Nonfinite penalized development loss")
                parameters = {"clearance_scale": math.exp(x[0]), "partition_scale": math.exp(x[1]),
                              "bidirectional_exchange_scale": math.exp(x[2])}
                parameters["blood_partition_scale" if variant == "blood_partition" else "exchange_shape_alpha"] = math.exp(x[3]) if variant == "blood_partition" else float(x[3])
                row = {"objective": objective, "log_parameters": np.asarray(x).tolist(), "parameters": parameters,
                       "development_ln_rmse": float(np.sqrt(np.mean(log_errors ** 2))),
                       "development_gmfe": math.exp(float(np.mean(np.abs(log_errors)))),
                       "observed_window_auc_mg_h_l": float(np.trapezoid(observed, times)),
                       "predicted_window_auc_mg_h_l": float(np.trapezoid(predicted, times)),
                       "start_index": index}
                if local_best is None or objective < local_best["objective"]:
                    local_best = row
                if best is None or objective < best["objective"]:
                    best = row
                return residuals
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                failures += 1
                local_failures += 1
                reasons[type(exc).__name__ + ": " + str(exc)] += 1
                return np.full(len(observed) + 3, 1e6)

        status = "converged"
        try:
            optimized = least_squares(residual, x0, bounds=(lower, upper), max_nfev=400,
                                      ftol=1e-8, xtol=1e-8, gtol=1e-8)
            if not optimized.success:
                status = "optimizer_limit"
        except CallBudgetExhausted:
            status = "call_budget_exhausted"
        if local_best is None:
            status = "no_feasible_evaluation"
        traces.append({"index": index, "initial_log_parameters": x0.tolist(), "status": status,
                       "calls": used, "failures": local_failures, "call_limit": quota,
                       "eligible": local_best is not None, "best": local_best})
    return {"version": VERSION, "variant": variant, "status": "fitted" if best is not None else "failed",
            "best": best, "starts": traces, "calls": calls, "failures": failures, "call_limit": budget,
            "infeasible_reasons": dict(reasons), "wall_seconds": time.monotonic() - beginning,
            "visible_points": len(observed), "no_warm_start": True, "target_observations_used": False,
            "nonrenal_clearance_bounds_l_h": list(nonrenal_bounds), "clearance_input": cl,
            "parameter_lower": lower.tolist(), "parameter_upper": upper.tolist(),
            "fourth_axis": "log_blood_partition_scale" if variant == "blood_partition" else "exchange_shape_alpha"}
