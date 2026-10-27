"""A finite union of coordinate profiles on visible development data.

Each of four log parameters is fixed in turn while the other three are fitted.
This covers more directions than an RBC-only profile, but is NOT a full joint
uncertainty sample, probability posterior, or calibrated confidence region.
No clinical target data, cross-profile continuation, or external warm start is
accepted. The caller protects measured evidence and selects eligible members.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares


VERSION = "four-axis-evidence-profile-union-1"
SEED = 20261008
AXES = ("log_nonrenal_clearance_scale", "log_tissue_partition_scale",
        "log_exchange_scale", "log_RBC_partition_scale")


class ProfileBudgetExhausted(RuntimeError):
    """The assigned start exhausted its actual model-call quota."""


def _integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return int(value)


def _quotas(total, count):
    return [total // count + int(i < total % count) for i in range(count)]


def _grid(lower, upper, count, fitted):
    values = np.linspace(lower, upper, count).tolist()
    if lower <= 0. <= upper:
        # Remove roundoff neighbors of an intended zero interior coordinate.
        values = [0. if lower < x < upper and abs(x) <= 1e-12 else x for x in values]
        values.append(0.)
    if fitted is not None and lower <= fitted <= upper:
        if not any(abs(fitted - x) <= 1e-12 for x in values):
            values.append(float(fitted))
    return sorted(set(values))


def analyze(predict, times, observed, lower4, upper4, *, grid_points=7,
            max_calls=65536, before_trial=None):
    """Profile all four coordinates with exactly the visible observations.

    ``predict(x, times)`` returns concentrations in mg/L for log coordinates in
    ORIGINAL [CL, K, PS, RBC] order. Each node has two independent deterministic
    nuisance starts. The visible-data location fit adds grid coordinates only;
    its fitted nuisance parameters are never reused as starts. All attempted
    model calls, finite differences and failures consume their assigned budget.
    Numerical truncation stays unsuccessful, even if it found a finite point.
    """
    if not callable(predict) or (before_trial is not None and not callable(before_trial)):
        raise ValueError("Predictor and optional guard must be callable")
    grid_points = _integer(grid_points, "grid_points", 3)
    max_calls = _integer(max_calls, "max_calls")
    t, y = np.array(times, dtype=float, copy=True), np.array(observed, dtype=float, copy=True)
    lower, upper = np.array(lower4, dtype=float, copy=True), np.array(upper4, dtype=float, copy=True)
    if (t.ndim != 1 or len(t) < 4 or y.shape != t.shape or not np.isfinite(t).all()
            or np.any(t < 0) or np.any(np.diff(t) <= 0)
            or not np.isfinite(y).all() or np.any(y <= 0)):
        raise ValueError("Require at least four ordered visible hours and positive finite mg/L observations")
    if (lower.shape != (4,) or upper.shape != (4,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower >= upper)):
        raise ValueError("Four finite ordered log-coordinate bounds are required")
    zeros = np.clip(np.zeros(4), lower, upper)
    independent = np.random.default_rng(SEED).uniform(lower, upper)
    calls = failures = 0
    reasons = Counter()

    def fit_at(initial, quota, *, axis=None, coordinate=None):
        nonlocal calls, failures
        free = np.array([i for i in range(4) if i != axis], dtype=int)
        initial = np.array(initial, dtype=float, copy=True)
        if axis is not None:
            initial[axis] = coordinate
        used = bad = 0
        best = None
        receipt = dict(initial_log_parameters=initial.tolist(), free_axes=free.tolist(),
                       axis=axis, fixed_coordinate=coordinate, call_limit=quota)
        if quota == 0:
            return receipt | dict(successful=False, status="no_call_quota", calls=0, failures=0, best=None)

        def residual(z):
            nonlocal calls, failures, used, bad, best
            if used >= quota or calls >= max_calls:
                raise ProfileBudgetExhausted()
            if before_trial is not None:
                before_trial()
            used += 1
            calls += 1
            x = initial.copy()
            x[free] = np.asarray(z, dtype=float)
            try:
                answer = predict(x.copy(), t.copy())
                if not isinstance(answer, Mapping) or "concentration_mg_l" not in answer:
                    raise ValueError("Predictor must return concentration_mg_l")
                predicted = np.asarray(answer["concentration_mg_l"], dtype=float)
                if predicted.shape != y.shape or not np.isfinite(predicted).all() or np.any(predicted <= 0):
                    raise ValueError("Visible predictions must be finite, positive, and match visible times")
                values = (np.log(predicted) - np.log(y)) / math.sqrt(len(y))
                mse = float(values @ values)
                if not math.isfinite(mse):
                    raise ValueError("Nonfinite visible-data loss")
                row = dict(mse=mse, log_parameters=x.tolist())
                if best is None or mse < best["mse"]:
                    best = row
                return values
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                failures += 1
                bad += 1
                reasons[type(exc).__name__ + ": " + str(exc)] += 1
                return np.full(len(y), 1e6)

        successful, status = False, "converged"
        try:
            result = least_squares(residual, initial[free], bounds=(lower[free], upper[free]),
                                   max_nfev=400, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            successful = bool(result.success) and best is not None
            if not result.success:
                status = "optimizer_limit"
        except ProfileBudgetExhausted:
            status = "call_budget_exhausted"
        if best is None:
            status = "no_feasible_evaluation"
        return receipt | dict(successful=successful, status=status, calls=used, failures=bad, best=best)

    # This data-only location estimate is not a MAP/posterior estimate. It may
    # contribute a grid coordinate even if numerically truncated, but never an
    # eligible committee member unless a profile node itself converges.
    full_fit = fit_at(zeros, max_calls // (4 * grid_points + 3))
    location = None if full_fit["best"] is None else full_fit["best"]["log_parameters"]
    grids = [_grid(lower[i], upper[i], grid_points, None if location is None else location[i]) for i in range(4)]
    nodes = [(i, point) for i, grid in enumerate(grids) for point in grid]
    node_quotas = _quotas(max_calls - calls, len(nodes))
    profile = []
    for (axis, coordinate), quota in zip(nodes, node_quotas):
        runs = [fit_at(initial, part, axis=axis, coordinate=coordinate)
                for initial, part in zip((zeros, independent), _quotas(quota, 2))]
        eligible = [run["best"] for run in runs if run["successful"]]
        best = min(eligible, key=lambda row: row["mse"], default=None)
        profile.append(dict(axis=axis, axis_name=AXES[axis], fixed_coordinate=coordinate,
            successful=best is not None, mse=None if best is None else best["mse"], best=best,
            starts=runs, calls=sum(row["calls"] for row in runs),
            failures=sum(row["failures"] for row in runs), call_limit=quota))
    complete = all(row["successful"] for row in profile)
    return dict(version=VERSION, status="profiled" if complete else "incomplete",
        profile=profile, full_fit=full_fit, visible_points=len(t), calls=calls, failures=failures,
        max_calls=max_calls, failure_reasons=dict(reasons), lower4=lower.tolist(), upper4=upper.tolist(),
        requested_grid_points=grid_points, actual_grid_points=len(profile),
        axis_grids={name:grid for name,grid in zip(AXES,grids)}, seed=SEED,
        target_observations_used=False, external_warm_start_used=False, neighbor_continuation_used=False,
        parameter_order=list(AXES), units=dict(time="h", concentration="mg/L", coordinates="log multipliers"),
        interpretation="Finite union of coordinate profiles; not a joint posterior or calibrated confidence region")
