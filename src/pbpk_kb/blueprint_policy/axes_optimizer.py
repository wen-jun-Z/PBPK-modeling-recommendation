"""Bounded calibration of physical PBPK scales through a pure trial callback.

Only development observations enter this optimizer. Every distinct callback
request, including failed physics and finite differences, uses the same budget.
The bounds are engineering exploration windows, not population prior intervals.
No single local sensitivity calculation freezes an otherwise active parameter.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import least_squares

VERSION = "physical-axes-optimizer-1"
PARAMETERS = ("clearance_scale", "partition_scale", "bidirectional_exchange_scale")
BOUNDS = {"clearance_scale": (.25, 4.), "partition_scale": (.1, 10.),
          "bidirectional_exchange_scale": (.01, 100.)}
FINITE_DIFFERENCE_STEP = .01
MINIMUM_PATTERN_STEP = .001
FAILURE_RESIDUAL = 1e6
_LOG_SCALES = np.log([BOUNDS[name][1] for name in PARAMETERS])


class BudgetExhausted(Exception):
    """The allowed number of distinct trial requests has been consumed."""


class _InvalidJacobian(Exception):
    """Native failures prevent a defensible local finite difference."""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _positive_vector(values, name):
    array = np.asarray(values)
    if (array.ndim != 1 or not array.size or array.dtype.kind not in "iuf"
            or not np.isfinite(array).all() or np.any(array <= 0)):
        raise ValueError(f"{name} must be a nonempty positive finite numeric vector")
    return array.astype(float)


def residual_vector(row, observations):
    """Validate the fixed loss and return its n+3 residual representation.

    The squared norm is mean squared natural-log concentration error plus
    .1*(ln(CL)/ln(2))**2 + .01*(ln(partition)/ln(10))**2 +
    .01*(ln(exchange)/ln(100))**2. Explicit native residuals must match this
    vector, not merely have the same squared norm. No alternate loss is allowed.
    """
    observed = _positive_vector(observations, "observations")
    predicted = _positive_vector(row["predicted_at_observed_times"], "predictions")
    if predicted.shape != observed.shape:
        raise ValueError("Development predictions must match observation shape")
    parameters = row.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != set(PARAMETERS):
        raise ValueError("Native parameters must contain exactly the three physical scales")
    scales = [_finite(parameters[name], name) for name in PARAMETERS]
    if any(value <= 0 for value in scales):
        raise ValueError("Physical scales must be positive")
    residuals = np.r_[
        (np.log(predicted) - np.log(observed)) / math.sqrt(observed.size),
        math.sqrt(.1) * math.log(scales[0]) / math.log(2),
        .1 * math.log(scales[1]) / math.log(10),
        .1 * math.log(scales[2]) / math.log(100),
    ]
    if "residual_vector" in row:
        supplied = np.asarray(row["residual_vector"], dtype=float)
        if supplied.shape != (observed.size + 3,) or not np.isfinite(supplied).all():
            raise ValueError("Native residual_vector must have n_observations + 3 finite entries")
        if not np.allclose(supplied, residuals, rtol=1e-9, atol=1e-11):
            raise ValueError("Native residual_vector differs from the fixed physical-axes loss")
    objective = _finite(row["objective"], "objective")
    if not math.isclose(float(residuals @ residuals), objective, rel_tol=1e-9, abs_tol=1e-11):
        raise ValueError("Native objective differs from the fixed physical-axes residual objective")
    return residuals


def _diagnostics(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    columns = matrix.shape[1]
    full_rank = rank == columns
    condition = float(singular[0] / singular[-1]) if full_rank else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    return {"rank": rank, "column_count": columns, "full_column_rank": full_rank,
            "condition_number": condition, "singular_values": singular.tolist(),
            "column_norms": np.linalg.norm(matrix, axis=0).tolist()}


def optimize(evaluate, observations, *, active_parameters=PARAMETERS, max_evaluations=48):
    """Run deterministic bounded TRF, with pattern fallback for invalid Jacobians.

    ``evaluate(parameters, purpose)`` receives all three scales, with inactive
    scales fixed to one. It returns an immutable trial mapping with ``status``,
    unique integer ``evaluation``, ``parameters``, ``eligible_for_selection``;
    successful trials also supply ``objective`` and development predictions.
    Callback scientific failures must return ineligible trial records; unexpected
    exceptions propagate so an infrastructure interruption can be resumed by
    replaying the same callback cache. No hidden warm start is introduced.

    The CL-only, CL+partition and all-axis fits use this same API and budget;
    each distinct Jacobian/design/search/failure request counts once. Local
    data-only Jacobian diagnostics exclude regularization residuals and make no
    global identifiability claim. Numerical convergence is not clinical validity.
    """
    if not callable(evaluate):
        raise ValueError("evaluate must be callable")
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral)
            or max_evaluations < 1):
        raise ValueError("max_evaluations must be a positive integer")
    if isinstance(active_parameters, (str, bytes)):
        raise ValueError("active_parameters must be a sequence")
    requested = tuple(active_parameters)
    if (any(not isinstance(name, str) or name not in PARAMETERS for name in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError("Unknown or duplicate active parameter")
    allowed = tuple(name for name in PARAMETERS if name in requested)
    active = [PARAMETERS.index(name) for name in allowed]
    observed = _positive_vector(observations, "observations")

    records, cache, jacobians, polls, scipy_runs = [], {}, [], [], []
    evaluated_coordinates = {}
    seen_evaluations = set()
    cache_hits = 0
    converged = fallback_used = exhausted = False
    termination, reason = "budget_exhausted", "Distinct trial budget exhausted"

    def full_point(x):
        values = np.asarray(x, dtype=float)
        if values.shape != (len(active),):
            raise ValueError("Incorrect normalized parameter dimension")
        z = np.zeros(3)
        z[active] = values
        if not np.isfinite(z).all() or np.any(np.abs(z) > 1 + 1e-12):
            raise ValueError("Optimizer requested a point outside normalized bounds")
        return np.clip(z, -1., 1.)

    def coordinates(row):
        # Retain the original optimizer coordinates: log(exp(z)) roundoff must
        # not turn a repeated center into another costly native callback.
        return evaluated_coordinates[row["evaluation"]].copy()

    def trial(x, purpose):
        nonlocal cache_hits
        z = full_point(x)
        parameters = {name: float(np.clip(math.exp(float(_LOG_SCALES[i] * z[i])), *BOUNDS[name]))
                      for i, name in enumerate(PARAMETERS)}
        key = tuple(parameters[name].hex() for name in PARAMETERS)
        if key in cache:
            cache_hits += 1
            return cache[key]
        if len(records) >= max_evaluations:
            raise BudgetExhausted
        returned = evaluate(deepcopy(parameters), purpose)
        if not isinstance(returned, Mapping):
            raise ValueError("Native callback must return a trial mapping")
        row = deepcopy(dict(returned))
        if row.get("parameters") != parameters:
            raise ValueError("Native trial returned different physical parameters")
        evaluation = row.get("evaluation")
        if (isinstance(evaluation, bool) or not isinstance(evaluation, Integral)
                or evaluation < 0 or evaluation in seen_evaluations):
            raise ValueError("Native evaluation must be a unique nonnegative integer")
        if not isinstance(row.get("status"), str) or not row["status"]:
            raise ValueError("Native trial must have an explicit status")
        if row.get("status") == "ok":
            if row.get("eligible_for_selection") is not True:
                raise ValueError("Successful native trials must be explicitly selectable")
            residual_vector(row, observed)
        elif row.get("eligible_for_selection") is not False:
            raise ValueError("A failed native trial must be explicitly ineligible")
        records.append(row)
        cache[key] = row
        seen_evaluations.add(evaluation)
        evaluated_coordinates[evaluation] = z[active].copy()
        return row

    def best():
        return min((row for row in records if row["status"] == "ok"),
                   key=lambda row: (row["objective"], row["evaluation"]), default=None)

    def residuals(x):
        row = trial(x, "trust_region_step")
        if row["status"] != "ok":
            return np.full(observed.size + 3, FAILURE_RESIDUAL)
        return residual_vector(row, observed)

    def jacobian(x):
        center = trial(x, "trust_region_jacobian_center")
        if center["status"] != "ok":
            raise _InvalidJacobian("Trust-region center has invalid native physics")
        base = residual_vector(center, observed)
        columns, probes, failed = [], [], []
        for index, parameter in enumerate(allowed):
            derivative = None
            signs = (1., -1.) if x[index] <= 0 else (-1., 1.)
            for sign in signs:
                delta = sign * min(FINITE_DIFFERENCE_STEP, 1 - sign * x[index])
                if abs(delta) < 1e-12:
                    continue
                point = np.array(x, copy=True)
                point[index] += delta
                row = trial(point, f"trust_region_jacobian_{parameter}_{'plus' if sign > 0 else 'minus'}")
                probes.append(row["evaluation"])
                if row["status"] == "ok":
                    derivative = (residual_vector(row, observed) - base) / delta
                    break
                failed.append(row["evaluation"])
            if derivative is None:
                jacobians.append({"center": center["evaluation"], "probes": probes,
                                  "failed_probes": failed, "status": "failed",
                                  "data_only": None, "regularized": None,
                                  "interpretation": "local_only_no_global_identifiability_claim"})
                raise _InvalidJacobian(f"Both finite-difference directions failed for {parameter}")
            columns.append(derivative)
        matrix = np.column_stack(columns)
        jacobians.append({"center": center["evaluation"], "parameters": deepcopy(center["parameters"]),
                          "probes": probes, "failed_probes": failed, "status": "ok",
                          "data_only": _diagnostics(matrix[:observed.size]),
                          "regularized": _diagnostics(matrix),
                          "coordinates": "normalized_log_scales",
                          "interpretation": "local_only_no_global_identifiability_claim"})
        return matrix

    def pattern():
        step = .25
        directions = [np.eye(len(active))[i] * sign for i in range(len(active)) for sign in (1., -1.)]
        while step >= MINIMUM_PATTERN_STEP:
            selected = best()
            anchor = np.zeros(len(active)) if selected is None else coordinates(selected)
            previous = None if selected is None else selected["objective"]
            count, failures = len(records), 0
            for direction in directions:
                row = trial(np.clip(anchor + step * direction, -1., 1.), "pattern_fallback_poll")
                failures += row["status"] != "ok"
            selected = best()
            improved = selected is not None and (previous is None or selected["objective"] < previous - 1e-12)
            polls.append({"step": step, "new_parameter_points": len(records) - count,
                          "improved": improved, "failed_probes": failures})
            if not improved:
                step *= .5
        return bool(best()) and bool(polls) and polls[-1]["failed_probes"] == 0

    try:
        trial(np.zeros(len(active)), "evidence_origin")
        if not active:
            termination, reason = "fixed_parameters_only", "All scales fixed to one"
        else:
            for index in range(len(active)):
                for sign in (1., -1.):
                    point = np.zeros(len(active))
                    point[index] = sign * .5
                    trial(point, f"initial_axis_{allowed[index]}_{'plus' if sign > 0 else 'minus'}")
            selected = best()
            if selected is None:
                raise _InvalidJacobian("No feasible origin or initial axis probe")
            start = np.clip(coordinates(selected), -1 + 1e-10, 1 - 1e-10)
            result = least_squares(residuals, start, jac=jacobian, bounds=(-1., 1.),
                                   method="trf", loss="linear", x_scale=1., tr_solver="exact",
                                   ftol=1e-6, xtol=1e-5, gtol=1e-5, max_nfev=max_evaluations)
            final = trial(result.x, "trust_region_final")
            selected = best()
            same_point = selected is not None and final["evaluation"] == selected["evaluation"]
            converged = bool(result.success and final["status"] == "ok" and same_point)
            scipy_runs.append({"status": int(result.status), "message": str(result.message),
                               "nfev": int(result.nfev), "njev": int(result.njev or 0),
                               "optimality": float(result.optimality),
                               "final_evaluation": final["evaluation"],
                               "selected_equals_final_iterate": same_point})
            termination = "converged" if converged else "best_native_trial"
            reason = str(result.message) if converged else "Best native trial retained; convergence unestablished"
    except _InvalidJacobian as exc:
        fallback_used = True
        reason = str(exc)
        try:
            converged = pattern()
            termination = "converged" if converged else "no_feasible_improvement"
            reason += ("; pattern step tolerance reached" if converged
                       else "; final poll contains failures, convergence unestablished")
        except BudgetExhausted:
            exhausted, converged, termination = True, False, "budget_exhausted"
            reason += "; remaining pattern-search budget exhausted"
    except BudgetExhausted:
        exhausted, converged, termination = True, False, "budget_exhausted"

    selected = best()
    return {
        "status": termination if selected else "no_feasible_point", "termination": termination,
        "reason": reason, "eligible": selected is not None, "converged": converged,
        "budget_exhausted": exhausted, "budget_remaining": int(max_evaluations - len(records)),
        "decision": deepcopy(selected), "evaluations": records, "evaluation_count": len(records),
        "failed_evaluation_count": sum(row["status"] != "ok" for row in records),
        "cache_hits": cache_hits, "active_parameters": list(allowed),
        "screening": {"status": "not_requested", "selected_parameters": list(allowed),
                      "reason": "No axis frozen by a single local sensitivity calculation"},
        "optimizer": {
            "version": VERSION, "method": "bounded_trf", "native_trial_budget": int(max_evaluations),
            "bounds": {name: list(BOUNDS[name]) for name in PARAMETERS},
            "bounds_interpretation": "engineering_windows_not_population_prior_intervals",
            "finite_difference_normalized_step": FINITE_DIFFERENCE_STEP,
            "initial_design": "origin_then_each_active_axis_plus_and_minus_half_range",
            "objective": "mean_log_concentration_MSE_plus_fixed_three_axis_quadratic_priors",
            "residual_count": int(observed.size + 3), "numerical_convergence_only": True,
            "active_parameters": list(allowed),
            "fixed_at_one": [name for name in PARAMETERS if name not in allowed],
            "jacobians": jacobians, "scipy_runs": scipy_runs, "polls": polls,
            "physics_failure_fallback_used": fallback_used,
            "budget_includes": ["origin", "initial axis probes", "Jacobian probes",
                                "failed physics", "trust-region trials", "fallback polls"],
            "reference": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html",
        },
    }
