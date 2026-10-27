"""Budgeted nested initialization and deterministic screening of physical axes.

This is a new search protocol, not a change to the physical model or objective.
Only the supplied development observations and callback are available here.
The nested incumbent is computed and charged within this run; no prior fit is
read or imported. A final selected point cannot be worse than that incumbent.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares

from .axes_optimizer import (
    BOUNDS, PARAMETERS, FINITE_DIFFERENCE_STEP, FAILURE_RESIDUAL,
    BudgetExhausted, _InvalidJacobian, _diagnostics, _positive_vector,
    optimize as _nested_optimize, residual_vector,
)

VERSION = "physical-axes-refined-optimizer-1"
CONFIG = {
    "nested_evaluation_cap": 18,
    "screening_evaluation_cap": 7,
    "nested_exchange_coordinate": 0.,
    "exchange_coordinates_at_nested_incumbent": (.5, -.5, 1., -1.),
    "joint_normalized_coordinates": ((0., -.5, .5), (0., .5, -.5), (0., .5, .5)),
    "joint_coordinate_reference": "absolute_normalized_log_scales",
    "ftol": 1e-6, "xtol": 1e-5, "gtol": 1e-5,
    "pattern_initial_step": .25, "pattern_minimum_step": .001,
}
_LOG_SCALES = np.log([BOUNDS[name][1] for name in PARAMETERS])


def optimize(evaluate, observations, *, active_parameters=PARAMETERS, max_evaluations=47):
    """Search all three axes using one shared budget, cache and incumbent.

    The caller charges the base model separately. Up to 18 distinct callbacks
    initialize the CL/partition subspace. Up to seven fixed screening points
    are then evaluated, and all remaining calls fund one three-axis TRF run.
    Unused stage allocations carry forward. Failed scientific trials count;
    infrastructure exceptions propagate for deterministic caller-cache replay.
    The incumbent guarantee covers this run's nested stage, not a separately
    optimized two-axis model with its own larger budget.
    """
    if not callable(evaluate):
        raise ValueError("evaluate must be callable")
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral)
            or max_evaluations < 1):
        raise ValueError("max_evaluations must be a positive integer")
    if isinstance(active_parameters, (str, bytes)):
        raise ValueError("This protocol requires all three physical axes")
    requested = tuple(active_parameters)
    if (len(requested) != len(PARAMETERS) or any(name not in PARAMETERS for name in requested)
            or len(set(requested)) != len(PARAMETERS)):
        raise ValueError("This protocol requires all three physical axes")
    observed = _positive_vector(observations, "observations")
    records, cache, coordinate_cache, coordinates_by_evaluation = [], {}, {}, {}
    seen_evaluations = set()
    jacobians, scipy_runs, polls, stages = [], [], [], []
    cache_hits = 0
    nested_incumbent = None
    full_start = None
    fallback_used = converged = exhausted = False
    termination, reason = "best_native_trial", "Numerical convergence unestablished"
    config = deepcopy(CONFIG)

    def best():
        return min((r for r in records if r["status"] == "ok"),
                   key=lambda r: (r["objective"], r["evaluation"]), default=None)

    def point_key(z):
        return tuple(float(v).hex() for v in z)

    def parameters_trial(parameters, purpose, z=None):
        nonlocal cache_hits
        key = tuple(float(parameters[name]).hex() for name in PARAMETERS)
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
        if row["status"] in {"infrastructure_failed", "infra_failed", "paused"}:
            raise RuntimeError("Infrastructure failure must be resumed, not scored as scientific failure")
        if row["status"] == "ok":
            if row.get("eligible_for_selection") is not True:
                raise ValueError("Successful native trials must be explicitly selectable")
            residual_vector(row, observed)
        elif row.get("eligible_for_selection") is not False:
            raise ValueError("A failed native trial must be explicitly ineligible")
        if z is None:
            z = np.log([parameters[name] for name in PARAMETERS]) / _LOG_SCALES
        z = np.asarray(z, dtype=float).copy()
        records.append(row)
        cache[key] = row
        seen_evaluations.add(evaluation)
        coordinates_by_evaluation[evaluation] = z
        coordinate_cache[point_key(z)] = row
        return row

    def trial(z, purpose):
        nonlocal cache_hits
        z = np.asarray(z, dtype=float)
        if z.shape != (3,) or not np.isfinite(z).all() or np.any(np.abs(z) > 1 + 1e-12):
            raise ValueError("Optimizer requested a point outside normalized bounds")
        z = np.clip(z, -1., 1.)
        # The exact retained coordinates make nested-to-full handoff a cache
        # hit, without a log/exp roundoff native solve at the same incumbent.
        coordinate_key = point_key(z)
        if coordinate_key in coordinate_cache:
            cache_hits += 1
            return coordinate_cache[coordinate_key]
        parameters = {name: float(np.clip(math.exp(float(_LOG_SCALES[i] * z[i])), *BOUNDS[name]))
                      for i, name in enumerate(PARAMETERS)}
        row = parameters_trial(parameters, purpose, z)
        coordinate_cache[coordinate_key] = row
        return row

    def residuals(z):
        row = trial(z, "full_trust_region_step")
        return (residual_vector(row, observed) if row["status"] == "ok"
                else np.full(observed.size + 3, FAILURE_RESIDUAL))

    def jacobian(z):
        center = trial(z, "full_trust_region_jacobian_center")
        if center["status"] != "ok":
            raise _InvalidJacobian("Three-axis center has invalid native physics")
        base = residual_vector(center, observed)
        columns, probes, failures = [], [], []
        for index, parameter in enumerate(PARAMETERS):
            derivative = None
            for sign in ((1., -1.) if z[index] <= 0 else (-1., 1.)):
                delta = sign * min(FINITE_DIFFERENCE_STEP, 1 - sign * z[index])
                if abs(delta) < 1e-12:
                    continue
                point = np.array(z, copy=True)
                point[index] += delta
                row = trial(point, f"full_jacobian_{parameter}_{'plus' if sign > 0 else 'minus'}")
                probes.append(row["evaluation"])
                if row["status"] == "ok":
                    derivative = (residual_vector(row, observed) - base) / delta
                    break
                failures.append(row["evaluation"])
            if derivative is None:
                jacobians.append({"stage": "three_axis", "center": center["evaluation"],
                    "probes": probes, "failed_probes": failures, "status": "failed",
                    "data_only": None, "regularized": None,
                    "interpretation": "local_only_no_global_identifiability_claim"})
                raise _InvalidJacobian(f"Both finite-difference directions failed for {parameter}")
            columns.append(derivative)
        matrix = np.column_stack(columns)
        data_information = np.sum(matrix[:observed.size] ** 2, axis=0)
        regularized_information = np.sum(matrix ** 2, axis=0)
        jacobians.append({"stage": "three_axis", "center": center["evaluation"],
            "parameters": deepcopy(center["parameters"]), "probes": probes,
            "failed_probes": failures, "status": "ok",
            "data_only": _diagnostics(matrix[:observed.size]), "regularized": _diagnostics(matrix),
            "prior_curvature_fraction_by_axis": dict(zip(PARAMETERS,
                (1 - data_information / regularized_information).tolist())),
            "coordinates": "normalized_log_scales",
            "interpretation": "local_only_no_global_identifiability_claim"})
        return matrix

    def pattern():
        step = config["pattern_initial_step"]
        directions = [np.eye(3)[i] * sign for i in range(3) for sign in (1., -1.)]
        while step >= config["pattern_minimum_step"]:
            incumbent = best()
            anchor = (np.zeros(3) if incumbent is None else
                      coordinates_by_evaluation[incumbent["evaluation"]])
            previous = None if incumbent is None else incumbent["objective"]
            count, failures = len(records), 0
            for direction in directions:
                row = trial(np.clip(anchor + step * direction, -1., 1.), "full_pattern_fallback_poll")
                failures += row["status"] != "ok"
            incumbent = best()
            improved = (incumbent is not None and
                        (previous is None or incumbent["objective"] < previous - 1e-12))
            polls.append({"stage": "three_axis", "step": step,
                          "new_parameter_points": len(records) - count,
                          "improved": improved, "failed_probes": failures})
            if not improved:
                step *= .5

    try:
        nested_cap = min(config["nested_evaluation_cap"], int(max_evaluations))
        nested = _nested_optimize(
            lambda parameters, purpose: parameters_trial(parameters, "nested_" + purpose),
            observed, active_parameters=PARAMETERS[:2], max_evaluations=nested_cap)
        nested_incumbent = deepcopy(best())
        stages.append({"stage": "nested_two_axis", "budget_limit": nested_cap,
            "evaluation_count": len(records), "status": nested["status"],
            "numerical_converged": nested["converged"],
            "incumbent_evaluation": None if nested_incumbent is None else nested_incumbent["evaluation"]})
        jacobians.extend({**deepcopy(j), "stage": "nested_two_axis"}
                         for j in nested["optimizer"]["jacobians"])
        scipy_runs.extend({**deepcopy(r), "stage": "nested_two_axis"}
                          for r in nested["optimizer"]["scipy_runs"])
        polls.extend({**deepcopy(p), "stage": "nested_two_axis"}
                     for p in nested["optimizer"]["polls"])
        fallback_used = nested["optimizer"]["physics_failure_fallback_used"]

        anchor = (np.zeros(3) if nested_incumbent is None else
                  coordinates_by_evaluation[nested_incumbent["evaluation"]].copy())
        screening_start = len(records)
        screening = {"stage": "fixed_screening", "budget_limit": config["screening_evaluation_cap"],
                     "evaluation_count": 0, "attempted_points": [], "status": "running"}
        stages.append(screening)
        candidates = []
        for exchange in config["exchange_coordinates_at_nested_incumbent"]:
            point = anchor.copy()
            point[2] = exchange
            candidates.append((point, "screen_exchange_at_nested_incumbent"))
        candidates.extend((np.array(point), "screen_fixed_joint_point")
                          for point in config["joint_normalized_coordinates"])
        for point, purpose in candidates:
            row = trial(point, purpose)
            screening["attempted_points"].append({"coordinates": point.tolist(), "evaluation": row["evaluation"]})
            screening["evaluation_count"] = len(records) - screening_start
        screening["status"] = "completed"

        selected = best()
        if selected is None:
            raise _InvalidJacobian("No feasible nested or screening point")
        full_start = selected["evaluation"]
        full_stage = {"stage": "three_axis", "start_evaluation": full_start,
                      "budget_limit": max_evaluations - len(records), "evaluation_count": 0,
                      "status": "running"}
        stages.append(full_stage)
        full_begin = len(records)
        start = coordinates_by_evaluation[full_start].copy()
        # Pass the selected point directly. scipy enforces strict feasibility
        # internally when this point is on a bound; no origin design is repeated.
        result = least_squares(residuals, start, jac=jacobian, bounds=(-1., 1.),
            method="trf", loss="linear", x_scale=1., tr_solver="exact",
            ftol=config["ftol"], xtol=config["xtol"], gtol=config["gtol"],
            max_nfev=max_evaluations)
        final = trial(result.x, "full_trust_region_final")
        selected = best()
        same_point = selected is not None and final["evaluation"] == selected["evaluation"]
        converged = bool(result.success and final["status"] == "ok" and same_point)
        criterion = {0: "function_evaluation_limit", 1: "gradient_tolerance", 2: "objective_change_tolerance",
                     3: "step_tolerance", 4: "objective_and_step_tolerance"}.get(int(result.status), "other")
        scipy_runs.append({"stage": "three_axis", "status": int(result.status),
            "message": str(result.message), "nfev": int(result.nfev), "njev": int(result.njev or 0),
            "optimality": float(result.optimality), "termination_criterion": criterion,
            "stationarity_tolerance_satisfied": bool(result.optimality <= config["gtol"]),
            "final_evaluation": final["evaluation"], "selected_equals_final_iterate": same_point})
        full_stage.update(evaluation_count=len(records) - full_begin,
                          status="converged" if converged else "best_native_trial")
        termination = "converged" if converged else "best_native_trial"
        reason = (str(result.message) if converged else
                  "Best actual development trial retained; three-axis convergence unestablished")
    except _InvalidJacobian as exc:
        fallback_used = True
        reason = str(exc)
        try:
            pattern()
            termination = "best_native_trial"
            reason += "; pattern step limit reached without verified three-axis stationarity"
        except BudgetExhausted:
            exhausted, termination = True, "budget_exhausted"
            reason += "; remaining search budget exhausted"
    except BudgetExhausted:
        exhausted, termination = True, "budget_exhausted"
        reason = "Distinct trial or caller native-attempt budget exhausted"

    selected = best()
    if stages and stages[-1]["status"] == "running":
        stages[-1]["status"] = termination
        if stages[-1]["stage"] == "three_axis":
            stages[-1]["evaluation_count"] = len(records) - full_begin
    selected_jacobians = [j for j in jacobians if j["stage"] == "three_axis" and j["status"] == "ok"
                          and selected is not None and j["center"] == selected["evaluation"]]
    selected_z = None if selected is None else coordinates_by_evaluation[selected["evaluation"]]
    return {
        "status": termination if selected else "no_feasible_point", "termination": termination,
        "reason": reason, "eligible": selected is not None, "converged": converged,
        "budget_exhausted": exhausted, "budget_remaining": int(max_evaluations - len(records)),
        "decision": deepcopy(selected), "evaluations": records, "evaluation_count": len(records),
        "failed_evaluation_count": sum(row["status"] != "ok" for row in records),
        "cache_hits": cache_hits, "active_parameters": list(PARAMETERS),
        "screening": {"status": "fixed_candidate_design", "selected_parameters": list(PARAMETERS),
                      "reason": "No axis frozen; only the development objective selects the start"},
        "optimizer": {
            "version": VERSION, "method": "nested_screened_bounded_trf", "config": config,
            "native_trial_budget": int(max_evaluations), "base_solve_included": False,
            "bounds": {name: list(BOUNDS[name]) for name in PARAMETERS},
            "bounds_interpretation": "engineering_windows_not_population_prior_intervals",
            "finite_difference_normalized_step": FINITE_DIFFERENCE_STEP,
            "objective": "mean_log_concentration_MSE_plus_fixed_three_axis_quadratic_priors",
            "residual_count": int(observed.size + 3), "numerical_convergence_only": True,
            "active_parameters": list(PARAMETERS), "fixed_at_one": [],
            "stages": stages, "nested_incumbent": nested_incumbent,
            "nested_incumbent_guarantee": "within_this_run_only_not_separate_full_budget_two_axis_fit",
            "three_axis_start_evaluation": full_start,
            "jacobians": jacobians, "scipy_runs": scipy_runs, "polls": polls,
            "selected_local_diagnostics": deepcopy(selected_jacobians[-1]) if selected_jacobians else None,
            "selected_boundary_axes": ([] if selected_z is None else
                [name for name, z in zip(PARAMETERS, selected_z) if abs(z) >= 1 - 1e-6]),
            "physics_failure_fallback_used": fallback_used,
            "budget_includes": ["nested trials", "fixed screening", "Jacobian probes",
                                "failed physics", "three-axis TRF trials", "fallback polls"],
            "prior_fit_inputs_used": False,
        },
    }
