"""Small, bounded development-curve calibration with explicit PK priors.

This numerical module never reads files or comparison data. Its callback is
responsible for executing and auditing the mechanistic model. The fixed loss
adapts normalized log concentration residuals and regularization; it is not an
implementation of an official OSP algorithm or a physiological confidence model.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from numbers import Integral, Real
from typing import Any


_STEP = 0.03
_SENSITIVITY_RMS_MIN = 1e-6
_RANK_RELATIVE_TOLERANCE = 1e-3
_SUCCESS = {"predicted", "ok", "success", "completed"}
_NAMES = ("effective_lipophilicity", "log_clearance_scale")
FAILURE_RECOVERY_VERSION = "failed-search-backtrack-1.0"
_BACKTRACK_FRACTIONS = (.5, .25, .125)
_MAX_RESTARTS = 2


class _Stop(Exception):
    def __init__(self, status: str, reason: str):
        self.status, self.reason = status, reason
        super().__init__(reason)


class _SearchFailure(_Stop):
    """Abort least_squares without returning invented or off-coordinate residuals."""
    def __init__(self, record):
        super().__init__("evaluation_failed", record["reason"])
        self.record = record


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + " must be finite" + (" and positive" if positive else ""))
    return value


def _observations(document):
    rows = document.get("observations") if isinstance(document, Mapping) else None
    if not isinstance(rows, (list, tuple)) or len(rows) < 2:
        raise ValueError("observations must contain at least two concentration records")
    times, values = [], []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each observation must be a mapping")
        times.append(_number(row.get("time_min"), "observation time"))
        values.append(_number(row.get("concentration_umol_per_l"), "observed concentration", positive=True))
    if times[0] < 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Observation times must be increasing and nonnegative")
    if document.get("time_unit", "min") != "min" or document.get("unit", "umol/l") != "umol/l":
        raise ValueError("Observation units must be min and umol/l")
    return times, values


def _sample_curve(curve, times):
    import numpy as np

    if not isinstance(curve, Mapping) or curve.get("time_unit") != "min" or curve.get("unit") != "umol/l":
        raise ValueError("Simulated curve must declare time_unit=min and unit=umol/l")
    raw_times, raw_values = curve.get("time_min"), curve.get("concentration")
    if (not isinstance(raw_times, (list, tuple)) or not isinstance(raw_values, (list, tuple))
            or len(raw_times) < 2 or len(raw_times) != len(raw_values)):
        raise ValueError("Simulated curve requires aligned time_min and concentration lists")
    pt = [_number(value, "simulated time") for value in raw_times]
    pv = [_number(value, "simulated concentration") for value in raw_values]
    if pt[0] < 0 or any(v < 0 for v in pv) or any(b <= a for a, b in zip(pt, pt[1:])):
        raise ValueError("Simulated times must increase; times and concentrations must be nonnegative")
    if times[0] < pt[0] or times[-1] > pt[-1]:
        raise ValueError("Simulated curve does not cover observations; extrapolation is forbidden")
    sampled = np.interp(times, pt, pv)
    if np.any(sampled <= 0) or not np.all(np.isfinite(sampled)):
        raise ValueError("Simulated concentrations must be positive at every observation time")
    return sampled


def optimize_development_curve(
    evaluate: Callable[[float, float], Mapping[str, Any]], *, observations,
    initial_lipophilicity, initial_clearance_scale=1., logp_prior,
    target_clearance_l_h, target_vss_l=None, max_evaluations=18,
    recover_failed_trials=False,
):
    """Calibrate at most two construction parameters on one development curve.

    Callback results require ``status``, positive ``clearance_l_h``, optional
    positive ``vss_l`` and the existing native curve shape: ``time_min``,
    ``concentration``, ``time_unit='min'``, ``unit='umol/l'``. Observations use
    ``observations[{time_min, concentration_umol_per_l}]``. No extrapolation,
    substituted observations or synthetic penalties for failed runs are used.

    All callback calls, including sensitivity and Jacobian calculations, count
    against the budget. Failed trials remain visible. On budget exhaustion or
    a failed search trial the best actually evaluated, admissible point remains
    available. Optional recovery aborts that least_squares invocation, evaluates
    up to three points from the best admissible point toward the failed point,
    and restarts only at a genuinely evaluated strict improvement (at most two
    restarts). Recovery shares the original call budget and frozen axes.
    Numerical convergence is separate from PK prior agreement.
    """
    if not callable(evaluate):
        raise ValueError("evaluate must be callable")
    times, observed = _observations(observations)
    initial = _number(initial_lipophilicity, "initial_lipophilicity")
    scale = _number(initial_clearance_scale, "initial_clearance_scale", positive=True)
    prior = _number(logp_prior, "logp_prior")
    target_cl = _number(target_clearance_l_h, "target_clearance_l_h", positive=True)
    target_vss = None if target_vss_l is None else _number(target_vss_l, "target_vss_l", positive=True)
    if not prior - 2.5 < prior < prior + 2.5 or not prior - 2.5 <= initial <= prior + 2.5:
        raise ValueError("Initial lipophilicity must lie in representable logp_prior +/- 2.5 bounds")
    if not .25 <= scale <= 4.:
        raise ValueError("Initial clearance scale must lie in [0.25, 4]")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    max_evaluations = int(max_evaluations)
    if not isinstance(recover_failed_trials, bool):
        raise ValueError("recover_failed_trials must be a boolean")

    import numpy as np
    from scipy.optimize import least_squares

    observed = np.asarray(observed, dtype=float)
    lower = np.array([prior - 2.5, math.log(.25)])
    upper = np.array([prior + 2.5, math.log(4.)])
    initial_coordinates = np.array([initial, math.log(scale)])
    records, cache = [], {}
    cache_hits = 0
    active = None
    sensitivity = {"status": "not_completed", "basis": "local derivative of log simulated concentrations at development sample times",
        "finite_difference_step": _STEP, "rms_sensitivity_minimum": _SENSITIVITY_RMS_MIN,
        "rank_relative_tolerance": _RANK_RELATIVE_TOLERANCE,
        "threshold_interpretation": "Computational diagnostics, not physiological uncertainty or global identifiability",
        "parameter_order": list(_NAMES), "columns": [], "rank": None,
        "condition_number": None, "active_parameters": [], "frozen_parameters": []}
    optimizer = {"method": "bounded_trust_region_least_squares", "success": False,
        "finite_difference_step": _STEP, "numerical_convergence_only": True}
    recovery = {"version": FAILURE_RECOVERY_VERSION, "enabled": recover_failed_trials,
        "backtrack_fractions": list(_BACKTRACK_FRACTIONS), "maximum_restarts": _MAX_RESTARTS,
        "acceptance": "Actually evaluated admissible point with objective strictly below the best feasible anchor",
        "budget": "All native calls, including failures and backtracks, share max_evaluations",
        "restart_count": 0, "events": []}

    def trial(coordinates, purpose):
        nonlocal cache_hits
        lip = float(coordinates[0])
        multiplier = float(np.clip(math.exp(float(coordinates[1])), .25, 4.))
        key = (lip, multiplier)
        if key in cache:
            cache_hits += 1
            return cache[key]
        if len(records) >= max_evaluations:
            raise _Stop("budget_exhausted", "Actual simulation call budget exhausted")
        record = {"evaluation": len(records) + 1, "purpose": purpose,
            "lipophilicity": lip, "clearance_scale": multiplier,
            "coordinates": [float(v) for v in coordinates], "status": "evaluation_failed",
            "eligible_for_selection": False}
        records.append(record)
        cache[key] = record
        try:
            result = evaluate(lip, multiplier)
            if not isinstance(result, Mapping):
                raise ValueError("evaluate must return a mapping")
            status = result.get("status")
            record["callback_status"] = status if isinstance(status, str) else None
            if status not in _SUCCESS:
                raise ValueError("Unsuccessful simulation status " + repr(status) + ": " + str(result.get("reason", "")))
            cl = _number(result.get("clearance_l_h"), "simulated clearance_l_h", positive=True)
            raw_vss = result.get("vss_l")
            vss = None if raw_vss is None and target_vss is None else _number(raw_vss, "simulated vss_l", positive=True)
            predicted = _sample_curve(result.get("curve"), times)
            log_errors = np.log(predicted) - np.log(observed)
            cl_log = math.log(cl) - math.log(target_cl)
            vss_log = None if target_vss is None else math.log(vss) - math.log(target_vss)
            lip_residual = (lip - prior) / 2.5
            components = {"mean_squared_log_curve_error": float(np.mean(log_errors ** 2)),
                "clearance_prior": .1 * (cl_log / math.log(2.)) ** 2,
                "vss_prior": 0. if vss_log is None else .1 * (vss_log / math.log(2.)) ** 2,
                "lipophilicity_prior": .01 * lip_residual ** 2}
            residuals = [float(v) / math.sqrt(len(times)) for v in log_errors]
            residuals.append(math.sqrt(.1) * cl_log / math.log(2.))
            if vss_log is not None:
                residuals.append(math.sqrt(.1) * vss_log / math.log(2.))
            residuals.append(math.sqrt(.01) * lip_residual)
            objective = math.fsum(components.values())
            if not math.isfinite(objective):
                raise ValueError("Objective is nonfinite")
            record.update(status="ok", eligible_for_selection=True, objective=objective,
                objective_components=components, residuals=residuals,
                sampled_log_ratios=log_errors.tolist(), predicted_at_observed_times=predicted.tolist(),
                clearance_l_h=cl, vss_l=vss,
                clearance_relative_error=abs(cl / target_cl - 1.),
                vss_relative_error=None if target_vss is None else abs(vss / target_vss - 1.),
                curve_metrics={"point_count": len(times), "gmfe": math.exp(float(np.mean(np.abs(log_errors)))),
                    "log_rmse": math.sqrt(components["mean_squared_log_curve_error"]),
                    "mean_log_ratio": float(np.mean(log_errors)),
                    "within_twofold_fraction": float(np.mean(np.abs(log_errors) <= math.log(2.)))})
        except Exception as exc:
            record.update(status="evaluation_failed", eligible_for_selection=False,
                          reason=type(exc).__name__ + ": " + str(exc))
        return record

    def perturb(coordinates, column):
        shifted = np.array(coordinates, copy=True)
        step = _STEP if shifted[column] + _STEP <= upper[column] else -_STEP
        shifted[column] = float(np.clip(shifted[column] + step, lower[column], upper[column]))
        actual_step = float(shifted[column] - coordinates[column])
        if not actual_step:
            raise _Stop("not_converged", "No representable finite difference inside bounds")
        return shifted, actual_step

    def enforce_frozen():
        if active is None:
            return
        frozen = set(range(2)) - set(active)
        for record in records:
            if record["status"] != "ok":
                continue
            if any(record["coordinates"][i] != float(initial_coordinates[i]) for i in frozen):
                record.update(eligible_for_selection=False,
                    selection_exclusion="Sensitivity-only perturbation of a subsequently frozen parameter")

    try:
        first = trial(initial_coordinates, "initial")
        if first["status"] != "ok":
            raise _Stop("initial_evaluation_failed", first["reason"])
        columns = {}
        for i in range(2):
            shifted, step = perturb(initial_coordinates, i)
            result = trial(shifted, "initial_sensitivity")
            info = {"parameter": _NAMES[i], "evaluation": result["evaluation"], "step": step}
            if result["status"] == "ok":
                derivative = (np.asarray(result["sampled_log_ratios"]) - first["sampled_log_ratios"]) / step
                norm = float(np.sqrt(np.mean(derivative ** 2)))
                info.update(status="evaluated", rms_derivative=norm, derivative=derivative.tolist())
                if norm >= _SENSITIVITY_RMS_MIN:
                    columns[i] = derivative
                else:
                    info["freeze_reason"] = "Curve response below the predeclared local sensitivity threshold"
            else:
                info.update(status="evaluation_failed", freeze_reason="Sensitivity perturbation was not a feasible audited simulation")
            sensitivity["columns"].append(info)
        candidates = list(columns)
        singular = np.linalg.svd(np.column_stack([columns[i] for i in candidates]), compute_uv=False) if candidates else np.array([])
        rank = int(np.sum(singular > singular[0] * _RANK_RELATIVE_TOLERANCE)) if len(singular) else 0
        if rank == len(candidates):
            active = candidates
        elif rank:
            active = [max(candidates, key=lambda i: float(np.linalg.norm(columns[i])))]
        else:
            active = []
        condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else None
        if condition is not None and not math.isfinite(condition):
            condition = None
        sensitivity.update(status="completed", rank=rank, singular_values=singular.tolist(), condition_number=condition,
            active_parameters=[_NAMES[i] for i in active])
        for i in set(range(2)) - set(active):
            item = next(r for r in sensitivity["columns"] if r["parameter"] == _NAMES[i])
            reason = item.get("freeze_reason", "Locally collinear curve effects; retained the column with greater RMS sensitivity")
            sensitivity["frozen_parameters"].append({"parameter": _NAMES[i], "coordinate_value": float(initial_coordinates[i]), "reason": reason})
        enforce_frozen()
        if not active:
            raise _Stop("no_identifiable_parameters", "No locally informative independent curve parameter remains")

        def full(reduced):
            coordinates = np.array(initial_coordinates, copy=True)
            coordinates[active] = reduced
            return coordinates

        def residual(reduced):
            result = trial(full(reduced), "optimization")
            if result["status"] != "ok":
                raise _SearchFailure(result)
            return np.asarray(result["residuals"])

        def jacobian(reduced):
            coordinates = full(reduced)
            base = residual(reduced)
            jac = np.empty((len(base), len(active)))
            for j, i in enumerate(active):
                shifted, step = perturb(coordinates, i)
                jac[:, j] = (residual(shifted[active]) - base) / step
            return jac

        def recover(failure):
            # Freeze the anchor before backtracking. Never retarget the segment
            # to an unaccepted probe or let a frozen sensitivity perturbation win.
            enforce_frozen()
            anchor = min((r for r in records if r["status"] == "ok" and r["eligible_for_selection"]),
                         key=lambda r: r["objective"])
            start = np.asarray(anchor["coordinates"], dtype=float)
            failed = np.asarray(failure.record["coordinates"], dtype=float)
            event = {"failure_evaluation": failure.record["evaluation"],
                "failed_coordinates": failed.tolist(), "anchor_evaluation": anchor["evaluation"],
                "anchor_coordinates": start.tolist(), "anchor_objective": anchor["objective"],
                "backtracks": [], "status": "not_recovered"}
            recovery["events"].append(event)
            if recovery["restart_count"] >= _MAX_RESTARTS:
                event["status"] = "restart_limit_reached"
                return None
            for fraction in _BACKTRACK_FRACTIONS:
                coordinates = np.clip(start + fraction * (failed - start), lower, upper)
                # This is redundant mathematically, but makes the frozen-axis
                # invariant explicit even under floating-point interpolation.
                for i in set(range(2)) - set(active):
                    coordinates[i] = initial_coordinates[i]
                attempt = {"fraction": fraction, "coordinates": coordinates.tolist()}
                event["backtracks"].append(attempt)
                before = len(records)
                try:
                    candidate = trial(coordinates, "failure_backtrack")
                except _Stop as stop:
                    attempt["status"] = stop.status
                    event["status"] = stop.status
                    raise
                enforce_frozen()
                acceptable = (candidate["status"] == "ok" and candidate["eligible_for_selection"]
                              and candidate["objective"] < anchor["objective"])
                attempt.update(evaluation=candidate["evaluation"], cached=len(records) == before,
                    status=candidate["status"], strictly_improves_anchor=acceptable)
                if candidate["status"] == "ok":
                    attempt["objective"] = candidate["objective"]
                if acceptable:
                    recovery["restart_count"] += 1
                    event.update(status="restarted", restart_number=recovery["restart_count"],
                        restart_evaluation=candidate["evaluation"],
                        restart_coordinates=list(candidate["coordinates"]))
                    return np.asarray(candidate["coordinates"], dtype=float)[active]
            return None

        restart_at = initial_coordinates[active]
        while True:
            try:
                result = least_squares(residual, restart_at, jac=jacobian,
                    bounds=(lower[active], upper[active]), method="trf", max_nfev=max_evaluations,
                    ftol=1e-6, xtol=1e-5, gtol=1e-6)
                break
            except _SearchFailure as failure:
                if not recover_failed_trials:
                    raise
                restart_at = recover(failure)
                if restart_at is None:
                    raise
        optimizer.update(success=bool(result.success), scipy_status=int(result.status), message=str(result.message),
            active_parameters=[_NAMES[i] for i in active])
        status = "converged" if result.success else "not_converged"
        reason = str(result.message)
    except _Stop as stop:
        status, reason = stop.status, stop.reason

    enforce_frozen()
    eligible = [r for r in records if r["status"] == "ok" and r["eligible_for_selection"]]
    best = min(eligible, key=lambda r: r["objective"]) if eligible else None
    decision = None if best is None else {k: best[k] for k in (
        "evaluation", "lipophilicity", "clearance_scale", "objective", "objective_components",
        "clearance_l_h", "vss_l", "clearance_relative_error", "vss_relative_error", "curve_metrics")}
    initial_objective = records[0].get("objective") if records else None
    return {"status": status, "converged": bool(optimizer["success"]), "reason": reason,
        "decision": decision, "evaluations": records, "evaluation_count": len(records),
        "failed_evaluation_count": sum(r["status"] != "ok" for r in records),
        "max_evaluations": max_evaluations, "cache_hits": cache_hits,
        "initial_objective": initial_objective,
        "improved_over_initial": bool(best is not None and initial_objective is not None and best["objective"] < initial_objective),
        "sensitivity": sensitivity, "optimizer": optimizer, "failure_recovery": recovery,
        "targets": {"clearance_l_h": target_cl, "vss_l": target_vss, "logp_prior": prior},
        "bounds": {"lipophilicity": [prior - 2.5, prior + 2.5], "clearance_scale": [.25, 4.],
                   "interpretation": "Computational search bounds, not physiological confidence intervals"},
        "objective_spec": {"formula": "mean(ln(P/O)^2)+0.1*(ln(CL/CLtarget)/ln(2))^2+optional_0.1*(ln(Vss/Vsstarget)/ln(2))^2+0.01*((logMA-logP)/2.5)^2",
            "curve_weight": 1., "clearance_prior_weight": .1, "vss_prior_weight": .1 if target_vss is not None else None,
            "lipophilicity_prior_weight": .01, "pk_log_scale": math.log(2.), "lipophilicity_scale": 2.5,
            "sampling": "Linear interpolation at development observation times only; no extrapolation",
            "derived_endpoint_residuals": False, "weights_are_fixed_engineering_regularization": True,
            "official_OSP_algorithm_reproduction": False},
        "interpretation": "Development-curve calibration; soft PK priors do not establish anchor tolerance compliance or independent accuracy"}
