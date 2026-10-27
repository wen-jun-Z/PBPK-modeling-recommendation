"""Bounded numerical matching of independent human PK evidence anchors.

This module receives only simulated CL/Vss and evidence targets. It never sees
clinical evaluation curves. Matching an anchor is parameter construction, not
external validation of a PBPK model or proof of fitness for a clinical decision.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from numbers import Integral, Real
from typing import Any


_SUCCESS_STATUSES = {"ok", "success", "completed", "predicted"}
_CLEARANCE_TOLERANCE = 0.05
_VSS_TOLERANCE = 0.10
_DIFFERENCE_STEP = 0.03


class _StopOptimization(Exception):
    def __init__(self, status: str, reason: str):
        self.status = status
        self.reason = reason
        super().__init__(reason)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


def optimize_pk_anchors(
    evaluate: Callable[[float, float], Mapping[str, Any]],
    initial_lipophilicity: float,
    target_clearance_l_h: float,
    target_vss_l: float | None = None,
    max_evaluations: int = 24,
) -> dict[str, Any]:
    """Fit effective lipophilicity and a clearance multiplier to PK anchors.

    ``evaluate(lipophilicity, clearance_scale)`` must return a successful
    ``status`` (``ok``, ``success``, ``completed`` or ``predicted``) and positive,
    finite ``clearance_l_h`` and, when targeted, ``vss_l``. A failed simulation
    or missing terminal endpoint stops the fit with its reason; no fabricated
    residual is substituted. With no Vss target, lipophilicity stays fixed.

    Real callback calls, including finite differences, are bounded by
    ``max_evaluations`` (1 through 24). Repeated coordinates use a cache. A ``decision`` is
    returned only when CL is within 5% and requested Vss within 10%. Otherwise
    ``decision`` is None and ``best_attempt`` remains explicitly unaccepted.

    Effective lipophilicity is bounded to its initial value +/- 2.5 log units;
    clearance scale is in [0.25, 4]. These are computational search bounds,
    not evidence intervals or physiological confidence intervals.
    """
    if not callable(evaluate):
        raise ValueError("evaluate must be callable")
    initial = _number(initial_lipophilicity, "initial_lipophilicity")
    target_cl = _number(target_clearance_l_h, "target_clearance_l_h", positive=True)
    target_vss = (None if target_vss_l is None else
                  _number(target_vss_l, "target_vss_l", positive=True))
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral)
            or not 1 <= max_evaluations <= 24):
        raise ValueError("max_evaluations must be an integer from 1 through 24")
    max_evaluations = int(max_evaluations)

    import numpy as np
    from scipy.optimize import least_squares

    with_vss = target_vss is not None
    if with_vss and not initial - 2.5 < initial < initial + 2.5:
        raise ValueError("initial_lipophilicity is too large to represent its search bounds")
    lower = np.array([initial - 2.5, math.log(0.25)] if with_vss else [math.log(0.25)])
    upper = np.array([initial + 2.5, math.log(4.0)] if with_vss else [math.log(4.0)])
    start = np.array([initial, 0.0] if with_vss else [0.0])
    evaluations: list[dict[str, Any]] = []
    cache: dict[tuple[float, float], dict[str, Any]] = {}
    best: dict[str, Any] | None = None
    cache_hits = 0
    optimizer_info: dict[str, Any] = {
        "method": "bounded_trust_region_least_squares",
        "objective": "sum_squared_log_simulated_over_evidence_anchor",
        "finite_difference_step": _DIFFERENCE_STEP,
        "variables": ["effective_lipophilicity", "log_clearance_scale"] if with_vss else ["log_clearance_scale"],
    }

    def residual(coordinates):
        nonlocal best, cache_hits
        lipophilicity = float(coordinates[0]) if with_vss else initial
        log_scale = float(coordinates[-1])
        clearance_scale = float(np.clip(math.exp(log_scale), 0.25, 4.0))
        key = (lipophilicity, clearance_scale)
        if key in cache:
            cache_hits += 1
            record = cache[key]
        else:
            if len(evaluations) >= max_evaluations:
                raise _StopOptimization("budget_exhausted", "Real simulation evaluation budget exhausted")
            record = {"evaluation": len(evaluations) + 1,
                      "lipophilicity": lipophilicity, "clearance_scale": clearance_scale,
                      "status": "evaluation_failed", "accepted": False}
            evaluations.append(record)
            cache[key] = record
            try:
                output = evaluate(lipophilicity, clearance_scale)
                if not isinstance(output, Mapping):
                    raise ValueError("evaluate must return a mapping")
                callback_status = output.get("status")
                record["callback_status"] = callback_status if isinstance(callback_status, str) else None
                if callback_status not in _SUCCESS_STATUSES:
                    detail = output.get("reason", output.get("errors", output.get("detail", "")))
                    raise ValueError(f"Unsuccessful simulation status {callback_status!r}: {detail}")
                simulated_cl = _number(output.get("clearance_l_h"), "simulated clearance_l_h", positive=True)
                simulated_vss = (None if output.get("vss_l") is None and not with_vss else
                                 _number(output.get("vss_l"), "simulated vss_l", positive=True))
                log_residuals = [math.log(simulated_cl) - math.log(target_cl)]
                if with_vss:
                    log_residuals.append(math.log(simulated_vss) - math.log(target_vss))
                cl_error = abs(simulated_cl / target_cl - 1.0)
                vss_error = None if not with_vss else abs(simulated_vss / target_vss - 1.0)
                accepted = cl_error <= _CLEARANCE_TOLERANCE and (not with_vss or vss_error <= _VSS_TOLERANCE)
                record.update(status="ok", clearance_l_h=simulated_cl, vss_l=simulated_vss,
                              clearance_relative_error=cl_error, vss_relative_error=vss_error,
                              log_residuals=log_residuals,
                              objective=sum(item * item for item in log_residuals), accepted=accepted)
            except Exception as exc:
                record["reason"] = f"{type(exc).__name__}: {exc}"
                raise _StopOptimization("evaluation_failed", record["reason"]) from exc
            if best is None or record["objective"] < best["objective"] or record["accepted"]:
                best = dict(record)
        if record["accepted"]:
            raise _StopOptimization("converged", "All requested PK anchors meet the declared relative-error tolerances")
        return np.asarray(record["log_residuals"], dtype=float)

    def jacobian(coordinates):
        base = residual(coordinates)
        jac = np.empty((len(base), len(coordinates)))
        for column in range(len(coordinates)):
            shifted = np.array(coordinates, copy=True)
            step = _DIFFERENCE_STEP
            if shifted[column] + step > upper[column]:
                step = -step
            shifted[column] = float(np.clip(shifted[column] + step, lower[column], upper[column]))
            actual_step = shifted[column] - coordinates[column]
            if actual_step == 0:
                raise _StopOptimization("not_converged", "No representable finite-difference step inside the search bounds")
            jac[:, column] = (residual(shifted) - base) / actual_step
        return jac

    try:
        result = least_squares(residual, start, jac=jacobian, bounds=(lower, upper),
                               method="trf", max_nfev=max_evaluations,
                               ftol=1e-6, xtol=1e-5, gtol=1e-6)
        status = "not_converged"
        reason = "Optimizer stopped before all requested PK anchors met their tolerances"
        optimizer_info.update(success=bool(result.success), status=int(result.status), message=str(result.message))
    except _StopOptimization as stop:
        status, reason = stop.status, stop.reason

    converged = status == "converged"
    return {
        "status": status, "converged": converged, "reason": reason,
        "decision": dict(best) if converged and best is not None else None,
        "best_attempt": best, "evaluations": evaluations,
        "evaluation_count": len(evaluations), "cache_hits": cache_hits,
        "max_evaluations": max_evaluations,
        "targets": {"clearance_l_h": target_cl, "vss_l": target_vss},
        "target_tolerances": {"clearance_relative_error": _CLEARANCE_TOLERANCE,
                              "vss_relative_error": _VSS_TOLERANCE if with_vss else None},
        "bounds": {"lipophilicity": [initial - 2.5, initial + 2.5] if with_vss else [initial, initial],
                   "clearance_scale": [0.25, 4.0],
                   "interpretation": "Computational search bounds, not physiological confidence intervals"},
        "optimizer": optimizer_info,
    }
