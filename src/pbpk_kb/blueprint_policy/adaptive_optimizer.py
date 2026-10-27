"""Budgeted, deterministic calibration of two normalized PBPK parameters.

The optimizer has no file, clinical-target, or native-runtime access. A caller
supplies immutable native trial records and the *development* concentrations.
TRF follows scipy.optimize.least_squares, including its bounded trust-region
method, but counts Jacobian probes and failed simulations against the same hard
budget. Absolute finite differences avoid infinitesimal native-solver probes.

Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
This is a numerical optimizer, not evidence of clinical superiority.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real
import math

import numpy as np
from scipy.optimize import least_squares

VERSION = "evidence-adaptive-optimizer-1"
PARAMETERS = ("lipophilicity", "clearance_scale")
FINITE_DIFFERENCE_STEP = .01
MINIMUM_PATTERN_STEP = 1e-3
FAILURE_RESIDUAL = 1e6


class BudgetExhausted(Exception):
    """No more native trial requests are allowed in this fit."""


class _InvalidJacobian(Exception):
    """A physics failure prevents estimating a trustworthy local Jacobian."""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _allowed(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("allowed_parameters must be a sequence")
    values = tuple(values)
    if any(x not in PARAMETERS for x in values) or len(set(values)) != len(values):
        raise ValueError("Unknown or duplicate allowed parameter")
    return tuple(name for name in PARAMETERS if name in values)


def residual_vector(row, observations, prior, prior_weight=.1):
    """Return residuals whose squared norm is the existing native objective.

    Mean squared log concentration error + clearance evidence regularization +
    weak effective-lipophilicity regularization. No robust-loss reweighting is
    applied, so comparisons isolate the optimizer rather than changing the loss.
    A mismatch with the caller's persisted objective is a hard integration error.
    """
    observed = np.asarray(observations, dtype=float)
    predicted = np.asarray(row["predicted_at_observed_times"], dtype=float)
    if (observed.ndim != 1 or not observed.size or predicted.shape != observed.shape
            or not np.isfinite(observed).all() or not np.isfinite(predicted).all()
            or np.any(observed <= 0) or np.any(predicted <= 0)):
        raise ValueError("Development predictions/observations must be matched, finite and positive")
    lip = _finite(row["lipophilicity"], "lipophilicity")
    scale = _finite(row["clearance_scale"], "clearance_scale")
    if scale <= 0:
        raise ValueError("clearance_scale must be positive")
    weight = _finite(prior_weight, "prior_weight")
    if weight < 0:
        raise ValueError("prior_weight must be nonnegative")
    prior = _finite(prior, "prior")
    if "residual_vector" in row:
        # The native adapter may explicitly freeze another development loss,
        # e.g. robust data residuals with quadratic evidence priors. Keeping it
        # in the immutable trial makes objective changes auditable and avoids
        # accidentally changing only the optimizer's view of that same trial.
        residuals = np.asarray(row["residual_vector"], dtype=float)
        if residuals.shape != (observed.size + 2,) or not np.isfinite(residuals).all():
            raise ValueError("Explicit residual_vector must have n_observations + 2 finite entries")
    else:
        residuals = np.concatenate(((np.log(predicted) - np.log(observed)) / math.sqrt(observed.size),
            [math.sqrt(weight) * math.log(scale) / math.log(2), .1 * (lip - prior) / 2.5]))
    objective = _finite(row["objective"], "objective")
    if not math.isclose(float(residuals @ residuals), objective, rel_tol=1e-9, abs_tol=1e-11):
        raise ValueError("Native objective differs from the optimizer residual objective")
    return residuals


def optimize(evaluate, prior, observations, *, allowed_parameters=PARAMETERS,
             max_evaluations=24, prior_weight=.1, method="trust_region",
             initial_design=4, budget_exceptions=()):
    """Optimize only allowed axes, with a hard count of unique callback calls.

    ``evaluate(lipophilicity, clearance_scale, purpose)`` must return a native
    row with status, evaluation, parameter values, eligible_for_selection and,
    for status='ok', objective and predicted_at_observed_times. Returned rows
    are not modified, so immutable trial/source hashes remain valid.

    ``budget_exceptions`` lets a native adapter interrupt before a request when
    native retries already consumed its stricter execution budget. Other errors
    propagate rather than becoming scientific failures. Replaying the same
    callback cache resumes the deterministic algorithm without hidden warm starts.

    ``pattern`` is a direct-search baseline with the same loss, bounds and call
    budget. Its initial_design argument is intentionally ignored: its initial
    exploration is its complete eight-direction poll, as in the old algorithm.
    """
    if not callable(evaluate):
        raise ValueError("evaluate must be callable")
    prior = _finite(prior, "prior")
    if not prior - 2.5 < prior < prior + 2.5:
        raise ValueError("Evidence lipophilicity bounds must be representable")
    weight = _finite(prior_weight, "prior_weight")
    if weight < 0:
        raise ValueError("prior_weight must be nonnegative")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if isinstance(initial_design, bool) or not isinstance(initial_design, Integral) or not 0 <= initial_design <= 4:
        raise ValueError("initial_design must be an integer between zero and four")
    if method not in ("trust_region", "pattern"):
        raise ValueError("Unknown optimizer method")
    allowed = _allowed(allowed_parameters)
    active = [PARAMETERS.index(name) for name in allowed]
    observed = np.asarray(observations, dtype=float)
    if observed.ndim != 1 or not observed.size or not np.isfinite(observed).all() or np.any(observed <= 0):
        raise ValueError("Development observations must be a positive finite vector")
    if not isinstance(budget_exceptions, tuple) or any(
        not isinstance(exc, type) or not issubclass(exc, Exception) for exc in budget_exceptions
    ):
        raise ValueError("budget_exceptions must be a tuple of Exception classes")

    records, cache, polls, jacobians, scipy_runs = [], {}, [], [], []
    cache_hits = 0
    termination, reason, converged = "budget_exhausted", "Native trial budget exhausted", False
    fallback_used = False

    def full_point(x):
        z = np.zeros(2)
        z[active] = np.asarray(x, dtype=float)
        if not np.isfinite(z).all() or np.any(np.abs(z) > 1 + 1e-12):
            raise ValueError("Optimizer requested a point outside normalized bounds")
        return np.clip(z, -1., 1.)

    def coordinates(row):
        return np.asarray([(row["lipophilicity"] - prior) / 2.5,
            math.log(row["clearance_scale"]) / math.log(4)])[active]

    def trial(x, purpose):
        nonlocal cache_hits
        z = full_point(x)
        lip, scale = prior + 2.5 * float(z[0]), math.exp(math.log(4) * float(z[1]))
        key = (lip.hex(), scale.hex())
        if key in cache:
            cache_hits += 1
            return cache[key]
        if len(records) >= max_evaluations:
            raise BudgetExhausted
        row = evaluate(lip, scale, purpose)
        if row.get("lipophilicity") != lip or row.get("clearance_scale") != scale:
            raise ValueError("Native trial returned different calibration parameters")
        if row.get("status") == "ok":
            residual_vector(row, observed, prior, weight)
        elif row.get("eligible_for_selection"):
            raise ValueError("A failed native trial cannot be eligible for selection")
        if row.get("status") == "ok" and not row.get("eligible_for_selection"):
            raise ValueError("All successful adaptive-optimizer trials must be selectable")
        records.append(row)
        cache[key] = row
        return row

    def best():
        return min((r for r in records if r.get("status") == "ok" and r.get("eligible_for_selection")),
            key=lambda r: (r["objective"], r["evaluation"]), default=None)

    def residuals(x, purpose="trust_region_step"):
        row = trial(x, purpose)
        if row.get("status") != "ok":
            return np.full(observed.size + 2, FAILURE_RESIDUAL)
        return residual_vector(row, observed, prior, weight)

    def jacobian(x):
        center = trial(x, "trust_region_jacobian_center")
        if center.get("status") != "ok":
            raise _InvalidJacobian("Current trust-region point has invalid native physics")
        base = residual_vector(center, observed, prior, weight)
        columns, used, failed = [], [], []
        for index, parameter in enumerate(allowed):
            # Prefer the direction with more room. If physics fails there, try
            # the opposite direction; never interpret a failure as insensitivity.
            directions = (1., -1.) if x[index] <= 0 else (-1., 1.)
            derivative = None
            for sign in directions:
                delta = sign * min(FINITE_DIFFERENCE_STEP, 1 - sign * x[index])
                if abs(delta) < 1e-12:
                    continue
                probe = np.array(x, copy=True)
                probe[index] += delta
                row = trial(probe, f"trust_region_jacobian_{parameter}_{'plus' if sign > 0 else 'minus'}")
                used.append(row["evaluation"])
                if row.get("status") == "ok":
                    derivative = (residual_vector(row, observed, prior, weight) - base) / delta
                    break
                failed.append(row["evaluation"])
            if derivative is None:
                jacobians.append({"center": center["evaluation"], "probes": used,
                    "failed_probes": failed, "status": "failed"})
                raise _InvalidJacobian(f"Both finite-difference directions failed for {parameter}")
            columns.append(derivative)
        matrix = np.column_stack(columns)
        jacobians.append({"center": center["evaluation"], "probes": used,
            "failed_probes": failed, "status": "ok", "rank": int(np.linalg.matrix_rank(matrix)),
            "column_norms": np.linalg.norm(matrix, axis=0).tolist()})
        return matrix

    def pattern(start_step=.25):
        """Complete positive-spanning polls, without convergence after failures."""
        step = start_step
        directions = [np.eye(len(active))[i] * sign for i in range(len(active)) for sign in (1., -1.)]
        if len(active) == 2:
            directions += [np.array([a, b]) for a, b in ((1., 1.), (1., -1.), (-1., 1.), (-1., -1.))]
        while step >= MINIMUM_PATTERN_STEP:
            selected = best()
            anchor = np.zeros(len(active)) if selected is None else coordinates(selected)
            previous = None if selected is None else selected["objective"]
            count, failures = len(records), 0
            for direction in directions:
                row = trial(np.clip(anchor + step * direction, -1., 1.), "pattern_poll")
                failures += row.get("status") != "ok"
            selected = best()
            improved = selected is not None and (previous is None or selected["objective"] < previous - 1e-12)
            polls.append({"step": step, "new_parameter_points": len(records) - count,
                "improved": improved, "failed_probes": failures})
            if not improved:
                step *= .5
        return bool(best()) and bool(polls) and polls[-1]["failed_probes"] == 0

    try:
        trial(np.zeros(len(active)), "evidence_prior")
        if not active:
            termination, reason = "fixed_evidence_only", "No permitted parameters; evidence point retained"
        elif method == "pattern":
            converged = pattern()
            termination = "converged" if converged else "no_feasible_improvement"
            reason = "Pattern step tolerance reached" if converged else "Final native poll failed; convergence unestablished"
        else:
            # Broad, fixed probes make no use of labels or earlier-drug results.
            designs = ([np.array([.65]), np.array([-.65])] if len(active) == 1 else
                [np.array(pair) for pair in ((.65, .65), (-.65, -.65), (.65, -.65), (-.65, .65))])
            for point in designs[:initial_design]:
                trial(point, "trust_region_initial_design")
            selected = best()
            if selected is None:
                raise _InvalidJacobian("No feasible evidence or broad-design point")
            start = np.clip(coordinates(selected), -1 + 1e-10, 1 - 1e-10)
            result = least_squares(residuals, start, jac=jacobian, bounds=(-1., 1.),
                method="trf", loss="linear", x_scale=1., tr_solver="exact",
                ftol=1e-6, xtol=1e-5, gtol=1e-5, max_nfev=max_evaluations)
            final = trial(result.x, "trust_region_final")
            selected = best()
            # A derivative probe can outperform the accepted iterate. Keep the
            # actual best native point, but do not call that different point a
            # converged optimum on the basis of SciPy's termination elsewhere.
            same_point = selected is not None and final["evaluation"] == selected["evaluation"]
            converged = bool(result.success and final.get("status") == "ok" and same_point)
            scipy_runs.append({"status": int(result.status), "message": str(result.message),
                "nfev": int(result.nfev), "njev": int(result.njev),
                "optimality": float(result.optimality), "final_evaluation": final["evaluation"],
                "selected_equals_final_iterate": same_point})
            termination = "converged" if converged else "best_native_trial"
            reason = str(result.message) if converged else "Best native trial retained; its convergence is not established"
    except _InvalidJacobian as exc:
        # A native-domain boundary must not produce a spurious zero-gradient
        # success. Direct search can still use the remaining call budget.
        fallback_used = True
        reason = str(exc)
        try:
            converged = pattern()
            termination = "converged" if converged else "no_feasible_improvement"
        except (BudgetExhausted, *budget_exceptions):
            termination, converged = "budget_exhausted", False
            reason += "; remaining direct-search budget exhausted"
    except (BudgetExhausted, *budget_exceptions):
        termination, converged = "budget_exhausted", False

    selected = best()
    return {"status": termination if selected else "no_feasible_point", "reason": reason,
        "converged": converged, "eligible": selected is not None,
        "decision": deepcopy(selected), "evaluations": records, "evaluation_count": len(records),
        "failed_evaluation_count": sum(row.get("status") != "ok" for row in records),
        "cache_hits": cache_hits, "active_parameters": list(allowed),
        "screening": {"status": "not_requested", "selected_parameters": list(allowed),
            "reason": "No hard axis screening from a single local Jacobian"},
        "optimizer": {"version": VERSION, "method": method, "native_trial_budget": int(max_evaluations),
            "initial_design_requested": int(initial_design) if method == "trust_region" else 0,
            "finite_difference_normalized_step": FINITE_DIFFERENCE_STEP,
            "prior_weight": weight, "residual_loss": "linear; optional explicit native residual transformation",
            "numerical_convergence_only": True,
            "active_parameters": list(allowed),
            "frozen_at_evidence_parameters": [p for p in PARAMETERS if p not in allowed],
            "jacobians": jacobians, "scipy_runs": scipy_runs, "polls": polls,
            "physics_failure_fallback_used": fallback_used,
            "budget_includes": ["initial design", "Jacobian probes", "failed physics", "search trials"],
            "reference": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html"}}
