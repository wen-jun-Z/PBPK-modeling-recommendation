"""Finite-ensemble expected endpoint variance reduction for one measurement.

Classical Bayesian-design utility is evaluated within an engineering ensemble:
U = Var(log E) - E_Y[Var(log E | Y)], with a uniform model prior and a Gaussian
log-observation likelihood. Neither this chosen prior nor the finite ensemble
is asserted to be a posterior learned from clinical data. Gauss-Hermite
quadrature integrates each component of the predictive observation mixture.

The implementation receives model predictions only. The caller must lock the
chosen sampling time before reading any held-out concentration or endpoint.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.special import logsumexp, roots_hermitenorm

from . import endpoint_evidence_design as linear


VERSION = "nonlinear-endpoint-evidence-design-1"
QUADRATURE_SEQUENCE = (7, 15, 31, 63, 127)
FLOAT_BOUND_RTOL = 1e-10
FLOAT_BOUND_ATOL = 1e-14


def _integrate_candidate(log_c: np.ndarray, centered_endpoints: np.ndarray,
                         sigma: float, nodes: int) -> dict[str, np.ndarray]:
    """Moments for all goals, under N equally weighted normal components.

    scipy's probabilists' Hermite weights integrate exp(-x*x/2); dividing by
    sqrt(2*pi) converts these weights to a standard-normal expectation.
    """
    abscissae, weights = roots_hermitenorm(nodes)
    weights = weights / np.sqrt(2 * np.pi)
    n = len(log_c)
    # Form standardized differences directly to avoid cancellation when sigma
    # is much smaller than the absolute value of the log concentration.
    with np.errstate(over="ignore", invalid="raise", divide="raise"):
        relative_centers = (log_c[:, None] - log_c[None, :]) / sigma
        standardized = relative_centers[:, None, :] + abscissae[None, :, None]
        log_likelihood = -.5 * standardized ** 2
    log_normalizer = logsumexp(log_likelihood, axis=-1, keepdims=True)
    conditioned_weights = np.exp(log_likelihood - log_normalizer).reshape(n * nodes, n)
    integration_weights = np.tile(weights / n, n)
    means = conditioned_weights @ centered_endpoints
    seconds = conditioned_weights @ (centered_endpoints ** 2)
    variances = seconds - means ** 2
    numerical_tolerance = FLOAT_BOUND_ATOL + FLOAT_BOUND_RTOL * np.maximum(seconds, means ** 2)
    if np.any(variances < -numerical_tolerance):
        raise ArithmeticError("negative conditional variance exceeds roundoff tolerance")
    variances = np.maximum(variances, 0.)
    mean = integration_weights @ means
    mean_squared = integration_weights @ (means ** 2)
    return {"expected_conditional_variance": integration_weights @ variances,
            "expected_conditional_mean": mean,
            "variance_conditional_mean": np.maximum(mean_squared - mean ** 2, 0.),
            "expected_conditional_second_moment": integration_weights @ seconds}


def choose(candidate_times_h: Any, observed_times_h: Any,
           concentrations_mg_l: Any, endpoints: Mapping[str, Any], *,
           log_noise_sd: float = .1, quadrature_nodes: int = 31) -> dict[str, Any]:
    """Return nonlinear goal choices plus unchanged linear/fixed comparators.

    Validation and comparator definitions are delegated to the frozen linear
    module. ``policies.goal_directed`` retains the linear comparator;
    ``policies.nonlinear_goal_directed`` is this module's new policy.

    Integration starts with 31 nodes per ensemble member by default. Allowed
    starting orders are 7, 15, 31, 63 and 127. If a utility
    violates [0, prior variance] beyond explicit floating-point tolerance, use
    successively higher orders from that sequence. Unresolved violations raise rather
    than silently clipping a large quadrature error. Only roundoff-sized bound
    excursions are clipped and both raw and adjusted utilities are returned.
    Bounds passing is not a convergence proof: law-of-total-variance and mean
    residuals are reported for every candidate; valid bounds are not a test
    of quadrature convergence or policy-ranking stability.
    """
    result = linear.choose(candidate_times_h, observed_times_h, concentrations_mg_l,
                           endpoints, log_noise_sd=log_noise_sd)
    if (isinstance(quadrature_nodes, bool) or
            not isinstance(quadrature_nodes, (int, np.integer)) or
            quadrature_nodes not in QUADRATURE_SEQUENCE):
        raise ValueError(f"quadrature_nodes must be one of {QUADRATURE_SEQUENCE}")
    sequence = tuple(value for value in QUADRATURE_SEQUENCE if value >= quadrature_nodes)
    result["comparator_version"] = result["version"]
    result["version"] = VERSION
    times = np.asarray(result["candidate_times_h"])
    log_c = np.log(np.asarray(concentrations_mg_l, dtype=float))
    names = list(endpoints)
    log_endpoints = np.column_stack([np.log(np.asarray(endpoints[name], dtype=float)) for name in names])
    centered = log_endpoints - np.mean(log_endpoints, axis=0)
    constant = np.ptp(log_endpoints, axis=0) == 0
    centered[:, constant] = 0.
    prior = np.mean(centered ** 2, axis=0)
    bound_tolerance = FLOAT_BOUND_ATOL + FLOAT_BOUND_RTOL * prior
    goals = {name: {
        "variance_log": float(prior[g]), "raw_utility_log_variance_reduction": [],
        "utility_log_variance_reduction": [], "expected_conditional_variance_log": [],
        "finite_spread": result["goals"][name]["finite_spread"],
        "quadrature": []} for g, name in enumerate(names)}

    for candidate in range(len(times)):
        attempts: list[dict[str, Any]] = []
        for nodes in sequence:
            integrals = _integrate_candidate(log_c[:, candidate], centered,
                                            float(log_noise_sd), nodes)
            remaining = integrals["expected_conditional_variance"]
            raw_utility = prior - remaining
            valid = (np.isfinite(raw_utility).all() and
                     np.all(raw_utility >= -bound_tolerance) and
                     np.all(raw_utility <= prior + bound_tolerance))
            attempts.append({"nodes": nodes, "utility_bounds_passed": bool(valid)})
            if valid:
                break
        else:
            raise ArithmeticError(f"quadrature utility bounds failed at candidate {candidate} after {list(sequence)}")

        adjusted = np.clip(raw_utility, 0., prior)
        closure = remaining + integrals["variance_conditional_mean"] - prior
        for g, name in enumerate(names):
            row = goals[name]
            row["raw_utility_log_variance_reduction"].append(float(raw_utility[g]))
            row["utility_log_variance_reduction"].append(float(adjusted[g]))
            row["expected_conditional_variance_log"].append(float(remaining[g]))
            row["quadrature"].append({
                "nodes_per_member": nodes, "attempts": [dict(attempt) for attempt in attempts],
                "utility_roundoff_adjustment": float(adjusted[g] - raw_utility[g]),
                "utility_bound_tolerance": float(bound_tolerance[g]),
                "prior_mean_closure_error_log": float(integrals["expected_conditional_mean"][g]),
                "total_variance_closure_error_log2": float(closure[g]),
                "second_moment_closure_error_log2": float(integrals["expected_conditional_second_moment"][g] - prior[g]),
                "convergence_verified": False})

    policies = result["policies"]
    policies["nonlinear_goal_directed"] = {}
    farthest = policies["time_farthest"]["selected_index"]
    for g, name in enumerate(names):
        scores = np.asarray(goals[name]["utility_log_variance_reduction"])
        policy = linear._action(scores, times, "maximum_expected_endpoint_log_variance_reduction")
        if constant[g]:
            policy = linear._action(scores, times, "no_endpoint_disagreement", farthest)
            policy["fallback"] = "time_farthest"
        policies["nonlinear_goal_directed"][name] = policy
    policies["uniform_random"]["expected_nonlinear_goal_utility"] = {
        name: float(np.mean(row["utility_log_variance_reduction"])) for name, row in goals.items()}
    result["nonlinear_goals"] = goals
    result["quadrature_starting_nodes_per_member"] = int(quadrature_nodes)
    result["nonlinear_assumptions"] = [
        "Classical expected conditional variance reduction under a finite uniform engineering prior.",
        "Independent Gaussian log-observation likelihood with prescribed noise standard deviation.",
        "Mixture expectation integrates each model component with deterministic Gauss-Hermite quadrature.",
        "Prespecified starting nodes per member; upward refinements only for utility bound violations.",
        "Passing utility bounds does not establish quadrature convergence; moment residuals are reported.",
        "The engineering ensemble is not a clinical posterior or a calibrated clinical confidence interval.",
        "Candidate observation values are absent from this API and must stay hidden until action lock.",
        "Expected variance reduction does not guarantee reduced observed error or correct model structure."]
    return result
