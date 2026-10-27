"""Choose one additional measurement for a requested PBPK endpoint.

This module only receives *model predictions*, never candidate observations.
Each ensemble row has equal probability. Its moments define a finite discrete
distribution, so covariance uses division by N (not an unbiased N-1 estimate).
The utility is the rank-one conditional covariance reduction in a joint linear
Gaussian approximation to log predictions with independent observation noise.
The ensemble is neither a posterior nor a calibrated clinical uncertainty set.

This is an application of established goal-oriented experimental design, not a
new statistical estimator. The caller must lock the action before revealing
the selected observation, and validate its actual predictive benefit separately.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


VERSION = "endpoint-evidence-design-1"
TIE_ATOL = 1e-12


def _positive_array(value: Any, name: str, dimensions: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values")
        result = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if result.ndim != dimensions or result.size == 0:
        raise ValueError(f"{name} must be a nonempty {dimensions}-dimensional array")
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _times(value: Any, name: str) -> np.ndarray:
    result = _positive_array(value, name, 1)
    if np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must be sorted and strictly increasing")
    return result


def _earliest_maximum(scores: np.ndarray) -> int:
    """Input times are sorted; absolute ties within 1e-12 prefer earlier time."""
    return int(np.flatnonzero(scores >= float(np.max(scores)) - TIE_ATOL)[0])


def _action(scores: np.ndarray, times: np.ndarray, reason: str,
            selected: int | None = None) -> dict[str, Any]:
    index = _earliest_maximum(scores) if selected is None else selected
    return {"selected_index": index, "selected_time_h": float(times[index]),
            "scores": scores.tolist(), "reason": reason}


def choose(candidate_times_h: Any, observed_times_h: Any,
           concentrations_mg_l: Any, endpoints: Mapping[str, Any], *,
           log_noise_sd: float = .1) -> dict[str, Any]:
    """Return endpoint-specific and fixed comparator measurement policies.

    concentrations_mg_l is N ensemble members by M candidate times. Each value
    in endpoints maps a nonempty goal name to an N-vector of positive endpoint
    predictions. Times must be positive, strictly increasing and disjoint.
    Duplicate ensemble members are allowed and retain their uniform weights;
    at least two rows are required, even when they happen to be identical.

    Zero endpoint variation causes a declared time-farthest fallback. The same
    fallback is used for the maximum-variance comparator when all candidate
    concentrations are constant. Uniform random is an expected action mixture,
    not a sampled action; this pure function uses no random numbers or files.
    """
    times = _times(candidate_times_h, "candidate_times_h")
    observed = _times(observed_times_h, "observed_times_h")
    if np.intersect1d(times, observed).size:
        raise ValueError("candidate and observed times must be disjoint")
    concentrations = _positive_array(concentrations_mg_l, "concentrations_mg_l", 2)
    members, candidates = concentrations.shape
    if members < 2 or candidates != len(times):
        raise ValueError("concentrations_mg_l needs at least two rows and one column per candidate")
    if not isinstance(endpoints, Mapping) or not endpoints:
        raise ValueError("endpoints must be a nonempty mapping")
    if isinstance(log_noise_sd, (bool, str, bytes)):
        raise ValueError("log_noise_sd must be finite and strictly positive")
    try:
        noise = float(log_noise_sd)
    except (ValueError, TypeError) as exc:
        raise ValueError("log_noise_sd must be finite and strictly positive") from exc
    if not np.isfinite(noise) or noise <= 0 or noise > np.sqrt(np.finfo(float).max):
        raise ValueError("log_noise_sd must be finite and strictly positive with finite variance")
    noise_variance = noise ** 2
    if noise_variance == 0:
        raise ValueError("log_noise_sd is too small to represent its variance")

    log_c = np.log(concentrations)
    centered_c = log_c - np.mean(log_c, axis=0)
    # Force exact zero for identical members instead of summation roundoff.
    constant_c = np.ptp(log_c, axis=0) == 0
    centered_c[:, constant_c] = 0.
    variance_c = np.mean(centered_c ** 2, axis=0)
    denominator = variance_c + noise_variance
    distance = np.min(np.abs(np.log(times)[:, None] - np.log(observed)[None, :]), axis=1)
    farthest = _earliest_maximum(distance)
    policies: dict[str, Any] = {
        "goal_directed": {},
        "time_farthest": _action(distance, times, "maximum_minimum_log_time_distance", farthest),
        "max_variance": _action(variance_c, times, "maximum_concentration_log_variance")
    }
    if np.all(constant_c):
        policies["max_variance"] = _action(variance_c, times, "no_concentration_disagreement", farthest)
        policies["max_variance"]["fallback"] = "time_farthest"

    goals: dict[str, Any] = {}
    for name, values in endpoints.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("endpoint goal names must be nonempty strings")
        endpoint = _positive_array(values, f"endpoints[{name}]", 1)
        if len(endpoint) != members:
            raise ValueError(f"endpoints[{name}] must contain one value per ensemble member")
        log_endpoint = np.log(endpoint)
        centered_endpoint = log_endpoint - np.mean(log_endpoint)
        constant_endpoint = bool(np.ptp(log_endpoint) == 0.)
        if constant_endpoint:
            centered_endpoint[:] = 0.
        variance_endpoint = float(np.mean(centered_endpoint ** 2))
        covariance = np.mean(centered_endpoint[:, None] * centered_c, axis=0)
        utility = covariance ** 2 / denominator
        conditional_variance = np.maximum(variance_endpoint - utility, 0.)
        goals[name] = {
            "variance_log": variance_endpoint, "covariance_log": covariance.tolist(),
            "utility_log_variance_reduction": utility.tolist(),
            "conditional_variance_log": conditional_variance.tolist(),
            "finite_spread": {"minimum": float(np.min(endpoint)), "maximum": float(np.max(endpoint)),
                              "log_span": float(np.ptp(log_endpoint))}
        }
        policies["goal_directed"][name] = _action(utility, times, "maximum_endpoint_log_variance_reduction")
        if constant_endpoint:
            policies["goal_directed"][name] = _action(utility, times, "no_endpoint_disagreement", farthest)
            policies["goal_directed"][name]["fallback"] = "time_farthest"

    policies["uniform_random"] = {
        "selected_index": None, "selected_time_h": None,
        "probabilities": np.full(candidates, 1. / candidates).tolist(),
        "expected_goal_utility": {name: float(np.mean(row["utility_log_variance_reduction"]))
                                  for name, row in goals.items()},
        "reason": "uniform_expected_action_not_a_random_draw"
    }
    return {
        "version": VERSION, "sample_count": members, "candidate_count": candidates,
        "candidate_times_h": times.tolist(), "observed_times_h": observed.tolist(),
        "log_noise_sd": noise, "concentration_log_variance": variance_c.tolist(),
        "concentration_finite_spread": {
            "minimum_mg_l": np.min(concentrations, axis=0).tolist(),
            "maximum_mg_l": np.max(concentrations, axis=0).tolist(),
            "log_span": np.ptp(log_c, axis=0).tolist()},
        "policies": policies, "goals": goals,
        "assumptions": [
            "Uniform finite ensemble weights; moments divide by N.",
            "Rank-one conditional covariance of a joint linear Gaussian log approximation.",
            "Independent homoscedastic log observation noise with user-specified standard deviation.",
            "Finite ensemble disagreement is not posterior uncertainty or a clinical confidence interval.",
            "The utility is an approximation, not a guarantee of reduced observed endpoint error.",
            "Candidate observations must remain hidden until the action is locked.",
            "Absolute score ties within 1e-12 select the earlier candidate time.",
            "Constant endpoint or universally constant concentrations use the time-farthest fallback."
        ]
    }
