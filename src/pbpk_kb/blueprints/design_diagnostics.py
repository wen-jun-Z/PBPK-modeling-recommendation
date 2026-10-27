"""Local sensitivity diagnostics and bounded scenario-call accounting.

This module does not construct observations, simulate models, or import model
runtimes. Numerical rank is a local diagnostic, not proof of global or practical
identifiability. The caller must construct the Jacobian from actual calibration
observations, with a documented observation mapping and parameter scaling.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable
from numbers import Integral, Real
from typing import Any, TypeVar

_T = TypeVar("_T")
_PHASES = ("search", "diagnostics")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def analyze_identifiability(
    parameter_ids: Iterable[str],
    observation_jacobian: Any,
    *,
    relative_threshold: float = 1e-6,
    max_free_parameters: int = 6,
) -> dict[str, Any]:
    """Diagnose a real calibration-observation Jacobian on CPU.

    Rows must correspond to actual calibration observations, never desired target
    outputs or synthetic observations added to increase rank. Their provenance
    cannot be inferred from a numerical matrix; the calling adapter must verify
    it. Columns follow ``parameter_ids``. Sensitivities should use a documented
    dimensionless parameter scale and observation/error weighting; this function
    does not silently normalize columns or assume an observation noise model.

    The selected subset is locally independent *conditional on fixing all other
    parameters*, with deterministic greedy column pivoting. It does not make the
    omitted parameters globally identifiable or eliminate their uncertainty.
    """
    if isinstance(parameter_ids, (str, bytes)):
        raise ValueError("parameter_ids must be a nonempty sequence of identifiers")
    try:
        ids = list(parameter_ids)
    except TypeError as exc:
        raise ValueError("parameter_ids must be iterable") from exc
    if not ids or any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("parameter_ids must contain nonempty string identifiers")
    if len(set(ids)) != len(ids):
        raise ValueError("parameter_ids must be unique")
    if (
        isinstance(relative_threshold, bool)
        or not isinstance(relative_threshold, Real)
        or not math.isfinite(relative_threshold)
        or not 0 < relative_threshold < 1
    ):
        raise ValueError("relative_threshold must be finite and strictly between 0 and 1")
    relative_threshold = float(relative_threshold)
    cap = _positive_integer(max_free_parameters, "max_free_parameters")

    # NumPy is intentionally imported only when this numerical function is used.
    import numpy as np

    try:
        raw = np.asarray(observation_jacobian)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_jacobian must be a rectangular real matrix") from exc
    if raw.ndim != 2 or not all(raw.shape):
        raise ValueError("observation_jacobian must be a nonempty two-dimensional matrix")
    if raw.shape[1] != len(ids):
        raise ValueError("Jacobian column count must equal parameter_ids length")
    if raw.dtype.kind not in "iuf":
        raise ValueError("observation_jacobian must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = raw.astype(float, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("observation_jacobian must contain only finite values")

    # Uniform scaling prevents overflow without changing the rank or pivot order.
    scale = float(np.max(np.abs(matrix)))
    scaled = matrix / scale if scale else matrix.copy()
    try:
        singular_scaled = np.linalg.svd(scaled, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Jacobian singular-value decomposition did not converge") from exc
    cutoff = float(singular_scaled[0]) * relative_threshold
    rank = int(np.count_nonzero(singular_scaled > cutoff))
    norms_scaled = np.linalg.norm(scaled, axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        singular_values = singular_scaled * scale
        column_norms = norms_scaled * scale
    if not np.isfinite(singular_values).all() or not np.isfinite(column_norms).all():
        raise ValueError("Jacobian magnitude is too large; provide explicitly scaled sensitivities")

    residual = scaled.copy()
    selected: list[int] = []
    for _ in range(min(rank, cap)):
        residual_norms = np.linalg.norm(residual, axis=0)
        if selected:
            residual_norms[selected] = -1.0
        pivot = int(np.argmax(residual_norms))
        pivot_norm = float(residual_norms[pivot])
        if pivot_norm <= cutoff:
            break
        trial = selected + [pivot]
        # Keep the subset independently resolvable at the full-matrix cutoff.
        trial_singular = np.linalg.svd(scaled[:, trial], compute_uv=False)
        if float(trial_singular[-1]) <= cutoff:
            break
        selected.append(pivot)
        direction = residual[:, pivot] / pivot_norm
        for _pass in range(2):
            residual -= np.outer(direction, direction @ residual)
        residual[:, pivot] = 0.0

    degenerate = [i for i, norm in enumerate(norms_scaled) if float(norm) <= cutoff]
    degenerate_set = set(degenerate)
    correlated_pairs: list[dict[str, Any]] = []
    cosine_threshold = 0.95
    for first in range(len(ids)):
        if first in degenerate_set:
            continue
        unit_first = scaled[:, first] / norms_scaled[first]
        for second in range(first + 1, len(ids)):
            if second in degenerate_set:
                continue
            cosine = float(np.clip(unit_first @ (scaled[:, second] / norms_scaled[second]), -1, 1))
            if abs(cosine) >= cosine_threshold:
                correlated_pairs.append({
                    "parameter_a": ids[first],
                    "parameter_b": ids[second],
                    "cosine_similarity": cosine,
                })

    selected_set = set(selected)
    return {
        "diagnostic_kind": "local_sensitivity_rank",
        "local_not_global": True,
        "source_requirement": "actual_calibration_observations_only",
        "observation_source_verified": False,
        "subset_interpretation": "conditional_on_unselected_parameters_fixed",
        "scaling_requirement": "document_parameter_scaling_and_observation_weighting",
        "observation_count": int(matrix.shape[0]),
        "parameter_count": len(ids),
        "parameter_ids": ids,
        "rank": rank,
        "relative_threshold": relative_threshold,
        "absolute_threshold": float(cutoff * scale),
        "singular_values": [float(value) for value in singular_values],
        "column_norms": {item: float(column_norms[i]) for i, item in enumerate(ids)},
        "selected_parameter_ids": [ids[i] for i in selected],
        "selected_parameter_indices": selected,
        "selected_parameter_count": len(selected),
        "unselected_parameter_ids": [item for i, item in enumerate(ids) if i not in selected_set],
        "degenerate_parameter_ids": [ids[i] for i in degenerate],
        "correlated_pairs": correlated_pairs,
        "correlation_measure": "uncentered_sensitivity_cosine",
        "correlation_warning_threshold": cosine_threshold,
        "max_free_parameters": cap,
        "parameter_cap_applied": rank > cap,
    }


class BudgetExceeded(RuntimeError):
    """Raised before a callback when its scenario calls exceed a hard budget."""

    def __init__(self, phase: str, requested: int, remaining: int) -> None:
        self.phase = phase
        self.requested = requested
        self.remaining = remaining
        super().__init__(
            f"{phase} solver-call budget exceeded: requested {requested}, remaining {remaining}"
        )


class BudgetLedger:
    """Thread-safe, non-transferable search and diagnostic scenario budgets.

    One unit is one logical scenario solve request, including cache hits and
    failed attempts. A multi-scenario callback must declare its actual scenario
    count, and every additional solver request must pass through this ledger.
    Physical counts describe noncached attempts, not successful simulations.
    Diagnostic allocation is rounded up; search receives the remainder.
    """

    def __init__(self, max_solver_calls: int = 5000, diagnostic_fraction: float = 0.2) -> None:
        self.max_solver_calls = _positive_integer(max_solver_calls, "max_solver_calls")
        if (
            isinstance(diagnostic_fraction, bool)
            or not isinstance(diagnostic_fraction, Real)
            or not math.isfinite(diagnostic_fraction)
            or not 0 <= diagnostic_fraction <= 1
        ):
            raise ValueError("diagnostic_fraction must be finite and between 0 and 1")
        self.diagnostic_fraction = float(diagnostic_fraction)
        self.diagnostics_limit = math.ceil(self.max_solver_calls * self.diagnostic_fraction)
        self.search_limit = self.max_solver_calls - self.diagnostics_limit
        self._limits = {"search": self.search_limit, "diagnostics": self.diagnostics_limit}
        self._counts = {
            phase: {"logical_calls": 0, "physical_calls": 0, "cache_hits": 0, "failed_calls": 0}
            for phase in _PHASES
        }
        self._lock = threading.Lock()

    def consume(self, scenario_calls: int = 1, *, phase: str = "search", cache_hit: bool = False) -> None:
        """Atomically charge attempts before calling a solver or cache callback."""
        calls = _positive_integer(scenario_calls, "scenario_calls")
        if phase not in _PHASES:
            raise ValueError("phase must be 'search' or 'diagnostics'")
        if not isinstance(cache_hit, bool):
            raise ValueError("cache_hit must be a boolean")
        with self._lock:
            counts = self._counts[phase]
            remaining = self._limits[phase] - counts["logical_calls"]
            if calls > remaining:
                raise BudgetExceeded(phase, calls, remaining)
            counts["logical_calls"] += calls
            counts["cache_hits" if cache_hit else "physical_calls"] += calls

    def run(
        self,
        callback: Callable[[], _T],
        *,
        scenario_calls: int = 1,
        phase: str = "search",
        cache_hit: bool = False,
    ) -> _T:
        """Charge before invoking a callback; a failure never refunds budget."""
        if not callable(callback):
            raise ValueError("callback must be callable")
        self.consume(scenario_calls, phase=phase, cache_hit=cache_hit)
        try:
            return callback()
        except BaseException:
            with self._lock:
                self._counts[phase]["failed_calls"] += int(scenario_calls)
            raise

    def snapshot(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable view of all counters and limits."""
        with self._lock:
            phases = {
                phase: {
                    "limit": self._limits[phase],
                    **self._counts[phase],
                    "remaining": self._limits[phase] - self._counts[phase]["logical_calls"],
                }
                for phase in _PHASES
            }
            totals = {
                key: sum(self._counts[phase][key] for phase in _PHASES)
                for key in ("logical_calls", "physical_calls", "cache_hits", "failed_calls")
            }
            return {
                "max_solver_calls": self.max_solver_calls,
                "diagnostic_fraction": self.diagnostic_fraction,
                "phases": phases,
                "totals": {**totals, "remaining": self.max_solver_calls - totals["logical_calls"]},
            }
