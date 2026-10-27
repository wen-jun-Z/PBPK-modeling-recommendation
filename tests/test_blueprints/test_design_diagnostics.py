"""Toy matrices and callback counters only: no model execution or real data."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pbpk_kb.blueprints.design_diagnostics import (
    BudgetExceeded,
    BudgetLedger,
    analyze_identifiability,
)


def test_full_rank_subset_uses_strongest_independent_columns() -> None:
    result = analyze_identifiability(["a", "b", "c"], np.diag([1.0, 3.0, 2.0]), max_free_parameters=2)
    assert result["rank"] == 3
    assert result["singular_values"] == pytest.approx([3.0, 2.0, 1.0])
    assert result["selected_parameter_ids"] == ["b", "c"]
    assert result["unselected_parameter_ids"] == ["a"]
    assert result["parameter_cap_applied"] is True
    assert result["local_not_global"] is True
    assert result["observation_source_verified"] is False
    assert result["subset_interpretation"] == "conditional_on_unselected_parameters_fixed"
    json.dumps(result, allow_nan=False)


def test_collinear_negative_and_zero_sensitivities_are_reported() -> None:
    result = analyze_identifiability(
        ["a", "opposite", "independent", "zero"],
        [[1.0, -2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, -2.0, 0.0, 0.0]],
    )
    assert result["rank"] == 2
    assert result["selected_parameter_ids"] == ["opposite", "independent"]
    assert result["degenerate_parameter_ids"] == ["zero"]
    assert result["correlated_pairs"] == [
        {"parameter_a": "a", "parameter_b": "opposite", "cosine_similarity": pytest.approx(-1.0)}
    ]


def test_all_zero_matrix_has_no_estimable_parameters() -> None:
    result = analyze_identifiability(["a", "b"], [[0.0, 0.0]])
    assert result["rank"] == 0
    assert result["selected_parameter_ids"] == []
    assert result["degenerate_parameter_ids"] == ["a", "b"]
    assert result["correlated_pairs"] == []
    json.dumps(result, allow_nan=False)


def test_rank_is_observation_limited_and_deterministic() -> None:
    result = analyze_identifiability(["a", "b", "c"], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert result["rank"] == 2
    assert result["selected_parameter_ids"] == ["a", "b"]


def test_small_singular_direction_is_not_selected() -> None:
    result = analyze_identifiability(["a", "b"], [[1.0, 0.0], [0.0, 1e-8]])
    assert result["rank"] == 1
    assert result["selected_parameter_ids"] == ["a"]
    assert result["degenerate_parameter_ids"] == ["b"]


def test_global_scaling_preserves_rank_and_column_selection() -> None:
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    normal = analyze_identifiability(["a", "b", "c"], matrix)
    for scale in (1e-200, 1e200):
        scaled = analyze_identifiability(["a", "b", "c"], matrix * scale)
        assert scaled["rank"] == normal["rank"]
        assert scaled["selected_parameter_ids"] == normal["selected_parameter_ids"]
        json.dumps(scaled, allow_nan=False)


@pytest.mark.parametrize("ids", [[], [""], [" "], [None], ["a", "a"], "a", None])
def test_invalid_parameter_identifiers_are_rejected(ids: object) -> None:
    with pytest.raises(ValueError):
        analyze_identifiability(ids, [[1.0]])


@pytest.mark.parametrize(
    "matrix",
    [[], [[]], [1.0], [[[1.0]]], [[1.0, 2.0]], [[1.0], [1.0, 2.0]],
     [[float("nan")]], [[float("inf")]], [[1j]], [["1.0"]], [[True]]],
)
def test_invalid_observation_matrices_are_rejected(matrix: object) -> None:
    with pytest.raises(ValueError):
        analyze_identifiability(["a"], matrix)


@pytest.mark.parametrize("threshold", [0.0, -1.0, 1.0, float("nan"), float("inf"), True])
def test_invalid_thresholds_are_rejected(threshold: object) -> None:
    with pytest.raises(ValueError):
        analyze_identifiability(["a"], [[1.0]], relative_threshold=threshold)


@pytest.mark.parametrize("cap", [0, -1, 1.5, True])
def test_invalid_parameter_caps_are_rejected(cap: object) -> None:
    with pytest.raises(ValueError):
        analyze_identifiability(["a"], [[1.0]], max_free_parameters=cap)


def test_extreme_unrepresentable_norm_is_rejected() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        analyze_identifiability(["a"], [[1.7e308], [1.7e308]])


def test_default_budget_and_rounding() -> None:
    ledger = BudgetLedger()
    assert ledger.search_limit == 4000
    assert ledger.diagnostics_limit == 1000
    small = BudgetLedger(3, 0.2)
    assert small.search_limit == 2
    assert small.diagnostics_limit == 1
    json.dumps(ledger.snapshot(), allow_nan=False)


def test_search_cannot_spend_reserved_diagnostics_budget() -> None:
    ledger = BudgetLedger(5, 0.4)
    ledger.consume(3)
    called = []
    before = ledger.snapshot()
    with pytest.raises(BudgetExceeded) as error:
        ledger.run(lambda: called.append(True))
    assert called == []
    assert error.value.phase == "search"
    assert error.value.remaining == 0
    assert ledger.snapshot() == before
    ledger.consume(2, phase="diagnostics")
    assert ledger.snapshot()["totals"]["remaining"] == 0


def test_diagnostics_cannot_borrow_search_budget() -> None:
    ledger = BudgetLedger(5, 0.4)
    with pytest.raises(BudgetExceeded):
        ledger.consume(3, phase="diagnostics")
    assert ledger.snapshot()["totals"]["logical_calls"] == 0


def test_cache_hits_and_failures_both_consume_logical_scenario_calls() -> None:
    ledger = BudgetLedger(10, 0.2)
    assert ledger.run(lambda: "cached", scenario_calls=2, cache_hit=True) == "cached"

    def failing_callback() -> None:
        raise RuntimeError("toy failure")

    with pytest.raises(RuntimeError, match="toy failure"):
        ledger.run(failing_callback, scenario_calls=3)
    snapshot = ledger.snapshot()
    assert snapshot["totals"] == {
        "logical_calls": 5, "physical_calls": 3, "cache_hits": 2,
        "failed_calls": 3, "remaining": 5,
    }
    assert snapshot["phases"]["search"]["remaining"] == 3


def test_multiscenario_callback_is_blocked_before_execution() -> None:
    ledger = BudgetLedger(10, 0.2)
    called = []
    with pytest.raises(BudgetExceeded):
        ledger.run(lambda: called.append(True), scenario_calls=9)
    assert called == []
    assert ledger.snapshot()["totals"]["logical_calls"] == 0


def test_snapshot_is_detached() -> None:
    ledger = BudgetLedger(10, 0.2)
    snapshot = ledger.snapshot()
    snapshot["phases"]["search"]["logical_calls"] = 100
    assert ledger.snapshot()["phases"]["search"]["logical_calls"] == 0


def test_concurrent_attempts_cannot_overspend() -> None:
    ledger = BudgetLedger(10, 0.2)

    def attempt(_: int) -> bool:
        try:
            ledger.consume()
        except BudgetExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(attempt, range(20)))
    assert sum(accepted) == 8
    assert ledger.snapshot()["totals"]["logical_calls"] == 8


@pytest.mark.parametrize("maximum", [0, -1, 1.5, True])
def test_invalid_solver_limits_are_rejected(maximum: object) -> None:
    with pytest.raises(ValueError):
        BudgetLedger(maximum)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_diagnostic_fractions_are_rejected(fraction: object) -> None:
    with pytest.raises(ValueError):
        BudgetLedger(diagnostic_fraction=fraction)


@pytest.mark.parametrize("calls", [0, -1, 1.5, True])
def test_invalid_scenario_counts_are_rejected(calls: object) -> None:
    ledger = BudgetLedger()
    with pytest.raises(ValueError):
        ledger.consume(calls)
    assert ledger.snapshot()["totals"]["logical_calls"] == 0


def test_invalid_phase_cache_flag_and_callback_do_not_consume_budget() -> None:
    ledger = BudgetLedger()
    with pytest.raises(ValueError):
        ledger.consume(phase="final_test")
    with pytest.raises(ValueError):
        ledger.consume(cache_hit=1)
    with pytest.raises(ValueError):
        ledger.run(None)
    assert ledger.snapshot()["totals"]["logical_calls"] == 0
