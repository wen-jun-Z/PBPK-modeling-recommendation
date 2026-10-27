"""Numerical engineering profiles, not statistical identifiability claims."""
from copy import deepcopy
import math
import types

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import selective_blood_partition as profile


def design(times):
    return np.column_stack([np.ones(len(times)), np.sin(times), np.cos(2*times), 2*np.sin(3*times)])


def problem(n=20, rbc=.6):
    times = np.linspace(.1, 5, n)
    true = np.array([.2, -.1, .3, rbc])
    observed = np.exp(design(times) @ true)
    lower, upper = np.array([-2., -2., -2., -2.]), np.array([2., 2., 2., 2.])
    def predict(x, t):
        return {"concentration_mg_l": np.exp(design(t) @ x)}
    return predict, times, observed, lower, upper


def test_bounded_profile_excludes_null_only_as_an_engineering_gate():
    predict, times, observed, lower, upper = problem()
    result = profile.analyze(predict, times, observed, lower, upper)
    assert result["status"] == "profiled"
    assert result["decision"]["selected_model"] == "blood_partition"
    assert result["null_mse"] > result["threshold"]
    assert result["connected_components"] == 1 and not result["touches_boundary"]
    assert result["calls"] <= 8192 and result["failures"] == 0
    assert any(row["log_rbc_scale"] == 0 for row in result["profile"])
    assert any(abs(row["log_rbc_scale"] - .6) < 1e-6 for row in result["profile"])
    assert result["threshold"] == pytest.approx(result["minimum_profile_mse"] + math.log(1.1)**2)
    assert result["target_observations_used"] is False and result["external_warm_start_used"] is False
    assert "not a statistical confidence" in result["assumptions"][0]
    # Every best point contains the fixed profile RBC coordinate; nuisance
    # fits consume only the same visible data, without an inferred extra label.
    for row in result["profile"]:
        assert row["best"]["log_parameters"][3] == row["log_rbc_scale"]
        assert len(row["starts"]) == 2


def test_null_retained_for_a_flat_uninformative_profile():
    _, times, _, lower, upper = problem()
    observed = np.exp(design(times)[:, :3] @ [.2, -.1, .3])
    result = profile.analyze(lambda x, t: {"concentration_mg_l": np.exp(design(t)[:, :3] @ x[:3])},
                             times, observed, lower, upper)
    assert result["status"] == "profiled" and result["decision"]["selected_model"] == "direct_relaxed"
    assert result["touches_boundary"] and not result["null_outside"]
    assert "null_within_engineering_tolerance" in result["decision"]["reasons"]


def synthetic_rows(coordinates, errors, successful=None):
    successful = [True]*len(coordinates) if successful is None else successful
    return [{"log_rbc_scale": x, "mse": e, "successful": good} for x,e,good in zip(coordinates, errors, successful)]


def test_boundary_profile_rejected_by_main_but_allowed_by_necessity_ablation():
    assessment = profile._assessment(synthetic_rows([-2, -1, 0, 1, 2], [0, .02, .1, .2, .3]), 20, math.log(1.1))
    assert assessment["decision"]["selected_model"] == "direct_relaxed"
    assert assessment["necessity_only_decision"]["selected_model"] == "blood_partition"
    assert assessment["touches_boundary"]
    assert "accepted_band_touches_numerical_boundary" in assessment["decision"]["reasons"]


def test_disjoint_accepted_components_are_not_a_supported_main_gate():
    assessment = profile._assessment(synthetic_rows([-2, -1, 0, 1, 2], [.2, 0, .2, 0, .2]), 20, math.log(1.1))
    assert assessment["connected_components"] == 2
    assert assessment["accepted_intervals_log"] == [[-1., -1.], [1., 1.]]
    assert not assessment["decision"]["accepted"]
    assert assessment["necessity_only_decision"]["accepted"]


def test_failed_profile_point_closes_both_decisions_even_if_other_points_exclude_null():
    rows = synthetic_rows([-2, -1, 0, 1, 2], [.2, 0, .2, None, .2], [True, True, True, False, True])
    assessment = profile._assessment(rows, 20, math.log(1.1))
    assert not assessment["decision"]["accepted"] and not assessment["necessity_only_decision"]["accepted"]
    assert "one_or_more_profile_points_failed" in assessment["decision"]["reasons"]


def test_five_point_result_is_exploratory_and_cannot_select_four_parameter_model():
    rows = synthetic_rows([-2, -1, 0, 1, 2], [.2, 0, .2, .2, .2])
    assessment = profile._assessment(rows, 5, math.log(1.1))
    assert assessment["raw_five_point_gate"] is True
    assert not assessment["decision"]["accepted"] and not assessment["necessity_only_decision"]["accepted"]
    assert "fewer_than_eight_visible_observations" in assessment["decision"]["reasons"]


def test_guard_failure_propagates_without_becoming_numerical_failure():
    predict, times, observed, lower, upper = problem()
    calls = []
    def guard():
        raise ValueError("Source hash changed")
    with pytest.raises(ValueError, match="Source hash changed"):
        profile.analyze(lambda *_: calls.append(1), times, observed, lower, upper, before_trial=guard)
    assert calls == []


def test_hard_budget_reserves_all_gridpoints_and_counts_failed_trials(monkeypatch):
    def hungry(fun, initial, **kwargs):
        for _ in range(10000):
            fun(initial)
        pytest.fail("Call budget failed")
    monkeypatch.setattr(profile, "least_squares", hungry)
    _, times, observed, lower, upper = problem()
    calls = []
    def bad(*_):
        calls.append(1)
        raise ValueError("Numerical failure")
    result = profile.analyze(bad, times, observed, lower, upper, grid_points=5, max_calls=37)
    assert result["calls"] == result["failures"] == len(calls) == 37
    assert result["status"] == "incomplete" and not result["decision"]["accepted"]
    assert len(result["profile"]) == 5
    assert all(row["calls"] > 0 for row in result["profile"])
    assert result["full_fit"]["calls"] + sum(row["calls"] for row in result["profile"]) == 37


def test_small_budget_fails_closed_without_skipping_grid_records():
    predict, times, observed, lower, upper = problem()
    result = profile.analyze(predict, times, observed, lower, upper, grid_points=5, max_calls=2)
    assert result["calls"] == 2 and len(result["profile"]) == 5
    assert not result["decision"]["accepted"] and result["status"] == "incomplete"


def test_observations_arrays_and_bounds_are_not_modified_and_mg_l_scale_cancels():
    predict, times, observed, lower, upper = problem()
    originals = [x.copy() for x in (times, observed, lower, upper)]
    baseline = profile.analyze(predict, times, observed, lower, upper, grid_points=5)
    scaled = profile.analyze(lambda x, t: {"concentration_mg_l": predict(x, t)["concentration_mg_l"]*1000},
                             times, observed*1000, lower, upper, grid_points=5)
    assert scaled["null_mse"] == pytest.approx(baseline["null_mse"], rel=1e-9)
    assert scaled["decision"] == baseline["decision"]
    for actual, original in zip((times, observed, lower, upper), originals):
        np.testing.assert_array_equal(actual, original)


def test_protected_blood_partition_is_never_replaced_by_numerical_gate():
    result = {"decision": {"selected_model": "blood_partition", "accepted": True, "reasons": []}}
    original = deepcopy(result)
    decision = profile.decision_for(result, protected=True)
    assert decision["selected_model"] == "direct_relaxed" and not decision["accepted"]
    assert decision["reasons"] == ["protected_blood_partition_input"]
    assert result == original


@pytest.mark.parametrize("name,value", [("grid_points", 2), ("grid_points", 3.), ("grid_points", True),
    ("max_calls", 0), ("max_calls", False), ("max_calls", 4.5), ("log_tolerance", 0), ("log_tolerance", float("nan"))])
def test_invalid_configuration_rejected_before_predicting(name, value):
    _, times, observed, lower, upper = problem()
    with pytest.raises(ValueError):
        profile.analyze(lambda *_: pytest.fail("Invalid configuration evaluated"), times, observed, lower, upper, **{name:value})


@pytest.mark.parametrize("fault", ["duplicate_time", "zero_observation", "nonfinite_bound", "wrong_bounds_dimension", "null_outside_bounds"])
def test_input_time_concentration_and_log_bounds_are_checked(fault):
    _, times, observed, lower, upper = problem()
    if fault == "duplicate_time": times[1] = times[0]
    elif fault == "zero_observation": observed[0] = 0.
    elif fault == "nonfinite_bound": lower[0] = -math.inf
    elif fault == "wrong_bounds_dimension": lower = lower[:3]
    elif fault == "null_outside_bounds": lower[3] = .1
    with pytest.raises(ValueError):
        profile.analyze(lambda *_: pytest.fail("Invalid input evaluated"), times, observed, lower, upper)
