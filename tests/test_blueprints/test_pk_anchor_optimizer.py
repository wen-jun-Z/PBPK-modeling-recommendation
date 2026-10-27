"""Numerical anchor construction tests; toy functions are not clinical data."""
from copy import deepcopy
import math

import pytest

from pbpk_kb.blueprints.pk_anchor_optimizer import optimize_pk_anchors


def coupled_pk(lipophilicity, clearance_scale):
    return {"status": "ok",
            "clearance_l_h": 4 * clearance_scale * math.exp(0.12 * lipophilicity + .025 * lipophilicity ** 2),
            "vss_l": 30 * math.exp(.65 * lipophilicity + .04 * lipophilicity ** 2) / (1 + .1 * (clearance_scale - 1))}


def test_nonlinear_coupled_targets_converge_with_verified_real_evaluations():
    target = coupled_pk(1.4, 1.7)
    calls = []

    def evaluate(lipophilicity, clearance_scale):
        calls.append((lipophilicity, clearance_scale))
        return coupled_pk(lipophilicity, clearance_scale)

    result = optimize_pk_anchors(evaluate, .2, target["clearance_l_h"], target["vss_l"])
    assert result["status"] == "converged" and result["converged"]
    decision = result["decision"]
    assert decision["clearance_relative_error"] <= .05
    assert decision["vss_relative_error"] <= .1
    assert decision["accepted"]
    assert len(calls) == result["evaluation_count"] <= 24
    assert len(calls) == len(set(calls))
    assert result["cache_hits"] > 0
    assert (decision["lipophilicity"], decision["clearance_scale"]) in calls
    assert decision["clearance_l_h"] == coupled_pk(decision["lipophilicity"], decision["clearance_scale"])["clearance_l_h"]


def test_no_vss_keeps_lipophilicity_fixed_and_accepts_missing_vss():
    calls = []

    def evaluate(lipophilicity, clearance_scale):
        calls.append(lipophilicity)
        return {"status": "ok", "clearance_l_h": 5 * clearance_scale ** .8}

    result = optimize_pk_anchors(evaluate, -.4, 9)
    assert result["converged"]
    assert all(value == -.4 for value in calls)
    assert result["decision"]["vss_l"] is None
    assert result["decision"]["vss_relative_error"] is None
    assert result["optimizer"]["variables"] == ["log_clearance_scale"]


def test_initial_acceptable_point_is_one_call_and_callback_output_is_unchanged():
    callback_output = {"status": "ok", "clearance_l_h": 10., "vss_l": 40., "other": [1, 2]}
    original = deepcopy(callback_output)
    result = optimize_pk_anchors(lambda *_: callback_output, 0., 10., 40.)
    assert result["converged"] and result["evaluation_count"] == 1
    assert result["decision"]["lipophilicity"] == 0.
    assert result["decision"]["clearance_scale"] == 1.
    assert callback_output == original


@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_budget_counts_finite_differences_and_no_extra_final_call(budget):
    calls = []

    def evaluate(lipophilicity, clearance_scale):
        calls.append((lipophilicity, clearance_scale))
        return coupled_pk(lipophilicity, clearance_scale)

    result = optimize_pk_anchors(evaluate, 0., 20., 100., max_evaluations=budget)
    assert result["status"] == "budget_exhausted"
    assert len(calls) == result["evaluation_count"] == budget
    assert result["decision"] is None and not result["converged"]
    assert result["best_attempt"] is not None


def test_unreachable_targets_do_not_turn_optimizer_success_into_anchor_success():
    result = optimize_pk_anchors(lambda *_: {"status": "ok", "clearance_l_h": 2., "vss_l": 10.},
                                 1., 20., 100.)
    assert result["status"] == "not_converged"
    assert result["optimizer"]["success"]
    assert result["decision"] is None and not result["converged"]


def test_bounds_hold_for_unreachable_high_and_low_targets():
    for target_cl, target_vss in [(1e6, 1e6), (.0001, .0001)]:
        result = optimize_pk_anchors(coupled_pk, .2, target_cl, target_vss)
        assert not result["converged"] and result["decision"] is None
        assert all(-2.3 <= row["lipophilicity"] <= 2.7 and .25 <= row["clearance_scale"] <= 4
                   for row in result["evaluations"])
        assert "not physiological confidence intervals" in result["bounds"]["interpretation"]


@pytest.mark.parametrize("bad_output", [
    {"status": "failed", "clearance_l_h": 10., "vss_l": 40., "reason": "terminal fit failed"},
    {"status": "ok", "clearance_l_h": None, "vss_l": 40.},
    {"status": "ok", "clearance_l_h": True, "vss_l": 40.},
    {"status": "ok", "clearance_l_h": float("nan"), "vss_l": 40.},
    {"status": "ok", "clearance_l_h": 10., "vss_l": None},
    {"status": "ok", "clearance_l_h": 10., "vss_l": 0.},
    None,
])
def test_missing_or_failed_endpoints_never_become_successful_residuals(bad_output):
    result = optimize_pk_anchors(lambda *_: bad_output, 0., 10., 40.)
    assert result["status"] == "evaluation_failed"
    assert result["decision"] is None and result["best_attempt"] is None
    assert result["evaluation_count"] == 1
    assert result["evaluations"][0]["reason"]
    assert "log_residuals" not in result["evaluations"][0]


def test_failed_finite_difference_preserves_previous_real_attempt_without_accepting_it():
    def evaluate(lipophilicity, clearance_scale):
        if lipophilicity > 0:
            raise RuntimeError("native solver failed")
        return coupled_pk(lipophilicity, clearance_scale)

    result = optimize_pk_anchors(evaluate, 0., 8., 60.)
    assert result["status"] == "evaluation_failed" and result["decision"] is None
    assert result["best_attempt"]["evaluation"] == 1
    assert "native solver failed" in result["reason"]
    assert result["evaluation_count"] == 2


@pytest.mark.parametrize("field,value", [
    ("initial_lipophilicity", True), ("initial_lipophilicity", float("inf")),
    ("target_clearance_l_h", 0), ("target_clearance_l_h", -1),
    ("target_clearance_l_h", True), ("target_clearance_l_h", "1"),
    ("target_vss_l", False), ("target_vss_l", float("nan")),
    ("max_evaluations", True), ("max_evaluations", 0), ("max_evaluations", 1.5),
    ("max_evaluations", 25), ("initial_lipophilicity", 1e30),
])
def test_bad_inputs_are_rejected_before_running_simulation(field, value):
    inputs = {"initial_lipophilicity": 0., "target_clearance_l_h": 10., "target_vss_l": 40.}
    inputs[field] = value
    with pytest.raises(ValueError, match=field):
        optimize_pk_anchors(lambda *_: pytest.fail("must not evaluate invalid input"), **inputs)
