"""Synthetic numerical and accounting tests, never clinical validation."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_policy.axes_optimizer import BOUNDS, PARAMETERS, optimize, residual_vector


def _evaluator(log_error, invalid=None):
    observations = np.array([2., 1., .5, .2, .1])
    records = []

    def evaluate(parameters, purpose):
        z = np.array([math.log(parameters[name]) / math.log(BOUNDS[name][1]) for name in PARAMETERS])
        row = {"status": "ok", "evaluation": len(records) + 1, "parameters": deepcopy(parameters),
               "eligible_for_selection": True, "purpose": purpose}
        if invalid is not None and invalid(z):
            row.update(status="simulation_failed", eligible_for_selection=False,
                       reason="synthetic invalid physics")
        else:
            differences = np.asarray(log_error(z))
            vector = np.r_[differences / math.sqrt(5), math.sqrt(.1) * 2 * z[0], .1 * z[1], .1 * z[2]]
            row.update(objective=float(vector @ vector), residual_vector=vector.tolist(),
                       predicted_at_observed_times=(observations * np.exp(differences)).tolist())
        records.append(deepcopy(row))
        return row

    return evaluate, observations, records


def _surface(z):
    x, y, w = z
    return np.array([x + .25, 2 * (y - x * x - .3), w + .4, y - .3, w - x])


def test_fixed_objective_and_n_plus_three_residuals_match_declared_loss():
    evaluate, observations, _ = _evaluator(_surface)
    parameters = dict(zip(PARAMETERS, [2., 10., 100.]))
    row = evaluate(parameters, "example")
    vector = residual_vector(row, observations)
    assert vector.shape == (8,)
    assert vector[-3:] == pytest.approx([math.sqrt(.1), .1, .1])
    assert float(vector @ vector) == pytest.approx(row["objective"])


def test_bounded_trf_improves_known_nonlinear_surface():
    evaluate, observations, calls = _evaluator(_surface)
    result = optimize(evaluate, observations)
    assert result["eligible"] and result["decision"]["status"] == "ok"
    assert result["decision"]["objective"] < .5 * calls[0]["objective"]
    assert len(calls) == result["evaluation_count"] <= 48
    assert result["optimizer"]["jacobians"]
    assert result["evaluations"] == calls


@pytest.mark.parametrize("active", [PARAMETERS[:1], PARAMETERS[:2], PARAMETERS, ()])
def test_nested_models_have_same_budget_and_inactive_axes_remain_one(active):
    evaluate, observations, calls = _evaluator(_surface)
    result = optimize(evaluate, observations, active_parameters=active, max_evaluations=18)
    assert result["optimizer"]["native_trial_budget"] == 18
    for row in calls:
        for name in PARAMETERS:
            assert BOUNDS[name][0] <= row["parameters"][name] <= BOUNDS[name][1]
            if name not in active:
                assert row["parameters"][name] == 1.
    if not active:
        assert len(calls) == 1 and result["status"] == "fixed_parameters_only"
        assert not result["converged"]


def test_initial_design_is_origin_then_fixed_half_range_axis_probes():
    evaluate, observations, calls = _evaluator(_surface)
    optimize(evaluate, observations, max_evaluations=7)
    points = [np.log([r["parameters"][name] for name in PARAMETERS]) /
              np.log([BOUNDS[name][1] for name in PARAMETERS]) for r in calls]
    assert points[0] == pytest.approx([0, 0, 0])
    for index in range(3):
        positive = np.zeros(3)
        positive[index] = .5
        assert points[1 + 2 * index] == pytest.approx(positive)
        assert points[2 + 2 * index] == pytest.approx(-positive)


@pytest.mark.parametrize("budget", [1, 2, 7, 8, 12, 24, 48])
def test_unique_calls_including_derivative_probes_never_exceed_budget(budget):
    evaluate, observations, calls = _evaluator(_surface)
    result = optimize(evaluate, observations, max_evaluations=budget)
    assert result["evaluation_count"] == len(calls) <= budget
    assert len({tuple(r["parameters"][name].hex() for name in PARAMETERS) for r in calls}) == len(calls)
    if result["budget_exhausted"]:
        assert len(calls) == budget and not result["converged"]


def test_data_only_rank_is_not_replaced_by_full_rank_regularization():
    evaluate, observations, _ = _evaluator(lambda z: np.repeat(z[0] - .2, 5))
    result = optimize(evaluate, observations)
    diagnostics = [j for j in result["optimizer"]["jacobians"] if j["status"] == "ok"]
    assert diagnostics
    for item in diagnostics:
        assert item["data_only"]["rank"] == 1
        assert item["data_only"]["condition_number"] is None
        assert item["regularized"]["rank"] == 3
        assert item["regularized"]["condition_number"] >= 1
        assert item["interpretation"] == "local_only_no_global_identifiability_claim"


def test_zero_origin_sensitivity_does_not_freeze_a_nonlinear_axis():
    evaluate, observations, calls = _evaluator(lambda z: np.repeat(z[1] ** 2 - .36, 5))
    result = optimize(evaluate, observations, active_parameters=("partition_scale",))
    assert result["active_parameters"] == ["partition_scale"]
    assert result["screening"]["status"] == "not_requested"
    assert result["decision"]["objective"] < .1 * calls[0]["objective"]


def test_failed_physics_uses_budget_and_cannot_claim_convergence():
    evaluate, observations, calls = _evaluator(_surface, invalid=lambda z: True)
    result = optimize(evaluate, observations, max_evaluations=16)
    assert len(calls) == result["failed_evaluation_count"] == 16
    assert result["status"] == "no_feasible_point" and result["decision"] is None
    assert result["budget_exhausted"] and not result["converged"]
    assert result["optimizer"]["physics_failure_fallback_used"]


def test_failed_both_sides_of_jacobian_activates_pattern_fallback():
    # A feasible origin and broad lattice, with both .01 derivative probes invalid.
    invalid = lambda z: np.any((np.abs(z) > 1e-8) & (np.abs(z) < .02))
    evaluate, observations, calls = _evaluator(lambda z: np.repeat(.1 + z[0], 5), invalid=invalid)
    result = optimize(evaluate, observations, active_parameters=PARAMETERS[:1], max_evaluations=20)
    assert result["eligible"]
    assert result["failed_evaluation_count"] >= 2
    assert result["optimizer"]["physics_failure_fallback_used"]
    assert any(j["status"] == "failed" for j in result["optimizer"]["jacobians"])
    assert any(r["purpose"] == "pattern_fallback_poll" for r in calls)


def test_cached_callback_replay_resumes_deterministically_without_rerunning_points():
    evaluate, observations, calls = _evaluator(_surface)
    cached = {}
    interrupted = False

    def callback(parameters, purpose):
        nonlocal interrupted
        key = tuple(parameters[name].hex() for name in PARAMETERS)
        if key in cached:
            return deepcopy(cached[key])
        if len(calls) == 9 and not interrupted:
            interrupted = True
            raise RuntimeError("synthetic infrastructure pause")
        cached[key] = evaluate(parameters, purpose)
        return deepcopy(cached[key])

    with pytest.raises(RuntimeError, match="infrastructure pause"):
        optimize(callback, observations, max_evaluations=24)
    resumed = optimize(callback, observations, max_evaluations=24)
    fresh, _, reference_calls = _evaluator(_surface)
    reference = optimize(fresh, observations, max_evaluations=24)
    assert resumed == reference and calls == reference_calls


@pytest.mark.parametrize("corruption", ["objective", "opposite_residual", "parameters", "failed_eligible"])
def test_callback_contract_mismatches_fail_closed(corruption):
    evaluate, observations, _ = _evaluator(_surface)

    def broken(parameters, purpose):
        row = evaluate(parameters, purpose)
        if corruption == "objective":
            row["objective"] += .1
        elif corruption == "opposite_residual":
            row["residual_vector"] = (-np.asarray(row["residual_vector"])).tolist()
        elif corruption == "parameters":
            row["parameters"]["partition_scale"] = 2.
        else:
            row["status"] = "failed"
        return row

    with pytest.raises(ValueError):
        optimize(broken, observations)


@pytest.mark.parametrize("options", [
    {"max_evaluations": 0}, {"max_evaluations": True},
    {"active_parameters": ["clearance_scale", "clearance_scale"]},
    {"active_parameters": "clearance_scale"}, {"active_parameters": ["lipophilicity"]},
])
def test_invalid_options_do_not_request_native_work(options):
    evaluate, observations, calls = _evaluator(_surface)
    with pytest.raises(ValueError):
        optimize(evaluate, observations, **options)
    assert not calls
