"""Development-only synthetic search tests; no clinical fitting or target data."""
from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pbpk_kb.blueprint_policy import refined_axes_optimizer as optimizer
from pbpk_kb.blueprint_policy.axes_optimizer import BudgetExhausted


def _evaluator(log_error, invalid=None):
    observations = np.array([2., 1., .5, .2, .1])
    records = []

    def evaluate(parameters, purpose):
        z = np.log([parameters[name] for name in optimizer.PARAMETERS]) / np.log(
            [optimizer.BOUNDS[name][1] for name in optimizer.PARAMETERS])
        row = {"status": "ok", "evaluation": len(records) + 1, "parameters": deepcopy(parameters),
               "eligible_for_selection": True, "purpose": purpose}
        if invalid is not None and invalid(z):
            row.update(status="scientific_failed", eligible_for_selection=False,
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


def test_three_axis_search_improves_and_never_loses_its_paid_nested_incumbent():
    evaluate, observed, calls = _evaluator(_surface)
    result = optimizer.optimize(evaluate, observed)
    nested = result["optimizer"]["nested_incumbent"]
    assert nested["parameters"]["bidirectional_exchange_scale"] == 1.
    assert result["decision"]["objective"] <= nested["objective"]
    assert result["decision"]["objective"] < .5 * calls[0]["objective"]
    assert result["decision"] == min(calls, key=lambda row: (row["objective"], row["evaluation"]))
    assert result["evaluations"] == calls
    assert result["optimizer"]["prior_fit_inputs_used"] is False
    assert result["optimizer"]["base_solve_included"] is False
    assert result["optimizer"]["native_trial_budget"] == 47
    assert result["optimizer"]["nested_incumbent_guarantee"] == "within_this_run_only_not_separate_full_budget_two_axis_fit"


@pytest.mark.parametrize("budget", [1, 5, 17, 18, 19, 24, 25, 26, 47])
def test_all_stages_share_one_unique_callback_budget(budget):
    evaluate, observed, calls = _evaluator(_surface)
    result = optimizer.optimize(evaluate, observed, max_evaluations=budget)
    assert result["evaluation_count"] == len(calls) <= budget
    assert len({tuple(r["parameters"][name].hex() for name in optimizer.PARAMETERS)
                for r in calls}) == len(calls)
    assert result["optimizer"]["stages"][0]["evaluation_count"] <= 18
    assert result["budget_remaining"] == budget - len(calls)
    if result["budget_exhausted"]:
        assert len(calls) == budget and not result["converged"]
    for row in calls:
        for name in optimizer.PARAMETERS:
            assert optimizer.BOUNDS[name][0] <= row["parameters"][name] <= optimizer.BOUNDS[name][1]


def test_screening_is_fixed_and_early_nested_completion_carries_budget_forward():
    evaluate, observed, calls = _evaluator(lambda z: np.repeat(z[2] - .5, 5))
    result = optimizer.optimize(evaluate, observed)
    nested, screening, full = result["optimizer"]["stages"]
    assert nested["evaluation_count"] < 18
    assert screening["evaluation_count"] <= 7
    assert len(screening["attempted_points"]) == 7
    points = [r["coordinates"] for r in screening["attempted_points"]]
    assert np.asarray(points[:4])[:, 2] == pytest.approx([.5, -.5, 1., -1.])
    assert np.asarray(points[4:]) == pytest.approx(np.asarray(optimizer.CONFIG["joint_normalized_coordinates"]))
    assert full["budget_limit"] == 47 - nested["evaluation_count"] - screening["evaluation_count"]
    assert full["budget_limit"] > 22
    assert result["decision"]["objective"] < .02
    assert all(row["parameters"]["bidirectional_exchange_scale"] == 1
               for row in calls[:nested["evaluation_count"]])


def test_full_trf_really_starts_at_selected_screen_point_not_origin(monkeypatch):
    evaluate, observed, calls = _evaluator(lambda z: np.repeat(z[2] - .5, 5))
    starts = []

    def inspect_start(fun, start, **kwargs):
        incumbent = min(calls, key=lambda row: (row["objective"], row["evaluation"]))
        expected = np.log([incumbent["parameters"][name] for name in optimizer.PARAMETERS]) / np.log(
            [optimizer.BOUNDS[name][1] for name in optimizer.PARAMETERS])
        assert start == pytest.approx(expected)
        assert start[2] == pytest.approx(.5)
        starts.append(start.copy())
        count = len(calls)
        fun(start)
        assert len(calls) == count  # handoff reuses its actual native result
        return OptimizeResult(x=start, success=False, status=0, message="synthetic incomplete run",
                              nfev=1, njev=0, optimality=1.)

    monkeypatch.setattr(optimizer, "least_squares", inspect_start)
    result = optimizer.optimize(evaluate, observed)
    assert len(starts) == 1
    assert not result["converged"] and result["status"] == "best_native_trial"
    assert result["optimizer"]["three_axis_start_evaluation"] == result["decision"]["evaluation"]


def test_nested_numerical_success_alone_cannot_mark_three_axis_search_converged():
    evaluate, observed, _ = _evaluator(lambda z: np.repeat(z[2] - .5, 5))
    result = optimizer.optimize(evaluate, observed, max_evaluations=10)
    assert result["optimizer"]["stages"][0]["numerical_converged"]
    assert result["budget_exhausted"] and not result["converged"]
    assert result["optimizer"]["three_axis_start_evaluation"] is None


def test_data_rank_excludes_prior_and_selected_diagnostics_match_selected_point():
    evaluate, observed, _ = _evaluator(lambda z: np.repeat(z[0] - .2, 5))
    result = optimizer.optimize(evaluate, observed)
    full = [j for j in result["optimizer"]["jacobians"] if j["stage"] == "three_axis" and j["status"] == "ok"]
    assert full
    for item in full:
        assert item["data_only"]["rank"] == 1
        assert item["data_only"]["condition_number"] is None
        assert item["regularized"]["rank"] == 3
        assert item["prior_curvature_fraction_by_axis"]["partition_scale"] == pytest.approx(1.)
        assert item["prior_curvature_fraction_by_axis"]["bidirectional_exchange_scale"] == pytest.approx(1.)
    selected = result["optimizer"]["selected_local_diagnostics"]
    if selected is not None:
        assert selected["center"] == result["decision"]["evaluation"]


def test_scientific_failures_are_charged_without_creating_a_feasible_solution():
    evaluate, observed, calls = _evaluator(_surface, invalid=lambda z: True)
    result = optimizer.optimize(evaluate, observed, max_evaluations=35)
    assert result["failed_evaluation_count"] == len(calls) == 35
    assert result["budget_exhausted"] and not result["converged"]
    assert result["status"] == "no_feasible_point" and result["decision"] is None
    assert result["optimizer"]["physics_failure_fallback_used"]


def test_failed_jacobians_use_fallback_but_cannot_claim_trf_convergence():
    invalid = lambda z: np.any((np.abs(z) > 1e-8) & (np.abs(z) < .02))
    evaluate, observed, calls = _evaluator(lambda z: np.repeat(.1 + z[0], 5), invalid=invalid)
    result = optimizer.optimize(evaluate, observed)
    assert result["eligible"] and result["failed_evaluation_count"] >= 2
    assert result["optimizer"]["physics_failure_fallback_used"]
    assert not result["converged"]
    assert any(row["purpose"] == "full_pattern_fallback_poll" for row in calls)


def test_infrastructure_interruption_replays_deterministically_through_shared_cache():
    evaluate, observed, calls = _evaluator(_surface)
    cache = {}
    interrupted = False

    def callback(parameters, purpose):
        nonlocal interrupted
        key = tuple(parameters[name].hex() for name in optimizer.PARAMETERS)
        if key in cache:
            return deepcopy(cache[key])
        if len(calls) == 27 and not interrupted:
            interrupted = True
            raise RuntimeError("synthetic infrastructure pause")
        cache[key] = evaluate(parameters, purpose)
        return deepcopy(cache[key])

    with pytest.raises(RuntimeError, match="infrastructure pause"):
        optimizer.optimize(callback, observed)
    resumed = optimizer.optimize(callback, observed)
    fresh, _, fresh_calls = _evaluator(_surface)
    reference = optimizer.optimize(fresh, observed)
    assert resumed == reference and calls == fresh_calls


def test_infrastructure_failure_record_is_not_a_scientific_penalty():
    def callback(parameters, purpose):
        return {"evaluation": 1, "parameters": parameters, "status": "infrastructure_failed",
                "eligible_for_selection": False}

    with pytest.raises(RuntimeError, match="Infrastructure failure"):
        optimizer.optimize(callback, [1.])


def test_caller_budget_exception_uses_the_existing_shared_exception_class():
    assert optimizer.BudgetExhausted is BudgetExhausted
    evaluate, observed, calls = _evaluator(_surface)

    def callback(parameters, purpose):
        if len(calls) == 12:
            raise BudgetExhausted
        return evaluate(parameters, purpose)

    result = optimizer.optimize(callback, observed)
    assert result["budget_exhausted"] and result["evaluation_count"] == 12
    assert result["eligible"] and not result["converged"]


@pytest.mark.parametrize("corruption", ["objective", "opposite_residual", "parameters", "failed_eligible"])
def test_callback_loss_and_selection_contract_fail_closed(corruption):
    evaluate, observed, _ = _evaluator(_surface)

    def broken(parameters, purpose):
        row = evaluate(parameters, purpose)
        if corruption == "objective":
            row["objective"] += .1
        elif corruption == "opposite_residual":
            row["residual_vector"] = (-np.asarray(row["residual_vector"])).tolist()
        elif corruption == "parameters":
            row["parameters"]["partition_scale"] = 2.
        else:
            row["status"] = "scientific_failed"
        return row

    with pytest.raises(ValueError):
        optimizer.optimize(broken, observed)


@pytest.mark.parametrize("options", [
    {"max_evaluations": 0}, {"max_evaluations": True},
    {"active_parameters": optimizer.PARAMETERS[:2]}, {"active_parameters": "clearance_scale"},
    {"active_parameters": ("clearance_scale", "clearance_scale", "partition_scale")},
])
def test_invalid_protocol_options_do_not_trigger_native_work(options):
    evaluate, observed, calls = _evaluator(_surface)
    with pytest.raises(ValueError):
        optimizer.optimize(evaluate, observed, **options)
    assert not calls
