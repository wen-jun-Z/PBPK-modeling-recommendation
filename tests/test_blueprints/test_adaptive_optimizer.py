"""Numerical and accounting contracts, with synthetic curves rather than PK claims."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_policy.adaptive_optimizer import optimize, residual_vector


def evaluator(log_error, *, prior=1.2, weight=.1, invalid=None, custom_residual=None):
    observed = np.array([1., 2., 1.5, .8, .3])
    records = []

    def evaluate(lip, scale, purpose):
        x, y = (lip - prior) / 2.5, math.log(scale) / math.log(4)
        row = {"evaluation": len(records) + 1, "lipophilicity": lip, "clearance_scale": scale,
            "purpose": purpose, "status": "ok", "eligible_for_selection": True}
        if invalid and invalid(x, y):
            row.update(status="evaluation_failed", eligible_for_selection=False, reason="synthetic invalid native region")
        else:
            differences = np.asarray(log_error(x, y), dtype=float)
            assert differences.shape == observed.shape
            vector = np.r_[differences / math.sqrt(len(observed)),
                math.sqrt(weight) * math.log(scale) / math.log(2), .1 * x]
            if custom_residual:
                vector = np.asarray(custom_residual(differences, x, y), dtype=float)
                row["residual_vector"] = vector.tolist()
            row.update(objective=float(vector @ vector), predicted_at_observed_times=(observed * np.exp(differences)).tolist())
        records.append(deepcopy(row))
        return row

    return evaluate, observed, records


def curved_valley(x, y):
    return np.array([2 * (y - x * x), .6 * (x - .55), .3 * (x - .55), .1 * (y - .3025), 0.])


def test_residual_norm_matches_existing_native_objective():
    evaluate, observed, _ = evaluator(lambda x, y: np.linspace(-1, 1, 5) * x + y)
    row = evaluate(2., .75, "example")
    residual = residual_vector(row, observed, 1.2)
    assert float(residual @ residual) == pytest.approx(row["objective"], rel=1e-12)
    assert residual[-2] == pytest.approx(math.sqrt(.1) * math.log(.75) / math.log(2))


def test_trust_region_improves_nonlinear_valley_under_same_budget():
    trust, observations, trust_calls = evaluator(curved_valley)
    direct, _, direct_calls = evaluator(curved_valley)
    result = optimize(trust, 1.2, observations, max_evaluations=24)
    baseline = optimize(direct, 1.2, observations, max_evaluations=24, method="pattern")
    assert len(trust_calls) <= 24 and len(direct_calls) <= 24
    assert result["decision"]["objective"] < trust_calls[0]["objective"] * .5
    assert result["decision"]["objective"] < baseline["decision"]["objective"]
    assert result["evaluation_count"] == len(trust_calls)
    assert result["optimizer"]["jacobians"]
    # This tests a known synthetic surface, not a superiority claim for clinical curves.


def test_pattern_baseline_matches_existing_unscreened_search_sequence():
    from pbpk_kb.blueprint_policy.native_calibration import _pattern, _BudgetExhausted
    new_evaluate, observations, new_calls = evaluator(curved_valley)
    old_evaluate, _, old_calls = evaluator(curved_valley)
    result = optimize(new_evaluate, 1.2, observations, max_evaluations=24, method="pattern")

    def old_budget(lip, scale, purpose):
        if len(old_calls) == 24:
            raise _BudgetExhausted
        return old_evaluate(lip, scale, purpose)

    reference = _pattern(old_budget, 1.2, ("lipophilicity", "clearance_scale"), False)
    assert [(r["lipophilicity"], r["clearance_scale"]) for r in new_calls] == [
        (r["lipophilicity"], r["clearance_scale"]) for r in old_calls]
    assert result["decision"]["objective"] == reference["decision"]["objective"]


def test_broad_probes_escape_zero_local_data_sensitivity_without_freezing():
    evaluate, observations, calls = evaluator(lambda x, y: np.array([x * x - .5, x * x - .5, y, y, 0.]), weight=0.)
    result = optimize(evaluate, 1.2, observations, max_evaluations=32, prior_weight=0.)
    assert result["active_parameters"] == ["lipophilicity", "clearance_scale"]
    assert result["decision"]["objective"] < calls[0]["objective"] * .1
    assert abs(result["decision"]["lipophilicity"] - 1.2) > 1.


@pytest.mark.parametrize("allowed", [("lipophilicity",), ("clearance_scale",), ()])
def test_fixed_axes_never_move_in_design_jacobian_or_search(allowed):
    evaluate, observations, calls = evaluator(lambda x, y: np.linspace(-1, 1, 5) * (x - .4) + y + .3)
    result = optimize(evaluate, 1.2, observations, allowed_parameters=allowed)
    if "lipophilicity" not in allowed:
        assert all(row["lipophilicity"] == 1.2 for row in calls)
    if "clearance_scale" not in allowed:
        assert all(row["clearance_scale"] == 1. for row in calls)
    if not allowed:
        assert len(calls) == 1 and result["status"] == "fixed_evidence_only" and not result["converged"]


def test_native_failures_consume_budget_and_are_not_zero_gradient_convergence():
    evaluate, observations, calls = evaluator(curved_valley, invalid=lambda x, y: True)
    result = optimize(evaluate, 1.2, observations, max_evaluations=13)
    assert len(calls) == result["failed_evaluation_count"] == 13
    assert result["decision"] is None and result["status"] == "no_feasible_point"
    assert not result["converged"] and result["optimizer"]["physics_failure_fallback_used"]


def test_invalid_region_retained_while_valid_region_still_optimized():
    evaluate, observations, calls = evaluator(lambda x, y: np.array([x + .4, x + .4, y + .2, y + .2, 0.]),
        invalid=lambda x, y: x > .2)
    result = optimize(evaluate, 1.2, observations, max_evaluations=24)
    assert result["failed_evaluation_count"] >= 2
    assert result["decision"]["objective"] < calls[0]["objective"]
    assert result["decision"]["status"] == "ok"
    assert all(row["evaluation"] == index + 1 for index, row in enumerate(result["evaluations"]))


@pytest.mark.parametrize("limit", [1, 2, 5, 8, 14, 24])
def test_hard_budget_includes_initial_design_and_jacobian_calls(limit):
    evaluate, observations, calls = evaluator(curved_valley)
    result = optimize(evaluate, 1.2, observations, max_evaluations=limit)
    assert len(calls) == result["evaluation_count"] <= limit
    assert len({(r["lipophilicity"].hex(), r["clearance_scale"].hex()) for r in calls}) == len(calls)
    if result["status"] == "budget_exhausted":
        assert len(calls) == limit and not result["converged"]


def test_stricter_adapter_budget_interruption_preserves_completed_trials():
    class NativeRetriesUsedBudget(Exception):
        pass
    evaluate, observations, calls = evaluator(curved_valley)

    def adapter(lip, scale, purpose):
        if len(calls) >= 7:
            raise NativeRetriesUsedBudget
        return evaluate(lip, scale, purpose)

    result = optimize(adapter, 1.2, observations, max_evaluations=24,
        budget_exceptions=(NativeRetriesUsedBudget,))
    assert len(calls) == result["evaluation_count"] == 7
    assert result["status"] == "budget_exhausted" and not result["converged"]


def test_infrastructure_failure_propagates_instead_of_becoming_scientific_failure():
    def outage(*args):
        raise RuntimeError("synthetic infrastructure pause")
    with pytest.raises(RuntimeError, match="infrastructure pause"):
        optimize(outage, 1.2, [1., 2.])


def test_immutable_trial_rows_and_replay_are_deterministic():
    first, observations, original = evaluator(curved_valley)
    result = optimize(first, 1.2, observations, max_evaluations=24)
    assert result["evaluations"] == original
    second, _, replay = evaluator(curved_valley)
    other = optimize(second, 1.2, observations, max_evaluations=24)
    assert replay == original and other == result


def test_objective_mismatch_fails_closed():
    evaluate, observations, _ = evaluator(curved_valley)
    def wrong_objective(*args):
        row = evaluate(*args)
        row["objective"] += .1
        return row
    with pytest.raises(ValueError, match="objective differs"):
        optimize(wrong_objective, 1.2, observations)


def test_explicit_robust_data_residual_preserves_quadratic_priors_and_matching_loss():
    def residual(differences, x, y):
        f = math.log(2)
        # Stable soft-L1 data residual transform; priors remain quadratic.
        z = differences / f
        data = np.sign(differences) * np.sqrt(2 * differences**2 / (np.sqrt(1 + z*z) + 1) / len(differences))
        return np.r_[data, math.sqrt(.1) * 2 * y, .1 * x]
    evaluate, observations, calls = evaluator(curved_valley, custom_residual=residual)
    result = optimize(evaluate, 1.2, observations, max_evaluations=24)
    assert result["decision"]["objective"] < calls[0]["objective"]
    for row in calls:
        assert float(residual_vector(row, observations, 1.2) @ residual_vector(row, observations, 1.2)) == pytest.approx(row["objective"])


def test_bad_explicit_residual_shape_and_nonfinite_values_rejected():
    evaluate, observations, _ = evaluator(curved_valley)
    row = evaluate(1.2, 1., "test")
    for bad in ([1.], [float("nan")] * 7):
        row["residual_vector"] = bad
        with pytest.raises(ValueError, match="n_observations"):
            residual_vector(row, observations, 1.2)


@pytest.mark.parametrize("kwargs", [
    {"max_evaluations": 0}, {"max_evaluations": True}, {"initial_design": 5},
    {"allowed_parameters": ["clearance_scale", "clearance_scale"]},
    {"allowed_parameters": ["renal_clearance"]}, {"prior_weight": -1},
    {"method": "unknown"}, {"budget_exceptions": [RuntimeError]},
])
def test_invalid_configuration_rejected_before_native_calls(kwargs):
    evaluate, observations, calls = evaluator(curved_valley)
    with pytest.raises(ValueError):
        optimize(evaluate, 1.2, observations, **kwargs)
    assert not calls
