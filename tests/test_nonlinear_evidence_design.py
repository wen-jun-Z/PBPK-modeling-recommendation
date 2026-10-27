"""Analytic and independent-integration checks of nonlinear design utility."""
from copy import deepcopy
import inspect
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from pbpk_kb.blueprint_policy import endpoint_evidence_design as linear
from pbpk_kb.blueprint_policy import nonlinear_evidence_design as design


def example():
    a = np.array([-2., -1., 1., 2.])
    return {"candidate_times_h": [2., 8.], "observed_times_h": [1., 3.],
            "concentrations_mg_l": np.exp(np.column_stack([np.zeros(4), a])),
            "endpoints": {"nonmonotonic": np.exp(a ** 2)}}


def test_nonlinear_dependency_informative_when_covariance_is_zero():
    result = design.choose(**example())
    assert result["policies"]["goal_directed"]["nonmonotonic"]["scores"] == pytest.approx([0., 0.])
    assert result["policies"]["goal_directed"]["nonmonotonic"]["selected_time_h"] == 2.
    policy = result["policies"]["nonlinear_goal_directed"]["nonmonotonic"]
    assert policy["selected_time_h"] == 8.
    assert policy["scores"][0] == pytest.approx(0., abs=1e-12)
    assert policy["scores"][1] == pytest.approx(2.25, rel=1e-4)


def test_default_31_and_explicit_seven_node_integration_are_identified():
    for nodes in (7, 31, 63):
        result = design.choose(**example(), quadrature_nodes=nodes)
        assert result["quadrature_starting_nodes_per_member"] == nodes
        for row in result["nonlinear_goals"]["nonmonotonic"]["quadrature"]:
            assert row["nodes_per_member"] == nodes
            assert not row["convergence_verified"]
    assert design.choose(**example())["quadrature_starting_nodes_per_member"] == 31


def test_expected_variance_agrees_with_independent_adaptive_continuous_integral():
    log_c = np.array([-.4, -.15, .15, .4])
    log_e = np.array([-.4, .1, .15, .7])
    sigma = .3
    centered_e = log_e - log_e.mean()
    prior = np.var(log_e)

    def integrand(y):
        likelihood = -.5 * ((y - log_c) / sigma) ** 2
        posterior = np.exp(likelihood - logsumexp(likelihood))
        mean = posterior @ centered_e
        var = posterior @ ((centered_e - mean) ** 2)
        density = np.exp(logsumexp(likelihood) - np.log(len(log_c) * sigma * np.sqrt(2 * np.pi)))
        return density * var

    expected, quadrature_error = quad(integrand, log_c.min() - 10 * sigma,
                                     log_c.max() + 10 * sigma, epsabs=1e-12, points=log_c)
    assert quadrature_error < 1e-9
    result = design.choose([2.], [1., 3.], np.exp(log_c[:, None]),
                           {"auc": np.exp(log_e)}, log_noise_sd=sigma)
    row = result["nonlinear_goals"]["auc"]
    assert row["expected_conditional_variance_log"][0] == pytest.approx(expected, rel=2e-6)
    assert row["utility_log_variance_reduction"][0] == pytest.approx(prior - expected, rel=2e-6)


def test_unresolved_model_alias_preserves_conditional_endpoint_variance():
    # Measuring the concentration identifies one of two groups, but cannot
    # distinguish the two endpoint values within either group.
    log_c = np.array([-2., -2., 2., 2.])
    log_e = np.array([-2., 0., 0., 2.])
    result = design.choose([2.], [1., 3.], np.exp(log_c[:, None]), {"auc": np.exp(log_e)})
    row = result["nonlinear_goals"]["auc"]
    assert row["variance_log"] == pytest.approx(2.)
    assert row["expected_conditional_variance_log"] == pytest.approx([1.])
    assert row["utility_log_variance_reduction"] == pytest.approx([1.])


def test_constant_endpoint_and_constant_concentration_are_not_fictitious_information():
    args = example()
    args["endpoints"]["constant"] = np.full(4, 1.3)
    result = design.choose(**args)
    assert result["nonlinear_goals"]["constant"]["utility_log_variance_reduction"] == [0., 0.]
    action = result["policies"]["nonlinear_goal_directed"]["constant"]
    assert action["reason"] == "no_endpoint_disagreement"
    assert action["selected_time_h"] == 8. and action["fallback"] == "time_farthest"
    assert result["nonlinear_goals"]["nonmonotonic"]["utility_log_variance_reduction"][0] == pytest.approx(0., abs=1e-12)


def test_all_comparators_are_preserved_exactly_and_random_is_expected_action():
    args = example()
    old = linear.choose(**args)
    new = design.choose(**args)
    for key in ("goal_directed", "max_variance", "time_farthest"):
        assert new["policies"][key] == old["policies"][key]
    rand = new["policies"]["uniform_random"]
    assert rand["selected_time_h"] is None and rand["probabilities"] == [.5, .5]
    assert rand["expected_nonlinear_goal_utility"]["nonmonotonic"] == pytest.approx(
        np.mean(new["nonlinear_goals"]["nonmonotonic"]["utility_log_variance_reduction"]))


def test_identical_candidate_prediction_ties_prefer_earlier_time():
    args = example()
    args["concentrations_mg_l"][:, 0] = args["concentrations_mg_l"][:, 1]
    assert design.choose(**args)["policies"]["nonlinear_goal_directed"]["nonmonotonic"]["selected_time_h"] == 2.


def test_uniform_member_duplication_and_unit_rescaling_are_invariant():
    args = example()
    expected = design.choose(**args)
    args["concentrations_mg_l"] = np.tile(args["concentrations_mg_l"], (2, 1)) * 1e5
    args["endpoints"] = {key: np.tile(value, 2) * 1e-5 for key, value in args["endpoints"].items()}
    actual = design.choose(**args)
    assert actual["nonlinear_goals"]["nonmonotonic"]["utility_log_variance_reduction"] == pytest.approx(
        expected["nonlinear_goals"]["nonmonotonic"]["utility_log_variance_reduction"], abs=1e-12)


def test_more_noise_reduces_expected_information_for_same_ensemble():
    args = example()
    precise = design.choose(**args, log_noise_sd=.05)
    noisy = design.choose(**args, log_noise_sd=2.)
    assert precise["nonlinear_goals"]["nonmonotonic"]["utility_log_variance_reduction"][1] > noisy["nonlinear_goals"]["nonmonotonic"]["utility_log_variance_reduction"][1]


def test_stable_likelihood_even_for_extremely_separated_members():
    result = design.choose([2.], [1., 3.], [[1e-200], [1e200]],
                           {"auc": [1e-100, 1e100]}, log_noise_sd=1e-100)
    row = result["nonlinear_goals"]["auc"]
    assert row["expected_conditional_variance_log"][0] == pytest.approx(0., abs=1e-8)
    assert row["utility_log_variance_reduction"][0] == pytest.approx(row["variance_log"])
    json.dumps(result, allow_nan=False)


def test_large_negative_quadrature_utility_triggers_refinement_not_clipping(monkeypatch):
    original = design._integrate_candidate
    calls = []

    def inaccurate(log_c, centered, sigma, nodes):
        calls.append(nodes)
        result = original(log_c, centered, sigma, nodes)
        if nodes == 31:
            result["expected_conditional_variance"] += 10.
        return result

    monkeypatch.setattr(design, "_integrate_candidate", inaccurate)
    result = design.choose(**example())
    assert calls == [31, 63, 31, 63]
    audit = result["nonlinear_goals"]["nonmonotonic"]["quadrature"][0]
    assert audit["nodes_per_member"] == 63
    assert audit["attempts"] == [{"nodes": 31, "utility_bounds_passed": False},
                                  {"nodes": 63, "utility_bounds_passed": True}]


def test_unresolved_quadrature_error_fails_closed(monkeypatch):
    original = design._integrate_candidate

    def inaccurate(log_c, centered, sigma, nodes):
        result = original(log_c, centered, sigma, nodes)
        result["expected_conditional_variance"] += 10.
        return result

    monkeypatch.setattr(design, "_integrate_candidate", inaccurate)
    with pytest.raises(ArithmeticError, match="quadrature utility bounds failed"):
        design.choose(**example())


def test_no_input_mutation_no_rng_use_and_no_hidden_observation_api():
    args = example()
    before = deepcopy(args)
    rng_before = np.random.get_state()
    result = design.choose(**args)
    np.testing.assert_array_equal(args["concentrations_mg_l"], before["concentrations_mg_l"])
    np.testing.assert_array_equal(args["endpoints"]["nonmonotonic"], before["endpoints"]["nonmonotonic"])
    assert args["candidate_times_h"] == before["candidate_times_h"]
    rng_after = np.random.get_state()
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[0] == rng_after[0] and rng_before[2:] == rng_after[2:]
    assert "observations" not in inspect.signature(design.choose).parameters
    assert design.choose(**args) == result
    json.dumps(result, allow_nan=False)
    assert "not a clinical posterior" in " ".join(result["nonlinear_assumptions"])


@pytest.mark.parametrize("nodes", [0, 1, 30, 32, 128, 31., "31", True, None])
def test_invalid_quadrature_order_is_rejected(nodes):
    with pytest.raises(ValueError, match="quadrature_nodes"):
        design.choose(**example(), quadrature_nodes=nodes)


@pytest.mark.parametrize("field,value", [
    ("candidate_times_h", [0., 8.]), ("candidate_times_h", [1., 8.]),
    ("candidate_times_h", [8., 2.]), ("observed_times_h", []),
    ("concentrations_mg_l", [[1., 2.]]), ("endpoints", {"a": [1., 2.]}),
    ("log_noise_sd", 0.), ("log_noise_sd", np.nan)])
def test_frozen_strict_input_validation_is_still_enforced(field, value):
    args = example()
    args[field] = value
    with pytest.raises(ValueError):
        design.choose(**args)
