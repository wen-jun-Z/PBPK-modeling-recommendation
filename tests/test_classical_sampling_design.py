"""Author-code provenance, Fisher adaptation and independent analytic checks."""

import json

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import classical_sampling_design as design


def inputs():
    return dict(
        visible_jacobian=np.eye(4),
        candidate_jacobian=np.array([[1., 0., 0., 0.], [0., 2., 0., 0.]]),
        goal_gradient=np.array([1., 0., 0., 0.]),
        regularization=np.ones(4),
    )


def test_author_code_scores_match_hand_computed_diagonal_fisher():
    result = design.c_optimal_scores(**inputs())
    assert result["scores"] == pytest.approx([1 / 3, 1 / 2], rel=1e-12)
    assert result["selected_index"] == 0
    other_goal = inputs()
    other_goal["goal_gradient"] = np.array([0., 1., 0., 0.])
    result = design.c_optimal_scores(**other_goal)
    assert result["scores"] == pytest.approx([1 / 2, 1 / 6], rel=1e-12)
    assert result["selected_index"] == 1


def test_full_correlated_design_agrees_with_independent_solve():
    visible = np.array([[1., .2, .1, -.3], [0., 1., .5, .4], [1., 1., -.2, 0.]])
    candidates = np.array([[.3, -.2, 1., .4], [1., 0., 0., 1.], [.2, 1., .2, 1.]])
    gradient = np.array([1., -.3, .4, .8])
    precision = np.array([.4, .2, .3, .8])
    actual = design.c_optimal_scores(visible, candidates, gradient, precision)
    expected = [float(gradient @ np.linalg.solve(
        visible.T @ visible + np.outer(row, row) + np.diag(precision), gradient
    )) for row in candidates]
    assert actual["scores"] == pytest.approx(expected, rel=2e-12)
    assert actual["selected_index"] == int(np.argmin(expected))


def test_actual_author_call_gets_visible_candidate_ridge_and_unit_weights(monkeypatch):
    criterion_type, _, _ = design._author_implementation()
    original = criterion_type.__call__
    calls = []

    def recording_call(self, weights):
        calls.append(weights.copy())
        return original(self, weights)

    monkeypatch.setattr(criterion_type, "__call__", recording_call)
    result = design.c_optimal_scores(**inputs())
    assert len(calls) == 2
    assert all(weights.shape == (9, 1) for weights in calls)
    assert all(np.array_equal(weights, np.ones((9, 1))) for weights in calls)
    assert result["source"]["commit"] == "0d9acd97c61ba11b478f76059d01031eed6321c5"
    assert result["source"]["license"] == "MIT"
    assert result["source"]["version"] == "2.0.0"
    assert "COptimalCriterion" in result["source"]["author_criterion"]


@pytest.mark.parametrize("noise", [.5, np.array([.5, 2.])])
def test_explicit_candidate_noise_agrees_with_prewhitened_inputs(noise):
    args = inputs()
    explicit = design.c_optimal_scores(**args, noise_sd_candidate=noise)
    n = np.broadcast_to(noise, (2,))
    args["candidate_jacobian"] = args["candidate_jacobian"] / n[:, None]
    prewhitened = design.c_optimal_scores(**args)
    assert explicit["scores"] == pytest.approx(prewhitened["scores"], rel=1e-12)
    assert explicit["candidate_whitening"] == "explicit_noise_sd"


def test_zero_visible_rows_and_tied_candidates_use_first_index():
    result = design.c_optimal_scores(np.empty((0, 4)), np.zeros((2, 4)), np.ones(4), np.ones(4))
    assert result["scores"] == pytest.approx([4., 4.])
    assert result["selected_index"] == 0


def test_near_tie_uses_absolute_tolerance():
    result = design.c_optimal_scores(np.empty((0, 4)),
                                    [[1., 0., 0., 0.], [1. + 1e-13, 0., 0., 0.]],
                                    [1., 0., 0., 0.], np.ones(4))
    assert result["scores"][1] < result["scores"][0]
    assert result["selected_index"] == 0


def test_zero_precision_allowed_when_visible_design_is_full_rank():
    args = inputs()
    args["regularization"] = np.zeros(4)
    assert design.c_optimal_scores(**args)["scores"] == pytest.approx([.5, 1.])


def test_singular_information_fails_instead_of_using_pseudoinverse():
    with pytest.raises(ValueError, match="not positive definite"):
        design.c_optimal_scores(np.empty((0, 4)), [[1., 0., 0., 0.]], np.ones(4), np.zeros(4))


def test_no_input_mutation_and_strict_json_result():
    args = inputs()
    copies = {key: value.copy() for key, value in args.items()}
    result = design.c_optimal_scores(**args)
    for key in args:
        np.testing.assert_array_equal(args[key], copies[key])
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert design.c_optimal_scores(**args) == result


@pytest.mark.parametrize("key,value", [
    ("visible_jacobian", np.zeros((3, 3))),
    ("candidate_jacobian", np.empty((0, 4))),
    ("candidate_jacobian", [[float("nan"), 0., 0., 0.]]),
    ("goal_gradient", [1., 0., 0.]),
    ("goal_gradient", [1j, 0., 0., 0.]),
    ("regularization", [1., 1., -1., 1.]),
    ("regularization", 1.),
    ("regularization", [float("inf"), 1., 1., 1.]),
    ("noise_sd_candidate", 0.),
    ("noise_sd_candidate", [-1., 1.]),
    ("noise_sd_candidate", [1., 1., 1.]),
])
def test_invalid_inputs_raise(key, value):
    args = inputs()
    args[key] = value
    with pytest.raises(ValueError):
        design.c_optimal_scores(**args)
