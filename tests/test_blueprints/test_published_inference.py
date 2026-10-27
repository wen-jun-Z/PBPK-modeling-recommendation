"""Independent algebraic/statistical checks for the published inference adapter."""
import inspect
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import published_inference as inference


def test_pss_squared_condition_number_rule():
    matrix = np.diag([10.0, 1.0, 0.01, 0.00001])
    result = inference.pss_svd(matrix, eta=1e-5)
    assert result["selected_indices"] == [0, 1]
    assert [x["removed_index"] for x in result["history"][:-1]] == [3, 2]
    assert result["history"][-1]["squared_singular_ratio"] == pytest.approx(0.01)


def test_pss_removes_dependent_column_and_retains_full_rank_subset():
    a = np.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 2.], [2., 1., 3.]])
    result = inference.pss_svd(a, eta=1e-8)
    selected = result["selected_indices"]
    assert len(selected) == 2
    assert np.linalg.matrix_rank(a[:, selected]) == 2


def test_pss_zero_sensitivity_is_not_declared_identifiable():
    result = inference.pss_svd(np.zeros((5, 4)))
    assert result["selected_indices"] == []


def test_pss_handles_fewer_observations_than_parameters():
    result = inference.pss_svd(np.array([[1., 2., 0.], [0., 1., 2.]]))
    assert len(result["selected_indices"]) <= 2


def test_sensitivity_agrees_with_closed_form_at_bound():
    a = np.array([[2., -1., 1., 3.], [0., 2., -2., 1.]])
    lower, upper = np.full(4, -3.), np.full(4, 3.)
    initial = np.array([-3., 0., 0., 3.])
    actual = inference.log_sensitivity(lambda x: a @ x, initial, lower, upper)
    np.testing.assert_allclose(actual, a, atol=2e-11)


def test_bayesian_prior_matches_frozen_calibration_objective():
    theta = np.array([0.5, -0.8, 0.6, 1.2])
    residual = np.array([0.3, -0.4, 0.1, 0.05, -0.1])
    sd, sigma, n = inference.default_prior_sd(5), 0.3, 5
    minus_twice_logpost = np.sum((residual/sigma)**2) + np.sum((theta/sd)**2)
    frozen_objective = np.mean(residual**2) + (theta[1]*0.1/np.log(10))**2
    frozen_objective += (theta[2]*0.1/np.log(100))**2 + (theta[3]*0.1/np.log(10))**2
    assert minus_twice_logpost * sigma**2/n == pytest.approx(frozen_objective)


@pytest.mark.parametrize("bad", [np.full((2, 4), np.nan), np.empty((0, 4)), np.ones(4)])
def test_invalid_sensitivity_is_rejected(bad):
    with pytest.raises(ValueError):
        inference.pss_svd(bad)


def test_inference_interface_has_no_target_input():
    names = set(inspect.signature(inference.fit).parameters)
    assert not any("target" in name for name in names)
    assert "observed_log" in names


@pytest.fixture(scope="module")
def gaussian_fit():
    # Linear Gaussian observation model has an independently calculable posterior.
    a = np.r_[np.eye(4), np.ones((1, 4))]
    truth = np.array([0.2, -0.4, 0.6, -0.2])
    y = a @ truth
    sd, sigma = np.full(4, 1.0), 0.3
    covariance = np.linalg.inv(a.T @ a / sigma**2 + np.diag(1/sd**2))
    mean = covariance @ (a.T @ y / sigma**2)
    result = inference.fit(lambda x: a @ x, y, np.array([1., -1., 0.3, 0.8]), np.full(4, -5.),
                           np.full(4, 5.), prior_sd=sd, sigma_log=sigma,
                           seed=219, chains=4, iterations=2000, burnin=500,
                           max_iterations=4000)
    return result, mean, covariance


def test_official_dram_matches_analytic_posterior(gaussian_fit):
    result, mean, covariance = gaussian_fit
    np.testing.assert_allclose(result["point_parameters"], mean, atol=0.065)
    actual_sd = result["posterior_samples"].std(axis=0)
    np.testing.assert_allclose(actual_sd, np.sqrt(np.diag(covariance)), rtol=0.18)
    assert result["diagnostics"]["sampler"] == "pymcmcstat DRAM"
    assert result["diagnostics"]["target_observations_used"] is False
    assert result["diagnostics"]["prior_mean"] == [0., 0., 0., 0.]
    assert result["simulator_calls"] > 4 * 1000


def test_rank_rhat_and_ess_reported(gaussian_fit):
    result, _, _ = gaussian_fit
    details = result["diagnostics"]
    assert len(details["rank_normalized_split_rhat"]) == 4
    assert min(details["bulk_ess"]) > 100
    assert details["converged"]


def test_pss_inactive_axes_stay_at_evidence_values():
    a = np.c_[np.arange(1., 6.), np.zeros((5, 3))]
    result = inference.fit(lambda x: a @ x, a @ np.array([0.2, 0., 0., 0.]),
                           np.zeros(4), np.full(4, -3.), np.full(4, 3.),
                           method="pss_bayesian_dram", seed=88,
                           iterations=500, burnin=100, max_iterations=500, chains=2)
    assert result["selected_indices"] == [0]
    np.testing.assert_array_equal(result["posterior_samples"][:, 1:], 0.)
    assert result["point_parameters"][0] == pytest.approx(0.2, abs=0.03)


def test_continuation_preserves_required_draw_count_and_exposes_warning():
    a = np.r_[np.eye(4), np.ones((1, 4))]
    result = inference.fit(lambda x: a @ x, np.zeros(5), np.zeros(4),
                           np.full(4, -3.), np.full(4, 3.), seed=999,
                           iterations=100, burnin=50, max_iterations=200, chains=2)
    details = result["diagnostics"]
    assert [x["iterations_per_chain"] for x in details["convergence_history"]] == [100, 200]
    assert all(x["draws"] == 200 for x in details["chains"])
    assert result["posterior_samples"].shape == (300, 4)
    assert details["status"] == "complete_with_convergence_warning"


def test_invalid_prediction_fails_before_sampler():
    with pytest.raises(ValueError, match="finite log concentration"):
        inference.fit(lambda x: np.zeros(2), np.zeros(5), np.zeros(4),
                       np.full(4, -3.), np.full(4, 3.))


def test_infrastructure_stop_is_not_swallowed_as_numerical_failure():
    calls = 0
    def predict(x):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("source hash changed: stop submitting work")
        return np.zeros(5)
    with pytest.raises(RuntimeError, match="source hash changed"):
        inference.fit(predict, np.zeros(5), np.zeros(4), np.full(4, -3.), np.full(4, 3.))


def test_seed_reproducibility_and_global_rng_restoration():
    a = np.c_[np.arange(1., 6.), np.zeros((5, 3))]
    kwargs = dict(method="pss_bayesian_dram", seed=173, iterations=100,
                  burnin=50, max_iterations=100, chains=2)
    np.random.seed(19)
    expected = np.random.random(3)
    np.random.seed(19)
    first = inference.fit(lambda x: a @ x, np.zeros(5), np.zeros(4),
                          np.full(4, -3.), np.full(4, 3.), **kwargs)
    np.testing.assert_array_equal(np.random.random(3), expected)
    second = inference.fit(lambda x: a @ x, np.zeros(5), np.zeros(4),
                           np.full(4, -3.), np.full(4, 3.), **kwargs)
    np.testing.assert_array_equal(first["posterior_samples"], second["posterior_samples"])
