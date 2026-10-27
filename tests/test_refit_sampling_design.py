"""Visible-only acquisition, actual fitting budgets and numerical failure contracts."""
import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import refit_sampling_design as design


TIMES = np.array([1., 2., 3., 4.])


def model(*, members=None, initial=0., predict=None, goal=None):
    # Early observations contain no information on x0; the later candidate
    # has more endpoint information. The response is nonlinear in x0.
    def response(x, times):
        sensitivity = np.where(times == 5., .1, np.where(times == 6., 2., 0.))
        return np.exp(sensitivity * (x[0] + .2 * x[0] ** 2))
    return dict(x=np.array([initial, 0., 0., 0.]),
                lower=np.full(4, -1.), upper=np.full(4, 1.),
                members=[np.array([v, 0., 0., 0.]) for v in (members or [-.5, .5])],
                predict=predict or response, goal=goal or (lambda x: math.exp(x[0])))


def selection(m=None, **kwargs):
    return design.choose([m or model()], 0, TIMES, np.ones(4), [5., 6.], **kwargs)


def test_real_nonlinear_refits_prefer_the_more_informative_measurement():
    result = selection(proportional_sd=.03, floor_fraction=0.)
    assert result['status'] == 'selected'
    assert result['selected_index'] == 1
    low, high = result['candidates']
    assert high['expected_log_endpoint_loss'] < low['expected_log_endpoint_loss']
    assert high['expected_reduction'] > 0
    assert result['resources']['local_fits'] == 2 * 2 * 3
    assert 0 < result['resources']['refit_calls'] <= 12 * 128
    assert result['resources']['scenario_calls'] == 2
    assert result['resources']['failed_calls'] == 0
    assert not result['hidden_values_used'] and not result['target_values_used']


def test_negative_expected_benefits_are_retained_and_ranked():
    result = selection(model(members=[0.]), proportional_sd=.1, floor_fraction=0.)
    assert result['initial_scenario_loss'] == 0.
    assert all(r['expected_reduction'] < 0 for r in result['candidates'])
    assert result['status'] == 'selected'
    assert result['selected_index'] == min(range(2), key=lambda j: result['candidates'][j]['expected_log_endpoint_loss'])


def test_local_objective_matches_final_blood_partition_regularization(monkeypatch):
    x = np.array([.6, -.4, .3, -.2])
    expected_prediction = np.array([1.1, 2.4, 1.3, 4.])
    observed = np.array([1., 2., 1., 2.])
    captured = {}
    def once(fun, initial, **kwargs):
        captured['residual'] = fun(initial)
        return SimpleNamespace(success=True)
    monkeypatch.setattr(design, 'least_squares', once)
    m = model(predict=lambda x, t: expected_prediction.copy())
    m['x'] = x
    result = design.local_refit(m, TIMES, observed, 1)
    expected = np.mean((np.log(expected_prediction) - np.log(observed)) ** 2)
    expected += (.1*x[1]/math.log(10))**2 + (.1*x[2]/math.log(100))**2 + (.1*x[3]/math.log(10))**2
    assert result['best']['objective'] == pytest.approx(expected, rel=1e-14)
    np.testing.assert_allclose(captured['residual'][:4], (np.log(expected_prediction)-np.log(observed))/2)
    assert len(captured['residual']) == 7  # CL receives no added penalty.


def test_log_loss_does_not_overflow_the_concentration_ratio(monkeypatch):
    monkeypatch.setattr(design, 'least_squares', lambda fun, x, **kw: (fun(x), SimpleNamespace(success=True))[1])
    result = design.local_refit(model(predict=lambda x,t: np.full(len(t), 1e300)), TIMES, np.full(4, 1e-300), 1)
    assert result['failures'] == 0
    assert result['best']['objective'] == pytest.approx((600*math.log(10))**2)


def test_actual_call_budget_includes_numerical_failures(monkeypatch):
    events = []
    def predict(x, times):
        events.append('predict')
        if events.count('predict') == 2:
            raise ValueError('native simulation failed')
        return np.ones(len(times))
    def unlimited(fun, x, **kwargs):
        for _ in range(20):
            fun(x)
    monkeypatch.setattr(design, 'least_squares', unlimited)
    result = design.local_refit(model(predict=predict), TIMES, np.ones(4), 3, lambda: events.append('guard'))
    assert events == ['guard', 'predict'] * 3
    assert result['calls'] == 3 and result['failures'] == 1
    assert result['status'] == 'call_budget_exhausted'
    assert result['best'] is not None  # Explicit approximate result, never labeled converged.
    assert result['reasons'] == {'ValueError: native simulation failed': 1}


def test_failed_residual_plateau_is_not_a_successful_fit():
    result = design.local_refit(model(predict=lambda x,t: np.zeros(len(t))), TIMES, np.ones(4), 128)
    assert result['status'] == 'failed'
    assert result['best'] is None
    assert result['calls'] == result['failures'] > 0


def test_guard_failure_propagates_instead_of_becoming_scientific_failure():
    called = []
    def guard():
        raise ValueError('source hash changed')
    with pytest.raises(ValueError, match='source hash changed'):
        design.local_refit(model(predict=lambda x,t: called.append(True)), TIMES, np.ones(4), 3, guard)
    assert called == []


def test_hypothetical_goal_guard_is_not_swallowed(monkeypatch):
    goals = []
    def goal(x):
        goals.append(True)
        return 1.
    def guard():
        # Scenario goal + current goal have already completed. The next guard
        # must protect the hypothetical goal and propagate a hash exception.
        if len(goals) == 2:
            raise ValueError('frozen inputs changed')
    monkeypatch.setattr(design, 'local_refit', lambda *args, **kwargs: dict(best=dict(objective=0., x=[0.]*4), calls=0, failures=0, status='converged'))
    with pytest.raises(ValueError, match='frozen inputs changed'):
        selection(model(members=[0.], goal=goal), guard=guard)
    assert len(goals) == 2


def test_callback_mutation_cannot_change_fit_coordinates_or_visible_arrays(monkeypatch):
    times, values = TIMES.copy(), np.ones(4)
    m = model()
    original_x = m['x'].copy()
    def destructive(x, t):
        x[:] = 99
        t[:] = 99
        return np.ones(len(t))
    m['predict'] = destructive
    monkeypatch.setattr(design, 'least_squares', lambda fun, x, **kw: (fun(x), SimpleNamespace(success=True))[1])
    result = design.local_refit(m, times, values, 1)
    np.testing.assert_array_equal(times, TIMES)
    np.testing.assert_array_equal(values, np.ones(4))
    np.testing.assert_array_equal(m['x'], original_x)
    np.testing.assert_array_equal(result['best']['x'], original_x)


def test_hidden_concentrations_have_no_selector_input():
    parameters = inspect.signature(design.choose).parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    assert 'candidate_values' not in parameters and 'target_values' not in parameters
    with pytest.raises(TypeError, match='candidate_values'):
        selection(candidate_values=[2., 3.])


@pytest.mark.parametrize('budget', [0, -1, 2.5, True, None])
def test_invalid_budget_rejected_before_model_or_guard_calls(budget):
    calls = []
    with pytest.raises(ValueError, match='budget'):
        selection(model(predict=lambda *args: calls.append('model')), local_budget=budget, guard=lambda: calls.append('guard'))
    assert calls == []


@pytest.mark.parametrize('noise', [dict(proportional_sd=0.), dict(proportional_sd=float('nan')), dict(proportional_sd=True), dict(floor_fraction=-1.), dict(floor_fraction=float('inf'))])
def test_noise_validation_precedes_all_model_calls(noise):
    calls = []
    with pytest.raises(ValueError, match='noise'):
        selection(model(predict=lambda *args: calls.append(True)), **noise)
    assert calls == []


def test_additive_floor_is_based_only_on_visible_concentrations():
    a = selection(model(members=[-.1]), local_budget=1, floor_fraction=.025)
    b = selection(model(members=[.9]), local_budget=1, floor_fraction=.025)
    c = design.choose([model(members=[.9])], 0, TIMES, [1., 2., 3., 4.], [5., 6.], local_budget=1, floor_fraction=.025)
    assert a['additive_floor_mg_l'] == b['additive_floor_mg_l'] == .025
    assert c['additive_floor_mg_l'] == .1
    assert 'median' in a['noise_convention'].lower()


def test_infeasible_hypothetical_fits_leave_no_successful_action():
    def predict(x, times):
        if len(times) == 5:
            return np.zeros(5)
        return np.ones(len(times))
    result = selection(model(members=[0.], predict=predict), local_budget=4)
    assert result['status'] == 'failed' and result['selected_index'] is None
    assert all(r['expected_log_endpoint_loss'] is None and len(r['failures']) == 3 for r in result['candidates'])
    assert result['resources']['failed_calls'] == result['resources']['refit_calls'] == 24


def test_one_failed_action_is_not_renormalized_into_a_false_low_loss():
    def predict(x, times):
        if len(times) == 5 and 5. in times:
            raise ValueError('unavailable candidate simulation')
        return np.ones(len(times))
    result = selection(model(members=[0.], predict=predict), local_budget=4)
    assert result['selected_index'] == 1
    assert result['candidates'][0]['expected_log_endpoint_loss'] is None
    assert result['candidates'][1]['expected_log_endpoint_loss'] == 0.
    assert result['resources']['failed_calls'] == 12


def test_unavailable_hypothetical_endpoint_is_reported_not_fabricated(monkeypatch):
    def goal(x):
        return None if x[0] == .7 else math.exp(x[0])
    monkeypatch.setattr(design, 'local_refit', lambda *args, **kwargs: dict(best=dict(objective=0., x=[.7, 0., 0., 0.]), calls=3, failures=0, status='converged'))
    result = selection(model(members=[0.], goal=goal))
    assert result['status'] == 'failed'
    assert all(len(r['failures']) == 3 and r['simulated_endpoints'] == 0 for r in result['candidates'])
    assert result['resources']['refit_calls'] == 18


def test_partial_hypothesis_failure_invalidates_whole_action(monkeypatch):
    count = 0
    def local(*args, **kwargs):
        nonlocal count
        count += 1
        return dict(best=None if count == 1 else dict(objective=0., x=[0.]*4), calls=2, failures=2 if count == 1 else 0, status='failed' if count == 1 else 'converged')
    monkeypatch.setattr(design, 'local_refit', local)
    result = selection(model(members=[0.]))
    assert result['candidates'][0]['expected_log_endpoint_loss'] is None
    assert result['candidates'][0]['simulated_endpoints'] == 2
    assert result['selected_index'] == 1


def test_two_formulas_have_equal_refit_budgets_and_no_observed_candidate_values(monkeypatch):
    seen = []
    def local(m, times, values, budget, guard):
        seen.append((m['name'], times.copy(), values.copy(), budget))
        return dict(best=dict(objective=0., x=[0.]*4), calls=budget, failures=0, status='call_budget_exhausted')
    monkeypatch.setattr(design, 'local_refit', local)
    models = [dict(model(members=[0.]), name=name) for name in ('a', 'b')]
    result = design.choose(models, 0, TIMES, [1., 2., 3., 4.], [5.], local_budget=17, floor_fraction=0.)
    assert len(seen) == 2 * 3 * 2
    assert [row[0] for row in seen] == ['a', 'b'] * 6
    for _, times, values, budget in seen:
        np.testing.assert_array_equal(times, [1., 2., 3., 4., 5.])
        np.testing.assert_array_equal(values[:4], [1., 2., 3., 4.])
        assert budget == 17
    assert result['resources']['refit_calls'] == 204
    assert result['resources']['budget_exhaustions'] == 12
