"""Development-only fold selection, weighted loss and resumable fitting contracts."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import tail_exchange_calibration as cal

spec = importlib.util.spec_from_file_location('_tail_contracts', Path(__file__).with_name('test_robust_exchange_calibration.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
contracts = helpers.contracts


def test_tail_membership_uses_time_only_and_end_of_infusion_is_not_postinfusion():
    times = [.1, .5, 1., 1.1, 2., 3., 4., 5., 8.]
    plan = cal.tail_weights(times, 1.)
    assert plan['tail_indices'] == [6, 7, 8]
    assert plan['body_indices'] == list(range(6))
    assert sum(plan['weights'][i] for i in plan['tail_indices']) == pytest.approx(.5)
    assert sum(plan['weights'][i] for i in plan['body_indices']) == pytest.approx(.5)
    assert plan['selection_uses_concentrations'] is False


@pytest.mark.parametrize('times,infusion', [([.1, .5, 1., 2.], 1.), ([1., 2., 3.], .5)])
def test_tail_fallback_is_explicit_and_uniform(times, infusion):
    plan = cal.tail_weights(times, infusion)
    assert plan['supported'] is False
    assert plan['fallback'] == 'uniform_logMSE_insufficient_phase_support'
    np.testing.assert_allclose(plan['weights'], np.full(len(times), 1/len(times)))


@pytest.mark.parametrize('n', [7, 8, 9, 10, 12, 13, 14, 16])
def test_folds_are_disjoint_have_six_training_points_and_protect_interior_anchors(n):
    times = np.geomspace(.1, 24, n)
    folds = cal.development_folds(times, .5)
    k = min(3, max(1, n//4), n-6)
    assert len(folds) == 2
    assert folds[0]['heldout_indices'] == list(range(n-k, n))
    for fold in folds:
        assert len(fold['training_indices']) >= 6
        assert len(fold['heldout_indices']) == k
        assert set(fold['training_indices']).isdisjoint(fold['heldout_indices'])
        assert sorted(fold['training_indices']+fold['heldout_indices']) == list(range(n))
    assert set(folds[1]['heldout_indices']).isdisjoint(folds[0]['heldout_indices'])
    assert set(folds[1]['heldout_indices']).isdisjoint(folds[1]['protected_interior_indices'])


def test_cv_does_not_silently_use_training_loss_if_folds_are_unsupported():
    with pytest.raises(ValueError, match='no full-data fallback'):
        cal.development_folds([1, 2, 3, 4, 5, 6], .5)
    plan = {'prior_weights': [.01, .1], 'folds': [{}, {}]}
    stages = {f'cv_{w:g}_{i}': {'heldout_score': None if w == .01 else 2.}
              for w in plan['prior_weights'] for i in range(2)}
    result = cal._cv_selection(stages, plan)
    assert result['selected']['penalty_weight'] == .1
    for stage in stages.values(): stage['heldout_score'] = None
    assert cal._cv_selection(stages, plan)['selected'] is None


def test_cv_exact_ties_prefer_stronger_prior():
    plan = {'prior_weights': [.01, .1], 'folds': [{}, {}]}
    stages = {f'cv_{w:g}_{i}': {'heldout_score': 1.} for w in plan['prior_weights'] for i in range(2)}
    assert cal._cv_selection(stages, plan)['selected']['penalty_weight'] == .1


@pytest.fixture(params=['tail_balanced', 'development_guarded'])
def fitting(request, tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, 'cal', cal)
    original = cal.fit_mapping
    def fit(**kwargs):
        kwargs['variant'] = request.param
        return original(**kwargs)
    monkeypatch.setattr(cal, 'fit_mapping', fit)
    result = contracts.fitting.__wrapped__(tmp_path, monkeypatch)
    root, req, source, calls, execute = result
    helpers.qualify(req, 'predicted')
    cal.save(root/'prior.json', {'kind': 'independent synthetic input'})
    req['evidence'][0]['source']['sha256'] = cal.sha256(root/'prior.json')
    doc = cal._read(root/'development.json')
    times = np.asarray([6., 12., 30., 60., 120., 180., 240., 480.])
    doc['observations'] = [{'time_min': float(t), 'concentration_umol_per_l': float(v)}
        for t, v in zip(times, 2*np.exp(1.6-(.5+np.exp(-.1))*times/60))]
    cal.save(root/'development.json', doc)
    source['sha256'] = cal.sha256(root/'development.json')
    return result


@pytest.mark.parametrize('contract', ['test_sealed_fit_selection_resume_and_cost',
    'test_fit_refuses_target_observations', 'test_completed_fit_refuses_source_or_code_changes',
    'test_subset_conditions_must_match_request', 'test_mapping_cannot_rewrite_both_start_scores_and_selected_decision'])
def test_full_fit_contracts(fitting, contract):
    getattr(contracts, contract)(fitting)


def test_loss_metadata_and_plain_rmse_are_kept_separate(fitting):
    root, req, source, calls, fit = fitting
    result = fit(); stage = result['stages']['full']; d = result['decision']
    w = np.asarray(stage['data_weights']); tr = cal._read(root/'development.json')
    t = np.array([r['time_min']/60 for r in tr['observations']]); o = np.array([r['concentration_umol_per_l']*.5 for r in tr['observations']])
    p = cal._simulate({}, req, t, result['variant'], cal.PARAMETERS[result['variant']], d['log_parameters'])['concentration_mg_l']
    errors = np.log(p/o)
    assert d['data_only_ln_rmse'] == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert d['curve_metrics']['gmfe'] == pytest.approx(np.exp(np.mean(abs(errors))))
    assert d['weighted_data_loss'] == pytest.approx(np.dot(w, errors**2))
    x = np.array(d['log_parameters']); penalty = stage['penalty_weight']*(x[0]/np.log(2))**2 + .01*(x[1]/np.log(10))**2 + .01*(x[2]/np.log(100))**2
    assert d['objective'] == pytest.approx(d['weighted_data_loss']+penalty)
    if result['variant'] == 'development_guarded':
        assert len(result['stages']) == 5
        assert result['formula_selection_score'] == result['cv_selection']['selected']['score']
        for name, value in result['stages'].items():
            assert len(value['starts']) == (16 if name == 'full' else 8)
            if name != 'full':
                np.testing.assert_allclose(value['heldout_score'], np.mean(np.array(value['heldout_log_errors'])**2))
    else:
        assert len(result['stages']) == 1
        assert result['formula_selection_score'] == d['objective']


def test_partial_start_resume_is_not_misreported_as_exact_cost(fitting, monkeypatch):
    root, req, source, calls, fit = fitting
    real = cal.least_squares
    def interrupted(function, initial, **kwargs):
        function(initial)
        raise RuntimeError('Synthetic infrastructure interruption')
    monkeypatch.setattr(cal, 'least_squares', interrupted)
    with pytest.raises(RuntimeError, match='interruption'): fit()
    assert not (root/'fit/mapping_locked.json').exists()
    monkeypatch.setattr(cal, 'least_squares', real)
    result = fit()
    assert result['eligible'] is True and result['cost_count_exact'] is False
    assert result['linear_solver_calls'] == calls.count('linear')
    assert result['decision'] == fit()['decision']


def test_observed_prior_is_never_relaxed_or_cross_validated_against_rf_strength(fitting):
    root, req, source, calls, fit = fitting
    req['evidence'][0]['basis'] = 'derived'
    req['evidence'][0].pop('prediction_method')
    result = fit()
    assert result['clearance_prior_policy']['penalty_weight'] == .1
    assert result['clearance_prior_policy']['scale_bounds'] == [.25, 4.]
    assert result['development_plan']['prior_weights'] == [.1]


def test_physics_is_exactly_the_frozen_flexible_model(monkeypatch):
    calls = []
    def spy(ph, request, times, family, keys, x):
        calls.append((ph, request, times, family, keys, x)); return {'fixed': True}
    monkeypatch.setattr(cal.base, '_simulate', spy)
    for variant in cal.PARAMETERS:
        assert cal._simulate({}, {}, [1.], variant, ('a',), [0.]) == {'fixed': True}
        assert calls[-1][3] == 'evidence_flexible'
