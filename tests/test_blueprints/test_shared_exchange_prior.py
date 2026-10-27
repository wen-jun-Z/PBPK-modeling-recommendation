"""Static/grouped learning and sparse adaptation guards, with synthetic data."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pbpk_kb.blueprint_learning import shared_exchange_prior as prior
from pbpk_kb.blueprint_policy import robust_exchange_model

spec = importlib.util.spec_from_file_location('_shared_physiology_fixture',
    Path(__file__).with_name('test_perfusion_model.py'))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def request(compound, index=0):
    def evidence(name, value, unit, basis='measured'):
        return {'parameter': name, 'compound_id': compound, 'value': value, 'unit': unit,
            'basis': basis, 'context': {'species': 'human', 'route': 'intravenous', 'matrix': 'plasma'}}
    return {'compound_id': compound, 'compound_name': compound, 'partition_method': 'PKSimStandard',
        'human': {'age_years': 30., 'weight_kg': 70., 'height_cm': 175., 'sex': 'MALE', 'population': 'test'},
        'dosing': {'route': 'intravenous', 'dose_mg': 100., 'infusion_minutes': 30., 'start_h': 0.},
        'evidence': [evidence('molecular_weight', 200.+index*15., 'g/mol'),
            evidence('logp', -.5+.2*index, 'Log Units'),
            evidence('fraction_unbound', .1+.02*index, 'fraction', 'predicted'),
            evidence('total_clearance', 8./4.2+index*.1, 'ml/min/kg', 'predicted'),
            evidence('pka_types', [{'Type': 'Acid', 'Pka': 4.+.1*index}], '1')],
        'forbidden_sources': []}


@pytest.fixture
def rows():
    return [{'compound_id': f'train{i}', 'study_ids': [f'study{i}'], 'request': request(f'train{i}', i),
             'log_parameters': [.05*i, -.1*i, .12*i]} for i in range(6)]


def fit(rows):
    return prior.fit_prior(rows, excluded_compound_ids=['heldout'], excluded_study_ids=['heldout_study'])


def test_static_features_ignore_person_dose_ids_and_all_extra_clinical_keys():
    req = request('heldout')
    expected = prior.static_features(req)
    changed = deepcopy(req)
    changed.update(compound_id='renamed', compound_name='no feature', human={'arbitrary': 999},
        dosing={'unrelated': float('inf')}, target={'cmax': 1e200},
        clinical_reference={'clearance': float('nan')}, observations=[{'concentration': 1e200}],
        teacher_parameters={'partition_scale': 1000})
    for entry in changed['evidence']: entry['compound_id'] = 'renamed'
    np.testing.assert_array_equal(prior.static_features(changed), expected)
    assert expected.shape == (8,)
    assert expected.tolist() == pytest.approx([math.log(200.), -.5, math.log(.1), math.log(8./4.2), 1, 0, 4., 4.])


@pytest.mark.parametrize('bad', ['measured_fu', 'measured_cl', 'renal', 'formula', 'duplicate', 'unit', 'fu'])
def test_static_contract_rejects_clinical_clearance_and_incompatible_inputs(bad):
    req = request('heldout')
    if bad.startswith('measured'):
        key = 'fraction_unbound' if bad.endswith('fu') else 'total_clearance'
        next(e for e in req['evidence'] if e['parameter'] == key)['basis'] = 'measured'
    elif bad == 'renal':
        row = deepcopy(req['evidence'][3]); row['parameter'] = 'renal_clearance'; req['evidence'].append(row)
    elif bad == 'formula': req['partition_method'] = 'RodgersRowland'
    elif bad == 'duplicate': req['evidence'].append(deepcopy(req['evidence'][0]))
    elif bad == 'unit': req['evidence'][0]['unit'] = 'mg'
    else: req['evidence'][2]['value'] = 1.1
    with pytest.raises(ValueError): prior.static_features(req)


@pytest.mark.parametrize('kind', ['compound', 'study'])
def test_heldout_rows_are_rejected_before_decoding_requests_or_teachers(rows, kind):
    poisoned = {'compound_id': 'heldout' if kind == 'compound' else 'other',
        'study_ids': ['unrelated'] if kind == 'compound' else ['heldout_study'],
        'request': None, 'log_parameters': None}
    with pytest.raises(ValueError, match='Held-out'):
        fit(rows + [poisoned])


def test_inner_study_groups_and_scalers_exclude_every_related_compound(rows):
    # train0 and train1 share a study; a second train1 row links study2 too.
    rows[1]['study_ids'] = ['study0']
    duplicate = deepcopy(rows[1]); duplicate['study_ids'] = ['study2']
    augmented = rows + [duplicate]
    bundle = fit(augmented)
    assert len(bundle['inner_folds']) == 4
    for fold in bundle['inner_folds']:
        assert set(fold['training_compound_ids']).isdisjoint(fold['validation_compound_ids'])
        assert set(fold['training_study_ids']).isdisjoint(fold['validation_study_ids'])
        train = [augmented[i] for i in fold['training_row_indices']]
        features = np.asarray([prior.static_features(row['request']) for row in train])
        weights = prior._weights([row['compound_id'] for row in train])
        np.testing.assert_allclose(fold['scaler_mean'], np.average(features, axis=0, weights=weights))
        if 'train0' in fold['validation_compound_ids']:
            assert set(fold['validation_compound_ids']) == {'train0', 'train1', 'train2'}
    for alpha, score in bundle['inner_teacher_mse_by_alpha'].items():
        expected = np.average([fold['teacher_mse_by_alpha'][alpha] for fold in bundle['inner_folds']],
            weights=[len(fold['validation_compound_ids']) for fold in bundle['inner_folds']])
        assert score == pytest.approx(expected)
    assert bundle['teacher_is_clinical_ground_truth'] is False
    assert bundle['native_execution_required'] is False


def test_compound_equal_weight_pooled_mean_and_serializable_predictions(rows):
    duplicated = rows + [deepcopy(rows[0]) for _ in range(5)]
    bundle = fit(duplicated)
    expected = np.mean([row['log_parameters'] for row in rows], axis=0)
    np.testing.assert_allclose(bundle['pooled_mean'], expected)
    saved = json.loads(json.dumps(bundle, allow_nan=False))
    req = request('heldout', 3)
    for method in ('rf_only', 'pooled_mean', 'ridge'):
        np.testing.assert_array_equal(prior.predict_prior(bundle, req, method), prior.predict_prior(saved, req, method))
    np.testing.assert_array_equal(prior.predict_prior(bundle, req, 'rf_only'), np.zeros(3))
    np.testing.assert_allclose(prior.predict_prior(bundle, req, 'pooled_mean'), expected)
    with pytest.raises(ValueError, match='excluded compound'):
        prior.predict_prior(bundle, rows[0]['request'])


def test_exact_inner_ties_favor_strong_ridge_and_predictions_are_bounded(rows):
    for row in rows: row['log_parameters'] = [0., 0., 0.]
    bundle = fit(rows)
    assert bundle['ridge']['alpha'] == 100.
    bundle['ridge']['intercept'] = [1e6, -1e6, 1e6]
    predicted = prior.predict_prior(bundle, request('heldout'))
    np.testing.assert_allclose(np.exp(predicted), [20., .1, 100.])
    bundle['feature_names'] = ['wrong']
    with pytest.raises(ValueError, match='schema'):
        prior.predict_prior(bundle, request('heldout'))


@pytest.fixture
def five_case():
    physiology = fixture_module.physiology.__wrapped__()
    req = request('heldout')
    times = np.array([.08, .3, .5, 1.4, 4.])
    concentration = robust_exchange_model.simulate(physiology, times_h=times, dose_mg=100.,
        infusion_duration_h=.5, total_clearance_l_h=8.)['concentration_mg_l']
    document = {'compound_id': 'heldout', 'study_id': 'heldout_development',
        'role': 'calibration_data_reconstruction', 'matrix': 'plasma', 'time_unit': 'min', 'unit': 'umol/l',
        'human': deepcopy(req['human']), 'dosing': deepcopy(req['dosing']),
        'observations': [{'time_min': float(t*60), 'concentration_umol_per_l': float(c*1000/200)}
                         for t, c in zip(times, concentration)]}
    return physiology, req, document


def test_five_fit_recovers_synthetic_baseline_with_mass_audit(five_case):
    physiology, req, document = five_case
    result = prior.fit_five(physiology, req, document, np.zeros(3))
    assert result['eligible'] and result['objective'] < 1e-14
    np.testing.assert_allclose(result['log_parameters'], np.zeros(3), atol=1e-6)
    assert len(result['starts']) == 16 and result['native_solver_calls'] == 0
    assert result['linear_solver_calls'] == sum(row['linear_solver_calls'] for row in result['starts'])
    assert result['failed_evaluation_count'] == 0
    assert result['maximum_mass_balance_relative_error'] < 1e-8
    assert result['weighted_data_loss'] + result['prior_loss'] == pytest.approx(result['objective'])
    assert result['full_curve_used'] is result['external_warm_start_used'] is False
    assert result['decision']['data_jacobian_includes_prior_rows'] is False
    json.dumps(result, allow_nan=False)


def test_all_methods_share_fixed_starts_and_regularization_even_for_bolus(five_case, monkeypatch):
    physiology, req, document = five_case
    req['dosing']['infusion_minutes'] = document['dosing']['infusion_minutes'] = 0.
    observed_initials = []
    def one_evaluation(fun, x0, **kwargs):
        assert kwargs['max_nfev'] == 400
        residual = fun(x0)
        assert residual.shape == (8,)
        observed_initials.append(x0.copy())
        return SimpleNamespace(x=x0.copy(), jac=np.zeros((8, 3)), success=False,
            nfev=1, message='synthetic fixed-budget probe')
    monkeypatch.setattr(prior, 'least_squares', one_evaluation)
    a = prior.fit_five(physiology, req, document, [0., 0., 0.])
    center = np.array([.2, -.3, .4])
    b = prior.fit_five(physiology, req, document, center)
    np.testing.assert_array_equal(observed_initials[:16], observed_initials[16:])
    assert a['starts'][0]['initial_log_parameters'] == [0., 0., 0.]
    expected = .1 * np.mean((center / np.log([2., 10., 100.]))**2)
    assert b['starts'][0]['prior_loss'] == pytest.approx(expected)
    assert a['linear_solver_calls'] == b['linear_solver_calls'] == 32


@pytest.mark.parametrize('bad', ['full', 'raw', 'label', 'teacher', 'conditions', 'role', 'forbidden'])
def test_five_refuses_full_curves_hidden_labels_and_wrong_sources(five_case, bad):
    physiology, req, document = five_case
    if bad == 'full': document['observations'].append({'time_min': 300., 'concentration_umol_per_l': .1})
    elif bad == 'raw': document['raw_observations'] = [0]*100
    elif bad == 'label': document['observations'][0]['target_label'] = 999.
    elif bad == 'teacher': document['teacher_parameters'] = {'clearance_scale': 2.}
    elif bad == 'conditions': document['human']['weight_kg'] = 999.
    elif bad == 'role': document['role'] = 'heldout_clinical_evaluation'
    else:
        document['source'] = {'path': 'heldout.json', 'sha256': 'a'*64}
        req['forbidden_sources'] = [document['source']]
    with pytest.raises(ValueError):
        prior.fit_five(physiology, req, document, [0., 0., 0.])


def test_all_failed_simulations_are_retained_and_callbacks_can_interrupt(five_case, monkeypatch):
    physiology, req, document = five_case
    def failure(*args, **kwargs): raise ArithmeticError('synthetic propagation failure')
    monkeypatch.setattr(prior.robust_exchange_model, 'simulate', failure)
    result = prior.fit_five(physiology, req, document, [0., 0., 0.])
    assert not result['eligible'] and result['decision'] is None and result['log_parameters'] is None
    assert len(result['starts']) == 16
    assert result['failed_evaluation_count'] == result['linear_solver_calls'] > 0
    assert all(not row['eligible'] for row in result['starts'])
    def stop(): raise RuntimeError('stop requested')
    with pytest.raises(RuntimeError, match='stop requested'):
        prior.fit_five(physiology, req, document, [0., 0., 0.], before_trial=stop)
