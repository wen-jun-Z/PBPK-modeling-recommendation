"""Subset isolation, sparse identifiability and unchanged strong analytic controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import blueprint_sparse_classical as sparse
import blueprint_validation_classical as frozen


@pytest.fixture
def synthetic(tmp_path):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip('Use a --basetemp inside project runtime/test-tmp')
    def save(name, obj):
        p = tmp_path/name
        p.write_text(json.dumps(obj))
        return {'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    human = {'age_years': 30., 'weight_kg': 70., 'height_cm': 175., 'sex': 'MALE',
             'population': 'European_ICRP_2002', 'seed': 17}
    dose = {'route': 'intravenous', 'dose_mg': 10., 'start_h': 0., 'infusion_minutes': 30.}
    source = save('evidence.json', {'role': 'synthetic independent CL'})
    mw = save('mw.json', {'identity': {'molecular_weight_g_mol': 400.}})
    request = {'compound_id': 'synthetic', 'human': human, 'dosing': dose,
        'target': {'window_start_h': 0., 'window_end_h': 8., 'terminal_fit': None},
        'evidence': [{'parameter': 'total_clearance', 'compound_id': 'synthetic',
            'value': 2., 'unit': 'ml/min/kg', 'source': source, 'basis': 'predicted'}],
        'evidence_ledger': source, 'forbidden_sources': []}
    case = {'case_id': 'sparse_synthetic', 'compound_id': 'synthetic', 'drug': 'Synthetic',
        'study_id': 'different_target', 'physchem': mw,
        'development': 'full_curve_must_not_be_read.json',
        'reference': 'target_outcomes_must_not_be_read.json',
        'target_request_metadata': {'human': dict(human, weight_kg=80.),
            'dosing': dict(dose, dose_mg=20., infusion_minutes=60.), 'window_end_h': 8.},
        'nca_policy': {'sampling_time_h': [0., .5, 1., 2., 8.]}}
    def training(times, unit='umol/l'):
        times = np.asarray(times)
        concentrations = sparse.concentration_mg_l([8., 10., 15., 30.], times,
                                                   **sparse.dosing_kwargs(dose))
        doc = {'study_id': 'synthetic_training', 'compound_id': 'synthetic',
            'time_unit': 'min', 'unit': unit, 'molecular_weight_g_mol': 400.,
            'human': human, 'dosing': dose, 'observations': [
                {'time_min': float(t*60), 'concentration_umol_per_l': float(c*1000/400)}
                for t, c in zip(times, concentrations)]}
        return save(f'training_{len(times)}_{unit.replace("/", "_")}.json', doc)
    return case, request, training, tmp_path


def test_three_observations_fit_without_inventing_data_and_predict_changed_conditions(synthetic):
    case, request, training, folder = synthetic
    source = training([.1, .5, 6.])
    result = sparse.fit_predict(case=case, request=request, training_source=source,
        output_dir=folder/'three', project_root=ROOT, with_cl_prior=False)
    fit = json.loads((ROOT/result['fit']['path']).read_text())
    curve = json.loads((ROOT/result['prediction']['path']).read_text())
    assert result['status'] == 'predicted' and not result['target_outcomes_used']
    assert len(fit['starts']) == 16
    assert fit['optimizer']['max_nfev_per_start'] == 400
    assert len(fit['development_curve']['observed_mg_l']) == 3
    assert fit['initialization']['training_observations'] == 3
    assert not fit['initialization']['old_fit_or_full_curve_read']
    ident = fit['identifiability']
    assert ident['underdetermined_by_observation_count']
    assert ident['data_rank_deficient'] and ident['data_only']['rank'] <= 3
    assert ident['data_only']['condition_number'] is None
    assert ident['with_regularizer'] is None
    assert ident['jacobian_source'] == 'selected scipy least_squares data rows'
    assert fit['quality_checks']['analytic_auc_mass_balance_relative_error'] < 1e-8
    assert fit['development_metrics']['log_mse'] < 1e-12
    fitted = np.array([fit['target_parameters'][k] for k in sparse.PARAMETERS])
    hours = np.asarray(curve['time_min'])/60
    expected = sparse.concentration_mg_l(fitted, hours,
        **sparse.dosing_kwargs(case['target_request_metadata']['dosing']))
    np.testing.assert_allclose(curve['concentration'], expected*1000/400, rtol=1e-14)
    assert curve['unit'] == 'umol/l' and curve['time_unit'] == 'min'
    assert (np.diff(hours) > 0).all()
    replay = sparse.fit_predict(case=case, request=request, training_source=source,
        output_dir=folder/'three', project_root=ROOT, with_cl_prior=False)
    assert replay['reused'] and replay['this_invocation_residual_calls'] == 0
    (ROOT/result['prediction']['path']).write_text('{"tampered": true}')
    with pytest.raises(ValueError, match='Frozen source changed'):
        sparse.fit_predict(case=case, request=request, training_source=source,
            output_dir=folder/'three', project_root=ROOT, with_cl_prior=False)


@pytest.mark.parametrize('with_prior', [False, True])
def test_default_five_point_results_match_frozen_control(synthetic, with_prior):
    case, request, training, folder = synthetic
    source = training([.1, .5, 1., 2., 6.])
    args = dict(case=case, request=request, training_source=source,
                project_root=ROOT, with_cl_prior=with_prior)
    old = frozen.fit_predict(**args, output_dir=folder/'frozen')
    new = sparse.fit_predict(**args, output_dir=folder/'new')
    oldfit = json.loads((ROOT/old['fit']['path']).read_text())
    newfit = json.loads((ROOT/new['fit']['path']).read_text())
    np.testing.assert_allclose(list(newfit['development_parameters'].values()),
                               list(oldfit['development_parameters'].values()), rtol=1e-12)
    assert newfit['selected_objective'] == pytest.approx(oldfit['selected_objective'], abs=1e-14)
    assert new['resources']['analytic_cpu_residual_calls'] == old['resources']['analytic_cpu_residual_calls']


def test_sparse_prior_multiplier_is_explicit_and_locked(synthetic):
    case, request, training, folder = synthetic
    source = training([.1, .5, 6.]); multiplier = 10/3
    args = dict(case=case, request=request, training_source=source,
                output_dir=folder/'weighted', project_root=ROOT, with_cl_prior=True)
    result = sparse.fit_predict(**args, prior_multiplier=multiplier)
    fit = json.loads((ROOT/result['fit']['path']).read_text())
    prior = fit['clearance_prior']
    assert prior['value_L_h'] == pytest.approx(8.4)
    assert prior['weight'] == pytest.approx(.1*multiplier)
    cl = fit['development_parameters']['CL_L_h']
    objective = fit['development_metrics']['log_mse'] + .1*multiplier*(np.log(cl/8.4)/np.log(2))**2
    assert fit['selected_objective'] == pytest.approx(objective, abs=1e-14)
    assert fit['identifiability']['data_only']['rank'] <= 3
    assert fit['identifiability']['with_regularizer']['rank'] <= 4
    assert fit['identifiability']['observations'] == 3
    with pytest.raises(ValueError, match='different'):
        sparse.fit_predict(**args, prior_multiplier=1.)


def test_residual_formula_and_invalid_weights():
    p = np.log([16.8, 10., 15., 30.]); t = np.array([.1, .5, 6.])
    kwargs = dict(dose_mg=10., infusion_h=.5, start_h=0.)
    obs = sparse.concentration_mg_l(np.exp(p), t, **kwargs)/1.5
    for mult in [1., 2., 10/3]:
        r = sparse._residual_vector(p, t, obs, kwargs, 8.4, mult)
        assert np.sum(r*r) == pytest.approx(np.log(1.5)**2+.1*mult, abs=1e-14)
    for bad in [True, 0., -1., float('inf'), float('nan'), '2']:
        with pytest.raises(ValueError, match='prior_multiplier'):
            sparse._valid_multiplier(bad, True)
    with pytest.raises(ValueError, match='requires'):
        sparse._valid_multiplier(2., False)


def test_wrong_units_and_too_few_points_fail_without_fitting(synthetic):
    case, request, training, folder = synthetic
    for source in [training([.1, .5, 6.], unit='mg/L'), training([.1, .5])]:
        with pytest.raises(ValueError, match='units|three'):
            sparse.fit_predict(case=case, request=request, training_source=source,
                output_dir=folder/'invalid', project_root=ROOT, with_cl_prior=False)
    assert not (folder/'invalid'/'fit.json').exists()
