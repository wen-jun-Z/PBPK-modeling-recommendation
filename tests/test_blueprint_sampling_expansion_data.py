"""Clinical-reference isolation, exact bolus math and sampler transfer checks."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import blueprint_sampling_expansion_data as data
from pbpk_kb.blueprint_policy import learned_sampling_design as learner


@pytest.fixture
def example():
    request = dict(compound_id='synthetic', human=dict(weight_kg=70., sex='MALE'),
        dosing=dict(route='intravenous', dose_mg=10., infusion_minutes=0., start_h=0.),
        evidence=[dict(parameter='molecular_weight', value=300., unit='g/mol')])
    times = [0., 1., 2., 3., 4., 6., 8.]
    concentrations = [1000 * math.exp(-.5*t) for t in times]
    reference = dict(study_id='target', compound_id='synthetic', human=request['human'],
        dosing=request['dosing'], matrix='plasma', time_unit='min', unit='umol/l',
        observations=[dict(time_min=t*60, concentration_umol_per_l=c/300)
                      for t, c in zip(times, concentrations)])
    case = dict(case_id='case', study_id='target', nca_policy={}, reference_source=reference)
    prediction = dict(time_min=[t*60 for t in times], concentration=[c/300 for c in concentrations],
                      time_unit='min', unit='umol/l')
    return case, reference, request, prediction


def test_bolus_exact_exponential_nca_without_fake_infusion():
    ts = [0., 1., 2., 3., 4., 6., 8.]
    c = [1000 * math.exp(-.5*t) for t in ts]
    result = data.observed_curve_nca(time_h=ts, concentration_ng_ml=c,
        dose_ng=1e7, infusion_h=0., weight_kg=70.)
    endpoints = result['endpoints']
    assert endpoints['auc_inf']['value'] == pytest.approx(2000.)
    assert endpoints['terminal_half_life']['value'] == pytest.approx(math.log(2)/.5)
    assert endpoints['clearance']['value'] == pytest.approx(5/70)
    assert endpoints['vz']['value'] == pytest.approx(10/70)
    assert endpoints['mrt']['value'] == pytest.approx(2.)
    assert endpoints['vss']['value'] == pytest.approx(10/70)
    assert result['diagnostics']['infusion_h'] == 0
    assert result['diagnostics']['manufactured_time_zero'] is False


def test_positive_infusion_full_schedule_matches_frozen_nca():
    from pbpk_kb.blueprints.clinical_prediction_nca import compute_clinical_prediction_nca
    ts = [0., .25, .5, 1., 2., 3., 4., 6., 8.]
    cs = [1000*(1-math.exp(-.5*t)) if t <= 1 else
          1000*(1-math.exp(-.5))*math.exp(-.5*(t-1)) for t in ts]
    new = data.observed_curve_nca(time_h=ts, concentration_ng_ml=cs, dose_ng=1e7,
                                  infusion_h=1., weight_kg=70., cutoff_ng_ml=.01)
    old = compute_clinical_prediction_nca(time_h=ts, concentration_ng_ml=cs, dose_ng=1e7,
        infusion_h=1., nominal_times_h=ts, lloq_ng_ml=.01, weight_kg=70.)
    for key in data.UNITS:
        assert new['endpoints'][key]['value'] == pytest.approx(old['endpoints'][key]['value'])


def test_without_actual_time_zero_full_exposure_references_stay_missing():
    ts = [1., 2., 3., 4., 6., 8.]
    result = data.observed_curve_nca(time_h=ts,
        concentration_ng_ml=[1000*math.exp(-.5*t) for t in ts],
        dose_ng=1e7, infusion_h=0., weight_kg=70.)
    assert result['endpoints']['auc']['value'] == pytest.approx(2000*(math.exp(-.5)-math.exp(-4)))
    assert result['endpoints']['terminal_half_life']['value'] == pytest.approx(math.log(2)/.5)
    assert all(result['endpoints'][g]['value'] is None for g in data.DEPENDENT)
    assert result['diagnostics']['window_start_h'] == 1


def test_no_tail_does_not_remove_sampled_cmax():
    result = data.observed_curve_nca(time_h=[1., 2., 3.], concentration_ng_ml=[1., 2., 3.],
        dose_ng=1e7, infusion_h=.5, weight_kg=70.)
    assert result['endpoints']['cmax']['value'] == 3
    assert result['endpoints']['tmax']['value'] == 3
    assert result['endpoints']['auc']['value'] == 4
    assert result['endpoints']['terminal_half_life']['value'] is None


def test_zero_last_point_does_not_create_zero_tail_area():
    result = data.observed_curve_nca(time_h=[0., 1., 2., 3., 4., 5., 6.],
        concentration_ng_ml=[100., 70., 40., 20., 10., 5., 0.],
        dose_ng=1e7, infusion_h=0, weight_kg=70.)
    assert result['endpoints']['terminal_half_life']['value'] is not None
    assert result['endpoints']['auc_inf']['value'] is None
    assert result['diagnostics']['window_end_h'] == 6


def test_metadata_contract_never_reads_candidate_or_target_concentrations(example):
    case, reference, req, _ = example
    first = data.build_target_contract(case, reference, req)
    hidden = deepcopy(reference)
    for row in hidden['observations']:
        del row['concentration_umol_per_l']
    hidden['endpoints'] = {'cmax': {'value': 'INVALID_HIDDEN_VALUE'}}
    assert data.build_target_contract(case, hidden, req) == first
    assert 'concentration' not in json.dumps(first)
    assert first['cutoff_ng_ml'] == 0
    assert first['cutoff_basis'] == 'not_reported_positive_points_only'


def test_metadata_legacy_clinical_curve_values_not_read(example):
    case, reference, req, _ = example
    legacy = dict(clinical_reference=dict(case_id='case', study_id='target',
        curve=dict(time_h=[r['time_min']/60 for r in reference['observations']],
                   unit='ng/mL', matrix='plasma', statistic='arithmetic_mean')))
    contract = data.build_target_contract(case, legacy, req)
    assert contract['sampling_time_h'] == [0., 1., 2., 3., 4., 6., 8.]


def test_model_only_full_prediction_never_fills_missing_reference(example):
    case, reference, req, prediction = example
    reference['observations'] = reference['observations'][1:]
    score = data.score_case(case, dict(request=req, prediction=prediction))
    assert score['endpoints']['auc_inf']['value'] == pytest.approx(2000)
    assert score['endpoints']['auc_inf']['reference'] is None
    assert score['endpoints']['auc_inf']['error'] is None
    assert score['endpoints']['cmax']['value'] == pytest.approx(1000*math.exp(-.5))
    assert score['endpoints']['cmax']['error'] == pytest.approx(1.)
    assert score['curve_metrics']['gmfe'] == pytest.approx(1.)
    assert score['reference_nca_diagnostics']['manufactured_time_zero'] is False
    assert len(score['endpoints']) == 10


def test_matched_finite_window_uses_actual_times_even_if_extra_prediction_peaks(example):
    case, reference, req, prediction = example
    reference['observations'] = reference['observations'][1:]
    prediction['time_min'].insert(1, 30.)
    prediction['concentration'].insert(1, 10000.)
    score = data.score_case(case, dict(request=req, prediction=prediction))
    assert score['endpoints']['cmax']['value'] == pytest.approx(1000*math.exp(-.5))
    assert score['endpoints']['auc']['error'] == pytest.approx(1.)


def test_score_scaling_and_tmax_absolute_error(example):
    case, _, req, prediction = example
    prediction['concentration'] = [2*c for c in prediction['concentration']]
    score = data.score_case(case, dict(request=req, prediction=prediction))
    assert score['curve_metrics']['gmfe'] == pytest.approx(2.)
    assert score['curve_metrics']['ln_rmse'] == pytest.approx(math.log(2))
    assert score['endpoints']['cmax']['error'] == pytest.approx(2.)
    assert score['endpoints']['auc_inf']['error'] == pytest.approx(2.)
    assert score['endpoints']['clearance']['error'] == pytest.approx(2.)
    assert score['endpoints']['tmax']['error'] == 0
    assert score['endpoints']['tmax']['error_metric'] == 'absolute_error_h'
    assert score['endpoints']['cmax']['reference_kind'] == 'observed_curve_nca'


def test_original_reported_scalars_separate_sensitivity_only(example):
    case, reference, req, prediction = example
    reference['endpoints'] = dict(cmax=dict(value=1.4, unit='mg/L', kind='clinical_study_reported'))
    score = data.score_case(case, dict(request=req, prediction=prediction))
    assert score['endpoints']['cmax']['reference'] == 1000.
    assert score['endpoints']['cmax']['error'] == 1.
    assert score['source_reported_sensitivity']['cmax']['reference'] == 1.4
    assert score['source_reported_sensitivity']['cmax']['error'] == pytest.approx(1.4)


def test_model_goals_has_no_clinical_values_access_and_missing_tail_is_local(example):
    case, reference, req, prediction = example
    contract = data.build_target_contract(case, reference, req)
    prediction['concentration'] = list(range(1, len(prediction['concentration'])+1))
    goals = data.model_goals(contract, prediction, req)
    assert goals['values']['cmax'] == 2100.
    assert goals['values']['terminal_half_life'] is None
    assert goals['values']['auc_inf'] is None
    assert goals['target_observation_values_used'] is False


@pytest.mark.parametrize('mutation', ['unknown', 'changed_dose', 'changed_mw', 'zero_claim'])
def test_frozen_metadata_contract_change_rejected(example, mutation):
    case, reference, req, prediction = example
    contract = data.build_target_contract(case, reference, req)
    if mutation == 'unknown':
        contract['hidden_target_values'] = [999.]
    elif mutation == 'changed_dose':
        contract['dosing']['dose_mg'] = 999.
    elif mutation == 'changed_mw':
        contract['molecular_weight_g_mol'] = 999.
    else:
        contract['actual_observed_time_zero'] = False
    with pytest.raises(ValueError):
        data.model_goals(contract, prediction, req)


def test_bolus_feature_rank_differences_invariant_to_time_reference():
    args = dict(times=[.1, .3, 2.], visible_times=[.2, .5, 1., 3.], visible_max=10., infusion_end=0.,
        concentrations=[[8., 6., 2.], [9., 7., 3.]], linear_utility=[.2, .5, .7],
        nonlinear_utility=[.1, .4, .8], goal_variance=1., c_optimal_scores=[.7, .5, .4], c_optimal_before=1.)
    bolus = data.acquisition_features(**args)
    tiny = learner.features(**{**args, 'infusion_end': .001})
    coef = np.arange(1., 9.)
    model = dict(scale=np.arange(2., 10.), coef=coef)
    a, b = learner.predict_scores(model, bolus), learner.predict_scores(model, tiny)
    assert a - a[0] == pytest.approx(b - b[0])
    assert np.all(bolus[:, -1] == 1)
    assert learner.choose_index(a, args['times']) == learner.choose_index(b, args['times'])
    assert args['infusion_end'] == 0


def test_real_infusion_features_unchanged():
    args = dict(times=[.1, .3, 2.], visible_times=[.2, .5, 1., 3.], visible_max=10., infusion_end=.5,
        concentrations=[[8., 6., 2.], [9., 7., 3.]], linear_utility=[.2, .5, .7],
        nonlinear_utility=[.1, .4, .8], goal_variance=1., c_optimal_scores=[.7, .5, .4], c_optimal_before=1.)
    assert np.array_equal(data.acquisition_features(**args), learner.features(**args))


def test_project_source_hash_and_boundaries(tmp_path):
    path = tmp_path/'source.json'; path.write_text('{"x": 1}')
    ref = dict(path='source.json', sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert data.read_reference(ref, tmp_path) == {'x': 1}
    path.write_text('{"x": 2}')
    with pytest.raises(ValueError, match='hash changed'):
        data.read_reference(ref, tmp_path)
    with pytest.raises(ValueError, match='outside project'):
        data.read_reference(dict(path='../outside.json', sha256='x'), tmp_path)


@pytest.mark.parametrize('field,value', [('infusion_h', -1.), ('dose_ng', 0.), ('weight_kg', 0.),
                                       ('cutoff_ng_ml', float('nan'))])
def test_invalid_physical_inputs_rejected(field, value):
    args = dict(time_h=[0., 1., 2., 3.], concentration_ng_ml=[10., 8., 6., 4.],
                dose_ng=1e7, infusion_h=0., weight_kg=70., cutoff_ng_ml=0.)
    args[field] = value
    with pytest.raises(ValueError):
        data.observed_curve_nca(**args)


def test_actual_zero_predose_bolus_is_not_postdose_initial_concentration():
    args = dict(time_h=[0., 1., 2., 3., 4., 5., 6.],
        concentration_ng_ml=[0., 70., 40., 20., 10., 5., 2.5], dose_ng=1e7, infusion_h=0., weight_kg=70.)
    observed = data.observed_curve_nca(**args)
    assert observed['endpoints']['terminal_half_life']['value'] is not None
    assert observed['endpoints']['auc_inf']['value'] is None
    assert observed['diagnostics']['complete_initial_condition'] is False
    predicted = data.observed_curve_nca(**args, model_initial_condition_known=True)
    assert predicted['endpoints']['auc_inf']['value'] is not None


@pytest.mark.parametrize('key,value', [('compound_id', 'other'), ('dosing', {'dose_mg': 999}),
                                    ('human', {'weight_kg': 999})])
def test_clinical_identity_and_conditions_must_match_request(example, key, value):
    case, reference, request, _ = example
    reference[key] = value
    with pytest.raises(ValueError, match='differ'):
        data.build_target_contract(case, reference, request)
