"""Synthetic analytic curves test numerical contracts, not clinical performance."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprints.full_endpoint_evaluation import (
    ENDPOINTS, qualify_endpoint_references, score_full_endpoints,
)


def case(*, infusion=False, end=24., step=.1):
    times = np.arange(0., end+step/2, step)
    # Synthetic one-compartment model: dose=100 mg, V=10 L, CL=5 L/h.
    if infusion:
        concentrations = np.where(times <= .5, 40*(1-np.exp(-.5*times)),
                                  40*(1-math.exp(-.25))*np.exp(-.5*(times-.5)))
    else:
        concentrations = 10*np.exp(-.5*times)
    molar = concentrations*10  # MW=100 g/mol.
    fit = {'start_h': end/2, 'end_h': end, 'min_points': 4,
           'min_adjusted_r2': .98, 'min_span_half_lives': 1.,
           'max_auc_extrapolated_fraction': .2, 'max_aumc_extrapolated_fraction': .2}
    request = {'compound_id': 'synthetic:test', 'dosing': {'route': 'intravenous',
        'dose_mg': 100., 'start_h': 0., 'infusion_minutes': 30. if infusion else 0.},
        'evidence': [{'parameter': 'molecular_weight', 'value': 100., 'unit': 'g/mol'}],
        'target': {'terminal_fit': fit}}
    observed = {'compound_id': 'synthetic:test', 'matrix': 'plasma', 'time_unit': 'min', 'unit': 'umol/l',
        'source': {'description': 'Synthetic analytic unit-test fixture; not clinical evidence'},
        'observations': [{'time_min': float(t*60), 'concentration_umol_per_l': float(c)} for t, c in zip(times, molar)]}
    prediction = {'time_unit': 'min', 'unit': 'umol/l', 'time_min': (times*60).tolist(), 'concentration': molar.tolist()}
    return observed, request, prediction


INITIAL = {'method': 'single_infusion_zero',
    'source_url': 'https://onlinehelp.certara.com/phoenix/8.3/topics/ncacomprules.htm',
    'rationale': 'Synthetic single-dose infusion without pre-existing drug; test of an explicitly declared boundary assumption'}
BOLUS_INITIAL = {**INITIAL, 'method': 'single_bolus_back_extrapolation',
    'rationale': 'Synthetic single IV bolus; initial log-linear decline permits explicitly declared NCA back-extrapolation'}


def test_complete_bolus_references_recover_analytic_pk_and_no_data_mutation():
    o, r, p = case(); before = deepcopy((o, r, p))
    reference = qualify_endpoint_references(o, r)
    assert reference['all_11_qualified'] and reference['qualified_endpoint_count'] == 11
    endpoints = reference['endpoints']
    expected = {'auc_inf': 20., 'clearance': 5., 'vz': 10., 'mrt': 2., 'vss': 10., 'terminal_half_life': math.log(2)/.5}
    for name, value in expected.items():
        assert endpoints[name]['value'] == pytest.approx(value, rel=.001)
    result = score_full_endpoints(o, r, p)
    assert result['status'] == 'completed' and result['scored_endpoints'] == 11
    assert result['endpoints']['plasma_concentration_time']['gmfe'] == 1.
    assert result['endpoints']['plasma_concentration_time']['log_rmse'] == 0.
    assert result['endpoints']['tmax']['absolute_error_h'] == 0.
    assert result['endpoints']['tmax']['fold_error'] is None
    assert (o, r, p) == before


def test_twofold_concentration_scaling_gives_correct_inverted_clearance_and_volume_errors():
    o, r, p = case(); p['concentration'] = [2*x for x in p['concentration']]
    result = score_full_endpoints(o, r, p)
    assert result['reference_qualification'] == qualify_endpoint_references(o, r)
    assert result['endpoints']['plasma_concentration_time']['gmfe'] == pytest.approx(2.)
    assert result['endpoints']['plasma_concentration_time']['log_rmse'] == pytest.approx(math.log(2))
    for name in ('auc', 'cmax', 'c_last', 'auc_inf', 'clearance', 'vz', 'vss'):
        assert result['endpoints'][name]['fold_error'] == pytest.approx(2.)
    for name in ('mrt', 'terminal_half_life'):
        assert result['endpoints'][name]['fold_error'] == pytest.approx(1.)
    assert result['endpoints']['clearance']['prediction'] == pytest.approx(result['endpoints']['clearance']['reference']/2)


def test_measured_infusion_zero_is_used_by_nca_but_excluded_from_log_error():
    o, r, p = case(infusion=True)
    result = score_full_endpoints(o, r, p)
    assert result['scored_endpoints'] == 11
    curve = result['endpoints']['plasma_concentration_time']
    assert curve['observed_point_count'] == 241
    assert curve['positive_points_scored'] == 240
    assert curve['zero_observations_excluded_from_log_error'] == 1
    assert not curve['nca_boundary_excluded_from_point_count']
    assert result['reference_qualification']['initial_condition']['boundary_is_observed']
    assert result['endpoints']['mrt']['reference'] == pytest.approx(2., rel=.001)


def test_missing_dose_time_is_strictly_unavailable_for_all_total_nca_endpoints():
    o, r, p = case(infusion=True); o['observations'] = o['observations'][1:]
    reference = qualify_endpoint_references(o, r)
    assert not reference['all_11_qualified']
    assert set(reference['unavailable_endpoints']) == {'auc_inf', 'clearance', 'vz', 'mrt', 'vss'}
    assert reference['qualified_endpoint_count'] == 6
    result = score_full_endpoints(o, r, p)
    assert result['scored_endpoints'] == 6


def test_explicit_initial_condition_is_separate_from_actual_observations():
    o, r, p = case(infusion=True); o['observations'] = o['observations'][1:]
    before = deepcopy(o)
    result = score_full_endpoints(o, r, p, initial_condition=INITIAL)
    assert result['scored_endpoints'] == 11
    assert o == before
    reference = result['reference_qualification']; boundary = reference['initial_condition']
    assert boundary['boundary_added'] and not boundary['boundary_is_observed']
    assert reference['observed_point_count'] == 240
    assert result['endpoints']['plasma_concentration_time']['positive_points_scored'] == 240
    first_area = .1*o['observations'][0]['concentration_umol_per_l']*.1/2
    assert boundary['first_interval_auc_mg_h_l'] == pytest.approx(first_area)
    assert 0 < boundary['first_interval_fraction_of_auc_last'] < .01
    assert result['endpoints']['auc']['reference_type'] == 'observed_curve_trapezoid'
    assert result['endpoints']['auc_inf']['reference_type'].endswith('declared_initial_condition')
    assert result['endpoints']['auc_inf']['fold_error'] == pytest.approx(1.)


def test_nca_boundary_requires_model_coverage_and_does_not_fabricate_prediction_zero():
    o, r, p = case(infusion=True); o['observations'] = o['observations'][1:]
    p['time_min'] = p['time_min'][1:]; p['concentration'] = p['concentration'][1:]
    result = score_full_endpoints(o, r, p, initial_condition=INITIAL)
    assert not result['prediction_available'] and result['scored_endpoints'] == 0
    assert 'extrapolation' in result['prediction_failure_reason']
    o, r, p = case(infusion=True); o['observations'] = o['observations'][1:]; p['concentration'][0] = 1.
    result = score_full_endpoints(o, r, p, initial_condition=INITIAL)
    assert result['endpoints']['auc_inf']['status'] == 'prediction_unavailable'
    assert 'zero predose' in result['endpoints']['auc_inf']['prediction_reason']


def test_bolus_extrapolation_uses_same_actual_times_independently_without_changing_sampled_endpoints():
    o, r, p = case(); o['observations'] = o['observations'][1:]
    before = deepcopy(o)
    strict = qualify_endpoint_references(o, r)
    p['concentration'] = [2*x for x in p['concentration']]
    p['concentration'][0] = 999.  # Dense-grid C0 must not replace matched back-extrapolation.
    result = score_full_endpoints(o, r, p, initial_condition=BOLUS_INITIAL)
    assert result['scored_endpoints'] == 11 and o == before
    ref = result['reference_qualification']; boundary = ref['initial_condition']
    assert boundary['concentration_umol_per_l'] == pytest.approx(100.)
    assert boundary['sample_indices'] == [0, 1]
    assert boundary['sample_times_min'] == pytest.approx([6., 12.])
    assert not boundary['boundary_is_observed'] and ref['observed_point_count'] == 240
    assert 0 < boundary['first_interval_fraction_of_auc_last'] < .1
    assert result['prediction_initial_condition']['concentration_umol_per_l'] == pytest.approx(200.)
    assert result['prediction_initial_condition']['sample_times_min'] == boundary['sample_times_min']
    for name in ('auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life'):
        assert ref['endpoints'][name] == strict['endpoints'][name]
    assert result['endpoints']['cmax']['reference'] < 10.
    assert result['endpoints']['plasma_concentration_time']['gmfe'] == pytest.approx(2.)
    for name in ('auc_inf', 'clearance', 'vz', 'vss'):
        assert result['endpoints'][name]['fold_error'] == pytest.approx(2.)
    assert result['endpoints']['tmax']['absolute_error_h'] == 0.


def test_bolus_back_extrapolation_needs_only_actual_scoring_coverage():
    o, r, p = case(); o['observations'] = o['observations'][1:]
    p['time_min'] = p['time_min'][1:]; p['concentration'] = p['concentration'][1:]
    result = score_full_endpoints(o, r, p, initial_condition=BOLUS_INITIAL)
    assert result['scored_endpoints'] == 11
    assert result['endpoints']['auc_inf']['fold_error'] == pytest.approx(1.)


def test_invalid_predicted_bolus_decline_keeps_curve_scores_but_never_borrows_reference_c0():
    o, r, p = case(); o['observations'] = o['observations'][1:]
    p['concentration'][1] = p['concentration'][2]/2
    result = score_full_endpoints(o, r, p, initial_condition=BOLUS_INITIAL)
    assert result['prediction_available'] and result['scored_endpoints'] == 6
    assert result['prediction_initial_condition']['status'] == 'not_estimable'
    assert result['endpoints']['plasma_concentration_time']['gmfe'] is not None
    for name in ('auc_inf', 'clearance', 'vz', 'mrt', 'vss'):
        assert result['endpoints'][name]['status'] == 'prediction_unavailable'
        assert 'strictly declining' in result['endpoints'][name]['prediction_reason']


@pytest.mark.parametrize('change', ['rising', 'flat', 'infusion', 'existing_zero', 'overflow'])
def test_unsafe_bolus_back_extrapolation_has_no_fallback(change):
    o, r, _ = case(infusion=change == 'infusion')
    if change != 'existing_zero': o['observations'] = o['observations'][1:]
    if change == 'rising': o['observations'][0]['concentration_umol_per_l'] = 1.
    if change == 'flat': o['observations'][0]['concentration_umol_per_l'] = o['observations'][1]['concentration_umol_per_l']
    if change == 'overflow':
        o['observations'][0]['concentration_umol_per_l'] = 1e308
        o['observations'][1]['concentration_umol_per_l'] = 1e-300
    with pytest.raises(ValueError):
        qualify_endpoint_references(o, r, initial_condition=BOLUS_INITIAL)


@pytest.mark.parametrize('change', ['bolus', 'existing_zero', 'missing_rationale', 'wrong_method'])
def test_unsafe_initial_conditions_are_rejected(change):
    o, r, _ = case(infusion=change != 'bolus'); initial = deepcopy(INITIAL)
    if change != 'existing_zero': o['observations'] = o['observations'][1:]
    if change == 'missing_rationale': initial['rationale'] = ''
    if change == 'wrong_method': initial['method'] = 'bolus_backextrapolation'
    with pytest.raises(ValueError): qualify_endpoint_references(o, r, initial_condition=initial)


def test_aumc_gate_remains_stricter_than_auc_gate():
    o, r, _ = case(end=4.)
    reference = qualify_endpoint_references(o, r)
    assert set(reference['unavailable_endpoints']) == {'mrt', 'vss'}
    assert reference['endpoints']['auc_inf']['status'] == 'qualified'
    assert 'AUMC tail fraction' in reference['endpoints']['vss']['reason']


def test_source_disqualification_cannot_be_overridden_by_good_numeric_tail():
    o, r, _ = case(); o.update(terminal_phase_qualified=False, terminal_phase_reason='Source review did not establish the terminal phase')
    reference = qualify_endpoint_references(o, r)
    assert set(reference['unavailable_endpoints']) == {'terminal_half_life', 'auc_inf', 'clearance', 'vz', 'mrt', 'vss'}
    assert reference['endpoints']['terminal_half_life']['value'] is None


def test_prediction_failure_is_retained_for_all_eleven_endpoints():
    o, r, _ = case()
    result = score_full_endpoints(o, r, None, prediction_failure_reason='Native execution failed')
    assert result['attempted_endpoints'] == 11 and result['scored_endpoints'] == 0
    assert set(result['endpoints']) == set(ENDPOINTS)
    assert all(v['status'] == 'prediction_unavailable' and v['fold_error'] is None for v in result['endpoints'].values())


def test_molecular_weight_units_must_be_explicit():
    o, r, _ = case(); r['evidence'][0]['unit'] = 'kg/mol'
    with pytest.raises(ValueError, match='g/mol'): qualify_endpoint_references(o, r)


def test_valid_prediction_has_no_stale_failure_reason():
    o, r, p = case()
    result = score_full_endpoints(o, r, p, prediction_failure_reason='Earlier unavailable run')
    assert result['prediction_failure_reason'] is None and result['scored_endpoints'] == 11


def test_no_log_epsilon_or_silent_terminal_window_change():
    o, r, p = case(); p['concentration'][3] = 0.
    result = score_full_endpoints(o, r, p)
    assert not result['prediction_available'] and 'no log floor' in result['prediction_failure_reason']
    changed = {**r['target']['terminal_fit'], 'start_h': 13.}
    with pytest.raises(ValueError, match='freeze'):
        qualify_endpoint_references(o, r, terminal_fit=changed)


@pytest.mark.parametrize('change', ['negative', 'duplicate_time', 'wrong_matrix', 'wrong_identity', 'censored'])
def test_invalid_observation_contract_is_rejected(change):
    o, r, _ = case()
    if change == 'negative': o['observations'][0]['concentration_umol_per_l'] = -1.
    if change == 'duplicate_time': o['observations'][1]['time_min'] = 0.
    if change == 'wrong_matrix': o['matrix'] = 'serum'
    if change == 'wrong_identity': o['compound_id'] = 'another:test'
    if change == 'censored': o['observations'][3]['censored'] = True
    with pytest.raises(ValueError): qualify_endpoint_references(o, r)
