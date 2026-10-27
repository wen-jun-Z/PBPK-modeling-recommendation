"""Protect endpoint definitions, units, unknowns, exact ties and error estimands."""
from pathlib import Path
import sys
import math

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recommendation_endpoint_extension_metrics import (descriptor, exact_position, hierarchical,
                                                        match_pair, normalized_unit, summarize_positions)


def scalar(endpoint='CL', value=1., unit='L/h', **extra):
    return {'scenario_id': 'case', 'publication_cluster_id': 'study', 'endpoint': endpoint,
            'quantity': endpoint, 'value': value, 'unit': unit, 'eligibility': True,
            'steady_state': False, 'body_weight_normalized': False, **extra}


def test_cl_unit_conversion_without_changing_apparent_clearance():
    result = match_pair(scalar(value=3.), scalar(value=50., unit='mL/min'))
    assert result['scorable'] and result['absolute_log_error'] == pytest.approx(0.)
    result = match_pair(scalar(quantity='CL'), scalar(quantity='CL/F'))
    assert not result['scorable'] and result['reason'] == 'different_quantity'


def test_fixed_auc_windows_and_steady_state_are_never_mixed():
    base = scalar('AUC_window', 100., 'ng*h/mL', window_h=[0, 24])
    assert not match_pair(base, {**base, 'window_h': [0, 72]})['scorable']
    assert not match_pair(base, {**base, 'steady_state': True})['scorable']
    assert descriptor(base, {'route': 'oral'})['endpoint_key'] != descriptor({**base, 'window_h': [0, 72]}, {'route': 'oral'})['endpoint_key']


def test_tmax_reports_absolute_hours_and_accepts_zero_time():
    result = match_pair(scalar('Tmax', 0., 'h'), scalar('Tmax', 90., 'min'))
    assert result['scorable'] and result['absolute_error_h'] == 1.5
    assert result['absolute_log_error'] is None and result['within_twofold'] is None


def test_missing_observations_do_not_become_zero_errors():
    result = match_pair(None, scalar())
    assert not result['scorable'] and result['absolute_log_error'] is None
    result = match_pair(scalar(value=None), scalar())
    assert not result['scorable'] and result['absolute_log_error'] is None


def test_exact_tie_is_conditioned_on_coverage_without_backfill():
    pairs = {'a': {'scorable': True, 'absolute_log_error': math.log(4)},
             'later': {'scorable': True, 'absolute_log_error': 0.}}
    ids = ['unknown_top', 'a', 'unknown_tie', 'later']
    scores = np.array([10., 9., 9., 8.])
    assert exact_position(scores, ids, pairs, 1)['absolute_log_error'] is None
    for rank in (2, 3):
        row = exact_position(scores, ids, pairs, rank)
        assert row['coverage'] == .5 and row['absolute_log_error'] == pytest.approx(math.log(4))


def test_molar_conversion_requires_known_molecular_weight():
    with pytest.raises(ValueError, match='molecular_weight_required'):
        normalized_unit(2., 'nM')
    assert normalized_unit(2., 'nM', 500.) == (1., 'ng/mL')


def test_equal_compounds_and_publications_do_not_overweight_dose_branches():
    rows = [{'compound': 'a', 'publication_cluster_id': 'a1', 'error': 0.},
            {'compound': 'a', 'publication_cluster_id': 'a1', 'error': 0.},
            {'compound': 'a', 'publication_cluster_id': 'a2', 'error': 4.},
            {'compound': 'b', 'publication_cluster_id': 'b1', 'error': 10.}]
    assert hierarchical(rows, 'error') == 6.


def test_zero_eligible_reference_has_zero_coverage_but_null_error():
    definition = {'endpoint_key': 'm', 'endpoint': 'Tmax', 'quantity': 'Tmax'}
    row = {'method_id': 'own', 'endpoint_key': 'm', 'analysis_stratum': 'main',
           'prediction_kind': 'protocol_matched', 'rank': 1, 'reference_eligible': False,
           'scenario_id': 'q', 'compound': 'a', 'publication_cluster_id': 'p', 'coverage': 0.,
           'fully_evaluable': False, 'absolute_log_error': None, 'absolute_error_h': None, 'within_twofold': None}
    result = summarize_positions([row], [{'method_id': 'own'}], {'m': definition})[0]
    assert result['eligible_reference_scenarios'] == 0
    assert result['coverage_eligible_reference'] == result['coverage_total41'] == 0.
    assert result['gmfe'] is None and result['mae_h'] is None


def test_auc_last_approximation_is_explicit_and_never_aliased_to_fixed_auc():
    observed = scalar('AUC_last', 100., 'ng*h/mL')
    prediction = scalar('AUC_last', 120., 'ng*h/mL', window_h=[0, 48])
    assert match_pair(observed, prediction)['scorable']
    definition = descriptor(observed, {'route': 'oral'})
    assert 'individual observed tlast is unavailable' in definition['comparison_qualification']
    assert not match_pair(observed, {**prediction, 'endpoint': 'AUC_window'})['scorable']
