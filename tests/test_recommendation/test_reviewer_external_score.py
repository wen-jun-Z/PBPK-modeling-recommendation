import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from reviewer_external_score import derive, position_row, OUT


def test_missing_first_candidate_is_not_backfilled():
    row = position_row(np.array([3., 2., 1.]), ['unknown', 'best', 'other'],
                       {'best': 0., 'other': math.log(3)}, rank=1, time_endpoint=False)
    assert row['coverage'] == 0
    assert row['gmfe'] is None


def test_partial_tie_retains_coverage_and_conditional_error():
    row = position_row(np.array([2., 2., 1.]), ['a', 'b', 'c'],
                       {'b': math.log(2), 'c': 0.}, rank=1, time_endpoint=False)
    assert row['coverage'] == .5
    assert row['gmfe'] == pytest.approx(2.)
    assert row['tie_size'] == 2


def test_tmax_is_absolute_hours_not_fold_error():
    row = position_row(np.array([1., 0.]), ['a', 'b'], {'a': .41, 'b': 0.}, rank=1, time_endpoint=True)
    assert row['gmfe'] is None
    assert row['tmax_mae_h'] == .41
    assert row['within_twofold'] is None


def test_nca_uses_locked_curve_and_recovers_exponential_tail():
    t = [0., 1., 2., 4., 8., 12., 24., 36.]
    c = [0.] + [100 * math.exp(-.1 * (x-1)) for x in t[1:]]
    prediction = {'time_h': t, 'concentration_ng_ml': c, 'sampling_time_h': t, 'sampled_concentration_ng_ml': c}
    scenario = {'normalized_design': {'regimen': {'dose_times_h': [0.]}, 'dose': {'value': 10.}, 'sampling': {'lloq_ng_ml': .1}}}
    values, unavailable, diagnostics = derive(prediction, scenario)
    assert values['Cmax'] == 100
    assert values['Tmax'] == 1
    assert values['terminal_half_life'] == pytest.approx(math.log(2)/.1)
    assert values['AUC_inf'] == pytest.approx(1050.)
    assert not unavailable
    assert diagnostics['observed_values_used'] is False


def test_repeated_dose_ambiguity_cannot_silently_use_single_dose_nca():
    prediction = {'time_h': [0, 1], 'concentration_ng_ml': [1, 2], 'sampling_time_h': [0, 1], 'sampled_concentration_ng_ml': [1, 2]}
    with pytest.raises(ValueError, match='must not reach scoring'):
        derive(prediction, {'normalized_design': {'regimen': {'dose_times_h': [0, 12]}}})


def test_all_tasks_and_unavailable_results_retained_separately():
    report = json.loads((OUT / 'report/results.json').read_text())
    assert len(report['candidate_results']) == 25
    assert len(report['positions']) == 5 * 5 * 4 * 3
    assert report['native_status_counts'] == {'prediction_locked': 16, 'blocked': 9}
    preg = next(r for r in report['membership'] if r['compound'] == 'pregabalin')
    assert (preg['qualified_versions'], preg['qualified_families']) == (2, 1)
    assert preg['independent_family_choice_possible'] is False
    assert not any(r['compound'] == 'metformin' for r in report['positions'])
    assert all(not r['endpoints'] for r in report['candidate_results'] if r['compound'] == 'metformin')
    assert len({r['candidate_id'] for r in report['candidate_results'] if r['compound'] == 'pregabalin' and r['status'] == 'prediction_locked'}) == 2
