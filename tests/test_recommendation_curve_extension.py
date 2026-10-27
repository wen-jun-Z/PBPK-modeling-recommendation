"""Scientifically material contracts of the concentration-curve extension."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pbpk_match_light_curves import curve_errors, positions, score_position


def test_curve_error_uses_all_matched_timepoints_not_peak_only():
    case = {'observation': {'time_h': [1., 2., 3.], 'concentration_ng_ml': [1., 2., 1.]}}
    curve = {'time_h': [1., 2., 3.], 'concentration_ng_ml': [4., 2., .25],
             'gmfe': math.exp(2 * math.log(4) / 3), 'log_rmse': math.sqrt(2 / 3) * math.log(4)}
    measured = curve_errors(case, curve)
    assert measured['curve_gmfe'] == pytest.approx(4 ** (2 / 3))
    assert measured['within_twofold_fraction'] == pytest.approx(1 / 3)
    assert measured['qualified_point_count'] == 3


def test_exact_ranks_do_not_backfill_and_partial_ties_keep_coverage():
    ids = ['unmeasured_top', 'measured_tie', 'unmeasured_tie', 'measured_later']
    candidates = {cid: {'family_name': cid, 'family_id': cid, 'compounds': ['drug']} for cid in ids}
    case = {'compound': 'drug', 'scenario_id': 'case', 'publication_cluster_id': 'study',
            'observation': {'time_h': [1, 2]},
            'curves': [{'candidate_id': ids[1], 'label': 'A'}, {'candidate_id': ids[3], 'label': 'B'}]}
    result = positions(np.array([10., 9., 9., 8.]), ids, candidates, case)
    assert [x['curve_coverage'] for x in result] == [0., .5, .5]
    metrics = {ids[1]: {'mean_abs_log_error': math.log(4), 'mean_squared_log_error': math.log(4) ** 2,
                        'within_twofold_fraction': 0.},
               ids[3]: {'mean_abs_log_error': 0., 'mean_squared_log_error': 0., 'within_twofold_fraction': 1.}}
    ranking = {'method_id': 'example', 'method': 'Example'}
    assert score_position(case, ranking, result[0], metrics)['curve_gmfe'] is None
    for position in result[1:]:
        row = score_position(case, ranking, position, metrics)
        assert row['curve_gmfe'] == pytest.approx(4.)
        assert row['curve_coverage'] == .5
        assert row['curve_labels'] == ['A']


def test_curve_metric_mismatch_fails_instead_of_silently_replacing_frozen_value():
    case = {'observation': {'time_h': [1., 2.], 'concentration_ng_ml': [1., 1.]}}
    curve = {'time_h': [1., 2.], 'concentration_ng_ml': [2., 2.], 'gmfe': 1., 'log_rmse': 0.}
    with pytest.raises(ValueError, match='differs from the locked result'):
        curve_errors(case, curve)
