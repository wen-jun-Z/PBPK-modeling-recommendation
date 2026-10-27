import math
import numpy as np
from pbpk_kb.recommendation.pbpk_match_v2.evaluation import (
    exact_rank_weights, clinical_evaluation, fit_oof_temperature, calibrate,
)


def test_positions_keep_ties_unknown_and_do_not_backfill():
    ids = ['a', 'b', 'c', 'd']
    contract = {'q': {'compound': 'x', 'publication_cluster_id': 'p',
        'endpoints': ['Cmax'], 'same_parent_ids': ids, 'errors': {
            'b': {'Cmax': {'absolute': math.log(2), 'signed': math.log(2)}},
            'c': {'Cmax': {'absolute': math.log(8), 'signed': math.log(8)}},
            'd': {'Cmax': {'absolute': 0., 'signed': 0.}}}}}
    result = clinical_evaluation([[4, 3, 3, 1]], ['q'], ids, method='x', contract=contract)
    rows = result['position_rows']
    assert rows[0]['coverage'] == 0 and rows[0]['absolute_log_error'] is None
    assert math.isclose(math.exp(rows[1]['absolute_log_error']), 4.)
    assert rows[1]['absolute_log_error'] == rows[2]['absolute_log_error']
    assert result['omissions'][0]['by_k']['3']['best_measured_miss_probability'] == 1.
    np.testing.assert_equal(exact_rank_weights([4, 3, 3, 1], 3), [0, .5, .5, 0])


def test_missing_calibration_class_is_explicit():
    p = np.array([[[.9, .05, .05], [.1, .8, .1]]])
    c = fit_oof_temperature(p, [[0, 1]], ['drug'])
    assert c['state'] == 'unavailable'
    np.testing.assert_allclose(calibrate(p, c['temperature']), p)


def test_temperature_uses_group_weighted_valid_labels():
    p = np.array([[[.4, .3, .3], [.3, .4, .3], [.3, .3, .4], [.8, .1, .1]]])
    c = fit_oof_temperature(p, [[0, 1, 2, -1]], ['a'])
    assert c['state'] == 'fitted'
    assert c['temperature'] < 1
    np.testing.assert_allclose(calibrate(p, c['temperature']).sum(axis=-1), 1.)
