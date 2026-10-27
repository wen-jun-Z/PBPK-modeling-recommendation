"""Synthetic statistical invariants; never open clinical observations."""
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from recommendation_execution_statistics import numerical_miss_audit, paired_method_statistics


def test_uniform_tie_omission_and_unknown_bounds():
    result = numerical_miss_audit(['a', 'b', 'c', 'outside'], ['a', 'b', 'c'],
                                 {'a': 1, 'b': 1, 'c': 0, 'outside': 2}, {'a': 0., 'b': math.log(3)}, ks=(1, 2))
    assert result['unmeasured_same_parent_count'] == 1
    assert result['by_k']['1']['best_measured_miss_probability'] == 1
    two = result['by_k']['2']['thresholds']['2.0']
    assert two['known_numeric_recall'] == .5
    assert two['exact_same_parent_numeric_recall'] is None
    assert two['recall_bounds_allowing_all_unknown_completions']['lower'] == .25
    assert two['recall_bounds_allowing_all_unknown_completions']['upper'] == .5
    assert two['missed_or_partly_missed_known_candidates'] == [{'candidate_id': 'a', 'exclusion_probability': .5}]


def test_no_usable_candidate_does_not_mean_perfect_recall():
    empty = numerical_miss_audit(['a', 'b'], ['a'], {'a': 0, 'b': 1}, {})
    assert empty['state'] == 'unavailable'
    assert empty['by_k']['1']['thresholds']['2.0']['known_numeric_recall'] is None
    assert empty['by_k']['1']['best_measured_hit_probability'] is None
    bad = numerical_miss_audit(['a'], ['a'], {'a': 1}, {'a': math.log(4)})
    assert bad['complete_same_parent_numeric_pool']
    assert bad['by_k']['1']['thresholds']['2.0']['exact_same_parent_numeric_recall'] is None


def test_full_pool_and_tied_best_are_not_id_ordered():
    first = numerical_miss_audit(['a', 'b', 'c'], ['a', 'b', 'c'], {'a': 2, 'b': 2, 'c': 2}, {'a': 0., 'b': 0., 'c': math.log(4)}, ks=(1, 2))
    second = numerical_miss_audit(['c', 'b', 'a'], ['a', 'b', 'c'], {'a': 2, 'b': 2, 'c': 2}, {'a': 0., 'b': 0., 'c': math.log(4)}, ks=(1, 2))
    assert first == second
    assert first['by_k']['1']['best_measured_hit_probability'] == pytest.approx(2 / 3)
    assert first['by_k']['2']['best_measured_hit_probability'] == 1
    assert first['by_k']['1']['thresholds']['2.0']['exact_same_parent_numeric_recall'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('errors', [{'outside': .1}, {'a': float('nan')}, {'a': -.1}, {'a': True}])
def test_reject_bad_numeric_truth(errors):
    with pytest.raises(ValueError):
        numerical_miss_audit(['a', 'outside'], ['a'], {'a': 1, 'outside': 0}, errors)


def rows(n=10, *, duplicate_study=False):
    return [{'method': method, 'stratum': 'reference_diagnostic', 'endpoint': 'Cmax', 'scenario_id': str(i),
             'compound': 'drug' + str(i), 'publication_cluster_id': 'shared' if duplicate_study else 'paper' + str(i),
             'top1_evaluable_probability': 1.0, 'top1_absolute_log_error': error}
            for i in range(n) for method, error in [('bm25_text', 1.), ('new_method', .5)]]


def test_paired_bootstrap_exact_sign_flip_and_holm_family():
    data = rows()
    report = paired_method_statistics(data, ['bm25_text', 'new_method', 'not_measured'], draws=100)
    r = report['comparisons'][0]
    assert r['state'] == 'computed'
    assert r['mean_paired_absolute_log_error_difference'] == -.5
    assert r['bootstrap_95ci'] == [-.5, -.5]
    assert r['two_sided_p'] == 2 / 1024
    assert r['holm_adjusted_p'] == 4 / 1024
    assert report == paired_method_statistics(data, ['bm25_text', 'new_method', 'not_measured'], draws=100)


@pytest.mark.parametrize('data', [rows(9), rows(10, duplicate_study=True)])
def test_insufficient_independence_does_not_generate_significance(data):
    r = paired_method_statistics(data, ['bm25_text', 'new_method'], draws=100)['comparisons'][0]
    assert r['state'] == 'unavailable'
    assert r['bootstrap_95ci'] is None
    assert r['two_sided_p'] is None


def test_partial_top1_tie_coverage_not_used_in_paired_error_claim():
    data = rows()
    data[0]['top1_evaluable_probability'] = .999
    r = paired_method_statistics(data, ['bm25_text', 'new_method'], draws=100)['comparisons'][0]
    assert r['compounds'] == 9
    assert r['excluded_scenarios'] == ['0']
    assert r['state'] == 'unavailable'


def test_repeat_cases_in_same_publication_do_not_raise_sample_size():
    data = rows()
    data += [{**r, 'scenario_id': r['scenario_id'] + 'extra'} for r in data[:2]]
    r = paired_method_statistics(data, ['bm25_text', 'new_method'], draws=100)['comparisons'][0]
    assert r['compounds'] == 10
    assert r['independent_compound_publication_blocks'] == 10
    assert r['mean_paired_absolute_log_error_difference'] == -.5


def test_zero_difference_p_one_and_no_result_dependent_direction():
    data = rows()
    for row in data:
        row['top1_absolute_log_error'] = .5
    r = paired_method_statistics(data, ['bm25_text', 'new_method'], draws=100)['comparisons'][0]
    assert r['two_sided_p'] == 1
    assert r['bootstrap_95ci'] == [0., 0.]


def test_duplicate_endpoint_unit_rejected():
    data = rows()
    with pytest.raises(ValueError):
        paired_method_statistics(data + [data[0]], ['bm25_text', 'new_method'], draws=100)
