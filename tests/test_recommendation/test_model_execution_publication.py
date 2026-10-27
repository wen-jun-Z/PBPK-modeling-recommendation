"""Rank-position diagnostics must preserve ties, unknowns, and study denominators."""
import math
import json
from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from build_model_execution_results import (
    common_position_summaries, exact_position, independent_blocks, summarize_positions,
)


def pair(fold):
    return {'predicted': fold, 'observed': 1.0}


def case(scores, ids, endpoints, position, *, sid='s', drug='d', study='p'):
    row = exact_position(scores, ids, endpoints, position)
    row.update(method_id='m', scenario_id=sid, compound=drug,
               publication_cluster_id=study, stratum='reference_diagnostic', endpoint='Cmax')
    return row


def test_missing_first_position_does_not_backfill_from_second_or_third():
    rows = [exact_position([3., 2., 1.], ['unmeasured', 'bad', 'good'],
                           {'bad': pair(4), 'good': pair(1)}, p) for p in (1, 2, 3)]
    assert rows[0]['evaluable_probability'] == 0
    assert rows[0]['conditional_absolute_log_error'] is None
    assert rows[0]['members'][0]['candidate_id'] == 'unmeasured'
    assert rows[1]['conditional_absolute_log_error'] == pytest.approx(math.log(4))
    assert rows[2]['conditional_absolute_log_error'] == 0


def test_position_uses_entire_exact_tie_block_not_a_topk_subset():
    rows = [exact_position([10., 10., 10., 2.], ['a', 'b', 'c', 'd'],
                           {'a': pair(1), 'b': pair(4), 'd': pair(1)}, p) for p in (1, 2, 3)]
    for row in rows:
        assert [m['candidate_id'] for m in row['members']] == ['a', 'b', 'c']
        assert [m['weight'] for m in row['members']] == [1 / 3] * 3
        assert row['tie_group_first_rank'] == 1 and row['tie_group_last_rank'] == 3
        assert row['evaluable_probability'] == 2 / 3
        assert row['conditional_absolute_log_error'] == pytest.approx(math.log(2))
        assert row['conditional_within2fold'] == .5
        assert not row['fully_evaluable']


def test_block_crossing_third_position_includes_fourth_and_fifth_candidates():
    row = exact_position([10, 9, 8, 8, 8], list('abcde'), {c: pair(1) for c in 'abcde'}, 3)
    assert [m['candidate_id'] for m in row['members']] == list('cde')
    assert row['tie_group_first_rank'] == 3 and row['tie_group_last_rank'] == 5
    assert row['fully_evaluable']


@pytest.mark.parametrize('bad_pair', [None, {'predicted': 0, 'observed': 1},
    {'predicted': 1, 'observed': -1}, {'predicted': True, 'observed': 1},
    {'predicted': float('nan'), 'observed': 1}, {'predicted': 1, 'observed': None}])
def test_invalid_endpoint_is_unknown_not_zero_error(bad_pair):
    row = exact_position([1], ['a'], {'a': bad_pair}, 1)
    assert row['members'][0]['state'] == 'unknown'
    assert row['conditional_absolute_log_error'] is None
    assert row['conditional_within2fold'] is None
    assert row['evaluable_probability'] == 0


@pytest.mark.parametrize('position', [0, 4, True, 1.5])
def test_position_bounds_rejected(position):
    with pytest.raises(ValueError, match='Invalid rank position'):
        exact_position([3, 2, 1], list('abc'), {}, position)


def test_near_scores_are_not_merged_into_an_approximate_tie():
    row = exact_position([1, 1 - 1e-12], ['a', 'b'], {'a': pair(4), 'b': pair(1)}, 2)
    assert row['tie_group_size'] == 1 and row['members'][0]['candidate_id'] == 'b'
    assert row['conditional_absolute_log_error'] == 0


def test_frozen_numpy_score_array_exports_json_native_rank_boundaries():
    row = exact_position(np.array([3., 2., 2.]), list('abc'), {'b': pair(2), 'c': pair(1)}, 2)
    assert json.loads(json.dumps(row, allow_nan=False)) == row
    assert type(row['tie_group_first_rank']) is type(row['tie_group_last_rank']) is int


def test_conditional_error_weights_studies_then_drugs_not_number_of_cases():
    rows = [case([1], ['a'], {'a': pair(fold)}, 1, sid=sid, drug=drug, study=study)
            for sid, drug, study, fold in [('s1', 'd1', 'p1', 1), ('s2', 'd1', 'p1', 1),
                                         ('s3', 'd1', 'p2', 16), ('s4', 'd2', 'p3', 1)]]
    rows.append(case([1], ['a'], {}, 1, sid='unknown', drug='d3', study='p4'))
    summary, = summarize_positions(rows)
    assert summary['requested_scenarios'] == 5
    assert summary['evaluable_probability'] == .8
    assert summary['covered_compounds'] == 2
    assert summary['covered_publication_compound_groups'] == 3
    assert summary['conditional_gmfe'] == pytest.approx(2)
    assert summary['conditional_within2fold'] == .75


def test_common_set_excludes_partial_tie_coverage_and_uses_paired_direction():
    rows = []
    for p in (1, 2, 3):
        rows.append(case([3, 2, 1], list('abc'), {'a': pair(2), 'b': pair(4), 'c': pair(1)}, p))
        rows.append(case([3, 2, 2], list('abc'), {'a': pair(2), 'b': pair(4)}, p, sid='partial'))
    summary, = common_position_summaries(rows)
    assert summary['common_scenarios'] == 1 and summary['scenario_ids'] == ['s']
    assert summary['excluded_scenario_ids'] == ['partial']
    assert [summary[f'rank{p}_gmfe'] for p in (1, 2, 3)] == pytest.approx([2, 4, 1])
    assert summary['rank2_minus_rank1_absolute_log_error'] == pytest.approx(math.log(2))
    assert summary['rank3_minus_rank1_absolute_log_error'] == pytest.approx(-math.log(2))
    assert summary['inferential_test_performed'] is False


def test_no_common_cases_means_null_errors_not_zero():
    rows = [case([3, 2, 1], list('abc'), {'a': pair(1), 'b': pair(2)}, p) for p in (1, 2, 3)]
    summary, = common_position_summaries(rows)
    assert summary['common_scenarios'] == summary['common_compounds'] == 0
    assert summary['independent_compound_publication_blocks'] == 0
    assert all(summary[f'rank{p}_gmfe'] is None for p in (1, 2, 3))
    assert summary['rank2_minus_rank1_absolute_log_error'] is None


def test_publication_overlap_joins_drugs_transitively():
    rows = [{'compound': d, 'publication_cluster_id': p} for d, p in
            [('a', 'study1'), ('b', 'study1'), ('b', 'study2'), ('c', 'study2'), ('d', 'study3')]]
    blocks = independent_blocks(rows)
    assert {tuple(b['compounds']) for b in blocks} == {('a', 'b', 'c'), ('d',)}


def test_rank_cases_cannot_be_silently_missing_or_duplicated():
    rows = [case([3, 2, 1], list('abc'), {}, p) for p in (1, 2, 3)]
    with pytest.raises(ValueError, match='Incomplete three-position'):
        common_position_summaries(rows[:2])
    with pytest.raises(ValueError, match='Duplicate position'):
        common_position_summaries(rows + [rows[0]])
