import itertools
import json

import numpy as np
import pytest

from pbpk_kb.recommendation.pbpk_match.evaluation import tie_aware_rank_metrics
from pbpk_kb.recommendation.pbpk_match.policy_evaluation import (
    evaluate_identity_policy, retained_pool_rank_metrics,
)


def records(q=1, c=3):
    queries = [{"query_id": f"q{i}", "compound_id": f"d{i}", "compounds": [f"drug{i}"],
                "task_type": "plasma_concentration_time_prediction"} for i in range(q)]
    candidates = [{"candidate_id": f"c{i}", "compounds": ["drug0"]} for i in range(c)]
    return queries, candidates


def fixed_policy(mask):
    return lambda query, candidates: {"eligible_mask": list(mask), "abstain": not any(mask), "scope": "test_only"}


def test_all_eligible_matches_frozen_known_pool_metric_including_ties():
    scores, grades = np.array([3, 2, 2, 0]), np.array([0, 3, 1, 2])
    original = tie_aware_rank_metrics(scores, grades)
    policy = retained_pool_rank_metrics(scores, grades, np.ones(4, bool))
    for name in ("ndcg@10", "recall@10", "mrr"):
        assert policy[name] == original[name]


def test_removing_known_positive_cannot_shrink_ideal_or_recall_denominator():
    result = retained_pool_rank_metrics([3, 2, 1], [3, 2, 0], [False, True, True])
    assert result["recall@10"] == .5
    assert result["ndcg@10"] == pytest.approx(3 / (7 + 3 / np.log2(3)))
    assert result["pre_policy_known_positives"] == 2
    assert result["mrr"] == 1


def test_only_reference_grade_after_filter_has_low_gain_not_a_dropped_query():
    result = retained_pool_rank_metrics([3, 2, 1], [3, 1, 0], [False, True, True])
    assert result["recall@10"] == result["mrr"] == 0
    assert result["ndcg@10"] == pytest.approx(1 / (7 + 1 / np.log2(3)))


def test_zero_eligible_is_zero_when_original_pool_has_positive_and_none_otherwise():
    zero = retained_pool_rank_metrics([2, 1], [3, 0], [False, False])
    assert all(zero[key] == 0 for key in ("ndcg@10", "recall@10", "mrr"))
    assert retained_pool_rank_metrics([2, 1], [1, 0], [False, False]) is None


def test_tie_expectation_with_fixed_prefilter_denominator_matches_enumeration():
    values = np.array([3, 2, 0, 1])
    expected = retained_pool_rank_metrics([9, 1, 1, 1], values, [False, True, True, True])
    enumerated = []
    for permutation in itertools.permutations(values[1:]):
        enumerated.append(retained_pool_rank_metrics([9, 3, 2, 1], [3, *permutation], [False, True, True, True]))
    for name in ("ndcg@10", "recall@10", "mrr"):
        assert expected[name] == pytest.approx(np.mean([row[name] for row in enumerated]))


def test_unknown_top_candidate_is_visible_and_never_assigned_negative_label():
    queries, candidates = records()
    raw = np.array([[4., 3., 2.]])
    saved = raw.copy()
    result = evaluate_identity_policy(raw, [[-1, 3, 0]], [[2, 0, 1]], queries, candidates,
                                      policy_function=fixed_policy([True, True, False]))
    np.testing.assert_array_equal(raw, saved)
    assert result["metrics"]["ndcg@10"] == 1  # Explicitly conditional known list.
    assert result["coverage"]["top1_unknown_grade_rate_among_nonempty_queries"] == 1
    assert result["coverage"]["eligible_unknown_candidate_count"] == 1
    assert result["raw_scores_modified"] is False
    assert result["policy_is_scientific_applicability_label"] is False


def test_abstained_positive_query_remains_in_metric_average_and_coverage():
    queries, candidates = records(2)
    def policy(query, candidates):
        mask = [False] * 3 if query["query_id"] == "q0" else [True] * 3
        return {"eligible_mask": mask}
    report = evaluate_identity_policy([[3, 2, 1]] * 2, [[3, 0, -1]] * 2, [[0, 1, 2]] * 2,
                                      queries, candidates, policy_function=policy)
    assert report["eligible_ranking_queries"] == 2
    assert report["metrics"]["ndcg@10"] == report["metrics"]["mrr"] == .5
    assert report["coverage"]["nonempty_candidate_coverage"] == .5
    assert report["coverage"]["zero_eligible_with_known_positive_queries"] == 1
    assert report["coverage"]["filtered_known_positive_count"] == 1
    assert report["per_query"][0]["filtered_known_positive_candidate_ids"] == ["c0"]
    json.dumps(report, allow_nan=False)


def test_top1_unknown_fraction_averages_all_tied_eligible_candidates():
    queries, candidates = records()
    report = evaluate_identity_policy([[5, 5, 9]], [[3, -1, 0]], [[0, 2, 1]], queries, candidates,
                                      policy_function=fixed_policy([True, True, False]))
    assert report["coverage"]["top1_unknown_grade_rate_among_nonempty_queries"] == .5
    assert report["per_query"][0]["policy_top1_tie_size"] == 2


def test_same_metadata_policy_independent_of_ranker_scores_or_targets():
    queries, candidates = records()
    a = evaluate_identity_policy([[3, 2, 1]], [[3, 0, -1]], [[0, 1, 2]], queries, candidates,
                                 policy_function=fixed_policy([True, False, True]))
    b = evaluate_identity_policy([[1, 2, 3]], [[-1, 3, 0]], [[2, 0, 1]], queries, candidates,
                                 policy_function=fixed_policy([True, False, True]))
    assert a["policy_mask_sha256"] == b["policy_mask_sha256"]
    assert a["raw_scores_sha256"] != b["raw_scores_sha256"]


def test_malformed_or_self_contradicting_policy_mask_is_rejected():
    queries, candidates = records()
    for response in ({"eligible_mask": [1, 0, 1]}, {"eligible_mask": [True]},
                     {"eligible_mask": [False] * 3, "abstain": False},
                     {"eligible_mask": [True] * 3, "eligible": [False] * 3}):
        with pytest.raises(ValueError):
            evaluate_identity_policy([[3, 2, 1]], [[3, 0, -1]], [[0, 1, 2]], queries, candidates,
                                     policy_function=lambda *args: response)


def test_real_catalog_policy_direct_and_template_share_original_grade_denominator():
    queries, candidates = records(2)
    queries[1].update(compounds=["drug0"], task_type="mechanistic_template_recommendation")
    candidates[1]["compounds"] = ["other"]
    candidates[2]["compounds"] = []
    # A high source grade can disagree with an identity restriction. The policy
    # must incur the resulting penalty rather than redefining it as a negative.
    result = evaluate_identity_policy([[1, 3, 2]] * 2, [[0, 3, -1]] * 2, [[1, 0, 2]] * 2,
                                      queries, candidates)
    direct, template = result["per_query"]
    assert direct["eligible_candidates"] == 1
    assert direct["filtered_known_positive_count"] == 1
    assert direct["metrics"]["ndcg@10"] == 0
    assert template["eligible_candidates"] == 3
    assert template["metrics"]["ndcg@10"] == 1
    assert result["metrics"]["ndcg@10"] == .5
    assert result["policy_versions"] == ["catalog-identity-v1"]
