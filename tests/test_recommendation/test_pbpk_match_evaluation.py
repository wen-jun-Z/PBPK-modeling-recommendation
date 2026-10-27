"""Scientific denominator, tie, and calibration-leakage regressions."""
import itertools
import json

import numpy as np
import pytest

from pbpk_kb.recommendation.pbpk_match.evaluation import (
    apply_probability_calibrator,
    compare_stability,
    evaluate_ranking,
    fit_probability_calibrator,
    tie_aware_rank_metrics,
)


def records(q=1, c=3):
    queries = [{"query_id": f"q{i}", "compound_id": f"d{i}", "compounds": ["drug"]} for i in range(q)]
    candidates = [{"candidate_id": f"c{i}", "family_id": f"f{i}", "compounds": ["drug"]} for i in range(c)]
    return queries, candidates


def test_tied_metrics_match_all_permutations():
    grades = np.array([3, 0, 2, 1])
    result = tie_aware_rank_metrics(np.zeros(4), grades)
    ordinary = [tie_aware_rank_metrics(np.arange(4)[::-1], np.array(p)) for p in itertools.permutations(grades)]
    for name in ("ndcg@10", "recall@10", "mrr", "top1_expected_grade"):
        assert result[name] == pytest.approx(np.mean([row[name] for row in ordinary]))


def test_tie_crossing_top10_boundary_is_fractional():
    result = tie_aware_rank_metrics(np.zeros(12), [3] + [0] * 11)
    assert result["recall@10"] == pytest.approx(10 / 12)
    assert result["mrr"] == pytest.approx(sum(1 / i for i in range(1, 13)) / 12)


def test_unknown_top_candidate_is_not_a_negative_or_a_measured_success():
    queries, candidates = records(2)
    result = evaluate_ranking([[9, 3, 1], [2, 1, 0]], [[-1, 3, 0], [-1, 1, 0]],
                              [[2, 0, 1], [-1, 2, 1]], queries, candidates)
    assert result["judged_only"]["eligible_queries"] == 1
    assert result["judged_only"]["metrics"]["ndcg@10"] == 1
    assert result["strict_full_pool"]["eligible_queries"] == 0
    assert result["strict_full_pool"]["metrics"]["ndcg@10"] is None
    assert result["coverage"]["full_catalog_top1_unknown_rate"] == 1
    assert result["coverage"]["known_grade_fraction"] == pytest.approx(4 / 6)
    assert result["coverage"]["queries_without_known_positive"] == 1
    json.dumps(result, allow_nan=False)


def test_top1_unknown_rate_averages_ties_without_arbitrary_winner():
    queries, candidates = records()
    result = evaluate_ranking([[1, 1, 0]], [[-1, 3, 0]], [[2, 0, 1]], queries, candidates)
    assert result["coverage"]["full_catalog_top1_unknown_rate"] == .5
    assert result["coverage"]["full_catalog_top1_contains_unknown_query_fraction"] == 1


def test_family_preference_excludes_unknown_equal_grades_and_shared_sources():
    queries, candidates = records(1, 5)
    candidates[0]["source_packet_sha256"] = candidates[1]["source_packet_sha256"] = "same"
    candidates[4]["compounds"] = ["other"]
    result = evaluate_ranking([[4, 4, 2, 1, 0]], [[3, 0, -1, 1, 3]], [[0, 1, 2, 2, 0]], queries, candidates)
    pref = result["same_drug_different_family_preference"]
    assert pref["pairs"] == 2  # 0-vs-3 correct; 1-vs-3 incorrect
    assert pref["pair_micro_accuracy"] == .5
    assert pref["source_indistinguishable_pairs_excluded"] == 1
    assert pref["independence_verified"] is False


def test_binary_scores_exclude_documented_unknown_and_invalid():
    queries, candidates = records(1, 4)
    probabilities = [[[.8, .1, .1], [.2, .7, .1], [.9, .05, .05], [0, 0, 1]]]
    result = evaluate_ranking([[3, 2, 1, 0]], [[3, 0, -1, -1]], [[0, 1, 2, -1]], queries, candidates, probabilities)
    app = result["applicability"]
    assert app["binary_known_pairs"] == 2
    assert app["valid_pairs"] == 3
    assert app["brier"] == pytest.approx(.04)
    assert app["ece"] == pytest.approx(.2)
    assert app["macro_f1"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert app["invalid_or_unprocessed_pairs"] == 1


def test_missing_classes_never_return_perfect_or_fabricated_metrics():
    queries, candidates = records()
    result = evaluate_ranking([[2, 1, 0]], [[0, 0, -1]], [[1, 1, 2]], queries, candidates,
                              [[[0, 1, 0], [0, 1, 0], [0, 0, 1]]])
    assert result["judged_only"]["metrics"]["mrr"] is None
    assert result["applicability"]["macro_f1"] is None
    assert result["applicability"]["per_class_f1"]["Y"] is None
    assert result["applicability"]["brier"] is None
    assert result["applicability"]["ece"] is None


def test_stability_respects_changed_subset_and_ties():
    left = [[1, 1, 0], [3, 2, 1]]
    right = [[1, 0, 1], [1, 2, 3]]
    result = compare_stability(left, right, "missing_evidence", np.array([True, False]))
    assert result["changed_input_queries"] == 1
    assert result["top1_set_jaccard"] == pytest.approx(1 / 3)
    assert result["unique_top1_consistency"] is None
    assert "neither change nor invariance" in result["interpretation"]
    empty = compare_stability(left, right, changed_query_mask=np.array([False, False]))
    assert empty["top1_set_jaccard"] is None


def calibration_data():
    return np.array([[[3., 0, 0], [0, 3, 0], [0, 0, 3]]]), [[0, 1, 2]], [{"compound_id": "calibration"}]


def fit_args():
    return {"allowed_compound_ids": ["fitted", "calibration"],
            "excluded_compound_ids": ["outer_validation", "test"], "model_fit_compound_ids": ["fitted"]}


def test_calibration_fits_only_disjoint_inner_subset_and_serializes():
    logits, app, queries = calibration_data()
    fitted = fit_probability_calibrator(logits, app, queries, **fit_args())
    assert fitted["state"] == "fitted"
    assert fitted["fit_compound_ids"] == ["calibration"]
    transformed = apply_probability_calibrator(logits, fitted)
    np.testing.assert_allclose(transformed.sum(axis=-1), 1)
    assert transformed[0, 0, 0] > .99
    json.dumps(fitted, allow_nan=False)


@pytest.mark.parametrize("drug", ["fitted", "outer_validation", "test", "undeclared"])
def test_calibration_rejects_in_sample_or_outer_holdout_labels(drug):
    logits, app, _ = calibration_data()
    with pytest.raises(ValueError):
        fit_probability_calibrator(logits, app, [{"compound_id": drug}], **fit_args())


def test_calibration_missing_class_is_unavailable_and_not_silently_identity():
    logits, _, queries = calibration_data()
    fitted = fit_probability_calibrator(logits, [[1, 1, 2]], queries, **fit_args())
    assert fitted["state"] == "unavailable"
    assert fitted["temperature"] is None
    with pytest.raises(ValueError):
        apply_probability_calibrator(logits, fitted)


@pytest.mark.parametrize("grades,app", [([[4, 0, 0]], [[0, 1, 2]]), ([[3, 0, 0]], [[3, 1, 2]])])
def test_invalid_label_encoding_fails_loudly(grades, app):
    queries, candidates = records()
    with pytest.raises(ValueError):
        evaluate_ranking([[3, 2, 1]], grades, app, queries, candidates)
