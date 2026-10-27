import copy
import json

import numpy as np
import pytest

from pbpk_kb.recommendation.pbpk_match.diagnostics import (
    bind_label_pools, compare_reports_paired, risk_coverage_from_predictions,
    summarize_grade_support,
)
from pbpk_kb.recommendation.pbpk_match.evaluation import evaluate_ranking


def reports(compounds=("a", "a", "b")):
    queries = [{"query_id": str(i), "compound_id": compound, "compounds": [compound]} for i, compound in enumerate(compounds)]
    candidates = [{"candidate_id": f"c{i}", "family_id": f"f{i}", "compounds": []} for i in range(3)]
    grades = np.tile([3, 0, -1], (len(queries), 1))
    apps = np.tile([0, 1, 2], (len(queries), 1))
    def report(scores):
        return bind_label_pools(evaluate_ranking(scores, grades, apps, queries, candidates), grades, queries, candidates)
    return report(np.tile([3, 2, 1], (len(queries), 1))), report(np.tile([2, 3, 1], (len(queries), 1))), grades, queries, candidates


def test_paired_bootstrap_uses_compounds_and_expected_direction():
    a, b, *_ = reports()
    result = compare_reports_paired(a, b, replicates=200)
    assert result["eligible_queries"] == 3
    assert result["eligible_compounds"] == 2
    assert result["metrics"]["mrr"]["point_difference"] == .5
    assert result["metrics"]["mrr"]["ci95"] == [.5, .5]
    assert result["metrics"]["recall@10"]["ci95"] == [0, 0]
    assert result == compare_reports_paired(a, b, replicates=200)
    assert result["scientific_gold"] is False
    json.dumps(result, allow_nan=False)


def test_cluster_bootstrap_preserves_all_rows_of_each_drug():
    a, b, *_ = reports()
    # Drug a has two queries with delta=1; drug b one query with delta=0.
    for i in range(3):
        a["per_query"][i]["judged_only"]["mrr"] = 1 if i < 2 else 0
        b["per_query"][i]["judged_only"]["mrr"] = 0
    result = compare_reports_paired(a, b, replicates=2000)
    metric = result["metrics"]["mrr"]
    assert metric["point_difference"] == pytest.approx(2 / 3)
    assert metric["compound_macro_point_difference"] == .5
    assert metric["ci95"] == [0, 1]


def test_comparison_refuses_missing_or_different_original_label_pools():
    a, b, grades, queries, candidates = reports()
    del b["per_query"][0]["candidate_grade_pool_sha256"]
    assert compare_reports_paired(a, b)["state"] == "unavailable"
    grades[0] = [3, -1, 0]  # Identical counts, different actual judged candidates.
    b = evaluate_ranking(np.tile([2, 3, 1], (3, 1)), grades, np.tile([0, 1, 2], (3, 1)), queries, candidates)
    b = bind_label_pools(b, grades, queries, candidates)
    result = compare_reports_paired(a, b)
    assert result["state"] == "unavailable"
    assert result["different_pool_queries"] == ["0"]
    assert result["metrics"]["mrr"]["point_difference"] is None


def test_binding_is_candidate_order_invariant_and_does_not_mutate_report():
    a, _, grades, queries, candidates = reports()
    plain = copy.deepcopy(a)
    for row in plain["per_query"]:
        del row["candidate_grade_pool_sha256"]
    bound = bind_label_pools(plain, grades[:, ::-1], queries, candidates[::-1])
    assert bound["per_query"][0]["candidate_grade_pool_sha256"] == a["per_query"][0]["candidate_grade_pool_sha256"]
    assert "candidate_grade_pool_sha256" not in plain["per_query"][0]


def test_one_cluster_has_no_fabricated_confidence_interval():
    a, b, *_ = reports(("a", "a"))
    result = compare_reports_paired(a, b)
    assert result["state"] == "point_only"
    assert result["metrics"]["mrr"]["point_difference"] == .5
    assert result["metrics"]["mrr"]["ci95"] is None


def test_only_common_eligible_queries_enter_the_paired_comparison():
    a, b, *_ = reports()
    b["per_query"][0]["judged_only"] = None
    b["per_query"].pop()
    result = compare_reports_paired(a, b)
    assert result["common_queries"] == 2
    assert result["eligible_queries"] == 1
    assert result["eligible_compounds"] == 1


def test_risk_coverage_ties_do_not_choose_good_candidate_or_hide_unknown():
    scores = [[2, 2, 2, 2], [4, 3, 2, 1]]
    probs = np.array([[[.8, .1, .1]] * 4, [[.6, .2, .2]] * 4])
    app = [[0, 1, 2, -1], [0, 1, 2, -1]]
    result = risk_coverage_from_predictions(scores, probs, app, (.5, 1))
    first = result["curve"][0]
    assert first["coverage"] == .5
    assert first["risk_known_only"] == .5
    assert first["known_expected_top1_fraction"] == .5
    assert first["unknown_expected_top1_fraction"] == .25
    assert first["invalid_expected_top1_fraction"] == .25
    assert result["curve"][1]["risk_known_only"] == pytest.approx(.25 / 1.5)
    assert result["curve"][1]["coverage"] == 1


def test_confidence_boundary_ties_include_every_query():
    result = risk_coverage_from_predictions([[1], [1], [1]], [[[.7, .2, .1]]] * 3, [[0], [1], [2]], (.25,))
    row = result["curve"][0]
    assert row["selected_queries"] == 3
    assert row["coverage"] == 1
    assert row["risk_known_only"] == .5
    assert row["unknown_expected_top1_fraction"] == pytest.approx(1 / 3)


def test_all_unknown_risk_is_unavailable_not_zero_or_one():
    result = risk_coverage_from_predictions([[1, 1]], [[[.8, .1, .1], [.7, .2, .1]]], [[2, -1]])
    assert all(row["risk_known_only"] is None for row in result["curve"])
    assert all(row["coverage"] == 1 for row in result["curve"])
    assert all(row["unknown_or_invalid_expected_top1_fraction"] == 1 for row in result["curve"])
    json.dumps(result, allow_nan=False)


def test_grade_support_exposes_absent_partial_applicability_and_clusters():
    report = summarize_grade_support([[3, 0, -1], [1, 0, -1]], [{"compound_id": "a"}, {"compound_id": "a"}])
    assert report["grade_counts"] == {"0": 2, "1": 1, "2": 0, "3": 1}
    assert report["known_fraction"] == pytest.approx(4 / 6)
    assert report["queries_with_known_positive"] == 1
    assert report["compounds_with_known_positive"] == 1
    assert report["grade2_observed"] is False
    assert report["grade2_learning_claim_supported"] is False
    assert report["scientific_gold"] is False


def test_diagnostics_reject_invalid_label_encoding_and_probabilities():
    with pytest.raises(ValueError):
        summarize_grade_support([[4]])
    with pytest.raises(ValueError):
        risk_coverage_from_predictions([[1]], [[[.9, .9, .1]]], [[0]])
    with pytest.raises(ValueError):
        risk_coverage_from_predictions([[1]], [[[.8, .1, .1]]], [[3]])
