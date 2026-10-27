from __future__ import annotations

from copy import deepcopy
import json

import pytest

from pbpk_kb.recommendation.model_matching.scientific_evaluation import evaluate, main


def _payload() -> dict:
    return {
        "queries": [{
            "query_id": "q1", "compound_id": "drug-a", "study_id": "held-out-study",
            "task_type": "direct_reuse", "candidate_ids": ["a", "b"],
            "input_evidence_ids": ["input-a"], "development_study_ids": ["training-study"],
        }],
        "judgments": [
            {"query_id": "q1", "candidate_id": candidate, "grade": grade,
             "applicable": applicable, "label_source": "independent_execution",
             "study_id": "held-out-study", "evidence_ids": [f"gold-{candidate}"],
             "candidate_compound_id": "drug-a", "family_id": f"family-{candidate}"}
            for candidate, grade, applicable in [("a", 3, True), ("b", 0, False)]
        ],
        "predictions": [
            {"query_id": "q1", "candidate_id": candidate, "method": "ranker", "seed": 17,
             "score": score, "probability": probability, "predicted_applicable": decision}
            for candidate, score, probability, decision in [("a", 2, 0.9, True), ("b", 1, 0.1, False)]
        ],
    }


def _overall(payload: dict, **kwargs) -> dict:
    return evaluate(payload, **kwargs)["results"][0]["overall"]


def _unknown(row: dict) -> None:
    row.update(grade=None, applicable=None, label_source="unjudged", evidence_ids=[], study_id=None)


def test_perfect_predictions_have_multidimensional_scores_without_certification() -> None:
    report = evaluate(_payload())
    assert report["scientific_claim_eligible"] is False
    assert report["audit"]["consistency_checks_passed"] is True
    result = report["results"][0]
    overall = result["overall"]
    assert overall["ranking"]["metrics"]["ndcg@10"] == 1
    assert overall["ranking"]["metrics"]["recall@1"] == 1
    assert overall["ranking"]["metrics"]["mrr"] == 1
    assert overall["within_drug_ranking"]["eligible_query_count"] == 1
    assert overall["applicability"]["macro_f1"] == 1
    assert overall["applicability"]["unsafe_recommendation_rate"] == 0
    assert overall["calibration"]["brier_score"] == pytest.approx(0.01)
    assert overall["calibration"]["ece"] == pytest.approx(0.1)
    assert result["compound_macro"]["ranking_compound_count"] == 1
    assert set(result["by_compound"]) == {"drug-a"}
    assert set(result["by_task_type"]) == {"direct_reuse"}


def test_unknown_judgments_are_never_negative() -> None:
    payload = _payload()
    _unknown(payload["judgments"][1])
    payload["predictions"][1]["predicted_applicable"] = True
    overall = _overall(payload)
    assert overall["ranking"]["metrics"]["ndcg@10"] is None
    assert overall["ranking"]["unjudged_query_count"] == 1
    assert overall["ranking"]["coverage"] == 0
    assert overall["applicability"]["support"] == 1
    assert overall["applicability"]["unknown_gold_count"] == 1
    assert overall["applicability"]["recommended_unknown_gold_count"] == 1
    assert overall["applicability"]["unsafe_recommendation_rate"] == 0
    assert overall["calibration"]["support"] == 1


def test_all_unjudged_returns_na_not_scientific_scores() -> None:
    payload = _payload()
    for row in payload["judgments"]:
        _unknown(row)
    overall = _overall(payload)
    assert all(value is None for value in overall["ranking"]["metrics"].values())
    assert overall["applicability"]["macro_f1"] is None
    assert overall["applicability"]["unsafe_recommendation_rate"] is None
    assert overall["calibration"]["ece"] is None
    assert overall["top1_applicability"]["inapplicable_rate"] is None


@pytest.mark.parametrize("source", ["metadata", "compatibility_rule", "rule", "llm_generated"])
def test_rule_or_unaudited_label_sources_rejected(source: str) -> None:
    payload = _payload()
    payload["judgments"][0]["label_source"] = source
    with pytest.raises(ValueError, match="rule/metadata labels forbidden"):
        evaluate(payload)


def test_same_source_input_and_gold_rejected() -> None:
    payload = _payload()
    payload["queries"][0]["input_evidence_ids"] = ["gold-a"]
    with pytest.raises(ValueError, match="overlaps judgment evidence globally"):
        evaluate(payload)


def test_across_query_evidence_and_development_study_leakage_rejected() -> None:
    payload = _payload()
    second = deepcopy(payload["queries"][0])
    second.update(query_id="q2", input_evidence_ids=["gold-a"])
    payload["queries"].append(second)
    payload["judgments"].extend([{**row, "query_id": "q2"} for row in list(payload["judgments"])])
    payload["predictions"].extend([{**row, "query_id": "q2"} for row in list(payload["predictions"])])
    with pytest.raises(ValueError, match="overlaps judgment evidence globally"):
        evaluate(payload)
    second["input_evidence_ids"] = []
    second["development_study_ids"] = ["held-out-study"]
    with pytest.raises(ValueError, match="Development studies overlap"):
        evaluate(payload)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "missing-query"])
def test_full_pool_prediction_contract(mutation: str) -> None:
    payload = _payload()
    if mutation == "missing":
        payload["predictions"].pop()
    elif mutation == "extra":
        payload["predictions"].append({**payload["predictions"][0], "candidate_id": "not-in-pool"})
    elif mutation == "duplicate":
        payload["predictions"].append(dict(payload["predictions"][0]))
    else:
        payload["queries"].append({**payload["queries"][0], "query_id": "q2"})
        payload["judgments"].extend([{**row, "query_id": "q2"} for row in list(payload["judgments"])])
    with pytest.raises(ValueError, match="prediction"):
        evaluate(payload)


def test_missing_judgment_must_be_explicitly_unjudged() -> None:
    payload = _payload()
    payload["judgments"].pop()
    with pytest.raises(ValueError, match="Missing judgments"):
        evaluate(payload)


@pytest.mark.parametrize("field,value", [("score", float("nan")), ("score", float("inf")), ("score", True), ("probability", 1.1), ("probability", -0.1), ("predicted_applicable", 1)])
def test_invalid_predictions_rejected(field: str, value) -> None:
    payload = _payload()
    payload["predictions"][0][field] = value
    with pytest.raises(ValueError):
        evaluate(payload)


def test_prediction_gold_fields_cannot_override_independent_judgments() -> None:
    payload = _payload()
    payload["predictions"][0].update(grade=0, applicable=False, relevance_grade=0)
    assert _overall(payload)["ranking"]["metrics"]["ndcg@1"] == 1


@pytest.mark.parametrize("reverse_grades", [False, True])
def test_equal_scores_get_half_pair_credit_in_either_label_direction(reverse_grades: bool) -> None:
    payload = _payload()
    for row in payload["predictions"]:
        row["score"] = 1
    if reverse_grades:
        payload["judgments"][0]["grade"], payload["judgments"][1]["grade"] = 0, 3
    metrics = _overall(payload)["ranking"]["metrics"]
    assert metrics["pairwise_accuracy"] == 0.5
    assert metrics["mrr"] == (0.5 if reverse_grades else 1)
    payload["predictions"].reverse()
    assert _overall(payload)["ranking"]["metrics"] == metrics


def test_ece_uses_equal_width_bins_including_one() -> None:
    payload = _payload()
    payload["predictions"][0]["probability"] = 0.49
    payload["predictions"][1]["probability"] = 0.01
    result = _overall(payload, calibration_bins=2)["calibration"]
    assert result["bins"][0]["count"] == 2
    assert result["bins"][1]["count"] == 0
    assert result["ece"] == pytest.approx(0.25)
    payload["predictions"][0]["probability"] = 1.0
    payload["predictions"][1]["probability"] = 0.0
    result = _overall(payload, calibration_bins=2)["calibration"]
    assert [row["count"] for row in result["bins"]] == [1, 1]
    assert result["ece"] == 0


def test_missing_decision_is_not_a_negative_prediction() -> None:
    payload = _payload()
    del payload["predictions"][0]["predicted_applicable"]
    payload["predictions"][1]["predicted_applicable"] = True
    result = _overall(payload)["applicability"]
    assert result["support"] == 1
    assert result["missing_prediction_with_known_gold_count"] == 1
    assert result["confusion_matrix"] == [[0, 1], [0, 0]]
    assert result["unsafe_recommendation_rate"] == 1
    assert result["recommended_known_gold_count"] == 1


def test_no_positive_queries_are_counted_and_not_scored_as_zero() -> None:
    payload = _payload()
    payload["judgments"][0]["grade"] = 1
    ranking = _overall(payload)["ranking"]
    assert ranking["no_positive_query_count"] == 1
    assert ranking["unjudged_query_count"] == 0
    assert ranking["metrics"]["mrr"] is None
    assert _overall(payload, positive_grade_threshold=1)["ranking"]["metrics"]["mrr"] == 1


@pytest.mark.parametrize("reason", ["same-family", "same-grade", "missing-family", "missing-compound"])
def test_within_drug_requires_independent_families_and_label_discrimination(reason: str) -> None:
    payload = _payload()
    if reason == "same-family":
        payload["judgments"][1]["family_id"] = "family-a"
    elif reason == "same-grade":
        payload["judgments"][1]["grade"] = 3
    elif reason == "missing-family":
        del payload["judgments"][1]["family_id"]
    else:
        del payload["judgments"][1]["candidate_compound_id"]
    result = _overall(payload)["within_drug_ranking"]
    assert result["eligible_query_count"] == 0
    assert result["coverage_of_all_queries"] == 0
    assert result["metrics"]["ndcg@1"] is None


def test_seeds_are_separate_and_compound_macro_does_not_count_requests_as_drugs() -> None:
    payload = _payload()
    payload["predictions"].extend([{**row, "seed": 29} for row in list(payload["predictions"])])
    report = evaluate(payload)
    assert len(report["results"]) == 2
    assert report["audit"]["query_count"] == 1
    assert report["audit"]["compound_count"] == 1
    for result in report["results"]:
        assert result["overall"]["ranking"]["query_count"] == 1
        assert result["compound_macro"]["ranking_compound_count"] == 1


def test_compound_macro_weights_drugs_equally_despite_different_query_counts() -> None:
    payload = _payload()
    for query_id in ("q2", "q3"):
        payload["queries"].append({**payload["queries"][0], "query_id": query_id, "compound_id": "drug-b"})
        payload["judgments"].extend([{**row, "query_id": query_id} for row in payload["judgments"][:2]])
        payload["predictions"].extend([{**row, "query_id": query_id, "score": -row["score"]} for row in payload["predictions"][:2]])
    result = evaluate(payload)["results"][0]
    assert result["overall"]["ranking"]["metrics"]["mrr"] == pytest.approx(2 / 3)
    assert result["compound_macro"]["ranking_metrics"]["mrr"] == 0.75
    assert result["compound_macro"]["ranking_compound_count"] == 2


def test_candidate_identity_cannot_change_between_queries() -> None:
    payload = _payload()
    payload["queries"].append({**payload["queries"][0], "query_id": "q2"})
    payload["judgments"].extend([{**row, "query_id": "q2", "family_id": "changed-family"} for row in payload["judgments"][:2]])
    with pytest.raises(ValueError, match="Inconsistent candidate metadata"):
        evaluate(payload)


def test_unjudged_evidence_also_cannot_appear_as_input() -> None:
    payload = _payload()
    _unknown(payload["judgments"][1])
    payload["judgments"][1]["evidence_ids"] = ["input-a"]
    with pytest.raises(ValueError, match="overlaps judgment evidence globally"):
        evaluate(payload)


def test_ddi_candidate_membership_counts_in_within_drug_comparison() -> None:
    payload = _payload()
    del payload["judgments"][1]["candidate_compound_id"]
    payload["judgments"][1]["candidate_compound_ids"] = ["drug-b", "drug-a"]
    result = _overall(payload)["within_drug_ranking"]
    assert result["eligible_query_count"] == 1
    assert result["metrics"]["ndcg@1"] == 1


def test_selective_risk_integrates_ordered_errors_at_actual_coverage() -> None:
    payload = _payload()
    payload["predictions"][1].update(predicted_applicable=True, probability=0.8)
    result = _overall(payload)["selective_classification"]
    assert result["support"] == 2
    assert result["evaluable_coverage"] == 1
    assert result["aurc"] == pytest.approx(0.25)
    assert result["full_coverage_risk"] == 0.5
    assert [point["risk"] for point in result["curve"]] == [None, 0, 0.5]
    assert [point["coverage"] for point in result["curve"]] == [0, 0.5, 1]


def test_confidence_ties_use_expected_risk_without_gold_or_input_order_tiebreaks() -> None:
    payload = _payload()
    payload["predictions"][0].update(predicted_applicable=False, probability=0.1)
    result = _overall(payload)["selective_classification"]
    assert result["aurc"] == pytest.approx(0.5)
    assert result["full_coverage_risk"] == 0.5
    assert len(result["curve"]) == 2  # Empty set and the whole tied block.
    payload["predictions"].reverse()
    assert _overall(payload)["selective_classification"] == result


def test_risk_does_not_turn_unknown_labels_or_missing_probabilities_into_errors() -> None:
    payload = _payload()
    _unknown(payload["judgments"][1])
    payload["predictions"][0].pop("probability")
    result = _overall(payload)["selective_classification"]
    assert result["support"] == 0
    assert result["aurc"] is None
    assert result["curve"] == []
    assert result["unknown_gold_count"] == 1
    assert result["missing_probability_with_known_gold_and_decision_count"] == 1
    assert result["evaluable_coverage"] == 0


def test_missing_decision_cannot_be_inferred_from_a_high_probability() -> None:
    payload = _payload()
    for prediction in payload["predictions"]:
        prediction.pop("predicted_applicable")
        prediction["probability"] = 1.0
    overall = _overall(payload)
    assert overall["selective_classification"]["support"] == 0
    assert overall["selective_classification"]["aurc"] is None
    decision = overall["recommendation_decisions"]
    assert decision["decision_coverage"] == 0
    assert decision["unknown_action_query_count"] == 1
    assert decision["recommendation_coverage"] == 0
    assert decision["false_rejection_rate"] is None
    assert decision["missing_action_with_known_available_query_count"] == 1


def test_reject_everything_reports_false_rejection_and_zero_recommendation_coverage() -> None:
    payload = _payload()
    for prediction in payload["predictions"]:
        prediction["predicted_applicable"] = False
    overall = _overall(payload)
    assert overall["applicability"]["false_rejection_rate"] == 1
    assert overall["applicability"]["recommendation_coverage"] == 0
    decision = overall["recommendation_decisions"]
    assert decision["false_rejection_rate"] == 1
    assert decision["false_rejection_denominator"] == 1
    assert decision["recommendation_coverage"] == 0
    assert decision["rejection_coverage"] == 1
    assert decision["unsafe_recommendation_rate"] is None
    assert decision["selected_recommendation_risk_coverage"]["aurc"] is None


def test_no_applicable_model_requires_entire_gold_pool_to_be_judged() -> None:
    payload = _payload()
    for prediction in payload["predictions"]:
        prediction["predicted_applicable"] = False
    for judgment in payload["judgments"]:
        judgment.update(grade=0, applicable=False)
    result = _overall(payload)["recommendation_decisions"]
    assert result["fully_judged_no_applicable_query_count"] == 1
    assert result["correct_rejection_rate"] == 1
    _unknown(payload["judgments"][1])
    result = _overall(payload)["recommendation_decisions"]
    assert result["fully_judged_no_applicable_query_count"] == 0
    assert result["unknown_availability_query_count"] == 1
    assert result["correct_rejection_rate"] is None


def test_query_policy_can_choose_an_accepted_lower_ranked_candidate() -> None:
    payload = _payload()
    payload["predictions"][0]["predicted_applicable"] = False
    payload["predictions"][1].update(predicted_applicable=True, probability=0.7)
    decision = _overall(payload)["recommendation_decisions"]
    assert decision["recommendation_query_count"] == 1
    assert decision["unsafe_recommendation_count"] == 1
    assert decision["unsafe_recommendation_rate"] == 1
    assert decision["false_rejection_rate"] == 0
    assert decision["selected_recommendation_risk_coverage"]["aurc"] == 1


def test_partial_prediction_pool_with_no_acceptance_does_not_establish_refusal() -> None:
    payload = _payload()
    payload["predictions"][0].pop("predicted_applicable")
    decision = _overall(payload)["recommendation_decisions"]
    assert decision["unknown_action_query_count"] == 1
    assert decision["rejection_query_count"] == 0
    assert decision["false_rejection_rate"] is None


def test_selected_unknown_gold_is_reported_without_an_unsafe_score() -> None:
    payload = _payload()
    _unknown(payload["judgments"][0])
    decision = _overall(payload)["recommendation_decisions"]
    assert decision["recommendation_coverage"] == 1
    assert decision["recommended_selected_gold_unknown_count"] == 1
    assert decision["unsafe_recommendation_rate"] is None
    assert decision["selected_recommendation_risk_coverage"]["aurc"] is None
    assert decision["selected_recommendation_risk_coverage"]["unknown_gold_count"] == 1


def test_ddi_membership_order_is_normalized_across_queries() -> None:
    payload = _payload()
    payload["judgments"][0]["candidate_compound_ids"] = ["drug-b", "drug-a"]
    payload["queries"].append({**payload["queries"][0], "query_id": "q2"})
    payload["judgments"].extend([{**row, "query_id": "q2"} for row in payload["judgments"][:2]])
    payload["judgments"][2]["candidate_compound_ids"] = ["drug-a", "drug-b"]
    payload["predictions"].extend([{**row, "query_id": "q2"} for row in payload["predictions"][:2]])
    assert evaluate(payload)["audit"]["query_count"] == 2
    payload["judgments"][2]["candidate_compound_ids"] = ["drug-a", "drug-c"]
    with pytest.raises(ValueError, match="Inconsistent candidate metadata"):
        evaluate(payload)


def test_single_and_multiple_compound_membership_cannot_contradict() -> None:
    payload = _payload()
    payload["judgments"][0]["candidate_compound_ids"] = ["drug-b"]
    with pytest.raises(ValueError, match="must belong"):
        evaluate(payload)


def test_duplicate_family_versions_make_family_selection_na_without_deduplicating_by_score() -> None:
    payload = _payload()
    payload["queries"][0]["candidate_ids"].append("c")
    payload["judgments"].append({**payload["judgments"][0], "candidate_id": "c", "grade": 2})
    payload["predictions"].append({**payload["predictions"][0], "candidate_id": "c", "score": 100})
    result = _overall(payload)
    within = result["within_drug_ranking"]
    assert result["ranking"]["eligible_query_count"] == 1
    assert within["eligible_query_count"] == 0
    assert within["duplicate_family_versions_query_count"] == 1
    assert within["metrics"]["ndcg@1"] is None
    payload["predictions"][-1]["score"] = -100
    assert _overall(payload)["within_drug_ranking"] == within


@pytest.mark.parametrize("state,reason", [("known", None), ("unknown", "unjudged_candidates"), ("no-positive", "no_positive_candidate")])
def test_per_query_records_preserve_cluster_ids_and_eligibility(state: str, reason: str | None) -> None:
    payload = _payload()
    if state == "unknown":
        _unknown(payload["judgments"][1])
    elif state == "no-positive":
        payload["judgments"][0]["grade"] = 1
    rows = evaluate(payload)["results"][0]["per_query"]
    assert len(rows) == 1
    row = rows[0]
    assert (row["query_id"], row["compound_id"], row["study_id"], row["task_type"]) == (
        "q1", "drug-a", "held-out-study", "direct_reuse"
    )
    assert row["ranking_eligible"] is (state == "known")
    assert row["ranking_omission_reason"] == reason
    assert row["grade_complete"] is (state != "unknown")
    assert row["grade_known_count"] == (1 if state == "unknown" else 2)
    assert row["ranking_metrics"]["mrr"] == (1 if state == "known" else None)


def test_cli_writes_json_without_mutating_input(tmp_path) -> None:
    payload = _payload()
    frozen = deepcopy(payload)
    source = tmp_path / "input.json"
    target = tmp_path / "result.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["--input", str(source), "--output", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["audit"]["query_count"] == 1
    assert payload == frozen
