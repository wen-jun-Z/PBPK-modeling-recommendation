"""Metadata policy tests: identity screening is never a scientific gold label."""

from copy import deepcopy

import numpy as np
import pytest

from pbpk_kb.recommendation.pbpk_match.data import TASKS
from pbpk_kb.recommendation.pbpk_match.policy import (
    DIRECT_TASKS, TEMPLATE_TASKS, catalog_identity_policy, select_policy_topk,
)


def query(task="plasma_concentration_time_prediction", compounds=None):
    return {"task_type": task, "compounds": ["metformin"] if compounds is None else compounds}


def candidates():
    return [
        {"candidate_id": "other", "compounds": ["isoniazid"]},
        {"candidate_id": "same", "compounds": ["metformin"]},
        {"candidate_id": "missing", "compounds": []},
    ]


def test_task_partition_covers_existing_public_tasks():
    assert DIRECT_TASKS.isdisjoint(TEMPLATE_TASKS)
    assert DIRECT_TASKS | TEMPLATE_TASKS == set(TASKS)


@pytest.mark.parametrize("task", sorted(DIRECT_TASKS))
def test_direct_request_never_returns_unregistered_other_drug(task):
    policy = catalog_identity_policy(query(task), candidates())
    assert policy["eligible_mask"] == [False, True, False]
    selection = select_policy_topk([100., -4., 50.], policy, k=1)
    assert [r["candidate_id"] for r in selection["items"]] == ["same"]
    assert selection["items"][0]["raw_rank"] == 3
    assert selection["items"][0]["policy_rank"] == 1
    assert len(selection["all_candidates"]) == 3
    assert not selection["abstain"]
    assert not policy["scope"]["scientific_applicability_assessed"]


def test_multiple_compounds_require_complete_registered_coverage():
    policy = catalog_identity_policy(query("ddi_exposure_prediction", ["Drug A", "drug b"]), [
        {"candidate_id": "a", "compounds": ["drug a"]},
        {"candidate_id": "both", "compounds": [" DRUG   A ", "DRUG B", "third drug"]},
        {"candidate_id": "b", "compounds": ["drug b"]},
    ])
    assert policy["eligible_mask"] == [False, True, False]
    assert policy["candidate_status"][0]["status"] == "partial_identity_coverage"
    assert policy["candidate_status"][0]["missing_required_compounds"] == ["drug b"]


def test_missing_identity_is_metadata_uncertainty_not_a_negative_label():
    policy = catalog_identity_policy(query(), [{"candidate_id": "unknown"}])
    row = policy["candidate_status"][0]
    assert row["status"] == "candidate_identity_unknown"
    assert row["metadata_only"]
    assert not row["eligible"]
    assert "grade" not in row and "applicability" not in row
    selection = select_policy_topk([999.], policy, k=1)
    assert selection["abstain"] and selection["needs_catalog_review"]
    assert selection["items"] == []
    assert len(selection["all_candidates"]) == 1


def test_missing_query_identity_does_not_vacuously_match_every_candidate():
    policy = catalog_identity_policy(query(compounds=[]), candidates())
    assert policy["eligible_mask"] == [False, False, False]
    assert policy["needs_request_review"]
    assert policy["abstention_reason"] == "query_compounds_missing"


def test_no_matching_catalog_entry_abstains_without_other_drug_fallback():
    policy = catalog_identity_policy(query(compounds=["new drug"]), candidates())
    selection = select_policy_topk([3., 2., 1.], policy)
    assert selection["items"] == []
    assert selection["abstain"]
    assert selection["abstention_reason"] == "no_catalog_candidate_covers_all_required_compounds"
    assert len(selection["all_candidates"]) == 3


@pytest.mark.parametrize("task", sorted(TEMPLATE_TASKS))
def test_templates_allow_other_or_unregistered_compounds(task):
    policy = catalog_identity_policy(query(task), candidates())
    assert policy["eligible_mask"] == [True, True, True]
    assert [r["candidate_id"] for r in select_policy_topk([3., 2., 1.], policy, k=1)["items"]] == ["other"]
    assert all(r["metadata_only"] for r in policy["candidate_status"])


def test_exact_boundary_ties_are_complete_and_raw_ranks_preserved():
    catalog = [
        {"candidate_id": "outside", "compounds": ["other"]},
        *({"candidate_id": f"c{i}", "compounds": ["metformin"]} for i in range(4)),
    ]
    policy = catalog_identity_policy(query(), catalog)
    scores = np.asarray([100., 5., 4., 4., 3.])
    before = scores.copy()
    result = select_policy_topk(scores, policy, k=2)
    assert [r["catalog_index"] for r in result["items"]] == [1, 2, 3]
    assert result["items"][1]["policy_rank"] == 2
    assert result["items"][1]["policy_rank_end"] == 3
    assert result["items"][1]["raw_rank"] == 3
    assert result["items"][1]["raw_rank_end"] == 4
    assert result["all_candidates"][0]["policy_rank"] is None
    np.testing.assert_array_equal(scores, before)


def test_close_but_distinct_scores_are_not_rounded_into_ties():
    policy = catalog_identity_policy(query("mechanistic_template_recommendation"), candidates())
    result = select_policy_topk([1., 1. - 1e-12, .5], policy, k=1)
    assert len(result["items"]) == 1


def test_catalog_and_query_are_unchanged_and_no_scientific_labels_are_read():
    q, cs = query(), candidates()
    q["grade"] = 999
    for c in cs:
        c.update(grade=999, documented_applicability="not a label", probabilities=[1, 0, 0])
    before = deepcopy((q, cs))
    policy = catalog_identity_policy(q, cs)
    assert (q, cs) == before
    assert policy["eligible_mask"] == [False, True, False]
    assert not policy["scope"]["labels_used"]


def test_compound_normalization_and_top_level_precedence_match_fields_for():
    q = {"task_type": "special_population_prediction", "features": {"compounds": [" MetFORMIN "]}}
    policy = catalog_identity_policy(q, [
        {"features": {"compounds": ["METformin"]}},
        {"compounds": [], "features": {"compounds": ["metformin"]}},
    ])
    assert policy["eligible_mask"] == [True, False]


def test_empty_catalog_and_unsupported_task_are_explicit_abstentions():
    empty = catalog_identity_policy(query(), [])
    assert select_policy_topk([], empty)["abstention_reason"] == "catalog_empty"
    unsupported = catalog_identity_policy(query("unexpected_new_task"), candidates())
    assert unsupported["eligible_mask"] == [False, False, False]
    assert unsupported["needs_request_review"]
    assert unsupported["abstention_reason"] == "unsupported_task_type"


@pytest.mark.parametrize("values", ["metformin", [12], {"name": "metformin"}])
def test_malformed_compounds_are_not_interpreted_as_drug_characters(values):
    with pytest.raises(ValueError, match="collection of strings"):
        catalog_identity_policy(query(compounds=values), candidates())


@pytest.mark.parametrize("scores,k", [([float("nan"), 1., 0.], 1), ([2., 1., 0.], 0), ([1., 0.], 1)])
def test_invalid_ranking_inputs_fail(scores, k):
    with pytest.raises(ValueError):
        select_policy_topk(scores, catalog_identity_policy(query(), candidates()), k=k)
