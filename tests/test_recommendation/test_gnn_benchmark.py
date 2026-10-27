from __future__ import annotations

import json

import pytest

from pbpk_kb.recommendation.gnn_benchmark import (
    _calibrate_category_alphas,
    build_pair_records,
    build_relation_split,
    build_semantic_group_split,
    lambda_ndcg_loss,
    macro_f1,
    normalize_query,
    ranking_metrics,
)


def _query(
    query_id: str,
    category: str,
    text: str,
    label: dict[str, object],
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "category": category,
        "query_text": text,
        "label_basis": label,
        "target_hop": 0,
        "positive_subgraph_ids": [f"subgraph:{query_id}:positive"],
        "hard_negative_subgraph_ids": [f"subgraph:{query_id}:negative"],
    }


def test_semantic_split_keeps_label_and_text_components_disjoint() -> None:
    queries = []
    categories = (
        "ddi_role",
        "entity_query",
        "multi_hop_path",
        "parameter_query",
        "scenario_query",
        "special_population",
    )
    for category in categories:
        for group in range(3):
            label = {"target": f"{category}-{group}"}
            for variant in range(2):
                queries.append(
                    _query(
                        f"{category}-{group}-{variant}",
                        category,
                        f"{category} target {group} variant {variant}",
                        label,
                    )
                )
    assignments, report = build_semantic_group_split(queries)
    assert len(assignments) == len(queries)
    assert report["query_count_by_split"]["train"] > 0
    assert report["query_count_by_split"]["validation"] > 0
    assert report["query_count_by_split"]["test"] > 0
    for overlap in report["cross_split_overlap"].values():
        assert overlap == {"labels": 0, "texts": 0, "groups": 0}


def test_pair_records_only_mark_locked_hard_negatives_supervised() -> None:
    query = _query("q1", "entity_query", "entity one", {"target": "one"})
    assignments = [
        {
            "query_id": "q1",
            "category": "entity_query",
            "split": "train",
            "semantic_group_id": "g1",
        }
    ]
    candidates = [
        {
            "query_id": "q1",
            "candidate_subgraph_id": "subgraph:q1:positive",
            "candidate_rank": 1,
            "retriever_ranks": {"dense": 1},
            "rrf_score": 0.1,
        },
        {
            "query_id": "q1",
            "candidate_subgraph_id": "subgraph:q1:negative",
            "candidate_rank": 2,
            "retriever_ranks": {"dense": 2},
            "rrf_score": 0.09,
        },
        {
            "query_id": "q1",
            "candidate_subgraph_id": "subgraph:q1:unlabeled",
            "candidate_rank": 3,
            "retriever_ranks": {"dense": 3},
            "rrf_score": 0.08,
        },
    ]
    reranker_scores = [
        {
            "query_id": row["query_id"],
            "candidate_subgraph_id": row["candidate_subgraph_id"],
            "reranker_score": 0.9 - index * 0.2,
            "final_ranking_score": 0.9 - index * 0.2,
            "graph_constraint_pass": True,
        }
        for index, row in enumerate(candidates)
    ]
    subgraphs = {
        value["candidate_subgraph_id"]: {
            "citation_count": int(index == 0)
        }
        for index, value in enumerate(candidates)
    }
    rows, report = build_pair_records(
        queries=[query],
        candidates=candidates,
        assignments=assignments,
        subgraphs=subgraphs,
        reranker_scores=reranker_scores,
    )
    assert [row["supervised_pair"] for row in rows] == [True, True, False]
    assert report["positive_pair_count"] == 1
    assert report["hard_negative_pair_count"] == 1
    assert report["reranker_prior_pair_count"] == 3
    assert report["missing_reranker_prior_pair_count"] == 0
    assert rows[0]["retriever_ranks"] == {"dense": 1}


def test_validation_alpha_never_uses_missing_category_or_unsafe_graph() -> None:
    rows = [
        {
            "query_id": "q1",
            "subgraph_id": "s1",
            "category": "entity_query",
            "relevance_label": 1,
        },
        {
            "query_id": "q1",
            "subgraph_id": "s2",
            "category": "entity_query",
            "relevance_label": 0,
        },
    ]
    alphas = _calibrate_category_alphas(
        rows=rows,
        teacher_scores=[1.0, 0.0],
        graph_scores=[0.0, 1.0],
    )
    assert alphas["entity_query"] == 0.0
    assert alphas["ddi_role"] == 0.0


def test_relation_split_is_source_document_disjoint() -> None:
    edges = []
    for source in range(20):
        for relation in ("MEASURES", "SUPPORTED_BY"):
            edges.append(
                {
                    "edge_id": f"edge:{source}:{relation}",
                    "predicate": relation,
                    "provenance": {
                        "source_id": "source",
                        "source_location": f"document-{source}#chunk=1",
                    },
                }
            )
    rows, report = build_relation_split(edges)
    assert len(rows) == len(edges)
    assert all(
        value == 0
        for value in report["cross_split_source_group_overlap"].values()
    )
    edge_splits = {
        row["edge_id"].split(":")[1]: row["split"] for row in rows
    }
    for source in range(20):
        assert edge_splits[str(source)]


def test_ranking_metrics_and_macro_f1() -> None:
    rows = [
        {
            "query_id": "q1",
            "subgraph_id": "s1",
            "category": "entity_query",
            "relevance_label": 1,
        },
        {
            "query_id": "q1",
            "subgraph_id": "s2",
            "category": "entity_query",
            "relevance_label": 0,
        },
    ]
    metrics = ranking_metrics(rows, [1.0, 0.0])
    assert metrics["overall"]["hit_at_10"] == 1.0
    assert metrics["overall"]["reciprocal_rank"] == 1.0
    assert metrics["overall"]["ndcg_at_10"] == 1.0
    assert macro_f1([0, 1], [0, 1]) == 1.0


def test_lambda_ndcg_loss_penalizes_inverted_ranking() -> None:
    torch = pytest.importorskip("torch")
    query_rows = torch.tensor([0, 0, 0], dtype=torch.long)
    labels = torch.tensor([1.0, 0.0, 0.0])
    correct = lambda_ndcg_loss(
        torch.tensor([2.0, 1.0, 0.0]),
        query_rows,
        labels,
    )
    inverted = lambda_ndcg_loss(
        torch.tensor([0.0, 1.0, 2.0]),
        query_rows,
        labels,
    )
    assert float(correct) < float(inverted)


def test_query_normalization_is_stable() -> None:
    assert normalize_query("AUC-ratio (DDI)") == "auc ratio ddi"
