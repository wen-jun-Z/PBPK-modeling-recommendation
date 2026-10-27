from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from pbpk_kb.recommendation.embedding_benchmark import (
    _model_factory,
    load_jsonl,
    sha256_file,
    stable_id,
)


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


GNN_BENCHMARK_VERSION = "pbpk-gnn-silver-3.0.0"
GNN_TRAINING_VERSION = "pbpk-gnn-training-3.0.0"
QUERY_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ARCHITECTURES = ("no_gnn", "rgcn", "hgt", "graph_transformer")
GNN_ARCHITECTURES = ("rgcn", "hgt", "graph_transformer")
SEEDS = (17, 29, 43)
PAIR_STRUCTURE_DIM = 16
LOSS_WEIGHTS = {
    "relevance": 1.0,
    "contrastive": 0.20,
    "relation_path": 0.15,
    "category": 0.10,
    "parameter": 0.10,
    "citation": 0.05,
    "teacher_anchor": 0.10,
}
TRAINING_CONFIG = {
    "hidden_dim": 256,
    "layers": 2,
    "heads": 4,
    "dropout": 0.15,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 1.0,
    "max_epochs": 200,
    "patience": 25,
    "seeds": list(SEEDS),
    "graph_shared_parameter_learning_rate_scale": 0.20,
    "hard_negative_weight": 2.0,
    "validation_alpha_grid": [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
    ],
    "contrastive_components": {
        "multi_positive_listwise": 0.5,
        "lambda_ndcg_at_10": 0.5,
    },
}
ACCEPTANCE_THRESHOLDS = {
    "minimum_composite_improvement_over_no_gnn": 0.01,
    "minimum_ndcg_at_10_improvement_over_no_gnn": 0.01,
    "maximum_category_hit_at_10_regression": 0.10,
    "maximum_composite_standard_deviation": 0.05,
    "minimum_relation_macro_f1_delta_over_no_gnn": 0.0,
    "maximum_quarantined_evidence_count": 0,
    "checkpoint_reproduction_tolerance": 1e-7,
}
CATEGORY_ORDER = (
    "ddi_role",
    "entity_query",
    "multi_hop_path",
    "parameter_query",
    "scenario_query",
    "special_population",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_query(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _semantic_components(
    queries: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    components: list[list[dict[str, Any]]] = []
    for category in CATEGORY_ORDER:
        category_queries = [
            query for query in queries if str(query["category"]) == category
        ]
        dsu = DisjointSet(len(category_queries))
        seen: dict[str, int] = {}
        for index, query in enumerate(category_queries):
            aliases = (
                f"label:{canonical_json(query['label_basis'])}",
                f"text:{normalize_query(str(query['query_text']))}",
            )
            for alias in aliases:
                if alias in seen:
                    dsu.union(index, seen[alias])
                else:
                    seen[alias] = index
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, query in enumerate(category_queries):
            grouped[dsu.find(index)].append(query)
        components.extend(grouped.values())
    return components


def build_semantic_group_split(
    queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        components = [
            component
            for component in _semantic_components(queries)
            if str(component[0]["category"]) == category
        ]
        components.sort(
            key=lambda rows: sha256_text(
                "|".join(sorted(str(row["query_id"]) for row in rows))
            )
        )
        count = len(components)
        if count < 2:
            raise RuntimeError(
                f"Category {category} has fewer than two disjoint semantic groups."
            )
        if count == 2:
            split_names = ["train", "test"]
        else:
            validation_count = max(1, round(count * 0.15))
            test_count = max(1, round(count * 0.15))
            while validation_count + test_count >= count:
                if validation_count >= test_count and validation_count > 1:
                    validation_count -= 1
                elif test_count > 1:
                    test_count -= 1
                else:
                    break
            train_count = count - validation_count - test_count
            split_names = (
                ["train"] * train_count
                + ["validation"] * validation_count
                + ["test"] * test_count
            )
        for component, split_name in zip(components, split_names, strict=True):
            group_id = stable_id(
                "gnn-semantic-group",
                category,
                sorted(str(row["query_id"]) for row in component),
            )
            label_fingerprints = sorted(
                {
                    sha256_text(canonical_json(row["label_basis"]))
                    for row in component
                }
            )
            normalized_texts = sorted(
                {normalize_query(str(row["query_text"])) for row in component}
            )
            component_rows.append(
                {
                    "group_id": group_id,
                    "category": category,
                    "split": split_name,
                    "query_count": len(component),
                    "query_ids": sorted(str(row["query_id"]) for row in component),
                    "label_fingerprints": label_fingerprints,
                    "normalized_query_texts": normalized_texts,
                }
            )
            for query in component:
                assignments.append(
                    {
                        "query_id": str(query["query_id"]),
                        "category": category,
                        "split": split_name,
                        "semantic_group_id": group_id,
                        "label_fingerprint": sha256_text(
                            canonical_json(query["label_basis"])
                        ),
                        "normalized_query_sha256": sha256_text(
                            normalize_query(str(query["query_text"]))
                        ),
                    }
                )
    assignments.sort(key=lambda row: str(row["query_id"]))
    split_sets: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"labels": set(), "texts": set(), "groups": set()}
    )
    for row in assignments:
        split_sets[str(row["split"])]["labels"].add(
            str(row["label_fingerprint"])
        )
        split_sets[str(row["split"])]["texts"].add(
            str(row["normalized_query_sha256"])
        )
        split_sets[str(row["split"])]["groups"].add(
            str(row["semantic_group_id"])
        )
    split_names = sorted(split_sets)
    overlap: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap[f"{left}__{right}"] = {
                key: len(split_sets[left][key] & split_sets[right][key])
                for key in ("labels", "texts", "groups")
            }
    if any(value for pair in overlap.values() for value in pair.values()):
        raise RuntimeError("Semantic split leakage was detected.")
    report = {
        "method": (
            "category-stratified connected components over canonical label basis "
            "and normalized query text"
        ),
        "component_count": len(component_rows),
        "query_count_by_split": dict(
            sorted(Counter(row["split"] for row in assignments).items())
        ),
        "category_query_count_by_split": {
            split: dict(
                sorted(
                    Counter(
                        row["category"]
                        for row in assignments
                        if row["split"] == split
                    ).items()
                )
            )
            for split in ("train", "validation", "test")
        },
        "cross_split_overlap": overlap,
        "components": component_rows,
    }
    return assignments, report


def _relation_targets(label_basis: dict[str, Any]) -> list[str]:
    targets: set[str] = set()
    for key in (
        "required_predicate",
        "required_direct_subject_predicate",
    ):
        value = label_basis.get(key)
        if value:
            targets.add(str(value))
    targets.update(
        str(value)
        for value in label_basis.get("required_two_edge_predicates") or ()
        if value
    )
    return sorted(targets)


def build_pair_records(
    *,
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    subgraphs: dict[str, dict[str, Any]],
    reranker_scores: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignment_by_query = {
        str(row["query_id"]): row for row in assignments
    }
    candidates_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        candidates_by_query[str(row["query_id"])].append(row)
    reranker_by_pair = {
        (
            str(row["query_id"]),
            str(row["candidate_subgraph_id"]),
        ): row
        for row in (reranker_scores or [])
    }
    query_by_id = {str(row["query_id"]): row for row in queries}
    pair_rows: list[dict[str, Any]] = []
    missing_positive_queries: list[str] = []
    for query_id, query in sorted(query_by_id.items()):
        positive_ids = {str(value) for value in query["positive_subgraph_ids"]}
        hard_negative_ids = {
            str(value) for value in query["hard_negative_subgraph_ids"]
        }
        observed_positive = 0
        for candidate in sorted(
            candidates_by_query.get(query_id, ()),
            key=lambda row: int(row["candidate_rank"]),
        ):
            subgraph_id = str(candidate["candidate_subgraph_id"])
            if subgraph_id not in subgraphs:
                raise RuntimeError(f"Candidate subgraph is missing: {subgraph_id}")
            relevance_label = 1 if subgraph_id in positive_ids else 0
            if relevance_label:
                observed_positive += 1
            subgraph = subgraphs[subgraph_id]
            supervised = relevance_label == 1 or subgraph_id in hard_negative_ids
            reranker = reranker_by_pair.get((query_id, subgraph_id), {})
            retriever_ranks = {
                str(name): int(rank)
                for name, rank in (candidate.get("retriever_ranks") or {}).items()
            }
            pair_rows.append(
                {
                    "pair_id": stable_id("gnn-query-subgraph", query_id, subgraph_id),
                    "query_id": query_id,
                    "subgraph_id": subgraph_id,
                    "split": assignment_by_query[query_id]["split"],
                    "category": query["category"],
                    "target_hop": int(query["target_hop"]),
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "relevance_label": relevance_label,
                    "positive_count_total": len(positive_ids),
                    "supervised_pair": supervised,
                    "hard_negative": subgraph_id in hard_negative_ids,
                    "rrf_score": float(candidate.get("rrf_score") or 0.0),
                    "retriever_ranks": retriever_ranks,
                    "reranker_score": float(
                        reranker.get("reranker_score") or 0.0
                    ),
                    "final_ranking_score": float(
                        reranker.get("final_ranking_score") or 0.0
                    ),
                    "graph_constraint_pass": bool(
                        reranker.get("graph_constraint_pass", True)
                    ),
                    "path_relation_targets": _relation_targets(
                        dict(query["label_basis"])
                    ),
                    "parameter_ranking_eligible": (
                        str(query["category"]) == "parameter_query"
                    ),
                    "citation_selection_label": int(
                        relevance_label == 1
                        and int(subgraph.get("citation_count") or 0) > 0
                    ),
                }
            )
        if not observed_positive:
            missing_positive_queries.append(query_id)
    report = {
        "pair_count": len(pair_rows),
        "supervised_pair_count": sum(
            int(row["supervised_pair"]) for row in pair_rows
        ),
        "positive_pair_count": sum(
            int(row["relevance_label"]) for row in pair_rows
        ),
        "hard_negative_pair_count": sum(
            int(row["hard_negative"]) for row in pair_rows
        ),
        "reranker_prior_pair_count": sum(
            int((row["query_id"], row["subgraph_id"]) in reranker_by_pair)
            for row in pair_rows
        ),
        "missing_reranker_prior_pair_count": sum(
            int((row["query_id"], row["subgraph_id"]) not in reranker_by_pair)
            for row in pair_rows
        ),
        "candidate_pool_miss_query_count": len(missing_positive_queries),
        "candidate_pool_miss_query_ids": missing_positive_queries,
        "pair_count_by_split": dict(
            sorted(Counter(row["split"] for row in pair_rows).items())
        ),
        "positive_pair_count_by_split": {
            split: sum(
                int(row["relevance_label"])
                for row in pair_rows
                if row["split"] == split
            )
            for split in ("train", "validation", "test")
        },
    }
    return pair_rows, report


def _source_document_key(edge: dict[str, Any]) -> str:
    provenance = edge.get("provenance") or {}
    location = str(provenance.get("source_location") or "")
    location = location.split("#", 1)[0]
    source_id = str(provenance.get("source_id") or "")
    if source_id or location:
        return f"{source_id}|{location}"
    return f"edge-only|{edge['edge_id']}"


def build_relation_split(
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        groups[_source_document_key(edge)].append(edge)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: sha256_text(item[0]),
    )
    group_count = len(ordered_groups)
    boundaries = {
        "context": round(group_count * 0.55),
        "train": round(group_count * 0.70),
        "validation": round(group_count * 0.85),
    }
    rows: list[dict[str, Any]] = []
    for index, (source_group, group_edges) in enumerate(ordered_groups):
        if index < boundaries["context"]:
            split = "context"
        elif index < boundaries["train"]:
            split = "train"
        elif index < boundaries["validation"]:
            split = "validation"
        else:
            split = "test"
        for edge in group_edges:
            rows.append(
                {
                    "edge_id": str(edge["edge_id"]),
                    "source_document_group_sha256": sha256_text(source_group),
                    "relation": str(edge["predicate"]),
                    "split": split,
                }
            )
    rows.sort(key=lambda row: str(row["edge_id"]))
    group_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_sets[str(row["split"])].add(
            str(row["source_document_group_sha256"])
        )
    overlaps: dict[str, int] = {}
    split_names = ("context", "train", "validation", "test")
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = len(
                group_sets[left] & group_sets[right]
            )
    if any(overlaps.values()):
        raise RuntimeError("Source-document relation split leakage was detected.")
    relation_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        relation_by_split[str(row["split"])].add(str(row["relation"]))
    report = {
        "method": "source-document-disjoint deterministic relation reconstruction split",
        "source_document_group_count": group_count,
        "edge_count_by_split": dict(
            sorted(Counter(row["split"] for row in rows).items())
        ),
        "relation_count_by_split": {
            split: len(relation_by_split[split]) for split in split_names
        },
        "test_relations_seen_in_train": len(
            relation_by_split["test"] & relation_by_split["train"]
        ),
        "cross_split_source_group_overlap": overlaps,
    }
    return rows, report


def _load_subgraphs(subgraph_dir: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for hop in range(3):
        for row in load_jsonl(subgraph_dir / f"subgraphs_{hop}hop.jsonl"):
            subgraph_id = str(row["subgraph_id"])
            if subgraph_id in output:
                raise RuntimeError(f"Duplicate subgraph ID: {subgraph_id}")
            output[subgraph_id] = row
    return output


def prepare_benchmark(
    *,
    graph_dir: Path,
    subgraph_dir: Path,
    retrieval_benchmark_dir: Path,
    candidate_pool_path: Path,
    reranker_scores_path: Path,
    embedding_index_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    device: str,
) -> dict[str, Any]:
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_manifest = json.loads(
        (graph_dir / "kg_extraction_manifest.json").read_text(encoding="utf-8")
    )
    graph_summary = json.loads(
        (graph_dir / "kg_build_summary.json").read_text(encoding="utf-8")
    )
    gate2_manifest = json.loads(
        (subgraph_dir / "gate2_manifest.json").read_text(encoding="utf-8")
    )
    retrieval_manifest = json.loads(
        (retrieval_benchmark_dir / "benchmark_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    index_manifest = json.loads(
        (embedding_index_dir / "embedding_index_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for name, manifest in (
        ("graph", graph_summary),
        ("subgraph", gate2_manifest),
        ("retrieval benchmark", retrieval_manifest),
        ("embedding index", index_manifest),
    ):
        if manifest.get("accepted") is not True:
            raise RuntimeError(f"Upstream {name} manifest is not accepted.")
    models = index_manifest.get("models") or {}
    if QUERY_MODEL not in models:
        raise RuntimeError("Locked Qwen3 query/node embedding is missing.")

    queries = load_jsonl(
        retrieval_benchmark_dir / "locked_retrieval_silver.jsonl"
    )
    candidates = load_jsonl(candidate_pool_path)
    reranker_scores = load_jsonl(reranker_scores_path)
    nodes = load_jsonl(graph_dir / "kg_nodes.jsonl")
    edges = load_jsonl(graph_dir / "kg_edges.jsonl")
    subgraphs = _load_subgraphs(subgraph_dir)
    assignments, split_report = build_semantic_group_split(queries)
    pair_rows, pair_report = build_pair_records(
        queries=queries,
        candidates=candidates,
        assignments=assignments,
        subgraphs=subgraphs,
        reranker_scores=reranker_scores,
    )
    if pair_report["missing_reranker_prior_pair_count"]:
        raise RuntimeError(
            "Gate 3 reranker priors are incomplete: "
            f"{pair_report['missing_reranker_prior_pair_count']} missing pairs."
        )
    relation_rows, relation_report = build_relation_split(edges)

    quarantined_count = sum(
        1
        for node in nodes
        if str((node.get("properties") or {}).get("quality_status") or "")
        == "quarantined"
    )
    if quarantined_count:
        raise RuntimeError("Quarantined evidence is present in the graph.")

    query_texts = [str(row["query_text"]) for row in queries]
    encoder = _model_factory(
        QUERY_MODEL,
        cache_dir=cache_dir,
        device=device,
        batch_size=8,
    )
    expected_revision = str(models[QUERY_MODEL]["revision"])
    if str(encoder.revision) != expected_revision:
        raise RuntimeError(
            f"Qwen3 revision mismatch: {encoder.revision} != {expected_revision}"
        )
    query_vectors = np.asarray(
        encoder.encode_queries(query_texts),
        dtype=np.float32,
    )
    if query_vectors.shape != (len(queries), 768):
        raise RuntimeError(f"Unexpected query embedding shape: {query_vectors.shape}")
    del encoder
    gc.collect()

    query_records = [
        {
            "query_row": index,
            "query_id": str(query["query_id"]),
            "query_text": str(query["query_text"]),
            "query_input_sha256": sha256_text(str(query["query_text"])),
            "category": str(query["category"]),
            "target_hop": int(query["target_hop"]),
            "split": next(
                row["split"]
                for row in assignments
                if row["query_id"] == query["query_id"]
            ),
        }
        for index, query in enumerate(queries)
    ]
    subgraph_records = [
        {
            "subgraph_row": index,
            "subgraph_id": subgraph_id,
            "center_id": row["center_id"],
            "center_type": row["center_type"],
            "hop": int(row["hop_limit"]),
            "node_ids": [str(node["node_id"]) for node in row["nodes"]],
            "edge_ids": [str(edge["edge_id"]) for edge in row["edges"]],
            "event_ids": [str(event["event_id"]) for event in row["events"]],
            "citation_count": int(row["citation_count"]),
            "node_count": int(row["node_count"]),
            "edge_count": int(row["edge_count"]),
            "event_count": int(row["event_count"]),
            "content_sha256": str(row["content_sha256"]),
        }
        for index, (subgraph_id, row) in enumerate(sorted(subgraphs.items()))
    ]

    atomic_jsonl(output_dir / "query_records.jsonl", query_records)
    atomic_npy(output_dir / "query_embeddings.npy", query_vectors)
    atomic_jsonl(output_dir / "semantic_split.jsonl", assignments)
    atomic_jsonl(output_dir / "query_subgraph_pairs.jsonl", pair_rows)
    atomic_jsonl(output_dir / "relation_split.jsonl", relation_rows)
    atomic_jsonl(output_dir / "subgraph_index.jsonl", subgraph_records)
    acceptance = {
        "benchmark_version": GNN_BENCHMARK_VERSION,
        "locked_before_training": True,
        "architectures": list(ARCHITECTURES),
        "training": TRAINING_CONFIG,
        "loss_weights": LOSS_WEIGHTS,
        "thresholds": ACCEPTANCE_THRESHOLDS,
        "primary_split": split_report["method"],
        "relation_split": relation_report["method"],
    }
    atomic_json(output_dir / "preregistered_acceptance.json", acceptance)

    output_hashes = {
        name: sha256_file(output_dir / filename)
        for name, filename in {
            "query_records": "query_records.jsonl",
            "query_embeddings": "query_embeddings.npy",
            "semantic_split": "semantic_split.jsonl",
            "query_subgraph_pairs": "query_subgraph_pairs.jsonl",
            "relation_split": "relation_split.jsonl",
            "subgraph_index": "subgraph_index.jsonl",
            "acceptance": "preregistered_acceptance.json",
        }.items()
    }
    manifest = {
        "accepted": True,
        "benchmark_level": "silver",
        "benchmark_version": GNN_BENCHMARK_VERSION,
        "generated_at": utc_now(),
        "query_count": len(queries),
        "subgraph_count": len(subgraphs),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "query_embedding_model": QUERY_MODEL,
        "query_embedding_revision": expected_revision,
        "query_embedding_shape": list(query_vectors.shape),
        "split_audit": split_report,
        "pair_audit": pair_report,
        "relation_split_audit": relation_report,
        "quarantined_evidence_count": quarantined_count,
        "input_hashes": {
            "graph_manifest": sha256_file(
                graph_dir / "kg_extraction_manifest.json"
            ),
            "graph_summary": sha256_file(graph_dir / "kg_build_summary.json"),
            "nodes": sha256_file(graph_dir / "kg_nodes.jsonl"),
            "edges": sha256_file(graph_dir / "kg_edges.jsonl"),
            "subgraph_manifest": sha256_file(
                subgraph_dir / "gate2_manifest.json"
            ),
            "retrieval_benchmark_manifest": sha256_file(
                retrieval_benchmark_dir / "benchmark_manifest.json"
            ),
            "retrieval_queries": sha256_file(
                retrieval_benchmark_dir / "locked_retrieval_silver.jsonl"
            ),
            "candidate_pool": sha256_file(candidate_pool_path),
            "reranker_scores": sha256_file(reranker_scores_path),
            "embedding_index_manifest": sha256_file(
                embedding_index_dir / "embedding_index_manifest.json"
            ),
        },
        "output_hashes": output_hashes,
        "limitations": [
            "This is a provenance-backed silver retrieval benchmark, not an expert gold standard.",
            "Gate 6 must separately run compound/source/model/time generalization splits.",
            "Publication time is not present in the current graph and cannot be inferred from acquisition time.",
        ],
    }
    atomic_json(output_dir / "gnn_benchmark_manifest.json", manifest)
    return manifest


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bounded_numeric(value: Any) -> tuple[float, float]:
    number = _float_value(value)
    if number is None:
        return 0.0, 1.0
    bounded = math.copysign(math.log1p(abs(number)), number)
    return math.tanh(bounded / 8.0), 0.0


def _categorical_vocab(values: Iterable[str]) -> dict[str, int]:
    return {"<missing>": 0, **{
        value: index + 1
        for index, value in enumerate(sorted({value for value in values if value}))
    }}


@dataclass
class GraphTensorBundle:
    text_features: Any
    numeric_features: Any
    node_type_ids: Any
    source_ids: Any
    unit_ids: Any
    context_ids: Any
    forward_edge_index: Any
    full_edge_index: Any
    full_edge_type: Any
    full_edge_numeric: Any
    full_edge_source_ids: Any
    full_edge_version_ids: Any
    edge_ids: list[str]
    relation_ids: list[int]
    relation_vocab: dict[str, int]
    node_ids: list[str]
    node_types: list[str]
    node_type_vocab: dict[str, int]
    source_vocab: dict[str, int]
    unit_vocab: dict[str, int]
    context_vocab: dict[str, int]
    edge_source_vocab: dict[str, int]
    edge_version_vocab: dict[str, int]
    hetero_node_indices: dict[str, Any]
    hetero_edge_indices: dict[tuple[str, str, str], Any]

    def to(self, device: Any) -> "GraphTensorBundle":
        tensor_fields = (
            "text_features",
            "numeric_features",
            "node_type_ids",
            "source_ids",
            "unit_ids",
            "context_ids",
            "forward_edge_index",
            "full_edge_index",
            "full_edge_type",
            "full_edge_numeric",
            "full_edge_source_ids",
            "full_edge_version_ids",
        )
        values = self.__dict__.copy()
        for name in tensor_fields:
            values[name] = values[name].to(device)
        values["hetero_node_indices"] = {
            key: value.to(device)
            for key, value in self.hetero_node_indices.items()
        }
        values["hetero_edge_indices"] = {
            key: value.to(device)
            for key, value in self.hetero_edge_indices.items()
        }
        return GraphTensorBundle(**values)


def build_graph_tensors(
    *,
    graph_dir: Path,
    embedding_index_dir: Path,
) -> GraphTensorBundle:
    import numpy as np
    import torch

    nodes = load_jsonl(graph_dir / "kg_nodes.jsonl")
    edges = load_jsonl(graph_dir / "kg_edges.jsonl")
    node_ids = [str(row["node_id"]) for row in nodes]
    node_index = {value: index for index, value in enumerate(node_ids)}
    if len(node_index) != len(nodes):
        raise RuntimeError("Duplicate node IDs in graph.")
    node_types = [str(row["node_type"]) for row in nodes]
    node_type_vocab = _categorical_vocab(node_types)

    index_manifest = json.loads(
        (embedding_index_dir / "embedding_index_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    qwen = index_manifest["models"][QUERY_MODEL]
    node_artifact = qwen["artifacts"]["node"]
    source_records = load_jsonl(
        embedding_index_dir
        / index_manifest["source_records"]["node"]["records_path"]
    )
    vectors = np.load(
        embedding_index_dir / node_artifact["vectors_path"],
        mmap_mode="r",
    )
    source_vector_by_id = {
        str(row["object_id"]): np.asarray(vectors[index], dtype=np.float32)
        for index, row in enumerate(source_records)
    }
    if set(source_vector_by_id) != set(node_ids):
        raise RuntimeError("Node embedding IDs do not align with graph nodes.")
    text_features = torch.from_numpy(
        np.stack([source_vector_by_id[node_id] for node_id in node_ids])
    )

    source_values: list[str] = []
    unit_values: list[str] = []
    context_values: list[str] = []
    numeric_rows: list[list[float]] = []
    for node in nodes:
        properties = node.get("properties") or {}
        source = str(properties.get("source_id") or "")
        unit = str(
            properties.get("canonical_unit")
            or properties.get("reported_unit")
            or ""
        )
        context = canonical_json(
            {
                key: properties.get(key)
                for key in (
                    "route",
                    "population",
                    "condition",
                    "formulation",
                    "dose",
                    "qualifier",
                    "evidence_type",
                )
                if properties.get(key) not in (None, "", [], {})
            }
        )
        if context == "{}":
            context = ""
        source_values.append(source)
        unit_values.append(unit)
        context_values.append(context)
        numeric: list[float] = []
        for key in ("value", "value_low", "value_high", "extraction_confidence"):
            transformed, missing = _bounded_numeric(properties.get(key))
            numeric.extend((transformed, missing))
        numeric.extend(
            (
                float(properties.get("quality_status") == "auto_validated"),
                float(bool(source)),
                float(bool(unit)),
                float(bool(context)),
            )
        )
        numeric_rows.append(numeric)
    source_vocab = _categorical_vocab(source_values)
    unit_vocab = _categorical_vocab(unit_values)
    context_vocab = _categorical_vocab(context_values)

    relation_names = sorted({str(row["predicate"]) for row in edges})
    relation_vocab = {value: index for index, value in enumerate(relation_names)}
    edge_source_values = [
        str((row.get("provenance") or {}).get("source_id") or "")
        for row in edges
    ]
    edge_version_values = [
        str((row.get("provenance") or {}).get("extractor_version") or "")
        for row in edges
    ]
    edge_source_vocab = _categorical_vocab(edge_source_values)
    edge_version_vocab = _categorical_vocab(edge_version_values)
    forward_sources: list[int] = []
    forward_targets: list[int] = []
    relation_ids: list[int] = []
    edge_numeric: list[list[float]] = []
    edge_source_ids: list[int] = []
    edge_version_ids: list[int] = []
    edge_ids: list[str] = []
    for edge in edges:
        subject = str(edge["subject_id"])
        object_ = str(edge["object_id"])
        if subject not in node_index or object_ not in node_index:
            raise RuntimeError(f"Edge endpoint is absent: {edge['edge_id']}")
        provenance = edge.get("provenance") or {}
        confidence, confidence_missing = _bounded_numeric(
            provenance.get("confidence")
        )
        source = str(provenance.get("source_id") or "")
        version = str(provenance.get("extractor_version") or "")
        context_present = bool(provenance.get("context_id"))
        edge_numeric.append(
            [
                confidence,
                float(provenance.get("negation_status") == "affirmed"),
                float(provenance.get("speculation_status") == "asserted"),
                float(bool(source)),
                float(context_present),
                float(
                    str(edge.get("subject_type") or "").endswith("Event")
                    or str(edge.get("subject_type") or "")
                    in {"PKMeasurement", "DoseRegimen"}
                ),
                float(bool(version)),
                0.0,
                confidence_missing,
                float(not source),
                float(not context_present),
                float(not version),
            ]
        )
        forward_sources.append(node_index[subject])
        forward_targets.append(node_index[object_])
        relation_ids.append(relation_vocab[str(edge["predicate"])])
        edge_source_ids.append(edge_source_vocab.get(source, 0))
        edge_version_ids.append(edge_version_vocab.get(version, 0))
        edge_ids.append(str(edge["edge_id"]))
    forward_edge_index = torch.tensor(
        [forward_sources, forward_targets], dtype=torch.long
    )
    relation_tensor = torch.tensor(relation_ids, dtype=torch.long)
    reverse_edge_index = forward_edge_index.flip(0)
    relation_count = len(relation_vocab)
    full_edge_type = torch.cat(
        (relation_tensor, relation_tensor + relation_count)
    )
    forward_edge_numeric = torch.tensor(edge_numeric, dtype=torch.float32)
    reverse_edge_numeric = forward_edge_numeric.clone()
    reverse_edge_numeric[:, 7] = 1.0
    full_edge_numeric = torch.cat(
        (forward_edge_numeric, reverse_edge_numeric), dim=0
    )
    edge_source_tensor = torch.tensor(edge_source_ids, dtype=torch.long)
    edge_version_tensor = torch.tensor(edge_version_ids, dtype=torch.long)

    hetero_node_indices: dict[str, Any] = {}
    local_index: dict[str, dict[int, int]] = {}
    for node_type in sorted(set(node_types)):
        indices = [
            index for index, value in enumerate(node_types) if value == node_type
        ]
        hetero_node_indices[node_type] = torch.tensor(indices, dtype=torch.long)
        local_index[node_type] = {
            global_index: local for local, global_index in enumerate(indices)
        }
    hetero_edges: dict[tuple[str, str, str], list[list[int]]] = defaultdict(
        lambda: [[], []]
    )
    for edge_index, edge in enumerate(edges):
        source_global = forward_sources[edge_index]
        target_global = forward_targets[edge_index]
        source_type = node_types[source_global]
        target_type = node_types[target_global]
        predicate = str(edge["predicate"])
        forward_key = (source_type, f"FWD__{predicate}", target_type)
        reverse_key = (target_type, f"REV__{predicate}", source_type)
        hetero_edges[forward_key][0].append(
            local_index[source_type][source_global]
        )
        hetero_edges[forward_key][1].append(
            local_index[target_type][target_global]
        )
        hetero_edges[reverse_key][0].append(
            local_index[target_type][target_global]
        )
        hetero_edges[reverse_key][1].append(
            local_index[source_type][source_global]
        )
    hetero_edge_indices = {
        key: torch.tensor(value, dtype=torch.long)
        for key, value in hetero_edges.items()
    }
    return GraphTensorBundle(
        text_features=text_features,
        numeric_features=torch.tensor(numeric_rows, dtype=torch.float32),
        node_type_ids=torch.tensor(
            [node_type_vocab[value] for value in node_types], dtype=torch.long
        ),
        source_ids=torch.tensor(
            [source_vocab.get(value, 0) for value in source_values],
            dtype=torch.long,
        ),
        unit_ids=torch.tensor(
            [unit_vocab.get(value, 0) for value in unit_values],
            dtype=torch.long,
        ),
        context_ids=torch.tensor(
            [context_vocab.get(value, 0) for value in context_values],
            dtype=torch.long,
        ),
        forward_edge_index=forward_edge_index,
        full_edge_index=torch.cat(
            (forward_edge_index, reverse_edge_index), dim=1
        ),
        full_edge_type=full_edge_type,
        full_edge_numeric=full_edge_numeric,
        full_edge_source_ids=torch.cat(
            (edge_source_tensor, edge_source_tensor)
        ),
        full_edge_version_ids=torch.cat(
            (edge_version_tensor, edge_version_tensor)
        ),
        edge_ids=edge_ids,
        relation_ids=relation_ids,
        relation_vocab=relation_vocab,
        node_ids=node_ids,
        node_types=node_types,
        node_type_vocab=node_type_vocab,
        source_vocab=source_vocab,
        unit_vocab=unit_vocab,
        context_vocab=context_vocab,
        edge_source_vocab=edge_source_vocab,
        edge_version_vocab=edge_version_vocab,
        hetero_node_indices=hetero_node_indices,
        hetero_edge_indices=hetero_edge_indices,
    )


class NodeFeatureEncoder:
    def __init__(
        self,
        *,
        hidden_dim: int,
        dropout: float,
        graph: GraphTensorBundle,
    ) -> None:
        import torch

        nn = torch.nn
        self.module = nn.Module()
        self.module.text = nn.Linear(768, hidden_dim)
        self.module.numeric = nn.Sequential(
            nn.Linear(int(graph.numeric_features.shape[1]), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.module.node_type = nn.Embedding(
            len(graph.node_type_vocab) + 1, hidden_dim
        )
        self.module.source = nn.Embedding(
            len(graph.source_vocab) + 1, hidden_dim
        )
        self.module.unit = nn.Embedding(len(graph.unit_vocab) + 1, hidden_dim)
        self.module.context = nn.Embedding(
            len(graph.context_vocab) + 1, hidden_dim
        )
        self.module.norm = nn.LayerNorm(hidden_dim)
        self.module.dropout = nn.Dropout(dropout)

    def __call__(self, graph: GraphTensorBundle) -> Any:
        import torch

        module = self.module
        value = (
            module.text(graph.text_features)
            + module.numeric(graph.numeric_features)
            + module.node_type(graph.node_type_ids)
            + module.source(graph.source_ids)
            + module.unit(graph.unit_ids)
            + module.context(graph.context_ids)
        )
        return module.dropout(torch.nn.functional.gelu(module.norm(value)))


def _register_child(parent: Any, name: str, child: NodeFeatureEncoder) -> None:
    setattr(parent, name, child.module)


def build_ranker(
    *,
    architecture: str,
    graph: GraphTensorBundle,
    category_count: int,
) -> Any:
    import torch
    from torch_geometric.nn import RGCNConv, TransformerConv
    from torch_geometric.utils import softmax

    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    nn = torch.nn
    hidden_dim = int(TRAINING_CONFIG["hidden_dim"])
    heads = int(TRAINING_CONFIG["heads"])
    layers = int(TRAINING_CONFIG["layers"])
    dropout = float(TRAINING_CONFIG["dropout"])

    class DeterministicHGTLayer(nn.Module):
        """HGT attention with explicit, checkpointed relation parameters."""

        def __init__(self) -> None:
            super().__init__()
            self.node_types = sorted(graph.hetero_node_indices)
            self.edge_types = sorted(graph.hetero_edge_indices)
            self.node_key = {
                node_type: f"n{index}"
                for index, node_type in enumerate(self.node_types)
            }
            self.edge_key = {
                edge_type: f"e{index}"
                for index, edge_type in enumerate(self.edge_types)
            }
            self.q = nn.ModuleDict(
                {
                    self.node_key[node_type]: nn.Linear(
                        hidden_dim, hidden_dim, bias=False
                    )
                    for node_type in self.node_types
                }
            )
            self.k = nn.ModuleDict(
                {
                    self.node_key[node_type]: nn.Linear(
                        hidden_dim, hidden_dim, bias=False
                    )
                    for node_type in self.node_types
                }
            )
            self.v = nn.ModuleDict(
                {
                    self.node_key[node_type]: nn.Linear(
                        hidden_dim, hidden_dim, bias=False
                    )
                    for node_type in self.node_types
                }
            )
            self.out = nn.ModuleDict(
                {
                    self.node_key[node_type]: nn.Linear(
                        hidden_dim, hidden_dim
                    )
                    for node_type in self.node_types
                }
            )
            head_dim = hidden_dim // heads
            self.relation_k = nn.ParameterDict(
                {
                    self.edge_key[edge_type]: nn.Parameter(
                        torch.empty(heads, head_dim, head_dim)
                    )
                    for edge_type in self.edge_types
                }
            )
            self.relation_v = nn.ParameterDict(
                {
                    self.edge_key[edge_type]: nn.Parameter(
                        torch.empty(heads, head_dim, head_dim)
                    )
                    for edge_type in self.edge_types
                }
            )
            self.relation_prior = nn.ParameterDict(
                {
                    self.edge_key[edge_type]: nn.Parameter(torch.ones(heads))
                    for edge_type in self.edge_types
                }
            )
            self.skip = nn.ParameterDict(
                {
                    self.node_key[node_type]: nn.Parameter(torch.ones(1))
                    for node_type in self.node_types
                }
            )
            for parameter in self.relation_k.values():
                nn.init.xavier_uniform_(parameter)
            for parameter in self.relation_v.values():
                nn.init.xavier_uniform_(parameter)

        def forward(
            self,
            x_dict: dict[str, Any],
            edge_index_dict: dict[tuple[str, str, str], Any],
        ) -> dict[str, Any]:
            head_dim = hidden_dim // heads
            q_dict = {
                node_type: self.q[self.node_key[node_type]](value).view(
                    -1, heads, head_dim
                )
                for node_type, value in x_dict.items()
            }
            k_dict = {
                node_type: self.k[self.node_key[node_type]](value).view(
                    -1, heads, head_dim
                )
                for node_type, value in x_dict.items()
            }
            v_dict = {
                node_type: self.v[self.node_key[node_type]](value).view(
                    -1, heads, head_dim
                )
                for node_type, value in x_dict.items()
            }
            incoming: dict[str, list[tuple[Any, Any, Any]]] = defaultdict(list)
            for edge_type in self.edge_types:
                edge_index = edge_index_dict[edge_type]
                if int(edge_index.shape[1]) == 0:
                    continue
                source_type, _, target_type = edge_type
                source_index = edge_index[0]
                target_index = edge_index[1]
                relation_key = self.edge_key[edge_type]
                transformed_k = torch.einsum(
                    "ehd,hdf->ehf",
                    k_dict[source_type][source_index],
                    self.relation_k[relation_key],
                )
                transformed_v = torch.einsum(
                    "ehd,hdf->ehf",
                    v_dict[source_type][source_index],
                    self.relation_v[relation_key],
                )
                logits = (
                    q_dict[target_type][target_index] * transformed_k
                ).sum(dim=-1)
                logits = (
                    logits
                    * self.relation_prior[relation_key].unsqueeze(0)
                    / math.sqrt(head_dim)
                )
                incoming[target_type].append(
                    (target_index, logits, transformed_v)
                )

            output: dict[str, Any] = {}
            for node_type, value in x_dict.items():
                if node_type not in incoming:
                    output[node_type] = value
                    continue
                target_indices = torch.cat(
                    [item[0] for item in incoming[node_type]], dim=0
                )
                logits = torch.cat(
                    [item[1] for item in incoming[node_type]], dim=0
                )
                messages = torch.cat(
                    [item[2] for item in incoming[node_type]], dim=0
                )
                weights = torch.stack(
                    [
                        softmax(
                            logits[:, head],
                            target_indices,
                            num_nodes=int(value.shape[0]),
                        )
                        for head in range(heads)
                    ],
                    dim=1,
                )
                aggregated = torch.zeros(
                    int(value.shape[0]),
                    heads,
                    head_dim,
                    device=value.device,
                    dtype=value.dtype,
                )
                aggregated.index_add_(
                    0,
                    target_indices,
                    messages * weights.unsqueeze(-1),
                )
                transformed = self.out[self.node_key[node_type]](
                    aggregated.reshape(-1, hidden_dim)
                )
                alpha = torch.sigmoid(self.skip[self.node_key[node_type]])
                output[node_type] = alpha * transformed
            return output

    class Ranker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            feature_encoder = NodeFeatureEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
                graph=graph,
            )
            _register_child(self, "features", feature_encoder)
            self._feature_encoder = feature_encoder
            self.architecture = architecture
            self.dropout = nn.Dropout(dropout)
            self.norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(layers)]
            )
            self.layer_gates = nn.Parameter(
                torch.full((layers,), -3.0),
                requires_grad=architecture != "no_gnn",
            )
            self.edge_relation_embedding = nn.Embedding(
                len(graph.relation_vocab) * 2, hidden_dim
            )
            self.edge_source_embedding = nn.Embedding(
                len(graph.edge_source_vocab) + 1, hidden_dim
            )
            self.edge_version_embedding = nn.Embedding(
                len(graph.edge_version_vocab) + 1, hidden_dim
            )
            self.edge_numeric = nn.Linear(
                int(graph.full_edge_numeric.shape[1]), hidden_dim
            )
            if architecture == "rgcn":
                self.layers = nn.ModuleList(
                    [
                        RGCNConv(
                            hidden_dim,
                            hidden_dim,
                            num_relations=len(graph.relation_vocab) * 2,
                            num_bases=min(8, len(graph.relation_vocab) * 2),
                        )
                        for _ in range(layers)
                    ]
                )
            elif architecture == "graph_transformer":
                self.layers = nn.ModuleList(
                    [
                        TransformerConv(
                            hidden_dim,
                            hidden_dim // heads,
                            heads=heads,
                            concat=True,
                            dropout=dropout,
                            edge_dim=hidden_dim,
                            beta=True,
                        )
                        for _ in range(layers)
                    ]
                )
            elif architecture == "hgt":
                self.layers = nn.ModuleList(
                    [DeterministicHGTLayer() for _ in range(layers)]
                )
            else:
                self.layers = nn.ModuleList()

            self.query_projection = nn.Sequential(
                nn.Linear(768, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.structure_projection = nn.Sequential(
                nn.Linear(PAIR_STRUCTURE_DIM, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            pair_dim = hidden_dim * 4
            self.pair_encoder = nn.Sequential(
                nn.Linear(pair_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.relevance_head = nn.Linear(hidden_dim, 1)
            self.category_head = nn.Linear(hidden_dim, category_count)
            self.parameter_head = nn.Linear(hidden_dim, 1)
            self.citation_head = nn.Linear(hidden_dim, 1)
            self.path_head = nn.Linear(hidden_dim, len(graph.relation_vocab))
            self.relation_head = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(graph.relation_vocab)),
            )

        def edge_features(self, current_graph: GraphTensorBundle) -> Any:
            return (
                self.edge_relation_embedding(current_graph.full_edge_type)
                + self.edge_source_embedding(
                    current_graph.full_edge_source_ids
                )
                + self.edge_version_embedding(
                    current_graph.full_edge_version_ids
                )
                + self.edge_numeric(current_graph.full_edge_numeric)
            )

        def encode_nodes(self, current_graph: GraphTensorBundle) -> Any:
            x = self._feature_encoder(current_graph)
            if self.architecture == "no_gnn":
                return x
            if self.architecture == "hgt":
                x_dict = {
                    node_type: x[indices]
                    for node_type, indices in current_graph.hetero_node_indices.items()
                }
                for layer_index, (layer, norm) in enumerate(
                    zip(self.layers, self.norms, strict=True)
                ):
                    update = layer(
                        x_dict,
                        current_graph.hetero_edge_indices,
                    )
                    next_dict = {}
                    for node_type, value in x_dict.items():
                        delta = update.get(node_type)
                        next_dict[node_type] = (
                            value
                            if delta is None
                            else value
                            + torch.sigmoid(self.layer_gates[layer_index])
                            * self.dropout(
                                torch.nn.functional.gelu(
                                    norm(delta)
                                )
                            )
                        )
                    x_dict = next_dict
                output = torch.empty_like(x)
                for node_type, indices in current_graph.hetero_node_indices.items():
                    output[indices] = x_dict[node_type]
                return output
            edge_features = self.edge_features(current_graph)
            for layer_index, (layer, norm) in enumerate(
                zip(self.layers, self.norms, strict=True)
            ):
                if self.architecture == "rgcn":
                    update = layer(
                        x,
                        current_graph.full_edge_index,
                        current_graph.full_edge_type,
                    )
                else:
                    update = layer(
                        x,
                        current_graph.full_edge_index,
                        edge_features,
                    )
                x = x + torch.sigmoid(self.layer_gates[layer_index]) * self.dropout(
                    torch.nn.functional.gelu(norm(update))
                )
            return x

        def score_pairs(
            self,
            *,
            node_embeddings: Any,
            query_embeddings: Any,
            pair_query_rows: Any,
            pair_member_nodes: Any,
            pair_member_rows: Any,
            structure_features: Any,
            return_attention: bool = False,
        ) -> dict[str, Any]:
            query_hidden = self.query_projection(
                query_embeddings[pair_query_rows]
            )
            member_query = query_hidden[pair_member_rows]
            member_nodes = node_embeddings[pair_member_nodes]
            attention_logits = (
                member_query * member_nodes
            ).sum(dim=1) / math.sqrt(hidden_dim)
            attention = softmax(attention_logits, pair_member_rows)
            pooled = torch.zeros_like(query_hidden)
            pooled.index_add_(
                0,
                pair_member_rows,
                member_nodes * attention.unsqueeze(1),
            )
            pooled = pooled + self.structure_projection(structure_features)
            pair_representation = self.pair_encoder(
                torch.cat(
                    (
                        query_hidden,
                        pooled,
                        query_hidden * pooled,
                        (query_hidden - pooled).abs(),
                    ),
                    dim=1,
                )
            )
            result = {
                "scores": self.relevance_head(pair_representation).squeeze(1),
                "category_logits": self.category_head(pair_representation),
                "parameter_scores": self.parameter_head(
                    pair_representation
                ).squeeze(1),
                "citation_scores": self.citation_head(
                    pair_representation
                ).squeeze(1),
                "path_logits": self.path_head(pair_representation),
                "pair_representation": pair_representation,
            }
            if return_attention:
                result["attention"] = attention
            return result

        def relation_logits(
            self,
            node_embeddings: Any,
            edge_index: Any,
        ) -> Any:
            source = node_embeddings[edge_index[0]]
            target = node_embeddings[edge_index[1]]
            return self.relation_head(
                torch.cat(
                    (
                        source,
                        target,
                        source * target,
                        (source - target).abs(),
                    ),
                    dim=1,
                )
            )

    return Ranker()


@dataclass
class PairTensorBundle:
    rows: list[dict[str, Any]]
    query_rows: Any
    labels: Any
    hard_negative: Any
    categories: Any
    parameter_eligible: Any
    citation_labels: Any
    path_targets: Any
    member_nodes: Any
    member_pair_rows: Any
    structure_features: Any
    query_ids: list[str]
    subgraph_ids: list[str]

    def to(self, device: Any) -> "PairTensorBundle":
        return PairTensorBundle(
            rows=self.rows,
            query_rows=self.query_rows.to(device),
            labels=self.labels.to(device),
            hard_negative=self.hard_negative.to(device),
            categories=self.categories.to(device),
            parameter_eligible=self.parameter_eligible.to(device),
            citation_labels=self.citation_labels.to(device),
            path_targets=self.path_targets.to(device),
            member_nodes=self.member_nodes.to(device),
            member_pair_rows=self.member_pair_rows.to(device),
            structure_features=self.structure_features.to(device),
            query_ids=self.query_ids,
            subgraph_ids=self.subgraph_ids,
        )


def build_pair_tensors(
    *,
    rows: list[dict[str, Any]],
    query_row_by_id: dict[str, int],
    subgraph_by_id: dict[str, dict[str, Any]],
    node_index: dict[str, int],
    relation_vocab: dict[str, int],
) -> PairTensorBundle:
    import torch

    member_nodes: list[int] = []
    member_pairs: list[int] = []
    structure: list[list[float]] = []
    path_targets: list[list[float]] = []
    for pair_index, row in enumerate(rows):
        subgraph = subgraph_by_id[str(row["subgraph_id"])]
        nodes = [
            node_index[str(node_id)]
            for node_id in subgraph["node_ids"]
            if str(node_id) in node_index
        ]
        if not nodes:
            raise RuntimeError(f"Subgraph has no graph nodes: {row['subgraph_id']}")
        member_nodes.extend(nodes)
        member_pairs.extend([pair_index] * len(nodes))
        retriever_ranks = [
            int(value)
            for value in (row.get("retriever_ranks") or {}).values()
            if int(value) > 0
        ]
        reranker_score = min(
            max(float(row.get("reranker_score") or 0.0), 1e-6),
            1.0 - 1e-6,
        )
        final_ranking_score = min(
            max(float(row.get("final_ranking_score") or 0.0), 1e-6),
            1.0 - 1e-6,
        )
        structure_row = [
            math.log1p(int(subgraph["node_count"])) / 6.0,
            math.log1p(int(subgraph["edge_count"])) / 6.0,
            math.log1p(int(subgraph["event_count"])) / 6.0,
            math.log1p(int(subgraph["citation_count"])) / 6.0,
            float(subgraph["hop"]) / 2.0,
            float(int(subgraph["citation_count"]) > 0),
            float(int(subgraph["edge_count"]) > 0),
            float(int(subgraph["event_count"]) > 0),
            1.0 / math.log2(int(row["candidate_rank"]) + 1.0),
            min(float(row.get("rrf_score") or 0.0) * 10.0, 1.0),
            math.log(reranker_score / (1.0 - reranker_score)) / 8.0,
            math.log(
                final_ranking_score / (1.0 - final_ranking_score)
            )
            / 8.0,
            float(bool(row.get("graph_constraint_pass", True))),
            min(len(retriever_ranks) / 6.0, 1.0),
            (
                sum(1.0 / rank for rank in retriever_ranks)
                / len(retriever_ranks)
                if retriever_ranks
                else 0.0
            ),
            max((1.0 / rank for rank in retriever_ranks), default=0.0),
        ]
        if len(structure_row) != PAIR_STRUCTURE_DIM:
            raise RuntimeError("Pair structure feature width changed.")
        structure.append(structure_row)
        target = [0.0] * len(relation_vocab)
        for relation in row["path_relation_targets"]:
            if relation in relation_vocab:
                target[relation_vocab[relation]] = 1.0
        path_targets.append(target)
    return PairTensorBundle(
        rows=rows,
        query_rows=torch.tensor(
            [query_row_by_id[str(row["query_id"])] for row in rows],
            dtype=torch.long,
        ),
        labels=torch.tensor(
            [float(row["relevance_label"]) for row in rows],
            dtype=torch.float32,
        ),
        hard_negative=torch.tensor(
            [bool(row["hard_negative"]) for row in rows],
            dtype=torch.bool,
        ),
        categories=torch.tensor(
            [CATEGORY_ORDER.index(str(row["category"])) for row in rows],
            dtype=torch.long,
        ),
        parameter_eligible=torch.tensor(
            [bool(row["parameter_ranking_eligible"]) for row in rows],
            dtype=torch.bool,
        ),
        citation_labels=torch.tensor(
            [float(row["citation_selection_label"]) for row in rows],
            dtype=torch.float32,
        ),
        path_targets=torch.tensor(path_targets, dtype=torch.float32),
        member_nodes=torch.tensor(member_nodes, dtype=torch.long),
        member_pair_rows=torch.tensor(member_pairs, dtype=torch.long),
        structure_features=torch.tensor(structure, dtype=torch.float32),
        query_ids=[str(row["query_id"]) for row in rows],
        subgraph_ids=[str(row["subgraph_id"]) for row in rows],
    )


def _subset_graph(
    graph: GraphTensorBundle,
    forward_indices: Sequence[int],
) -> GraphTensorBundle:
    import torch

    device = graph.forward_edge_index.device
    indices = torch.tensor(
        list(forward_indices), dtype=torch.long, device=device
    )
    count = int(graph.forward_edge_index.shape[1])
    reverse_indices = indices + count
    full_indices = torch.cat((indices, reverse_indices))
    values = graph.__dict__.copy()
    values["forward_edge_index"] = graph.forward_edge_index[:, indices]
    values["full_edge_index"] = graph.full_edge_index[:, full_indices]
    values["full_edge_type"] = graph.full_edge_type[full_indices]
    values["full_edge_numeric"] = graph.full_edge_numeric[full_indices]
    values["full_edge_source_ids"] = graph.full_edge_source_ids[full_indices]
    values["full_edge_version_ids"] = graph.full_edge_version_ids[full_indices]
    selected_edge_ids = {graph.edge_ids[index] for index in forward_indices}
    values["edge_ids"] = [graph.edge_ids[index] for index in forward_indices]
    values["relation_ids"] = [
        graph.relation_ids[index] for index in forward_indices
    ]

    local_index: dict[str, dict[int, int]] = {}
    for node_type, tensor in graph.hetero_node_indices.items():
        local_index[node_type] = {
            int(global_index): local
            for local, global_index in enumerate(tensor.tolist())
        }
    hetero_edges: dict[tuple[str, str, str], list[list[int]]] = defaultdict(
        lambda: [[], []]
    )
    for index in forward_indices:
        source_global = int(graph.forward_edge_index[0, index])
        target_global = int(graph.forward_edge_index[1, index])
        source_type = graph.node_types[source_global]
        target_type = graph.node_types[target_global]
        relation_name = next(
            key
            for key, value in graph.relation_vocab.items()
            if value == graph.relation_ids[index]
        )
        forward_key = (source_type, f"FWD__{relation_name}", target_type)
        reverse_key = (target_type, f"REV__{relation_name}", source_type)
        hetero_edges[forward_key][0].append(
            local_index[source_type][source_global]
        )
        hetero_edges[forward_key][1].append(
            local_index[target_type][target_global]
        )
        hetero_edges[reverse_key][0].append(
            local_index[target_type][target_global]
        )
        hetero_edges[reverse_key][1].append(
            local_index[source_type][source_global]
        )
    # HGT metadata is fixed at construction time, so retain empty edge types.
    values["hetero_edge_indices"] = {
        key: torch.tensor(
            hetero_edges.get(key, [[], []]),
            dtype=torch.long,
            device=device,
        ).reshape(2, -1)
        for key in graph.hetero_edge_indices
    }
    if selected_edge_ids != set(values["edge_ids"]):
        raise RuntimeError("Graph subset alignment failed.")
    return GraphTensorBundle(**values)


def _pair_forward(
    model: Any,
    node_embeddings: Any,
    query_embeddings: Any,
    pairs: PairTensorBundle,
    *,
    return_attention: bool = False,
) -> dict[str, Any]:
    return model.score_pairs(
        node_embeddings=node_embeddings,
        query_embeddings=query_embeddings,
        pair_query_rows=pairs.query_rows,
        pair_member_nodes=pairs.member_nodes,
        pair_member_rows=pairs.member_pair_rows,
        structure_features=pairs.structure_features,
        return_attention=return_attention,
    )


def multi_positive_contrastive_loss(
    scores: Any,
    query_rows: Any,
    labels: Any,
) -> Any:
    import torch

    losses = []
    for query_row in torch.unique(query_rows):
        mask = query_rows == query_row
        query_scores = scores[mask]
        query_labels = labels[mask] > 0.5
        if query_labels.any() and (~query_labels).any():
            losses.append(
                -(
                    torch.logsumexp(query_scores[query_labels], dim=0)
                    - torch.logsumexp(query_scores, dim=0)
                )
            )
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def lambda_ndcg_loss(
    scores: Any,
    query_rows: Any,
    labels: Any,
    *,
    limit: int = 10,
) -> Any:
    import torch

    losses = []
    for query_row in torch.unique(query_rows):
        mask = query_rows == query_row
        query_scores = scores[mask]
        query_labels = labels[mask] > 0.5
        positive_indices = torch.where(query_labels)[0]
        negative_indices = torch.where(~query_labels)[0]
        if not len(positive_indices) or not len(negative_indices):
            continue

        order = torch.argsort(query_scores.detach(), descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(
            1,
            len(order) + 1,
            device=order.device,
        )

        def discount(selected_ranks: Any) -> Any:
            values = 1.0 / torch.log2(selected_ranks.float() + 1.0)
            return torch.where(
                selected_ranks <= limit,
                values,
                torch.zeros_like(values),
            )

        positive_discount = discount(ranks[positive_indices]).unsqueeze(1)
        negative_discount = discount(ranks[negative_indices]).unsqueeze(0)
        ideal_count = min(int(query_labels.sum().item()), limit)
        ideal_ranks = torch.arange(
            1,
            ideal_count + 1,
            device=scores.device,
        )
        ideal_dcg = discount(ideal_ranks).sum().clamp_min(1e-12)
        delta_ndcg = (
            positive_discount - negative_discount
        ).abs() / ideal_dcg
        pairwise = torch.nn.functional.softplus(
            -(
                query_scores[positive_indices].unsqueeze(1)
                - query_scores[negative_indices].unsqueeze(0)
            )
        )
        weight_sum = delta_ndcg.sum()
        if float(weight_sum.detach().cpu()) > 0.0:
            losses.append((pairwise * delta_ndcg).sum() / weight_sum)
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def query_balanced_relevance_loss(
    scores: Any,
    query_rows: Any,
    labels: Any,
    hard_negative: Any,
) -> Any:
    import torch

    losses = []
    for query_row in torch.unique(query_rows):
        mask = query_rows == query_row
        query_labels = labels[mask]
        positive_count = query_labels.sum()
        negative_count = float(len(query_labels)) - positive_count
        positive_weight = negative_count / positive_count.clamp_min(1.0)
        weights = torch.ones_like(query_labels)
        weights = torch.where(
            query_labels > 0.5,
            weights * positive_weight,
            weights,
        )
        weights = torch.where(
            hard_negative[mask],
            weights * float(TRAINING_CONFIG["hard_negative_weight"]),
            weights,
        )
        losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                scores[mask],
                query_labels,
                weight=weights,
            )
        )
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def _dcg(labels: list[int], limit: int) -> float:
    return sum(
        label / math.log2(index + 2)
        for index, label in enumerate(labels[:limit])
    )


def ranking_metrics(
    rows: list[dict[str, Any]],
    scores: Sequence[float],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[str(row["query_id"])].append((row, float(score)))
    per_query: list[dict[str, Any]] = []
    for query_id, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: (-item[1], str(item[0]["subgraph_id"])))
        labels = [int(item[0]["relevance_label"]) for item in candidates]
        positive_count = int(
            candidates[0][0].get("positive_count_total") or sum(labels)
        )
        if positive_count <= 0:
            raise RuntimeError(f"Query has no locked positives: {query_id}")
        first_positive = next(
            (index + 1 for index, label in enumerate(labels) if label),
            None,
        )
        ideal = [1] * min(positive_count, len(labels))
        ideal.extend([0] * (len(labels) - len(ideal)))
        ndcg = _dcg(labels, 10) / max(_dcg(ideal, 10), 1e-12)
        top10 = labels[:10]
        per_query.append(
            {
                "query_id": query_id,
                "category": str(candidates[0][0]["category"]),
                "hit_at_10": float(any(top10)),
                "reciprocal_rank": (
                    0.0 if first_positive is None else 1.0 / first_positive
                ),
                "ndcg_at_10": ndcg,
                "recall_at_10": sum(top10) / positive_count,
            }
        )

    def aggregate(values: list[dict[str, Any]]) -> dict[str, float]:
        if not values:
            return {
                "hit_at_10": 0.0,
                "reciprocal_rank": 0.0,
                "ndcg_at_10": 0.0,
                "recall_at_10": 0.0,
                "composite": 0.0,
            }
        result = {
            key: sum(float(row[key]) for row in values) / len(values)
            for key in (
                "hit_at_10",
                "reciprocal_rank",
                "ndcg_at_10",
                "recall_at_10",
            )
        }
        result["composite"] = (
            result["hit_at_10"]
            + result["reciprocal_rank"]
            + result["ndcg_at_10"]
        ) / 3.0
        return result

    return {
        "query_count": len(per_query),
        "overall": aggregate(per_query),
        "by_category": {
            category: aggregate(
                [row for row in per_query if row["category"] == category]
            )
            for category in CATEGORY_ORDER
            if any(row["category"] == category for row in per_query)
        },
        "per_query": per_query,
    }


def macro_f1(
    true_values: Sequence[int],
    predicted_values: Sequence[int],
) -> float:
    labels = sorted(set(true_values) | set(predicted_values))
    scores = []
    for label in labels:
        true_positive = sum(
            int(true == label and predicted == label)
            for true, predicted in zip(
                true_values, predicted_values, strict=True
            )
        )
        false_positive = sum(
            int(true != label and predicted == label)
            for true, predicted in zip(
                true_values, predicted_values, strict=True
            )
        )
        false_negative = sum(
            int(true == label and predicted != label)
            for true, predicted in zip(
                true_values, predicted_values, strict=True
            )
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        scores.append(
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    return sum(scores) / max(1, len(scores))


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def _losses(
    *,
    outputs: dict[str, Any],
    pairs: PairTensorBundle,
    relation_logits: Any,
    relation_targets: Any,
    teacher_scores: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    import torch

    functional = torch.nn.functional
    relevance = query_balanced_relevance_loss(
        outputs["scores"],
        pairs.query_rows,
        pairs.labels,
        pairs.hard_negative,
    )
    contrastive = (
        float(
            TRAINING_CONFIG["contrastive_components"][
                "multi_positive_listwise"
            ]
        )
        * multi_positive_contrastive_loss(
            outputs["scores"],
            pairs.query_rows,
            pairs.labels,
        )
        + float(
            TRAINING_CONFIG["contrastive_components"]["lambda_ndcg_at_10"]
        )
        * lambda_ndcg_loss(
            outputs["scores"],
            pairs.query_rows,
            pairs.labels,
        )
    )
    edge_relation = functional.cross_entropy(
        relation_logits,
        relation_targets,
    )
    path_mask = (pairs.path_targets.sum(dim=1) > 0) & (pairs.labels > 0.5)
    path_loss = (
        functional.binary_cross_entropy_with_logits(
            outputs["path_logits"][path_mask],
            pairs.path_targets[path_mask],
        )
        if path_mask.any()
        else outputs["path_logits"].sum() * 0.0
    )
    relation_path = 0.5 * edge_relation + 0.5 * path_loss
    category_scores = outputs["category_logits"].gather(
        1, pairs.categories.unsqueeze(1)
    ).squeeze(1)
    category = functional.binary_cross_entropy_with_logits(
        category_scores,
        pairs.labels,
    )
    parameter_mask = pairs.parameter_eligible
    parameter = (
        functional.binary_cross_entropy_with_logits(
            outputs["parameter_scores"][parameter_mask],
            pairs.labels[parameter_mask],
        )
        if parameter_mask.any()
        else outputs["parameter_scores"].sum() * 0.0
    )
    citation = functional.binary_cross_entropy_with_logits(
        outputs["citation_scores"],
        pairs.citation_labels,
    )
    teacher_anchor = (
        functional.smooth_l1_loss(outputs["scores"], teacher_scores)
        if teacher_scores is not None
        else outputs["scores"].sum() * 0.0
    )
    values = {
        "relevance": relevance,
        "contrastive": contrastive,
        "relation_path": relation_path,
        "category": category,
        "parameter": parameter,
        "citation": citation,
        "teacher_anchor": teacher_anchor,
    }
    total = sum(LOSS_WEIGHTS[key] * value for key, value in values.items())
    return total, {
        key: float(value.detach().cpu()) for key, value in values.items()
    }


def _relation_indices(
    graph: GraphTensorBundle,
    relation_rows: list[dict[str, Any]],
) -> dict[str, list[int]]:
    edge_index = {edge_id: index for index, edge_id in enumerate(graph.edge_ids)}
    output: dict[str, list[int]] = defaultdict(list)
    for row in relation_rows:
        output[str(row["split"])].append(edge_index[str(row["edge_id"])])
    return output


def _evaluate_model(
    *,
    model: Any,
    graph: GraphTensorBundle,
    context_graph: GraphTensorBundle,
    query_embeddings: Any,
    pairs: PairTensorBundle,
    relation_edge_index: Any,
    relation_targets: Any,
) -> dict[str, Any]:
    import torch

    model.eval()
    with torch.inference_mode():
        node_embeddings = model.encode_nodes(graph)
        outputs = _pair_forward(
            model,
            node_embeddings,
            query_embeddings,
            pairs,
        )
        metrics = ranking_metrics(
            pairs.rows,
            outputs["scores"].detach().cpu().tolist(),
        )
        context_embeddings = model.encode_nodes(context_graph)
        relation_logits = model.relation_logits(
            context_embeddings,
            relation_edge_index,
        )
        predicted = relation_logits.argmax(dim=1).detach().cpu().tolist()
        truth = relation_targets.detach().cpu().tolist()
        metrics["relation_macro_f1"] = macro_f1(truth, predicted)
        metrics["relation_accuracy"] = sum(
            int(left == right)
            for left, right in zip(truth, predicted, strict=True)
        ) / max(1, len(truth))
    return metrics


def _predict_pair_scores(
    *,
    model: Any,
    graph: GraphTensorBundle,
    query_embeddings: Any,
    pairs: PairTensorBundle,
) -> Any:
    import torch

    model.eval()
    with torch.inference_mode():
        node_embeddings = model.encode_nodes(graph)
        scores = _pair_forward(
            model,
            node_embeddings,
            query_embeddings,
            pairs,
        )["scores"].detach()
    return scores.clone()


def _calibrate_category_alphas(
    *,
    rows: list[dict[str, Any]],
    teacher_scores: Sequence[float],
    graph_scores: Sequence[float],
) -> dict[str, float]:
    alpha_grid = [
        float(value)
        for value in TRAINING_CONFIG["validation_alpha_grid"]
    ]
    result: dict[str, float] = {}
    for category in CATEGORY_ORDER:
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row["category"]) == category
        ]
        if not indices:
            result[category] = 0.0
            continue
        category_rows = [rows[index] for index in indices]
        baseline = ranking_metrics(
            category_rows,
            [teacher_scores[index] for index in indices],
        )["overall"]
        best_alpha = 0.0
        best_composite = float(baseline["composite"])
        best_objective = 0.5 * best_composite + 0.5 * float(
            baseline["ndcg_at_10"]
        )
        for alpha in alpha_grid:
            scores = [
                teacher_scores[index]
                + alpha * (graph_scores[index] - teacher_scores[index])
                for index in indices
            ]
            metrics = ranking_metrics(category_rows, scores)["overall"]
            if (
                float(metrics["hit_at_10"])
                + 1e-12
                < float(baseline["hit_at_10"])
                or float(metrics["ndcg_at_10"])
                + 1e-12
                < float(baseline["ndcg_at_10"])
            ):
                continue
            composite = float(metrics["composite"])
            objective = 0.5 * composite + 0.5 * float(
                metrics["ndcg_at_10"]
            )
            if (
                objective > best_objective + 1e-12
                or (
                    abs(objective - best_objective) <= 1e-12
                    and composite > best_composite + 1e-12
                )
            ):
                best_alpha = alpha
                best_composite = composite
                best_objective = objective
        result[category] = best_alpha
    return result


def _blend_category_scores(
    *,
    rows: list[dict[str, Any]],
    teacher_scores: Sequence[float],
    graph_scores: Sequence[float],
    alphas: dict[str, float],
) -> list[float]:
    return [
        float(teacher)
        + float(alphas.get(str(row["category"]), 0.0))
        * (float(graph) - float(teacher))
        for row, teacher, graph in zip(
            rows,
            teacher_scores,
            graph_scores,
            strict=True,
        )
    ]


def _checkpoint_hash(checkpoint: dict[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                key: value
                for key, value in checkpoint.items()
                if key != "state_dict"
            }
        )
    )


def train_one(
    *,
    architecture: str,
    seed: int,
    graph: GraphTensorBundle,
    context_graph: GraphTensorBundle,
    query_embeddings: Any,
    pairs_by_split: dict[str, PairTensorBundle],
    relation_splits: dict[str, tuple[Any, Any]],
    output_dir: Path,
    device: Any,
    input_hash: str,
    teacher_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    import torch

    _set_seed(seed)
    model = build_ranker(
        architecture=architecture,
        graph=graph,
        category_count=len(CATEGORY_ORDER),
    ).to(device)
    teacher = None
    teacher_checkpoint = None
    if architecture != "no_gnn":
        if teacher_checkpoint_path is None or not teacher_checkpoint_path.exists():
            raise RuntimeError(
                f"Missing NoGNN teacher checkpoint for {architecture} seed {seed}."
            )
        teacher_checkpoint = torch.load(
            teacher_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if teacher_checkpoint["input_hash"] != input_hash:
            raise RuntimeError("NoGNN teacher input hash mismatch.")
        teacher = build_ranker(
            architecture="no_gnn",
            graph=graph,
            category_count=len(CATEGORY_ORDER),
        )
        teacher.load_state_dict(teacher_checkpoint["state_dict"])
        teacher.to(device)
        teacher.eval()
        current_state = model.state_dict()
        shared_state = {
            key: value
            for key, value in teacher_checkpoint["state_dict"].items()
            if key in current_state
            and current_state[key].shape == value.shape
        }
        model.load_state_dict(shared_state, strict=False)

    base_learning_rate = float(TRAINING_CONFIG["learning_rate"])
    if architecture == "no_gnn":
        parameter_groups = [{"params": list(model.parameters()), "lr": base_learning_rate}]
    else:
        graph_parameters = []
        shared_parameters = []
        for name, parameter in model.named_parameters():
            if (
                name.startswith("layers.")
                or name.startswith("norms.")
                or name.startswith("edge_")
                or name == "layer_gates"
            ):
                graph_parameters.append(parameter)
            else:
                shared_parameters.append(parameter)
        parameter_groups = [
            {"params": graph_parameters, "lr": base_learning_rate},
            {
                "params": shared_parameters,
                "lr": base_learning_rate
                * float(
                    TRAINING_CONFIG[
                        "graph_shared_parameter_learning_rate_scale"
                    ]
                ),
            },
        ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(TRAINING_CONFIG["weight_decay"]),
    )
    train_pairs = pairs_by_split["train"]
    validation_pairs = pairs_by_split["validation"]
    relation_train_index, relation_train_targets = relation_splits["train"]
    relation_validation_index, relation_validation_targets = relation_splits[
        "validation"
    ]
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = output_dir / architecture / f"seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    teacher_train_scores = None
    teacher_validation_scores = None
    if teacher is not None:
        teacher_train_scores = _predict_pair_scores(
            model=teacher,
            graph=graph,
            query_embeddings=query_embeddings,
            pairs=train_pairs,
        )
        teacher_validation_scores = (
            _predict_pair_scores(
                model=teacher,
                graph=graph,
                query_embeddings=query_embeddings,
                pairs=validation_pairs,
            )
            .detach()
            .cpu()
            .tolist()
        )

    for epoch in range(1, int(TRAINING_CONFIG["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        node_embeddings = model.encode_nodes(graph)
        pair_outputs = _pair_forward(
            model,
            node_embeddings,
            query_embeddings,
            train_pairs,
        )
        context_embeddings = model.encode_nodes(context_graph)
        relation_logits = model.relation_logits(
            context_embeddings,
            relation_train_index,
        )
        total_loss, component_losses = _losses(
            outputs=pair_outputs,
            pairs=train_pairs,
            relation_logits=relation_logits,
            relation_targets=relation_train_targets,
            teacher_scores=teacher_train_scores,
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(TRAINING_CONFIG["gradient_clip"]),
        )
        optimizer.step()

        raw_validation = _evaluate_model(
            model=model,
            graph=graph,
            context_graph=context_graph,
            query_embeddings=query_embeddings,
            pairs=validation_pairs,
            relation_edge_index=relation_validation_index,
            relation_targets=relation_validation_targets,
        )
        validation_alphas = {
            category: 1.0 for category in CATEGORY_ORDER
        }
        validation = raw_validation
        if teacher is not None and teacher_validation_scores is not None:
            graph_validation_scores = (
                _predict_pair_scores(
                    model=model,
                    graph=graph,
                    query_embeddings=query_embeddings,
                    pairs=validation_pairs,
                )
                .detach()
                .cpu()
                .tolist()
            )
            validation_alphas = _calibrate_category_alphas(
                rows=validation_pairs.rows,
                teacher_scores=teacher_validation_scores,
                graph_scores=graph_validation_scores,
            )
            validation = ranking_metrics(
                validation_pairs.rows,
                _blend_category_scores(
                    rows=validation_pairs.rows,
                    teacher_scores=teacher_validation_scores,
                    graph_scores=graph_validation_scores,
                    alphas=validation_alphas,
                ),
            )
            validation["relation_macro_f1"] = raw_validation[
                "relation_macro_f1"
            ]
            validation["relation_accuracy"] = raw_validation[
                "relation_accuracy"
            ]
        validation_score = float(validation["overall"]["composite"])
        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total_loss.detach().cpu()),
                "component_losses": component_losses,
                "validation_composite": validation_score,
                "validation_ndcg_at_10": float(
                    validation["overall"]["ndcg_at_10"]
                ),
                "validation_relation_macro_f1": float(
                    validation["relation_macro_f1"]
                ),
                "validation_alphas": validation_alphas,
            }
        )
        if validation_score > best_score + 1e-8:
            best_score = validation_score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "training_version": GNN_TRAINING_VERSION,
                    "architecture": architecture,
                    "seed": seed,
                    "epoch": epoch,
                    "best_validation_composite": validation_score,
                    "input_hash": input_hash,
                    "training_config": TRAINING_CONFIG,
                    "loss_weights": LOSS_WEIGHTS,
                    "validation_alphas": validation_alphas,
                    "teacher_checkpoint_path": (
                        None
                        if teacher_checkpoint_path is None
                        else str(teacher_checkpoint_path)
                    ),
                    "teacher_checkpoint_sha256": (
                        None
                        if teacher_checkpoint_path is None
                        else sha256_file(teacher_checkpoint_path)
                    ),
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= int(TRAINING_CONFIG["patience"]):
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["input_hash"] != input_hash:
        raise RuntimeError("Checkpoint input hash mismatch.")
    model.load_state_dict(checkpoint["state_dict"])
    # GPU scatter kernels used by heterogeneous message passing can introduce
    # tiny ordering differences even when cuBLAS is deterministic. Final model
    # selection and checkpoint reproduction therefore run on CPU from the
    # frozen weights; training remains on the requested accelerator.
    evaluation_device = torch.device("cpu")
    model.to(evaluation_device)
    evaluation_graph = graph.to(evaluation_device)
    evaluation_context_graph = context_graph.to(evaluation_device)
    evaluation_queries = query_embeddings.to(evaluation_device)
    evaluation_pairs = {
        name: value.to(evaluation_device)
        for name, value in pairs_by_split.items()
    }
    evaluation_relations = {
        name: (
            edge_index.to(evaluation_device),
            targets.to(evaluation_device),
        )
        for name, (edge_index, targets) in relation_splits.items()
    }
    raw_validation_metrics = _evaluate_model(
        model=model,
        graph=evaluation_graph,
        context_graph=evaluation_context_graph,
        query_embeddings=evaluation_queries,
        pairs=evaluation_pairs["validation"],
        relation_edge_index=evaluation_relations["validation"][0],
        relation_targets=evaluation_relations["validation"][1],
    )
    raw_test_metrics = _evaluate_model(
        model=model,
        graph=evaluation_graph,
        context_graph=evaluation_context_graph,
        query_embeddings=evaluation_queries,
        pairs=evaluation_pairs["test"],
        relation_edge_index=evaluation_relations["test"][0],
        relation_targets=evaluation_relations["test"][1],
    )
    validation_metrics = raw_validation_metrics
    test_metrics = raw_test_metrics
    final_alphas = {
        category: 1.0 for category in CATEGORY_ORDER
    }
    evaluation_teacher = None
    if teacher_checkpoint is not None:
        evaluation_teacher = build_ranker(
            architecture="no_gnn",
            graph=graph,
            category_count=len(CATEGORY_ORDER),
        )
        evaluation_teacher.load_state_dict(teacher_checkpoint["state_dict"])
        evaluation_teacher.to(evaluation_device)
        teacher_validation = (
            _predict_pair_scores(
                model=evaluation_teacher,
                graph=evaluation_graph,
                query_embeddings=evaluation_queries,
                pairs=evaluation_pairs["validation"],
            )
            .cpu()
            .tolist()
        )
        teacher_test = (
            _predict_pair_scores(
                model=evaluation_teacher,
                graph=evaluation_graph,
                query_embeddings=evaluation_queries,
                pairs=evaluation_pairs["test"],
            )
            .cpu()
            .tolist()
        )
        graph_validation = (
            _predict_pair_scores(
                model=model,
                graph=evaluation_graph,
                query_embeddings=evaluation_queries,
                pairs=evaluation_pairs["validation"],
            )
            .cpu()
            .tolist()
        )
        graph_test = (
            _predict_pair_scores(
                model=model,
                graph=evaluation_graph,
                query_embeddings=evaluation_queries,
                pairs=evaluation_pairs["test"],
            )
            .cpu()
            .tolist()
        )
        final_alphas = {
            str(key): float(value)
            for key, value in checkpoint["validation_alphas"].items()
        }
        validation_metrics = ranking_metrics(
            evaluation_pairs["validation"].rows,
            _blend_category_scores(
                rows=evaluation_pairs["validation"].rows,
                teacher_scores=teacher_validation,
                graph_scores=graph_validation,
                alphas=final_alphas,
            ),
        )
        test_metrics = ranking_metrics(
            evaluation_pairs["test"].rows,
            _blend_category_scores(
                rows=evaluation_pairs["test"].rows,
                teacher_scores=teacher_test,
                graph_scores=graph_test,
                alphas=final_alphas,
            ),
        )
        for metrics, raw_metrics in (
            (validation_metrics, raw_validation_metrics),
            (test_metrics, raw_test_metrics),
        ):
            metrics["relation_macro_f1"] = raw_metrics["relation_macro_f1"]
            metrics["relation_accuracy"] = raw_metrics["relation_accuracy"]
    first_test = canonical_json(test_metrics)

    reloaded = build_ranker(
        architecture=architecture,
        graph=graph,
        category_count=len(CATEGORY_ORDER),
    )
    reloaded.load_state_dict(checkpoint["state_dict"])
    reloaded.to(evaluation_device)
    reproduced_raw_metrics = _evaluate_model(
        model=reloaded,
        graph=evaluation_graph,
        context_graph=evaluation_context_graph,
        query_embeddings=evaluation_queries,
        pairs=evaluation_pairs["test"],
        relation_edge_index=evaluation_relations["test"][0],
        relation_targets=evaluation_relations["test"][1],
    )
    reproduced_metrics = reproduced_raw_metrics
    if evaluation_teacher is not None:
        reproduced_graph_test = (
            _predict_pair_scores(
                model=reloaded,
                graph=evaluation_graph,
                query_embeddings=evaluation_queries,
                pairs=evaluation_pairs["test"],
            )
            .cpu()
            .tolist()
        )
        reproduced_teacher_test = (
            _predict_pair_scores(
                model=evaluation_teacher,
                graph=evaluation_graph,
                query_embeddings=evaluation_queries,
                pairs=evaluation_pairs["test"],
            )
            .cpu()
            .tolist()
        )
        reproduced_metrics = ranking_metrics(
            evaluation_pairs["test"].rows,
            _blend_category_scores(
                rows=evaluation_pairs["test"].rows,
                teacher_scores=reproduced_teacher_test,
                graph_scores=reproduced_graph_test,
                alphas=final_alphas,
            ),
        )
        reproduced_metrics["relation_macro_f1"] = reproduced_raw_metrics[
            "relation_macro_f1"
        ]
        reproduced_metrics["relation_accuracy"] = reproduced_raw_metrics[
            "relation_accuracy"
        ]
    reproduction_delta = abs(
        float(test_metrics["overall"]["composite"])
        - float(reproduced_metrics["overall"]["composite"])
    )
    if first_test != canonical_json(reproduced_metrics):
        if reproduction_delta > float(
            ACCEPTANCE_THRESHOLDS["checkpoint_reproduction_tolerance"]
        ):
            raise RuntimeError(
                "Reloaded checkpoint did not reproduce test metrics: "
                f"composite_delta={reproduction_delta}"
            )

    atomic_json(
        output_dir / architecture / f"seed_{seed}_history.json",
        history,
    )
    report = {
        "architecture": architecture,
        "seed": seed,
        "best_epoch": best_epoch,
        "epoch_count": len(history),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata_sha256": _checkpoint_hash(checkpoint),
        "checkpoint_reproduction_delta": reproduction_delta,
        "final_evaluation_device": str(evaluation_device),
        "validation_alphas": final_alphas,
        "raw_validation": raw_validation_metrics,
        "raw_test": raw_test_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    atomic_json(
        output_dir / architecture / f"seed_{seed}_metrics.json",
        report,
    )
    del (
        model,
        reloaded,
        teacher,
        evaluation_teacher,
        optimizer,
        evaluation_graph,
        evaluation_context_graph,
        evaluation_queries,
        evaluation_pairs,
        evaluation_relations,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(
        sum((value - mean) ** 2 for value in values) / max(1, len(values))
    )


def aggregate_architecture(
    architecture: str,
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    validation_composite = [
        float(row["validation"]["overall"]["composite"]) for row in reports
    ]
    test_metric_names = (
        "hit_at_10",
        "reciprocal_rank",
        "ndcg_at_10",
        "recall_at_10",
        "composite",
    )
    test_means = {
        key: _mean(
            [float(row["test"]["overall"][key]) for row in reports]
        )
        for key in test_metric_names
    }
    test_standard_deviations = {
        key: _std(
            [float(row["test"]["overall"][key]) for row in reports]
        )
        for key in test_metric_names
    }
    category_hit_at_10 = {
        category: _mean(
            [
                float(
                    row["test"]["by_category"]
                    .get(category, {})
                    .get("hit_at_10", 0.0)
                )
                for row in reports
            ]
        )
        for category in CATEGORY_ORDER
    }
    return {
        "architecture": architecture,
        "seed_count": len(reports),
        "validation_composite_mean": _mean(validation_composite),
        "validation_composite_std": _std(validation_composite),
        "test_mean": test_means,
        "test_std": test_standard_deviations,
        "category_hit_at_10_mean": category_hit_at_10,
        "relation_macro_f1_mean": _mean(
            [float(row["test"]["relation_macro_f1"]) for row in reports]
        ),
        "relation_accuracy_mean": _mean(
            [float(row["test"]["relation_accuracy"]) for row in reports]
        ),
        "all_checkpoints_reproduced": all(
            float(row["checkpoint_reproduction_delta"])
            <= float(
                ACCEPTANCE_THRESHOLDS["checkpoint_reproduction_tolerance"]
            )
            for row in reports
        ),
        "runs": reports,
    }


def _model_input_hash(
    benchmark_manifest: dict[str, Any],
    acceptance_path: Path,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "training_version": GNN_TRAINING_VERSION,
                "benchmark_output_hashes": benchmark_manifest["output_hashes"],
                "benchmark_input_hashes": benchmark_manifest["input_hashes"],
                "acceptance_sha256": sha256_file(acceptance_path),
            }
        )
    )


def train_benchmark(
    *,
    graph_dir: Path,
    embedding_index_dir: Path,
    benchmark_dir: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    benchmark_manifest_path = benchmark_dir / "gnn_benchmark_manifest.json"
    benchmark_manifest = json.loads(
        benchmark_manifest_path.read_text(encoding="utf-8")
    )
    acceptance_path = benchmark_dir / "preregistered_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if benchmark_manifest.get("accepted") is not True:
        raise RuntimeError("GNN benchmark is not accepted.")
    if acceptance.get("locked_before_training") is not True:
        raise RuntimeError("GNN acceptance was not locked before training.")
    if acceptance.get("architectures") != list(ARCHITECTURES):
        raise RuntimeError("GNN architecture comparison was changed.")
    for name, expected in benchmark_manifest["output_hashes"].items():
        filenames = {
            "query_records": "query_records.jsonl",
            "query_embeddings": "query_embeddings.npy",
            "semantic_split": "semantic_split.jsonl",
            "query_subgraph_pairs": "query_subgraph_pairs.jsonl",
            "relation_split": "relation_split.jsonl",
            "subgraph_index": "subgraph_index.jsonl",
            "acceptance": "preregistered_acceptance.json",
        }
        if sha256_file(benchmark_dir / filenames[name]) != expected:
            raise RuntimeError(f"GNN benchmark checksum mismatch: {name}")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph_tensors(
        graph_dir=graph_dir,
        embedding_index_dir=embedding_index_dir,
    )
    node_index = {
        node_id: index for index, node_id in enumerate(graph.node_ids)
    }
    query_records = load_jsonl(benchmark_dir / "query_records.jsonl")
    query_row_by_id = {
        str(row["query_id"]): int(row["query_row"])
        for row in query_records
    }
    query_embeddings = torch.from_numpy(
        np.load(benchmark_dir / "query_embeddings.npy")
    ).to(device)
    pair_rows = load_jsonl(benchmark_dir / "query_subgraph_pairs.jsonl")
    subgraph_rows = load_jsonl(benchmark_dir / "subgraph_index.jsonl")
    subgraph_by_id = {
        str(row["subgraph_id"]): row for row in subgraph_rows
    }
    train_rows = [row for row in pair_rows if row["split"] == "train"]
    pairs_by_split = {
        "train": build_pair_tensors(
            rows=train_rows,
            query_row_by_id=query_row_by_id,
            subgraph_by_id=subgraph_by_id,
            node_index=node_index,
            relation_vocab=graph.relation_vocab,
        ),
        "validation": build_pair_tensors(
            rows=[row for row in pair_rows if row["split"] == "validation"],
            query_row_by_id=query_row_by_id,
            subgraph_by_id=subgraph_by_id,
            node_index=node_index,
            relation_vocab=graph.relation_vocab,
        ),
        "test": build_pair_tensors(
            rows=[row for row in pair_rows if row["split"] == "test"],
            query_row_by_id=query_row_by_id,
            subgraph_by_id=subgraph_by_id,
            node_index=node_index,
            relation_vocab=graph.relation_vocab,
        ),
    }
    graph = graph.to(device)
    pairs_by_split = {
        key: value.to(device) for key, value in pairs_by_split.items()
    }

    relation_rows = load_jsonl(benchmark_dir / "relation_split.jsonl")
    relation_indices = _relation_indices(graph, relation_rows)
    context_graph = _subset_graph(
        graph,
        relation_indices["context"],
    ).to(device)
    relation_splits: dict[str, tuple[Any, Any]] = {}
    for split in ("train", "validation", "test"):
        indices = relation_indices[split]
        relation_splits[split] = (
            graph.forward_edge_index[
                :,
                torch.tensor(indices, dtype=torch.long, device=device),
            ],
            torch.tensor(
                [graph.relation_ids[index] for index in indices],
                dtype=torch.long,
                device=device,
            ),
        )

    input_hash = _model_input_hash(benchmark_manifest, acceptance_path)
    reports_by_architecture: dict[str, list[dict[str, Any]]] = {}
    for architecture in ARCHITECTURES:
        reports_by_architecture[architecture] = []
        for seed in SEEDS:
            report = train_one(
                architecture=architecture,
                seed=seed,
                graph=graph,
                context_graph=context_graph,
                query_embeddings=query_embeddings,
                pairs_by_split=pairs_by_split,
                relation_splits=relation_splits,
                output_dir=output_dir,
                device=device,
                input_hash=input_hash,
                teacher_checkpoint_path=(
                    None
                    if architecture == "no_gnn"
                    else output_dir / "no_gnn" / f"seed_{seed}.pt"
                ),
            )
            reports_by_architecture[architecture].append(report)
            print(
                canonical_json(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "validation_composite": report["validation"]["overall"][
                            "composite"
                        ],
                        "test_composite": report["test"]["overall"]["composite"],
                    }
                ),
                flush=True,
            )

    aggregated = {
        architecture: aggregate_architecture(
            architecture,
            reports_by_architecture[architecture],
        )
        for architecture in ARCHITECTURES
    }
    selected_architecture = max(
        GNN_ARCHITECTURES,
        key=lambda name: float(
            aggregated[name]["validation_composite_mean"]
        ),
    )
    selected = aggregated[selected_architecture]
    baseline = aggregated["no_gnn"]
    category_regressions = {
        category: (
            float(baseline["category_hit_at_10_mean"][category])
            - float(selected["category_hit_at_10_mean"][category])
        )
        for category in CATEGORY_ORDER
    }
    checks = {
        "all_checkpoints_reproduced": all(
            row["all_checkpoints_reproduced"] for row in aggregated.values()
        ),
        "composite_improvement": (
            float(selected["test_mean"]["composite"])
            - float(baseline["test_mean"]["composite"])
            >= float(
                ACCEPTANCE_THRESHOLDS[
                    "minimum_composite_improvement_over_no_gnn"
                ]
            )
        ),
        "ndcg_improvement": (
            float(selected["test_mean"]["ndcg_at_10"])
            - float(baseline["test_mean"]["ndcg_at_10"])
            >= float(
                ACCEPTANCE_THRESHOLDS[
                    "minimum_ndcg_at_10_improvement_over_no_gnn"
                ]
            )
        ),
        "category_regression_within_limit": max(
            category_regressions.values()
        )
        <= float(
            ACCEPTANCE_THRESHOLDS["maximum_category_hit_at_10_regression"]
        ),
        "composite_stability": float(selected["test_std"]["composite"])
        <= float(
            ACCEPTANCE_THRESHOLDS[
                "maximum_composite_standard_deviation"
            ]
        ),
        "relation_macro_f1_not_worse": (
            float(selected["relation_macro_f1_mean"])
            - float(baseline["relation_macro_f1_mean"])
            >= float(
                ACCEPTANCE_THRESHOLDS[
                    "minimum_relation_macro_f1_delta_over_no_gnn"
                ]
            )
        ),
        "quarantined_evidence_absent": int(
            benchmark_manifest["quarantined_evidence_count"]
        )
        <= int(
            ACCEPTANCE_THRESHOLDS["maximum_quarantined_evidence_count"]
        ),
    }
    comparison = {
        "accepted": all(checks.values()),
        "training_version": GNN_TRAINING_VERSION,
        "benchmark_level": "silver",
        "benchmark_version": benchmark_manifest["benchmark_version"],
        "generated_at": utc_now(),
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "input_hash": input_hash,
        "acceptance_sha256": sha256_file(acceptance_path),
        "selected_architecture": selected_architecture,
        "selection_basis": "highest mean validation composite across locked seeds",
        "checks": checks,
        "thresholds": ACCEPTANCE_THRESHOLDS,
        "category_hit_at_10_regressions": category_regressions,
        "architectures": aggregated,
        "limitations": [
            "Results are measured on a provenance-backed silver benchmark.",
            "Independent expert gold recommendation labels remain mandatory at Gate 6.",
        ],
    }
    atomic_json(output_dir / "gnn_comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    if not comparison["accepted"]:
        raise RuntimeError("Gate 4 GNN acceptance thresholds were not met.")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and train the locked PBPK GNN benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--graph-dir", type=Path, required=True)
    prepare.add_argument("--subgraph-dir", type=Path, required=True)
    prepare.add_argument("--retrieval-benchmark-dir", type=Path, required=True)
    prepare.add_argument("--candidate-pool", type=Path, required=True)
    prepare.add_argument("--reranker-scores", type=Path, required=True)
    prepare.add_argument("--embedding-index-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--device", default="cuda")

    train = subparsers.add_parser("train")
    train.add_argument("--graph-dir", type=Path, required=True)
    train.add_argument("--embedding-index-dir", type=Path, required=True)
    train.add_argument("--benchmark-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = prepare_benchmark(
            graph_dir=args.graph_dir,
            subgraph_dir=args.subgraph_dir,
            retrieval_benchmark_dir=args.retrieval_benchmark_dir,
            candidate_pool_path=args.candidate_pool,
            reranker_scores_path=args.reranker_scores,
            embedding_index_dir=args.embedding_index_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            device=args.device,
        )
    else:
        result = train_benchmark(
            graph_dir=args.graph_dir,
            embedding_index_dir=args.embedding_index_dir,
            benchmark_dir=args.benchmark_dir,
            output_dir=args.output_dir,
            device_name=args.device,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
