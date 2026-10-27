from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.paths import resolve_recorded_path
from pbpk_kb.recommendation.embedding_benchmark import load_jsonl, sha256_file
from pbpk_kb.recommendation.gnn_benchmark import (
    CATEGORY_ORDER,
    GNN_TRAINING_VERSION,
    _blend_category_scores,
    _pair_forward,
    atomic_json,
    atomic_jsonl,
    atomic_npy,
    build_graph_tensors,
    build_pair_tensors,
    build_ranker,
    canonical_json,
    ranking_metrics,
    sha256_text,
    utc_now,
)


GNN_INDEX_VERSION = "pbpk-gnn-index-1.0.0"
GNN_VECTOR_DIMENSION = 256


def _load_subgraphs(subgraph_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for hop in range(3):
        for row in load_jsonl(subgraph_dir / f"subgraphs_{hop}hop.jsonl"):
            rows[str(row["subgraph_id"])] = row
    return rows


def _selected_run(comparison: dict[str, Any]) -> dict[str, Any]:
    architecture = str(comparison["selected_architecture"])
    runs = list(comparison["architectures"][architecture]["runs"])
    return max(
        runs,
        key=lambda row: (
            float(row["validation"]["overall"]["composite"]),
            -int(row["seed"]),
        ),
    )


def _normalise(value: Any) -> Any:
    import torch

    return torch.nn.functional.normalize(value, p=2, dim=-1)


def _subgraph_vector(
    *,
    node_embeddings: Any,
    member_indices: Any,
    center_index: int,
) -> tuple[Any, Any]:
    import torch

    members = node_embeddings[member_indices]
    anchor = node_embeddings[center_index]
    logits = (members * anchor.unsqueeze(0)).sum(dim=1) / math.sqrt(
        int(node_embeddings.shape[1])
    )
    attention = torch.softmax(logits, dim=0)
    pooled = (members * attention.unsqueeze(1)).sum(dim=0)
    vector = _normalise(anchor + pooled)
    return vector, attention


def _top_nodes(
    *,
    node_ids: list[str],
    member_indices: list[int],
    attention: list[float],
    node_by_id: dict[str, dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    rows = []
    for graph_index, weight in zip(member_indices, attention, strict=True):
        node_id = node_ids[graph_index]
        node = node_by_id[node_id]
        rows.append(
            {
                "node_id": node_id,
                "node_type": str(node["node_type"]),
                "preferred_name": str(node.get("preferred_name") or node_id),
                "attention": float(weight),
            }
        )
    rows.sort(key=lambda row: (-row["attention"], row["node_id"]))
    return rows[:limit]


def _edge_and_path_explanations(
    *,
    subgraph: dict[str, Any],
    node_attention: dict[str, float],
    edge_by_id: dict[str, dict[str, Any]],
    path_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edge_rows = []
    edge_scores: dict[str, float] = {}
    edge_ids = [
        str(edge["edge_id"])
        for edge in subgraph.get("edges") or ()
    ]
    for edge_id in edge_ids:
        edge = edge_by_id[str(edge_id)]
        provenance = dict(edge.get("provenance") or {})
        score = (
            node_attention.get(str(edge["subject_id"]), 0.0)
            + node_attention.get(str(edge["object_id"]), 0.0)
        ) / 2.0
        score *= float(provenance.get("confidence") or 0.0)
        edge_scores[str(edge_id)] = score
        edge_rows.append(
            {
                "edge_id": str(edge_id),
                "subject_id": str(edge["subject_id"]),
                "predicate": str(edge["predicate"]),
                "object_id": str(edge["object_id"]),
                "contribution": score,
                "source_id": provenance.get("source_id"),
                "source_location": provenance.get("source_location"),
                "supporting_text": provenance.get("supporting_text"),
            }
        )
    edge_rows.sort(key=lambda row: (-row["contribution"], row["edge_id"]))

    path_rows = []
    for path_id in subgraph.get("path_ids") or ():
        path = path_by_id.get(str(path_id))
        if path is None:
            continue
        values = [
            edge_scores.get(str(edge_id), 0.0)
            for edge_id in path.get("edge_ids") or ()
        ]
        path_rows.append(
            {
                "path_id": str(path_id),
                "node_ids": [str(value) for value in path.get("node_ids") or ()],
                "edge_ids": [str(value) for value in path.get("edge_ids") or ()],
                "hop": int(path.get("hop") or 0),
                "contribution": sum(values) / max(1, len(values)),
            }
        )
    path_rows.sort(key=lambda row: (-row["contribution"], row["path_id"]))
    return edge_rows[:10], path_rows[:5]


def _index_metrics(
    *,
    query_records: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    query_vectors: Any,
    subgraph_vectors: Any,
    subgraph_records: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    positives: dict[str, set[str]] = defaultdict(set)
    for row in pair_rows:
        if int(row["relevance_label"]) == 1:
            positives[str(row["query_id"])].add(str(row["subgraph_id"]))
    by_hop: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(subgraph_records):
        by_hop[int(row["hop"])].append(index)
    per_query = []
    for query in query_records:
        query_id = str(query["query_id"])
        indices = by_hop[int(query["target_hop"])]
        scores = subgraph_vectors[indices] @ query_vectors[int(query["query_row"])]
        order = np.argsort(-scores, kind="stable")
        returned = [
            str(subgraph_records[indices[int(index)]]["subgraph_id"])
            for index in order[:100]
        ]
        expected = positives[query_id]
        per_query.append(
            {
                "query_id": query_id,
                "category": str(query["category"]),
                "positive_count": len(expected),
                "hit_at_100": float(bool(set(returned) & expected)),
                "recall_at_100": len(set(returned) & expected)
                / max(1, len(expected)),
            }
        )
    return {
        "query_count": len(per_query),
        "hit_at_100": sum(row["hit_at_100"] for row in per_query)
        / max(1, len(per_query)),
        "recall_at_100": sum(row["recall_at_100"] for row in per_query)
        / max(1, len(per_query)),
        "by_category": {
            category: {
                "query_count": len(
                    selected := [
                        row
                        for row in per_query
                        if row["category"] == category
                    ]
                ),
                "hit_at_100": sum(row["hit_at_100"] for row in selected)
                / max(1, len(selected)),
                "recall_at_100": sum(
                    row["recall_at_100"] for row in selected
                )
                / max(1, len(selected)),
            }
            for category in CATEGORY_ORDER
        },
        "per_query": per_query,
    }


def build_gnn_index(
    *,
    graph_dir: Path,
    subgraph_dir: Path,
    embedding_index_dir: Path,
    benchmark_dir: Path,
    training_dir: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = training_dir / "gnn_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("accepted") is not True:
        raise RuntimeError("Gate 4 comparison is not accepted.")
    selected = _selected_run(comparison)
    architecture = str(comparison["selected_architecture"])
    checkpoint_path = resolve_recorded_path(
        str(selected["checkpoint_path"]),
        base=Path.cwd(),
    )
    if sha256_file(checkpoint_path) != str(selected["checkpoint_sha256"]):
        raise RuntimeError("Selected checkpoint checksum mismatch.")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["training_version"] != GNN_TRAINING_VERSION:
        raise RuntimeError("GNN training version mismatch.")
    teacher_path = training_dir / "no_gnn" / f"seed_{selected['seed']}.pt"
    teacher_checkpoint = torch.load(
        teacher_path,
        map_location="cpu",
        weights_only=False,
    )

    graph = build_graph_tensors(
        graph_dir=graph_dir,
        embedding_index_dir=embedding_index_dir,
    )
    device = torch.device(device_name)
    model = build_ranker(
        architecture=architecture,
        graph=graph,
        category_count=len(CATEGORY_ORDER),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    teacher = build_ranker(
        architecture="no_gnn",
        graph=graph,
        category_count=len(CATEGORY_ORDER),
    )
    teacher.load_state_dict(teacher_checkpoint["state_dict"])
    teacher.to(device).eval()
    graph = graph.to(device)
    with torch.inference_mode():
        node_embeddings = model.encode_nodes(graph)
        teacher_node_embeddings = teacher.encode_nodes(graph)

    graph_nodes = load_jsonl(graph_dir / "kg_nodes.jsonl")
    graph_edges = load_jsonl(graph_dir / "kg_edges.jsonl")
    node_by_id = {str(row["node_id"]): row for row in graph_nodes}
    edge_by_id = {str(row["edge_id"]): row for row in graph_edges}
    path_by_id = {
        str(row["path_id"]): row
        for row in load_jsonl(subgraph_dir / "traversal_paths.jsonl")
    }
    subgraph_by_id = _load_subgraphs(subgraph_dir)
    benchmark_subgraphs = load_jsonl(benchmark_dir / "subgraph_index.jsonl")
    node_index = {
        node_id: index for index, node_id in enumerate(graph.node_ids)
    }

    vector_rows = []
    vectors = []
    for row_index, benchmark_subgraph in enumerate(benchmark_subgraphs):
        subgraph_id = str(benchmark_subgraph["subgraph_id"])
        subgraph = subgraph_by_id[subgraph_id]
        member_indices = [
            node_index[str(node_id)]
            for node_id in benchmark_subgraph["node_ids"]
        ]
        center_index = node_index[str(benchmark_subgraph["center_id"])]
        with torch.inference_mode():
            vector, attention = _subgraph_vector(
                node_embeddings=node_embeddings,
                member_indices=torch.tensor(
                    member_indices,
                    dtype=torch.long,
                    device=device,
                ),
                center_index=center_index,
            )
        vector_numpy = vector.detach().cpu().numpy().astype(np.float32)
        vectors.append(vector_numpy)
        vector_rows.append(
            {
                "embedding_row": row_index,
                "subgraph_id": subgraph_id,
                "center_id": str(benchmark_subgraph["center_id"]),
                "center_type": str(benchmark_subgraph["center_type"]),
                "hop": int(benchmark_subgraph["hop"]),
                "node_count": int(benchmark_subgraph["node_count"]),
                "edge_count": int(benchmark_subgraph["edge_count"]),
                "event_count": int(benchmark_subgraph["event_count"]),
                "citation_count": int(benchmark_subgraph["citation_count"]),
                "subgraph_content_sha256": str(
                    benchmark_subgraph["content_sha256"]
                ),
                "vector_sha256": hashlib.sha256(
                    vector_numpy.tobytes()
                ).hexdigest(),
                "top_node_attention": _top_nodes(
                    node_ids=graph.node_ids,
                    member_indices=member_indices,
                    attention=attention.detach().cpu().tolist(),
                    node_by_id=node_by_id,
                ),
                "source_ids": sorted(
                    {
                        str(
                            (edge_by_id[str(edge_id)].get("provenance") or {}).get(
                                "source_id"
                            )
                        )
                        for edge_id in benchmark_subgraph["edge_ids"]
                        if (edge_by_id[str(edge_id)].get("provenance") or {}).get(
                            "source_id"
                        )
                    }
                ),
                "quality_statuses": sorted(
                    {
                        str(
                            (edge_by_id[str(edge_id)].get("provenance") or {}).get(
                                "quality_status"
                            )
                        )
                        for edge_id in benchmark_subgraph["edge_ids"]
                        if (edge_by_id[str(edge_id)].get("provenance") or {}).get(
                            "quality_status"
                        )
                    }
                ),
            }
        )
    subgraph_vectors = np.stack(vectors).astype(np.float32)
    if (
        subgraph_vectors.shape
        != (len(benchmark_subgraphs), GNN_VECTOR_DIMENSION)
        or not np.isfinite(subgraph_vectors).all()
        or not np.allclose(
            np.linalg.norm(subgraph_vectors, axis=1),
            1.0,
            atol=1e-5,
        )
    ):
        raise RuntimeError("Invalid GNN subgraph vectors.")

    query_records = load_jsonl(benchmark_dir / "query_records.jsonl")
    pair_rows = load_jsonl(benchmark_dir / "query_subgraph_pairs.jsonl")
    query_embeddings = torch.from_numpy(
        np.load(benchmark_dir / "query_embeddings.npy")
    ).to(device)
    with torch.inference_mode():
        query_vectors = (
            _normalise(model.query_projection(query_embeddings))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    index_metrics = _index_metrics(
        query_records=query_records,
        pair_rows=pair_rows,
        query_vectors=query_vectors,
        subgraph_vectors=subgraph_vectors,
        subgraph_records=vector_rows,
    )

    pair_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        pair_by_query[str(row["query_id"])].append(row)
    query_record_by_id = {
        str(row["query_id"]): row for row in query_records
    }
    subgraph_index_by_id = {
        str(row["subgraph_id"]): row for row in benchmark_subgraphs
    }
    explanations = []
    final_metrics_rows = []
    for query_id, rows in sorted(pair_by_query.items()):
        rows.sort(key=lambda row: int(row["candidate_rank"]))
        pairs = build_pair_tensors(
            rows=rows,
            query_row_by_id={
                str(row["query_id"]): int(row["query_row"])
                for row in query_records
            },
            subgraph_by_id=subgraph_index_by_id,
            node_index=node_index,
            relation_vocab=graph.relation_vocab,
        ).to(device)
        with torch.inference_mode():
            graph_outputs = _pair_forward(
                model,
                node_embeddings,
                query_embeddings,
                pairs,
                return_attention=True,
            )
            teacher_outputs = _pair_forward(
                teacher,
                teacher_node_embeddings,
                query_embeddings,
                pairs,
            )
        graph_scores = graph_outputs["scores"].detach().cpu().tolist()
        teacher_scores = teacher_outputs["scores"].detach().cpu().tolist()
        alpha = float(
            selected["validation_alphas"].get(
                str(query_record_by_id[query_id]["category"]),
                0.0,
            )
        )
        final_scores = _blend_category_scores(
            rows=rows,
            teacher_scores=teacher_scores,
            graph_scores=graph_scores,
            alphas={
                str(query_record_by_id[query_id]["category"]): alpha
            },
        )
        final_metrics_rows.extend(
            {
                **row,
                "_final_score": score,
            }
            for row, score in zip(rows, final_scores, strict=True)
        )
        attention = graph_outputs["attention"].detach().cpu().tolist()
        member_rows = pairs.member_pair_rows.detach().cpu().tolist()
        member_nodes = pairs.member_nodes.detach().cpu().tolist()
        node_attention_by_pair: dict[int, dict[str, float]] = defaultdict(dict)
        for pair_index, graph_node, weight in zip(
            member_rows,
            member_nodes,
            attention,
            strict=True,
        ):
            node_attention_by_pair[int(pair_index)][
                graph.node_ids[int(graph_node)]
            ] = float(weight)
        order = sorted(
            range(len(rows)),
            key=lambda index: (
                -final_scores[index],
                str(rows[index]["subgraph_id"]),
            ),
        )
        ranked = []
        for rank, pair_index in enumerate(order[:10], start=1):
            pair = rows[pair_index]
            subgraph_id = str(pair["subgraph_id"])
            subgraph = subgraph_by_id[subgraph_id]
            node_attention = node_attention_by_pair[pair_index]
            edge_rows, path_rows = _edge_and_path_explanations(
                subgraph=subgraph,
                node_attention=node_attention,
                edge_by_id=edge_by_id,
                path_by_id=path_by_id,
            )
            ranked.append(
                {
                    "rank": rank,
                    "subgraph_id": subgraph_id,
                    "relevance_label": int(pair["relevance_label"]),
                    "teacher_score": float(teacher_scores[pair_index]),
                    "raw_gnn_score": float(graph_scores[pair_index]),
                    "residual_alpha": alpha,
                    "final_score": float(final_scores[pair_index]),
                    "top_node_attention": _top_nodes(
                        node_ids=graph.node_ids,
                        member_indices=[
                            node_index[node_id]
                            for node_id in node_attention
                        ],
                        attention=list(node_attention.values()),
                        node_by_id=node_by_id,
                    ),
                    "top_edge_contribution": edge_rows,
                    "top_path_contribution": path_rows,
                }
            )
        explanations.append(
            {
                **query_record_by_id[query_id],
                "selected_architecture": architecture,
                "selected_seed": int(selected["seed"]),
                "residual_alpha": alpha,
                "ranked_subgraphs": ranked,
            }
        )

    final_metrics = ranking_metrics(
        final_metrics_rows,
        [float(row["_final_score"]) for row in final_metrics_rows],
    )
    probe_queries = []
    seen_by_category: dict[str, int] = defaultdict(int)
    for row in query_records:
        category = str(row["category"])
        if seen_by_category[category] >= 2:
            continue
        seen_by_category[category] += 1
        probe_queries.append(row)
    probes = []
    probe_vectors = []
    subgraph_rows_by_hop: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(vector_rows):
        subgraph_rows_by_hop[int(row["hop"])].append(index)
    positive_by_query: dict[str, set[str]] = defaultdict(set)
    for row in pair_rows:
        if int(row["relevance_label"]) == 1:
            positive_by_query[str(row["query_id"])].add(str(row["subgraph_id"]))
    for probe in probe_queries:
        query_row = int(probe["query_row"])
        vector = query_vectors[query_row]
        probe_vectors.append(vector)
        indices = subgraph_rows_by_hop[int(probe["target_hop"])]
        scores = subgraph_vectors[indices] @ vector
        order = np.argsort(-scores, kind="stable")[:10]
        returned = [
            str(vector_rows[indices[int(index)]]["subgraph_id"])
            for index in order
        ]
        positives = positive_by_query[str(probe["query_id"])]
        probes.append(
            {
                "query_id": str(probe["query_id"]),
                "category": str(probe["category"]),
                "target_hop": int(probe["target_hop"]),
                "query_row": query_row,
                "reference_top10_subgraph_ids": returned,
                "positive_subgraph_ids": sorted(positives),
                "reference_positive_hit": bool(set(returned) & positives),
            }
        )

    vector_path = output_dir / "subgraph_vectors.npy"
    query_vector_path = output_dir / "query_vectors.npy"
    record_path = output_dir / "subgraph_records.jsonl"
    explanation_path = output_dir / "query_explanations.jsonl"
    probe_path = output_dir / "probe_metadata.jsonl"
    probe_vector_path = output_dir / "probe_vectors.npy"
    metrics_path = output_dir / "gnn_index_metrics.json"
    atomic_npy(vector_path, subgraph_vectors)
    atomic_npy(query_vector_path, query_vectors)
    atomic_jsonl(record_path, vector_rows)
    atomic_jsonl(explanation_path, explanations)
    atomic_jsonl(probe_path, probes)
    atomic_npy(probe_vector_path, np.stack(probe_vectors).astype(np.float32))
    atomic_json(
        metrics_path,
        {
            "coarse_gnn_index": index_metrics,
            "selected_checkpoint_final_ranking": final_metrics,
        },
    )

    graph_manifest_path = graph_dir / "kg_extraction_manifest.json"
    manifest = {
        "accepted": True,
        "gnn_index_version": GNN_INDEX_VERSION,
        "generated_at": utc_now(),
        "kg_run_id": str(
            json.loads(
                graph_manifest_path.read_text(encoding="utf-8")
            )["run_id"]
        ),
        "selected_architecture": architecture,
        "selected_seed": int(selected["seed"]),
        "selection_basis": "highest validation composite within selected architecture",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "teacher_checkpoint_path": str(teacher_path),
        "teacher_checkpoint_sha256": sha256_file(teacher_path),
        "training_input_hash": str(checkpoint["input_hash"]),
        "embedding_model": f"PBPK-HGT/{GNN_TRAINING_VERSION}",
        "embedding_revision": sha256_file(checkpoint_path)[:24],
        "embedding_dimension": GNN_VECTOR_DIMENSION,
        "subgraph_count": len(vector_rows),
        "query_count": len(query_records),
        "explanation_count": len(explanations),
        "probe_count": len(probes),
        "input_hashes": {
            "graph_manifest": sha256_file(graph_manifest_path),
            "comparison": sha256_file(comparison_path),
            "benchmark_manifest": sha256_file(
                benchmark_dir / "gnn_benchmark_manifest.json"
            ),
            "subgraph_manifest": sha256_file(
                subgraph_dir / "gate2_manifest.json"
            ),
            "embedding_index_manifest": sha256_file(
                embedding_index_dir / "embedding_index_manifest.json"
            ),
        },
        "artifacts": {
            "subgraph_vectors": {
                "path": vector_path.name,
                "sha256": sha256_file(vector_path),
                "shape": list(subgraph_vectors.shape),
            },
            "query_vectors": {
                "path": query_vector_path.name,
                "sha256": sha256_file(query_vector_path),
                "shape": list(query_vectors.shape),
            },
            "subgraph_records": {
                "path": record_path.name,
                "sha256": sha256_file(record_path),
                "count": len(vector_rows),
            },
            "query_explanations": {
                "path": explanation_path.name,
                "sha256": sha256_file(explanation_path),
                "count": len(explanations),
            },
            "probe_metadata": {
                "path": probe_path.name,
                "sha256": sha256_file(probe_path),
                "count": len(probes),
            },
            "probe_vectors": {
                "path": probe_vector_path.name,
                "sha256": sha256_file(probe_vector_path),
                "shape": [len(probes), GNN_VECTOR_DIMENSION],
            },
            "metrics": {
                "path": metrics_path.name,
                "sha256": sha256_file(metrics_path),
            },
        },
        "content_sha256": sha256_text(
            canonical_json(
                {
                    "checkpoint": sha256_file(checkpoint_path),
                    "vectors": sha256_file(vector_path),
                    "records": sha256_file(record_path),
                    "explanations": sha256_file(explanation_path),
                }
            )
        ),
    }
    atomic_json(output_dir / "gnn_index_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build checkpointed HGT subgraph vectors and explanations."
    )
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--embedding-index-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_gnn_index(
        graph_dir=args.graph_dir,
        subgraph_dir=args.subgraph_dir,
        embedding_index_dir=args.embedding_index_dir,
        benchmark_dir=args.benchmark_dir,
        training_dir=args.training_dir,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
