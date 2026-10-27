from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.paths import recommendation_model_cache_dir
from pbpk_kb.recommendation.embedding_benchmark import (
    _model_factory,
    load_jsonl,
    MODEL_INPUT_SERIALIZATION_VERSION,
    prepare_model_input,
    sha256_file,
    stable_id,
)


INDEX_VERSION = "pbpk-recommendation-multiretriever-index-2.0.0"
EXPECTED_DIMENSION = 768
LOCKED_DENSE_MODELS = (
    "ncbi/MedCPT",
    "NeuML/pubmedbert-base-embeddings",
    "Alibaba-NLP/gte-modernbert-base",
    "BAAI/bge-base-en-v1.5",
    "Qwen/Qwen3-Embedding-0.6B",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


def _atomic_npy(path: Path, matrix: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        import numpy as np

        np.save(handle, matrix, allow_pickle=False)
    os.replace(temporary, path)


def _input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _node_text(node: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"NODE TYPE: {node['node_type']}",
            f"NAME: {node['preferred_name']}",
            "IDENTIFIERS: "
            + json.dumps(node.get("standard_identifiers") or {}, ensure_ascii=False, sort_keys=True),
            "PROPERTIES: "
            + json.dumps(node.get("properties") or {}, ensure_ascii=False, sort_keys=True),
        )
    )


def _edge_text(edge: dict[str, Any], node_by_id: dict[str, dict[str, Any]]) -> str:
    subject = node_by_id[str(edge["subject_id"])]
    object_ = node_by_id[str(edge["object_id"])]
    provenance = edge.get("provenance") or {}
    quote = " ".join(str(provenance.get("supporting_text") or "").split())[:1500]
    return "\n".join(
        (
            f"SUBJECT [{subject['node_type']}]: {subject['preferred_name']}",
            f"RELATION: {edge['predicate']}",
            f"OBJECT [{object_['node_type']}]: {object_['preferred_name']}",
            "PROPERTIES: "
            + json.dumps(edge.get("properties") or {}, ensure_ascii=False, sort_keys=True),
            f"SOURCE: {provenance.get('source_id')} {provenance.get('source_location')}",
            f"SUPPORTING EVIDENCE: {quote}",
        )
    )


def _build_records(
    *,
    graph_dir: Path,
    subgraph_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    nodes = load_jsonl(graph_dir / "kg_nodes.jsonl")
    edges = load_jsonl(graph_dir / "kg_edges.jsonl")
    node_by_id = {str(node["node_id"]): node for node in nodes}
    records: dict[str, list[dict[str, Any]]] = {}

    records["node"] = [
        {
            "object_id": str(node["node_id"]),
            "object_kind": "node",
            "hop": -1,
            "embedding_text": _node_text(node),
            "metadata": {
                "node_type": node["node_type"],
                "preferred_name": node["preferred_name"],
                "standard_identifiers": node.get("standard_identifiers") or {},
            },
        }
        for node in nodes
    ]
    records["edge"] = [
        {
            "object_id": str(edge["edge_id"]),
            "object_kind": "edge",
            "hop": -1,
            "embedding_text": _edge_text(edge, node_by_id),
            "metadata": {
                "subject_id": edge["subject_id"],
                "predicate": edge["predicate"],
                "object_id": edge["object_id"],
                "evidence_id": (edge.get("provenance") or {}).get("evidence_id"),
            },
        }
        for edge in edges
    ]

    edges_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        evidence_id = str((edge.get("provenance") or {}).get("evidence_id") or "")
        if evidence_id:
            edges_by_evidence[evidence_id].append(edge)
    evidence_records: list[dict[str, Any]] = []
    for evidence_id, evidence_edges in sorted(edges_by_evidence.items()):
        evidence_node = node_by_id.get(evidence_id)
        if not evidence_node or evidence_node.get("node_type") != "Evidence":
            continue
        provenance = evidence_edges[0].get("provenance") or {}
        relation_lines = []
        for edge in sorted(evidence_edges, key=lambda row: str(row["edge_id"]))[:50]:
            subject = node_by_id[str(edge["subject_id"])]
            object_ = node_by_id[str(edge["object_id"])]
            relation_lines.append(
                f"{subject['preferred_name']} -[{edge['predicate']}]-> {object_['preferred_name']}"
            )
        text = "\n".join(
            (
                f"EVIDENCE: {evidence_node['preferred_name']}",
                f"SOURCE: {provenance.get('source_id')} {provenance.get('source_location')}",
                "GRAPH FACTS:",
                *relation_lines,
                "SOURCE TEXT: " + str(provenance.get("supporting_text") or ""),
            )
        )
        evidence_records.append(
            {
                "object_id": evidence_id,
                "object_kind": "evidence",
                "hop": -1,
                "embedding_text": text,
                "metadata": {
                    "source_id": provenance.get("source_id"),
                    "source_location": provenance.get("source_location"),
                    "source_text_hash": provenance.get("source_text_hash"),
                    "relation_count": len(evidence_edges),
                },
            }
        )
    records["evidence"] = evidence_records

    for hop in (0, 1, 2):
        records[f"subgraph_{hop}hop"] = [
            {
                "object_id": str(row["subgraph_id"]),
                "object_kind": "subgraph",
                "hop": hop,
                "embedding_text": str(row["embedding_text"]),
                "metadata": {
                    "center_id": row["center_id"],
                    "center_type": row["center_type"],
                    "center_name": row["center_name"],
                    "node_count": row["node_count"],
                    "edge_count": row["edge_count"],
                    "citation_count": row["citation_count"],
                    "subgraph_content_sha256": row["content_sha256"],
                },
            }
            for row in load_jsonl(subgraph_dir / f"subgraphs_{hop}hop.jsonl")
        ]
    return records


def build_embedding_index(
    *,
    graph_dir: Path,
    subgraph_dir: Path,
    gate1_audit_path: Path,
    gate2_audit_path: Path,
    benchmark_dir: Path,
    comparison_path: Path,
    output_dir: Path,
    cache_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("accepted") is not True:
        raise RuntimeError("Gate 3 multi-retriever comparison is not accepted.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA index build requested without a CUDA device.")

    candidate_generation = comparison.get("candidate_generation") or {}
    dense_models = tuple(str(value) for value in candidate_generation.get("dense_models") or ())
    if dense_models != LOCKED_DENSE_MODELS:
        raise RuntimeError(
            "Formal index model list differs from the locked Gate 3 candidate generators: "
            f"{dense_models!r}"
        )
    dense_revisions = {
        str(key): str(value)
        for key, value in (candidate_generation.get("dense_revisions") or {}).items()
    }
    if set(dense_revisions) != set(LOCKED_DENSE_MODELS):
        raise RuntimeError("Gate 3 comparison does not contain every locked dense revision.")

    benchmark_query_path = benchmark_dir / "locked_retrieval_silver.jsonl"
    benchmark_acceptance_path = benchmark_dir / "preregistered_acceptance.json"
    benchmark_query_sha256 = sha256_file(benchmark_query_path)
    benchmark_acceptance_sha256 = sha256_file(benchmark_acceptance_path)
    if benchmark_query_sha256 != str(comparison.get("benchmark_query_sha256") or ""):
        raise RuntimeError("Locked benchmark query checksum differs from Gate 3 comparison.")
    if benchmark_acceptance_sha256 != str(comparison.get("acceptance_sha256") or ""):
        raise RuntimeError("Locked benchmark acceptance checksum differs from Gate 3 comparison.")

    records_by_kind = _build_records(graph_dir=graph_dir, subgraph_dir=subgraph_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_record_dir = output_dir / "source_records"
    source_record_dir.mkdir(parents=True, exist_ok=True)
    source_artifacts: dict[str, dict[str, Any]] = {}
    for kind, records in records_by_kind.items():
        for index, record in enumerate(records):
            record["embedding_row"] = index
            record["input_sha256"] = _input_hash(record["embedding_text"])
        record_path = source_record_dir / f"{kind}_records.jsonl"
        _atomic_jsonl(record_path, records)
        source_artifacts[kind] = {
            "record_count": len(records),
            "records_path": str(record_path.relative_to(output_dir)),
            "records_sha256": sha256_file(record_path),
        }

    queries = load_jsonl(benchmark_query_path)
    selected_probe_queries: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for query in sorted(queries, key=lambda row: str(row["query_id"])):
        category = str(query["category"])
        if category_counts[category] >= 2:
            continue
        category_counts[category] += 1
        selected_probe_queries.append(query)
    if len(category_counts) != 6 or len(selected_probe_queries) != 12:
        raise RuntimeError(
            "Expected two deterministic probes for each of the six retrieval categories."
        )

    model_manifests: dict[str, dict[str, Any]] = {}
    started = time.time()
    for model_name in LOCKED_DENSE_MODELS:
        expected_revision = dense_revisions[model_name]
        encoder = _model_factory(
            model_name,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
        )
        if encoder.revision != expected_revision:
            raise RuntimeError(
                f"Embedding revision changed for {model_name}: "
                f"{encoder.revision} != {expected_revision}"
            )
        if encoder.dimension != EXPECTED_DIMENSION:
            raise RuntimeError(
                f"Expected {EXPECTED_DIMENSION}-dimensional {model_name} vectors, "
                f"got {encoder.dimension}."
            )

        model_key = re.sub(r"[^A-Za-z0-9._-]+", "__", model_name)
        model_dir = output_dir / "dense_models" / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, dict[str, Any]] = {}
        for kind, records in records_by_kind.items():
            prepared = [
                prepare_model_input(
                    encoder=encoder,
                    text=str(record["embedding_text"]),
                    object_kind=str(record["object_kind"]),
                )
                for record in records
            ]
            texts = [value[0] for value in prepared]
            compaction_stats = [value[1] for value in prepared]
            token_stats = encoder.document_token_stats(texts)
            if len(token_stats) != len(records):
                raise RuntimeError(f"Token audit row mismatch for {model_name}/{kind}.")
            token_rows = []
            for record, model_input, compaction, token_stat in zip(
                records,
                texts,
                compaction_stats,
                token_stats,
                strict=True,
            ):
                section_counts = {
                    "edge": sum(
                        line.startswith("EDGE ")
                        for line in record["embedding_text"].splitlines()
                    ),
                    "event": sum(
                        line.startswith("EVENT ")
                        for line in record["embedding_text"].splitlines()
                    ),
                    "citation": sum(
                        line.startswith("CITATION ")
                        for line in record["embedding_text"].splitlines()
                    ),
                    "source_family_summary": int(
                        "SOURCE FAMILIES " in record["embedding_text"]
                    ),
                    "relation_type_summary": int(
                        "RELATION TYPES " in record["embedding_text"]
                    ),
                }
                if (
                    record["object_kind"] == "subgraph"
                    and record["hop"] > 0
                    and (
                        not section_counts["relation_type_summary"]
                        or not section_counts["source_family_summary"]
                    )
                ):
                    raise RuntimeError(
                        "Subgraph embedding text lacks critical sections: "
                        f"{record['object_id']}"
                    )
                if record["object_kind"] == "subgraph" and bool(token_stat["truncated"]):
                    raise RuntimeError(
                        f"Formal subgraph was truncated for {model_name}: "
                        f"{record['object_id']}"
                    )
                token_rows.append(
                    {
                        "embedding_row": record["embedding_row"],
                        "embedding_record_id": stable_id(
                            "recommendation-embedding",
                            benchmark_query_sha256,
                            model_name,
                            expected_revision,
                            kind,
                            record["object_id"],
                            record["hop"],
                        ),
                        "object_id": record["object_id"],
                        "canonical_input_sha256": record["input_sha256"],
                        "model_input_text": model_input,
                        "model_input_sha256": _input_hash(model_input),
                        "tokenization": token_stat
                        | {
                            "section_counts": section_counts,
                            "compaction": compaction,
                        },
                    }
                )

            vectors = np.asarray(encoder.encode_documents(texts), dtype=np.float32)
            if vectors.shape != (len(records), EXPECTED_DIMENSION):
                raise RuntimeError(
                    f"Unexpected {model_name}/{kind} embedding shape: {vectors.shape}"
                )
            norms = np.linalg.norm(vectors, axis=1)
            if not np.isfinite(vectors).all() or not np.allclose(
                norms, 1.0, atol=1e-4
            ):
                raise RuntimeError(
                    f"{model_name}/{kind} embeddings are not finite unit vectors."
                )
            token_path = model_dir / f"{kind}_token_audit.jsonl"
            vectors_path = model_dir / f"{kind}_embeddings.npy"
            _atomic_jsonl(token_path, token_rows)
            _atomic_npy(vectors_path, vectors)
            artifacts[kind] = {
                "record_count": len(records),
                "token_audit_path": str(token_path.relative_to(output_dir)),
                "token_audit_sha256": sha256_file(token_path),
                "vectors_path": str(vectors_path.relative_to(output_dir)),
                "vectors_sha256": sha256_file(vectors_path),
                "shape": list(vectors.shape),
                "token_summary": {
                    "maximum_original_tokens": max(
                        (int(row["original_token_count"]) for row in token_stats),
                        default=0,
                    ),
                    "maximum_model_input_tokens": max(
                        (int(row["model_input_token_count"]) for row in token_stats),
                        default=0,
                    ),
                    "truncated_record_count": sum(
                        bool(row["truncated"]) for row in token_stats
                    ),
                    "model_max_length": (
                        int(token_stats[0]["model_max_length"]) if token_stats else 0
                    ),
                    "tokenizer_class": (
                        str(token_stats[0]["tokenizer_class"]) if token_stats else None
                    ),
                    "tokenizer_revision": (
                        str(token_stats[0]["tokenizer_revision"]) if token_stats else None
                    ),
                },
            }
            print(
                json.dumps(
                    {
                        "embedded_model": model_name,
                        "embedded_kind": kind,
                        "record_count": len(records),
                    }
                ),
                flush=True,
            )

        subgraph_vectors = {
            hop: np.load(
                output_dir / artifacts[f"subgraph_{hop}hop"]["vectors_path"],
                mmap_mode="r",
            )
            for hop in (0, 1, 2)
        }
        subgraph_records = {
            hop: records_by_kind[f"subgraph_{hop}hop"] for hop in (0, 1, 2)
        }
        probe_matrix = np.asarray(
            encoder.encode_queries(
                [str(query["query_text"]) for query in selected_probe_queries]
            ),
            dtype=np.float32,
        )
        probe_rows: list[dict[str, Any]] = []
        for probe_row, (query, vector) in enumerate(
            zip(selected_probe_queries, probe_matrix, strict=True)
        ):
            hop = int(query["target_hop"])
            scores = subgraph_vectors[hop] @ vector
            top_indices = np.argsort(-scores, kind="stable")[:10]
            top_ids = [
                str(subgraph_records[hop][int(index)]["object_id"])
                for index in top_indices
            ]
            positives = sorted(str(value) for value in query["positive_subgraph_ids"])
            probe_rows.append(
                {
                    "probe_row": probe_row,
                    "query_id": query["query_id"],
                    "query_text": query["query_text"],
                    "category": query["category"],
                    "target_hop": hop,
                    "positive_subgraph_ids": positives,
                    "reference_top10_subgraph_ids": top_ids,
                    "reference_positive_hit": bool(set(positives) & set(top_ids)),
                }
            )
        probe_metadata_path = model_dir / "probe_queries.jsonl"
        probe_vectors_path = model_dir / "probe_query_embeddings.npy"
        _atomic_jsonl(probe_metadata_path, probe_rows)
        _atomic_npy(probe_vectors_path, probe_matrix)
        model_manifests[model_name] = {
            "revision": expected_revision,
            "dimension": EXPECTED_DIMENSION,
            "normalization": "l2",
            "artifacts": artifacts,
            "probe_queries": {
                "record_count": len(probe_rows),
                "category_counts": dict(sorted(category_counts.items())),
                "metadata_path": str(probe_metadata_path.relative_to(output_dir)),
                "metadata_sha256": sha256_file(probe_metadata_path),
                "vectors_path": str(probe_vectors_path.relative_to(output_dir)),
                "vectors_sha256": sha256_file(probe_vectors_path),
                "shape": list(probe_matrix.shape),
            },
        }
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    graph_manifest_path = graph_dir / "kg_extraction_manifest.json"
    gate2_manifest_path = subgraph_dir / "gate2_manifest.json"
    manifest = {
        "accepted": True,
        "index_version": INDEX_VERSION,
        "kg_run_id": json.loads(graph_manifest_path.read_text(encoding="utf-8"))["run_id"],
        "embedding_dimension": EXPECTED_DIMENSION,
        "model_input_serialization_version": MODEL_INPUT_SERIALIZATION_VERSION,
        "dense_model_order": list(LOCKED_DENSE_MODELS),
        "models": model_manifests,
        "source_records": source_artifacts,
        "lexical_index": {
            "revision": str(candidate_generation.get("lexical_revision") or ""),
            "source": "shared canonical source_records",
            "postgres_configuration": "english",
            "index_method": "generated tsvector plus GIN",
        },
        "fine_retrieval": {
            "reranker_model": str((comparison.get("reranker") or {}).get("model") or ""),
            "reranker_revision": str(
                (comparison.get("reranker") or {}).get("revision") or ""
            ),
            "graph_constraints": "ModelingRequest-derived hard compatibility tier",
        },
        "device": device,
        "elapsed_seconds": round(time.time() - started, 3),
        "input_hashes": {
            "graph_manifest": sha256_file(graph_manifest_path),
            "gate1_audit": sha256_file(gate1_audit_path),
            "gate2_manifest": sha256_file(gate2_manifest_path),
            "gate2_audit": sha256_file(gate2_audit_path),
            "benchmark_queries": benchmark_query_sha256,
            "benchmark_acceptance": benchmark_acceptance_sha256,
            "model_comparison": sha256_file(comparison_path),
        },
        "generated_at": utc_now(),
    }
    _atomic_json(output_dir / "embedding_index_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the selected biomedical embedding index.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--gate1-audit", type=Path, required=True)
    parser.add_argument("--gate2-audit", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=recommendation_model_cache_dir())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_embedding_index(
        graph_dir=args.graph_dir,
        subgraph_dir=args.subgraph_dir,
        gate1_audit_path=args.gate1_audit,
        gate2_audit_path=args.gate2_audit,
        benchmark_dir=args.benchmark_dir,
        comparison_path=args.comparison,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
