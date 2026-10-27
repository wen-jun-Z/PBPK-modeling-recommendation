from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from pbpk_kb.config.settings import get_settings
from pbpk_kb.recommendation.embedding_benchmark import (
    load_jsonl,
    sha256_file,
    stable_id,
)
from pbpk_kb.storage.import_recommendation_embeddings import vector_literal


TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS recommendation_gnn_embedding_record (
    gnn_embedding_record_id TEXT PRIMARY KEY,
    kg_run_id TEXT NOT NULL,
    subgraph_id TEXT NOT NULL,
    center_id TEXT NOT NULL,
    hop SMALLINT NOT NULL CHECK (hop BETWEEN 0 AND 2),
    embedding_row INTEGER NOT NULL CHECK (embedding_row >= 0),
    subgraph_content_sha256 TEXT NOT NULL CHECK (
        subgraph_content_sha256 ~ '^[a-f0-9]{64}$'
    ),
    graph_manifest_sha256 TEXT NOT NULL CHECK (
        graph_manifest_sha256 ~ '^[a-f0-9]{64}$'
    ),
    checkpoint_sha256 TEXT NOT NULL CHECK (
        checkpoint_sha256 ~ '^[a-f0-9]{64}$'
    ),
    embedding vector(256) NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_revision TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension = 256),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (
        kg_run_id, embedding_model, embedding_revision, subgraph_id, hop
    ),
    UNIQUE (
        kg_run_id, embedding_model, embedding_revision, hop, embedding_row
    )
);
CREATE INDEX IF NOT EXISTS recommendation_gnn_embedding_object_idx
    ON recommendation_gnn_embedding_record(kg_run_id, subgraph_id, hop);
CREATE INDEX IF NOT EXISTS recommendation_gnn_embedding_model_idx
    ON recommendation_gnn_embedding_record(
        kg_run_id, embedding_model, embedding_revision, hop
    );
"""


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _index_name(model: str, revision: str) -> str:
    suffix = hashlib.sha256(f"{model}\x1f{revision}".encode()).hexdigest()[:16]
    return f"recommendation_gnn_hnsw_{suffix}_idx"


def _batches(rows: list[dict[str, Any]], size: int = 128) -> Any:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _artifact_path(
    index_dir: Path,
    manifest: dict[str, Any],
    name: str,
) -> Path:
    artifact = manifest["artifacts"][name]
    path = index_dir / str(artifact["path"])
    if sha256_file(path) != str(artifact["sha256"]):
        raise RuntimeError(f"GNN index artifact checksum mismatch: {name}")
    return path


def import_gnn_index(
    index_dir: Path,
    database_url: str,
) -> dict[str, Any]:
    import numpy as np

    manifest_path = index_dir / "gnn_index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("accepted") is not True:
        raise RuntimeError("GNN index manifest is not accepted.")
    if int(manifest.get("embedding_dimension") or 0) != 256:
        raise RuntimeError("GNN index must be 256 dimensional.")

    record_path = _artifact_path(index_dir, manifest, "subgraph_records")
    vector_path = _artifact_path(index_dir, manifest, "subgraph_vectors")
    probe_path = _artifact_path(index_dir, manifest, "probe_metadata")
    probe_vector_path = _artifact_path(index_dir, manifest, "probe_vectors")
    records = load_jsonl(record_path)
    vectors = np.load(vector_path, mmap_mode="r")
    probes = load_jsonl(probe_path)
    probe_vectors = np.load(probe_vector_path, mmap_mode="r")
    if tuple(vectors.shape) != (len(records), 256):
        raise RuntimeError("GNN record/vector shape mismatch.")
    if tuple(probe_vectors.shape) != (len(probes), 256):
        raise RuntimeError("GNN probe shape mismatch.")
    if not np.isfinite(vectors).all() or not np.isfinite(probe_vectors).all():
        raise RuntimeError("GNN index contains non-finite vectors.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise RuntimeError("GNN subgraph vectors are not unit-normalized.")
    for index, (record, vector) in enumerate(
        zip(records, vectors, strict=True)
    ):
        if int(record["embedding_row"]) != index:
            raise RuntimeError("GNN embedding rows are not aligned.")
        if hashlib.sha256(vector.tobytes()).hexdigest() != str(
            record["vector_sha256"]
        ):
            raise RuntimeError("GNN per-vector checksum mismatch.")

    kg_run_id = str(manifest["kg_run_id"])
    model = str(manifest["embedding_model"])
    revision = str(manifest["embedding_revision"])
    checkpoint_sha256 = str(manifest["checkpoint_sha256"])
    graph_manifest_sha256 = str(
        manifest["input_hashes"]["graph_manifest"]
    )
    insert_rows = []
    for record, vector in zip(records, vectors, strict=True):
        metadata = {
            key: record[key]
            for key in (
                "center_type",
                "node_count",
                "edge_count",
                "event_count",
                "citation_count",
                "vector_sha256",
                "top_node_attention",
                "source_ids",
                "quality_statuses",
            )
        }
        insert_rows.append(
            {
                "gnn_embedding_record_id": stable_id(
                    "recommendation-gnn-embedding",
                    kg_run_id,
                    model,
                    revision,
                    record["subgraph_id"],
                    record["hop"],
                ),
                "kg_run_id": kg_run_id,
                "subgraph_id": str(record["subgraph_id"]),
                "center_id": str(record["center_id"]),
                "hop": int(record["hop"]),
                "embedding_row": int(record["embedding_row"]),
                "subgraph_content_sha256": str(
                    record["subgraph_content_sha256"]
                ),
                "graph_manifest_sha256": graph_manifest_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "embedding": vector_literal(vector),
                "embedding_model": model,
                "embedding_revision": revision,
                "metadata_json": json.dumps(metadata, ensure_ascii=False),
            }
        )

    engine = create_engine(database_url, future=True)
    index_name = _index_name(model, revision)
    with engine.begin() as connection:
        for statement in TABLE_SQL.split(";"):
            if statement.strip():
                connection.execute(text(statement))
        connection.execute(
            text(
                """
                DELETE FROM recommendation_gnn_embedding_record
                WHERE kg_run_id = :kg_run_id
                  AND embedding_model = :model
                  AND embedding_revision = :revision
                """
            ),
            {
                "kg_run_id": kg_run_id,
                "model": model,
                "revision": revision,
            },
        )
        insert_sql = text(
            """
            INSERT INTO recommendation_gnn_embedding_record (
                gnn_embedding_record_id, kg_run_id, subgraph_id, center_id,
                hop, embedding_row, subgraph_content_sha256,
                graph_manifest_sha256, checkpoint_sha256, embedding,
                embedding_model, embedding_revision, embedding_dimension,
                metadata_json
            ) VALUES (
                :gnn_embedding_record_id, :kg_run_id, :subgraph_id, :center_id,
                :hop, :embedding_row, :subgraph_content_sha256,
                :graph_manifest_sha256, :checkpoint_sha256,
                CAST(:embedding AS vector), :embedding_model,
                :embedding_revision, 256, CAST(:metadata_json AS jsonb)
            )
            """
        )
        for batch in _batches(insert_rows):
            connection.execute(insert_sql, batch)
        connection.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON recommendation_gnn_embedding_record
                USING hnsw (embedding vector_cosine_ops)
                WHERE embedding_model = {_sql_literal(model)}
                  AND embedding_revision = {_sql_literal(revision)}
                """
            )
        )

        exact_results = []
        connection.execute(text("SET LOCAL enable_indexscan = off"))
        connection.execute(text("SET LOCAL enable_bitmapscan = off"))
        for probe, vector in zip(probes, probe_vectors, strict=True):
            returned = [
                str(row["subgraph_id"])
                for row in connection.execute(
                    text(
                        """
                        SELECT subgraph_id
                        FROM recommendation_gnn_embedding_record
                        WHERE kg_run_id = :kg_run_id
                          AND embedding_model = :model
                          AND embedding_revision = :revision
                          AND hop = :hop
                        ORDER BY embedding <=> CAST(:embedding AS vector),
                                 subgraph_id
                        LIMIT 10
                        """
                    ),
                    {
                        "kg_run_id": kg_run_id,
                        "model": model,
                        "revision": revision,
                        "hop": int(probe["target_hop"]),
                        "embedding": vector_literal(vector),
                    },
                ).mappings()
            ]
            reference = [
                str(value)
                for value in probe["reference_top10_subgraph_ids"]
            ]
            exact_results.append(
                {
                    "query_id": str(probe["query_id"]),
                    "returned": returned,
                    "reference": reference,
                    "exact_top10_reproduced": returned == reference,
                    "positive_hit_reproduced": (
                        bool(
                            set(returned)
                            & set(probe["positive_subgraph_ids"])
                        )
                        == bool(probe["reference_positive_hit"])
                    ),
                }
            )
        if not all(
            row["exact_top10_reproduced"]
            and row["positive_hit_reproduced"]
            for row in exact_results
        ):
            raise RuntimeError("Exact GNN pgvector probes did not reproduce.")

        connection.execute(text("SET LOCAL enable_indexscan = on"))
        connection.execute(text("SET LOCAL enable_bitmapscan = on"))
        connection.execute(text("SET LOCAL enable_seqscan = on"))
        connection.execute(text("SET LOCAL enable_sort = on"))
        connection.execute(text("SET LOCAL hnsw.ef_search = 1000"))
        connection.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
        first = probes[0]
        first_parameters = {
            "kg_run_id": kg_run_id,
            "model": model,
            "revision": revision,
            "hop": int(first["target_hop"]),
            "embedding": vector_literal(probe_vectors[0]),
        }
        plan_sql = text(
            """
            EXPLAIN (FORMAT JSON)
            SELECT subgraph_id
            FROM recommendation_gnn_embedding_record
            WHERE kg_run_id = :kg_run_id
              AND embedding_model = :model
              AND embedding_revision = :revision
              AND hop = :hop
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 10
            """
        )
        default_plan = connection.execute(
            plan_sql, first_parameters
        ).scalar_one()
        connection.execute(text("SET LOCAL enable_seqscan = off"))
        connection.execute(text("SET LOCAL enable_bitmapscan = off"))
        connection.execute(text("SET LOCAL enable_sort = off"))
        forced_plan = connection.execute(
            plan_sql, first_parameters
        ).scalar_one()
        if index_name not in json.dumps(forced_plan, ensure_ascii=False):
            raise RuntimeError("GNN HNSW index is not usable.")

        query_sql = text(
            """
            SELECT subgraph_id
            FROM recommendation_gnn_embedding_record
            WHERE kg_run_id = :kg_run_id
              AND embedding_model = :model
              AND embedding_revision = :revision
              AND hop = :hop
            ORDER BY embedding <=> CAST(:embedding AS vector), subgraph_id
            LIMIT 10
            """
        )
        hnsw_results = []
        for probe, vector, exact in zip(
            probes,
            probe_vectors,
            exact_results,
            strict=True,
        ):
            returned = [
                str(row["subgraph_id"])
                for row in connection.execute(
                    query_sql,
                    {
                        "kg_run_id": kg_run_id,
                        "model": model,
                        "revision": revision,
                        "hop": int(probe["target_hop"]),
                        "embedding": vector_literal(vector),
                    },
                ).mappings()
            ]
            recall = len(set(returned) & set(exact["reference"])) / max(
                1, len(exact["reference"])
            )
            hnsw_results.append(
                {
                    "query_id": str(probe["query_id"]),
                    "top10_recall_against_exact": recall,
                    "positive_hit_reproduced": (
                        bool(
                            set(returned)
                            & set(probe["positive_subgraph_ids"])
                        )
                        == bool(probe["reference_positive_hit"])
                    ),
                }
            )
        if (
            min(row["top10_recall_against_exact"] for row in hnsw_results)
            < 0.9
            or not all(
                row["positive_hit_reproduced"] for row in hnsw_results
            )
        ):
            raise RuntimeError("GNN HNSW probes failed.")
        count = connection.execute(
            text(
                """
                SELECT count(*)
                FROM recommendation_gnn_embedding_record
                WHERE kg_run_id = :kg_run_id
                  AND embedding_model = :model
                  AND embedding_revision = :revision
                """
            ),
            {
                "kg_run_id": kg_run_id,
                "model": model,
                "revision": revision,
            },
        ).scalar_one()
    engine.dispose()
    return {
        "accepted": True,
        "kg_run_id": kg_run_id,
        "embedding_model": model,
        "embedding_revision": revision,
        "checkpoint_sha256": checkpoint_sha256,
        "imported_count": len(insert_rows),
        "database_count": int(count),
        "probe_count": len(probes),
        "all_exact_probes_reproduced": True,
        "partial_hnsw_index": index_name,
        "default_plan_uses_hnsw": index_name
        in json.dumps(default_plan, ensure_ascii=False),
        "forced_hnsw_plan_verified": True,
        "minimum_hnsw_top10_recall": min(
            row["top10_recall_against_exact"] for row in hnsw_results
        ),
        "mean_hnsw_top10_recall": sum(
            row["top10_recall_against_exact"] for row in hnsw_results
        )
        / len(hnsw_results),
        "manifest_sha256": sha256_file(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the checkpointed 256-dimensional HGT index."
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--database-url", default=get_settings().database_url)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = import_gnn_index(args.index_dir, args.database_url)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
