from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from pbpk_kb.config.settings import get_settings
from pbpk_kb.recommendation.embedding_benchmark import load_jsonl, sha256_file, stable_id


TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS recommendation_embedding_record (
    embedding_record_id TEXT PRIMARY KEY,
    kg_run_id TEXT NOT NULL,
    object_kind TEXT NOT NULL CHECK (
        object_kind IN ('node', 'edge', 'evidence', 'subgraph')
    ),
    object_id TEXT NOT NULL,
    hop SMALLINT NOT NULL DEFAULT -1 CHECK (hop BETWEEN -1 AND 2),
    embedding_row INTEGER NOT NULL CHECK (embedding_row >= 0),
    embedding_text TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (input_sha256 ~ '^[a-f0-9]{64}$'),
    graph_manifest_sha256 TEXT NOT NULL CHECK (graph_manifest_sha256 ~ '^[a-f0-9]{64}$'),
    embedding vector(768) NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_revision TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension = 768),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kg_run_id, embedding_model, embedding_revision, object_kind, object_id, hop),
    UNIQUE (kg_run_id, embedding_model, embedding_revision, object_kind, hop, embedding_row)
);
CREATE INDEX IF NOT EXISTS recommendation_embedding_object_idx
    ON recommendation_embedding_record(kg_run_id, object_kind, hop, object_id);
CREATE INDEX IF NOT EXISTS recommendation_embedding_model_idx
    ON recommendation_embedding_record(
        kg_run_id, embedding_model, embedding_revision, object_kind, hop
    );
CREATE TABLE IF NOT EXISTS recommendation_lexical_record (
    lexical_record_id TEXT PRIMARY KEY,
    kg_run_id TEXT NOT NULL,
    object_kind TEXT NOT NULL CHECK (
        object_kind IN ('node', 'edge', 'evidence', 'subgraph')
    ),
    object_id TEXT NOT NULL,
    hop SMALLINT NOT NULL DEFAULT -1 CHECK (hop BETWEEN -1 AND 2),
    lexical_row INTEGER NOT NULL CHECK (lexical_row >= 0),
    canonical_text TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (input_sha256 ~ '^[a-f0-9]{64}$'),
    graph_manifest_sha256 TEXT NOT NULL CHECK (graph_manifest_sha256 ~ '^[a-f0-9]{64}$'),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english'::regconfig, canonical_text)
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kg_run_id, object_kind, object_id, hop),
    UNIQUE (kg_run_id, object_kind, hop, lexical_row)
);
CREATE INDEX IF NOT EXISTS recommendation_lexical_object_idx
    ON recommendation_lexical_record(kg_run_id, object_kind, hop, object_id);
CREATE INDEX IF NOT EXISTS recommendation_lexical_search_idx
    ON recommendation_lexical_record USING GIN (search_vector);
"""


def vector_literal(vector: Any) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _partial_index_name(model: str, revision: str) -> str:
    suffix = hashlib.sha256(f"{model}\x1f{revision}".encode()).hexdigest()[:16]
    return f"recommendation_embedding_hnsw_{suffix}_idx"


def _batched(rows: list[dict[str, Any]], size: int = 128) -> Any:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def import_index(index_dir: Path, database_url: str) -> dict[str, Any]:
    import numpy as np

    manifest_path = index_dir / "embedding_index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("accepted") is not True:
        raise RuntimeError("Recommendation embedding index manifest is not accepted.")
    if int(manifest.get("embedding_dimension") or 0) != 768:
        raise RuntimeError("Only the validated 768-dimensional index can be imported.")
    models = manifest.get("models") or {}
    model_order = [str(value) for value in manifest.get("dense_model_order") or ()]
    if not model_order or set(model_order) != set(models):
        raise RuntimeError("Multi-model index manifest is incomplete.")

    graph_hash = str((manifest.get("input_hashes") or {}).get("graph_manifest") or "")
    kg_run_id = str(manifest["kg_run_id"])
    engine = create_engine(database_url, future=True)
    imported_vectors = 0
    imported_lexical = 0
    model_reports: dict[str, dict[str, Any]] = {}
    with engine.begin() as connection:
        for statement in TABLE_SQL.split(";"):
            if statement.strip():
                connection.execute(text(statement))
        connection.execute(text("DROP INDEX IF EXISTS recommendation_embedding_hnsw_idx"))
        connection.execute(
            text(
                """
                DELETE FROM recommendation_embedding_record
                WHERE kg_run_id = :kg_run_id
                """
            ),
            {"kg_run_id": kg_run_id},
        )
        connection.execute(
            text("DELETE FROM recommendation_lexical_record WHERE kg_run_id = :kg_run_id"),
            {"kg_run_id": kg_run_id},
        )

        source_rows_by_kind: dict[str, list[dict[str, Any]]] = {}
        for kind, source_artifact in sorted(
            (manifest.get("source_records") or {}).items()
        ):
            source_path = index_dir / source_artifact["records_path"]
            if sha256_file(source_path) != source_artifact["records_sha256"]:
                raise RuntimeError(f"Source record checksum mismatch: {kind}")
            source_rows = load_jsonl(source_path)
            if len(source_rows) != int(source_artifact["record_count"]):
                raise RuntimeError(f"Source record count mismatch: {kind}")
            source_rows_by_kind[str(kind)] = source_rows
            lexical_rows = [
                {
                    "lexical_record_id": stable_id(
                        "recommendation-lexical",
                        kg_run_id,
                        row["object_kind"],
                        row["object_id"],
                        row["hop"],
                    ),
                    "kg_run_id": kg_run_id,
                    "object_kind": row["object_kind"],
                    "object_id": row["object_id"],
                    "hop": row["hop"],
                    "lexical_row": row["embedding_row"],
                    "canonical_text": row["embedding_text"],
                    "input_sha256": row["input_sha256"],
                    "graph_manifest_sha256": graph_hash,
                    "metadata_json": json.dumps(
                        row.get("metadata") or {}, ensure_ascii=False
                    ),
                }
                for row in source_rows
            ]
            insert_lexical = text(
                """
                INSERT INTO recommendation_lexical_record (
                    lexical_record_id, kg_run_id, object_kind, object_id, hop,
                    lexical_row, canonical_text, input_sha256,
                    graph_manifest_sha256, metadata_json
                ) VALUES (
                    :lexical_record_id, :kg_run_id, :object_kind, :object_id, :hop,
                    :lexical_row, :canonical_text, :input_sha256,
                    :graph_manifest_sha256, CAST(:metadata_json AS jsonb)
                )
                """
            )
            for batch in _batched(lexical_rows):
                connection.execute(insert_lexical, batch)
            imported_lexical += len(lexical_rows)

        for model in model_order:
            model_manifest = models[model]
            revision = str(model_manifest["revision"])
            model_imported = 0
            for artifact_name, artifact in sorted(
                (model_manifest.get("artifacts") or {}).items()
            ):
                source_rows = source_rows_by_kind.get(str(artifact_name))
                if source_rows is None:
                    raise RuntimeError(f"Missing source records for {artifact_name}.")
                token_path = index_dir / artifact["token_audit_path"]
                vectors_path = index_dir / artifact["vectors_path"]
                if sha256_file(token_path) != artifact["token_audit_sha256"]:
                    raise RuntimeError(
                        f"Token audit checksum mismatch: {model}/{artifact_name}"
                    )
                if sha256_file(vectors_path) != artifact["vectors_sha256"]:
                    raise RuntimeError(
                        f"Vector checksum mismatch: {model}/{artifact_name}"
                    )
                token_rows = load_jsonl(token_path)
                vectors = np.load(vectors_path, mmap_mode="r")
                if (
                    tuple(vectors.shape) != tuple(artifact["shape"])
                    or len(source_rows) != len(vectors)
                    or len(token_rows) != len(vectors)
                ):
                    raise RuntimeError(f"Artifact shape mismatch: {model}/{artifact_name}")
                insert_rows = []
                for source_row, token_row, vector in zip(
                    source_rows, token_rows, vectors, strict=True
                ):
                    if (
                        int(source_row["embedding_row"])
                        != int(token_row["embedding_row"])
                        or source_row["object_id"] != token_row["object_id"]
                        or source_row["input_sha256"]
                        != token_row["canonical_input_sha256"]
                    ):
                        raise RuntimeError(
                            f"Record alignment mismatch: {model}/{artifact_name}"
                        )
                    metadata = dict(source_row.get("metadata") or {})
                    metadata["canonical_input_sha256"] = source_row["input_sha256"]
                    metadata["tokenization"] = token_row["tokenization"]
                    insert_rows.append(
                        {
                            "embedding_record_id": token_row["embedding_record_id"],
                            "kg_run_id": kg_run_id,
                            "object_kind": source_row["object_kind"],
                            "object_id": source_row["object_id"],
                            "hop": source_row["hop"],
                            "embedding_row": source_row["embedding_row"],
                            "embedding_text": token_row["model_input_text"],
                            "input_sha256": token_row["model_input_sha256"],
                            "graph_manifest_sha256": graph_hash,
                            "embedding": vector_literal(vector),
                            "embedding_model": model,
                            "embedding_revision": revision,
                            "metadata_json": json.dumps(metadata, ensure_ascii=False),
                        }
                    )
                insert_vector = text(
                    """
                    INSERT INTO recommendation_embedding_record (
                        embedding_record_id, kg_run_id, object_kind, object_id, hop,
                        embedding_row, embedding_text, input_sha256,
                        graph_manifest_sha256, embedding, embedding_model,
                        embedding_revision, embedding_dimension, metadata_json
                    ) VALUES (
                        :embedding_record_id, :kg_run_id, :object_kind, :object_id, :hop,
                        :embedding_row, :embedding_text, :input_sha256,
                        :graph_manifest_sha256, CAST(:embedding AS vector), :embedding_model,
                        :embedding_revision, 768, CAST(:metadata_json AS jsonb)
                    )
                    """
                )
                for batch in _batched(insert_rows):
                    connection.execute(insert_vector, batch)
                imported_vectors += len(insert_rows)
                model_imported += len(insert_rows)

            index_name = _partial_index_name(model, revision)
            connection.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON recommendation_embedding_record
                    USING hnsw (embedding vector_cosine_ops)
                    WHERE embedding_model = {_sql_literal(model)}
                      AND embedding_revision = {_sql_literal(revision)}
                    """
                )
            )

            probe_artifact = model_manifest["probe_queries"]
            probe_path = index_dir / probe_artifact["metadata_path"]
            probe_vectors_path = index_dir / probe_artifact["vectors_path"]
            if sha256_file(probe_path) != probe_artifact["metadata_sha256"]:
                raise RuntimeError(f"Probe metadata checksum mismatch: {model}")
            if sha256_file(probe_vectors_path) != probe_artifact["vectors_sha256"]:
                raise RuntimeError(f"Probe vector checksum mismatch: {model}")
            probes = load_jsonl(probe_path)
            probe_vectors = np.load(probe_vectors_path, mmap_mode="r")
            if len(probes) != len(probe_vectors):
                raise RuntimeError(f"Probe shape mismatch: {model}")

            connection.execute(text("SET LOCAL enable_indexscan = off"))
            connection.execute(text("SET LOCAL enable_bitmapscan = off"))
            exact_results = []
            for probe, vector in zip(probes, probe_vectors, strict=True):
                rows = connection.execute(
                    text(
                        """
                        SELECT object_id
                        FROM recommendation_embedding_record
                        WHERE kg_run_id = :kg_run_id
                          AND embedding_model = :model
                          AND embedding_revision = :revision
                          AND object_kind = 'subgraph'
                          AND hop = :hop
                        ORDER BY embedding <=> CAST(:embedding AS vector), object_id
                        LIMIT 10
                        """
                    ),
                    {
                        "embedding": vector_literal(vector),
                        "kg_run_id": kg_run_id,
                        "model": model,
                        "revision": revision,
                        "hop": int(probe["target_hop"]),
                    },
                ).mappings().all()
                returned = [str(row["object_id"]) for row in rows]
                reference = [
                    str(value) for value in probe["reference_top10_subgraph_ids"]
                ]
                exact_results.append(
                    {
                        "query_id": probe["query_id"],
                        "category": probe["category"],
                        "top10_set_reproduced": set(returned) == set(reference),
                        "positive_hit_reproduced": bool(
                            set(returned) & set(probe["positive_subgraph_ids"])
                        )
                        == bool(probe["reference_positive_hit"]),
                    }
                )
            if not exact_results or not all(
                row["top10_set_reproduced"] and row["positive_hit_reproduced"]
                for row in exact_results
            ):
                raise RuntimeError(f"Exact pgvector probe validation failed for {model}.")

            connection.execute(text("SET LOCAL enable_indexscan = on"))
            connection.execute(text("SET LOCAL enable_bitmapscan = on"))
            connection.execute(text("SET LOCAL enable_seqscan = on"))
            connection.execute(text("SET LOCAL enable_sort = on"))
            connection.execute(text("SET LOCAL hnsw.ef_search = 1000"))
            connection.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
            connection.execute(text("SET LOCAL hnsw.max_scan_tuples = 100000"))
            first_probe = probes[0]
            first_vector = probe_vectors[0]
            query_sql = text(
                """
                SELECT object_id
                FROM recommendation_embedding_record
                WHERE kg_run_id = :kg_run_id
                  AND embedding_model = :model
                  AND embedding_revision = :revision
                  AND object_kind = 'subgraph'
                  AND hop = :hop
                ORDER BY embedding <=> CAST(:embedding AS vector), object_id
                LIMIT 10
                """
            )
            plan_sql = text(
                """
                EXPLAIN (FORMAT JSON)
                SELECT object_id
                FROM recommendation_embedding_record
                WHERE kg_run_id = :kg_run_id
                  AND embedding_model = :model
                  AND embedding_revision = :revision
                  AND object_kind = 'subgraph'
                  AND hop = :hop
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT 10
                """
            )
            first_parameters = {
                "embedding": vector_literal(first_vector),
                "kg_run_id": kg_run_id,
                "model": model,
                "revision": revision,
                "hop": int(first_probe["target_hop"]),
            }
            default_plan = connection.execute(
                plan_sql,
                first_parameters,
            ).scalar_one()

            # At the current graph size an exact B-tree filter followed by a sort
            # can legitimately be cheaper. Disable competing paths only in this
            # validation session to prove the isolated HNSW path is executable.
            connection.execute(text("SET LOCAL enable_seqscan = off"))
            connection.execute(text("SET LOCAL enable_bitmapscan = off"))
            connection.execute(text("SET LOCAL enable_sort = off"))
            forced_plan = connection.execute(
                plan_sql,
                first_parameters,
            ).scalar_one()
            forced_plan_text = json.dumps(forced_plan, ensure_ascii=False)
            if index_name not in forced_plan_text:
                raise RuntimeError(f"Partial HNSW index is not usable for {model}.")

            hnsw_results = []
            for probe, vector, exact_result in zip(
                probes, probe_vectors, exact_results, strict=True
            ):
                parameters = {
                    "embedding": vector_literal(vector),
                    "kg_run_id": kg_run_id,
                    "model": model,
                    "revision": revision,
                    "hop": int(probe["target_hop"]),
                }
                returned = [
                    str(row["object_id"])
                    for row in connection.execute(
                        query_sql,
                        parameters,
                    ).mappings()
                ]
                reference = [
                    str(value) for value in probe["reference_top10_subgraph_ids"]
                ]
                recall = len(set(returned) & set(reference)) / max(1, len(reference))
                positive_hit = bool(
                    set(returned) & set(probe["positive_subgraph_ids"])
                )
                hnsw_results.append(
                    {
                        "query_id": probe["query_id"],
                        "category": probe["category"],
                        "top10_recall_against_exact": recall,
                        "positive_hit_reproduced": (
                            positive_hit == bool(probe["reference_positive_hit"])
                        ),
                        "exact_probe_valid": bool(
                            exact_result["top10_set_reproduced"]
                            and exact_result["positive_hit_reproduced"]
                        ),
                    }
                )
            minimum_hnsw_recall = min(
                row["top10_recall_against_exact"] for row in hnsw_results
            )
            mean_hnsw_recall = sum(
                row["top10_recall_against_exact"] for row in hnsw_results
            ) / len(hnsw_results)
            if minimum_hnsw_recall < 0.9 or not all(
                row["positive_hit_reproduced"] and row["exact_probe_valid"]
                for row in hnsw_results
            ):
                raise RuntimeError(f"HNSW probe validation failed for {model}.")

            connection.execute(text("SET LOCAL enable_sort = on"))
            connection.execute(text("SET LOCAL enable_seqscan = on"))
            connection.execute(text("SET LOCAL enable_bitmapscan = on"))

            count = connection.execute(
                text(
                    """
                    SELECT count(*) FROM recommendation_embedding_record
                    WHERE kg_run_id = :kg_run_id
                      AND embedding_model = :model
                      AND embedding_revision = :revision
                    """
                ),
                {"kg_run_id": kg_run_id, "model": model, "revision": revision},
            ).scalar_one()
            model_reports[model] = {
                "revision": revision,
                "imported_count": model_imported,
                "database_count": int(count),
                "probe_count": len(exact_results),
                "all_exact_probes_reproduced": True,
                "partial_hnsw_index": index_name,
                "hnsw_plan_verified": True,
                "default_plan_uses_hnsw": index_name
                in json.dumps(default_plan, ensure_ascii=False),
                "forced_hnsw_probe_count": len(hnsw_results),
                "minimum_hnsw_top10_recall": minimum_hnsw_recall,
                "mean_hnsw_top10_recall": mean_hnsw_recall,
                "all_hnsw_positive_hits_reproduced": True,
            }

        lexical_count = connection.execute(
            text(
                """
                SELECT count(*) FROM recommendation_lexical_record
                WHERE kg_run_id = :kg_run_id
                """
            ),
            {"kg_run_id": kg_run_id},
        ).scalar_one()
    engine.dispose()
    return {
        "accepted": True,
        "kg_run_id": kg_run_id,
        "model_count": len(model_reports),
        "models": model_reports,
        "imported_vector_count": imported_vectors,
        "imported_lexical_count": imported_lexical,
        "database_lexical_count": int(lexical_count),
        "index_manifest_sha256": sha256_file(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import recommendation vectors into pgvector.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--database-url", default=get_settings().database_url)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = import_index(args.index_dir, args.database_url)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
