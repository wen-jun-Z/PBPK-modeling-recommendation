from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import text

from pbpk_kb.config.settings import get_settings
from pbpk_kb.recommendation.gate6_evaluation import canonical_json, sha256_file
from pbpk_kb.recommendation.graph_prompt import (
    CITATION_CONTEXT_VERSION,
    GRAPH_PROMPT_VERSION,
    RECOMMENDATION_SCHEMA_VERSION,
    VALIDATOR_VERSION,
)
from pbpk_kb.storage.db import create_db_engine


RECOMMENDATION_TABLE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS recommendation_model_run (
        recommendation_model_run_id TEXT PRIMARY KEY,
        gate5_manifest_sha256 TEXT NOT NULL UNIQUE CHECK (
            gate5_manifest_sha256 ~ '^[a-f0-9]{64}$'
        ),
        gate6_manifest_sha256 TEXT NOT NULL UNIQUE CHECK (
            gate6_manifest_sha256 ~ '^[a-f0-9]{64}$'
        ),
        kg_run_id TEXT NOT NULL,
        evidence_run_id TEXT NOT NULL,
        llm_model TEXT NOT NULL,
        llm_revision TEXT NOT NULL,
        llm_dtype TEXT NOT NULL,
        projector_checkpoint_sha256 TEXT NOT NULL CHECK (
            projector_checkpoint_sha256 ~ '^[a-f0-9]{64}$'
        ),
        benchmark_level TEXT NOT NULL CHECK (benchmark_level = 'silver'),
        engineering_acceptance BOOLEAN NOT NULL,
        generalization_claim_eligible BOOLEAN NOT NULL,
        gold_claim BOOLEAN NOT NULL DEFAULT FALSE,
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendation_result (
        recommendation_result_id TEXT PRIMARY KEY,
        recommendation_model_run_id TEXT NOT NULL
            REFERENCES recommendation_model_run(
                recommendation_model_run_id
            ) ON DELETE RESTRICT,
        example_id TEXT NOT NULL,
        compound_node_id TEXT NOT NULL,
        compound_name TEXT NOT NULL,
        objective TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status = 'recommendation_validated'),
        request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[a-f0-9]{64}$'),
        recommendation_sha256 TEXT NOT NULL CHECK (
            recommendation_sha256 ~ '^[a-f0-9]{64}$'
        ),
        overall_confidence DOUBLE PRECISION NOT NULL CHECK (
            overall_confidence BETWEEN 0 AND 1
        ),
        citation_count INTEGER NOT NULL CHECK (citation_count >= 0),
        evidence_gap_count INTEGER NOT NULL CHECK (evidence_gap_count >= 0),
        request_json JSONB NOT NULL,
        recommendation_json JSONB NOT NULL,
        validation_json JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (recommendation_model_run_id, example_id),
        UNIQUE (recommendation_model_run_id, request_sha256)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS recommendation_model_run_acceptance_idx
        ON recommendation_model_run(
            engineering_acceptance, generalization_claim_eligible, created_at
        )
    """,
    """
    CREATE INDEX IF NOT EXISTS recommendation_result_compound_idx
        ON recommendation_result(
            recommendation_model_run_id, compound_node_id, objective
        )
    """,
    """
    CREATE INDEX IF NOT EXISTS recommendation_result_request_idx
        ON recommendation_result(request_sha256)
    """,
)


def stable_id(namespace: str, *values: str) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()
    return f"{namespace}:{digest[:24]}"


def _request_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _missing_expected_result_ids(
    result_rows: list[dict[str, Any]],
    persisted_result_ids: set[str],
) -> list[str]:
    expected_result_ids = {
        str(row["recommendation_result_id"])
        for row in result_rows
    }
    return sorted(expected_result_ids - persisted_result_ids)


def _validated_run_artifacts(
    run_dir: Path,
    *,
    require_human_base_pk: bool,
) -> dict[str, Any]:
    paths = {
        "manifest": run_dir / "recommendation_run_manifest.json",
        "request": run_dir / "recommendation_request.json",
        "recommendation": run_dir / "recommendation_final.json",
        "validation": run_dir / "recommendation_validation.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Incomplete online recommendation run {run_dir.name}: "
            + ", ".join(missing)
        )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    expected_versions = {
        "assembly_version": GRAPH_PROMPT_VERSION,
        "recommendation_schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "citation_context_version": CITATION_CONTEXT_VERSION,
    }
    version_errors = {
        key: {"expected": expected, "observed": manifest.get(key)}
        for key, expected in expected_versions.items()
        if manifest.get(key) != expected
    }
    if version_errors:
        raise RuntimeError(
            f"Recommendation contract version mismatch: {run_dir.name}: "
            f"{json.dumps(version_errors, sort_keys=True)}"
        )
    artifacts = manifest.get("artifacts") or {}
    for path in paths.values():
        if path == paths["manifest"]:
            continue
        expected = str(artifacts.get(path.name) or "")
        if expected != sha256_file(path):
            raise RuntimeError(
                f"Artifact checksum mismatch: {run_dir.name}/{path.name}"
            )
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    recommendation = json.loads(
        paths["recommendation"].read_text(encoding="utf-8")
    )
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "recommendation_validated"
        or recommendation.get("status") != "recommendation_validated"
        or validation.get("status") != "recommendation_validated"
        or validation.get("error_count") != 0
    ):
        raise RuntimeError(f"Run is not independently validated: {run_dir.name}")
    if recommendation.get("generated_from_auto_validated_only") is not True:
        raise RuntimeError(
            f"Run includes non-validated evidence: {run_dir.name}"
        )
    if recommendation.get("schema_version") != RECOMMENDATION_SCHEMA_VERSION:
        raise RuntimeError(
            f"Recommendation schema mismatch: {run_dir.name}"
        )
    if recommendation.get("citation_context_version") != CITATION_CONTEXT_VERSION:
        raise RuntimeError(
            f"Recommendation citation context mismatch: {run_dir.name}"
        )
    if validation.get("validator_version") != VALIDATOR_VERSION:
        raise RuntimeError(
            f"Recommendation validator mismatch: {run_dir.name}"
        )
    if recommendation.get("resolved_modeling_request") != request:
        raise RuntimeError(
            f"Resolved request does not match persisted request: {run_dir.name}"
        )
    if int(validation.get("citation_count") or 0) != len(
        recommendation.get("citations") or ()
    ):
        raise RuntimeError(
            f"Recommendation citation audit mismatch: {run_dir.name}"
        )
    if require_human_base_pk and (
        request.get("objective") != "base_pk"
        or request.get("species") != "human"
    ):
        raise RuntimeError(
            f"Serving batch requires human/base_pk requests: {run_dir.name}"
        )
    compound = recommendation.get("compound") or {}
    if str(compound.get("preferred_name") or "").casefold() != str(
        request.get("compound") or ""
    ).casefold():
        raise RuntimeError(
            f"Request/recommendation compound mismatch: {run_dir.name}"
        )
    required_runtime_hashes = {
        "graph_manifest",
        "subgraph_manifest",
        "embedding_index_manifest",
        "benchmark_manifest",
        "gnn_comparison",
        "gnn_index_manifest",
    }
    runtime_hashes = manifest.get("runtime_artifact_hashes") or {}
    if set(runtime_hashes) != required_runtime_hashes or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in runtime_hashes.values()
    ):
        raise RuntimeError(
            f"Runtime artifact chain is incomplete: {run_dir.name}"
        )
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "request": request,
        "recommendation": recommendation,
        "validation": validation,
        "request_sha256": _request_hash(request),
        "recommendation_sha256": sha256_file(paths["recommendation"]),
    }


def _validated_online_row(run_dir: Path) -> dict[str, Any]:
    return _validated_run_artifacts(
        run_dir,
        require_human_base_pk=True,
    )


def replace_kb_component_recommendations(
    *,
    runs_dir: Path,
    database_url: str,
) -> dict[str, Any]:
    rows = [
        _validated_online_row(run_dir)
        for run_dir in sorted(runs_dir.iterdir())
        if run_dir.is_dir()
    ]
    if not rows:
        raise RuntimeError("The component recommendation serving batch is empty.")
    gate_pairs = {
        (
            str(row["manifest"]["gate5_manifest_sha256"]),
            str(row["manifest"]["gate6_manifest_sha256"]),
        )
        for row in rows
    }
    if len(gate_pairs) != 1:
        raise RuntimeError("Serving batch mixes different Gate 5/6 models.")
    runtime_chains = {
        canonical_json(row["manifest"]["runtime_artifact_hashes"])
        for row in rows
    }
    if len(runtime_chains) != 1:
        raise RuntimeError("Serving batch mixes different runtime artifact chains.")
    compound_names = [
        str(row["recommendation"]["compound"]["preferred_name"]).casefold()
        for row in rows
    ]
    if len(compound_names) != len(set(compound_names)):
        raise RuntimeError("Serving batch contains duplicate compounds.")
    request_hashes = [str(row["request_sha256"]) for row in rows]
    if len(request_hashes) != len(set(request_hashes)):
        raise RuntimeError("Serving batch contains duplicate requests.")

    gate5_hash, gate6_hash = next(iter(gate_pairs))
    engine = create_db_engine(database_url)
    with engine.begin() as connection:
        model_run = connection.execute(
            text(
                """
                SELECT recommendation_model_run_id
                FROM recommendation_model_run
                WHERE gate5_manifest_sha256 = :gate5_hash
                  AND gate6_manifest_sha256 = :gate6_hash
                  AND engineering_acceptance IS TRUE
                """
            ),
            {"gate5_hash": gate5_hash, "gate6_hash": gate6_hash},
        ).scalar_one_or_none()
        if model_run is None:
            raise RuntimeError(
                "The accepted Gate 5/6 model run must be imported first."
            )
        expected_compounds = {
            str(row["compound_id"]): str(row["preferred_name"])
            for row in connection.execute(
                text(
                    """
                    SELECT compound_id::text, preferred_name::text
                    FROM compound_identity
                    """
                )
            ).mappings()
        }
        actual_compounds = {
            str(row["recommendation"]["compound"]["compound_id"])
            .removeprefix("Compound:"): str(
                row["recommendation"]["compound"]["preferred_name"]
            )
            for row in rows
        }
        if {
            key: value.casefold() for key, value in actual_compounds.items()
        } != {
            key: value.casefold() for key, value in expected_compounds.items()
        }:
            raise RuntimeError(
                "Serving batch compounds do not exactly match compound_identity."
            )
        connection.execute(
            text(
                """
                DELETE FROM recommendation_result
                WHERE recommendation_model_run_id = :model_run
                """
            ),
            {"model_run": model_run},
        )
        result_ids = []
        for row in rows:
            recommendation = row["recommendation"]
            compound = recommendation["compound"]
            result_id = stable_id(
                "recommendation-result",
                str(model_run),
                str(row["manifest"]["request_id"]),
                str(row["recommendation_sha256"]),
            )
            result_ids.append(result_id)
            connection.execute(
                text(
                    """
                    INSERT INTO recommendation_result (
                        recommendation_result_id,
                        recommendation_model_run_id,
                        example_id,
                        compound_node_id,
                        compound_name,
                        objective,
                        status,
                        request_sha256,
                        recommendation_sha256,
                        overall_confidence,
                        citation_count,
                        evidence_gap_count,
                        request_json,
                        recommendation_json,
                        validation_json
                    ) VALUES (
                        :result_id, :model_run, :example_id,
                        :compound_node_id, :compound_name, 'base_pk',
                        'recommendation_validated', :request_sha256,
                        :recommendation_sha256, :overall_confidence,
                        :citation_count, :evidence_gap_count,
                        CAST(:request_json AS jsonb),
                        CAST(:recommendation_json AS jsonb),
                        CAST(:validation_json AS jsonb)
                    )
                    """
                ),
                {
                    "result_id": result_id,
                    "model_run": model_run,
                    "example_id": str(row["manifest"]["request_id"]),
                    "compound_node_id": str(compound["compound_id"]),
                    "compound_name": str(compound["preferred_name"]),
                    "request_sha256": str(row["request_sha256"]),
                    "recommendation_sha256": str(
                        row["recommendation_sha256"]
                    ),
                    "overall_confidence": float(
                        recommendation.get("overall_confidence") or 0.0
                    ),
                    "citation_count": len(
                        recommendation.get("citations") or ()
                    ),
                    "evidence_gap_count": len(
                        recommendation.get("evidence_gaps") or ()
                    ),
                    "request_json": json.dumps(
                        row["request"], ensure_ascii=False
                    ),
                    "recommendation_json": json.dumps(
                        recommendation, ensure_ascii=False
                    ),
                    "validation_json": json.dumps(
                        row["validation"], ensure_ascii=False
                    ),
                },
            )
        batch_manifest = {
            "serving_policy": "one_validated_human_base_pk_result_per_kb_compound",
            "run_count": len(rows),
            "compounds": sorted(actual_compounds.values(), key=str.casefold),
            "runtime_artifact_hashes": rows[0]["manifest"][
                "runtime_artifact_hashes"
            ],
            "recommendation_result_ids": sorted(result_ids),
        }
        connection.execute(
            text(
                """
                UPDATE recommendation_model_run
                SET metadata_json = metadata_json || jsonb_build_object(
                    'serving_batch', CAST(:batch_manifest AS jsonb)
                )
                WHERE recommendation_model_run_id = :model_run
                """
            ),
            {
                "model_run": model_run,
                "batch_manifest": json.dumps(
                    batch_manifest, ensure_ascii=False
                ),
            },
        )
        persisted = connection.execute(
            text(
                """
                SELECT COUNT(*), COUNT(DISTINCT compound_node_id)
                FROM recommendation_result
                WHERE recommendation_model_run_id = :model_run
                  AND objective = 'base_pk'
                  AND status = 'recommendation_validated'
                """
            ),
            {"model_run": model_run},
        ).one()
        if tuple(map(int, persisted)) != (len(rows), len(expected_compounds)):
            raise RuntimeError("Serving batch database verification failed.")
    return {
        "recommendation_model_run_id": str(model_run),
        "imported_recommendation_count": len(rows),
        "compound_count": len(expected_compounds),
        "compounds": sorted(actual_compounds.values(), key=str.casefold),
        "gate5_manifest_sha256": gate5_hash,
        "gate6_manifest_sha256": gate6_hash,
    }


def import_recommendation_runs(
    *,
    gate5_dir: Path,
    gate6_dir: Path,
    database_url: str,
    kg_run_id: str,
    evidence_run_id: str,
) -> dict[str, Any]:
    gate5_path = gate5_dir / "gate5_manifest.json"
    gate6_path = gate6_dir / "gate6_manifest.json"
    gate5 = json.loads(gate5_path.read_text(encoding="utf-8"))
    gate6 = json.loads(gate6_path.read_text(encoding="utf-8"))
    if gate5.get("accepted") is not True:
        raise RuntimeError("Recommendation import requires accepted Gate 5.")
    if gate6.get("engineering_acceptance") is not True:
        raise RuntimeError(
            "Recommendation import requires Gate 6 engineering acceptance."
        )
    gate5_hash = sha256_file(gate5_path)
    gate6_hash = sha256_file(gate6_path)
    if (gate6.get("input_hashes") or {}).get("gate5_manifest.json") != gate5_hash:
        raise RuntimeError("Gate 6 was not evaluated from this Gate 5 manifest.")
    llm = gate5.get("llm") or {}
    projector = gate5.get("projector") or {}
    model_run_id = stable_id(
        "recommendation-model-run",
        gate5_hash,
        gate6_hash,
    )
    result_rows = []
    for run_dir in sorted((gate5_dir / "recommendation_runs").iterdir()):
        if not run_dir.is_dir():
            continue
        final_path = run_dir / "recommendation_final.json"
        validation_path = run_dir / "recommendation_validation.json"
        request_path = run_dir / "recommendation_request.json"
        if not all(
            path.exists()
            for path in (final_path, validation_path, request_path)
        ):
            raise RuntimeError(f"Incomplete recommendation run: {run_dir}")
        recommendation = json.loads(final_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if recommendation.get("status") != "recommendation_validated":
            continue
        if validation.get("status") != "recommendation_validated":
            raise RuntimeError(
                f"Final recommendation has non-validated audit: {run_dir.name}"
            )
        if recommendation.get("generated_from_auto_validated_only") is not True:
            raise RuntimeError(
                f"Recommendation is not auto-validated-only: {run_dir.name}"
            )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        recommendation_hash = sha256_file(final_path)
        compound = recommendation.get("compound") or {}
        result_rows.append(
            {
                "recommendation_result_id": stable_id(
                    "recommendation-result",
                    model_run_id,
                    run_dir.name,
                ),
                "recommendation_model_run_id": model_run_id,
                "example_id": run_dir.name,
                "compound_node_id": str(compound["compound_id"]),
                "compound_name": str(compound["preferred_name"]),
                "objective": str(request["objective"]),
                "status": "recommendation_validated",
                "request_sha256": _request_hash(request),
                "recommendation_sha256": recommendation_hash,
                "overall_confidence": float(
                    recommendation.get("overall_confidence") or 0.0
                ),
                "citation_count": len(
                    recommendation.get("citations") or ()
                ),
                "evidence_gap_count": len(
                    recommendation.get("evidence_gaps") or ()
                ),
                "request_json": json.dumps(request, ensure_ascii=False),
                "recommendation_json": json.dumps(
                    recommendation,
                    ensure_ascii=False,
                ),
                "validation_json": json.dumps(
                    validation,
                    ensure_ascii=False,
                ),
            }
        )
    expected_count = int(
        ((gate5.get("artifacts") or {}).get("validated_recommendations") or {}).get(
            "count"
        )
        or 0
    )
    if len(result_rows) != expected_count:
        raise RuntimeError(
            f"Validated recommendation count mismatch: "
            f"{len(result_rows)} != {expected_count}"
        )

    engine = create_db_engine(database_url)
    with engine.begin() as connection:
        for statement in RECOMMENDATION_TABLE_STATEMENTS:
            connection.execute(text(statement))
        existing = connection.execute(
            text(
                """
                SELECT gate5_manifest_sha256, gate6_manifest_sha256
                FROM recommendation_model_run
                WHERE recommendation_model_run_id = :run_id
                """
            ),
            {"run_id": model_run_id},
        ).mappings().first()
        if existing and (
            existing["gate5_manifest_sha256"] != gate5_hash
            or existing["gate6_manifest_sha256"] != gate6_hash
        ):
            raise RuntimeError(
                "Existing recommendation model run has different manifests."
            )
        connection.execute(
            text(
                """
                INSERT INTO recommendation_model_run (
                    recommendation_model_run_id,
                    gate5_manifest_sha256,
                    gate6_manifest_sha256,
                    kg_run_id,
                    evidence_run_id,
                    llm_model,
                    llm_revision,
                    llm_dtype,
                    projector_checkpoint_sha256,
                    benchmark_level,
                    engineering_acceptance,
                    generalization_claim_eligible,
                    gold_claim,
                    metadata_json
                ) VALUES (
                    :recommendation_model_run_id,
                    :gate5_manifest_sha256,
                    :gate6_manifest_sha256,
                    :kg_run_id,
                    :evidence_run_id,
                    :llm_model,
                    :llm_revision,
                    :llm_dtype,
                    :projector_checkpoint_sha256,
                    'silver',
                    :engineering_acceptance,
                    :generalization_claim_eligible,
                    FALSE,
                    CAST(:metadata_json AS JSONB)
                )
                ON CONFLICT (recommendation_model_run_id) DO NOTHING
                """
            ),
            {
                "recommendation_model_run_id": model_run_id,
                "gate5_manifest_sha256": gate5_hash,
                "gate6_manifest_sha256": gate6_hash,
                "kg_run_id": kg_run_id,
                "evidence_run_id": evidence_run_id,
                "llm_model": str(llm["model"]),
                "llm_revision": str(llm["revision"]),
                "llm_dtype": str(llm["dtype"]),
                "projector_checkpoint_sha256": str(
                    projector["checkpoint_sha256"]
                ),
                "engineering_acceptance": bool(
                    gate6["engineering_acceptance"]
                ),
                "generalization_claim_eligible": bool(
                    gate6["generalization_claim_eligible"]
                ),
                "metadata_json": json.dumps(
                    {
                        "gate5_manifest": gate5,
                        "gate6_manifest": gate6,
                    },
                    ensure_ascii=False,
                ),
            },
        )
        for row in result_rows:
            existing_result = connection.execute(
                text(
                    """
                    SELECT recommendation_sha256
                    FROM recommendation_result
                    WHERE recommendation_result_id = :result_id
                    """
                ),
                {"result_id": row["recommendation_result_id"]},
            ).scalar_one_or_none()
            if (
                existing_result is not None
                and existing_result != row["recommendation_sha256"]
            ):
                raise RuntimeError(
                    "Existing recommendation result has different content: "
                    f"{row['example_id']}"
                )
            connection.execute(
                text(
                    """
                    INSERT INTO recommendation_result (
                        recommendation_result_id,
                        recommendation_model_run_id,
                        example_id,
                        compound_node_id,
                        compound_name,
                        objective,
                        status,
                        request_sha256,
                        recommendation_sha256,
                        overall_confidence,
                        citation_count,
                        evidence_gap_count,
                        request_json,
                        recommendation_json,
                        validation_json
                    ) VALUES (
                        :recommendation_result_id,
                        :recommendation_model_run_id,
                        :example_id,
                        :compound_node_id,
                        :compound_name,
                        :objective,
                        :status,
                        :request_sha256,
                        :recommendation_sha256,
                        :overall_confidence,
                        :citation_count,
                        :evidence_gap_count,
                        CAST(:request_json AS JSONB),
                        CAST(:recommendation_json AS JSONB),
                        CAST(:validation_json AS JSONB)
                    )
                    ON CONFLICT (recommendation_result_id) DO NOTHING
                    """
                ),
                row,
            )
        persisted_result_ids = {
            str(value)
            for value in connection.execute(
                text(
                    """
                    SELECT recommendation_result_id
                    FROM recommendation_result
                    WHERE recommendation_model_run_id = :run_id
                    """
                ),
                {"run_id": model_run_id},
            ).scalars()
        }
    missing_result_ids = _missing_expected_result_ids(
        result_rows,
        persisted_result_ids,
    )
    if missing_result_ids:
        raise RuntimeError(
            "Database is missing imported recommendation results: "
            + ", ".join(missing_result_ids)
        )
    return {
        "recommendation_model_run_id": model_run_id,
        "imported_recommendation_count": len(result_rows),
        "gate5_manifest_sha256": gate5_hash,
        "gate6_manifest_sha256": gate6_hash,
        "engineering_acceptance": True,
        "generalization_claim_eligible": bool(
            gate6["generalization_claim_eligible"]
        ),
    }


def import_online_recommendation_run(
    *,
    run_dir: Path,
    database_url: str,
) -> dict[str, Any]:
    validated = _validated_run_artifacts(
        run_dir,
        require_human_base_pk=False,
    )
    manifest = validated["manifest"]
    request = validated["request"]
    recommendation = validated["recommendation"]
    validation = validated["validation"]
    final_path = run_dir / "recommendation_final.json"
    gate5_hash = str(manifest["gate5_manifest_sha256"])
    gate6_hash = str(manifest["gate6_manifest_sha256"])
    engine = create_db_engine(database_url)
    with engine.begin() as connection:
        model_run = connection.execute(
            text(
                """
                SELECT recommendation_model_run_id
                FROM recommendation_model_run
                WHERE gate5_manifest_sha256 = :gate5_hash
                  AND gate6_manifest_sha256 = :gate6_hash
                  AND engineering_acceptance IS TRUE
                """
            ),
            {"gate5_hash": gate5_hash, "gate6_hash": gate6_hash},
        ).scalar_one_or_none()
        if model_run is None:
            raise RuntimeError(
                "The accepted Gate 5/6 model run must be imported first."
            )
        request_hash = _request_hash(request)
        # Duplicate callers can finish the same GPU task together. Serialize the
        # compare-and-insert transaction, preserving the conflicting-output check.
        import_lock = int.from_bytes(
            hashlib.sha256(f"{model_run}:{request_hash}".encode()).digest()[:8],
            "big", signed=True,
        )
        connection.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": import_lock},
        )
        existing = connection.execute(
            text(
                """
                SELECT recommendation_result_id, recommendation_sha256
                FROM recommendation_result
                WHERE recommendation_model_run_id = :model_run
                  AND request_sha256 = :request_hash
                """
            ),
            {
                "model_run": model_run,
                "request_hash": request_hash,
            },
        ).mappings().first()
        recommendation_hash = str(validated["recommendation_sha256"])
        if existing:
            if existing["recommendation_sha256"] != recommendation_hash:
                raise RuntimeError(
                    "This ModelingRequest already has different validated output."
                )
            return {
                "recommendation_model_run_id": model_run,
                "recommendation_result_id": existing[
                    "recommendation_result_id"
                ],
                "already_present": True,
            }
        compound = recommendation["compound"]
        result_id = stable_id(
            "recommendation-result",
            str(model_run),
            str(manifest["request_id"]),
        )
        connection.execute(
            text(
                """
                INSERT INTO recommendation_result (
                    recommendation_result_id,
                    recommendation_model_run_id,
                    example_id,
                    compound_node_id,
                    compound_name,
                    objective,
                    status,
                    request_sha256,
                    recommendation_sha256,
                    overall_confidence,
                    citation_count,
                    evidence_gap_count,
                    request_json,
                    recommendation_json,
                    validation_json
                ) VALUES (
                    :result_id,
                    :model_run,
                    :example_id,
                    :compound_node_id,
                    :compound_name,
                    :objective,
                    'recommendation_validated',
                    :request_sha256,
                    :recommendation_sha256,
                    :overall_confidence,
                    :citation_count,
                    :evidence_gap_count,
                    CAST(:request_json AS JSONB),
                    CAST(:recommendation_json AS JSONB),
                    CAST(:validation_json AS JSONB)
                )
                """
            ),
            {
                "result_id": result_id,
                "model_run": model_run,
                "example_id": str(manifest["request_id"]),
                "compound_node_id": str(compound["compound_id"]),
                "compound_name": str(compound["preferred_name"]),
                "objective": str(request["objective"]),
                "request_sha256": request_hash,
                "recommendation_sha256": recommendation_hash,
                "overall_confidence": float(
                    recommendation.get("overall_confidence") or 0.0
                ),
                "citation_count": len(
                    recommendation.get("citations") or ()
                ),
                "evidence_gap_count": len(
                    recommendation.get("evidence_gaps") or ()
                ),
                "request_json": json.dumps(request, ensure_ascii=False),
                "recommendation_json": json.dumps(
                    recommendation,
                    ensure_ascii=False,
                ),
                "validation_json": json.dumps(
                    validation,
                    ensure_ascii=False,
                ),
            },
        )
    return {
        "recommendation_model_run_id": model_run,
        "recommendation_result_id": result_id,
        "already_present": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import accepted Gate 5/6 PBPK recommendation runs."
    )
    parser.add_argument("--gate5-dir", type=Path, required=True)
    parser.add_argument("--gate6-dir", type=Path, required=True)
    parser.add_argument(
        "--database-url",
        default=get_settings().database_url,
    )
    parser.add_argument("--kg-run-id", required=True)
    parser.add_argument("--evidence-run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = import_recommendation_runs(
        gate5_dir=args.gate5_dir,
        gate6_dir=args.gate6_dir,
        database_url=args.database_url,
        kg_run_id=args.kg_run_id,
        evidence_run_id=args.evidence_run_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
