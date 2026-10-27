from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pbpk_kb.recommendation.gate6_evaluation import canonical_json, sha256_file
from pbpk_kb.recommendation.graph_prompt import (
    CITATION_CONTEXT_VERSION,
    GRAPH_PROMPT_VERSION,
    RECOMMENDATION_SCHEMA_VERSION,
    VALIDATOR_VERSION,
)
from pbpk_kb.storage.import_recommendation_runs import (
    _missing_expected_result_ids,
    _validated_online_row,
)


def test_batch_import_verification_allows_additional_online_results() -> None:
    expected_rows = [
        {"recommendation_result_id": "batch-1"},
        {"recommendation_result_id": "batch-2"},
    ]

    assert _missing_expected_result_ids(
        expected_rows,
        {"batch-1", "batch-2", "online-1"},
    ) == []


def test_batch_import_verification_reports_missing_expected_results() -> None:
    expected_rows = [
        {"recommendation_result_id": "batch-1"},
        {"recommendation_result_id": "batch-2"},
    ]

    assert _missing_expected_result_ids(expected_rows, {"batch-1"}) == [
        "batch-2"
    ]


def _write_validated_online_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "online-recommendation:test"
    run_dir.mkdir()
    request = {
        "compound": "caffeine",
        "objective": "base_pk",
        "species": "human",
    }
    recommendation = {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "citation_context_version": CITATION_CONTEXT_VERSION,
        "status": "recommendation_validated",
        "resolved_modeling_request": request,
        "compound": {
            "compound_id": "Compound:compound-id",
            "preferred_name": "caffeine",
        },
        "generated_from_auto_validated_only": True,
        "scientific_use_status": "requirements_evidence_only",
        "overall_confidence": 0.8,
        "citations": [],
        "evidence_gaps": [],
    }
    validation = {
        "validator_version": VALIDATOR_VERSION,
        "status": "recommendation_validated",
        "error_count": 0,
        "errors": [],
        "citation_count": 0,
    }
    files = {
        "recommendation_request.json": request,
        "recommendation_final.json": recommendation,
        "recommendation_validation.json": validation,
    }
    for name, value in files.items():
        (run_dir / name).write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    runtime_hashes = {
        key: hashlib.sha256(key.encode()).hexdigest()
        for key in (
            "graph_manifest",
            "subgraph_manifest",
            "embedding_index_manifest",
            "benchmark_manifest",
            "gnn_comparison",
            "gnn_index_manifest",
        )
    }
    manifest = {
        "assembly_version": GRAPH_PROMPT_VERSION,
        "recommendation_schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "citation_context_version": CITATION_CONTEXT_VERSION,
        "status": "recommendation_validated",
        "request_id": "online-recommendation:test",
        "gate5_manifest_sha256": "a" * 64,
        "gate6_manifest_sha256": "b" * 64,
        "runtime_artifact_hashes": runtime_hashes,
        "artifacts": {
            name: sha256_file(run_dir / name) for name in files
        },
    }
    (run_dir / "recommendation_run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_validated_online_row_checks_full_serving_contract(
    tmp_path: Path,
) -> None:
    run_dir = _write_validated_online_run(tmp_path)

    row = _validated_online_row(run_dir)

    assert row["request"]["compound"] == "caffeine"
    assert row["request_sha256"] == hashlib.sha256(
        canonical_json(row["request"]).encode("utf-8")
    ).hexdigest()


def test_validated_online_row_rejects_tampered_final_result(
    tmp_path: Path,
) -> None:
    run_dir = _write_validated_online_run(tmp_path)
    with (run_dir / "recommendation_final.json").open("a") as handle:
        handle.write("tampered")

    with pytest.raises(RuntimeError, match="Artifact checksum mismatch"):
        _validated_online_row(run_dir)
