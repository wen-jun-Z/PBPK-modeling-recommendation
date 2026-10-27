from __future__ import annotations

import json
import hashlib
from pathlib import Path

from pbpk_kb.recommendation.agent import PBPKRecommendationAgent
from pbpk_kb.recommendation.graph_store import RecommendationGraphStore
from pbpk_kb.recommendation.online_agent import (
    _resolve_projector_checkpoint,
)
from pbpk_kb.recommendation.schemas import RecommendationRequest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_online_projector_uses_manifest_checkpoint_path(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "training" / "projectors" / "seed_29.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"locked-projector")
    manifest = {
        "projector": {
            "selected_seed": 29,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
        }
    }

    assert _resolve_projector_checkpoint(
        tmp_path / "accepted-gate5",
        manifest,
    ) == checkpoint


def test_online_projector_resolves_repo_relative_manifest_path(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    gate5_dir = repository / "data" / "runs" / "accepted-gate5"
    gate5_dir.mkdir(parents=True)
    checkpoint = repository / "data" / "training" / "projectors" / "seed_29.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"locked-projector")
    manifest = {
        "projector": {
            "selected_seed": 29,
            "checkpoint_path": "data/training/projectors/seed_29.pt",
            "checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
        }
    }

    assert _resolve_projector_checkpoint(gate5_dir, manifest) == checkpoint


def quality_fixture(tmp_path: Path) -> Path:
    evidence = []
    for index, (kind, parameter, source) in enumerate([
        ("physicochemical", "MolecularWeight", "pubchem"),
        ("physicochemical", "XLogP", "pubchem"),
        ("adme", "absorption", "dailymed"),
        ("adme", "metabolism CYP3A4", "pubmed"),
        ("clinical_pk", "AUC", "pkdb"),
        ("clinical_pk", "clearance", "dailymed"),
        ("public_pbpk_model", "midazolam PBPK model", "osp_model_library"),
    ]):
        evidence.append({
            "evidence_id": f"e{index}",
            "compound_id": "c-midazolam",
            "compound_name": "midazolam",
            "quality_status": "auto_validated",
            "evidence_type": kind,
            "parameter": parameter,
            "source_id": source,
            "source_location": f"{source}:record-{index}",
            "source_text": f"Midazolam evidence for {parameter}.",
            "source_text_hash": f"hash-{index}",
            "extraction_confidence": 0.9,
            "validation_method": "deterministic_quality_gate_v1",
            "route": "oral" if kind in {"adme", "clinical_pk"} else None,
        })
    nodes = [
        {"node_type": "Compound", "node_id": "c-midazolam", "properties": {"name": "midazolam"}},
    ]
    edges = []
    for row in evidence:
        nodes.append({"node_type": "Evidence", "node_id": row["evidence_id"], "properties": {"quality_status": "auto_validated"}})
        edges.append({
            "from_type": "Compound",
            "from_id": "c-midazolam",
            "relationship_type": "HAS_EVIDENCE",
            "to_type": "Evidence",
            "to_id": row["evidence_id"],
            "properties": {
                "evidence_id": row["evidence_id"],
                "source_id": row["source_id"],
                "source_location": row["source_location"],
                "source_text_hash": row["source_text_hash"],
                "quality_status": "auto_validated",
                "validation_method": row["validation_method"],
            },
        })
    write_jsonl(tmp_path / "final_auto_validated_evidence.jsonl", evidence)
    write_jsonl(tmp_path / "kg_ready_nodes.jsonl", nodes)
    write_jsonl(tmp_path / "kg_ready_edges.jsonl", edges)
    return tmp_path


def test_recommendation_uses_only_grounded_evidence_and_reports_gaps(tmp_path: Path) -> None:
    store = RecommendationGraphStore(quality_fixture(tmp_path))

    result = PBPKRecommendationAgent(store).recommend(
        RecommendationRequest(compound="midazolam", objective="base_pk", route="oral")
    )

    assert result.compound_name == "midazolam"
    assert result.generated_from_auto_validated_only is True
    assert result.ranked_subgraphs
    cited = {
        citation.evidence_id
        for component in result.components
        for citation in component.citations
    }
    assert cited <= set(store.evidence_by_id)
    assert all(store.evidence_by_id[evidence_id]["quality_status"] == "auto_validated" for evidence_id in cited)
    assert any(component.component == "public_model_template" for component in result.components)
    assert any("distribution" in gap for gap in result.evidence_gaps)
    assert all("special_population" not in gap for gap in result.evidence_gaps)


def test_unknown_compound_is_rejected(tmp_path: Path) -> None:
    store = RecommendationGraphStore(quality_fixture(tmp_path))
    agent = PBPKRecommendationAgent(store)

    try:
        agent.recommend(RecommendationRequest(compound="unknown", objective="base_pk"))
    except LookupError as exc:
        assert "not present" in str(exc)
    else:
        raise AssertionError("Unknown compound must not be recommended from")


def test_modeling_request_reports_ddi_and_regimen_gaps() -> None:
    request = RecommendationRequest(
        compound="midazolam",
        objective="ddi_victim",
        dose=5,
        dose_unit="mg",
    )

    assert "DDI perpetrator drug is missing." in request.request_gaps()
    assert "Dose regimen is incomplete: missing route, frequency." in request.request_gaps()


def test_modeling_request_maps_legacy_interacting_drug_to_explicit_role() -> None:
    request = RecommendationRequest(
        compound="midazolam",
        objective="ddi_victim",
        interacting_drug="ketoconazole",
    )

    assert request.perpetrator_drug == "ketoconazole"
    assert "DDI perpetrator drug is missing." not in request.request_gaps()
