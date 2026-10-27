from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID

from pbpk_kb.recommendation.requirement_profile_sources import (
    build_profile_bundles,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_locked_silver_source_is_mapped_with_real_provenance(
    tmp_path: Path,
) -> None:
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    evidence_id = "68b102ee-6b5a-45e2-8d59-3a245151e92d"
    compound_id = "Compound:PUBLIC:4192"
    _write_jsonl(
        graph_dir / "kg_nodes.jsonl",
        [
            {
                "node_id": compound_id,
                "node_type": "Compound",
                "preferred_name": "Midazolam",
                "standard_identifiers": {
                    "PUBCHEM.COMPOUND": "4192",
                },
            },
            {
                "node_id": evidence_id,
                "node_type": "Evidence",
                "preferred_name": "clinical_pk:clearance",
            },
            {
                "node_id": "PKParameter:clearance",
                "node_type": "PKParameter",
                "preferred_name": "intrinsic clearance",
            },
        ],
    )
    _write_jsonl(
        graph_dir / "kg_edges.jsonl",
        [
            {
                "edge_id": "edge-1",
                "subject_id": evidence_id,
                "predicate": "MEASURES",
                "object_id": "PKParameter:clearance",
                "provenance": {
                    "evidence_id": evidence_id,
                    "quality_status": "auto_validated",
                    "negation_status": "affirmed",
                    "source_id": "pubmed",
                    "source_location": "pubmed:123#abstract",
                    "supporting_text": "Intrinsic clearance was measured.",
                    "confidence": 0.91,
                },
            }
        ],
    )
    (graph_dir / "kg_extraction_manifest.json").write_text(
        '{"accepted":true}\n',
        encoding="utf-8",
    )

    silver_path = tmp_path / "recommendation_silver.jsonl"
    _write_jsonl(
        silver_path,
        [
            {
                "example_id": "silver:1",
                "compound_id": compound_id,
                "split": "test",
                "modeling_request": {
                    "compound": "midazolam",
                    "objective": "ddi_victim",
                    "perpetrator_drug": "rifampicin",
                },
                "target": {
                    "component_ids": ["metabolism"],
                    "evidence_ids": [evidence_id],
                    "gap_codes": ["missing_component:ddi"],
                },
            }
        ],
    )
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "accepted": True,
                "benchmark_level": "silver",
                "artifacts": {
                    "silver": {"sha256": _sha256(silver_path)}
                },
                "input_hashes": {
                    "graph_manifest": _sha256(
                        graph_dir / "kg_extraction_manifest.json"
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    bundles, manifest = build_profile_bundles(
        silver_path=silver_path,
        dataset_manifest_path=dataset_manifest,
        graph_dir=graph_dir,
    )

    assert manifest["bundle_count"] == 1
    assert manifest["benchmark_level"] == "silver"
    bundle = bundles[0]
    UUID(bundle["compound_identity"]["compound_id"])
    assert (
        bundle["compound_identity"]["identity_resolution_method"]
        == "evidence_kg_canonical_node"
    )
    component = bundle["component_recommendation"]["components"][0]
    assert component["component"] == "metabolism"
    assert component["status"] == "recommended"
    assert component["citations"][0]["source_id"] == "pubmed"
    assert (
        component["citations"][0]["source_text"]
        == "Intrinsic clearance was measured."
    )
