from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import pytest

from pbpk_kb.processing.raw_processor import RawProcessingPipeline, _extension, _file_family


def write_accepted_run_scaffold(run_root: Path, *, snapshot_root: Path | None = None) -> None:
    (run_root / "RUN_STATUS.json").write_text(
        json.dumps({"status": "completed", "accepted_as_final_result": True}), encoding="utf-8"
    )
    (run_root / "summary.json").write_text("{}", encoding="utf-8")
    (run_root / "download_plan.json").write_text("{}", encoding="utf-8")
    (run_root / "failures.jsonl").write_text("", encoding="utf-8")
    finalization = run_root / "finalization"
    finalization.mkdir()
    (finalization / "FINAL_ACQUISITION_RESULT.json").write_text(
        json.dumps(
            {
                "accepted_as_complete_raw_content": True,
                "osp_pbpk_model_library_snapshot": {
                    "snapshot_root": str(snapshot_root) if snapshot_root else None,
                },
            }
        ),
        encoding="utf-8",
    )
    (finalization / "FINAL_INTEGRITY_AUDIT.json").write_text(
        json.dumps({"status": "passed", "issue_count": 0}), encoding="utf-8"
    )
    (finalization / "failure_resolutions.jsonl").write_text("", encoding="utf-8")


def test_raw_processing_keeps_provenance_for_json_asset(tmp_path: Path) -> None:
    run_root = tmp_path / "run_20260629T105513Z"
    run_root.mkdir()
    write_accepted_run_scaffold(run_root)
    raw = run_root / "pubchem.json"
    payload = {
        "PropertyTable": {
            "Properties": [
                {
                    "MolecularFormula": "C18H13ClFN3",
                    "MolecularWeight": 325.8,
                    "InChIKey": "example",
                    "XLogP": 3.2,
                }
            ]
        }
    }
    raw.write_text(json.dumps(payload), encoding="utf-8")
    checksum = __import__("hashlib").sha256(raw.read_bytes()).hexdigest()
    asset = {
        "asset_id": "asset:one",
        "source_id": "pubchem",
        "compound": "midazolam",
        "group": "compound_api",
        "storage_path": str(raw),
        "url": "https://example.test/pubchem",
        "downloaded_at": "2026-06-29T00:00:00Z",
        "checksum": checksum,
        "content_type": "application/json",
        "request_id": "pubchem:properties",
        "request_key": "properties_by_name",
    }
    (run_root / "assets.jsonl").write_text(json.dumps(asset) + "\n", encoding="utf-8")

    pipeline = RawProcessingPipeline(run_root=run_root, output_root=tmp_path / "data_process", llm_mode="off", embedding_mode="off")
    summary = pipeline.process()

    assert summary["raw_asset_count"] == 1
    assert summary["evidence_candidate_count"] >= 1
    evidence_path = tmp_path / "data_process" / run_root.name / "04_evidence_candidates" / "evidence_candidates.jsonl"
    evidence = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert evidence
    assert all(row["raw_asset_id"] == "asset:one" for row in evidence)
    assert all(row["checksum"] == checksum for row in evidence)
    assert all(row["source_text"] for row in evidence)
    assert all(row["source_location"] for row in evidence)


# Keyword-only section matches are retrieval/LLM hints, not evidence rows.
def test_section_matches_are_pointers_not_evidence(tmp_path: Path) -> None:
    from pbpk_kb.processing.raw_processor import RawAsset, RawProcessingPipeline

    run_root = tmp_path / "run_x"
    run_root.mkdir()
    pipe = RawProcessingPipeline(run_root=run_root, output_root=tmp_path / "out", llm_mode="off", embedding_mode="off")
    base = dict(
        run_id="run_x", raw_asset_id="asset:1", compound="midazolam", group=None,
        storage_path="/x", source_url="http://x", downloaded_at=None, checksum="c",
        content_type=None, file_extension=".xml", file_family="xml", file_size=10,
        request_id=None, request_key=None, asset_metadata_json={},
    )
    asset = RawAsset(source_id="dailymed", **base)
    doc = {"parsed_document_id": "pd1"}
    prose = "In healthy adult volunteers the drug was administered orally. " * 5
    chunk = {
        "chunk_id": "ch1",
        "source_location": "loc",
        "chunk_text": prose + "Midazolam is primarily metabolized by CYP3A4 in the liver.",
    }
    pipe._extract_rule_evidence_from_chunk(asset, doc, chunk)

    ev_path = pipe.output_paths.path("evidence_candidates.jsonl", for_write=True)
    ptr_path = pipe.output_paths.path("evidence_pointers.jsonl", for_write=True)
    evidence = [json.loads(l) for l in ev_path.read_text().splitlines()] if ev_path.exists() else []
    pointers = [json.loads(l) for l in ptr_path.read_text().splitlines()] if ptr_path.exists() else []

    # No section_candidate rows leak into the evidence layer.
    assert all(row["extraction_method"] != "section_candidate" for row in evidence)
    # The ADME keyword match becomes a pointer instead.
    assert any(p["topic"] == "adme" and p["matched_term"] == "CYP3A4" for p in pointers)
    assert all(p["evidence_tier"] == "pointer" for p in pointers)
    # Any real evidence rows carry an explicit tier.
    assert all(row.get("evidence_tier") in {"quantitative", "qualitative"} for row in evidence)


def test_json_processing_chunks_only_source_specific_free_text_fields(tmp_path: Path) -> None:
    run_root = tmp_path / "run_20260629T105513Z"
    run_root.mkdir()
    write_accepted_run_scaffold(run_root)
    raw = run_root / "clinicaltrials.json"
    payload = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000000"},
            "statusModule": {"studyFirstSubmitDate": "2026-01-01"},
            "descriptionModule": {
                "briefSummary": "This study evaluates the pharmacokinetics of midazolam in patients with renal impairment after oral administration.",
            },
        }
    }
    raw.write_text(json.dumps(payload), encoding="utf-8")
    checksum = __import__("hashlib").sha256(raw.read_bytes()).hexdigest()
    asset = {
        "asset_id": "asset:ct",
        "source_id": "clinicaltrials",
        "compound": "midazolam",
        "group": "compound_api",
        "storage_path": str(raw),
        "url": "https://example.test/clinicaltrials",
        "downloaded_at": "2026-06-29T00:00:00Z",
        "checksum": checksum,
        "content_type": "application/json",
        "request_id": "clinicaltrials:studies",
        "request_key": "studies_by_compound",
    }
    (run_root / "assets.jsonl").write_text(json.dumps(asset) + "\n", encoding="utf-8")

    pipeline = RawProcessingPipeline(run_root=run_root, output_root=tmp_path / "data_process", llm_mode="off", embedding_mode="off")
    pipeline.process()

    chunks_path = tmp_path / "data_process" / run_root.name / "03_chunks_and_embeddings" / "text_chunks.jsonl"
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines()]
    assert chunks
    assert all("#json_path=$#chunk=" not in row["source_location"] for row in chunks)
    assert any("briefSummary" in row["source_location"] for row in chunks)
    assert all("studyFirstSubmitDate" not in row["chunk_text"] for row in chunks)


@pytest.mark.parametrize("overlap", [False, True])
def test_inventory_merges_final_snapshot_sniffs_bin_and_resolves_historical_failures(tmp_path: Path, overlap) -> None:
    run_root = tmp_path / "run_current"
    run_root.mkdir()
    snapshot_root = tmp_path / "snapshot_final"
    snapshot_root.mkdir()
    write_accepted_run_scaffold(run_root, snapshot_root=snapshot_root)
    (snapshot_root / "RUN_STATUS.json").write_text(
        json.dumps({"status": "completed", "accepted_as_final_result": True}), encoding="utf-8"
    )

    primary = run_root / "pubchem_payload.bin"
    primary.write_text(json.dumps({"CID": 123}), encoding="utf-8")
    shared = snapshot_root / "model.pksim5"
    shared.write_bytes(b"PK-Sim model bytes")
    primary_asset = {
        "asset_id": "asset:primary",
        "source_id": "pubchem",
        "compound": "midazolam",
        "storage_path": str(primary),
        "url": "https://example.test/payload",
        "checksum": sha256(primary.read_bytes()).hexdigest(),
        "byte_count": primary.stat().st_size,
        "request_id": "pubchem:one",
    }
    shared_asset = {
        "asset_id": "asset:shared",
        "source_id": "osp_pbpk_model_library",
        "compound": None,
        "storage_path": str(shared),
        "url": "https://example.test/model.pksim5",
        "checksum": sha256(shared.read_bytes()).hexdigest(),
        "byte_count": shared.stat().st_size,
        "request_id": "osp:model",
    }
    (run_root / "assets.jsonl").write_text(json.dumps(primary_asset) + "\n", encoding="utf-8")
    if overlap:
        with (run_root / "assets.jsonl").open("a") as handle:
            handle.write(json.dumps(shared_asset) + "\n")
    (snapshot_root / "assets.jsonl").write_text(json.dumps(shared_asset) + "\n", encoding="utf-8")

    failure = {
        "request_id": "pmc:failed-once",
        "source_id": "pmc_oa",
        "compound": "midazolam",
        "request_key": "oa_package",
        "url": "https://example.test/oa",
        "error": "temporary error",
    }
    (run_root / "failures.jsonl").write_text(
        json.dumps(failure) + "\n" + json.dumps(failure) + "\n", encoding="utf-8"
    )
    resolution = {
        "resolution_id": "resolution:one",
        "original_failure_request_id": "pmc:failed-once",
        "status": "resolved",
        "resolution_method": "retry_succeeded",
    }
    (run_root / "finalization" / "failure_resolutions.jsonl").write_text(
        json.dumps(resolution) + "\n", encoding="utf-8"
    )

    pipeline = RawProcessingPipeline(
        run_root=run_root,
        output_root=tmp_path / "processed",
        checksum_mode="verify",
        llm_mode="off",
        embedding_mode="off",
        stage="inventory",
    )
    summary = pipeline.process()

    assert summary["accepted"] is True
    assert summary["raw_asset_count"] == 2
    assert summary["input_ledger_counts"] == ({"primary_run": 2} if overlap else {"final_shared_snapshot": 1, "primary_run": 1})
    assert summary["acquisition_terminal_status_counts"] == {"resolved": 1}
    inventory_path = tmp_path / "processed" / run_root.name / "01_raw_inventory" / "raw_inventory.jsonl"
    inventory = [json.loads(line) for line in inventory_path.read_text(encoding="utf-8").splitlines()]
    assert {row["file_family"] for row in inventory} == {"json", "repo"}
    failure_path = tmp_path / "processed" / run_root.name / "00_run_manifest" / "processing_failures.jsonl"
    assert failure_path.read_text(encoding="utf-8") == ""
    status_path = tmp_path / "processed" / run_root.name / "00_run_manifest" / "acquisition_failure_status.jsonl"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["historical_failure_record_count"] == 2
    assert status["terminal_status"] == "resolved"


def test_inventory_file_family_classification_for_pmc_supplements(tmp_path: Path) -> None:
    samples = {
        "figure.jpg": (b"\xff\xd8\xff\xe0" + b"image", "image"),
        "video.mp4": (b"\x00\x00\x00\x18ftypmp42" + b"media", "media"),
        "dataset.unk": (b"SQLite format 3\x00" + b"database", "database"),
        "analysis.pzf": (b"PCFFGRA4" + b"prism", "specialized_binary"),
        "supplement.rar": (b"Rar!\x1a\x07\x00" + b"archive", "archive"),
        "pubmed.fcgi": (b'{"header":{"type":"esearch"}}', "json"),
        "supplement.docx": (b"PK\x03\x04" + b"ooxml", "document"),
        "legacy.doc": (bytes.fromhex("d0cf11e0a1b11ae1") + b"word", "document"),
        "slides.pptx": (b"PK\x03\x04" + b"ooxml", "presentation"),
        "legacy.ppt": (bytes.fromhex("d0cf11e0a1b11ae1") + b"powerpoint", "presentation"),
    }

    for file_name, (payload, expected_family) in samples.items():
        path = tmp_path / file_name
        path.write_bytes(payload)
        extension = _extension(path, None, None)
        assert _file_family(path, extension, None, "pmc_oa") == expected_family
