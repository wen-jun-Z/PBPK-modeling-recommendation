from __future__ import annotations

import json
from pathlib import Path

from pbpk_kb.processing.rag_index_stage import RagIndexStage, build_embedding_text


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def make_stage(tmp_path: Path) -> tuple[RagIndexStage, Path, Path]:
    raw = tmp_path / "raw" / "run_x"
    final = tmp_path / "final" / "run_x"
    write_json(
        final / "00_run_manifest" / "pipeline_state.json",
        {
            "run_id": "run_x",
            "current_stage": "reviewable_evidence_ready",
            "stages": {
                "evidence_layers": {"status": "completed", "accepted": True},
                "rag_index": {"status": "not_started"},
            },
        },
    )
    stage = RagIndexStage(raw_processing_dir=raw, final_evidence_dir=final, reset=False)
    return stage, raw, final


def test_prepare_builds_allowlisted_context_and_structured_embedding_text(tmp_path: Path) -> None:
    stage, raw, final = make_stage(tmp_path)
    reviewable = [
        {
            "run_id": "run_x",
            "evidence_id": "e1",
            "compound_id": "c1",
            "compound_name": "caffeine",
            "evidence_type": "physicochemical",
            "parameter": "MolecularWeight",
            "value": 194.19,
            "unit": "g/mol",
            "source_id": "pubchem",
            "source_location": "pubchem:file#json_path=$.weight",
            "source_url": "https://example.test/pubchem",
            "storage_path": "/raw/pubchem.json",
            "checksum": "raw-checksum",
            "source_text": "194.19",
            "source_text_hash": "source-hash",
            "chunk_id": "chunk-2",
            "parsed_document_id": "doc-1",
            "parsed_table_id": "table-1",
            "curation_status": "needs_review",
            "provenance_status": "passed",
        }
    ]
    write_jsonl(final / "05_evidence_layers" / "reviewable_evidence.jsonl", reviewable)
    write_jsonl(final / "05_evidence_layers" / "accepted_evidence.jsonl", [])
    chunks = [
        chunk("chunk-1", 1, "source:file#section=pk#chunk=1", "preceding context"),
        chunk("chunk-2", 2, "source:file#section=pk#chunk=2", "direct context"),
        chunk("chunk-3", 3, "source:file#section=pk#chunk=3", "following context"),
        chunk("chunk-cross", 3, "source:file#section=other#chunk=3", "must be excluded"),
        chunk("chunk-far", 6, "source:file#section=pk#chunk=6", "must be excluded"),
    ]
    write_jsonl(raw / "03_chunks_and_embeddings" / "text_chunks.jsonl", chunks)
    write_jsonl(
        raw / "02_parsed_outputs" / "parsed_tables.jsonl",
        [
            {
                "run_id": "run_x",
                "parsed_table_id": "table-1",
                "raw_asset_id": "raw-1",
                "source_id": "pubchem",
                "source_location": "pubchem:file#table=1",
                "source_url": "https://example.test/pubchem",
                "storage_path": "/raw/pubchem.json",
                "checksum": "raw-checksum",
                "table_hash": "table-hash",
                "table_text": "Parameter | Value\nMolecular weight | 194.19 g/mol",
            }
        ],
    )
    write_jsonl(raw / "06_table_recovery" / "recovered_tables.jsonl", [])

    summary = stage.prepare()

    records = read_jsonl(stage.records_path)
    assert summary["record_count"] == 5
    assert {text for row in records for text in [row["source_text"]]} == {
        "194.19",
        "preceding context",
        "direct context",
        "following context",
        "Parameter | Value\nMolecular weight | 194.19 g/mol",
    }
    structured = next(row for row in records if row["source_text"] == "194.19")
    assert "Compound: caffeine" in structured["embedding_text"]
    assert "Parameter: MolecularWeight" in structured["embedding_text"]
    assert "194.19 g/mol" in structured["embedding_text"]
    assert [row["embedding_row"] for row in records] == list(range(len(records)))
    state = json.loads(stage.state_path.read_text())
    assert state["stages"]["rag_index"]["status"] == "prepared"


def test_prepare_merges_identical_records_without_losing_links(tmp_path: Path) -> None:
    stage, raw, final = make_stage(tmp_path)
    base = {
        "run_id": "run_x",
        "compound_id": "c1",
        "compound_name": "caffeine",
        "evidence_type": "adme",
        "parameter": "absorption",
        "source_id": "pmc_oa",
        "source_location": "pmc:file#p=1",
        "source_text": "Caffeine is rapidly absorbed.",
        "curation_status": "needs_review",
        "provenance_status": "passed",
    }
    write_jsonl(
        final / "05_evidence_layers" / "reviewable_evidence.jsonl",
        [base | {"evidence_id": "e1"}, base | {"evidence_id": "e2"}],
    )
    write_jsonl(final / "05_evidence_layers" / "accepted_evidence.jsonl", [])
    write_jsonl(raw / "03_chunks_and_embeddings" / "text_chunks.jsonl", [])
    write_jsonl(raw / "02_parsed_outputs" / "parsed_tables.jsonl", [])
    write_jsonl(raw / "06_table_recovery" / "recovered_tables.jsonl", [])

    summary = stage.prepare()
    records = read_jsonl(stage.records_path)

    assert summary["record_count"] == 1
    assert records[0]["evidence_ids"] == ["e1", "e2"]


def test_prepare_reconstructs_reliable_recovered_table_from_saved_rows(tmp_path: Path) -> None:
    stage, raw, final = make_stage(tmp_path)
    evidence = {
        "run_id": "run_x",
        "evidence_id": "e-table",
        "compound_id": "c1",
        "compound_name": "midazolam",
        "evidence_type": "clinical_pk",
        "parameter": "Cmax",
        "source_id": "dailymed",
        "source_location": "dailymed:file#table=1#row=1",
        "source_text": "Cmax 28.0 ng/mL",
        "recovered_table_id": "recovered-1",
        "curation_status": "needs_review",
        "provenance_status": "passed",
    }
    write_jsonl(final / "05_evidence_layers" / "reviewable_evidence.jsonl", [evidence])
    write_jsonl(final / "05_evidence_layers" / "accepted_evidence.jsonl", [])
    write_jsonl(raw / "03_chunks_and_embeddings" / "text_chunks.jsonl", [])
    write_jsonl(raw / "02_parsed_outputs" / "parsed_tables.jsonl", [])
    write_jsonl(
        raw / "06_table_recovery" / "recovered_tables.jsonl",
        [
            {
                "run_id": "run_x",
                "recovered_table_id": "recovered-1",
                "raw_asset_id": "raw-1",
                "source_id": "dailymed",
                "source_location": "dailymed:file#table=1",
                "source_url": "https://example.test/label",
                "storage_path": "/raw/label.xml",
                "checksum": "raw-checksum",
                "caption": "PK parameters",
                "columns_json": ["Dose", "Cmax"],
                "metadata_json": {"structure_reliable": True},
            }
        ],
    )
    write_jsonl(
        raw / "06_table_recovery" / "recovered_table_rows.jsonl",
        [
            {"recovered_table_id": "recovered-1", "row_index": 2, "row_text": "1 mg | 40 ng/mL"},
            {"recovered_table_id": "recovered-1", "row_index": 1, "row_text": "0.5 mg | 28 ng/mL"},
        ],
    )

    summary = stage.prepare()
    records = read_jsonl(stage.records_path)

    assert summary["reconstructed_recovered_table_count"] == 1
    table = next(row for row in records if "recovered_table" in row["record_kinds"])
    assert table["source_text"] == "PK parameters\nDose | Cmax\n0.5 mg | 28 ng/mL\n1 mg | 40 ng/mL"
    assert table["recovered_table_ids"] == ["recovered-1"]


def test_build_embedding_text_never_replaces_source_text() -> None:
    record = {
        "compound_name": "warfarin",
        "compound_id": "c2",
        "evidence_types": ["clinical_pk"],
        "parameters": ["AUC"],
        "values": ["12 mg*h/L"],
        "record_kinds": ["evidence"],
        "source_text": "Original sentence.",
    }
    assert build_embedding_text(record).endswith("Source text: Original sentence.")
    assert record["source_text"] == "Original sentence."


def chunk(chunk_id: str, index: int, location: str, text: str) -> dict:
    return {
        "run_id": "run_x",
        "chunk_id": chunk_id,
        "chunk_index": index,
        "chunk_text": text,
        "chunk_text_hash": f"hash-{chunk_id}",
        "parsed_document_id": "doc-1",
        "raw_asset_id": "raw-1",
        "source_id": "pmc_oa",
        "source_location": location,
        "source_url": "https://example.test/article",
        "storage_path": "/raw/article.xml",
        "checksum": "raw-checksum",
    }
