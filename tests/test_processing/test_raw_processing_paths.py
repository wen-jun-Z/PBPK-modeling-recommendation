from __future__ import annotations

from pathlib import Path

from pbpk_kb.processing.raw_processing_paths import raw_processing_layout


def test_raw_processing_layout_writes_to_stage_subdirectories(tmp_path: Path) -> None:
    layout = raw_processing_layout(tmp_path / "run_x")

    assert layout.path("raw_inventory.jsonl", for_write=True) == tmp_path / "run_x" / "01_raw_inventory" / "raw_inventory.jsonl"
    assert layout.path("text_chunks.jsonl", for_write=True) == tmp_path / "run_x" / "03_chunks_and_embeddings" / "text_chunks.jsonl"
    assert layout.path("evidence_candidates.jsonl", for_write=True) == tmp_path / "run_x" / "04_evidence_candidates" / "evidence_candidates.jsonl"
    assert layout.path("llm_extraction_requests_live.jsonl", for_write=True) == tmp_path / "run_x" / "05_llm_extraction" / "llm_extraction_requests_live.jsonl"
    assert layout.path("processing_summary.json", for_write=True) == tmp_path / "run_x" / "00_run_manifest" / "processing_summary.json"
    assert layout.path("inventory_summary.json", for_write=True) == tmp_path / "run_x" / "00_run_manifest" / "inventory_summary.json"
    assert layout.path("pipeline_state.json", for_write=True) == tmp_path / "run_x" / "00_run_manifest" / "pipeline_state.json"
    assert layout.path("parse_summary.json", for_write=True) == tmp_path / "run_x" / "00_run_manifest" / "parse_summary.json"
    assert layout.path("parse_progress.json", for_write=True) == tmp_path / "run_x" / "00_run_manifest" / "parse_progress.json"
    assert layout.path("deterministic_extract_summary.json", for_write=True) == tmp_path / "run_x" / "00_run_manifest" / "deterministic_extract_summary.json"
    assert layout.path("deterministic_extract_progress.json", for_write=True) == tmp_path / "run_x" / "00_run_manifest" / "deterministic_extract_progress.json"
    assert layout.path("asset_parse_status.jsonl", for_write=True) == tmp_path / "run_x" / "00_run_manifest" / "asset_parse_status.jsonl"
    assert layout.path("raw_asset_aliases.jsonl", for_write=True) == tmp_path / "run_x" / "01_raw_inventory" / "raw_asset_aliases.jsonl"
    assert layout.path("parsed_records.jsonl", for_write=True) == tmp_path / "run_x" / "02_parsed_outputs" / "parsed_records.jsonl"
    assert layout.path("structured_record_processing_status.jsonl", for_write=True) == tmp_path / "run_x" / "02_parsed_outputs" / "structured_record_processing_status.jsonl"
    assert layout.path("structured_table_processing_status.jsonl", for_write=True) == tmp_path / "run_x" / "02_parsed_outputs" / "structured_table_processing_status.jsonl"
    assert layout.path("archive_members.jsonl", for_write=True) == tmp_path / "run_x" / "02_parsed_outputs" / "archive_members.jsonl"
    assert layout.path("asset_links.jsonl", for_write=True) == tmp_path / "run_x" / "02_parsed_outputs" / "asset_links.jsonl"
    assert layout.path("evidence_candidates_deterministic.jsonl", for_write=True) == tmp_path / "run_x" / "04_evidence_candidates" / "evidence_candidates_deterministic.jsonl"


def test_raw_processing_layout_reads_staged_before_flat_legacy_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_x"
    flat = run_dir / "evidence_candidates.jsonl"
    staged = run_dir / "04_evidence_candidates" / "evidence_candidates.jsonl"
    flat.parent.mkdir(parents=True)
    staged.parent.mkdir(parents=True)
    flat.write_text("flat\n", encoding="utf-8")
    staged.write_text("staged\n", encoding="utf-8")

    assert raw_processing_layout(run_dir).path("evidence_candidates.jsonl") == staged


def test_raw_processing_layout_accepts_stage_subdirectory_input(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run_x" / "03_chunks_and_embeddings"
    layout = raw_processing_layout(stage_dir)

    assert layout.root == tmp_path / "run_x"
    assert layout.path("text_chunks.jsonl", for_write=True) == stage_dir / "text_chunks.jsonl"
    assert layout.path("evidence_candidates.jsonl", for_write=True) == tmp_path / "run_x" / "04_evidence_candidates" / "evidence_candidates.jsonl"
