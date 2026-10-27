from __future__ import annotations

import csv
import json
from pathlib import Path

from pbpk_kb.processing.completeness_stage import CompletenessStage, topic_for_table_review, topics_for_asset
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row))
            file.write("\n")


def test_topics_use_source_responsibility_and_pubmed_query_family() -> None:
    assert topics_for_asset({"source_id": "pubchem", "group": "compound_api"}) == {
        "identity",
        "physicochemical",
    }
    assert topics_for_asset({"source_id": "pubmed", "group": "pubmed/esearch/ddi"}) == {
        "literature",
        "ddi",
    }
    assert topics_for_asset(
        {"source_id": "pubmed", "storage_path": "/raw/pubmed/esearch/pbpk_model/page.xml"}
    ) == {"literature", "public_pbpk_model"}


def test_table_review_topic_is_conservative() -> None:
    assert topic_for_table_review({"review_reason": "ddi_relation_requires_table_semantic_review"}) == "ddi"
    assert topic_for_table_review({"columns_json": ["Renal impairment", "AUC ratio"]}) == "special_population"
    assert topic_for_table_review({"caption": "Population PK summary", "columns_json": ["Cmax"]}) == "clinical_pk"


def test_completeness_materializes_cross_product_and_preserves_shared_assets(tmp_path: Path) -> None:
    output_dir = tmp_path / "run_test"
    layout = raw_processing_layout(output_dir)
    layout.ensure_dirs()
    inventory = [
        {
            "run_id": "run_test",
            "raw_asset_id": "pubchem-raw",
            "source_id": "pubchem",
            "compound": "midazolam",
            "group": "compound_api",
            "storage_path": "/raw/pubchem.json",
        },
        {
            "run_id": "run_test",
            "raw_asset_id": "label-raw",
            "source_id": "dailymed",
            "compound": "midazolam",
            "group": "dailymed/spl_xml",
            "storage_path": "/raw/label.xml",
        },
        {
            "run_id": "run_test",
            "raw_asset_id": "osp-raw",
            "source_id": "osp_pbpk_model_library",
            "compound": None,
            "group": "shared/osp_pbpk_model_library/github_selected_files",
            "storage_path": "/raw/model.pkml",
        },
    ]
    write_jsonl(layout.path("raw_inventory.jsonl", for_write=True), inventory)
    write_jsonl(
        layout.path("parsed_documents.jsonl", for_write=True),
        [
            {
                "run_id": "run_test",
                "parsed_document_id": "label-document",
                "raw_asset_id": "label-raw",
                "source_id": "dailymed",
                "compound": "midazolam",
            }
        ],
    )
    write_jsonl(
        layout.path("parsed_records.jsonl", for_write=True),
        [
            {
                "run_id": "run_test",
                "parsed_record_id": "pubchem-record",
                "raw_asset_id": "pubchem-raw",
                "source_id": "pubchem",
                "compound": "midazolam",
            },
            {
                "run_id": "run_test",
                "parsed_record_id": "osp-record",
                "raw_asset_id": "osp-raw",
                "source_id": "osp_pbpk_model_library",
                "compound": None,
            },
        ],
    )
    write_jsonl(
        layout.path("parsed_tables.jsonl", for_write=True),
        [
            {
                "run_id": "run_test",
                "parsed_table_id": "label-table",
                "raw_asset_id": "label-raw",
                "source_id": "dailymed",
                "compound": "midazolam",
            }
        ],
    )
    write_jsonl(layout.path("recovered_tables.jsonl", for_write=True), [])
    write_jsonl(
        layout.path("evidence_candidates.jsonl", for_write=True),
        [
            {
                "run_id": "run_test",
                "evidence_id": "mw-evidence",
                "evidence_type": "physicochemical",
                "raw_asset_id": "pubchem-raw",
                "parsed_record_id": "pubchem-record",
                "source_id": "pubchem",
                "compound": "midazolam",
                "curation_status": "needs_review",
            }
        ],
    )
    write_jsonl(
        layout.path("table_review_queue.jsonl", for_write=True),
        [
            {
                "run_id": "run_test",
                "review_id": "ddi-review",
                "review_reason": "ddi_relation_requires_table_semantic_review",
                "raw_asset_id": "label-raw",
                "parsed_table_id": "label-table",
                "source_id": "dailymed",
                "compound": "midazolam",
            }
        ],
    )
    write_jsonl(
        layout.path("asset_parse_status.jsonl", for_write=True),
        [
            {"raw_asset_id": row["raw_asset_id"], "source_id": row["source_id"], "compound": row["compound"], "terminal_status": "parsed"}
            for row in inventory
        ],
    )
    write_jsonl(layout.path("acquisition_failure_status.jsonl", for_write=True), [])
    write_jsonl(layout.path("processing_failures.jsonl", for_write=True), [])
    write_json(
        layout.path("pipeline_state.json", for_write=True),
        {
            "run_id": "run_test",
            "current_stage": "table_recovery",
            "stages": {
                "table_recovery": {"status": "completed", "accepted": True},
                "completeness": {"status": "not_started"},
                "embedding": {"status": "not_started"},
            },
        },
    )

    summary = CompletenessStage(output_dir=output_dir).run()

    assert summary["accepted"] is True
    assert summary["compound_count"] == 1
    assert summary["source_count"] == 3
    assert summary["expected_cross_product_row_count"] == 33
    assert summary["matrix_row_count"] == 34
    assert summary["shared_row_count"] == 1
    with layout.path("completeness_matrix.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    by_key = {(row["compound"], row["source_id"], row["topic"]): row for row in rows}
    assert by_key[("midazolam", "pubchem", "physicochemical")]["status"] == "candidate_extracted"
    assert "evidence_ids" not in by_key[("midazolam", "pubchem", "physicochemical")]
    provenance = [
        json.loads(line)
        for line in layout.path("completeness_provenance.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    provenance_by_id = {row["provenance_record_id"]: row for row in provenance}
    physicochemical = by_key[("midazolam", "pubchem", "physicochemical")]
    assert provenance_by_id[physicochemical["provenance_record_id"]]["evidence_ids"] == ["mw-evidence"]
    assert by_key[("midazolam", "pubchem", "identity")]["status"] == "parsed"
    assert by_key[("midazolam", "dailymed", "label")]["status"] == "parsed"
    assert by_key[("midazolam", "dailymed", "ddi")]["status"] == "needs_review"
    assert by_key[("midazolam", "osp_pbpk_model_library", "public_pbpk_model")]["status"] == "missing_from_raw"
    assert by_key[("shared", "osp_pbpk_model_library", "public_pbpk_model")]["status"] == "parsed"
    state = json.loads(layout.path("pipeline_state.json").read_text(encoding="utf-8"))
    assert state["current_stage"] == "completeness"
    assert state["stages"]["completeness"]["accepted"] is True
    assert state["stages"]["embedding"]["status"] == "not_started"
