from pbpk_kb.normalization.compound_resolver import ResolvedCompound
import json
from pathlib import Path

from pbpk_kb.storage.build_milestone3_index import (
    Milestone3Indexer,
    STAGE3_RESET_TABLES,
    compound_queries_from_args,
    replace_dailymed_placeholders_with_history,
)


def test_osp_model_compound_link_requires_exact_normalized_match() -> None:
    compound = ResolvedCompound(
        preferred_name="Midazolam",
        normalized_name="midazolam",
        synonyms=["Midazolam Hydrochloride"],
    )

    assert Milestone3Indexer._model_matches_compound({"model_folder_name": "Midazolam"}, compound)
    assert not Milestone3Indexer._model_matches_compound({"model_folder_name": "Alfentanil"}, compound)


def test_pubchem_molecular_weight_is_stored_as_numeric_value() -> None:
    value, unit = Milestone3Indexer._pubchem_numeric_property("MolecularWeight", "325.8")

    assert value == 325.8
    assert unit == "g/mol"


def test_dailymed_numeric_candidates_are_conservative_and_trace_units() -> None:
    candidates = Milestone3Indexer._extract_dailymed_numeric_candidates(
        "Elimination",
        (
            "Elimination The mean elimination half-life of midazolam ranged from 2.2 to 6.8 hours "
            "following single oral doses. In the same group of patients receiving the 0.15 mg/kg IV dose, "
            "the mean total clearance ranged from 9.3 to 11.0 mL/min/kg."
        ),
    )

    assert candidates == [
        {
            "table": "clinical_pk_evidence",
            "parameter_name": "elimination half-life",
            "value": None,
            "value_text": "2.2 to 6.8",
            "unit": "hours",
            "evidence_text": "Elimination The mean elimination half-life of midazolam ranged from 2.2 to 6.8 hours following single oral doses.",
            "confidence_score": 0.72,
        },
        {
            "table": "clinical_pk_evidence",
            "parameter_name": "total clearance",
            "value": None,
            "value_text": "9.3 to 11.0",
            "unit": "mL/min/kg",
            "evidence_text": "In the same group of patients receiving the 0.15 mg/kg IV dose, the mean total clearance ranged from 9.3 to 11.0 mL/min/kg.",
            "confidence_score": 0.72,
        },
    ]


def test_dailymed_numeric_candidates_extract_common_non_midazolam_pk_patterns() -> None:
    candidates = Milestone3Indexer._extract_dailymed_numeric_candidates(
        "Clinical Pharmacology",
        (
            "The apparent volume of distribution is approximately 1.4 L/kg. "
            "Mean Cmax was 250 ng/mL after oral dosing. "
            "The AUC0-inf was 1100 ng*h/mL. "
            "Protein binding is approximately 92%."
        ),
    )

    assert candidates == [
        {
            "table": "adme_parameter_evidence",
            "parameter_name": "plasma protein binding",
            "value": 92.0,
            "value_text": "92.0",
            "unit": "%",
            "evidence_text": "Protein binding is approximately 92%.",
            "confidence_score": 0.72,
        },
        {
            "table": "clinical_pk_evidence",
            "parameter_name": "volume of distribution",
            "value": 1.4,
            "value_text": "1.4",
            "unit": "L/kg",
            "evidence_text": "The apparent volume of distribution is approximately 1.4 L/kg.",
            "confidence_score": 0.72,
        },
        {
            "table": "clinical_pk_evidence",
            "parameter_name": "Cmax",
            "value": 250.0,
            "value_text": "250.0",
            "unit": "ng/mL",
            "evidence_text": "Mean Cmax was 250 ng/mL after oral dosing.",
            "confidence_score": 0.68,
        },
        {
            "table": "clinical_pk_evidence",
            "parameter_name": "AUC",
            "value": 1100.0,
            "value_text": "1100.0",
            "unit": "ng*h/mL",
            "evidence_text": "The AUC0-inf was 1100 ng*h/mL.",
            "confidence_score": 0.68,
        },
    ]


def test_compound_queries_from_args_accepts_repeated_and_comma_separated_values() -> None:
    queries = compound_queries_from_args(["midazolam", "warfarin,caffeine"], "Warfarin, metformin")

    assert queries == ["midazolam", "warfarin", "caffeine", "metformin"]


def test_connector_summary_keeps_stage_b_counts_and_error_messages() -> None:
    summary = Milestone3Indexer._connector_summary(
        [
            {
                "source_id": "pubchem",
                "status": "succeeded",
                "raw_document_count": 2,
                "raw_record_count": 2,
                "parsed_record_count": 1,
                "used_sample_placeholder": False,
                "source_version": "snapshot",
                "source_checksum": "abc",
                "raw_documents": [{"large": "payload"}],
                "errors": [{"message": "recoverable"}],
            }
        ]
    )

    assert summary == [
        {
            "source_id": "pubchem",
            "status": "succeeded",
            "raw_document_count": 2,
            "raw_record_count": 2,
            "parsed_record_count": 1,
            "used_sample_placeholder": False,
            "source_version": "snapshot",
            "source_checksum_present": True,
            "errors": ["recoverable"],
        }
    ]


def test_reset_stage3_rows_includes_all_generated_evidence_tables() -> None:
    assert "tissue_expression_evidence" in STAGE3_RESET_TABLES
    assert "model_parameter_evidence" in STAGE3_RESET_TABLES
    assert STAGE3_RESET_TABLES.index("tissue_expression_evidence") < STAGE3_RESET_TABLES.index("protein_target_identity")
    assert STAGE3_RESET_TABLES.index("model_parameter_evidence") < STAGE3_RESET_TABLES.index("public_pbpk_model_registry")


def test_dailymed_placeholder_rebuild_uses_matching_historical_label(tmp_path: Path) -> None:
    records_dir = tmp_path / "parsed" / "dailymed" / "20260615T000000Z_real"
    records_dir.mkdir(parents=True)
    raw_dir = tmp_path / "raw" / "dailymed" / "20260615T000000Z_real"
    raw_dir.mkdir(parents=True)
    raw_file = raw_dir / "label.xml"
    raw_file.write_text("<document />", encoding="utf-8")
    raw_metadata = {
        "source_id": "dailymed",
        "source_url": "https://dailymed.nlm.nih.gov/example.xml",
        "raw_storage_path": "/work/raw/dailymed/20260615T000000Z_real/label.xml",
        "raw_checksum": "abc",
        "download_timestamp": "2026-06-15T00:00:00Z",
        "content_type": "application/xml",
        "source_version": "snapshot",
        "raw_record_count": 1,
    }
    (raw_file.with_suffix(raw_file.suffix + ".metadata.json")).write_text(json.dumps(raw_metadata), encoding="utf-8")

    historical_record = {
        "source_id": "dailymed",
        "source_record_id": "set-1",
        "source_url": "https://dailymed.nlm.nih.gov/example",
        "source_location": "SPL_SET_ID:set-1",
        "record_type": "label_detail",
        "raw_document_path": "/work/raw/dailymed/20260615T000000Z_real/label.xml",
        "parser_name": "label_parser.py",
        "parser_version": "0.1.0",
        "record": {
            "title": "MIDAZOLAM HYDROCHLORIDE SYRUP",
            "active_ingredients": [{"name": "Midazolam Hydrochloride"}],
            "pk_section_candidates": [{"title": "Elimination", "text": "half-life text"}],
        },
    }
    (records_dir / "records.jsonl").write_text(json.dumps(historical_record) + "\n", encoding="utf-8")

    records = [
        {
            "source_id": "dailymed",
            "source_record_id": "placeholder",
            "record_type": "sample_placeholder",
            "record": {"is_sample_placeholder": True},
        }
    ]
    logs = {"dailymed": {"source_id": "dailymed", "raw_documents": [], "parsed_records_path": "placeholder.jsonl"}}

    replaced_records, replaced_logs = replace_dailymed_placeholders_with_history(tmp_path, records, logs, "midazolam")

    assert len(replaced_records) == 1
    assert replaced_records[0]["record_type"] == "label_detail"
    assert replaced_records[0]["source_record_id"] == "set-1"
    assert replaced_records[0]["_ingestion_log"]["parsed_records_path"].endswith("records.jsonl")
    assert replaced_logs["dailymed"]["raw_documents"] == [raw_metadata]
