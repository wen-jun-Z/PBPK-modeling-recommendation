from __future__ import annotations

import json
from pathlib import Path

import pytest

from pbpk_kb.processing.deterministic_extract_stage import (
    DeterministicExtractStage,
    contains_exact_name,
    ddi_relation_rejection_reason,
    explicit_compound_pk_relation,
    explicit_compound_target_relation,
    pk_narrative_rejection_reason,
    valid_pk_value_and_unit,
)
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
from pbpk_kb.processing.raw_processor import RawProcessingPipeline, _sentence_around


RUN_ID = "run_test"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row))
            file.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parsed_record(asset: dict, record_id: str, payload: dict, *, source_location: str | None = None) -> dict:
    return {
        "run_id": RUN_ID,
        "parsed_record_id": record_id,
        "raw_asset_id": asset["raw_asset_id"],
        "source_id": asset["source_id"],
        "compound": asset.get("compound"),
        "storage_path": asset["storage_path"],
        "source_url": asset.get("source_url"),
        "checksum": asset["checksum"],
        "parser_name": "json_parser",
        "parser_version": "raw-parse-v3",
        "record_type": "json_root",
        "source_location": source_location or f"{asset['source_id']}:{Path(asset['storage_path']).name}#json_path=$",
        "record_json": payload,
        "record_hash": "hash",
        "metadata_json": {},
    }


def parsed_table(asset: dict) -> dict:
    return {
        "run_id": RUN_ID,
        "parsed_table_id": "table-1",
        "raw_asset_id": asset["raw_asset_id"],
        "source_id": asset["source_id"],
        "compound": asset.get("compound"),
        "storage_path": asset["storage_path"],
        "source_url": asset.get("source_url"),
        "checksum": asset["checksum"],
        "parser_name": "table_parser",
        "parser_version": "raw-parse-v3",
        "source_location": "pmc_oa:table.xlsx#sheet=PK#rows=1-2",
        "columns_json": ["Parameter", "Value"],
        "cells_json": [["Cmax", "12 ng/mL"]],
        "table_text": "Cmax | 12 ng/mL",
        "table_hash": "table-hash",
        "metadata_json": {"llm_eligible": True},
    }


def chunk(asset: dict, chunk_id: str, text: str) -> dict:
    return {
        "run_id": RUN_ID,
        "chunk_id": chunk_id,
        "raw_asset_id": asset["raw_asset_id"],
        "parsed_document_id": f"doc-{chunk_id}",
        "parsed_table_id": None,
        "source_id": asset["source_id"],
        "compound": asset.get("compound"),
        "storage_path": asset["storage_path"],
        "source_url": asset.get("source_url"),
        "checksum": asset["checksum"],
        "source_location": f"{asset['source_id']}:{Path(asset['storage_path']).name}#section=test#chunk={chunk_id}",
        "chunk_index": 1,
        "chunk_text": text,
        "chunk_text_hash": "chunk-hash",
        "token_count": 30,
        "metadata_json": {"section_title": "Clinical Pharmacology", "llm_eligible": True},
    }


def build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "raw" / RUN_ID
    run_root.mkdir(parents=True)
    output_dir = tmp_path / "processed" / RUN_ID
    layout = raw_processing_layout(output_dir)
    layout.ensure_dirs()

    pubchem = {
        "raw_asset_id": "raw-pubchem",
        "source_id": "pubchem",
        "compound": "caffeine",
        "storage_path": "/raw/pubchem.json",
        "source_url": "https://pubchem.example/caffeine",
        "checksum": "a" * 64,
        "asset_metadata_json": {},
    }
    chembl = {
        "raw_asset_id": "raw-chembl",
        "source_id": "chembl",
        "compound": "caffeine",
        "storage_path": "/raw/chembl.json",
        "source_url": "https://chembl.example/caffeine",
        "checksum": "b" * 64,
        "asset_metadata_json": {},
    }
    osp_manifest = {
        "raw_asset_id": "raw-osp-manifest",
        "source_id": "osp_pbpk_model_library",
        "compound": None,
        "storage_path": "/raw/repository_inventory.json",
        "source_url": "https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library",
        "checksum": "c" * 64,
        "asset_metadata_json": {},
    }
    osp_model = {
        "raw_asset_id": "raw-osp-model",
        "source_id": "osp_pbpk_model_library",
        "compound": None,
        "storage_path": "/raw/Midazolam-Model.pksim5",
        "source_url": "https://raw.example/Midazolam/Midazolam-Model.pksim5",
        "checksum": "d" * 64,
        "asset_metadata_json": {"metadata": {"repository_path": "Midazolam/Midazolam-Model.pksim5"}},
    }
    osp_report = {
        "raw_asset_id": "raw-osp-report",
        "source_id": "osp_pbpk_model_library",
        "compound": None,
        "storage_path": "/raw/Midazolam_evaluation_report.pdf",
        "source_url": "https://raw.example/Midazolam/Midazolam_evaluation_report.pdf",
        "checksum": "e" * 64,
        "asset_metadata_json": {"metadata": {"repository_path": "Midazolam/Midazolam_evaluation_report.pdf"}},
    }
    caffeine_text = {
        "raw_asset_id": "raw-caffeine-text",
        "source_id": "pmc_oa",
        "compound": "caffeine",
        "storage_path": "/raw/caffeine.xml",
        "source_url": "https://pmc.example/caffeine",
        "checksum": "f" * 64,
        "asset_metadata_json": {},
    }
    midazolam_text = {
        "raw_asset_id": "raw-midazolam-text",
        "source_id": "pmc_oa",
        "compound": "midazolam",
        "storage_path": "/raw/midazolam.xml",
        "source_url": "https://pmc.example/midazolam",
        "checksum": "1" * 64,
        "asset_metadata_json": {},
    }
    table_asset = {
        "raw_asset_id": "raw-table",
        "source_id": "pmc_oa",
        "compound": "midazolam",
        "storage_path": "/raw/table.xlsx",
        "source_url": "https://pmc.example/table",
        "checksum": "2" * 64,
        "asset_metadata_json": {},
    }
    inventory = [pubchem, chembl, osp_manifest, osp_model, osp_report, caffeine_text, midazolam_text, table_asset]
    write_jsonl(layout.path("raw_inventory.jsonl", for_write=True), inventory)
    write_jsonl(layout.path("raw_asset_aliases.jsonl", for_write=True), [])

    records = [
        parsed_record(
            pubchem,
            "record-pubchem",
            {
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 2519,
                            "MolecularFormula": "C8H10N4O2",
                            "MolecularWeight": "194.19",
                            "InChIKey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                            "XLogP": -0.1,
                            "TPSA": 58.4,
                        }
                    ]
                }
            },
        ),
        parsed_record(
            chembl,
            "record-chembl",
            {
                "molecules": [
                    {
                        "molecule_chembl_id": "CHEMBL113",
                        "pref_name": "CAFFEINE",
                        "molecule_synonyms": [],
                        "molecule_properties": {"full_mwt": "194.19", "alogp": "-0.07"},
                    },
                    {
                        "molecule_chembl_id": "CHEMBL999",
                        "pref_name": "CAFFEINE CITRATE",
                        "molecule_synonyms": [],
                        "molecule_properties": {"full_mwt": "386.31"},
                    },
                ]
            },
        ),
        parsed_record(
            osp_manifest,
            "record-osp",
            {
                "target_compound_files": [
                    {"path": "Midazolam/Midazolam-Model.pksim5", "raw_url": osp_model["source_url"]},
                    {"path": "Midazolam/Midazolam_evaluation_report.pdf", "raw_url": osp_report["source_url"]},
                    {"path": "Unrelated/Unrelated-Model.pksim5", "raw_url": "https://raw.example/unrelated"},
                ]
            },
        ),
    ]
    tables = [parsed_table(table_asset)]
    chunks = [
        chunk(caffeine_text, "chunk-pk", "After oral dosing, caffeine Cmax was 10 ng/mL in healthy adults."),
        chunk(midazolam_text, "chunk-ddi", "Midazolam is a sensitive substrate of CYP3A4 in healthy subjects."),
        chunk(caffeine_text, "chunk-dose", "The caffeine Cmax assessment followed a single 40 mg oral dose."),
        chunk(caffeine_text, "chunk-none", "This article describes routine analytical procedures for the study."),
    ]
    write_jsonl(layout.path("parsed_records.jsonl", for_write=True), records)
    write_jsonl(layout.path("parsed_tables.jsonl", for_write=True), tables)
    write_jsonl(layout.path("text_chunks.jsonl", for_write=True), chunks)
    write_json(layout.path("parse_summary.json", for_write=True), {
        "stage": "parse",
        "status": "completed",
        "accepted": True,
        "run_id": RUN_ID,
        "expected_raw_asset_count": len(inventory),
        "parsed_document_count": 4,
        "parsed_record_count": len(records),
        "parsed_table_count": len(tables),
        "text_chunk_count": len(chunks),
    })
    write_json(layout.path("pipeline_state.json", for_write=True), {
        "run_id": RUN_ID,
        "run_root": str(run_root),
        "current_stage": "parse",
        "stages": {
            "inventory": {"status": "completed", "accepted": True},
            "parse": {"status": "completed", "accepted": True},
            "deterministic_extract": {"status": "not_started"},
            "llm_extract": {"status": "not_started"},
            "completeness": {"status": "not_started"},
            "embedding": {"status": "not_started"},
        },
    })
    layout.path("processing_failures.jsonl", for_write=True).touch()
    return run_root, output_dir


def test_deterministic_stage_maps_only_high_precision_traceable_candidates(tmp_path: Path) -> None:
    run_root, output_dir = build_fixture(tmp_path)
    summary = DeterministicExtractStage(run_root=run_root, output_dir=output_dir, classifier_min_chars=20).run()
    layout = raw_processing_layout(output_dir)

    assert summary["accepted"] is True
    assert summary["structured_record_status_count"] == 3
    assert summary["structured_table_status_count"] == 1
    assert summary["chunk_processing_status_count"] == 4
    assert summary["llm_requests_created"] == 0
    assert summary["embeddings_created"] == 0
    assert summary["downstream_files_unchanged"] is True

    deterministic = read_jsonl(layout.path("evidence_candidates_deterministic.jsonl"))
    merged = read_jsonl(layout.path("evidence_candidates.jsonl"))
    assert deterministic == merged
    assert len({row["evidence_id"] for row in deterministic}) == len(deterministic)
    assert all(row["curation_status"] == "needs_review" for row in deterministic)
    assert all(row["raw_asset_id"] and row["checksum"] and row["source_location"] and row["source_text"] for row in deterministic)

    molecular_weights = [row for row in deterministic if row["parameter"] == "full_mwt"]
    assert [row["value"] for row in molecular_weights] == [194.19]
    assert not any(row.get("value") == 386.31 for row in deterministic)
    assert any(row["evidence_type"] == "public_pbpk_model" and row["compound"] == "midazolam" for row in deterministic)
    assert any(row["evidence_type"] == "clinical_pk" and row["parameter"] == "Cmax" and row["value"] == 10 for row in deterministic)
    assert not any(row["evidence_type"] == "clinical_pk" and row.get("value") == 40 for row in deterministic)
    assert any(row["evidence_type"] == "ddi" and row["compound"] == "midazolam" for row in deterministic)

    table_status = read_jsonl(layout.path("structured_table_processing_status.jsonl"))
    assert table_status[0]["processing_status"] == "deferred_complex_table"
    state = json.loads(layout.path("pipeline_state.json").read_text(encoding="utf-8"))
    assert state["stages"]["deterministic_extract"]["accepted"] is True
    assert state["stages"]["llm_extract"]["status"] == "not_started"
    assert not layout.path("embedding_index.jsonl").exists()
    assert not layout.path("llm_extraction_requests_live.jsonl").exists()


def test_deterministic_stage_refuses_started_downstream_stage(tmp_path: Path) -> None:
    run_root, output_dir = build_fixture(tmp_path)
    layout = raw_processing_layout(output_dir)
    state = json.loads(layout.path("pipeline_state.json").read_text(encoding="utf-8"))
    state["stages"]["llm_extract"] = {"status": "completed", "accepted": True}
    write_json(layout.path("pipeline_state.json", for_write=True), state)

    with pytest.raises(ValueError, match="already started"):
        DeterministicExtractStage(run_root=run_root, output_dir=output_dir).run()


def test_pipeline_does_not_initialize_llm_agent_for_staged_execution(tmp_path: Path) -> None:
    run_root = tmp_path / RUN_ID
    run_root.mkdir()
    pipeline = RawProcessingPipeline(
        run_root=run_root,
        output_root=tmp_path / "processed",
        stage="deterministic_extract",
        llm_mode="off",
        embedding_mode="off",
    )
    assert pipeline.agent is None


def test_pk_unit_gate_rejects_dose_units() -> None:
    assert valid_pk_value_and_unit("Cmax", 10, "ng/mL") is True
    assert valid_pk_value_and_unit("Cmax", 40, "mg") is False
    assert valid_pk_value_and_unit("bioavailability", 44, "%") is True
    assert valid_pk_value_and_unit("bioavailability", 144, "%") is False


def test_sentence_extraction_preserves_decimals_and_abbreviations() -> None:
    text = "Prior sentence. Midazolam half-life was 4.19 h vs. 7.2 h; follow-up sentence."
    start = text.index("half-life")
    source_text = _sentence_around(text, start, start + len("half-life was 4.19 h"))
    assert source_text == "Midazolam half-life was 4.19 h vs. 7.2 h;"


def test_narrative_quality_gates_reject_renal_criteria_relative_values_and_metabolite_names() -> None:
    assert (
        pk_narrative_rejection_reason(
            "clearance",
            "Metformin trial eligibility required creatinine clearance above 60 mL/min.",
            "metformin",
        )
        == "subject_renal_function_not_drug_clearance"
    )
    assert (
        pk_narrative_rejection_reason(
            "bioavailability",
            "Midazolam bioavailability was 43% and 100% higher in cirrhosis.",
            "midazolam",
            value_text="43",
            unit="%",
        )
        == "relative_change_not_absolute_pk_parameter"
    )
    assert (
        pk_narrative_rejection_reason(
            "clearance",
            "Rosuvastatin exposure increased in severe renal impairment (CL cr <30 mL/min).",
            "rosuvastatin",
            value_text="30",
            unit="mL/min",
        )
        == "subject_renal_function_not_drug_clearance"
    )
    assert (
        pk_narrative_rejection_reason(
            "clearance",
            "Metformin is dialyzable with a clearance of up to 170 mL/min.",
            "metformin",
            value_text="170",
            unit="mL/min",
        )
        == "dialysis_context_not_systemic_drug_clearance"
    )
    assert (
        pk_narrative_rejection_reason(
            "Tmax",
            "The median Tmax for metformin occurred 2 hours later after a high-fat meal.",
            "metformin",
            value_text="2",
            unit="hours",
        )
        == "relative_change_not_absolute_pk_parameter"
    )
    assert contains_exact_name("1-hydroxy-midazolam had a longer half-life.", "midazolam") is False
    assert (
        pk_narrative_rejection_reason(
            "Tmax",
            "For 1′-OH midazolam, median Tmax was 0.75 h.",
            "midazolam",
            value_text="0.75",
            unit="h",
            rule_match="Tmax was 0.75 h",
        )
        == "metabolite_subject_requires_resolution"
    )


def test_pk_relation_gate_requires_local_compound_parameter_value_syntax() -> None:
    assert explicit_compound_pk_relation(
        "The plasma clearance of omeprazole was 250 mL/min.",
        "omeprazole",
        parameter="clearance",
        rule_match="clearance of omeprazole was 250 mL/min",
        value_text="250",
    )
    assert explicit_compound_pk_relation(
        "Warfarin shows a volume of distribution of about 0.14 L/kg.",
        "warfarin",
        parameter="volume_of_distribution",
        rule_match="volume of distribution of about 0.14 L/kg",
        value_text="0.14",
    )
    assert explicit_compound_pk_relation(
        "Caffeine has a terminal half-life of 4 h.",
        "caffeine",
        parameter="half_life",
        rule_match="half-life of 4 h",
        value_text="4",
    )
    assert not explicit_compound_pk_relation(
        "Coadministration of metformin did not alter trospium absorption, with Tmax approximately 4 h for trospium.",
        "metformin",
        parameter="Tmax",
        rule_match="Tmax approximately 4 h",
        value_text="4",
    )
    assert not explicit_compound_pk_relation(
        "Alprazolam was shorter-lived (t1/2 = 32 min) than caffeine (t1/2 = 3 h).",
        "caffeine",
        parameter="half_life",
        rule_match="t1/2 = 32 min",
        value_text="32",
    )
    assert not explicit_compound_pk_relation(
        "Mean Tmax, concentration at 24 hours (C24), and renal digoxin clearance were determined.",
        "digoxin",
        parameter="Tmax",
        rule_match="Tmax, concentration at 24 hours",
        value_text="24",
    )
    assert (
        pk_narrative_rejection_reason(
            "Cmax",
            "Orange line marks the metformin Cmax threshold at 5 mg/L.",
            "metformin",
            value_text="5",
            unit="mg/L",
            rule_match="Cmax threshold at 5 mg/L",
        )
        == "model_or_threshold_context_not_clinical_observation"
    )
    assert (
        pk_narrative_rejection_reason(
            "clearance",
            "Plasma clearance in Digoxin was set from 2.00 mL/min/kg.",
            "digoxin",
            value_text="2.00",
            unit="mL/min/kg",
            rule_match="clearance in Digoxin was set from 2.00 mL/min/kg",
            source_location="supplement.pksim5#sqlite_table=COMMANDS#row=1",
        )
        == "model_configuration_not_clinical_observation"
    )
    assert not explicit_compound_pk_relation(
        "For metformin, Tmax and T1/2 for metformin are 2 h and 6 h.",
        "metformin",
        parameter="half_life",
        rule_match="T1/2 for metformin are 2 h",
        value_text="2",
    )


def test_ddi_relation_gate_requires_compound_as_explicit_relation_subject() -> None:
    assert explicit_compound_target_relation("Midazolam is a sensitive substrate of CYP3A4.", "midazolam", "CYP3A4") == "substrate"
    assert explicit_compound_target_relation("Midazolam is a sensitive CYP3A4 substrate.", "midazolam", "CYP3A4") == "substrate"
    assert explicit_compound_target_relation("Metformin was used as an OCT1 substrate.", "metformin", "OCT1") == "substrate"
    assert explicit_compound_target_relation("Midazolam is metabolized by CYP3A.", "midazolam", "CYP3A") == "substrate"
    assert explicit_compound_target_relation("Inhibition of CYP2C19 by omeprazole was observed.", "omeprazole", "CYP2C19") == "inhibitor"
    assert explicit_compound_target_relation("PA-824 inhibits CYP3A and alters the PK of midazolam.", "midazolam", "CYP3A") is None
    assert explicit_compound_target_relation("Propofol reduced midazolam clearance through inhibition of CYP3A4.", "midazolam", "CYP3A4") is None
    assert explicit_compound_target_relation("Like midazolam, alfentanil is not a substrate for P-gp.", "midazolam", "P-glycoprotein") is None
    assert explicit_compound_target_relation("Digoxin Is Not a P-gp Specific Probe Substrate.", "digoxin", "P-glycoprotein") is None
    assert (
        ddi_relation_rejection_reason(
            "Midazolam is not a substrate for P-glycoprotein.",
            "midazolam",
            "P-glycoprotein",
            "substrate",
        )
        == "negated_ddi_relation"
    )
    assert (
        ddi_relation_rejection_reason(
            "We investigated whether or not metformin is a substrate for P-glycoprotein.",
            "metformin",
            "P-glycoprotein",
            "substrate",
        )
        == "question_or_hypothesis_not_asserted_relation"
    )
    assert (
        ddi_relation_rejection_reason(
            "Inhibition of CYP2C19 by omeprazole and/or its metabolites was observed.",
            "omeprazole",
            "CYP2C19",
            "inhibitor",
        )
        == "parent_metabolite_relation_subject_ambiguous"
    )
    assert (
        ddi_relation_rejection_reason(
            "Midazolam is a sensitive CYP3A4 substrate and its exposure was not altered by tofacitinib.",
            "midazolam",
            "CYP3A4",
            "substrate",
        )
        is None
    )
