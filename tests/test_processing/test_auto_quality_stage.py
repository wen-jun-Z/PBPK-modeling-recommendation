from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pbpk_kb.processing.auto_quality_stage import AutoQualityStage, build_kg_ready, deterministic_repair, evaluate_evidence


def evidence(tmp_path: Path, **overrides: object) -> dict[str, object]:
    raw = tmp_path / "raw.json"
    raw.write_text("{}", encoding="utf-8")
    source_text = "After dosing, caffeine Cmax was 10 ng/mL."
    row: dict[str, object] = {
        "evidence_id": "e1",
        "compound_id": "c1",
        "compound_name": "caffeine",
        "compound_role": "target_compound",
        "evidence_type": "clinical_pk",
        "parameter": "Cmax",
        "value": 10.0,
        "unit": "ng/mL",
        "source_id": "pubmed",
        "raw_asset_id": "r1",
        "storage_path": str(raw),
        "checksum": "checksum",
        "source_location": "pubmed:1#abstract",
        "source_text": source_text,
        "source_text_hash": hashlib.sha256(source_text.encode()).hexdigest(),
        "extraction_method": "rule_based",
        "extractor_version": "1",
        "conflict_statuses": [],
    }
    row.update(overrides)
    return row


def test_quantitative_unit_is_recovered_from_verbatim_text(tmp_path: Path) -> None:
    row = evidence(tmp_path, unit=None, canonical_unit=None, canonical_value=None)
    repaired, repairs = deterministic_repair(row)

    assert repaired["unit"] == "ng/mL"
    assert repaired["canonical_unit"] == "ng/mL"
    assert repaired["canonical_value"] == 10.0
    assert {item["field"] for item in repairs} == {"unit", "canonical_unit", "canonical_value"}
    assert evaluate_evidence(repaired, repairs)["quality_status"] == "auto_validated"


def test_relative_effect_without_reference_is_repair_pending(tmp_path: Path) -> None:
    text = "Caffeine AUC increased by 40%."
    row = evidence(
        tmp_path,
        evidence_type="ddi",
        parameter="AUC change",
        value=40.0,
        unit="%",
        source_text=text,
        source_text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    decision = evaluate_evidence(row, [])

    assert decision["quality_status"] == "repair_pending"
    assert "relative_effect_missing_reference" in decision["repairable_blockers"]


def test_final_quality_blocks_truncated_shared_endpoint_percent(tmp_path: Path) -> None:
    text = "Ketoconazole increased the AUC 24h, C max and C min of bedaquiline by 22%"
    row = evidence(tmp_path, compound_name="bedaquiline", evidence_type="ddi", parameter="Cmax",
                   value=22.0, unit="%", source_text=text,
                   source_text_hash=hashlib.sha256(text.encode()).hexdigest())
    decision = evaluate_evidence(row, [])
    assert decision["quality_status"] != "auto_validated"
    assert "truncated_multi_endpoint_percent_assignment" in decision["fatal_blockers"]


def test_absolute_percentage_does_not_require_reference(tmp_path: Path) -> None:
    text = "Approximately 90% of caffeine is absorbed from the stomach."
    row = evidence(
        tmp_path,
        evidence_type="adme",
        parameter="absorption",
        value=90.0,
        unit="%",
        source_text=text,
        source_text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )

    assert evaluate_evidence(row, [])["quality_status"] == "auto_validated"


def test_excreted_unchanged_is_not_a_relative_comparison(tmp_path: Path) -> None:
    text = "Approximately 90% of caffeine is excreted unchanged in urine."
    row = evidence(tmp_path, evidence_type="adme", parameter="fraction_excreted_unchanged", value=90, unit="%",
                   source_text=text, source_text_hash=hashlib.sha256(text.encode()).hexdigest())
    assert "relative_effect_missing_reference" not in evaluate_evidence(row, [])["quality_reasons"]
    from pbpk_kb.processing.final_evidence_quality_stage import evaluate_domain_gate
    assert evaluate_domain_gate(row)["domain_gate_status"] == "passed"
    text = "Caffeine Cmax was unchanged."
    row.update(parameter="Cmax", evidence_type="clinical_pk", value=None, unit=None,
               source_text=text, source_text_hash=hashlib.sha256(text.encode()).hexdigest())
    assert "relative_effect_missing_reference" in evaluate_evidence(row, [])["quality_reasons"]


def test_study_intent_without_result_is_quarantined(tmp_path: Path) -> None:
    text = "The study was designed to assess the effect of ketoconazole on caffeine pharmacokinetics."
    row = evidence(
        tmp_path,
        evidence_type="ddi",
        parameter="pharmacokinetics",
        value=None,
        unit=None,
        condition="with ketoconazole",
        source_text=text,
        source_text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    decision = evaluate_evidence(row, [])

    assert decision["quality_status"] == "quarantined"
    assert "study_intent_without_observed_result" in decision["fatal_blockers"]


def test_provenance_failure_is_quarantined(tmp_path: Path) -> None:
    row = evidence(tmp_path, source_text_hash="wrong")
    decision = evaluate_evidence(row, [])

    assert decision["quality_status"] == "quarantined"
    assert "source_text_hash_mismatch" in decision["fatal_blockers"]


def test_stage_partitions_rows_and_only_validated_rows_make_kg(tmp_path: Path) -> None:
    run = tmp_path / "run"
    layers = run / "05_evidence_layers"
    layers.mkdir(parents=True)
    rows = [evidence(tmp_path), evidence(tmp_path, evidence_id="e2", source_text_hash="wrong")]
    with (layers / "reviewable_evidence.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = AutoQualityStage(final_run_dir=run).run()

    assert summary["status_counts"] == {"auto_validated": 1, "quarantined": 1}
    validated = (run / "06_auto_quality_control" / "auto_validated_evidence.jsonl").read_text()
    assert '"evidence_id": "e1"' in validated
    assert '"evidence_id": "e2"' not in validated
    assert summary["kg_edge_count"] == 3


def test_kg_ready_adds_population_context_and_public_model_nodes(tmp_path: Path) -> None:
    row = evidence(
        tmp_path,
        evidence_type="public_pbpk_model",
        parameter="Caffeine whole-body PBPK model",
        population="healthy adults",
        route="oral",
        condition="fasted",
        metadata_json={"model_name": "Caffeine model", "software": "PK-Sim"},
        quality_status="auto_validated",
        validation_method="deterministic_quality_gate_v1",
    )

    nodes, edges = build_kg_ready([row])

    assert {"Compound", "Evidence", "Parameter", "SourceDocument", "Population", "StudyContext", "PublicModel"} <= {
        node["node_type"] for node in nodes
    }
    assert {"STUDIED_IN", "HAS_STUDY_CONTEXT", "DESCRIBED_BY_MODEL"} <= {
        edge["relationship_type"] for edge in edges
    }


def test_range_remains_a_range_through_final_quality_gate(tmp_path):
    text = "Caffeine Tmax ranged from 30 minutes to 2 hours."
    row = evidence(tmp_path, parameter="Tmax", value=None, value_low=.5, value_high=2,
                   unit="hours", source_text=text, source_text_hash=hashlib.sha256(text.encode()).hexdigest())
    repaired, repairs = deterministic_repair(row)
    assert repaired["value"] is None and repaired.get("canonical_value") is None
    assert repaired["canonical_value_low"] == .5 and repaired["canonical_value_high"] == 2
    assert evaluate_evidence(repaired, repairs)["quality_status"] == "auto_validated"
    for changed in ({"value": .5}, {"value_low": 3}, {"value_high": None}, {"unit": "kg", "canonical_unit": "kg"}):
        assert evaluate_evidence({**repaired, **changed}, [])["quality_status"] == "quarantined"


def test_percent_bioequivalence_interval_needs_reference_but_is_not_wrong_auc_unit(tmp_path):
    text = "Caffeine AUC was unchanged (90% confidence interval: 87.9%-102%) in the presence of venlafaxine."
    row = evidence(tmp_path, parameter="AUC", value=None, value_low=87.9, value_high=102, unit="%",
                   source_text=text, source_text_hash=hashlib.sha256(text.encode()).hexdigest())
    decision = evaluate_evidence(row, [])
    assert decision["quality_status"] == "repair_pending"
    assert "parameter_unit_dimension_mismatch" not in decision["fatal_blockers"]
    assert "relative_effect_missing_reference" in decision["repairable_blockers"]



def test_final_quality_quarantines_other_drug_numeric_binding_but_keeps_qualitative_target(tmp_path):
    text = ("Caffeine is negligibly bound to plasma proteins, in contrast to comparator drugs, "
            "which are more than 90% protein bound.")
    row = evidence(
        tmp_path, evidence_type="adme", parameter="protein_binding", value=90.0, unit="%",
        value_qualifier=">", source_text=text,
        source_text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    repaired, repairs = deterministic_repair(row)
    decision = evaluate_evidence(repaired, repairs)
    assert decision["quality_status"] == "quarantined"
    assert "numeric_value_belongs_to_other_substance" in decision["fatal_blockers"]
    qualitative = {**row, "value": None, "unit": None, "value_qualifier": None}
    assert evaluate_evidence(qualitative, [])["quality_status"] == "auto_validated"



def test_final_quality_rejects_percent_range_mislabeled_as_hours(tmp_path):
    text = "Caffeine had an increase in plasma half-life (t(1/2) 14-16%) compared with baseline."
    row = evidence(
        tmp_path, parameter="half_life", value=None, value_low=14, value_high=16, unit="h",
        source_text=text, source_text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    repaired, repairs = deterministic_repair(row)
    decision = evaluate_evidence(repaired, repairs)
    assert decision["quality_status"] == "quarantined"
    assert "numeric_unit_conflicts_with_source_assertion" in decision["fatal_blockers"]
