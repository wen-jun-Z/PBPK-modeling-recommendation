from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from pbpk_kb.recommendation.biorex_relation import (
    BIOREX_LABELS,
    finalize_biorex_signals,
    prepare_biorex_input,
)
from pbpk_kb.recommendation.entity_extraction import EntityCatalog, NullNERBackend, PBPKEntityExtractor
from pbpk_kb.recommendation.entity_link_benchmark import build_locked_silver_examples
from pbpk_kb.recommendation.entity_link_validation import (
    _automatic_dual_source_decision,
    _candidate_options,
    _hard_rejection_reason,
    _validate_model_decision,
)
from pbpk_kb.recommendation.ddi_relation_validation import (
    build_relation_task,
    deterministic_decision,
    validate_model_decision as validate_ddi_model_decision,
)
from pbpk_kb.recommendation.ddi_decision_revalidation import revalidate_decisions
from pbpk_kb.recommendation.complex_entity_extraction import (
    apply_deterministic_type_constraints,
    validate_generation as validate_complex_entity_generation,
)
from pbpk_kb.recommendation.catalog_merge import merge_catalogs
from pbpk_kb.recommendation.entity_resolution import (
    _canonicalize_identifier_punctuation,
    collect_candidates,
    resolve_candidate,
)
from pbpk_kb.recommendation.embedding_benchmark import (
    _evaluate_rrf,
    _rank_metrics,
    prepare_model_input,
)
from pbpk_kb.recommendation.final_kg_pipeline import run_pipeline
from pbpk_kb.recommendation.formal_graph import FormalGraphStore, TraversalPolicy
from pbpk_kb.recommendation.local_llm import StructuredGeneration
from pbpk_kb.recommendation.ner_benchmark import build_silver_examples, evaluate_acceptance
from pbpk_kb.recommendation.ontology import (
    RELATIONS,
    RELATION_COMPACT_GLOSSES,
    RELATION_GLOSSES,
    pk_parameter_semantics,
    validate_relation,
)
from pbpk_kb.recommendation.relation_extraction import PBPKRelationExtractor
from pbpk_kb.recommendation.reranker_benchmark import (
    _candidate_metrics,
    build_candidate_pool,
    graph_constraint_passes,
    infer_query_graph_constraints,
    validate_query_references,
)
from pbpk_kb.recommendation.supervised_relation import (
    build_supervised_pairs,
    evaluate_locked_silver,
    mark_entity_pair,
)
from pbpk_kb.storage.import_recommendation_embeddings import _partial_index_name


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_embedding_models_use_isolated_partial_hnsw_names() -> None:
    first = _partial_index_name("ncbi/MedCPT", "revision-a")
    second = _partial_index_name(
        "NeuML/pubmedbert-base-embeddings",
        "revision-b",
    )

    assert first.startswith("recommendation_embedding_hnsw_")
    assert first.endswith("_idx")
    assert first != second
    assert _partial_index_name("ncbi/MedCPT", "revision-a") == first


def test_model_specific_subgraph_compaction_preserves_critical_lines() -> None:
    class WhitespaceTokenizer:
        @staticmethod
        def __call__(
            text,
            *,
            add_special_tokens,
            truncation,
            padding,
            return_length=False,
            max_length=None,
        ):
            values = text.split()
            if truncation and max_length:
                values = values[: max_length - 2]
            result = {"input_ids": list(range(len(values) + 2))}
            if return_length:
                result["length"] = [len(values) + 2]
            return result

    class Model:
        tokenizer = WhitespaceTokenizer()
        max_seq_length = 80

    class Encoder:
        name = "test-biomedical-encoder"
        model = Model()

    text_value = "\n".join(
        [
            "CENTER [Compound] midazolam",
            "CENTER IDENTIFIERS {PUBCHEM:4192}",
            "HOP LIMIT 2",
            "ONTOLOGY VERSION test",
            "SOURCE FAMILIES pubmed",
            "RELATION TYPES VICTIM_IN=1",
            "RELATION GLOSS VICTIM_IN=The compound whose exposure is altered.",
            "EDGE midazolam -[VICTIM_IN]-> ddi-event {}",
            "EVENT [DDIEvent] trigger=increased exposure",
            "CITATION source=pubmed location=pmid:1 quote=supported",
            *[
                f"NODE [Compound] optional-{index} with many low priority tokens"
                for index in range(30)
            ],
        ]
    )

    prepared, audit = prepare_model_input(
        encoder=Encoder(),
        text=text_value,
        object_kind="subgraph",
    )

    assert audit["compacted"] is True
    assert audit["prepared_token_count"] <= 80
    assert "RELATION GLOSS VICTIM_IN=" in prepared
    assert "\nEDGE " in prepared
    assert "\nEVENT " in prepared
    assert "\nCITATION " in prepared


def evidence_row(
    *,
    evidence_id: str,
    compound_id: str,
    compound_name: str,
    source_text: str,
    evidence_type: str = "clinical_pk",
    parameter: str = "AUC",
    compound_role: str = "target_compound",
    metadata_json: dict | None = None,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "compound_id": compound_id,
        "compound_name": compound_name,
        "compound_role": compound_role,
        "evidence_type": evidence_type,
        "parameter": parameter,
        "value": 100.0 if evidence_type == "clinical_pk" else None,
        "unit": "ng/mL" if evidence_type == "clinical_pk" else None,
        "canonical_value": 100.0 if evidence_type == "clinical_pk" else None,
        "canonical_unit": "ng/mL" if evidence_type == "clinical_pk" else None,
        "route": "oral",
        "dose": "5 mg",
        "population": "healthy adults",
        "source_id": "pubmed",
        "source_location": f"pubmed:{evidence_id}#abstract",
        "source_text": source_text,
        "source_text_hash": hashlib.sha256(source_text.encode()).hexdigest(),
        "raw_asset_id": f"raw-{evidence_id}",
        "storage_path": f"/evidence/{evidence_id}.json",
        "checksum": hashlib.sha256(evidence_id.encode()).hexdigest(),
        "extraction_method": "llm_assisted",
        "extractor_version": "test",
        "extraction_confidence": 0.9,
        "quality_status": "auto_validated",
        "metadata_json": metadata_json or {},
    }


def test_ontology_rejects_invalid_domain_range() -> None:
    assert validate_relation("VICTIM_IN", "Compound", "DDIEvent") is None
    assert validate_relation("VICTIM_IN", "Evidence", "DDIEvent") == (
        "invalid_subject_type:VICTIM_IN:Evidence"
    )


def test_every_relation_has_full_and_compact_versioned_semantics() -> None:
    assert set(RELATION_GLOSSES) == set(RELATIONS)
    assert set(RELATION_COMPACT_GLOSSES) == set(RELATIONS)
    assert all(value.strip() for value in RELATION_COMPACT_GLOSSES.values())


def test_entity_and_event_extraction_preserves_provenance() -> None:
    row = evidence_row(
        evidence_id="e1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="After oral midazolam, AUC was 100 ng/mL in healthy adults.",
    )
    catalog = EntityCatalog([row])
    mentions, unresolved = PBPKEntityExtractor(catalog, NullNERBackend()).extract(row)
    result = PBPKRelationExtractor(catalog).extract(row, mentions)

    assert any(mention.linked_node_id == "Compound:c1" for mention in mentions)
    assert any(event.event_type == "PKMeasurement" for event in result.events)
    assert any(edge.predicate == "OF_COMPOUND" and edge.object_id == "Compound:c1" for edge in result.edges)
    assert all(edge.provenance.evidence_id == "e1" for edge in result.edges)
    assert all(edge.provenance.supporting_text in row["source_text"] for edge in result.edges)
    assert all(item.mention_id for item in unresolved)


def test_ddi_requires_linked_victim_and_perpetrator() -> None:
    ddi = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Repaglinide increased midazolam AUC through inhibition of CYP3A4.",
        evidence_type="ddi",
        parameter="AUC",
        compound_role="victim_drug",
        metadata_json={"perpetrator_drug": "repaglinide"},
    )
    repaglinide = evidence_row(
        evidence_id="pk2",
        compound_id="c2",
        compound_name="repaglinide",
        source_text="Repaglinide AUC was measured after oral dosing.",
    )
    catalog = EntityCatalog([ddi, repaglinide])
    mentions, _ = PBPKEntityExtractor(catalog, NullNERBackend()).extract(ddi)
    result = PBPKRelationExtractor(catalog).extract(ddi, mentions)

    assert len(result.events) == 1
    predicates = {edge.predicate for edge in result.edges}
    assert {"VICTIM_IN", "PERPETRATOR_IN", "MEDIATED_BY", "AFFECTS_PARAMETER"} <= predicates


def test_ddi_uses_unique_non_target_compound_as_counterpart() -> None:
    ddi = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Ketoconazole increased midazolam AUC through inhibition of CYP3A4.",
        evidence_type="ddi",
        parameter="AUC",
        compound_role="victim_drug",
    )
    catalog = EntityCatalog(
        [ddi],
        [
            {
                "node_id": "Compound:PUBLIC:47576",
                "node_type": "Compound",
                "preferred_name": "ketoconazole",
                "aliases": ["ketoconazole"],
                "standard_identifiers": {"PUBCHEM.COMPOUND": "47576"},
            }
        ],
    )
    mentions, _ = PBPKEntityExtractor(catalog, NullNERBackend()).extract(ddi)
    result = PBPKRelationExtractor(catalog).extract(ddi, mentions)

    assert len(result.events) == 1
    assert any(
        edge.predicate == "PERPETRATOR_IN" and edge.subject_id == "Compound:PUBLIC:47576"
        for edge in result.edges
    )


def test_ddi_does_not_guess_between_multiple_non_target_compounds() -> None:
    ddi = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Ketoconazole and ritonavir increased midazolam AUC.",
        evidence_type="ddi",
        parameter="AUC",
        compound_role="victim_drug",
    )
    catalog = EntityCatalog(
        [ddi],
        [
            {
                "node_id": "Compound:PUBLIC:47576",
                "node_type": "Compound",
                "preferred_name": "ketoconazole",
                "aliases": ["ketoconazole"],
                "standard_identifiers": {"PUBCHEM.COMPOUND": "47576"},
            },
            {
                "node_id": "Compound:PUBLIC:392622",
                "node_type": "Compound",
                "preferred_name": "ritonavir",
                "aliases": ["ritonavir"],
                "standard_identifiers": {"PUBCHEM.COMPOUND": "392622"},
            },
        ],
    )
    mentions, _ = PBPKEntityExtractor(catalog, NullNERBackend()).extract(ddi)
    result = PBPKRelationExtractor(catalog).extract(ddi, mentions)

    assert not result.events
    assert result.rejected[0].reasons == ["perpetrator compound was not uniquely linked"]


def test_ddi_relation_task_routes_unique_counterpart_deterministically() -> None:
    row = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Ketoconazole increased midazolam AUC.",
        evidence_type="ddi",
        parameter="AUC",
        compound_role="victim_drug",
    )
    mentions = [
        {"linked_node_id": "Compound:c1", "text": "midazolam"},
        {"linked_node_id": "Compound:PUBLIC:47576", "text": "Ketoconazole"},
        {"linked_node_id": "PKParameter:AUC", "text": "AUC"},
    ]

    task = build_relation_task(row, mentions)
    decision = deterministic_decision(task)

    assert task["routing"] == "deterministic_unique_counterpart"
    assert decision["status"] == "accepted"
    assert decision["events"][0]["victim_node_id"] == "Compound:c1"
    assert decision["events"][0]["perpetrator_node_id"] == "Compound:PUBLIC:47576"


def test_structured_route_creates_regimen_route_edge_without_verbatim_mention() -> None:
    row = evidence_row(
        evidence_id="pk-route",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Midazolam exposure was measured after administration.",
    )
    catalog = EntityCatalog([row])
    mentions, _ = PBPKEntityExtractor(catalog, NullNERBackend()).extract(row)

    result = PBPKRelationExtractor(catalog).extract(row, mentions)

    assert any(edge.predicate == "UNDER_REGIMEN" for edge in result.edges)
    assert any(edge.predicate == "HAS_ROUTE" for edge in result.edges)


def test_route_catalog_normalizes_iv_infusion_and_intranasal() -> None:
    catalog = EntityCatalog([])

    iv, _, _ = catalog.link("AdministrationRoute", "i.v.")
    infusion, _, _ = catalog.link("AdministrationRoute", "IV infusion")
    intranasal, _, _ = catalog.link("AdministrationRoute", "intranasal")

    assert iv is not None and iv.preferred_name == "intravenous"
    assert infusion is not None and infusion.node_id == iv.node_id
    assert intranasal is not None
    assert intranasal.standard_identifiers["SNOMEDCT"] == "260617000"


def test_measurement_uses_structured_unit_and_event_specific_relation() -> None:
    row = evidence_row(
        evidence_id="pk-unit",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Midazolam AUC was measured after administration.",
    )
    catalog = EntityCatalog([row])
    mentions, _ = PBPKEntityExtractor(catalog, NullNERBackend()).extract(row)

    result = PBPKRelationExtractor(catalog).extract(row, mentions)

    assert any(edge.predicate == "HAS_UNIT" for edge in result.edges)
    assert any(edge.predicate == "HAS_MEASUREMENT" for edge in result.edges)


def test_literature_reported_model_is_not_marked_reusable() -> None:
    row = evidence_row(
        evidence_id="model-report",
        compound_id="c1",
        compound_name="midazolam",
        source_text="A PBPK model of midazolam was developed.",
        evidence_type="public_pbpk_model",
        parameter="PBPK model",
    )
    catalog = EntityCatalog([row])
    mentions, _ = PBPKEntityExtractor(catalog, NullNERBackend()).extract(row)

    result = PBPKRelationExtractor(catalog).extract(row, mentions)

    model = next(node for node in result.nodes if node.node_type == "PBPKModel")
    assert model.properties["availability_status"] == "literature_reported"
    assert model.properties["reusable_candidate"] is False
    assert "MODEL_CHECKSUM" not in model.standard_identifiers


def test_ddi_relation_validator_rejects_invented_node_and_quote() -> None:
    row = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Ketoconazole and ritonavir increased midazolam AUC.",
        evidence_type="ddi",
        parameter="AUC",
        compound_role="victim_drug",
    )
    task = build_relation_task(
        row,
        [
            {"linked_node_id": "Compound:c1", "text": "midazolam"},
            {"linked_node_id": "Compound:PUBLIC:47576", "text": "Ketoconazole"},
            {"linked_node_id": "Compound:PUBLIC:392622", "text": "ritonavir"},
            {"linked_node_id": "PKParameter:AUC", "text": "AUC"},
        ],
    )
    generation = StructuredGeneration(
        parsed={
            "events": [
                {
                    "victim_node_id": "Compound:c1",
                    "perpetrator_node_id": "Compound:INVENTED",
                    "supporting_quote": "A paraphrase not in the source.",
                    "interaction_effect": "increased",
                    "parameter_node_ids": ["PKParameter:AUC"],
                    "mediator_node_ids": [],
                    "confidence": 1.0,
                    "reason": "invented",
                }
            ]
        },
        raw_text='{"events":[]}',
        error=None,
    )

    decision = validate_ddi_model_decision(task, generation)

    assert decision["status"] == "unresolved"
    errors = decision["rejected_events"][0]["validation_errors"]
    assert "perpetrator_not_in_candidate_set" in errors
    assert "supporting_quote_not_verbatim" in errors


def test_ddi_complex_routing_requires_positive_supervised_target_pair() -> None:
    row = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Ketoconazole and ritonavir increased midazolam AUC.",
        evidence_type="ddi",
        parameter="AUC",
        compound_role="victim_drug",
    )
    mentions = [
        {"linked_node_id": "Compound:c1", "text": "midazolam"},
        {"linked_node_id": "Compound:PUBLIC:47576", "text": "Ketoconazole"},
        {"linked_node_id": "Compound:PUBLIC:392622", "text": "ritonavir"},
    ]
    rejected = build_relation_task(
        row,
        mentions,
        [
            {
                "pair_id": "non-target-pair",
                "entity1_node_id": "Compound:PUBLIC:47576",
                "entity2_node_id": "Compound:PUBLIC:392622",
                "positive_signal": True,
            }
        ],
        supervised_gate_required=True,
    )
    accepted = build_relation_task(
        row,
        mentions,
        [
            {
                "pair_id": "target-pair",
                "entity1_node_id": "Compound:c1",
                "entity2_node_id": "Compound:PUBLIC:47576",
                "positive_signal": True,
                "positive_probability": 0.95,
            }
        ],
        supervised_gate_required=True,
    )

    assert rejected["routing"] == "unresolved_supervised_re_no_positive_target_pair"
    assert accepted["routing"] == "llm_complex"
    assert accepted["supervised_candidate_pairs"][0]["counterpart_node_id"] == (
        "Compound:PUBLIC:47576"
    )


def test_ddi_validator_rejects_pair_outside_supervised_gate() -> None:
    row = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Ketoconazole and ritonavir increased midazolam AUC.",
        evidence_type="ddi",
        parameter="AUC",
        compound_role="victim_drug",
    )
    task = build_relation_task(
        row,
        [
            {"linked_node_id": "Compound:c1", "text": "midazolam"},
            {"linked_node_id": "Compound:PUBLIC:47576", "text": "Ketoconazole"},
            {"linked_node_id": "Compound:PUBLIC:392622", "text": "ritonavir"},
            {"linked_node_id": "PKParameter:AUC", "text": "AUC"},
        ],
        [
            {
                "pair_id": "target-ketoconazole",
                "entity1_node_id": "Compound:c1",
                "entity2_node_id": "Compound:PUBLIC:47576",
                "positive_signal": True,
                "positive_probability": 0.95,
            }
        ],
        supervised_gate_required=True,
    )
    generation = StructuredGeneration(
        parsed={
            "events": [
                {
                    "victim_node_id": "Compound:c1",
                    "perpetrator_node_id": "Compound:PUBLIC:392622",
                    "supporting_quote": row["source_text"],
                    "interaction_effect": "increased",
                    "parameter_node_ids": ["PKParameter:AUC"],
                    "mediator_node_ids": [],
                    "confidence": 0.95,
                    "reason": "not accepted by supervised gate",
                }
            ]
        },
        raw_text='{"events":[]}',
        error=None,
    )

    decision = validate_ddi_model_decision(task, generation)

    assert decision["status"] == "unresolved"
    assert "pair_not_accepted_by_supervised_relation_gate" in (
        decision["rejected_events"][0]["validation_errors"]
    )


def test_ddi_revalidation_binds_supervised_candidate_pairs(tmp_path: Path) -> None:
    task = {
        "evidence_id": "ddi1",
        "source_text": "Ketoconazole increased midazolam AUC.",
        "source_text_hash": "a" * 64,
        "target_node_id": "Compound:c1",
        "target_role": "victim_drug",
        "candidate_compounds": [
            {"node_id": "Compound:c1", "mentions": ["midazolam"]},
            {"node_id": "Compound:ketoconazole", "mentions": ["Ketoconazole"]},
        ],
        "parameter_candidates": [{"node_id": "PKParameter:AUC", "mention": "AUC"}],
        "mediator_candidates": [],
        "supervised_candidate_pairs": [
            {
                "pair_id": "pair1",
                "target_node_id": "Compound:c1",
                "counterpart_node_id": "Compound:ketoconazole",
            }
        ],
        "supervised_gate_required": True,
        "routing": "llm_complex",
    }
    decision = {
        "evidence_id": "ddi1",
        "status": "accepted",
        "decision_method": "qwen_structured_relation_extraction",
        "events": [
            {
                "victim_node_id": "Compound:c1",
                "perpetrator_node_id": "Compound:ketoconazole",
                "supporting_quote": task["source_text"],
                "interaction_effect": "increased",
                "parameter_node_ids": ["PKParameter:AUC"],
                "mediator_node_ids": [],
                "confidence": 0.95,
                "reason": "validated",
                "validation_status": "auto_validated",
            }
        ],
        "rejected_events": [],
        "raw_model_output": "{}",
        "validation_errors": [],
        "source_text_hash": task["source_text_hash"],
        "prompt_version": "test",
        "validator_version": "test",
        "supervised_relation_signals": [],
    }
    tasks_path = tmp_path / "tasks.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    decisions_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")

    summary = revalidate_decisions(
        tasks_path=tasks_path,
        decisions_path=decisions_path,
        output_dir=tmp_path / "out",
    )

    assert summary["accepted"] is True
    row = json.loads(
        (tmp_path / "out" / "validated_ddi_relation_decisions.jsonl").read_text()
    )
    assert row["supervised_candidate_pairs"][0]["pair_id"] == "pair1"
    assert row["events"][0]["revalidation_version"]


def test_validated_ddi_decision_can_create_multiple_events() -> None:
    ddi = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Ketoconazole and ritonavir increased midazolam AUC.",
        evidence_type="ddi",
        parameter="AUC",
        compound_role="victim_drug",
    )
    external = [
        {
            "node_id": "Compound:PUBLIC:47576",
            "node_type": "Compound",
            "preferred_name": "ketoconazole",
            "aliases": ["ketoconazole"],
            "standard_identifiers": {"PUBCHEM.COMPOUND": "47576"},
        },
        {
            "node_id": "Compound:PUBLIC:392622",
            "node_type": "Compound",
            "preferred_name": "ritonavir",
            "aliases": ["ritonavir"],
            "standard_identifiers": {"PUBCHEM.COMPOUND": "392622"},
        },
    ]
    catalog = EntityCatalog([ddi], external)
    mentions, _ = PBPKEntityExtractor(catalog, NullNERBackend()).extract(ddi)
    decision = {
        "evidence_id": "ddi1",
        "status": "accepted",
        "decision_method": "qwen_structured_relation_extraction",
        "prompt_version": "p1",
        "validator_version": "v1",
        "source_text_hash": ddi["source_text_hash"],
        "events": [
            {
                "victim_node_id": "Compound:c1",
                "perpetrator_node_id": node_id,
                "supporting_quote": ddi["source_text"],
                "interaction_effect": "increased",
                "parameter_node_ids": ["PKParameter:AUC"],
                "mediator_node_ids": [],
                "confidence": 0.95,
                "validation_status": "auto_validated",
            }
            for node_id in ("Compound:PUBLIC:47576", "Compound:PUBLIC:392622")
        ],
    }

    result = PBPKRelationExtractor(catalog, {"ddi1": decision}).extract(ddi, mentions)

    assert len(result.events) == 2
    assert {edge.subject_id for edge in result.edges if edge.predicate == "PERPETRATOR_IN"} == {
        "Compound:PUBLIC:47576",
        "Compound:PUBLIC:392622",
    }


def test_supervised_relation_pair_markers_preserve_verbatim_entities() -> None:
    text = "Ketoconazole increased midazolam AUC."
    marked = mark_entity_pair(
        text,
        {"start_char": 0, "end_char": 12},
        {"start_char": 23, "end_char": 32},
    )

    assert marked == "[E1]Ketoconazole[/E1] increased [E2]midazolam[/E2] AUC."


def test_supervised_relation_silver_labels_target_and_non_target_pairs() -> None:
    row = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Ketoconazole and ritonavir increased midazolam AUC.",
        evidence_type="ddi",
        parameter="AUC",
        compound_role="victim_drug",
    )
    pairs = build_supervised_pairs(
        row,
        [
            {"linked_node_id": "Compound:PUBLIC:47576", "text": "Ketoconazole", "start_char": 0, "end_char": 12},
            {"linked_node_id": "Compound:PUBLIC:392622", "text": "ritonavir", "start_char": 17, "end_char": 26},
            {"linked_node_id": "Compound:c1", "text": "midazolam", "start_char": 37, "end_char": 46},
        ],
    )

    assert Counter(pair["silver_label"] for pair in pairs) == Counter(
        {"unlabeled": 2, "hard_negative": 1}
    )


def test_supervised_relation_benchmark_fails_when_negative_pairs_are_accepted() -> None:
    rows = [
        {"silver_label": "positive", "positive_signal": True},
        {"silver_label": "hard_negative", "positive_signal": True},
    ]

    report = evaluate_locked_silver(rows)

    assert report["accepted"] is False
    assert report["metrics"]["positive_recall"] == 1.0
    assert report["metrics"]["hard_negative_specificity"] == 0.0


def test_complex_entity_extraction_keeps_specific_compound_and_rejects_target() -> None:
    task = {
        "evidence_id": "ddi1",
        "source_text": "Rilaciclib increased metformin exposure.",
        "source_text_hash": hashlib.sha256(
            b"Rilaciclib increased metformin exposure."
        ).hexdigest(),
        "target_name": "metformin",
    }
    generation = StructuredGeneration(
        parsed={
            "entities": [
                {
                    "mention_text": "Rilaciclib",
                    "entity_type": "specific_compound",
                    "role": "perpetrator_drug",
                    "supporting_quote": "Rilaciclib increased metformin exposure.",
                    "confidence": 0.95,
                    "reason": "named non-target drug",
                },
                {
                    "mention_text": "metformin",
                    "entity_type": "specific_compound",
                    "role": "victim_drug",
                    "supporting_quote": "Rilaciclib increased metformin exposure.",
                    "confidence": 0.99,
                    "reason": "target",
                },
            ]
        },
        raw_text='{"entities":[]}',
        error=None,
    )

    decision = validate_complex_entity_generation(task, generation)

    assert [item["mention_text"] for item in decision["validated_entities"]] == ["Rilaciclib"]
    assert "target_compound_cannot_be_counterpart_candidate" in (
        decision["rejected_entities"][0]["validation_errors"]
    )


def test_complex_entity_constraints_reclassify_abbreviation_and_drug_class() -> None:
    abbreviation, abbreviation_reason = apply_deterministic_type_constraints(
        {"mention_text": "RGE", "entity_type": "specific_compound"}
    )
    drug_class, drug_class_reason = apply_deterministic_type_constraints(
        {"mention_text": "Quinolone", "entity_type": "specific_compound"}
    )
    named_compound, named_reason = apply_deterministic_type_constraints(
        {"mention_text": "PF-06835919", "entity_type": "specific_compound"}
    )

    assert abbreviation["entity_type"] == "unknown_abbreviation"
    assert abbreviation_reason == "short_unexpanded_uppercase_token_reclassified"
    assert drug_class["entity_type"] == "drug_class"
    assert drug_class_reason == "drug_class_term_reclassified"
    assert named_compound["entity_type"] == "specific_compound"
    assert named_reason is None


def test_catalog_merge_rejects_ambiguous_aliases(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps(
            {
                "node_id": "Compound:1",
                "node_type": "Compound",
                "preferred_name": "same name",
                "aliases": ["same name"],
                "standard_identifiers": {"PUBCHEM.COMPOUND": "1"},
            }
        )
        + "\n"
    )
    second.write_text(
        json.dumps(
            {
                "node_id": "Compound:2",
                "node_type": "Compound",
                "preferred_name": "same name",
                "aliases": ["same name"],
                "standard_identifiers": {"PUBCHEM.COMPOUND": "2"},
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="Ambiguous aliases"):
        merge_catalogs(
            [first, second],
            tmp_path / "merged.jsonl",
            tmp_path / "summary.json",
        )


def test_pipeline_writes_formal_graph_and_manifest(tmp_path: Path) -> None:
    row = evidence_row(
        evidence_id="e1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="After oral midazolam, AUC was 100 ng/mL in healthy adults.",
    )
    input_path = tmp_path / "final_auto_validated_evidence.jsonl"
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_dir = tmp_path / "08_recommendation_kg"

    summary = run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        run_id="test-run",
        ner_backend="none",
        ner_model="unused",
        ner_threshold=0.45,
        device="cpu",
        reset=True,
    )

    assert summary["accepted"] is True
    assert summary["evidence_count"] == 1
    assert summary["event_count"] == 1
    assert (output_dir / "kg_nodes.jsonl").exists()
    assert (output_dir / "kg_edges.jsonl").exists()
    assert (output_dir / "kg_events.jsonl").exists()
    assert (output_dir / "kg_extraction_manifest.json").exists()


def test_ner_silver_benchmark_is_deterministic_and_verbatim() -> None:
    rows = [
        evidence_row(
            evidence_id="e1",
            compound_id="c1",
            compound_name="midazolam",
            source_text="After oral midazolam, AUC was 100 ng/mL.",
        ),
        evidence_row(
            evidence_id="e2",
            compound_id="c2",
            compound_name="caffeine",
            source_text="Caffeine clearance was reported.",
            parameter="clearance",
        ),
    ]
    first = build_silver_examples(rows, per_type=10)
    second = build_silver_examples(list(reversed(rows)), per_type=10)

    assert first == second
    assert all(
        example["text"][mention["start"] : mention["end"]] == mention["text"]
        for example in first
        for mention in example["expected"]
    )


def test_external_compound_catalog_links_exact_alias_and_keeps_evidence_units() -> None:
    row = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Ketoconazole increased midazolam AUC measured in ng/mL.",
        evidence_type="ddi",
        parameter="AUC",
    )
    catalog = EntityCatalog(
        [row],
        [
            {
                "node_id": "Compound:PUBLIC:456201",
                "node_type": "Compound",
                "preferred_name": "ketoconazole",
                "aliases": ["Ketoconazole"],
                "standard_identifiers": {"PUBCHEM.COMPOUND": "456201"},
            }
        ],
    )

    compound, method, confidence = catalog.link("Compound", "Ketoconazole")
    unit, _, _ = catalog.link("Unit", "ng/mL")

    assert compound is not None
    assert compound.node_id == "Compound:PUBLIC:456201"
    assert method == "exact_controlled_alias"
    assert confidence == 1.0
    assert unit is not None
    assert unit.node_type == "Unit"


def test_external_identity_is_merged_into_existing_target_compound() -> None:
    row = evidence_row(
        evidence_id="pk1",
        compound_id="internal-midazolam",
        compound_name="midazolam",
        source_text="Midazolam AUC was measured.",
    )
    catalog = EntityCatalog(
        [row],
        [
            {
                "node_id": "Compound:PUBLIC:4192",
                "node_type": "Compound",
                "preferred_name": "Midazolam",
                "aliases": ["midazolam"],
                "standard_identifiers": {
                    "PUBCHEM.COMPOUND": "4192",
                    "RXNORM": "6960",
                },
            }
        ],
    )

    compound, method, _ = catalog.link("Compound", "midazolam")

    assert compound is not None
    assert compound.node_id == "Compound:internal-midazolam"
    assert compound.standard_identifiers["PUBCHEM.COMPOUND"] == "4192"
    assert compound.standard_identifiers["RXNORM"] == "6960"
    assert method == "exact_controlled_alias"
    assert "Compound:PUBLIC:4192" not in catalog.entries


def test_resolution_candidates_are_limited_to_ddi_compounds() -> None:
    ddi = evidence_row(
        evidence_id="ddi1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="Ketoconazole increased midazolam AUC.",
        evidence_type="ddi",
        metadata_json={"perpetrator_drug": "ketoconazole"},
    )
    pk = evidence_row(
        evidence_id="pk1",
        compound_id="c1",
        compound_name="midazolam",
        source_text="A placebo and midazolam were compared.",
    )
    unresolved = [
        {
            "evidence_id": "ddi1",
            "entity_type": "Compound",
            "mention_text": "Ketoconazole",
            "recognition_confidence": 0.95,
        },
        {
            "evidence_id": "pk1",
            "entity_type": "Compound",
            "mention_text": "placebo",
            "recognition_confidence": 0.99,
        },
    ]

    candidates = collect_candidates([ddi, pk], unresolved, min_confidence=0.75)

    assert [item["normalized_name"] for item in candidates] == ["ketoconazole"]
    assert set(candidates[0]["sources"]) == {"ddi_ner_mention", "structured_perpetrator_drug"}


def test_resolver_never_accepts_an_ambiguous_authoritative_match() -> None:
    candidate = {
        "name": "ambiguous drug",
        "normalized_name": "ambiguous drug",
        "evidence_ids": ["ddi1"],
        "sources": ["ddi_ner_mention"],
        "roles": [],
    }
    with (
        patch(
            "pbpk_kb.recommendation.entity_resolution.resolve_pubchem",
            return_value={"status": "ambiguous", "reason": "multiple_exact_pubchem_cids"},
        ),
        patch(
            "pbpk_kb.recommendation.entity_resolution.resolve_rxnorm",
            return_value={"status": "unresolved", "reason": "no_exact_rxnorm_concept"},
        ),
    ):
        decision = resolve_candidate(candidate, object())

    assert decision["status"] == "ambiguous"
    assert "catalog_entry" not in decision


def test_entity_resolution_normalizes_unicode_identifier_dashes_only() -> None:
    assert _canonicalize_identifier_punctuation("PF‐06835919") == "PF-06835919"
    assert _canonicalize_identifier_punctuation("rilaciclib") == "rilaciclib"


def test_dual_source_entity_link_requires_verbatim_context() -> None:
    resolver_decision = {
        "name": "ketoconazole",
        "roles": ["perpetrator_drug"],
        "pubchem": {"status": "resolved"},
        "rxnorm": {"status": "resolved"},
        "catalog_entry": {
            "node_id": "Compound:PUBLIC:47576",
            "node_type": "Compound",
            "preferred_name": "Ketoconazole",
            "aliases": ["ketoconazole"],
            "standard_identifiers": {
                "PUBCHEM.COMPOUND": "47576",
                "RXNORM": "6135",
            },
        },
    }
    candidates = _candidate_options(resolver_decision)

    accepted = _automatic_dual_source_decision(
        resolver_decision,
        [{"evidence_id": "ddi1", "text": "Ketoconazole increased midazolam AUC."}],
        candidates,
    )
    missing_context = _automatic_dual_source_decision(
        resolver_decision,
        [{"evidence_id": "ddi1", "text": "The inhibitor increased midazolam AUC."}],
        candidates,
    )

    assert accepted is not None
    assert accepted["decision"] == "accepted"
    assert accepted["selected_candidate_id"] == "Compound:PUBLIC:47576"
    assert missing_context is None


def test_context_reranker_fails_closed_on_nonverbatim_quote() -> None:
    result = StructuredGeneration(
        parsed={
            "decision": "accepted",
            "selected_candidate_id": "Compound:PUBLIC:47576",
            "supporting_quote": "A paraphrased statement.",
            "entity_role": "perpetrator_drug",
            "confidence": 0.99,
            "reason": "matches",
        },
        raw_text='{"decision":"accepted"}',
        error=None,
    )
    decision = _validate_model_decision(
        result,
        mention="ketoconazole",
        contexts=[{"evidence_id": "ddi1", "text": "Ketoconazole increased midazolam AUC."}],
        candidates=[
            {
                "candidate_id": "Compound:PUBLIC:47576",
                "preferred_name": "Ketoconazole",
                "standard_identifiers": {"PUBCHEM.COMPOUND": "47576"},
                "resolver_sources": ["PubChem"],
                "catalog_entry": {},
            }
        ],
        min_confidence=0.8,
    )

    assert decision["decision"] == "unresolved"
    assert "supporting_quote_not_verbatim" in decision["validation_errors"]
    assert "mention_missing_from_quote" in decision["validation_errors"]


def test_context_reranker_rejects_selected_candidate_name_mismatch() -> None:
    result = StructuredGeneration(
        parsed={
            "decision": "accepted",
            "selected_candidate_id": "Compound:PUBLIC:104865",
            "supporting_quote": "Atazanavir exposure was measured.",
            "entity_role": "drug",
            "confidence": 1.0,
            "reason": "accepted",
        },
        raw_text='{"decision":"accepted"}',
        error=None,
    )
    decision = _validate_model_decision(
        result,
        mention="atazanavir",
        contexts=[{"evidence_id": "ddi1", "text": "Atazanavir exposure was measured."}],
        candidates=[
            {
                "candidate_id": "Compound:PUBLIC:104865",
                "preferred_name": "Bosentan",
                "standard_identifiers": {"PUBCHEM.COMPOUND": "104865"},
                "resolver_sources": ["PubChem"],
                "catalog_entry": {
                    "preferred_name": "Bosentan",
                    "aliases": ["bosentan"],
                },
            }
        ],
        min_confidence=0.8,
    )

    assert decision["decision"] == "unresolved"
    assert "selected_candidate_name_mismatch" in decision["validation_errors"]


def test_entity_link_silver_benchmark_pairs_positive_and_hard_negative() -> None:
    decisions = []
    evidence = {}
    for index, name in enumerate(("ketoconazole", "midazolam"), start=1):
        evidence_id = f"ddi{index}"
        evidence[evidence_id] = {"evidence_id": evidence_id, "source_text": f"{name} was administered."}
        decisions.append(
            {
                "name": name,
                "normalized_name": name,
                "evidence_ids": [evidence_id],
                "roles": ["perpetrator_drug"],
                "pubchem": {"status": "resolved"},
                "rxnorm": {"status": "resolved"},
                "catalog_entry": {
                    "node_id": f"Compound:PUBLIC:{index}",
                    "node_type": "Compound",
                    "preferred_name": name,
                    "aliases": [name],
                    "standard_identifiers": {
                        "PUBCHEM.COMPOUND": str(index),
                        "RXNORM": str(index),
                    },
                },
            }
        )

    examples = build_locked_silver_examples(decisions, evidence, positive_count=2)

    assert len(examples) == 4
    assert [item["label"] for item in examples] == [
        "accepted",
        "unresolved",
        "accepted",
        "unresolved",
    ]
    assert examples[1]["candidates"][0]["candidate_id"] != examples[0]["expected_candidate_id"]


def test_noncompound_and_unexpanded_terms_are_rejected_before_llm_linking() -> None:
    assert _hard_rejection_reason("quinolone") == "drug_class_term_cannot_enter_compound_nodes"
    assert _hard_rejection_reason("RGE") == "unexpanded_short_abbreviation_requires_unique_context_resolution"
    assert (
        _hard_rejection_reason("goldenseal extract")
        == "botanical_or_food_product_requires_noncompound_ontology_resolution"
    )
    assert _hard_rejection_reason("ABT-335") is None
    assert _hard_rejection_reason("ketoconazole") is None


def test_ner_gate_requires_compound_and_pk_parameter_recall() -> None:
    passing = {
        "expected_count": 200,
        "exact_recall": 0.82,
        "overlap_type_recall": 0.90,
        "per_type": {
            "Compound": {"exact_recall": 0.93},
            "PKParameter": {"exact_recall": 0.68},
        },
    }
    failing = {
        **passing,
        "per_type": {
            "Compound": {"exact_recall": 0.93},
            "PKParameter": {"exact_recall": 0.40},
        },
    }

    assert all(evaluate_acceptance(passing).values())
    assert evaluate_acceptance(failing)["pk_parameter_exact_recall"] is False


def test_official_biorex_adapter_preserves_pair_provenance(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.jsonl"
    mentions_path = tmp_path / "mentions.jsonl"
    evidence = {
        "evidence_id": "ev-ddi",
        "evidence_type": "ddi",
        "compound_id": "midazolam",
        "compound_role": "victim",
        "source_text": "Ketoconazole increased midazolam exposure.",
        "source_text_hash": "source-hash",
    }
    evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    mentions = [
        {
            "evidence_id": "ev-ddi",
            "linked_node_id": "Compound:ketoconazole",
            "text": "Ketoconazole",
            "start_char": 0,
            "end_char": 12,
        },
        {
            "evidence_id": "ev-ddi",
            "linked_node_id": "Compound:midazolam",
            "text": "midazolam",
            "start_char": 23,
            "end_char": 32,
        },
    ]
    mentions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in mentions),
        encoding="utf-8",
    )
    prepared_dir = tmp_path / "prepared"
    summary = prepare_biorex_input(
        evidence_path=evidence_path,
        mentions_path=mentions_path,
        output_dir=prepared_dir,
    )
    assert summary["pair_count"] == 1
    mapping = json.loads((prepared_dir / "biorex_pair_mapping.jsonl").read_text())
    document_id = mapping["document_id"]
    processed_path = tmp_path / "processed.tsv"
    processed_path.write_text(
        "\t".join(
            [
                document_id,
                "unused",
                "unused",
                mapping["pubtator_entity1_id"],
                mapping["pubtator_entity2_id"],
                "True",
                "0",
                "marked text",
                "neighbors",
                "None",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logits = [-10.0] * len(BIOREX_LABELS)
    logits[BIOREX_LABELS.index("Drug_Interaction")] = 10.0
    logits_path = tmp_path / "logits.tsv"
    logits_path.write_text(
        "\t".join(str(value) for value in logits) + "\n",
        encoding="utf-8",
    )
    result = finalize_biorex_signals(
        prepared_dir=prepared_dir,
        processed_tsv_path=processed_path,
        logits_path=logits_path,
        model_config_path=None,
        output_dir=tmp_path / "finalized",
    )
    assert result["pair_count"] == 1
    signal = json.loads(
        (tmp_path / "finalized" / "official_biorex_relation_signals.jsonl").read_text()
    )
    assert signal["pair_id"] == mapping["pair_id"]
    assert signal["positive_signal"] is True
    assert signal["model_commit"] == "1f199d8c359eaceb5469538f4daa24aac44c4eae"


def _formal_graph_fixture(tmp_path: Path) -> tuple[Path, Path]:
    graph_dir = tmp_path / "formal-graph"
    graph_dir.mkdir()
    source_text = "Ketoconazole increased midazolam AUC."
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()
    nodes = [
        {
            "node_id": "Compound:midazolam",
            "node_type": "Compound",
            "preferred_name": "midazolam",
            "standard_identifiers": {"PUBCHEM.COMPOUND": "4192"},
            "properties": {"aliases": ["midazolam"]},
            "ontology_version": "pbpk-recommendation-ontology-1.0.0",
        },
        {
            "node_id": "event:ddi",
            "node_type": "DDIEvent",
            "preferred_name": "DDI event",
            "standard_identifiers": {},
            "properties": {},
            "ontology_version": "pbpk-recommendation-ontology-1.0.0",
        },
        {
            "node_id": "PKParameter:AUC",
            "node_type": "PKParameter",
            "preferred_name": "AUC",
            "standard_identifiers": {"PBPKKB.PARAMETER": "AUC"},
            "properties": {},
            "ontology_version": "pbpk-recommendation-ontology-1.0.0",
        },
        {
            "node_id": "evidence:ddi",
            "node_type": "Evidence",
            "preferred_name": "ddi:AUC",
            "standard_identifiers": {"PBPKKB": "evidence:ddi"},
            "properties": {
                "source_id": "pubmed",
                "source_location": "pubmed:1#abstract",
                "source_text_hash": source_hash,
                "quality_status": "auto_validated",
            },
            "ontology_version": "pbpk-recommendation-ontology-1.0.0",
        },
        {
            "node_id": "pbpk-model:reported",
            "node_type": "PBPKModel",
            "preferred_name": "literature model",
            "standard_identifiers": {"PBPKKB.EVIDENCE": "e1"},
            "properties": {
                "availability_status": "literature_reported",
                "reusable_candidate": False,
            },
            "ontology_version": "pbpk-recommendation-ontology-1.0.0",
        },
    ]
    provenance = {
        "evidence_id": "evidence:ddi",
        "source_id": "pubmed",
        "source_location": "pubmed:1#abstract",
        "supporting_text": source_text,
        "source_text_hash": source_hash,
        "extraction_method": "test",
        "extractor_version": "test",
        "confidence": 0.9,
        "quality_status": "auto_validated",
        "negation_status": "affirmed",
        "speculation_status": "asserted",
        "context_id": "context:test",
    }
    edges = [
        {
            "edge_id": "edge:victim",
            "subject_id": "Compound:midazolam",
            "subject_type": "Compound",
            "predicate": "VICTIM_IN",
            "object_id": "event:ddi",
            "object_type": "DDIEvent",
            "provenance": provenance,
            "properties": {},
            "ontology_version": "pbpk-recommendation-ontology-1.0.0",
        },
        {
            "edge_id": "edge:parameter",
            "subject_id": "event:ddi",
            "subject_type": "DDIEvent",
            "predicate": "AFFECTS_PARAMETER",
            "object_id": "PKParameter:AUC",
            "object_type": "PKParameter",
            "provenance": provenance,
            "properties": {},
            "ontology_version": "pbpk-recommendation-ontology-1.0.0",
        },
        {
            "edge_id": "edge:support",
            "subject_id": "event:ddi",
            "subject_type": "DDIEvent",
            "predicate": "SUPPORTED_BY",
            "object_id": "evidence:ddi",
            "object_type": "Evidence",
            "provenance": provenance,
            "properties": {},
            "ontology_version": "pbpk-recommendation-ontology-1.0.0",
        },
    ]
    events = [
        {
            "event_id": "event:ddi",
            "event_type": "DDIEvent",
            "evidence_id": "evidence:ddi",
            "trigger_text": "increased",
            "argument_edge_ids": ["edge:victim", "edge:parameter", "edge:support"],
            "context": {},
            "extraction_method": "test",
            "extractor_version": "test",
            "confidence": 0.9,
            "validation_status": "validated",
            "validation_reasons": [],
        }
    ]
    _write_jsonl(graph_dir / "kg_nodes.jsonl", nodes)
    _write_jsonl(graph_dir / "kg_edges.jsonl", edges)
    _write_jsonl(graph_dir / "kg_events.jsonl", events)
    (graph_dir / "kg_extraction_manifest.json").write_text(
        json.dumps({"accepted": True}), encoding="utf-8"
    )
    (graph_dir / "kg_build_summary.json").write_text(
        json.dumps({"accepted": True}), encoding="utf-8"
    )
    manifest_hash = hashlib.sha256(
        (graph_dir / "kg_extraction_manifest.json").read_bytes()
    ).hexdigest()
    audit_path = tmp_path / "gate1-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "inputs": {"graph_manifest": {"sha256": manifest_hash}},
            }
        ),
        encoding="utf-8",
    )
    return graph_dir, audit_path


def test_formal_graph_uses_true_adjacency_and_reconstructable_paths(tmp_path: Path) -> None:
    graph_dir, audit_path = _formal_graph_fixture(tmp_path)
    store = FormalGraphStore(graph_dir, audit_path)

    traversal = store.traverse(
        "Compound:midazolam",
        policy=TraversalPolicy(max_hops=2, max_nodes=20, max_edges=20),
        objective="ddi_victim",
    )
    one_hop = store.serialize_subgraph(traversal, 1)
    two_hop = store.serialize_subgraph(traversal, 2)

    assert {node["node_id"] for node in one_hop["nodes"]} == {
        "Compound:midazolam",
        "event:ddi",
    }
    assert {"PKParameter:AUC", "evidence:ddi"} <= {
        node["node_id"] for node in two_hop["nodes"]
    }
    evidence_path = next(
        path for path in traversal["paths"] if path["target_id"] == "evidence:ddi"
    )
    assert evidence_path["node_ids"] == [
        "Compound:midazolam",
        "event:ddi",
        "evidence:ddi",
    ]
    assert evidence_path["edge_ids"] == ["edge:victim", "edge:support"]
    assert two_hop["citations"][0]["source_location"] == "pubmed:1#abstract"
    assert "VICTIM_IN=DDI affected victim drug" in two_hop["embedding_text"]
    assert "RELATION GLOSS VICTIM_IN=" in two_hop["embedding_text"]
    assert "SOURCE FAMILIES pubmed" in two_hop["embedding_text"]
    assert "CITATION source=pubmed location=pubmed:1#abstract" in two_hop["embedding_text"]
    assert two_hop["ontology_semantics"]["relation_definitions"]["VICTIM_IN"]


def test_pk_parameter_ontology_semantics_expand_abbreviations() -> None:
    tpsa = pk_parameter_semantics("TPSA")
    xlogp = pk_parameter_semantics("XLogP")

    assert tpsa is not None
    assert "topological polar surface area" in tpsa[0]
    assert "polar atoms" in tpsa[1]
    assert xlogp is not None
    assert "octanol-water partition coefficient" in xlogp[0]


def test_formal_graph_excludes_nonreusable_literature_model_center(tmp_path: Path) -> None:
    graph_dir, audit_path = _formal_graph_fixture(tmp_path)
    store = FormalGraphStore(graph_dir, audit_path)

    eligible_ids = {node["node_id"] for node in store.eligible_centers()}

    assert "Compound:midazolam" in eligible_ids
    assert "pbpk-model:reported" not in eligible_ids


def test_embedding_benchmark_reports_true_recall_and_hit_rate() -> None:
    metrics = _rank_metrics(
        scores=[0.9, 0.8, 0.7, 0.6],
        corpus_ids=["positive-a", "negative", "positive-b", "negative-2"],
        positives={"positive-a", "positive-b"},
    )

    assert metrics["hit_rate_at_1"] == 1.0
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_5"] == 1.0
    assert metrics["reciprocal_rank"] == 1.0


def test_rrf_combines_dense_and_lexical_rankings_without_score_calibration() -> None:
    query = {
        "query_id": "q1",
        "query_text": "victim drug exposure interaction",
        "category": "ddi_role",
        "target_hop": 2,
        "positive_subgraph_ids": ["positive"],
    }
    dense_result = {
        "model": "biomedical-model",
        "revision": "rev-1",
        "dimension": 768,
    }

    result = _evaluate_rrf(
        dense_result=dense_result,
        dense_rankings={"q1": ["dense-only", "positive", "lexical-only"]},
        lexical_rankings={"q1": ["lexical-only", "positive", "dense-only"]},
        queries=[query],
        k=60,
    )

    assert result["retrieval_method"] == "dense_lexical_rrf"
    assert result["overall"]["hit_rate_at_5"] == 1.0
    assert result["dense_model"] == "biomedical-model"


def test_multi_retriever_candidate_pool_is_deterministic_and_auditable() -> None:
    queries = [
        {
            "query_id": "q1",
            "category": "ddi_role",
            "target_hop": 2,
            "positive_subgraph_ids": ["positive"],
        }
    ]
    rankings = {
        "dense-a": {"q1": ["a", "positive", "b"]},
        "dense-b": {"q1": ["positive", "c", "a"]},
        "lexical": {"q1": ["c", "positive", "d"]},
    }

    candidates, audit_rows = build_candidate_pool(
        retriever_rankings=rankings,
        queries=queries,
        top_k_per_retriever=3,
        pool_size=4,
        rrf_k=60,
    )
    metrics = _candidate_metrics(candidates, queries)

    assert candidates["q1"][0] == "positive"
    assert metrics["overall"]["hit_rate_at_100"] == 1.0
    assert audit_rows[0]["retriever_ranks"] == {
        "dense-a": 2,
        "dense-b": 1,
        "lexical": 2,
    }


def test_candidate_pool_does_not_read_silver_label_metadata() -> None:
    queries = [
        {
            "query_id": "q1",
            "category": "entity_query",
            "target_hop": 0,
            "positive_subgraph_ids": ["secret-positive"],
            "label_basis": {"center_id": "secret-positive"},
            "hard_negative_subgraph_ids": ["negative"],
        }
    ]
    rankings = {
        "dense": {"q1": ["negative", "ordinary"]},
        "lexical": {"q1": ["ordinary", "negative"]},
    }

    candidates, _ = build_candidate_pool(
        retriever_rankings=rankings,
        queries=queries,
        top_k_per_retriever=2,
        pool_size=2,
    )

    assert "secret-positive" not in candidates["q1"]


def test_reranker_benchmark_rejects_stale_subgraph_references(tmp_path: Path) -> None:
    for hop in (0, 1, 2):
        _write_jsonl(
            tmp_path / f"subgraphs_{hop}hop.jsonl",
            [{"subgraph_id": f"current-{hop}"}],
        )
    queries = [
        {
            "query_id": "q1",
            "target_hop": 1,
            "positive_subgraph_ids": ["stale-id"],
            "hard_negative_subgraph_ids": ["current-1"],
        }
    ]

    with pytest.raises(RuntimeError, match="reference mismatch"):
        validate_query_references(queries, tmp_path)


def test_graph_constraint_parser_and_validator_use_actual_path() -> None:
    query = "DDI victim linked to the pharmacokinetic outcome changed by the interaction"
    subgraph = {
        "center_id": "compound:midazolam",
        "edges": [
            {
                "subject_id": "compound:midazolam",
                "predicate": "VICTIM_IN",
                "object_id": "event:ddi",
            },
            {
                "subject_id": "event:ddi",
                "predicate": "AFFECTS_PARAMETER",
                "object_id": "parameter:auc",
            },
        ],
    }

    assert infer_query_graph_constraints(query) == {
        "required_path": ["VICTIM_IN", "AFFECTS_PARAMETER"]
    }
    assert graph_constraint_passes(query, subgraph)[0] is True
    subgraph["edges"][1]["predicate"] = "MEDIATED_BY"
    assert graph_constraint_passes(query, subgraph)[0] is False


def test_graph_constraint_parser_assigns_affected_drug_to_victim_role() -> None:
    assert infer_query_graph_constraints(
        "drug whose exposure is altered by a coadministered perpetrator"
    ) == {"required_center_predicate": "VICTIM_IN"}
    assert infer_query_graph_constraints(
        "affected compound with a two edge path to an exposure endpoint"
    ) == {"required_path": ["VICTIM_IN", "AFFECTS_PARAMETER"]}
    assert infer_query_graph_constraints(
        "causal interacting compound with a two edge path to metabolic machinery"
    ) == {"required_path": ["PERPETRATOR_IN", "MEDIATED_BY"]}
