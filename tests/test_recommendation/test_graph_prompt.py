from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.recommendation.graph_prompt import (
    GraphPromptCorpus,
    _canonicalize_parameter_numeric_fields,
    _species_context_assessment,
    _single_percent_is_nearest_parameter_value,
    assemble_recommendation,
    derive_silver_target,
    parse_selection,
    serialize_text_graph,
    validate_recommendation,
    validate_selection,
)
from pbpk_kb.recommendation.schemas import ModelingRequest
from pbpk_kb.recommendation.graph_prompt_dataset import _allowed_ids
from pbpk_kb.recommendation.online_agent import (
    ONLINE_PATH_ENV,
    _prompt_row,
    _validate_runtime_artifact_chain,
    configured_online_agent_kwargs,
    online_thread_settings,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_parameter_gate_converts_reported_range_to_canonical_unit() -> None:
    values, reasons = _canonicalize_parameter_numeric_fields(
        {
            "value_low": 6.0,
            "value_high": 10.0,
            "value_text": "6.0-10.0",
            "reported_unit": "mg/L",
            "canonical_unit": "ng/mL",
        },
        "Cmax ranged from 6 to 10 mg/L.",
    )

    assert reasons == []
    assert values["reported_value_low"] == 6.0
    assert values["reported_value_high"] == 10.0
    assert values["value_low"] == 6000.0
    assert values["value_high"] == 10000.0
    assert values["value_text"] == "6000-10000"
    assert values["canonical_unit"] == "ng/mL"


def test_parameter_gate_recognizes_already_canonical_stored_value() -> None:
    values, reasons = _canonicalize_parameter_numeric_fields(
        {
            "value": 2010.0,
            "value_text": "2010.0",
            "reported_unit": "ug/mL",
            "canonical_unit": "ng/mL",
        },
        "The reported Cmax was 2.01 ug/mL.",
    )

    assert reasons == []
    assert values["reported_value"] == pytest.approx(2.01)
    assert values["value"] == 2010.0
    assert values["unit_conversion_modes"]["value"] == (
        "already_canonical_value"
    )


def test_parameter_gate_allows_identical_domain_specific_unit() -> None:
    values, reasons = _canonicalize_parameter_numeric_fields(
        {
            "value": -1.03,
            "value_text": "-1.03",
            "reported_unit": "logP",
            "canonical_unit": "logP",
        },
        "The reported alogp was -1.03.",
    )

    assert reasons == []
    assert values["value"] == -1.03
    assert values["reported_value"] == -1.03
    assert values["unit_conversion_modes"]["value"] == "identity_conversion"


def test_human_parameter_gate_rejects_unresolved_literature_species() -> None:
    basis, reasons = _species_context_assessment(
        canonical_parameter="fraction_metabolized_via_cyp3a",
        request=ModelingRequest(
            compound="digoxin",
            objective="base_pk",
            species="human",
        ),
        source_id="pmc_oa",
        context={},
        supporting_text="CYP3A metabolism comprised 60% of the dose.",
    )

    assert basis == "species_context_absent"
    assert reasons == ["human_species_context_not_explicit"]


def test_human_parameter_gate_accepts_regulatory_label_context() -> None:
    basis, reasons = _species_context_assessment(
        canonical_parameter="clearance",
        request=ModelingRequest(
            compound="metformin",
            objective="base_pk",
            species="human",
        ),
        source_id="dailymed",
        context={},
        supporting_text="Clearance was 600 mL/min.",
    )

    assert basis == "human_regulatory_or_trial_source"
    assert reasons == []


def test_species_independent_property_does_not_require_population() -> None:
    basis, reasons = _species_context_assessment(
        canonical_parameter="molecular_weight",
        request=ModelingRequest(
            compound="warfarin",
            objective="base_pk",
            species="human",
        ),
        source_id="chembl",
        context={},
        supporting_text="308.33",
    )

    assert basis == "species_independent_property"
    assert reasons == []


def _corpus(tmp_path: Path) -> GraphPromptCorpus:
    graph = tmp_path / "graph"
    subgraphs = tmp_path / "subgraphs"
    gnn = tmp_path / "gnn"
    compound = {
        "node_id": "compound:caffeine",
        "node_type": "Compound",
        "preferred_name": "caffeine",
        "properties": {"aliases": ["Caffeine"]},
        "standard_identifiers": {"PUBCHEM.COMPOUND": "2519"},
    }
    evidence = {
        "node_id": "evidence:1",
        "node_type": "Evidence",
        "preferred_name": "clinical_pk:clearance",
        "properties": {
            "quality_status": "auto_validated",
            "extraction_confidence": 0.99,
        },
        "standard_identifiers": {},
    }
    event = {
        "node_id": "event:1",
        "node_type": "PKMeasurement",
        "preferred_name": "clearance measurement",
        "properties": {},
        "standard_identifiers": {},
    }
    parameter = {
        "node_id": "parameter:clearance",
        "node_type": "PKParameter",
        "preferred_name": "clearance",
        "properties": {},
        "standard_identifiers": {},
    }
    value = {
        "node_id": "value:1",
        "node_type": "ParameterValue",
        "preferred_name": "9.5",
        "properties": {
            "value": 9.5,
            "value_low": None,
            "value_high": None,
            "value_text": "9.5",
            "canonical_unit": "L/h",
        },
        "standard_identifiers": {},
    }
    source = {
        "node_id": "source:1",
        "node_type": "SourceDocument",
        "preferred_name": "source",
        "properties": {},
        "standard_identifiers": {},
    }
    model = {
        "node_id": "model:1",
        "node_type": "PBPKModel",
        "preferred_name": "Caffeine Model",
        "properties": {
            "reusable_candidate": True,
            "availability_status": "repository_model",
            "model_checksum": "abc",
            "software": "PK-Sim",
        },
        "standard_identifiers": {"MODEL_CHECKSUM": "abc"},
    }
    provenance = {
        "evidence_id": "evidence:1",
        "source_id": "pubmed",
        "source_location": "pubmed:1#abstract",
        "supporting_text": "Caffeine clearance was 9.5 L/h in healthy adults.",
        "source_text_hash": "hash1",
        "quality_status": "auto_validated",
        "negation_status": "affirmed",
        "confidence": 0.99,
    }

    def edge(edge_id: str, subject: dict, predicate: str, object_: dict) -> dict:
        return {
            "edge_id": edge_id,
            "subject_id": subject["node_id"],
            "subject_type": subject["node_type"],
            "predicate": predicate,
            "object_id": object_["node_id"],
            "object_type": object_["node_type"],
            "properties": {},
            "provenance": provenance,
        }

    edges = [
        edge("edge:report", evidence, "REPORTS", event),
        edge("edge:compound", event, "OF_COMPOUND", compound),
        edge("edge:measure", event, "MEASURES", parameter),
        edge("edge:value", event, "HAS_VALUE", value),
        edge("edge:support", event, "SUPPORTED_BY", evidence),
        edge("edge:source", evidence, "DERIVED_FROM", source),
        edge("edge:model", compound, "DESCRIBED_BY_MODEL", model),
    ]
    event_row = {
        "event_id": "event:1",
        "event_type": "PKMeasurement",
        "argument_edge_ids": [
            "edge:compound",
            "edge:measure",
            "edge:value",
            "edge:support",
        ],
        "evidence_id": "evidence:1",
        "context": {"population": "healthy adults", "species": "human"},
        "trigger_text": "clearance",
    }
    _write_jsonl(
        graph / "kg_nodes.jsonl",
        [compound, evidence, event, parameter, value, source, model],
    )
    _write_jsonl(graph / "kg_edges.jsonl", edges)
    _write_jsonl(graph / "kg_events.jsonl", [event_row])
    (graph / "kg_extraction_manifest.json").write_text("{}", encoding="utf-8")
    citation = {
        **provenance,
        "quality_status": "auto_validated",
    }
    for hop in range(3):
        row = {
            "subgraph_id": f"subgraph:{hop}",
            "center_id": "compound:caffeine",
            "center_type": "Compound",
            "center_name": "caffeine",
            "hop_limit": hop,
            "nodes": [compound, evidence, event, parameter, value, source, model],
            "edges": edges if hop else [],
            "events": [event_row] if hop else [],
            "citations": [citation] if hop else [],
            "path_ids": ["path:1"] if hop else [],
            "embedding_text": (
                "CENTER caffeine\nEDGE caffeine clearance 9.5 L/h\n"
                "CITATION pubmed healthy adults"
            ),
        }
        _write_jsonl(subgraphs / f"subgraphs_{hop}hop.jsonl", [row])
    (subgraphs / "gate2_manifest.json").write_text("{}", encoding="utf-8")
    _write_jsonl(
        gnn / "subgraph_records.jsonl",
        [
            {
                "subgraph_id": f"subgraph:{hop}",
                "embedding_row": hop,
            }
            for hop in range(3)
        ],
    )
    np.save(gnn / "subgraph_vectors.npy", np.eye(3, 256, dtype=np.float32))
    (gnn / "gnn_index_manifest.json").write_text("{}", encoding="utf-8")
    return GraphPromptCorpus(
        graph_dir=graph,
        subgraph_dir=subgraphs,
        gnn_index_dir=gnn,
    )


def _inject_foreign_ddi(corpus: GraphPromptCorpus) -> None:
    foreign_compound = {
        "node_id": "compound:digoxin",
        "node_type": "Compound",
        "preferred_name": "digoxin",
        "properties": {"aliases": ["Digoxin"]},
        "standard_identifiers": {},
    }
    foreign_evidence = {
        "node_id": "evidence:foreign-ddi",
        "node_type": "Evidence",
        "preferred_name": "ddi:digoxin interaction",
        "properties": {"quality_status": "auto_validated"},
        "standard_identifiers": {},
    }
    foreign_event_node = {
        "node_id": "event:foreign-ddi",
        "node_type": "DDIEvent",
        "preferred_name": "digoxin DDI",
        "properties": {},
        "standard_identifiers": {},
    }
    provenance = {
        "evidence_id": "evidence:foreign-ddi",
        "source_id": "pubmed",
        "source_location": "pubmed:foreign#abstract",
        "supporting_text": "Cyclosporine reduced the clearance of digoxin.",
        "source_text_hash": "hash-foreign",
        "quality_status": "auto_validated",
        "negation_status": "affirmed",
        "confidence": 0.98,
    }
    victim_edge = {
        "edge_id": "edge:foreign-victim",
        "subject_id": "compound:digoxin",
        "subject_type": "Compound",
        "predicate": "VICTIM_IN",
        "object_id": "event:foreign-ddi",
        "object_type": "DDIEvent",
        "properties": {},
        "provenance": provenance,
    }
    foreign_event = {
        "event_id": "event:foreign-ddi",
        "event_type": "DDIEvent",
        "argument_edge_ids": ["edge:foreign-victim"],
        "evidence_id": "evidence:foreign-ddi",
        "context": {"condition": "with cyclosporine"},
        "trigger_text": "Cyclosporine reduced the clearance of digoxin.",
    }
    for node in (foreign_compound, foreign_evidence, foreign_event_node):
        corpus.nodes.append(node)
        corpus.node_by_id[node["node_id"]] = node
    corpus.compounds[foreign_compound["node_id"]] = foreign_compound
    corpus.events.append(foreign_event)
    corpus.event_by_id[foreign_event["event_id"]] = foreign_event
    corpus.edges.append(victim_edge)
    corpus.edge_by_id[victim_edge["edge_id"]] = victim_edge
    corpus.edges_by_evidence[foreign_evidence["node_id"]].append(victim_edge)
    citation = {**provenance, "quality_status": "auto_validated"}
    for subgraph in corpus.subgraphs.values():
        if int(subgraph.get("hop_limit") or 0) == 0:
            continue
        subgraph["nodes"].extend(
            [foreign_compound, foreign_evidence, foreign_event_node]
        )
        subgraph["edges"].append(victim_edge)
        subgraph["events"].append(foreign_event)
        subgraph["citations"].append(citation)


def test_graph_prompt_retrieval_and_grounded_assembly(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    request = ModelingRequest(
        compound="caffeine",
        objective="public_model_reuse",
    )
    retrieved = corpus.retrieve(request)
    target = derive_silver_target(corpus, request, retrieved)
    prompt = serialize_text_graph(corpus, request, retrieved)

    assert retrieved
    assert "evidence:1" in target["evidence_ids"]
    assert target["public_model_ids"] == ["model:1"]
    assert "CITATION" in prompt

    recommendation = assemble_recommendation(
        corpus=corpus,
        request=request,
        subgraphs=retrieved,
        selection=target,
    )
    validation = validate_recommendation(
        corpus=corpus,
        request=request,
        recommendation=recommendation,
        allowed_subgraph_ids={row.subgraph_id for row in retrieved},
    )

    assert validation["status"] == "recommendation_validated"
    assert validation["citation_provenance_completeness"] == 1.0
    assert recommendation["parameter_candidates"][0]["value"] == 9.5
    assert recommendation["citations"][0]["target_compound"]["node_id"] == (
        "compound:caffeine"
    )
    assert recommendation["citations"][0]["compound_attribution"][
        "status"
    ] == "validated_graph_binding"
    assert recommendation["citations"][0]["structured_measurements"][0][
        "study_context"
    ] == {"population": "healthy adults", "species": "human"}
    assert recommendation["selected_public_models"][0]["model_checksum"] == "abc"


def test_allowed_gap_codes_are_constrained_by_objective(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    ddi_request = ModelingRequest(
        compound="caffeine",
        objective="ddi_victim",
        perpetrator_drug="ketoconazole",
    )
    model_request = ModelingRequest(
        compound="caffeine",
        objective="public_model_reuse",
    )

    ddi_allowed = _allowed_ids(
        corpus,
        ddi_request,
        corpus.retrieve(ddi_request),
    )
    model_allowed = _allowed_ids(
        corpus,
        model_request,
        corpus.retrieve(model_request),
    )

    assert ddi_allowed["gap_codes"] == [
        "missing_component:ddi",
        "missing_component:metabolism",
    ]
    assert model_allowed["gap_codes"] == [
        "missing_component:public_model_template",
        "no_reusable_public_model",
    ]


def test_online_prompt_is_built_without_silver_target(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    request = ModelingRequest(
        compound="caffeine",
        objective="public_model_reuse",
    )
    retrieved = corpus.retrieve(request)
    retrieval = {
        "selected_subgraph_ids": [row.subgraph_id for row in retrieved],
        "ranked_candidates": [
            {
                "subgraph_id": row.subgraph_id,
                "final_score": row.score,
                "rrf_score": 0.1,
                "reranker_score": 0.9,
                "teacher_score": 0.8,
                "raw_hgt_score": 0.7,
                "residual_alpha": 0.5,
                "graph_constraint_pass": True,
            }
            for row in retrieved
        ],
    }

    row, subgraphs = _prompt_row(
        corpus=corpus,
        request=request,
        retrieval=retrieval,
    )

    assert subgraphs
    assert "target" not in row
    assert "positive_subgraph_ids" not in row
    assert row["allowed_output_ids"]["gap_codes"] == [
        "missing_component:public_model_template",
        "no_reusable_public_model",
    ]


def test_online_agent_paths_are_server_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, environment in ONLINE_PATH_ENV.items():
        path = tmp_path / key
        if key != "output_root":
            path.mkdir()
        monkeypatch.setenv(environment, str(path))
    monkeypatch.setenv("PBPK_RECOMMENDATION_DEVICE", "cuda:0")

    values = configured_online_agent_kwargs()

    assert values["model_path"] == tmp_path / "model_path"
    assert values["output_root"] == tmp_path / "output_root"
    assert values["device"] == "cuda:0"
    assert values["blas_threads"] == 1
    assert values["torch_threads"] == 4
    assert values["torch_interop_threads"] == 1


def test_online_agent_thread_settings_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PBPK_RECOMMENDATION_BLAS_THREADS", "2")
    monkeypatch.setenv("PBPK_RECOMMENDATION_TORCH_THREADS", "6")
    monkeypatch.setenv("PBPK_RECOMMENDATION_TORCH_INTEROP_THREADS", "1")

    assert online_thread_settings() == {
        "blas_threads": 2,
        "torch_threads": 6,
        "torch_interop_threads": 1,
    }

    monkeypatch.setenv("PBPK_RECOMMENDATION_BLAS_THREADS", "0")
    with pytest.raises(RuntimeError, match="must be a positive integer"):
        online_thread_settings()


def test_online_agent_rejects_runtime_graph_manifest_drift(
    tmp_path: Path,
) -> None:
    graph = tmp_path / "graph"
    subgraphs = tmp_path / "subgraphs"
    embeddings = tmp_path / "embeddings"
    benchmark = tmp_path / "benchmark"
    training = tmp_path / "training"
    gnn = tmp_path / "gnn"
    for path in (graph, subgraphs, embeddings, benchmark, training, gnn):
        path.mkdir()
    manifest_paths = {
        graph / "kg_extraction_manifest.json": b"graph",
        subgraphs / "gate2_manifest.json": b"subgraph",
        embeddings / "embedding_index_manifest.json": b"{}",
        benchmark / "gnn_benchmark_manifest.json": b"{}",
        training / "gnn_comparison.json": b"{}",
        gnn / "gnn_index_manifest.json": b"{}",
    }
    for path, content in manifest_paths.items():
        path.write_bytes(content)

    with pytest.raises(
        RuntimeError,
        match="Runtime graph_manifest does not match accepted Gate 5",
    ):
        _validate_runtime_artifact_chain(
            gate5={
                "input_hashes": {
                    "graph_manifest": "0" * 64,
                    "subgraph_manifest": "0" * 64,
                    "gnn_index_manifest": "0" * 64,
                }
            },
            graph_dir=graph,
            subgraph_dir=subgraphs,
            embedding_index_dir=embeddings,
            gnn_benchmark_dir=benchmark,
            gnn_training_dir=training,
            gnn_index_dir=gnn,
        )


def test_validator_rejects_tampered_canonical_numeric_value(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    request = ModelingRequest(compound="caffeine", objective="base_pk")
    retrieved = corpus.retrieve(request)
    selection = derive_silver_target(corpus, request, retrieved)
    recommendation = assemble_recommendation(
        corpus=corpus,
        request=request,
        subgraphs=retrieved,
        selection=selection,
    )
    recommendation["parameter_candidates"][0]["value"] = 999.0

    validation = validate_recommendation(
        corpus=corpus,
        request=request,
        recommendation=recommendation,
        allowed_subgraph_ids={row.subgraph_id for row in retrieved},
    )

    assert validation["status"] != "recommendation_validated"
    assert "parameter_candidates_not_exactly_graph_derived" in validation[
        "errors"
    ]


def test_parameter_gate_rejects_unit_dimension_and_relative_change(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    value = corpus.node_by_id["value:1"]["properties"]
    value["canonical_unit"] = "%"
    for edge in corpus.edges:
        edge["provenance"]["supporting_text"] = (
            "Caffeine clearance decreased by 9.5% after coadministration."
        )
    request = ModelingRequest(compound="caffeine", objective="base_pk")
    retrieved = corpus.retrieve(request)
    recommendation = assemble_recommendation(
        corpus=corpus,
        request=request,
        subgraphs=retrieved,
        selection=derive_silver_target(corpus, request, retrieved),
    )

    assert recommendation["parameter_candidates"] == []
    rejection = recommendation["parameter_candidate_rejections"][0]
    assert "parameter_unit_dimension_mismatch" in rejection["reasons"]
    assert (
        "relative_or_comparative_result_not_absolute_parameter"
        in rejection["reasons"]
    )


def test_parameter_gate_rejects_wrong_number_from_multi_quantity_sentence(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    corpus.node_by_id["parameter:clearance"]["preferred_name"] = "protein binding"
    corpus.node_by_id["parameter:clearance"]["properties"] = {
        "aliases": ["protein binding"]
    }
    corpus.node_by_id["evidence:1"]["preferred_name"] = "adme:excretion"
    corpus.node_by_id["value:1"]["properties"].update(
        {"value": 0.5, "value_text": "0.5", "canonical_unit": "%"}
    )
    corpus.event_by_id["event:1"]["event_type"] = "ExcretionEvent"
    source_text = (
        "Caffeine exhibits 94% plasma protein binding, and only 0.5% is "
        "excreted unchanged."
    )
    for edge in corpus.edges:
        edge["provenance"]["supporting_text"] = source_text
    request = ModelingRequest(compound="caffeine", objective="base_pk")
    retrieved = corpus.retrieve(request)
    recommendation = assemble_recommendation(
        corpus=corpus,
        request=request,
        subgraphs=retrieved,
        selection=derive_silver_target(corpus, request, retrieved),
    )

    assert recommendation["parameter_candidates"] == []
    reasons = recommendation["parameter_candidate_rejections"][0]["reasons"]
    assert "evidence_parameter_semantics_mismatch" in reasons
    assert "event_parameter_semantics_mismatch" in reasons
    assert "numeric_value_not_nearest_parameter_mention" in reasons


def test_percent_binding_can_match_a_specific_later_parameter_phrase() -> None:
    text = (
        "Excretion: 37-52% of the dose was recovered in urine. Less than 2% "
        "was excreted unchanged as parent drug."
    )
    assert _single_percent_is_nearest_parameter_value(
        2.0,
        text,
        ("excreted unchanged", "excretion"),
    )


def test_metabolism_fraction_requires_a_resolved_pathway(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    corpus.node_by_id["parameter:clearance"]["preferred_name"] = "metabolism"
    corpus.node_by_id["parameter:clearance"]["properties"] = {
        "aliases": ["metabolism"]
    }
    corpus.node_by_id["evidence:1"]["preferred_name"] = "adme:metabolism"
    corpus.node_by_id["value:1"]["properties"].update(
        {"value": 50.0, "value_text": "50", "canonical_unit": "%"}
    )
    request = ModelingRequest(compound="caffeine", objective="base_pk")
    retrieved = corpus.retrieve(request)

    for edge in corpus.edges:
        edge["provenance"]["supporting_text"] = (
            "CYP2C8 and CYP3A4 each contributed less than 50% to caffeine "
            "metabolism."
        )
    ambiguous = assemble_recommendation(
        corpus=corpus,
        request=request,
        subgraphs=retrieved,
        selection=derive_silver_target(corpus, request, retrieved),
    )
    assert ambiguous["parameter_candidates"] == []
    assert "multiple_metabolic_pathways_not_resolved" in ambiguous[
        "parameter_candidate_rejections"
    ][0]["reasons"]

    for edge in corpus.edges:
        edge["provenance"]["supporting_text"] = (
            "Metabolism by CYP3A4 comprised 50% of the caffeine dose."
        )
    resolved = assemble_recommendation(
        corpus=corpus,
        request=request,
        subgraphs=retrieved,
        selection=derive_silver_target(corpus, request, retrieved),
    )
    assert resolved["parameter_candidates"][0]["parameter"] == (
        "fraction_metabolized_via_cyp3a4"
    )

def test_validator_rejects_tampered_parameter_gate_audit(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    request = ModelingRequest(compound="caffeine", objective="base_pk")
    retrieved = corpus.retrieve(request)
    recommendation = assemble_recommendation(
        corpus=corpus,
        request=request,
        subgraphs=retrieved,
        selection=derive_silver_target(corpus, request, retrieved),
    )
    recommendation["parameter_candidate_rejections"] = [
        {"event_id": "invented"}
    ]

    validation = validate_recommendation(
        corpus=corpus,
        request=request,
        recommendation=recommendation,
        allowed_subgraph_ids={row.subgraph_id for row in retrieved},
    )

    assert validation["status"] == "recommendation_quarantined"
    assert "parameter_candidate_rejections_not_exactly_graph_derived" in validation[
        "errors"
    ]


def test_assembly_and_validator_block_cross_compound_subgraph_leakage(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    _inject_foreign_ddi(corpus)
    request = ModelingRequest(
        compound="caffeine",
        objective="public_model_reuse",
    )
    retrieved = corpus.retrieve(request)
    selection = derive_silver_target(corpus, request, retrieved)

    recommendation = assemble_recommendation(
        corpus=corpus,
        request=request,
        subgraphs=retrieved,
        selection=selection,
    )

    assert recommendation["ddi_mechanisms"] == []
    assert all(
        row["evidence_id"] != "evidence:foreign-ddi"
        for row in recommendation["citations"]
    )

    recommendation["citations"].append(
        corpus.evidence_record("evidence:foreign-ddi")
    )
    validation = validate_recommendation(
        corpus=corpus,
        request=request,
        recommendation=recommendation,
        allowed_subgraph_ids={row.subgraph_id for row in retrieved},
    )

    assert validation["status"] == "recommendation_quarantined"
    assert any(
        row.startswith(
            "citation_not_grounded_to_target_compound_and_objective:"
        )
        for row in validation["errors"]
    )


def test_validator_rejects_tampered_citation_compound_context(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    request = ModelingRequest(compound="caffeine", objective="base_pk")
    retrieved = corpus.retrieve(request)
    recommendation = assemble_recommendation(
        corpus=corpus,
        request=request,
        subgraphs=retrieved,
        selection=derive_silver_target(corpus, request, retrieved),
    )
    recommendation["citations"][0]["target_compound"]["node_id"] = (
        "compound:wrong"
    )

    validation = validate_recommendation(
        corpus=corpus,
        request=request,
        recommendation=recommendation,
        allowed_subgraph_ids={row.subgraph_id for row in retrieved},
    )

    assert validation["status"] != "recommendation_validated"
    assert any(
        row.startswith("citation_target_compound_mismatch:")
        for row in validation["errors"]
    )


def test_selection_parser_rejects_unknown_handles() -> None:
    selection = parse_selection(
        '{"component_ids":["C01"],"evidence_ids":["E01"],'
        '"public_model_ids":[],"gap_codes":[]}'
    )
    errors = validate_selection(
        selection,
        {
            "component_ids": ["C01"],
            "evidence_ids": [],
            "public_model_ids": [],
            "gap_codes": [],
        },
    )

    assert errors == ["unsupported_evidence_ids:E01"]


def test_parse_selection_requires_all_list_fields() -> None:
    with pytest.raises(ValueError):
        parse_selection('{"component_ids":[]}')
