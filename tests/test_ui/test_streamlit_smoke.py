from __future__ import annotations

from pathlib import Path

import pytest

from pbpk_kb.ui import api_client, data


streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).resolve().parents[2]


def sample_compound() -> dict:
    return {
        "compound_id": "compound-1",
        "preferred_name": "Midazolam",
        "normalized_name": "midazolam",
        "synonyms": ["Midazolam Hydrochloride"],
        "pubchem_cid": "4192",
        "chembl_id": None,
        "drugcentral_id": None,
        "rxnorm_rxcui": None,
        "inchi_key": "DDLIGBOFAVUZHB-UHFFFAOYSA-N",
        "approval_status": None,
        "source_id": "pubchem",
        "source_location": "CID:4192",
        "confidence_score": 0.99,
        "curation_status": "machine_extracted",
        "evidence_count": 27,
        "clinical_pk_count": 14,
        "label_count": 1,
        "public_model_count": 0,
    }


def sample_source() -> dict:
    return {
        "source_id": "pubchem",
        "source_name": "PubChem",
        "source_family": "compound",
        "access_method": "REST API",
        "expected_format": ["JSON"],
        "license_notes": "public",
        "connector": "pubchem",
        "parser": "pubchem_json_parser",
        "entities": ["compound"],
        "refresh_frequency": "on demand",
        "enabled": True,
        "source_version": None,
        "checksum": None,
        "access_url": "https://pubchem.ncbi.nlm.nih.gov",
        "api_url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
        "raw_document_count": 1,
        "parsed_document_count": 1,
        "chunk_count": 1,
        "evidence_count": 3,
        "model_count": 0,
        "latest_ingestion_status": "succeeded",
        "latest_ingestion_completed_at": "2026-06-13T00:00:00",
        "latest_ingestion_log_path": "ingestion_logs/pubchem/latest.json",
        "updated_at": "2026-06-13T00:00:00",
    }


def sample_evidence() -> dict:
    return {
        "evidence_type": "clinical_pk",
        "evidence_id": "evidence-1",
        "compound_id": "compound-1",
        "compound_name": "Midazolam",
        "parameter_name": "total clearance",
        "numeric_value": None,
        "value_text": "9.3 to 11.0",
        "unit": "mL/min/kg",
        "context": "DailyMed rule-based numeric candidate",
        "source_id": "dailymed",
        "source_location": "SPL_SET_ID:example::SECTION:Elimination",
        "evidence_text": "The mean total clearance ranged from 9.3 to 11.0 mL/min/kg.",
        "extraction_method": "rule_based",
        "confidence_score": 0.72,
        "curation_status": "needs_review",
        "duplicate_risk": "low",
        "missing_unit": False,
        "missing_context": False,
        "created_at": "2026-06-13T00:00:00",
    }


def sample_model() -> dict:
    return {
        "model_id": "model-1",
        "model_name": "7E3",
        "model_type": "public_repository_folder",
        "software": "Open Systems Pharmacology",
        "species": None,
        "population": None,
        "route": None,
        "compound_name": None,
        "model_source": "https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library/tree/master/7E3",
        "source_id": "osp_pbpk_model_library",
        "source_name": "OSP-PBPK-Model-Library",
        "model_file_path": "7E3",
        "parameter_file_path": None,
        "evaluation_report_path": None,
        "publication_id": None,
        "validation_status": "metadata_indexed",
        "curation_status": "needs_review",
        "source_location": "GITHUB_PATH:7E3",
        "notes": "Compound linkage not asserted until curator review.",
        "created_at": "2026-06-13T00:00:00",
    }


def sample_detail() -> dict:
    evidence = sample_evidence()
    return {
        "summary": evidence,
        "extracted_fields": {
            "pk_parameter": "total clearance",
            "value_text": "9.3 to 11.0",
            "unit": "mL/min/kg",
        },
        "normalized_fields": {
            "preferred_name": "Midazolam",
            "pubchem_cid": "4192",
            "inchi_key": "DDLIGBOFAVUZHB-UHFFFAOYSA-N",
        },
        "source_document": {
            "source_id": "dailymed",
            "title": "Midazolam label",
            "source_location": "SPL_SET_ID:example",
            "raw_storage_path": "data/raw/dailymed/example.json",
        },
        "related_evidence": [sample_evidence()],
        "curation_history": [
            {
                "curation_timestamp": "2026-06-13T00:00:00",
                "previous_status": "machine_extracted",
                "new_status": "needs_review",
                "curator": "stage1_curator",
                "curator_comment": "ready",
                "field_changes": {"action": "flag_for_review"},
            }
        ],
        "editable_fields": ["value_text", "unit", "confidence_score"],
        "allowed_actions": ["accept", "reject", "edit", "note"],
    }


def sample_qa_result() -> dict:
    evidence = {**sample_evidence(), "citation_id": "E1"}
    graph_relationship = {
        "citation_id": "G1",
        "relationship_id": "rel-1",
        "compound_name": "Midazolam",
        "relationship_type": "inhibits_or_binds",
        "target_name": "Cytochrome P450 3A4",
        "gene_symbol": "CYP3A4",
        "target_type": "enzyme",
        "source_id": "chembl",
        "source_name": "ChEMBL",
        "source_location": "CHEMBL_MOLECULE:CHEMBL655::CHEMBL_ACTIVITY:1",
        "extraction_method": "machine_extracted",
        "curation_status": "needs_review",
    }
    return {
        "question": "What public clinical PK evidence exists for midazolam?",
        "answer": "The KB found 1 structured evidence record. [E1]",
        "compound": sample_compound(),
        "key_evidence": [evidence],
        "model_records": [{**sample_model(), "citation_id": "M1"}],
        "graph_relationships": [graph_relationship],
        "document_chunks": [
            {
                "citation_id": "D1",
                "chunk_id": "chunk-1",
                "source_id": "dailymed",
                "source_name": "DailyMed",
                "document_id": "document-1",
                "title": "Midazolam label",
                "section_title": "Clinical Pharmacology",
                "source_location": "SPL_SET_ID:example::SECTION:Clinical Pharmacology",
                "chunk_text": "Midazolam pharmacokinetic information.",
                "embedding_model": "NeuML/pubmedbert-base-embeddings",
                "score": 0.75,
            }
        ],
        "citations": [
            {
                "citation_id": "E1",
                "kind": "structured_evidence",
                "source_id": "dailymed",
                "source_location": "SPL_SET_ID:example",
                "curation_status": "needs_review",
            }
        ],
        "curation_status_counts": {"needs_review": 1},
        "uncertainties_and_gaps": ["Record needs review."],
        "retrieval_summary": {
            "structured_evidence_count": 1,
            "model_record_count": 1,
            "graph_relationship_count": 1,
            "document_chunk_count": 1,
        },
    }


def sample_graph() -> dict:
    compound = sample_compound()
    evidence = sample_evidence()
    return {
        "compound": compound,
        "nodes": [
            {
                "id": f"compound:{compound['compound_id']}",
                "type": "compound",
                "label": compound["preferred_name"],
                "properties": compound,
            },
            {
                "id": f"evidence:{evidence['evidence_type']}:{evidence['evidence_id']}",
                "type": "evidence",
                "label": evidence["parameter_name"],
                "properties": evidence,
            },
            {
                "id": "source:dailymed",
                "type": "source",
                "label": "DailyMed",
                "properties": {"source_id": "dailymed"},
            },
            {
                "id": "protein_target:target-1",
                "type": "protein_target",
                "label": "Cytochrome P450 3A4",
                "properties": {"gene_symbol": "CYP3A4"},
            },
        ],
        "edges": [
            {
                "source": f"compound:{compound['compound_id']}",
                "target": f"evidence:{evidence['evidence_type']}:{evidence['evidence_id']}",
                "type": "HAS_EVIDENCE",
            },
            {
                "source": f"evidence:{evidence['evidence_type']}:{evidence['evidence_id']}",
                "target": "source:dailymed",
                "type": "DERIVED_FROM",
            },
            {
                "source": f"compound:{compound['compound_id']}",
                "target": "protein_target:target-1",
                "type": "inhibits_or_binds",
            },
        ],
        "counts": {"nodes": 4, "edges": 3, "evidence": 1, "models": 0, "relationships": 1},
    }


def sample_export_result() -> dict:
    return {
        "export_job": {
            "export_job_id": "export-1",
            "export_type": "evidence_csv",
            "status": "succeeded",
            "output_path": "exports/evidence_csv/example.csv",
            "included_curation_statuses": ["needs_review"],
        },
        "output_path": "exports/evidence_csv/example.csv",
        "metadata_path": "exports/evidence_csv/example.metadata.json",
        "row_count": 1,
    }


def assert_no_streamlit_exceptions(app_test) -> None:
    assert len(app_test.exception) == 0


def test_home_dashboard_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        data,
        "dashboard_metrics",
        lambda: {
            "source_count": 5,
            "compound_count": 1,
            "evidence_count": 27,
            "auto_validated_count": 27,
            "raw_document_count": 8,
            "parsed_document_count": 8,
            "rag_record_count": 60,
            "public_model_count": 2,
        },
    )
    monkeypatch.setattr(data, "evidence_status_counts", lambda: [{"curation_status": "auto_validated", "record_count": 27}])
    monkeypatch.setattr(data, "source_registry_rows", lambda family="", enabled="all": [sample_source()])
    monkeypatch.setattr(data, "latest_ingestion_runs", lambda limit=10: [])
    monkeypatch.setattr(data, "latest_export_jobs", lambda limit=10: [])

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/streamlit_app.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "PBPK Stage 1 Public Knowledge Base" for title in app.title)
    assert len(app.dataframe) >= 1


def test_recommendation_agent_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_client,
        "recommendation_graph_summary",
        lambda: {
            "validated_recommendation_count": 1,
            "compound_count": 1,
            "knowledge_base_compound_count": 1,
            "benchmark_level": "silver",
            "online_inference_enabled": False,
            "serving_mode": "validated_results_only",
        },
    )
    validated_row = {
        "recommendation_result_id": "result-1",
        "compound_name": "dextromethorphan",
        "objective": "base_pk",
        "request_json": {"compound": "dextromethorphan"},
    }
    monkeypatch.setattr(
        api_client,
        "recommendation_compounds",
        lambda: [
            {
                "compound_id": "compound-dextromethorphan",
                "compound_name": "dextromethorphan",
                "pubchem_cid": "5360696",
                "validated_scenario_count": 1,
                "validated_objectives": ["base_pk"],
                "auto_validated_evidence_count": 31,
            }
        ],
    )
    monkeypatch.setattr(
        api_client,
        "validated_recommendations",
        lambda compound="", objective="", limit=100: [validated_row],
    )
    monkeypatch.setattr(
        api_client,
        "validated_recommendation_detail",
        lambda value: {
            **validated_row,
            "status": "recommendation_validated",
            "recommendation_sha256": "a" * 64,
            "request_sha256": "b" * 64,
            "model_run": {
                "benchmark_level": "silver",
                "generalization_claim_eligible": False,
                "gold_claim": False,
            },
            "validation_json": {
                "status": "recommendation_validated",
                "citation_provenance_completeness": 1.0,
                "unsupported_numeric_parameter_rate": 0.0,
            },
            "recommendation_json": {
                "status": "recommendation_validated",
                "overall_confidence": 1.0,
                "generated_from_auto_validated_only": True,
                "component_recommendations": [
                    {
                        "component": "absorption",
                        "status": "evidence_supported_candidate",
                        "evidence_ids": ["evidence-1"],
                        "source_count": 1,
                    }
                ],
                "recommended_model_structure": {
                    "whole_body_or_reduced_pbpk": "unspecified"
                },
                "parameter_candidates": [],
                "metabolism_pathways": [],
                "transport_pathways": [],
                "citations": [
                    {
                        "evidence_id": "evidence-1",
                        "evidence_type": "adme",
                        "parameter": "absorption",
                        "source_id": "dailymed",
                        "confidence": 1.0,
                        "supporting_text": "source-backed text",
                        "source_location": "dailymed:label#section",
                    }
                ],
                "retrieved_subgraphs": [],
                "traversed_path_ids": [],
                "evidence_gaps": [],
            },
        },
    )
    monkeypatch.setattr(
        api_client,
        "public_model_catalog_summary",
        lambda: {
            "model_count": 41,
            "model_version_count": 49,
            "runtime_verified_version_count": 49,
            "reusable_version_count": 1,
            "blueprint_generation_enabled": False,
            "compounds": [
                {
                    "compound_id": "compound-1",
                    "compound_name": "rosuvastatin",
                    "same_compound_model_version_count": 1,
                    "exact_family_model_version_count": 1,
                    "reusable_model_version_count": 1,
                }
            ],
        },
    )

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/14_recommendation_agent.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(
        title.value == "PBPK Recommendation Agent"
        for title in app.title
    )
    assert any(
        header.value == "PBPK Component Recommendation"
        for header in app.header
    )
    assert any(
        metric.label == "Validated scenarios" and metric.value == "1"
        for metric in app.metric
    )
    assert any(
        metric.label == "Evidence-supported candidates"
        and metric.value == "1"
        for metric in app.metric
    )
    assert any(
        metric.label == "KB compounds covered" and metric.value == "1 / 1"
        for metric in app.metric
    )
    assert any(
        metric.label == "Online inference" and metric.value == "Disabled"
        for metric in app.metric
    )
    assert not any(
        button.label == "Generate validated component recommendation"
        for button in app.button
    )
    component_compound = next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "Component recommendation compound"
    )
    assert component_compound.value == "dextromethorphan"
    assert any(
        button.label == "Rank downloaded public models"
        for button in app.button
    )
    target_compound = next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "Target compound"
    )
    assert target_compound.options == ["dextromethorphan"]
    assert target_compound.value == "dextromethorphan"


@pytest.fixture(autouse=True)
def model_matching_evaluation_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client,
        "public_model_matching_evaluation",
        lambda: {
            "status": "not_available",
            "scientific_claim_eligible": False,
        },
    )


def test_online_component_request_is_restricted_to_kb_compounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client,
        "recommendation_graph_summary",
        lambda: {
            "validated_recommendation_count": 2,
            "compound_count": 2,
            "knowledge_base_compound_count": 2,
            "benchmark_level": "silver",
            "online_inference_enabled": True,
            "serving_mode": "validated_results_and_online_inference",
            "supported_objectives": ["base_pk"],
        },
    )
    monkeypatch.setattr(
        api_client,
        "recommendation_compounds",
        lambda: [
            {
                "compound_id": "compound-caffeine",
                "compound_name": "caffeine",
                "validated_scenario_count": 1,
                "validated_objectives": ["base_pk"],
                "auto_validated_evidence_count": 10,
            },
            {
                "compound_id": "compound-midazolam",
                "compound_name": "midazolam",
                "validated_scenario_count": 1,
                "validated_objectives": ["base_pk"],
                "auto_validated_evidence_count": 10,
            },
        ],
    )

    app = AppTest.from_file(
        str(ROOT / "pbpk_kb/ui/pages/14_recommendation_agent.py")
    ).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    selector = next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "Component target compound"
    )
    assert selector.options == ["caffeine", "midazolam"]
    assert selector.value == "midazolam"


@pytest.fixture(autouse=True)
def recommendation_agent_api_defaults(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if "recommendation_agent" not in request.node.name:
        return
    monkeypatch.setattr(
        api_client,
        "recommendation_graph_summary",
        lambda: {
            "validated_recommendation_count": 0,
            "compound_count": 0,
            "knowledge_base_compound_count": 0,
            "benchmark_level": "silver",
            "online_inference_enabled": False,
            "serving_mode": "validated_results_only",
        },
    )
    monkeypatch.setattr(
        api_client,
        "recommendation_compounds",
        lambda: [],
    )
    monkeypatch.setattr(
        api_client,
        "validated_recommendations",
        lambda compound="", objective="", limit=100: [],
    )


def test_recommendation_agent_exposes_online_form_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client,
        "recommendation_graph_summary",
        lambda: {
            "validated_recommendation_count": 0,
            "compound_count": 0,
            "knowledge_base_compound_count": 0,
            "benchmark_level": "silver",
            "online_inference_enabled": True,
            "serving_mode": "validated_results_and_online_inference",
            "supported_objectives": ["base_pk", "ddi_victim"],
        },
    )
    monkeypatch.setattr(
        api_client,
        "public_model_catalog_summary",
        lambda: {
            "model_count": 0,
            "model_version_count": 0,
            "runtime_verified_version_count": 0,
            "reusable_version_count": 0,
            "blueprint_generation_enabled": False,
            "compounds": [],
        },
    )

    app = AppTest.from_file(
        str(ROOT / "pbpk_kb/ui/pages/14_recommendation_agent.py")
    ).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(
        button.label == "Generate validated component recommendation"
        for button in app.button
    )
    assert any(
        metric.label == "Online inference" and metric.value == "Enabled"
        for metric in app.metric
    )


def test_recommendation_agent_shows_uncovered_kb_compound_without_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client,
        "recommendation_graph_summary",
        lambda: {
            "validated_recommendation_count": 0,
            "compound_count": 0,
            "knowledge_base_compound_count": 1,
            "benchmark_level": "silver",
            "online_inference_enabled": False,
            "serving_mode": "validated_results_only",
        },
    )
    monkeypatch.setattr(
        api_client,
        "recommendation_compounds",
        lambda: [
            {
                "compound_id": "compound-caffeine",
                "compound_name": "caffeine",
                "pubchem_cid": "2519",
                "validated_scenario_count": 0,
                "validated_objectives": [],
                "auto_validated_evidence_count": 127,
            }
        ],
    )
    monkeypatch.setattr(
        api_client,
        "public_model_catalog_summary",
        lambda: {
            "model_count": 0,
            "model_version_count": 0,
            "runtime_verified_version_count": 0,
            "reusable_version_count": 0,
            "blueprint_generation_enabled": False,
            "compounds": [],
        },
    )

    app = AppTest.from_file(
        str(ROOT / "pbpk_kb/ui/pages/14_recommendation_agent.py")
    ).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    component_compound = next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "Component recommendation compound"
    )
    assert component_compound.value == "caffeine"
    assert any(
        "no component recommendation has completed" in warning.value
        for warning in app.warning
    )
    assert not any(
        selectbox.label == "Validated modeling scenario"
        for selectbox in app.selectbox
    )


def test_recommendation_agent_renders_layered_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client,
        "public_model_catalog_summary",
        lambda: {
            "model_count": 41,
            "model_version_count": 49,
            "runtime_verified_version_count": 49,
            "reusable_version_count": 1,
            "blueprint_generation_enabled": False,
            "compounds": [
                {
                    "compound_id": "compound-1",
                    "compound_name": "midazolam",
                    "same_compound_model_version_count": 1,
                    "exact_family_model_version_count": 1,
                    "reusable_model_version_count": 0,
                }
            ],
        },
    )
    candidate = {
        "rank": 1,
        "external_model_version_id": "model-version:midazolam",
        "model_name": "Midazolam PBPK model",
        "project_relative_path": "Midazolam/model.pksim5",
        "candidate_layer": "same_compound_qualification_candidate",
        "candidate_layer_label": "Same-compound qualification candidate",
        "candidate_class": "exact_compound_model",
        "direct_reuse_allowed": False,
        "deterministic_compatibility": 0.8,
        "run_label_grade": 0,
        "license_status": "unresolved",
        "execution_probe_status": "passed",
        "supported_compounds": ["Midazolam"],
        "routes": ["Oral"],
        "exclusion_reason_codes": ["LICENSE_UNRESOLVED"],
        "qualification_actions": ["resolve_model_license"],
        "hard_blockers": [],
        "asset_locations": [],
    }
    monkeypatch.setattr(
        api_client,
        "rank_public_model",
        lambda payload: {
            "status": "same_compound_candidates_ranked_reuse_blocked",
            "decision": "abstain",
            "candidate_summary": {
                "retrieved": 49,
                "catalog_scanned": 49,
                "relevant_candidates": 1,
                "excluded_unrelated": 48,
                "same_compound": 1,
                "reuse_ready": 0,
                "validated_reuse": 0,
                "controlled_adaptation": 0,
                "same_compound_qualification": 1,
                "mechanistic_reference": 0,
                "blocked": 0,
                "mechanistic_template": 48,
            },
            "candidates": [candidate],
            "candidates_by_layer": {
                "same_compound_qualification_candidate": [candidate]
            },
            "same_compound_candidates": [candidate],
            "mechanistic_templates": [],
            "top_same_compound_candidate": candidate,
            "ranking": {"candidates": []},
            "selected_model": None,
            "abstention_reason_codes": ["LICENSE_UNRESOLVED"],
            "blueprint_status": "eligible_for_template_lookup",
            "requirement_profile": {},
            "methodology": {},
        },
    )

    app = AppTest.from_file(
        str(ROOT / "pbpk_kb/ui/pages/14_recommendation_agent.py")
    ).run(timeout=10)
    rank_button = next(
        button
        for button in app.button
        if button.label == "Rank downloaded public models"
    )
    app = rank_button.click().run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(
        metric.label == "Catalog scanned" and metric.value == "49"
        for metric in app.metric
    )
    assert any(
        metric.label == "Relevant candidates" and metric.value == "1"
        for metric in app.metric
    )
    assert any(
        metric.label == "Unrelated excluded" and metric.value == "48"
        for metric in app.metric
    )
    assert any(
        selectbox.label == "Candidate layer"
        for selectbox in app.selectbox
    )
    assert len(app.dataframe) >= 1


def test_recommendation_agent_renders_executable_blueprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_client, "blueprint_package_bytes", lambda value: b"PK-test-package")
    blueprint_id = "a3e90c0f-70a5-5f92-b55f-2101591b90d4"
    monkeypatch.setattr(
        api_client,
        "public_model_catalog_summary",
        lambda: {
            "model_count": 41,
            "model_version_count": 49,
            "runtime_verified_version_count": 49,
            "reusable_version_count": 1,
            "blueprint_generation_enabled": True,
            "executable_blueprint_template_count": 1,
            "compounds": [
                {
                    "compound_id": "compound-1",
                    "compound_name": "Caffeine",
                    "same_compound_model_version_count": 0,
                    "exact_family_model_version_count": 0,
                    "reusable_model_version_count": 0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        api_client,
        "rank_public_model",
        lambda payload: {
            "status": "no_same_compound_public_model",
            "decision": "abstain",
            "candidate_summary": {
                "retrieved": 0,
                "same_compound": 0,
                "reuse_ready": 0,
                "validated_reuse": 0,
                "controlled_adaptation": 0,
                "same_compound_qualification": 0,
                "mechanistic_reference": 0,
                "blocked": 0,
                "mechanistic_template": 0,
            },
            "candidates": [],
            "candidates_by_layer": {},
            "same_compound_candidates": [],
            "mechanistic_templates": [],
            "abstention_reason_codes": ["NO_EXACT_COMPOUND_MODEL"],
            "blueprint_status": "eligible_for_template_lookup",
            "requirement_profile": {},
            "methodology": {},
        },
    )
    monkeypatch.setattr(
        api_client,
        "create_blueprint",
        lambda payload: {
            "status": "executable_template_available",
            "decision": "blueprint",
            "blueprint": {"blueprint_id": blueprint_id},
            "reason_codes": [],
        },
    )
    monkeypatch.setattr(
        api_client,
        "blueprint_detail",
        lambda value: {
            "blueprint_id": blueprint_id,
            "compound_name": "Caffeine",
            "objective": "public_model_reuse",
            "current_status": "executable",
            "scientific_status": "absorption_structure_failed",
            "registration_allowed": False,
            "blocker_codes": ["ABSORPTION_SET_EMPTY"],
            "external_template_id": "caffeine-pksim-v5",
            "template_version": "1.0.0",
            "target_runtime_adapter": "pksim_cli_snapshot",
            "runtime_version": "12.3.173",
            "counts": {
                "compartments": 1,
                "processes": 1,
                "parameters": 1,
                "artifacts": 1,
                "simulation_runs": 1,
                "outputs": 3,
            },
            "package": {
                "available": True,
                "byte_size": 100,
                "sha256": "a" * 64,
            },
            "blueprint_checksum": "b" * 64,
            "template_checksum": "c" * 64,
            "profile_checksum": "d" * 64,
            "validation_report_json": {"accepted": True},
            "compilation_report_json": {"status": "compiled"},
        },
    )
    monkeypatch.setattr(
        api_client,
        "blueprint_versions",
        lambda value: [
            {
                "blueprint_version": "1.0.3",
                "status": "executable",
                "blueprint_checksum": "b" * 64,
            }
        ],
    )
    monkeypatch.setattr(
        api_client,
        "blueprint_parameters",
        lambda value: [
            {
                "parameter_name": "Molecular weight",
                "semantic_id": "parameter:mw",
                "origin": "observed",
                "value": 194.19,
                "initial_value": None,
                "canonical_unit": "g/mol",
                "blocking": True,
                "provenance_json": [],
            }
        ],
    )
    monkeypatch.setattr(
        api_client,
        "blueprint_components",
        lambda value, kind: [
            {
                "source_location": "PKML:model",
                "payload_json": (
                    {
                        "name": "CYP1A2 metabolism",
                        "process_type": "metabolism",
                    }
                    if kind == "process"
                    else {
                        "name": "Liver",
                        "compartment_type": "organ",
                    }
                ),
            }
        ],
    )
    monkeypatch.setattr(
        api_client,
        "blueprint_artifacts",
        lambda value: [
            {
                "artifact_kind": "model",
                "relative_path": "model.pkml",
                "byte_size": 100,
                "sha256": "e" * 64,
            }
        ],
    )
    monkeypatch.setattr(
        api_client,
        "blueprint_simulation_runs",
        lambda value: [
            {
                "scenario_id": "CAL_PO",
                "run_ordinal": 1,
                "status": "succeeded",
                "output_count": 3,
            }
        ],
    )

    app = AppTest.from_file(
        str(ROOT / "pbpk_kb/ui/pages/14_recommendation_agent.py")
    ).run(timeout=10)
    rank_button = next(
        button
        for button in app.button
        if button.label == "Rank downloaded public models"
    )
    app = rank_button.click().run(timeout=10)
    blueprint_button = next(
        button
        for button in app.button
        if button.label == "Check registered executable blueprint"
    )
    app = blueprint_button.click().run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(
        subheader.value == "Executable blueprint workspace"
        for subheader in app.subheader
    )
    assert any(
        metric.label == "Scientific status"
        and metric.value == "absorption_structure_failed"
        for metric in app.metric
    )
    assert len(app.get("download_button")) >= 1


def test_compound_search_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "search_compounds", lambda query, limit=25: [sample_compound()])

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/03_compound_search.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "Compound Search" for title in app.title)
    assert len(app.dataframe) == 1
    assert any(button.label == "Open profile" for button in app.button)


def test_evidence_search_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "source_ids", lambda: ["dailymed"])
    monkeypatch.setattr(data, "source_families", lambda: ["label"])
    monkeypatch.setattr(data, "curation_statuses", lambda: ["needs_review"])
    monkeypatch.setattr(data, "evidence_search", lambda **kwargs: [sample_evidence()])

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/05_evidence_search.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "Evidence Search" for title in app.title)
    assert len(app.dataframe) == 1


def test_model_registry_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "source_ids", lambda: ["osp_pbpk_model_library"])
    monkeypatch.setattr(data, "validation_statuses", lambda: ["metadata_indexed"])
    monkeypatch.setattr(data, "model_registry", lambda **kwargs: [sample_model()])
    monkeypatch.setattr(
        api_client,
        "get_model_detail",
        lambda model_id: {
            "model": sample_model(),
            "versions": [],
            "assets": [],
            "simulations": [],
            "validation_runs": [],
            "validation_datasets": [],
            "validation_metrics": [],
            "parameters": [],
        },
    )

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/06_model_registry.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "Public PBPK Model Registry" for title in app.title)
    assert len(app.dataframe) == 1


def test_admin_settings_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_client, "list_sources", lambda **kwargs: [sample_source()])
    monkeypatch.setattr(
        api_client,
        "list_ingestion_runs",
        lambda **kwargs: [
            {
                "ingestion_run_id": "11111111-1111-1111-1111-111111111111",
                "source_id": "pubchem",
                "status": "failed",
                "completed_at": "2026-06-13T00:00:01",
                "raw_record_count": 0,
                "parsed_record_count": 0,
                "log_path": "ingestion_logs/pubchem/latest.json",
            }
        ],
    )

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/13_admin_settings.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "Admin Settings" for title in app.title)
    assert len(app.dataframe) >= 1


def test_ingestion_monitor_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_client, "list_sources", lambda **kwargs: [sample_source()])
    monkeypatch.setattr(
        api_client,
        "list_ingestion_runs",
        lambda **kwargs: [
            {
                "ingestion_run_id": "11111111-1111-1111-1111-111111111111",
                "source_id": "pubchem",
                "status": "succeeded",
                "started_at": "2026-06-13T00:00:00",
                "completed_at": "2026-06-13T00:00:01",
                "raw_record_count": 1,
                "parsed_record_count": 1,
                "evidence_record_count": 3,
                "source_version": "snapshot:2026-06-13",
                "log_path": "ingestion_logs/pubchem/latest.json",
                "error_summary": None,
            }
        ],
    )

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/11_ingestion_monitor.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "Ingestion Monitor" for title in app.title)
    assert len(app.dataframe) >= 1


def test_knowledge_graph_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "search_compounds", lambda query, limit=250: [sample_compound()])
    monkeypatch.setattr(
        api_client,
        "graph_summary",
        lambda: {
            "compounds": 1,
            "evidence_records": 1,
            "sources": 1,
            "public_models": 0,
            "protein_targets": 1,
            "enzyme_transporter_relationships": 1,
        },
    )
    monkeypatch.setattr(api_client, "compound_graph", lambda compound_id: sample_graph())

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/12_knowledge_graph.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "Knowledge Graph Explorer" for title in app.title)
    assert len(app.dataframe) >= 2


def test_review_queue_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "source_ids", lambda: ["dailymed"])
    monkeypatch.setattr(api_client, "review_queue", lambda **kwargs: [sample_evidence()])

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/07_review_queue.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "Review Queue" for title in app.title)
    assert len(app.dataframe) >= 1
    assert any(button.label == "Open evidence detail" for button in app.button)


def test_evidence_detail_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_client, "review_queue", lambda **kwargs: [sample_evidence()])
    monkeypatch.setattr(api_client, "evidence_detail", lambda evidence_type, evidence_id: sample_detail())

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/08_evidence_detail.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "Evidence Detail" for title in app.title)
    assert len(app.dataframe) >= 3


def test_qa_assistant_streamlit_smoke() -> None:
    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/09_qa_assistant.py"))
    app.session_state["qa_result"] = sample_qa_result()

    app.run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "Q&A Assistant" for title in app.title)
    assert len(app.dataframe) >= 4


def test_exports_streamlit_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "source_ids", lambda: ["dailymed"])
    monkeypatch.setattr(data, "curation_statuses", lambda: ["needs_review"])
    monkeypatch.setattr(data, "validation_statuses", lambda: ["metadata_indexed"])
    monkeypatch.setattr(api_client, "export_jobs", lambda limit=25: [sample_export_result()["export_job"]])

    app = AppTest.from_file(str(ROOT / "pbpk_kb/ui/pages/10_exports.py")).run(timeout=10)

    assert_no_streamlit_exceptions(app)
    assert any(title.value == "Exports" for title in app.title)
    assert len(app.dataframe) >= 1
