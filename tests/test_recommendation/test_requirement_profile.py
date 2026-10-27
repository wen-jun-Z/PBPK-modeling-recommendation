from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from pbpk_kb.recommendation.requirement_profile import (
    RequirementProfileBuildError,
    audit_requirement_profiles,
    build_requirement_profile,
)
from pbpk_kb.recommendation.schemas import (
    Citation,
    ComponentRecommendation,
    CompoundRef,
    ModelingRequest,
    RecommendationResponse,
    RequirementProfile,
)


def _compound() -> CompoundRef:
    return CompoundRef(
        compound_id=UUID("187ad469-6d60-42f4-a556-8b84c47d4df5"),
        preferred_name="Midazolam",
        active_moiety="Midazolam",
        external_identifiers={"pubchem_cid": "4192"},
        identity_resolution_method="database_normalized_name",
        identity_confidence=1.0,
    )


def _component(name: str) -> ComponentRecommendation:
    evidence_id = uuid4()
    return ComponentRecommendation(
        component=name,
        status="recommended",
        rationale="Validated evidence coverage.",
        citations=[
            Citation(
                evidence_id=str(evidence_id),
                source_id="test_source",
                source_location=f"TEST:{evidence_id}",
                source_text=f"{name} evidence",
                confidence=0.9,
            )
        ],
        evidence_count=1,
        source_count=1,
        confidence=0.9,
    )


def _recommendation() -> RecommendationResponse:
    return RecommendationResponse(
        compound_id=str(_compound().compound_id),
        compound_name="Midazolam",
        objective="base_pk",
        retrieval_policy="test",
        coverage_score=1.0,
        overall_confidence=0.9,
        components=[
            _component("absorption"),
            _component("distribution"),
            _component("metabolism"),
            _component("excretion"),
        ],
        ranked_subgraphs=[],
        public_model_candidates=[],
        evidence_gaps=[],
        warnings=[],
    )


def test_requirement_profile_is_typed_grounded_and_checksum_valid() -> None:
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="base_pk",
            route="oral",
            dose=5,
            dose_unit="mg",
            frequency="single dose",
            prediction_endpoints=["AUC", "Cmax"],
        ),
        compound_identity=_compound(),
        source_evidence_run_id=uuid4(),
        ontology_version="pbpk-ontology-1.0.0",
        component_recommendation=_recommendation(),
    )

    assert profile.ranking_eligibility == "ready"
    assert profile.critical_gaps == []
    assert {
        item.semantic_id
        for item in profile.requirements
        if item.category == "adme_process"
    } == {
        "process:absorption",
        "process:distribution",
        "process:metabolism",
        "process:excretion",
    }
    assert RequirementProfile.model_validate(
        profile.model_dump(mode="json")
    ) == profile
    audit = audit_requirement_profiles([profile])
    assert audit["accepted"] is True

    known_provenance_ids = {
        evidence_id
        for requirement in profile.requirements
        if requirement.origin != "user_request"
        for evidence_id in requirement.evidence_ids
    }
    strict_audit = audit_requirement_profiles(
        [profile],
        known_provenance_ids=known_provenance_ids,
    )
    assert strict_audit["accepted"] is True
    assert strict_audit["checks"]["provenance_references_resolve"] is True
    assert {
        item.preferred_name
        for item in profile.requirements
        if item.category == "dose_regimen"
    } == {"dose: 5.0", "frequency: single dose"}


def test_explicit_unmodeled_request_fields_become_hard_requirements() -> None:
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="public_model_reuse",
            salt_form="hydrochloride",
            body_weight_kg=70,
            validation_datasets=["external-study-1"],
        ),
        compound_identity=_compound(),
        source_evidence_run_id=uuid4(),
        ontology_version="pbpk-ontology-1.0.0",
    )

    constraints = [
        item
        for item in profile.requirements
        if item.category == "request_constraint"
    ]
    assert {item.preferred_name.split(":", 1)[0] for item in constraints} == {
        "salt_form",
        "body_weight_kg",
        "validation_dataset",
    }
    assert all(item.necessity == "hard_required" for item in constraints)


def test_requirement_profile_blocks_missing_hard_evidence() -> None:
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="base_pk",
        ),
        compound_identity=_compound(),
        source_evidence_run_id=uuid4(),
        ontology_version="pbpk-ontology-1.0.0",
    )

    assert profile.ranking_eligibility == "blocked"
    assert len(profile.critical_gaps) == 4
    assert all(gap.blocking for gap in profile.critical_gaps)


def test_requirement_profile_rejects_non_uuid_evidence_reference() -> None:
    recommendation = _recommendation()
    recommendation.components[0].citations[0].evidence_id = "legacy-id"

    with pytest.raises(
        RequirementProfileBuildError,
        match="must reference a UUID",
    ):
        build_requirement_profile(
            request=ModelingRequest(
                compound="midazolam",
                objective="base_pk",
            ),
            compound_identity=_compound(),
            source_evidence_run_id=uuid4(),
            ontology_version="pbpk-ontology-1.0.0",
            component_recommendation=recommendation,
        )


def test_public_model_reuse_defers_mechanism_checks_to_candidates() -> None:
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="Rosuvastatin",
            objective="public_model_reuse",
            species="human",
            route="oral",
        ),
        compound_identity=CompoundRef(
            compound_id=UUID("06c1679c-baed-5f3a-b87e-a16549e00e7a"),
            preferred_name="Rosuvastatin",
            active_moiety="Rosuvastatin",
            external_identifiers={"pubchem_cid": "446157"},
            identity_resolution_method="database_exact_id",
            identity_confidence=1.0,
        ),
        source_evidence_run_id=UUID(
            "bca5afe3-bdd7-5df1-bcf7-0b37789ec8af"
        ),
        ontology_version="pbpk-recommendation-ontology-1.1.0",
    )

    assert profile.ranking_eligibility == "ready"
    assert profile.critical_gaps == []
    assert {
        requirement.preferred_name
        for requirement in profile.requirements
        if requirement.category == "adme_process"
    } == {"absorption", "distribution", "metabolism", "excretion"}
