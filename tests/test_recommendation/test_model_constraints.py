from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from pbpk_kb.model_assets.models import (
    ExecutionProbeStatus,
    LicenseDecision,
    LicenseStatus,
    ModelValidationRunLabel,
    ModelValidationScenario,
    ModelValidationScope,
    OSPModelVersionAssetReference,
    OSPModelVersionSnapshot,
    ParseStatus,
)
from pbpk_kb.recommendation.model_constraints import (
    evaluate_model_constraints,
    model_version_uuid,
)
from pbpk_kb.recommendation.model_candidate_retrieval import (
    build_ranked_candidate,
    candidate_layer_decision,
    retrieve_model_candidates,
)
from pbpk_kb.recommendation.model_ranker import (
    audit_ranking_run,
    build_split_manifest,
    build_ranking_recommendation,
)
from pbpk_kb.recommendation.model_relevance import (
    ModelMechanismIndex,
    ModelMechanismPath,
    TargetMechanism,
    candidate_relevance_decision,
    unavailable_model_mechanism_index,
)
from pbpk_kb.recommendation.requirement_profile import (
    build_requirement_profile,
)
from pbpk_kb.recommendation.schemas import CompoundRef, ModelingRequest


def _profile():
    return build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="ddi_victim",
            perpetrator_drug="rifampicin",
            required_enzymes=["CYP3A4"],
        ),
        compound_identity=CompoundRef(
            compound_id=UUID(
                "187ad469-6d60-42f4-a556-8b84c47d4df5"
            ),
            preferred_name="Midazolam",
            active_moiety="Midazolam",
            identity_resolution_method="database_normalized_name",
            identity_confidence=1.0,
        ),
        source_evidence_run_id=UUID(
            "4e5935bb-f6dc-4470-a434-3cd40aa0c38f"
        ),
        ontology_version="pbpk-ontology-1.0.0",
    )


def _model() -> OSPModelVersionSnapshot:
    source_location = (
        "GITHUB_BLOB:commit/Midazolam/Midazolam-Model.pksim5"
    )
    checksum = "a" * 64
    return OSPModelVersionSnapshot(
        model_id="model:midazolam",
        model_version_id="model-version:midazolam",
        model_family="Midazolam",
        project_key="b" * 32,
        project_relative_path="Midazolam/Midazolam-Model.pksim5",
        project_sha256=checksum,
        source_id="osp_pbpk_model_library",
        repository_url="https://example.test/osp.git",
        repository_commit="c" * 40,
        software="PK-Sim",
        software_version="12.3.173",
        license=LicenseDecision(
            status=LicenseStatus.UNRESOLVED,
            reason_code="LICENSE_FILE_MISSING",
        ),
        static_parse_status=ParseStatus.REGISTERED_NOT_EXECUTABLE,
        execution_probe_status=ExecutionProbeStatus.PASSED,
        pkml_parse_status=ParseStatus.PARSED,
        reusable_candidate=False,
        compounds=["Midazolam", "Rifampicin"],
        metabolites=[],
        species=["Human"],
        populations=["European"],
        routes=["Intravenous", "Oral"],
        formulations=["Solution"],
        enzymes=["CYP3A4"],
        transporters=[],
        compartments=["Liver", "Kidney", "Lung", "Gut", "Plasma"],
        process_types=["absorption", "metabolism", "renal_excretion"],
        output_names=["Midazolam plasma concentration", "AUC"],
        simulation_count=1,
        parameter_record_count=10,
        component_record_count=5,
        process_record_count=3,
        report_count=0,
        dataset_reference_count=0,
        validation_metric_count=0,
        assets=[
            OSPModelVersionAssetReference(
                asset_kind="pksim_project",
                relative_path=(
                    "Midazolam/Midazolam-Model.pksim5"
                ),
                sha256=checksum,
                source_location=source_location,
            )
        ],
        report_paths=[],
        blocking_reasons=["LICENSE_FILE_MISSING"],
    )


def _grade_3_model() -> OSPModelVersionSnapshot:
    source_snapshot_checksum = "4" * 64
    payload = _model().model_dump(mode="python")
    payload.update(
        {
            "license": LicenseDecision(
                status=LicenseStatus.ALLOWED,
                spdx_id="GPL-2.0-only",
                source_location="GITHUB_BLOB:commit/LICENSE",
                source_checksum="d" * 64,
                reason_code="LICENSE_ALLOWED_BY_SPDX_POLICY",
            ),
            "static_parse_status": ParseStatus.PARSED,
            "reusable_candidate": True,
            "validation_run_label": ModelValidationRunLabel(
                validation_run_id="snapshot-validation-run:test",
                protocol_id="snapshot-reuse-test",
                protocol_sha256="1" * 64,
                windows_manifest_sha256="2" * 64,
                audit_report_sha256="3" * 64,
                source_snapshot_sha256=source_snapshot_checksum,
                validation_grade="external_study_validated",
                run_label_grade=3,
                ranking_label="reuse_positive",
                engineering_accepted=True,
                scientific_accepted=True,
                reusable_candidate=True,
                validation_independence=(
                    "repository_preregistered_holdout"
                ),
                verification_dataset_count=2,
                pooled_observation_count=10,
                pooled_gmfe=1.2,
                pooled_geometric_bias=1.0,
                pooled_two_fold_fraction=0.9,
                endpoint_scenario_pass_fraction=1.0,
                protocol_path="/audit/protocol.json",
                windows_manifest_path="/audit/windows.json",
                audit_report_path="/audit/report.json",
                validation_dataset_metrics_path="/audit/datasets.jsonl",
                validated_at=datetime.now(UTC),
            ),
            "validation_scope": ModelValidationScope(
                species=["Human"],
                populations=["Healthy adults"],
                routes=["Oral"],
                formulations=["Solution"],
                evaluated_endpoints=["concentration_time_profile"],
                dataset_names=["validation-1", "validation-2"],
                scenarios=[
                    ModelValidationScenario(
                        simulation_id=f"validation-{index}",
                        simulation_name=f"po 10 mg validation {index}",
                        dataset_name=f"validation-{index}",
                        source_location=f"TEST:validation-{index}",
                        species=["Human"],
                        populations=["Healthy adults"],
                        routes=["Oral"],
                        formulations=["Solution"],
                        evaluated_endpoints=[
                            "concentration_time_profile"
                        ],
                        dose=10.0,
                        dose_unit="mg",
                        frequency="single_dose",
                        single_or_multiple="single",
                    )
                    for index in (1, 2)
                ],
            ),
            "blocking_reasons": [],
        }
    )
    payload["assets"].append(
        OSPModelVersionAssetReference(
            asset_kind="repository_snapshot",
            relative_path="repository/source_snapshot.tar.zst",
            sha256=source_snapshot_checksum,
            source_location="GITHUB_ARCHIVE:commit",
        )
    )
    return OSPModelVersionSnapshot.model_validate(payload)


def test_reusable_snapshot_requires_bound_grade_3_label() -> None:
    payload = _model().model_dump(mode="python")
    payload["reusable_candidate"] = True

    with pytest.raises(
        ValueError,
        match="must be derived from a bound Grade 3 validation label",
    ):
        OSPModelVersionSnapshot.model_validate(payload)


def test_reusable_snapshot_requires_matching_source_asset() -> None:
    payload = _grade_3_model().model_dump(mode="python")
    payload["assets"] = [
        asset
        for asset in payload["assets"]
        if asset["asset_kind"] != "repository_snapshot"
    ]

    with pytest.raises(
        ValueError,
        match="checksum-bound project and source assets",
    ):
        OSPModelVersionSnapshot.model_validate(payload)


def test_constraints_preserve_assets_and_block_unresolved_license() -> None:
    model = _model()
    checks = evaluate_model_constraints(
        profile=_profile(),
        model=model,
    )

    assert all(
        check.model_version_id
        == model_version_uuid(model.model_version_id)
        for check in checks
    )
    license_check = next(
        check for check in checks if ":license:" in check.constraint_id
    )
    assert license_check.result == "fail"
    assert license_check.reason_code == "LICENSE_UNRESOLVED"
    assert license_check.model_asset_locations
    identity_check = next(
        check
        for check in checks
        if ":compound_identity:" in check.constraint_id
    )
    assert identity_check.result == "pass"
    enzyme_check = next(
        check for check in checks if ":enzyme:" in check.constraint_id
    )
    assert enzyme_check.result == "pass"
    parser_check = next(
        check for check in checks if ":parser:" in check.constraint_id
    )
    assert parser_check.result == "pass"
    gap_checks = [
        check
        for check in checks
        if ":profile_gap:" in check.constraint_id
    ]
    assert gap_checks
    assert {
        check.reason_code for check in gap_checks
    } == {"CRITICAL_EVIDENCE_GAP"}


def test_candidate_keeps_constraint_matrix_and_does_not_score_reuse() -> None:
    candidate = build_ranked_candidate(
        profile=_profile(),
        model=_model(),
    )

    assert candidate.candidate_class == "exact_compound_model"
    assert candidate.eligibility == "ineligible"
    assert candidate.final_score is None
    assert candidate.learned_graph_compatibility is None
    assert "LICENSE_UNRESOLVED" in candidate.exclusion_reason_codes
    assert "CRITICAL_EVIDENCE_GAP" in candidate.exclusion_reason_codes
    assert all(
        feature.weight == 0
        for feature in candidate.score_features
    )


def test_unresolved_same_compound_model_is_qualification_candidate() -> None:
    profile = _profile()
    model = _model()
    candidate = build_ranked_candidate(profile=profile, model=model)

    layer = candidate_layer_decision(
        profile=profile,
        model=model,
        candidate=candidate,
    )

    assert layer["candidate_layer"] == (
        "same_compound_qualification_candidate"
    )
    assert layer["direct_reuse_allowed"] is False
    assert "resolve_model_license" in layer["qualification_actions"]
    assert "complete_independent_external_validation" in (
        layer["qualification_actions"]
    )


def test_different_compound_model_requires_relevance_proof() -> None:
    profile = _profile()
    model = _model().model_copy(
        update={
            "model_family": "Caffeine",
            "compounds": ["Caffeine"],
        }
    )
    candidate = build_ranked_candidate(profile=profile, model=model)

    with pytest.raises(ValueError, match="verified shared-mechanism"):
        candidate_layer_decision(
            profile=profile,
            model=model,
            candidate=candidate,
        )


def test_verified_shared_process_becomes_mechanistic_reference() -> None:
    profile = _profile()
    model = _model().model_copy(
        update={
            "model_family": "Caffeine",
            "compounds": ["Caffeine"],
        }
    )
    candidate = build_ranked_candidate(profile=profile, model=model)
    mechanism = TargetMechanism(
        mechanism_type="enzyme",
        canonical_name="CYP3A4",
        aliases=("Cytochrome P450 3A4",),
        origins=("auto_validated_evidence",),
        relationship_types=("substrate_of",),
        relationship_ids=("relationship-1",),
        evidence_ids=("evidence-1",),
        source_locations=("SOURCE:target-evidence",),
    )
    mechanism_index = ModelMechanismIndex(
        status="accepted",
        reason_code=None,
        graph_summary_sha256="a" * 64,
        graph_sqlite_sha256="b" * 64,
        model_version_count=1,
        paths=(
            ModelMechanismPath(
                model_version_id=model.model_version_id,
                project_relative_path=model.project_relative_path,
                process_node_id="process:caffeine-metabolism",
                process_name="Caffeine-CYP3A4 metabolism",
                process_type="metabolism",
                mechanism_type="enzyme",
                mechanism_name="CYP3A4",
                compound_educt_names=("Caffeine",),
                ownership_edge_id="edge:owns-process",
                mechanism_edge_id="edge:modified-by",
                source_id="osp_pbpk_model_library",
                source_location="PKML:process:caffeine-metabolism",
                asset_sha256="c" * 64,
                extraction_method="pkml_structured_parser",
            ),
        ),
    )
    relevance = candidate_relevance_decision(
        profile=profile,
        model=model,
        candidate=candidate,
        target_mechanisms=[mechanism],
        mechanism_index=mechanism_index,
    )

    layer = candidate_layer_decision(
        profile=profile,
        model=model,
        candidate=candidate,
        relevance_decision=relevance,
    )

    assert relevance["is_relevant"] is True
    assert relevance["relevance_basis"][0]["target_evidence_ids"] == [
        "evidence-1"
    ]
    assert relevance["relevance_basis"][0]["model_process_paths"][0][
        "mechanism_edge_id"
    ] == "edge:modified-by"
    assert layer["candidate_layer"] == "mechanistic_reference"
    assert layer["direct_reuse_allowed"] is False
    assert "do_not_transfer_compound_specific_parameters" in (
        layer["qualification_actions"]
    )


def test_different_compound_model_without_target_mechanism_is_unrelated() -> None:
    profile = _profile()
    model = _model().model_copy(
        update={"model_family": "Caffeine", "compounds": ["Caffeine"]}
    )
    candidate = build_ranked_candidate(profile=profile, model=model)

    relevance = candidate_relevance_decision(
        profile=profile,
        model=model,
        candidate=candidate,
        target_mechanisms=[],
        mechanism_index=unavailable_model_mechanism_index(
            "MODEL_KG_NOT_NEEDED_NO_TARGET_MECHANISM"
        ),
    )

    assert relevance["is_relevant"] is False
    assert relevance["relevance_reason_codes"] == [
        "NO_VERIFIED_TARGET_DISPOSITION_MECHANISM"
    ]


def test_different_compound_model_fails_closed_without_accepted_kg() -> None:
    profile = _profile()
    model = _model().model_copy(
        update={"model_family": "Caffeine", "compounds": ["Caffeine"]}
    )
    candidate = build_ranked_candidate(profile=profile, model=model)
    mechanism = TargetMechanism(
        mechanism_type="enzyme",
        canonical_name="CYP3A4",
        aliases=(),
        origins=("user_request",),
        relationship_types=("explicit_requirement",),
        relationship_ids=(),
        evidence_ids=(),
        source_locations=(),
    )

    relevance = candidate_relevance_decision(
        profile=profile,
        model=model,
        candidate=candidate,
        target_mechanisms=[mechanism],
        mechanism_index=unavailable_model_mechanism_index(
            "MODEL_KG_INTEGRITY_OR_READ_FAILURE"
        ),
    )

    assert relevance["is_relevant"] is False
    assert relevance["relevance_reason_codes"] == [
        "MODEL_KG_INTEGRITY_OR_READ_FAILURE"
    ]


def test_failed_execution_model_is_blocked() -> None:
    profile = _profile()
    model = _model().model_copy(
        update={
            "execution_probe_status": ExecutionProbeStatus.FAILED,
        }
    )
    candidate = build_ranked_candidate(profile=profile, model=model)

    layer = candidate_layer_decision(
        profile=profile,
        model=model,
        candidate=candidate,
    )

    assert layer["candidate_layer"] == "blocked"
    assert "EXECUTION_PROBE_FAILED" in layer["technical_blocker_codes"]
    assert "rerun_runtime_execution_probe" in (
        layer["qualification_actions"]
    )


def test_secondary_drug_in_ddi_model_is_not_exact_family_model() -> None:
    candidate = build_ranked_candidate(
        profile=_profile(),
        model=_model().model_copy(
            update={
                "model_family": "Rifampicin",
                "compounds": ["Rifampicin", "Midazolam"],
            }
        ),
    )

    assert candidate.candidate_class == "mechanistic_template"


def test_general_request_prefers_general_over_pediatric_model() -> None:
    adult = _model()
    pediatric = _model().model_copy(
        update={
            "model_version_id": "model-version:midazolam-pediatric",
            "project_relative_path": (
                "Midazolam/Midazolam_Pediatrics.pksim5"
            ),
            "populations": ["Neonatal infant", "Pediatric"],
            "parameter_record_count": 100,
        }
    )

    candidates = retrieve_model_candidates(
        profile=_profile(),
        models=[pediatric, adult],
    )

    assert candidates[0].source_model_version_key == adult.model_version_id


def test_rule_only_ranker_forces_auditable_abstention() -> None:
    profile = _profile()
    candidate = build_ranked_candidate(
        profile=profile,
        model=_model(),
    )
    recommendation = build_ranking_recommendation(
        profile=profile,
        candidates=[candidate],
        model_run_manifest_id=UUID(
            "ee94d0c3-c9bb-464a-bd0f-c9f91be183c9"
        ),
        auto_reuse_enabled=False,
    )

    assert recommendation.decision == "abstain"
    assert recommendation.selected_model_version_id is None
    assert "LICENSE_UNRESOLVED" in (
        recommendation.abstention_reason_codes
    )
    assert "TRAINING_LABELS_INSUFFICIENT" in (
        recommendation.abstention_reason_codes
    )
    assert recommendation.auto_reuse_enabled is False


def test_grade_3_exact_candidate_enables_deterministic_reuse() -> None:
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="public_model_reuse",
            route="oral",
            formulation="solution",
            dose=10,
            dose_unit="mg",
            frequency="single dose",
            single_or_multiple="single",
            prediction_endpoints=["concentration_time_profile"],
        ),
        compound_identity=_profile().compound_identity,
        source_evidence_run_id=UUID(
            "4e5935bb-f6dc-4470-a434-3cd40aa0c38f"
        ),
        ontology_version="pbpk-ontology-1.0.0",
    )
    model = _grade_3_model()
    candidate = build_ranked_candidate(profile=profile, model=model)

    recommendation = build_ranking_recommendation(
        profile=profile,
        candidates=[candidate],
        model_run_manifest_id=UUID(
            "ee94d0c3-c9bb-464a-bd0f-c9f91be183c9"
        ),
        auto_reuse_enabled=True,
    )

    assert candidate.eligibility == "eligible"
    assert candidate.run_label_grade == 3
    assert recommendation.decision == "reuse"
    assert recommendation.selected_model_version_id == candidate.model_version_id
    assert recommendation.abstention_reason_codes == []
    layer = candidate_layer_decision(
        profile=profile,
        model=model,
        candidate=candidate,
    )
    assert layer["candidate_layer"] == "validated_reuse"
    assert layer["direct_reuse_allowed"] is True


def test_grade_3_reuse_abstains_when_scope_request_is_incomplete() -> None:
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="public_model_reuse",
        ),
        compound_identity=_profile().compound_identity,
        source_evidence_run_id=UUID(
            "4e5935bb-f6dc-4470-a434-3cd40aa0c38f"
        ),
        ontology_version="pbpk-ontology-1.0.0",
    )
    candidate = build_ranked_candidate(
        profile=profile,
        model=_grade_3_model(),
    )

    assert candidate.eligibility == "ineligible"
    scope_check = next(
        check
        for check in candidate.constraint_checks
        if ":validation_scope_transferability:" in check.constraint_id
    )
    assert scope_check.result == "unknown"
    assert scope_check.reason_code == "VALIDATION_NOT_TRANSFERABLE"
    assert scope_check.observed_value["missing_required_scope_fields"] == [
        "route",
        "dose",
        "dose_unit",
        "frequency",
        "single_or_multiple",
        "formulation",
        "prediction_endpoints",
    ]


def test_grade_3_reuse_does_not_treat_missing_regimen_as_wildcard() -> None:
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="public_model_reuse",
            route="oral",
            prediction_endpoints=["concentration_time_profile"],
        ),
        compound_identity=_profile().compound_identity,
        source_evidence_run_id=UUID(
            "4e5935bb-f6dc-4470-a434-3cd40aa0c38f"
        ),
        ontology_version="pbpk-ontology-1.0.0",
    )

    candidate = build_ranked_candidate(profile=profile, model=_grade_3_model())
    scope_check = next(
        check
        for check in candidate.constraint_checks
        if ":validation_scope_transferability:" in check.constraint_id
    )

    assert candidate.eligibility == "ineligible"
    assert scope_check.result == "unknown"
    assert scope_check.observed_value["missing_required_scope_fields"] == [
        "dose",
        "dose_unit",
        "frequency",
        "single_or_multiple",
        "formulation",
    ]


def test_grade_3_base_model_does_not_transfer_to_ddi_request() -> None:
    profile = _profile().model_copy(
        update={"critical_gaps": [], "conflicts": []}
    )
    model = _grade_3_model()

    candidate = build_ranked_candidate(profile=profile, model=model)

    assert candidate.eligibility == "ineligible"
    assert "VALIDATION_NOT_TRANSFERABLE" in candidate.exclusion_reason_codes


def test_grade_3_reuse_requires_matching_validation_scenario() -> None:
    identity = _profile().compound_identity
    matched_profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="public_model_reuse",
            route="oral",
            formulation="solution",
            dose=10,
            dose_unit="mg",
            frequency="single dose",
            single_or_multiple="single",
            prediction_endpoints=["concentration_time_profile"],
        ),
        compound_identity=identity,
        source_evidence_run_id=UUID(
            "4e5935bb-f6dc-4470-a434-3cd40aa0c38f"
        ),
        ontology_version="pbpk-ontology-1.0.0",
    )
    unmatched_profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="public_model_reuse",
            route="oral",
            formulation="solution",
            dose=20,
            dose_unit="mg",
            frequency="single dose",
            single_or_multiple="single",
            prediction_endpoints=["concentration_time_profile"],
        ),
        compound_identity=identity,
        source_evidence_run_id=UUID(
            "4e5935bb-f6dc-4470-a434-3cd40aa0c38f"
        ),
        ontology_version="pbpk-ontology-1.0.0",
    )

    matched = build_ranked_candidate(
        profile=matched_profile,
        model=_grade_3_model(),
    )
    unmatched = build_ranked_candidate(
        profile=unmatched_profile,
        model=_grade_3_model(),
    )

    assert matched.eligibility == "eligible"
    scope_check = next(
        check
        for check in matched.constraint_checks
        if ":validation_scope_transferability:" in check.constraint_id
    )
    assert scope_check.result == "pass"
    assert unmatched.eligibility == "ineligible"
    assert "VALIDATION_NOT_TRANSFERABLE" in (
        unmatched.exclusion_reason_codes
    )


def test_derived_endpoint_requires_registered_validation_metric() -> None:
    model_payload = _grade_3_model().model_dump(mode="python")
    model_payload["output_names"] = ["Midazolam plasma concentration"]
    model_payload["validation_scope"]["evaluated_endpoints"] = sorted(
        [
            *model_payload["validation_scope"]["evaluated_endpoints"],
            "AUC",
        ],
        key=str.casefold,
    )
    for scenario in model_payload["validation_scope"]["scenarios"]:
        scenario["evaluated_endpoints"] = sorted(
            [*scenario["evaluated_endpoints"], "AUC"],
            key=str.casefold,
        )
    model = OSPModelVersionSnapshot.model_validate(model_payload)
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="public_model_reuse",
            route="oral",
            formulation="solution",
            dose=10,
            dose_unit="mg",
            frequency="single dose",
            single_or_multiple="single",
            prediction_endpoints=["AUC"],
        ),
        compound_identity=_profile().compound_identity,
        source_evidence_run_id=UUID(
            "4e5935bb-f6dc-4470-a434-3cd40aa0c38f"
        ),
        ontology_version="pbpk-ontology-1.0.0",
    )

    candidate = build_ranked_candidate(profile=profile, model=model)

    assert candidate.eligibility == "eligible"
    endpoint_check = next(
        check
        for check in candidate.constraint_checks
        if ":prediction_endpoint:" in check.constraint_id
    )
    assert endpoint_check.result == "pass"


def test_small_benchmark_audit_reports_size_limits() -> None:
    profile = _profile()
    recommendation = build_ranking_recommendation(
        profile=profile,
        candidates=[
            build_ranked_candidate(
                profile=profile,
                model=_model(),
            )
        ],
        model_run_manifest_id=UUID(
            "ee94d0c3-c9bb-464a-bd0f-c9f91be183c9"
        ),
        auto_reuse_enabled=False,
    )

    audit = audit_ranking_run(
        profiles=[profile],
        recommendations=[recommendation],
        model_count=1,
        split_manifest={
            "compound_disjoint": {"status": "failed"}
        },
    )

    assert audit["accepted"] is False
    assert audit["checks"]["minimum_request_count_30"] is False
    assert audit["checks"]["false_reuse_count_zero"] is True


def test_compound_disjoint_split_is_rechecked() -> None:
    profile = _profile()
    split = build_split_manifest(
        profile_index=[
            {
                "profile_id": str(profile.profile_id),
                "benchmark_split": "test",
                "compound_id": str(
                    profile.compound_identity.compound_id
                ),
            }
        ],
        profiles=[profile],
        models=[_model()],
    )

    assert split["compound_disjoint"]["status"] == "failed"
    assert split["generalization_claim_eligible"] is False
    assert (
        split["source_repository_disjoint"]["status"]
        == "ineligible_single_source_repository"
    )


@pytest.mark.parametrize(
    ("dose", "unit", "expected_class"),
    [
        (9.99, "mg", "numerically_equivalent"),
        (10_000.0, "mcg", "exact"),
    ],
)
def test_grade_3_reuse_accepts_unit_normalized_equivalent_dose(
    dose: float,
    unit: str,
    expected_class: str,
) -> None:
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="public_model_reuse",
            route="oral",
            formulation="solution",
            dose=dose,
            dose_unit=unit,
            frequency="single dose",
            single_or_multiple="single",
            prediction_endpoints=["concentration_time_profile"],
        ),
        compound_identity=_profile().compound_identity,
        source_evidence_run_id=UUID(
            "4e5935bb-f6dc-4470-a434-3cd40aa0c38f"
        ),
        ontology_version="pbpk-ontology-1.0.0",
    )

    candidate = build_ranked_candidate(
        profile=profile,
        model=_grade_3_model(),
    )
    scope_check = next(
        check
        for check in candidate.constraint_checks
        if ":validation_scope_transferability:" in check.constraint_id
    )

    assert candidate.eligibility == "eligible"
    assert scope_check.result == "pass"
    assert (
        scope_check.observed_value["numeric_transferability"]["match_class"]
        == expected_class
    )


def test_changed_dose_remains_a_ranked_adaptation_candidate() -> None:
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="public_model_reuse",
            route="oral",
            formulation="solution",
            dose=15.0,
            dose_unit="mg",
            frequency="single dose",
            single_or_multiple="single",
            prediction_endpoints=["concentration_time_profile"],
        ),
        compound_identity=_profile().compound_identity,
        source_evidence_run_id=UUID(
            "4e5935bb-f6dc-4470-a434-3cd40aa0c38f"
        ),
        ontology_version="pbpk-ontology-1.0.0",
    )
    model = _grade_3_model()

    candidate = build_ranked_candidate(profile=profile, model=model)
    layer = candidate_layer_decision(
        profile=profile,
        model=model,
        candidate=candidate,
    )
    scope_check = next(
        check
        for check in candidate.constraint_checks
        if ":validation_scope_transferability:" in check.constraint_id
    )

    assert candidate.candidate_class == "same_compound_adaptation"
    assert layer["candidate_layer"] == "controlled_adaptation_candidate"
    assert layer["direct_reuse_allowed"] is False
    assert (
        scope_check.observed_value["numeric_transferability"]["match_class"]
        == "extrapolation"
    )


def test_unregistered_continuous_covariate_keeps_same_compound_candidate() -> None:
    profile = build_requirement_profile(
        request=ModelingRequest(
            compound="midazolam",
            objective="public_model_reuse",
            route="oral",
            formulation="solution",
            dose=10.0,
            dose_unit="mg",
            frequency="single dose",
            single_or_multiple="single",
            body_weight_kg=69.9,
            prediction_endpoints=["concentration_time_profile"],
        ),
        compound_identity=_profile().compound_identity,
        source_evidence_run_id=UUID(
            "4e5935bb-f6dc-4470-a434-3cd40aa0c38f"
        ),
        ontology_version="pbpk-ontology-1.0.0",
    )
    model = _grade_3_model()

    candidate = build_ranked_candidate(profile=profile, model=model)
    layer = candidate_layer_decision(
        profile=profile,
        model=model,
        candidate=candidate,
    )

    assert candidate.candidate_class == "same_compound_adaptation"
    assert layer["candidate_layer"] == "controlled_adaptation_candidate"
    assert layer["direct_reuse_allowed"] is False
    assert any(
        check.reason_code == "VALIDATION_NOT_TRANSFERABLE"
        and check.observed_value.get("requested") == 69.9
        for check in candidate.constraint_checks
        if isinstance(check.observed_value, dict)
    )
