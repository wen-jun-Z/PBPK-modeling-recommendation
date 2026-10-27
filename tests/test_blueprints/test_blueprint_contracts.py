from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from pbpk_kb.blueprints.osp_compiler import (
    OSPCompilationError,
    compile_osp_blueprint,
)
from pbpk_kb.blueprints.parameter_resolver import (
    resolve_template_parameters,
)
from pbpk_kb.blueprints.schemas import (
    AbsorptionProcess,
    BlueprintTemplate,
    CalibrationPlan,
    CompartmentDefinition,
    DistributionProcess,
    DosingRegimen,
    EquationTemplate,
    MetabolismProcess,
    ObservedParameter,
    ParameterContext,
    PBPKBlueprint,
    ParameterSlot,
    ProcessParticipant,
    ProvenanceRef,
    RenalExcretionProcess,
    SensitivityUncertaintyPlan,
    SimulationScenario,
    TargetRuntime,
    TemplateReference,
    ValidationPlan,
    advance_blueprint_status,
    build_blueprint,
)
from pbpk_kb.blueprints.template_catalog import (
    TemplateCatalog,
    TemplateCatalogError,
    select_template,
)
from pbpk_kb.blueprints.validators import validate_blueprint
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import LicenseDecision, LicenseStatus
from pbpk_kb.recommendation.requirement_profile import (
    build_requirement_profile,
)
from pbpk_kb.recommendation.schemas import (
    Citation,
    ComponentRecommendation,
    CompoundRef,
    ModelingRequest,
    RecommendationResponse,
)


COMPOUND_ID = UUID("6ef60faa-3f99-559d-bc2a-7d9da4764fce")
PROVENANCE_ID = UUID("2417ec4a-196c-445a-91fd-80a33f754f19")
SOURCE_CHECKSUM = "a" * 64
PROJECT_CHECKSUM = "b" * 64
LICENSE_CHECKSUM = "c" * 64
R_PACKAGE_CHECKSUM = "d" * 64
DOTNET_METADATA_CHECKSUM = "f" * 64
SOURCE_COMMIT = "e" * 40


def _provenance() -> ProvenanceRef:
    return ProvenanceRef(
        provenance_id=PROVENANCE_ID,
        provenance_kind="model_asset",
        source_id="licensed_template",
        source_location="GITHUB_BLOB:commit/template.pkml",
        source_checksum=SOURCE_CHECKSUM,
    )


def _component(name: str, evidence_id: UUID) -> ComponentRecommendation:
    return ComponentRecommendation(
        component=name,
        status="recommended",
        rationale="Grounded test evidence.",
        citations=[
            Citation(
                evidence_id=str(evidence_id),
                source_id="test",
                source_location=f"TEST:{evidence_id}",
                source_text=f"{name} evidence",
                confidence=1.0,
            )
        ],
        evidence_count=1,
        source_count=1,
        confidence=1.0,
    )


def _profile() -> object:
    evidence_ids = {
        name: UUID(value)
        for name, value in {
            "absorption": "0cb5942a-d00a-456c-938c-149302324dd6",
            "distribution": "1bf943af-a975-479c-a14a-8aa60dccd153",
            "metabolism": "929bb870-ae23-4af8-b7cb-a3ef559c59ad",
            "excretion": "0b5f38ce-f91f-464f-a958-597b2e110001",
        }.items()
    }
    compound = CompoundRef(
        compound_id=COMPOUND_ID,
        preferred_name="caffeine",
        active_moiety="caffeine",
        external_identifiers={"pubchem_cid": "2519"},
        identity_resolution_method="database_exact_id",
        identity_confidence=1.0,
    )
    recommendation = RecommendationResponse(
        compound_id=str(COMPOUND_ID),
        compound_name="caffeine",
        objective="base_pk",
        retrieval_policy="test",
        coverage_score=1.0,
        overall_confidence=1.0,
        components=[
            _component(name, evidence_id)
            for name, evidence_id in evidence_ids.items()
        ],
        ranked_subgraphs=[],
        public_model_candidates=[],
        evidence_gaps=[],
        warnings=[],
    )
    return build_requirement_profile(
        request=ModelingRequest(
            compound="caffeine",
            objective="base_pk",
            species="human",
            route="oral",
            dose=250,
            dose_unit="mg",
            frequency="single dose",
        ),
        compound_identity=compound,
        source_evidence_run_id=UUID(
            "4a6f2b30-8c2e-45b7-87b3-e814bb74262a"
        ),
        ontology_version="pbpk-ontology-test-1.0.0",
        component_recommendation=recommendation,
    )


def _target_runtime() -> TargetRuntime:
    return TargetRuntime(
        adapter="ospsuite_r",
        ospsuite_r_version="12.4.3",
        package_sha256=R_PACKAGE_CHECKSUM,
        r_version="4.4.3",
        dotnet_runtime_version="8.0.29",
        dotnet_release_metadata_sha256=DOTNET_METADATA_CHECKSUM,
        platform="linux-x86_64",
    )


def _template(
    *,
    supported_process_types: list[str] | None = None,
) -> BlueprintTemplate:
    process_types = supported_process_types or [
        "absorption",
        "distribution",
        "metabolism",
        "renal_excretion",
    ]
    slots = [
        ParameterSlot(
            slot_id=f"{name}_slot",
            semantic_id=f"parameter:{name}_rate",
            model_path=f"Organism|Caffeine|{name} rate",
            dimension="rate",
            canonical_unit="1/min",
            template_value=float(index),
            source_location=f"PKML:template#parameter={name}",
            source_checksum=SOURCE_CHECKSUM,
            minimum_value=0,
            maximum_value=100,
            compound_specific=True,
        )
        for index, name in enumerate(
            ("absorption", "distribution", "metabolism", "excretion"),
            start=1,
        )
    ]
    return BlueprintTemplate(
        template_id="osp-caffeine-test",
        template_version="1.0.0",
        target_runtime=_target_runtime(),
        source_skeleton_relative_path="template.pkml",
        source_skeleton_sha256=SOURCE_CHECKSUM,
        source_project_relative_path="template.pksim5",
        source_project_sha256=PROJECT_CHECKSUM,
        source_repository="https://example.test/template.git",
        source_tag="v1.0.0",
        source_commit=SOURCE_COMMIT,
        license_id="GPL-2.0-only",
        license_relative_path="LICENSE.pdf",
        license_sha256=LICENSE_CHECKSUM,
        license_decision=LicenseDecision(
            status=LicenseStatus.ALLOWED,
            spdx_id="GPL-2.0-only",
            source_location="LICENSE.pdf",
            source_checksum=LICENSE_CHECKSUM,
            reason_code="REPOSITORY_LICENSE_EXPLICIT",
        ),
        species="human",
        skeleton_simulation_id="test-simulation-id",
        skeleton_simulation_name="test-simulation",
        skeleton_compound_names=["caffeine"],
        skeleton_instantiated_process_count=len(process_types),
        skeleton_process_type_counts={
            process_type: 1 for process_type in process_types
        },
        compound_semantic_ids=[f"compound:{COMPOUND_ID}"],
        model_structure="whole_body",
        supported_objectives=["base_pk"],
        supported_routes=["oral"],
        supported_process_types=process_types,
        required_compartment_ids=[
            "lumen",
            "plasma",
            "tissue",
            "liver",
            "urine",
            "sink",
        ],
        parameter_slots=slots,
        equation_templates=[
            EquationTemplate(
                equation_template_id=f"equation:{name}",
                process_type=process_type,
                equation_dimension="rate",
                source_formula_id=f"formula:{name}",
                source_location=f"PKML:template#formula={name}",
                source_checksum=SOURCE_CHECKSUM,
                mass_balance_validated=True,
                required_parameter_semantic_ids=[
                    f"parameter:{name}_rate"
                ],
            )
            for name, process_type in (
                ("absorption", "absorption"),
                ("distribution", "distribution"),
                ("metabolism", "metabolism"),
                ("excretion", "renal_excretion"),
            )
            if process_type in process_types
        ],
        validation_scenario_ids=["oral-250-mg"],
        validation_status="structural_template_runtime_verified",
    )


def _blueprint(
    profile: object,
    template: BlueprintTemplate,
) -> PBPKBlueprint:
    compound = f"compound:{COMPOUND_ID}"
    metabolite = "compound:caffeine-metabolite"
    participants = lambda output=compound: [
        ProcessParticipant(
            compound_semantic_id=compound,
            role="input",
            stoichiometric_coefficient=1,
        ),
        ProcessParticipant(
            compound_semantic_id=output,
            role="output",
            stoichiometric_coefficient=1,
        ),
    ]
    processes = [
        AbsorptionProcess(
            process_id="absorption",
            name="Oral absorption",
            process_type="absorption",
            input_compartment_id="lumen",
            output_compartment_id="plasma",
            equation_template_id="equation:absorption",
            equation_dimension="rate",
            parameter_ids=[],
            participants=participants(),
            evidence=[_provenance()],
            route="oral",
        ),
        DistributionProcess(
            process_id="distribution",
            name="Tissue distribution",
            process_type="distribution",
            input_compartment_id="plasma",
            output_compartment_id="tissue",
            equation_template_id="equation:distribution",
            equation_dimension="rate",
            parameter_ids=[],
            participants=participants(),
            evidence=[_provenance()],
            distribution_method="whole_body_partitioning",
        ),
        MetabolismProcess(
            process_id="metabolism",
            name="CYP1A2 metabolism",
            process_type="metabolism",
            input_compartment_id="liver",
            output_compartment_id="sink",
            equation_template_id="equation:metabolism",
            equation_dimension="rate",
            parameter_ids=[],
            participants=participants(metabolite),
            evidence=[_provenance()],
            enzyme_semantic_id="enzyme:CYP1A2",
            parent_compound_semantic_id=compound,
            metabolite_semantic_id=metabolite,
        ),
        RenalExcretionProcess(
            process_id="excretion",
            name="Renal excretion",
            process_type="renal_excretion",
            input_compartment_id="plasma",
            output_compartment_id="urine",
            equation_template_id="equation:excretion",
            equation_dimension="rate",
            parameter_ids=[],
            participants=participants(),
            evidence=[_provenance()],
            renal_mechanism="combined",
        ),
    ]
    process_ids_by_semantic = {
        "parameter:absorption_rate": ["absorption"],
        "parameter:distribution_rate": ["distribution"],
        "parameter:metabolism_rate": ["metabolism"],
        "parameter:excretion_rate": ["excretion"],
    }
    resolved_parameters = {
        f"Organism|Caffeine|{name} rate": ObservedParameter(
            parameter_id=f"resolved-{name}-rate",
            semantic_id=f"parameter:{name}_rate",
            name=f"{name} rate",
            model_path=f"Organism|Caffeine|{name} rate",
            dimension="rate",
            canonical_unit="1/min",
            context=ParameterContext(
                compound_semantic_id=compound,
                species="human",
            ),
            used_by_process_ids=[name],
            provenance=[_provenance()],
            blocking=True,
            origin="observed",
            value=float(index),
            evidence_id=PROVENANCE_ID,
        )
        for index, name in enumerate(
            ("absorption", "distribution", "metabolism", "excretion"),
            start=1,
        )
    }
    parameters = resolve_template_parameters(
        template=template,
        compound_semantic_id=compound,
        species="human",
        process_ids_by_semantic_id=process_ids_by_semantic,
        resolved_parameters=resolved_parameters,
    )
    parameter_id_by_semantic = {
        item.semantic_id: item.parameter_id for item in parameters
    }
    for process, semantic in zip(
        processes,
        process_ids_by_semantic,
        strict=True,
    ):
        process.parameter_ids.append(
            parameter_id_by_semantic[semantic]
        )
    compartments = [
        CompartmentDefinition(
            compartment_id=name,
            semantic_id=f"compartment:{name}",
            name=name,
            compartment_type=(
                "lumen"
                if name == "lumen"
                else "urine"
                if name == "urine"
                else "sink"
                if name == "sink"
                else "tissue"
            ),
            provenance=[_provenance()],
        )
        for name in template.required_compartment_ids
    ]
    payload = {
        "blueprint_id": UUID(
            "f55dfb03-45a9-4132-aac2-8b848e40ffdb"
        ),
        "blueprint_version": "1.0.0",
        "profile_id": profile.profile_id,
        "profile_checksum": profile.checksum,
        "compound_semantic_id": compound,
        "compound_name": "caffeine",
        "species": "human",
        "objective": "base_pk",
        "template": TemplateReference(
            template_id=template.template_id,
            template_version=template.template_version,
            skeleton_relative_path=(
                template.source_skeleton_relative_path
            ),
            skeleton_sha256=template.source_skeleton_sha256,
            license_id=template.license_id,
            license_sha256=template.license_sha256,
            license_decision=template.license_decision,
            source_repository=template.source_repository,
            source_tag=template.source_tag,
            source_commit=template.source_commit,
            validation_status=template.validation_status,
        ),
        "compartments": compartments,
        "processes": processes,
        "parameters": parameters,
        "dosing_regimens": [
            DosingRegimen(
                regimen_id="oral-250-mg",
                route="oral",
                dose=250,
                dose_unit="mg",
                frequency="single dose",
                evidence=[_provenance()],
            )
        ],
        "simulation_scenarios": [
            SimulationScenario(
                scenario_id="oral-250-mg",
                name="Caffeine oral 250 mg",
                regimen_id="oral-250-mg",
                output_paths=[
                    "Organism|VenousBlood|Caffeine|Concentration"
                ],
                start_time=0,
                end_time=1440,
                time_unit="min",
            )
        ],
        "calibration_plan": CalibrationPlan(),
        "validation_plan": ValidationPlan(),
        "sensitivity_uncertainty_plan": (
            SensitivityUncertaintyPlan(enabled=False)
        ),
        "target_runtime": template.target_runtime,
        "satisfied_requirement_ids": [
            item.requirement_id
            for item in profile.requirements
            if item.necessity == "hard_required"
        ],
        "unresolved_requirement_ids": [],
        "unresolved_blockers": [],
        "status": "draft_grounded",
        "created_at": datetime(2026, 7, 31, tzinfo=UTC),
    }
    return build_blueprint(payload)


def test_blueprint_validates_and_advances_one_status_at_a_time() -> None:
    profile = _profile()
    template = _template()
    blueprint = _blueprint(profile, template)

    report = validate_blueprint(
        blueprint=blueprint,
        template=template,
        profile=profile,
    )

    assert report.accepted
    assert report.compile_eligible
    structural = advance_blueprint_status(
        blueprint,
        status="structurally_validated",
        created_at=datetime(2026, 7, 31, 1, tzinfo=UTC),
    )
    assert structural.parent_blueprint_checksum == blueprint.checksum
    assert structural.blueprint_version == "1.0.1"

    with pytest.raises(ValueError, match="Invalid blueprint status"):
        advance_blueprint_status(
            structural,
            status="executable",
            created_at=datetime(2026, 7, 31, 2, tzinfo=UTC),
        )


def test_template_selection_rejects_missing_hard_process() -> None:
    profile = _profile()
    template = _template(
        supported_process_types=[
            "absorption",
            "distribution",
            "renal_excretion",
        ]
    )

    with pytest.raises(
        TemplateCatalogError,
        match="NO_EXECUTABLE_TEMPLATE",
    ):
        select_template(
            catalog=TemplateCatalog(
                catalog_schema_version="1.0.0",
                templates=[template],
            ),
            profile=profile,
        )


def test_blueprint_checksum_tampering_is_rejected() -> None:
    profile = _profile()
    blueprint = _blueprint(profile, _template())
    payload = blueprint.model_dump(mode="json")
    payload["compound_name"] = "not-caffeine"

    with pytest.raises(ValidationError, match="checksum mismatch"):
        PBPKBlueprint.model_validate(payload)


def test_compiler_rejects_unvalidated_draft(tmp_path: Path) -> None:
    profile = _profile()
    template = _template()
    blueprint = _blueprint(profile, template)
    report = validate_blueprint(
        blueprint=blueprint,
        template=template,
        profile=profile,
    )

    with pytest.raises(
        OSPCompilationError,
        match="structurally_validated",
    ):
        compile_osp_blueprint(
            blueprint=blueprint,
            validation_report=report,
            template=template,
            asset_root=tmp_path,
            output_directory=tmp_path / "compile",
            r_executable=tmp_path / "R",
            r_library=tmp_path / "library",
            ospsuite_package_archive=tmp_path / "ospsuite.tar.gz",
            dotnet_root=tmp_path / "dotnet",
            dotnet_release_metadata_path=(
                tmp_path / "dotnet-releases.json"
            ),
        )
