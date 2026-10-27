from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pbpk_kb.blueprints.parameter_resolver import (
    resolve_template_parameters,
)
from pbpk_kb.blueprints.schemas import (
    AbsorptionProcess,
    BiliaryExcretionProcess,
    BlueprintTemplate,
    CalibrationPlan,
    CompartmentDefinition,
    DistributionProcess,
    DosingRegimen,
    EquationTemplate,
    MetabolismProcess,
    PBPKBlueprint,
    ParameterDefinition,
    ProcessDefinition,
    ProcessParticipant,
    ProvenanceRef,
    RenalExcretionProcess,
    SensitivityUncertaintyPlan,
    SimulationScenario,
    TemplateReference,
    TransportProcess,
    ValidationPlan,
    build_blueprint,
)
from pbpk_kb.blueprints.template_catalog import (
    audit_template_skeleton,
    instantiated_processes,
    resolve_template_asset_path,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    OSPPKMLComponentRecord,
    OSPPKMLProcessParticipant,
    OSPPKMLProcessRecord,
)
from pbpk_kb.model_assets.osp_pkml_parser import OSPPKMLParsedRecords
from pbpk_kb.recommendation.schemas import RequirementProfile


BLUEPRINT_BUILDER_VERSION = "osp-template-blueprint-builder-1.0.0"
BLUEPRINT_NAMESPACE = uuid.UUID("6d5e529a-8e51-4c28-889e-4057be313917")
PROVENANCE_NAMESPACE = uuid.UUID("ae4c4cf5-a109-401c-89c2-70172dbd6adc")


class BlueprintBuildError(RuntimeError):
    pass


def _normalized_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or uuid.uuid5(BLUEPRINT_NAMESPACE, value).hex


def _provenance(
    *,
    source_id: str,
    source_location: str,
    source_checksum: str,
) -> ProvenanceRef:
    return ProvenanceRef(
        provenance_id=uuid.uuid5(
            PROVENANCE_NAMESPACE,
            f"{source_id}\n{source_location}\n{source_checksum}",
        ),
        provenance_kind="model_asset",
        source_id=source_id,
        source_location=source_location,
        source_checksum=source_checksum,
    )


def _compartment_type(name: str) -> str:
    normalized = name.casefold()
    if normalized in {"model", "organism"}:
        return "other"
    if "urine" in normalized:
        return "urine"
    if any(value in normalized for value in ("bile", "gallbladder")):
        return "bile"
    if any(
        value in normalized
        for value in (
            "lumen",
            "stomach",
            "duodenum",
            "jejunum",
            "ileum",
            "caecum",
            "colon",
            "rectum",
            "feces",
        )
    ):
        return "lumen"
    if any(
        value in normalized
        for value in (
            "plasma",
            "blood",
            "artery",
            "vein",
            "vascular",
        )
    ):
        return "vascular"
    if normalized in {
        "liver",
        "kidney",
        "lung",
        "heart",
        "brain",
        "muscle",
        "skin",
        "bone",
        "pancreas",
        "spleen",
        "gonads",
        "fat",
    }:
        return "organ"
    return "tissue"


def _molecule_semantic_id(
    participant: OSPPKMLProcessParticipant,
    *,
    compound_names: set[str],
    compound_semantic_id: str,
) -> str:
    name = participant.name or participant.identifier
    if name.casefold() in compound_names:
        return compound_semantic_id
    prefix = {
        "compound": "compound",
        "metabolite": "metabolite",
        "enzyme": "enzyme",
        "transporter": "transporter",
        "protein": "protein",
        "molecular_complex": "complex",
    }.get(participant.component_type or "", "molecule")
    return f"{prefix}:{_normalized_identifier(name)}"


def _participants(
    process: OSPPKMLProcessRecord,
    *,
    compound_names: set[str],
    compound_semantic_id: str,
) -> list[ProcessParticipant]:
    participants: list[ProcessParticipant] = []
    for role, values in (
        ("input", process.educt_participants),
        ("output", process.product_participants),
        ("modifier", process.modifier_participants),
    ):
        for value in values:
            participants.append(
                ProcessParticipant(
                    compound_semantic_id=_molecule_semantic_id(
                        value,
                        compound_names=compound_names,
                        compound_semantic_id=compound_semantic_id,
                    ),
                    role=role,
                    stoichiometric_coefficient=(
                        value.stoichiometric_coefficient or 1.0
                    ),
                )
            )
    if not any(item.role == "input" for item in participants):
        raise BlueprintBuildError(
            f"Process has no input participant: {process.process_id}"
        )
    if not any(item.role == "output" for item in participants):
        raise BlueprintBuildError(
            f"Process has no output participant: {process.process_id}"
        )
    return participants


def _process_evidence(process: OSPPKMLProcessRecord) -> list[ProvenanceRef]:
    return [
        _provenance(
            source_id="osp_template_skeleton",
            source_location=process.source.source_location,
            source_checksum=process.source.sha256,
        )
    ]


def _equation_template_id(process: OSPPKMLProcessRecord) -> str:
    if process.formula_id is None:
        raise BlueprintBuildError(
            f"Process has no OSP formula ID: {process.process_id}"
        )
    return f"osp-formula:{process.formula_id}"


def derive_equation_templates(
    records: OSPPKMLParsedRecords,
) -> list[EquationTemplate]:
    equations: list[EquationTemplate] = []
    seen: set[str] = set()
    for process in instantiated_processes(records):
        if process.process_type == "other":
            raise BlueprintBuildError(
                f"Unsupported OSP process type: {process.process_id}"
            )
        equation_id = _equation_template_id(process)
        if equation_id in seen:
            raise BlueprintBuildError(
                f"OSP formula is shared by multiple instantiated processes: "
                f"{process.formula_id}"
            )
        seen.add(equation_id)
        equations.append(
            EquationTemplate(
                equation_template_id=equation_id,
                process_type=process.process_type,
                equation_dimension="rate",
                parameterization="locked_template_formula",
                source_formula_id=process.formula_id or "",
                source_location=(
                    process.source.source_location
                    + f"&formula_id={process.formula_id}"
                ),
                source_checksum=process.source.sha256,
                mass_balance_validated=True,
                required_parameter_semantic_ids=[],
            )
        )
    return equations


def _required_container_ids(
    *,
    records: OSPPKMLParsedRecords,
    processes: list[OSPPKMLProcessRecord],
) -> set[str]:
    containers = {
        component.component_id: component
        for component in records.components
        if component.element_type == "Container"
    }
    required = {
        compartment_id
        for process in processes
        for compartment_id in (
            process.input_compartment_id,
            process.output_compartment_id,
        )
        if compartment_id is not None
    }
    if required - containers.keys():
        raise BlueprintBuildError(
            "An instantiated process references an unknown container."
        )
    pending = list(required)
    while pending:
        component = containers[pending.pop()]
        parent_id = component.parent_container_id
        if parent_id is None or parent_id in required:
            continue
        if parent_id not in containers:
            raise BlueprintBuildError(
                f"Container parent is unresolved: {parent_id}"
            )
        required.add(parent_id)
        pending.append(parent_id)
    return required


def derive_compartments(
    records: OSPPKMLParsedRecords,
) -> list[CompartmentDefinition]:
    processes = instantiated_processes(records)
    required = _required_container_ids(
        records=records,
        processes=processes,
    )
    containers: dict[str, OSPPKMLComponentRecord] = {
        component.component_id: component
        for component in records.components
        if component.element_type == "Container"
        and component.component_id in required
    }
    return [
        CompartmentDefinition(
            compartment_id=component.component_id,
            semantic_id=f"osp-container:{component.component_id}",
            name=component.name,
            compartment_type=_compartment_type(component.name),
            parent_compartment_id=(
                component.parent_container_id
                if component.parent_container_id in required
                else None
            ),
            provenance=[
                _provenance(
                    source_id="osp_template_skeleton",
                    source_location=component.source.source_location,
                    source_checksum=component.source.sha256,
                )
            ],
        )
        for component in sorted(
            containers.values(),
            key=lambda item: item.component_id,
        )
    ]


def _transport_mechanism(process: OSPPKMLProcessRecord) -> str:
    normalized = process.name.casefold()
    if process.modifier_participants:
        return "carrier_mediated"
    if "passive" in normalized:
        return "passive"
    if "masstransfer" in normalized or "mass transfer" in normalized:
        return "bulk_flow"
    return "other"


def derive_processes(
    *,
    records: OSPPKMLParsedRecords,
    compound_semantic_id: str,
    compound_names: set[str],
    route: str,
) -> list[ProcessDefinition]:
    result: list[ProcessDefinition] = []
    for process in instantiated_processes(records):
        if (
            process.input_compartment_id is None
            or process.output_compartment_id is None
        ):
            raise BlueprintBuildError(
                f"Process compartment is unresolved: {process.process_id}"
            )
        participants = _participants(
            process,
            compound_names=compound_names,
            compound_semantic_id=compound_semantic_id,
        )
        common = {
            "process_id": process.process_id,
            "name": process.name,
            "process_type": process.process_type,
            "input_compartment_id": process.input_compartment_id,
            "output_compartment_id": process.output_compartment_id,
            "equation_template_id": _equation_template_id(process),
            "equation_dimension": "rate",
            "parameterization": "locked_template_formula",
            "parameter_ids": [],
            "participants": participants,
            "evidence": _process_evidence(process),
        }
        if process.process_type == "absorption":
            item = AbsorptionProcess(**common, route=route)
        elif process.process_type == "distribution":
            item = DistributionProcess(
                **common,
                distribution_method="osp_passive_diffusion",
            )
        elif process.process_type == "metabolism":
            inputs = [
                item.compound_semantic_id
                for item in participants
                if item.role == "input"
            ]
            outputs = [
                item.compound_semantic_id
                for item in participants
                if item.role == "output"
            ]
            modifiers = [
                item.compound_semantic_id
                for item in participants
                if item.role == "modifier"
            ]
            if not inputs or not outputs or not modifiers:
                raise BlueprintBuildError(
                    f"Metabolism topology is incomplete: {process.process_id}"
                )
            item = MetabolismProcess(
                **common,
                enzyme_semantic_id=modifiers[0],
                parent_compound_semantic_id=inputs[0],
                metabolite_semantic_id=outputs[0],
            )
        elif process.process_type == "transport":
            mechanism = _transport_mechanism(process)
            transporter = next(
                (
                    item.compound_semantic_id
                    for item in participants
                    if item.role == "modifier"
                ),
                None,
            )
            item = TransportProcess(
                **common,
                transport_mechanism=mechanism,
                transporter_semantic_id=transporter,
                direction="influx",
            )
        elif process.process_type == "renal_excretion":
            item = RenalExcretionProcess(
                **common,
                renal_mechanism="combined",
            )
        elif process.process_type == "biliary_excretion":
            item = BiliaryExcretionProcess(
                **common,
                biliary_mechanism="combined",
            )
        else:
            raise BlueprintBuildError(
                f"Unsupported OSP process type: {process.process_type}"
            )
        result.append(item)
    return result


def _hard_objective(profile: RequirementProfile) -> str:
    values = {
        str(item.expected_value)
        for item in profile.requirements
        if item.category == "validation_context"
        and item.necessity == "hard_required"
    }
    if len(values) != 1:
        raise BlueprintBuildError(
            "Requirement profile must define exactly one hard objective."
        )
    return next(iter(values))


def build_blueprint_from_template(
    *,
    profile: RequirementProfile,
    template: BlueprintTemplate,
    records: OSPPKMLParsedRecords,
    asset_root: Path,
    dose: float,
    dose_unit: str,
    route: str,
    formulation: str | None,
    frequency: str | None,
    output_paths: list[str],
    start_time: float,
    end_time: float,
    time_unit: str,
    created_at: datetime | None = None,
    resolved_parameters: dict[str, ParameterDefinition] | None = None,
    calibration_plan: CalibrationPlan | None = None,
    validation_plan: ValidationPlan | None = None,
) -> PBPKBlueprint:
    if profile.ranking_eligibility != "ready":
        raise BlueprintBuildError(
            "Blocked requirement profiles cannot generate blueprints."
        )
    skeleton_path = resolve_template_asset_path(
        asset_root,
        template.source_skeleton_relative_path,
    )
    if sha256_file(skeleton_path) != template.source_skeleton_sha256:
        raise BlueprintBuildError("Template skeleton checksum drift.")
    template_audit = audit_template_skeleton(
        template=template,
        asset_root=asset_root,
    )
    if not template_audit.accepted:
        raise BlueprintBuildError(
            "Template skeleton audit failed: "
            + ",".join(template_audit.errors)
        )
    compound_semantic_id = (
        f"compound:{profile.compound_identity.compound_id}"
    )
    if compound_semantic_id not in template.compound_semantic_ids:
        raise BlueprintBuildError("Template compound identity mismatch.")
    objective = _hard_objective(profile)
    if objective.casefold() not in set(
        map(str.casefold, template.supported_objectives)
    ):
        raise BlueprintBuildError("Template objective mismatch.")
    if route.casefold() not in set(
        map(str.casefold, template.supported_routes)
    ):
        raise BlueprintBuildError("Template route mismatch.")
    actual_equations = derive_equation_templates(records)
    def equation_identity(item: EquationTemplate) -> tuple[object, ...]:
        return (
            item.equation_template_id,
            item.process_type,
            item.equation_dimension,
            item.parameterization,
            item.source_formula_id,
            item.source_checksum,
            item.mass_balance_validated,
            tuple(item.required_parameter_semantic_ids),
        )

    if {
        equation_identity(item) for item in actual_equations
    } != {
        equation_identity(item) for item in template.equation_templates
    }:
        raise BlueprintBuildError(
            "Template equation inventory differs from the locked PKML."
        )
    compound_names = {
        item.casefold() for item in template.skeleton_compound_names
    }
    processes = derive_processes(
        records=records,
        compound_semantic_id=compound_semantic_id,
        compound_names=compound_names,
        route=route,
    )
    compartments = derive_compartments(records)
    if {
        item.compartment_id for item in compartments
    } != set(template.required_compartment_ids):
        raise BlueprintBuildError(
            "Template compartment inventory differs from the locked PKML."
        )
    parameters = resolve_template_parameters(
        template=template,
        compound_semantic_id=compound_semantic_id,
        species=template.species,
        resolved_parameters=resolved_parameters,
    )
    template_provenance = _provenance(
        source_id="osp_template_skeleton",
        source_location=(
            "PKML:"
            + template.source_skeleton_relative_path
            + f"#simulation={template.skeleton_simulation_name}"
        ),
        source_checksum=template.source_skeleton_sha256,
    )
    hard_requirement_ids = [
        item.requirement_id
        for item in profile.requirements
        if item.necessity == "hard_required"
    ]
    unresolved_requirement_ids = [
        item.requirement_id
        for item in profile.requirements
        if item.necessity != "hard_required"
    ]
    created_at = created_at or datetime.now(UTC)
    blueprint_id = uuid.uuid5(
        BLUEPRINT_NAMESPACE,
        (
            f"{profile.checksum}\n{template.template_id}\n"
            f"{template.template_version}\n"
            f"{template.source_skeleton_sha256}"
        ),
    )
    payload = {
        "blueprint_id": blueprint_id,
        "blueprint_version": "1.0.0",
        "parent_blueprint_checksum": None,
        "profile_id": profile.profile_id,
        "profile_checksum": profile.checksum,
        "compound_semantic_id": compound_semantic_id,
        "compound_name": profile.compound_identity.preferred_name,
        "species": template.species,
        "population": (
            records.summary.populations[0]
            if records.summary.populations
            else None
        ),
        "objective": objective,
        "template": TemplateReference(
            template_id=template.template_id,
            template_version=template.template_version,
            skeleton_relative_path=template.source_skeleton_relative_path,
            skeleton_sha256=template.source_skeleton_sha256,
            license_id=template.license_id,
            license_sha256=template.license_sha256,
            license_decision=template.license_decision,
            source_repository=template.source_repository,
            source_tag=template.source_tag,
            source_commit=template.source_commit,
            source_manifest_sha256=template.source_manifest_sha256,
            validation_status=template.validation_status,
        ),
        "compartments": compartments,
        "processes": processes,
        "parameters": parameters,
        "dosing_regimens": [
            DosingRegimen(
                regimen_id="template-regimen",
                route=route,
                dose=dose,
                dose_unit=dose_unit,
                formulation=formulation,
                frequency=frequency,
                evidence=[template_provenance],
            )
        ],
        "simulation_scenarios": [
            SimulationScenario(
                scenario_id=template.skeleton_simulation_id,
                name=template.skeleton_simulation_name,
                regimen_id="template-regimen",
                output_paths=output_paths,
                start_time=start_time,
                end_time=end_time,
                time_unit=time_unit,
            )
        ],
        "calibration_plan": calibration_plan or CalibrationPlan(),
        "validation_plan": validation_plan or ValidationPlan(),
        "sensitivity_uncertainty_plan": SensitivityUncertaintyPlan(
            enabled=False,
        ),
        "target_runtime": template.target_runtime,
        "satisfied_requirement_ids": hard_requirement_ids,
        "unresolved_requirement_ids": unresolved_requirement_ids,
        "unresolved_blockers": [],
        "status": "draft_grounded",
        "created_at": created_at,
    }
    return build_blueprint(payload)
