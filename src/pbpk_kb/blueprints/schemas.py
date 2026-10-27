from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from pbpk_kb.model_assets.models import LicenseDecision


BlueprintStatus = Literal[
    "draft_grounded",
    "structurally_validated",
    "compiled",
    "executable",
    "calibrated",
    "externally_validated",
    "quarantined",
]
ParameterDimension = Literal[
    "dimensionless",
    "amount",
    "mass",
    "concentration",
    "volume",
    "time",
    "flow",
    "clearance",
    "rate",
    "fraction",
    "molecular_weight",
    "permeability",
    "unknown",
]


class StrictBlueprintModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProvenanceRef(StrictBlueprintModel):
    provenance_id: UUID
    provenance_kind: Literal[
        "validated_evidence",
        "model_asset",
        "ontology_rule",
        "calibration_run",
        "user_request",
    ]
    source_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    source_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class ParameterContext(StrictBlueprintModel):
    compound_semantic_id: str | None = None
    species: str
    population: str | None = None
    compartment_id: str | None = None
    route: str | None = None
    formulation: str | None = None


class ParameterBase(StrictBlueprintModel):
    parameter_id: str = Field(min_length=1)
    semantic_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    dimension: ParameterDimension
    canonical_unit: str | None = None
    context: ParameterContext
    used_by_process_ids: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRef] = Field(min_length=1)
    blocking: bool

    @field_validator("used_by_process_ids")
    @classmethod
    def process_ids_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Parameter process references must be unique.")
        return value


class ObservedParameter(ParameterBase):
    origin: Literal["observed"]
    value: float
    canonical_unit: str = Field(min_length=1)
    evidence_id: UUID
    uncertainty: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def evidence_is_present(self) -> "ObservedParameter":
        if self.evidence_id not in {
            item.provenance_id for item in self.provenance
        }:
            raise ValueError(
                "Observed parameter evidence_id must resolve in provenance."
            )
        return self


class DerivedParameter(ParameterBase):
    origin: Literal["derived"]
    value: float | None = None
    canonical_unit: str = Field(min_length=1)
    derivation_expression: str = Field(min_length=1)
    expression_version: str = Field(min_length=1)
    input_parameter_ids: list[str] = Field(default_factory=list)
    uncertainty: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def derivation_is_traceable(self) -> "DerivedParameter":
        if not self.input_parameter_ids and not self.derivation_expression.startswith(
            "inherit_template_value:"
        ):
            raise ValueError(
                "Derived parameters require inputs unless they inherit a "
                "checksum-locked template value."
            )
        if (
            self.derivation_expression.startswith("inherit_template_value:")
            and not any(
                item.provenance_kind == "model_asset"
                for item in self.provenance
            )
        ):
            raise ValueError(
                "Template-inherited values require model-asset provenance."
            )
        return self


class EstimatedParameter(ParameterBase):
    origin: Literal["estimated"]
    initial_value: float
    value: float | None = None
    canonical_unit: str = Field(min_length=1)
    lower_bound: float
    upper_bound: float
    calibration_dataset_id: UUID
    objective_function: str = Field(min_length=1)
    calibration_run_id: UUID | None = None
    uncertainty: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def estimate_is_bounded(self) -> "EstimatedParameter":
        if self.lower_bound > self.upper_bound:
            raise ValueError("Estimated parameter bounds are reversed.")
        if not self.lower_bound <= self.initial_value <= self.upper_bound:
            raise ValueError(
                "Estimated parameter initial value is outside locked bounds."
            )
        if self.value is not None and not (
            self.lower_bound <= self.value <= self.upper_bound
        ):
            raise ValueError(
                "Estimated parameter value is outside locked bounds."
            )
        if self.value is not None and self.calibration_run_id is None:
            raise ValueError(
                "A fitted value requires a calibration_run_id."
            )
        return self


class MissingParameter(ParameterBase):
    origin: Literal["missing"]
    value: Literal[None] = None
    canonical_unit: str | None = None
    missing_reason_code: str = Field(min_length=1)


ParameterDefinition = Annotated[
    ObservedParameter
    | DerivedParameter
    | EstimatedParameter
    | MissingParameter,
    Field(discriminator="origin"),
]


class CompartmentDefinition(StrictBlueprintModel):
    compartment_id: str = Field(min_length=1)
    semantic_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    compartment_type: Literal[
        "organ",
        "tissue",
        "vascular",
        "lumen",
        "urine",
        "bile",
        "sink",
        "other",
    ]
    parent_compartment_id: str | None = None
    provenance: list[ProvenanceRef] = Field(min_length=1)


class ProcessParticipant(StrictBlueprintModel):
    compound_semantic_id: str = Field(min_length=1)
    role: Literal["input", "output", "modifier"]
    stoichiometric_coefficient: float = Field(gt=0)


class ProcessBase(StrictBlueprintModel):
    process_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    input_compartment_id: str
    output_compartment_id: str
    equation_template_id: str = Field(min_length=1)
    equation_dimension: ParameterDimension
    parameterization: Literal[
        "explicit_blueprint_parameters",
        "locked_template_formula",
    ] = "explicit_blueprint_parameters"
    parameter_ids: list[str] = Field(default_factory=list)
    participants: list[ProcessParticipant] = Field(min_length=1)
    evidence: list[ProvenanceRef] = Field(min_length=1)

    @field_validator("parameter_ids")
    @classmethod
    def parameter_ids_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Process parameter IDs must be unique.")
        return value

class AbsorptionProcess(ProcessBase):
    process_type: Literal["absorption"]
    route: str = Field(min_length=1)


class DistributionProcess(ProcessBase):
    process_type: Literal["distribution"]
    distribution_method: str = Field(min_length=1)


class MetabolismProcess(ProcessBase):
    process_type: Literal["metabolism"]
    enzyme_semantic_id: str = Field(min_length=1)
    parent_compound_semantic_id: str = Field(min_length=1)
    metabolite_semantic_id: str | None = None


class TransportProcess(ProcessBase):
    process_type: Literal["transport"]
    transport_mechanism: Literal[
        "passive",
        "carrier_mediated",
        "bulk_flow",
        "other",
    ]
    transporter_semantic_id: str | None = None
    direction: Literal["influx", "efflux"]

    @model_validator(mode="after")
    def carrier_transport_has_transporter(self) -> "TransportProcess":
        if (
            self.transport_mechanism == "carrier_mediated"
            and not self.transporter_semantic_id
        ):
            raise ValueError(
                "Carrier-mediated transport requires a transporter semantic ID."
            )
        return self


class RenalExcretionProcess(ProcessBase):
    process_type: Literal["renal_excretion"]
    renal_mechanism: Literal[
        "filtration",
        "secretion",
        "reabsorption",
        "combined",
    ]


class BiliaryExcretionProcess(ProcessBase):
    process_type: Literal["biliary_excretion"]
    biliary_mechanism: Literal["passive", "active", "combined"]


class DDIProcess(ProcessBase):
    process_type: Literal["ddi"]
    victim_compound_semantic_id: str = Field(min_length=1)
    perpetrator_compound_semantic_id: str = Field(min_length=1)
    target_semantic_id: str = Field(min_length=1)
    ddi_mechanism: Literal[
        "competitive_inhibition",
        "mechanism_based_inhibition",
        "induction",
        "transporter_inhibition",
    ]


ProcessDefinition = Annotated[
    AbsorptionProcess
    | DistributionProcess
    | MetabolismProcess
    | TransportProcess
    | RenalExcretionProcess
    | BiliaryExcretionProcess
    | DDIProcess,
    Field(discriminator="process_type"),
]


class DosingRegimen(StrictBlueprintModel):
    regimen_id: str = Field(min_length=1)
    route: str = Field(min_length=1)
    dose: float = Field(gt=0)
    dose_unit: str = Field(min_length=1)
    formulation: str | None = None
    frequency: str | None = None
    evidence: list[ProvenanceRef] = Field(min_length=1)


class SimulationScenario(StrictBlueprintModel):
    scenario_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    regimen_id: str
    output_paths: list[str] = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    time_unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def end_after_start(self) -> "SimulationScenario":
        if self.end_time <= self.start_time:
            raise ValueError("Simulation end time must follow start time.")
        return self


class DatasetRef(StrictBlueprintModel):
    dataset_id: UUID
    study_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject_or_cohort_ids: list[str] = Field(min_length=1)
    split: Literal["calibration", "validation", "test"]


class CalibrationPlan(StrictBlueprintModel):
    datasets: list[DatasetRef] = Field(default_factory=list)
    estimated_parameter_ids: list[str] = Field(default_factory=list)
    objective_function: str | None = None


class ValidationPlan(StrictBlueprintModel):
    datasets: list[DatasetRef] = Field(default_factory=list)
    acceptance_criteria_id: str | None = None
    endpoints: list[str] = Field(default_factory=list)


class SensitivityUncertaintyPlan(StrictBlueprintModel):
    enabled: bool
    parameter_ids: list[str] = Field(default_factory=list)
    method: str | None = None
    sample_count: int | None = Field(default=None, ge=1)


class TargetRuntime(StrictBlueprintModel):
    adapter: Literal["ospsuite_r", "pksim_cli_snapshot"]
    ospsuite_r_version: str | None = None
    package_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    r_version: str | None = None
    dotnet_runtime_version: str | None = None
    dotnet_release_metadata_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    pksim_cli_version: str | None = None
    pksim_cli_executable_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    platform: Literal["linux-x86_64", "windows-x86_64"]

    @model_validator(mode="after")
    def runtime_fields_match_adapter(self) -> "TargetRuntime":
        if self.adapter == "ospsuite_r":
            required = (
                self.ospsuite_r_version,
                self.package_sha256,
                self.r_version,
                self.dotnet_runtime_version,
                self.dotnet_release_metadata_sha256,
            )
            if self.platform != "linux-x86_64" or not all(required):
                raise ValueError(
                    "OSPSuite-R requires the locked Linux R and .NET runtime."
                )
        else:
            required = (
                self.pksim_cli_version,
                self.pksim_cli_executable_sha256,
            )
            if self.platform != "windows-x86_64" or not all(required):
                raise ValueError(
                    "PK-Sim snapshot compilation requires the locked Windows CLI."
                )
        return self


class TemplateReference(StrictBlueprintModel):
    template_id: str = Field(min_length=1)
    template_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    skeleton_relative_path: str = Field(min_length=1)
    skeleton_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_id: str = Field(min_length=1)
    license_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_decision: LicenseDecision
    source_repository: str = Field(min_length=1)
    source_tag: str = Field(min_length=1)
    source_commit: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
    )
    source_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    validation_status: Literal[
        "structural_template_runtime_verified"
    ]

    @model_validator(mode="after")
    def source_revision_is_locked(self) -> "TemplateReference":
        if self.source_commit is None and self.source_manifest_sha256 is None:
            raise ValueError(
                "Template source requires a Git commit or manifest checksum."
            )
        return self


def blueprint_checksum(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "checksum"}
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


class PBPKBlueprint(StrictBlueprintModel):
    blueprint_id: UUID
    blueprint_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    parent_blueprint_checksum: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    profile_id: UUID
    profile_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    compound_semantic_id: str = Field(min_length=1)
    compound_name: str = Field(min_length=1)
    species: str = Field(min_length=1)
    population: str | None = None
    objective: str = Field(min_length=1)
    template: TemplateReference
    compartments: list[CompartmentDefinition] = Field(min_length=1)
    processes: list[ProcessDefinition] = Field(min_length=1)
    parameters: list[ParameterDefinition] = Field(min_length=1)
    dosing_regimens: list[DosingRegimen] = Field(min_length=1)
    simulation_scenarios: list[SimulationScenario] = Field(min_length=1)
    calibration_plan: CalibrationPlan
    validation_plan: ValidationPlan
    sensitivity_uncertainty_plan: SensitivityUncertaintyPlan
    target_runtime: TargetRuntime
    satisfied_requirement_ids: list[UUID] = Field(min_length=1)
    unresolved_requirement_ids: list[UUID] = Field(default_factory=list)
    unresolved_blockers: list[str] = Field(default_factory=list)
    status: BlueprintStatus
    created_at: datetime
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("created_at")
    @classmethod
    def created_at_timezone_aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("Blueprint created_at must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_internal_references(self) -> "PBPKBlueprint":
        identifiers = {
            "compartment": [
                item.compartment_id for item in self.compartments
            ],
            "process": [item.process_id for item in self.processes],
            "parameter": [
                item.parameter_id for item in self.parameters
            ],
            "regimen": [
                item.regimen_id for item in self.dosing_regimens
            ],
            "scenario": [
                item.scenario_id for item in self.simulation_scenarios
            ],
        }
        for kind, values in identifiers.items():
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {kind} identifiers.")
        if set(self.satisfied_requirement_ids) & set(
            self.unresolved_requirement_ids
        ):
            raise ValueError(
                "A requirement cannot be satisfied and unresolved."
            )
        if len(self.satisfied_requirement_ids) != len(
            set(self.satisfied_requirement_ids)
        ) or len(self.unresolved_requirement_ids) != len(
            set(self.unresolved_requirement_ids)
        ):
            raise ValueError("Requirement references must be unique.")
        parameter_ids = set(identifiers["parameter"])
        process_ids = set(identifiers["process"])
        compartment_ids = set(identifiers["compartment"])
        regimen_ids = set(identifiers["regimen"])
        for parameter in self.parameters:
            if not set(parameter.used_by_process_ids).issubset(
                process_ids
            ):
                raise ValueError(
                    "Parameter references an unknown process."
                )
        for process in self.processes:
            if not set(process.parameter_ids).issubset(parameter_ids):
                raise ValueError("Process references an unknown parameter.")
            if (
                process.parameterization
                == "explicit_blueprint_parameters"
                and not process.parameter_ids
            ):
                raise ValueError(
                    "Explicit process parameterization requires parameter IDs."
                )
            if (
                process.parameterization == "locked_template_formula"
                and process.parameter_ids
            ):
                raise ValueError(
                    "Locked template formulas cannot declare editable "
                    "parameter IDs."
                )
            if (
                process.input_compartment_id not in compartment_ids
                or process.output_compartment_id not in compartment_ids
            ):
                raise ValueError(
                    "Process references an unknown compartment."
                )
        for scenario in self.simulation_scenarios:
            if scenario.regimen_id not in regimen_ids:
                raise ValueError(
                    "Simulation scenario references an unknown regimen."
                )
        expected = blueprint_checksum(self.model_dump(mode="json"))
        if self.checksum != expected:
            raise ValueError("PBPKBlueprint checksum mismatch.")
        return self


_BLUEPRINT_STATUS_ORDER: tuple[BlueprintStatus, ...] = (
    "draft_grounded",
    "structurally_validated",
    "compiled",
    "executable",
    "calibrated",
    "externally_validated",
)


def build_blueprint(payload: dict[str, Any]) -> PBPKBlueprint:
    value = dict(payload)
    value.pop("checksum", None)
    draft = PBPKBlueprint.model_construct(
        **value,
        checksum="0" * 64,
    )
    value["checksum"] = blueprint_checksum(
        draft.model_dump(mode="json")
    )
    return PBPKBlueprint.model_validate(value)


def advance_blueprint_status(
    blueprint: PBPKBlueprint,
    *,
    status: BlueprintStatus,
    created_at: datetime,
) -> PBPKBlueprint:
    if status == "quarantined":
        next_status = status
    else:
        current_index = _BLUEPRINT_STATUS_ORDER.index(blueprint.status)
        if current_index + 1 >= len(_BLUEPRINT_STATUS_ORDER):
            raise ValueError("Blueprint is already at its terminal status.")
        next_status = _BLUEPRINT_STATUS_ORDER[current_index + 1]
        if status != next_status:
            raise ValueError(
                f"Invalid blueprint status transition: "
                f"{blueprint.status} -> {status}."
            )
    if created_at.utcoffset() is None:
        raise ValueError("Blueprint transition time must be timezone-aware.")
    major, minor, patch = map(int, blueprint.blueprint_version.split("."))
    payload = dict(blueprint.__dict__)
    payload.update(
        {
            "blueprint_version": f"{major}.{minor}.{patch + 1}",
            "parent_blueprint_checksum": blueprint.checksum,
            "status": next_status,
            "created_at": created_at,
        }
    )
    return build_blueprint(payload)


class EquationTemplate(StrictBlueprintModel):
    equation_template_id: str
    process_type: Literal[
        "absorption",
        "distribution",
        "metabolism",
        "transport",
        "renal_excretion",
        "biliary_excretion",
        "ddi",
    ]
    equation_dimension: ParameterDimension
    parameterization: Literal[
        "explicit_blueprint_parameters",
        "locked_template_formula",
    ] = "explicit_blueprint_parameters"
    source_formula_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    source_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    mass_balance_validated: Literal[True]
    required_parameter_semantic_ids: list[str] = Field(
        default_factory=list
    )


class ParameterSlot(StrictBlueprintModel):
    slot_id: str
    semantic_id: str
    model_path: str
    dimension: ParameterDimension
    canonical_unit: str
    template_value: float
    source_location: str
    source_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    minimum_value: float | None = None
    maximum_value: float | None = None
    compound_specific: bool
    apply_to_runtime: bool = True

    @model_validator(mode="after")
    def bounds_valid(self) -> "ParameterSlot":
        if (
            self.minimum_value is not None
            and self.maximum_value is not None
            and self.minimum_value > self.maximum_value
        ):
            raise ValueError("Template parameter slot bounds are reversed.")
        return self


class BlueprintTemplate(StrictBlueprintModel):
    template_id: str
    template_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    target_runtime: TargetRuntime
    source_skeleton_relative_path: str
    source_skeleton_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_project_relative_path: str
    source_project_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_repository: str
    source_tag: str
    source_commit: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
    )
    source_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    license_id: str
    license_relative_path: str
    license_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_decision: LicenseDecision
    species: str
    skeleton_simulation_id: str = Field(min_length=1)
    skeleton_simulation_name: str = Field(min_length=1)
    skeleton_compound_names: list[str] = Field(min_length=1)
    skeleton_instantiated_process_count: int = Field(gt=0)
    skeleton_process_type_counts: dict[str, int]
    compound_semantic_ids: list[str] = Field(min_length=1)
    model_structure: Literal["whole_body", "reduced"]
    supported_objectives: list[str] = Field(min_length=1)
    supported_routes: list[str] = Field(min_length=1)
    supported_process_types: list[
        Literal[
            "absorption",
            "distribution",
            "metabolism",
            "transport",
            "renal_excretion",
            "biliary_excretion",
            "ddi",
        ]
    ] = Field(min_length=1)
    required_compartment_ids: list[str] = Field(min_length=1)
    optional_compartment_ids: list[str] = Field(default_factory=list)
    parameter_slots: list[ParameterSlot] = Field(min_length=1)
    equation_templates: list[EquationTemplate] = Field(min_length=1)
    allowed_topology_transformations: list[str] = Field(
        default_factory=list
    )
    validation_scenario_ids: list[str] = Field(min_length=1)
    validation_status: Literal[
        "structural_template_runtime_verified"
    ]

    @model_validator(mode="after")
    def template_identifiers_unique(self) -> "BlueprintTemplate":
        if self.source_commit is None and self.source_manifest_sha256 is None:
            raise ValueError(
                "Template source requires a Git commit or manifest checksum."
            )
        slot_ids = [item.slot_id for item in self.parameter_slots]
        paths = [item.model_path for item in self.parameter_slots]
        equations = [
            item.equation_template_id
            for item in self.equation_templates
        ]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("Template slot IDs must be unique.")
        if len(paths) != len(set(paths)):
            raise ValueError("Template parameter paths must be unique.")
        if len(equations) != len(set(equations)):
            raise ValueError("Equation template IDs must be unique.")
        if any(value <= 0 for value in self.skeleton_process_type_counts.values()):
            raise ValueError(
                "Template process type counts must contain positive values."
            )
        if sum(self.skeleton_process_type_counts.values()) != (
            self.skeleton_instantiated_process_count
        ):
            raise ValueError(
                "Template process type counts do not match the instantiated "
                "process total."
            )
        if not set(self.supported_process_types).issubset(
            self.skeleton_process_type_counts
        ):
            raise ValueError(
                "Supported process types must exist in the locked skeleton."
            )
        return self


class BlueprintValidationReport(StrictBlueprintModel):
    validation_schema_version: Literal["1.0.0"] = "1.0.0"
    validator_version: str
    validated_at: datetime
    blueprint_id: UUID
    blueprint_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_id: str
    accepted: bool
    compile_eligible: bool
    checks: dict[str, bool]
    counts: dict[str, int]
    errors: dict[str, list[dict[str, Any]]]


class CompilationArtifact(StrictBlueprintModel):
    artifact_kind: Literal[
        "input_blueprint",
        "validation_report",
        "template_skeleton",
        "compiler_script",
        "compiled_simulation",
        "simulation_results",
        "stdout_log",
        "stderr_log",
        "runtime_session",
        "dependency_lock",
        "dotnet_release_metadata",
        "executable_blueprint",
        "compiled_project",
        "input_snapshot",
        "roundtrip_snapshot",
        "roundtrip_audit",
        "scenario_manifest",
        "parameter_manifest",
        "dataset_manifest",
        "scientific_audit",
        "pkml_simulation",
        "simulation_structure_xml",
        "runtime_manifest",
    ]
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)


class RuntimeDependency(StrictBlueprintModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class OSPCompilerRuntimeIdentity(StrictBlueprintModel):
    runtime_kind: Literal["ospsuite_r"] = "ospsuite_r"
    platform: Literal["linux-x86_64"]
    r_version: str = Field(min_length=1)
    r_executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rscript_executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ospsuite_r_version: str = Field(min_length=1)
    ospsuite_r_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dotnet_runtime_version: str = Field(min_length=1)
    dotnet_executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dotnet_release_metadata_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    dependencies: list[RuntimeDependency] = Field(min_length=1)
    environment_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class PKSimCLICompilerRuntimeIdentity(StrictBlueprintModel):
    runtime_kind: Literal["pksim_cli_snapshot"] = "pksim_cli_snapshot"
    platform: Literal["windows-x86_64"]
    pksim_cli_version: str = Field(min_length=1)
    pksim_cli_executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roundtrip_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


CompilerRuntimeIdentity = Annotated[
    OSPCompilerRuntimeIdentity | PKSimCLICompilerRuntimeIdentity,
    Field(discriminator="runtime_kind"),
]


def compilation_report_checksum(value: dict[str, Any]) -> str:
    payload = {
        key: item for key, item in value.items() if key != "checksum"
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


class BlueprintCompilationReport(StrictBlueprintModel):
    compilation_schema_version: Literal["1.1.0"] = "1.1.0"
    compiler_version: str = Field(min_length=1)
    compilation_id: UUID
    compiled_at: datetime
    input_blueprint_id: UUID
    input_blueprint_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_blueprint_checksum: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    validation_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_id: str = Field(min_length=1)
    template_skeleton_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal[
        "compiled",
        "executable",
        "compile_failed",
        "execution_failed",
    ]
    compile_accepted: bool
    execution_accepted: bool
    command: list[str] = Field(min_length=1)
    runtime: CompilerRuntimeIdentity
    artifacts: list[CompilationArtifact] = Field(min_length=1)
    checks: dict[str, bool]
    counts: dict[str, int]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("compiled_at")
    @classmethod
    def compiled_at_timezone_aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError(
                "Compilation timestamp must be timezone-aware."
            )
        return value

    @model_validator(mode="after")
    def validate_compilation_status(self) -> "BlueprintCompilationReport":
        if self.execution_accepted != (self.status == "executable"):
            raise ValueError(
                "Only an executable report may accept execution."
            )
        if self.execution_accepted and not self.compile_accepted:
            raise ValueError(
                "Execution cannot pass when compilation failed."
            )
        if self.execution_accepted and self.output_blueprint_checksum is None:
            raise ValueError(
                "Executable compilation requires an output blueprint."
            )
        expected = compilation_report_checksum(
            self.model_dump(mode="json")
        )
        if self.checksum != expected:
            raise ValueError("BlueprintCompilationReport checksum mismatch.")
        return self
