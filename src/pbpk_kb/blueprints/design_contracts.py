"""Static input contracts for evidence-grounded human PK blueprint drafts.

These objects describe a design request, not executable equations or a validated
PBPK model. No solver, model service, database, or network is invoked here.
"""
from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import (
    CompartmentDefinition,
    ParameterDefinition,
    ParameterDimension,
    ProvenanceRef,
)

Identifier = Annotated[str, Field(min_length=1)]
Route = Literal["oral", "intravenous"]
Checksum = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _unique(values: list[str], label: str) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} must not contain blank identifiers.")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique.")


def _acyclic(graph: dict[str, list[str]], label: str) -> None:
    active: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            raise ValueError(f"{label} contains a cycle at {node}.")
        if node in done:
            return
        active.add(node)
        for dependency in graph[node]:
            if dependency not in graph:
                raise ValueError(f"{label} references unknown identifier {dependency}.")
            visit(dependency)
        active.remove(node)
        done.add(node)

    for node in graph:
        visit(node)


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Design inputs must not contain NaN or infinity.")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


class DesignModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def finite_nested_inputs(self) -> "DesignModel":
        # Existing schema models do not all reject non-finite optional values.
        _finite(self.model_dump())
        return self


class DesignTarget(DesignModel):
    target_id: Identifier
    analyte: Literal["parent", "metabolite"] = "parent"
    matrix: Literal["plasma", "whole_blood", "liver", "kidney", "urine", "tissue"]
    quantity: Literal["concentration", "amount", "fraction_excreted"]
    binding: Literal["total", "unbound"] = "total"
    required: bool = True


class DesignRequirement(DesignModel):
    compound_semantic_id: Identifier
    name: Identifier
    population: Identifier = "healthy_adult"
    route: Route
    targets: list[DesignTarget] = Field(min_length=1, max_length=32)
    dose: float = Field(gt=0)
    dose_unit: Identifier
    formulation: Identifier | None = None
    species: Literal["human"] = "human"

    @model_validator(mode="after")
    def unique_targets(self) -> "DesignRequirement":
        _unique([target.target_id for target in self.targets], "Target IDs")
        return self


class StudyReference(DesignModel):
    study_id: Identifier
    source_id: Identifier
    source_checksum: Checksum
    role: Literal["calibration", "selection", "test"]
    cohort_ids: list[Identifier] = Field(default_factory=list)
    observed_target_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_links(self) -> "StudyReference":
        _unique(self.cohort_ids, "Study cohort IDs")
        _unique(self.observed_target_ids, "Study observed target IDs")
        return self


class DesignEvidence(DesignModel):
    evidence_id: Identifier
    compound_semantic_id: Identifier
    kind: Literal["mechanism", "parameter"]
    mechanism: Identifier | None = None
    parameter_semantic_id: Identifier | None = None
    value: float | None = None
    unit: Identifier | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    provenance: ProvenanceRef
    study_id: Identifier | None = None
    species: Literal["human"] = "human"
    population: Identifier | None = None
    route: Route | None = None
    formulation: Identifier | None = None

    @model_validator(mode="after")
    def evidence_payload(self) -> "DesignEvidence":
        if (self.lower_bound is None) != (self.upper_bound is None):
            raise ValueError("Parameter bounds must provide both lower and upper bounds.")
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValueError("Parameter bounds are reversed.")
            if self.value is not None and not self.lower_bound <= self.value <= self.upper_bound:
                raise ValueError("Parameter value lies outside its evidence bounds.")
        if self.kind == "mechanism":
            if self.mechanism is None:
                raise ValueError("Mechanism evidence requires a mechanism identifier.")
            if any(value is not None for value in (
                self.parameter_semantic_id, self.value, self.unit,
                self.lower_bound, self.upper_bound,
            )):
                raise ValueError("Quantitative evidence must use kind='parameter'.")
        elif self.parameter_semantic_id is None or self.unit is None:
            raise ValueError("Parameter evidence requires a semantic ID and explicit unit.")
        elif self.value is None and self.lower_bound is None:
            raise ValueError("Parameter evidence requires a value or a bounded range.")
        return self


class HumanPhysiologyBase(DesignModel):
    base_id: Identifier
    version: Identifier
    compartments: list[CompartmentDefinition] = Field(min_length=1, max_length=256)
    parameters: list[ParameterDefinition] = Field(default_factory=list)
    supported_populations: list[Identifier] = Field(min_length=1)
    provenance: list[ProvenanceRef] = Field(min_length=1)

    @model_validator(mode="after")
    def physiology_is_compound_independent(self) -> "HumanPhysiologyBase":
        _unique(self.supported_populations, "Supported populations")
        compartment_ids = [item.compartment_id for item in self.compartments]
        _unique(compartment_ids, "Compartment IDs")
        _unique([item.parameter_id for item in self.parameters], "Physiology parameter IDs")
        _acyclic({
            item.compartment_id: ([item.parent_compartment_id] if item.parent_compartment_id else [])
            for item in self.compartments
        }, "Compartment parent hierarchy")
        for parameter in self.parameters:
            context = parameter.context
            if context.compound_semantic_id is not None:
                raise ValueError("Human physiology must not contain compound-specific parameters.")
            if context.species.strip().casefold() != "human":
                raise ValueError("Human physiology parameters require human species context.")
            if context.population is not None and context.population not in self.supported_populations:
                raise ValueError("Physiology parameter population is not declared by the base.")
            if context.compartment_id is not None and context.compartment_id not in compartment_ids:
                raise ValueError("Physiology parameter references an unknown compartment.")
        return self


class ModuleParameterSpec(DesignModel):
    semantic_id: Identifier
    dimension: ParameterDimension
    unit: Identifier
    required: bool = True
    estimate_allowed: bool = False
    route_independent: bool = False
    formulation_independent: bool = False
    observation_target_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_observations(self) -> "ModuleParameterSpec":
        _unique(self.observation_target_ids, "Parameter observation target IDs")
        return self


class MechanismModule(DesignModel):
    module_id: Identifier
    version: Identifier
    mechanism: Identifier
    role: Literal["input", "distribution", "elimination", "metabolism", "transport", "other"] = "other"
    routes: list[Route] = Field(default_factory=lambda: ["oral", "intravenous"], min_length=1)
    required_compartments: list[Identifier] = Field(default_factory=list)
    parameter_specs: list[ModuleParameterSpec] = Field(default_factory=list)
    provides_targets: list[Identifier] = Field(default_factory=list)
    requires_modules: list[Identifier] = Field(default_factory=list)
    excludes_modules: list[Identifier] = Field(default_factory=list)
    clearance_group: Identifier | None = None
    formulations: list[Identifier] = Field(default_factory=list)
    requires_evidence: bool = True
    provenance: list[ProvenanceRef] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_module_links(self) -> "MechanismModule":
        for field in ("routes", "formulations", "required_compartments", "provides_targets", "requires_modules", "excludes_modules"):
            _unique(getattr(self, field), f"Module {field}")
        _unique([item.semantic_id for item in self.parameter_specs], "Module parameter IDs")
        if self.module_id in self.requires_modules or self.module_id in self.excludes_modules:
            raise ValueError("A module cannot require or exclude itself.")
        if set(self.requires_modules) & set(self.excludes_modules):
            raise ValueError("A module cannot require and exclude the same module.")
        return self


class DesignBudget(DesignModel):
    beam_width: int = Field(default=4, ge=1, le=4, strict=True)
    max_rounds: int = Field(default=3, ge=1, le=3, strict=True)
    max_candidates: int = Field(default=24, ge=1, le=24, strict=True)
    max_free_parameters: int = Field(default=6, ge=0, le=6, strict=True)
    max_solver_calls: int = Field(default=5000, ge=1, le=5000, strict=True)


class RegisteredHumanModelRef(DesignModel):
    """A pinned project registry reference, not a client assertion of validity."""

    registry_id: Literal["pksim-human-standard-small-molecule"] = "pksim-human-standard-small-molecule"
    definition_checksum: Checksum


class HumanModelInstanceInput(DesignModel):
    """Explicit source-linked PK-Sim input blocks; no runtime is invoked.

    Snapshot field semantics and compatibility with the registered provider are
    checked by the builder. This contract only enforces identity and provenance.
    """

    population_context: Identifier
    individual: dict[str, Any]
    compound: dict[str, Any]
    protocol: dict[str, Any]
    snapshot_version: int = Field(default=80, ge=1, strict=True)
    simulation_name: Identifier
    human_source: ProvenanceRef
    compound_sources: dict[Identifier, ProvenanceRef] = Field(min_length=1)
    protocol_source: ProvenanceRef
    formulation: dict[str, Any] | None = None
    formulation_source: ProvenanceRef | None = None
    output_schema: list[dict[str, Any]] | None = None
    output_selections: list[Identifier] | None = None
    generated_demographics: list[Literal["Weight", "Height"]] = Field(default_factory=list)
    expression_profiles: list[dict[str, Any]] = Field(default_factory=list, max_length=8)
    expression_profile_sources: dict[Identifier, ProvenanceRef] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identity_and_optional_source(self) -> "HumanModelInstanceInput":
        _unique(self.generated_demographics, "Generated demographics")
        origin = self.individual.get("OriginData")
        if not isinstance(origin, dict) or origin.get("Species") != "Human":
            raise ValueError("Individual OriginData.Species must be exactly 'Human'.")
        if (self.formulation is None) != (self.formulation_source is None):
            raise ValueError("Formulation and its source must be supplied together.")
        return self


class BlueprintDesignRequest(DesignModel):
    requirement: DesignRequirement
    human_model: RegisteredHumanModelRef | None = None
    instance: HumanModelInstanceInput | None = None
    base: HumanPhysiologyBase | None = None
    evidence: list[DesignEvidence] = Field(default_factory=list, max_length=2048)
    modules: list[MechanismModule] = Field(default_factory=list, max_length=64)
    studies: list[StudyReference] = Field(default_factory=list, max_length=256)
    budget: DesignBudget = Field(default_factory=DesignBudget)
    forbidden_source_ids: list[Identifier] = Field(default_factory=list)
    forbidden_study_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def references_and_source_isolation(self) -> "BlueprintDesignRequest":
        _unique([item.evidence_id for item in self.evidence], "Evidence IDs")
        _unique([item.module_id for item in self.modules], "Module IDs")
        _unique([item.study_id for item in self.studies], "Study IDs")
        _unique(self.forbidden_source_ids, "Forbidden source IDs")
        _unique(self.forbidden_study_ids, "Forbidden study IDs")
        targets = {item.target_id for item in self.requirement.targets}
        compartments = {item.compartment_id for item in self.base.compartments} if self.base is not None else set()
        modules = {item.module_id: item for item in self.modules}
        _acyclic({key: item.requires_modules for key, item in modules.items()}, "Module dependencies")
        specs: dict[str, tuple[str, str]] = {}
        for module in self.modules:
            if self.human_model is None and self.base is not None and not set(module.required_compartments) <= compartments:
                raise ValueError("Module references a compartment absent from the physiology base.")
            if not set(module.excludes_modules) <= modules.keys():
                raise ValueError("Module exclusion references an unknown module.")
            if not set(module.provides_targets) <= targets:
                raise ValueError("Module output references an undeclared design target.")
            for parameter in module.parameter_specs:
                if not set(parameter.observation_target_ids) <= targets:
                    raise ValueError("Parameter references an undeclared observation target.")
                definition = (parameter.dimension, parameter.unit)
                if parameter.semantic_id in specs and specs[parameter.semantic_id] != definition:
                    raise ValueError("Shared module parameters have inconsistent dimensions or units.")
                specs[parameter.semantic_id] = definition
        studies = {item.study_id: item for item in self.studies}
        roles: dict[tuple[str, str], str] = {}
        source_checksums: dict[str, str] = {}
        blocked_sources = set(self.forbidden_source_ids)
        blocked_studies = set(self.forbidden_study_ids)
        blocked_checksums: set[str] = set()
        for study in self.studies:
            if not set(study.observed_target_ids) <= targets:
                raise ValueError("Study references an undeclared observation target.")
            identities = [("source", study.source_id), ("checksum", study.source_checksum)]
            identities.extend(("cohort", cohort) for cohort in study.cohort_ids)
            for identity in identities:
                if identity in roles and roles[identity] != study.role:
                    raise ValueError("Study sources or cohorts cannot cross data roles.")
                roles[identity] = study.role
            if study.source_id in source_checksums and source_checksums[study.source_id] != study.source_checksum:
                raise ValueError("A source ID cannot identify different source checksums.")
            source_checksums[study.source_id] = study.source_checksum
            if study.role in {"selection", "test"} or study.study_id in blocked_studies:
                blocked_sources.add(study.source_id)
                blocked_studies.add(study.study_id)
                blocked_checksums.add(study.source_checksum)
        for study in self.studies:
            if study.role == "calibration" and (study.source_id in blocked_sources or study.source_checksum in blocked_checksums):
                raise ValueError("Development studies include a forbidden or test source.")
        provenance = list(self.base.provenance) if self.base is not None else []
        if self.instance is not None:
            if self.instance.population_context != self.requirement.population:
                raise ValueError("Instance population_context must exactly match the requested population.")
            if self.instance.compound.get("Name") != self.requirement.name:
                raise ValueError("Instance compound Name must exactly match the requested compound name.")
            provenance.extend([self.instance.human_source, self.instance.protocol_source])
            provenance.extend(self.instance.compound_sources.values())
            provenance.extend(self.instance.expression_profile_sources.values())
            if self.instance.formulation_source is not None:
                provenance.append(self.instance.formulation_source)
        if self.base is not None:
            for item in self.base.compartments:
                provenance.extend(item.provenance)
            for item in self.base.parameters:
                provenance.extend(item.provenance)
        for module in self.modules:
            provenance.extend(module.provenance)
        for item in self.evidence:
            if item.compound_semantic_id != self.requirement.compound_semantic_id:
                raise ValueError("Evidence belongs to a different compound.")
            if item.study_id is not None:
                if item.study_id in blocked_studies:
                    raise ValueError("Usable evidence must not come from selection, test or forbidden studies.")
                if item.study_id not in studies:
                    raise ValueError("Evidence references an undeclared study.")
                study = studies[item.study_id]
                if (item.provenance.source_id, item.provenance.source_checksum) != (study.source_id, study.source_checksum):
                    raise ValueError("Evidence provenance does not match its declared study source.")
            provenance.append(item.provenance)
        for reference in provenance:
            if reference.source_id in blocked_sources or reference.source_checksum in blocked_checksums:
                raise ValueError("Usable design inputs include a selection, forbidden or test source.")
            previous = source_checksums.get(reference.source_id)
            if previous is not None and previous != reference.source_checksum:
                raise ValueError("Source provenance checksum conflicts with the declared source.")
            source_checksums[reference.source_id] = reference.source_checksum
        return self
