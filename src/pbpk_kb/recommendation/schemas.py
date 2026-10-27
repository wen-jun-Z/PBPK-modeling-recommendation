from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


RecommendationObjective = Literal[
    "base_pk",
    "first_in_human",
    "dose_selection",
    "formulation_comparison",
    "food_effect",
    "bioequivalence",
    "ddi_victim",
    "ddi_perpetrator",
    "renal_impairment",
    "hepatic_impairment",
    "pediatric",
    "geriatric",
    "pregnancy",
    "pharmacogenomics",
    "tissue_exposure",
    "public_model_reuse",
    # Backward-compatible request aliases retained until the final Agent replaces
    # the Stage 1 rule-based endpoint.
    "oral_absorption",
    "ddi",
    "special_population",
]


class DoseRegimenInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: str | None = None
    dose: float | None = Field(default=None, gt=0)
    dose_unit: str | None = None
    single_or_multiple: Literal["single", "multiple"] | None = None
    frequency: str | None = None
    infusion_duration: str | None = None
    fed_or_fasted: Literal["fed", "fasted", "not_applicable"] | None = None
    formulation: str | None = None
    dosage_form: str | None = None
    release_type: str | None = None


class ModelingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Compound identity and form
    compound: str = Field(min_length=2)
    active_moiety: str | None = None
    salt_form: str | None = None
    metabolites_to_model: list[str] = Field(default_factory=list)
    candidate_public_model: str | None = None

    # Modeling purpose
    objective: RecommendationObjective = "base_pk"

    # Biological system
    species: str = "human"
    population: str | None = None
    age_range: str | None = None
    sex: Literal["female", "male", "all", "unspecified"] | None = None
    body_weight_kg: float | None = Field(default=None, gt=0)
    ethnicity: str | None = None
    renal_function: str | None = None
    hepatic_function: str | None = None
    disease_state: str | None = None
    genotype: str | None = None
    pregnancy_status: str | None = None

    # Administration and formulation
    route: str | None = None
    dose: float | None = Field(default=None, gt=0)
    dose_unit: str | None = None
    single_or_multiple: Literal["single", "multiple"] | None = None
    frequency: str | None = None
    infusion_duration: str | None = None
    fed_or_fasted: Literal["fed", "fasted", "not_applicable"] | None = None
    formulation: str | None = None
    dosage_form: str | None = None
    release_type: str | None = None

    # DDI scenario
    victim_drug: str | None = None
    perpetrator_drug: str | None = None
    interacting_drug: str | None = None
    enzyme: list[str] = Field(default_factory=list)
    transporter: list[str] = Field(default_factory=list)
    interaction_mechanism: str | None = None
    interaction_duration: str | None = None
    victim_regimen: DoseRegimenInput | None = None
    perpetrator_regimen: DoseRegimenInput | None = None

    # Prediction endpoints
    prediction_endpoints: list[
        Literal[
            "AUC",
            "Cmax",
            "Tmax",
            "half_life",
            "clearance",
            "volume_of_distribution",
            "concentration_time_profile",
            "tissue_concentration",
            "fraction_absorbed",
            "bioavailability",
            "renal_excretion",
            "DDI_AUC_ratio",
            "DDI_Cmax_ratio",
        ]
    ] = Field(default_factory=list)

    # Model constraints
    whole_body_or_reduced_pbpk: Literal["whole_body", "reduced", "unspecified"] | None = None
    parent_or_parent_metabolite: Literal["parent", "parent_metabolite"] | None = None
    required_organs: list[str] = Field(default_factory=list)
    required_enzymes: list[str] = Field(default_factory=list)
    required_transporters: list[str] = Field(default_factory=list)
    software_preference: list[str] = Field(default_factory=list)
    parameter_uncertainty: str | None = None
    calibration_datasets: list[str] = Field(default_factory=list)
    validation_datasets: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    sensitivity_analysis: bool | None = None
    uncertainty_analysis: bool | None = None
    constraints: str | None = None

    # Retrieval controls
    top_k_subgraphs: int = Field(default=8, ge=1, le=30)
    max_evidence_per_component: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def normalize_legacy_ddi_field(self) -> "ModelingRequest":
        if self.interacting_drug:
            if self.objective == "ddi_victim" and self.perpetrator_drug is None:
                self.perpetrator_drug = self.interacting_drug
            elif self.objective == "ddi_perpetrator" and self.victim_drug is None:
                self.victim_drug = self.interacting_drug
        return self

    def request_gaps(self) -> list[str]:
        gaps: list[str] = []
        if self.objective in {"ddi", "ddi_victim", "ddi_perpetrator"}:
            if self.objective == "ddi_victim" and not self.perpetrator_drug:
                gaps.append("DDI perpetrator drug is missing.")
            elif self.objective == "ddi_perpetrator" and not self.victim_drug:
                gaps.append("DDI victim drug is missing.")
            elif self.objective == "ddi" and not (
                self.interacting_drug or (self.victim_drug and self.perpetrator_drug)
            ):
                gaps.append("DDI victim/perpetrator roles are not fully specified.")
        if self.objective in {
            "special_population",
            "renal_impairment",
            "hepatic_impairment",
            "pediatric",
            "geriatric",
            "pregnancy",
        } and not any(
            (
                self.population,
                self.age_range,
                self.renal_function,
                self.hepatic_function,
                self.pregnancy_status,
            )
        ):
            gaps.append("Target population definition is missing.")
        regimen_fields: dict[str, Any] = {
            "route": self.route,
            "dose": self.dose,
            "dose_unit": self.dose_unit,
            "frequency": self.frequency,
        }
        if self.objective != "public_model_reuse" and any(
            value is not None for value in regimen_fields.values()
        ):
            missing = [name for name, value in regimen_fields.items() if value is None]
            if missing:
                gaps.append(f"Dose regimen is incomplete: missing {', '.join(missing)}.")
        return gaps


class RecommendationRequest(ModelingRequest):
    """Backward-compatible public request name for the recommendation API."""


class Citation(BaseModel):
    evidence_id: str
    source_id: str
    source_location: str
    source_text: str
    parameter: str | None = None
    value: float | str | None = None
    unit: str | None = None
    confidence: float | None = None


class ComponentRecommendation(BaseModel):
    component: str
    status: Literal["recommended", "conditional", "insufficient_evidence"]
    rationale: str
    parameters: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence_count: int = 0
    source_count: int = 0
    confidence: float = 0.0


class SubgraphResult(BaseModel):
    center_id: str
    center_label: str
    hop: int
    score: float
    matched_domains: list[str]
    evidence_ids: list[str]
    node_ids: list[str]
    edge_count: int


class RecommendationResponse(BaseModel):
    compound_id: str
    compound_name: str
    objective: RecommendationObjective
    retrieval_policy: str
    coverage_score: float
    overall_confidence: float
    components: list[ComponentRecommendation]
    ranked_subgraphs: list[SubgraphResult]
    public_model_candidates: list[ComponentRecommendation]
    evidence_gaps: list[str]
    warnings: list[str]
    generated_from_auto_validated_only: bool = True


RequirementCategory = Literal[
    "compound_identity",
    "biological_scope",
    "model_structure",
    "compartment",
    "adme_process",
    "enzyme",
    "transporter",
    "ddi_role",
    "population_feature",
    "administration_route",
    "formulation",
    "dose_regimen",
    "request_constraint",
    "prediction_endpoint",
    "validation_context",
    "parameter",
]
RequirementNecessity = Literal[
    "hard_required",
    "preferred",
    "optional",
    "unsupported",
]
RequirementOrigin = Literal[
    "user_request",
    "validated_evidence",
    "ontology_rule",
]


class StrictRecommendationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequirementRange(StrictRecommendationModel):
    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "RequirementRange":
        if self.lower > self.upper:
            raise ValueError("Requirement range lower bound exceeds upper bound.")
        return self


class RequirementContext(StrictRecommendationModel):
    species: str | None = None
    population: str | None = None
    route: str | None = None
    formulation: str | None = None
    dose: float | None = Field(default=None, gt=0)
    dose_unit: str | None = None


class CompoundRef(StrictRecommendationModel):
    compound_id: UUID
    preferred_name: str = Field(min_length=1)
    active_moiety: str = Field(min_length=1)
    salt_form: str | None = None
    external_identifiers: dict[str, str] = Field(default_factory=dict)
    identity_resolution_method: Literal[
        "database_exact_id",
        "database_normalized_name",
        "request_explicit_id",
        "evidence_kg_canonical_node",
    ]
    identity_confidence: float = Field(ge=0, le=1)


class RequirementItem(StrictRecommendationModel):
    requirement_id: UUID
    category: RequirementCategory
    semantic_id: str = Field(min_length=1)
    preferred_name: str = Field(min_length=1)
    necessity: RequirementNecessity
    expected_value: str | float | int | RequirementRange | None = None
    canonical_unit: str | None = None
    context: RequirementContext
    evidence_ids: list[UUID] = Field(default_factory=list)
    origin: RequirementOrigin
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_provenance(self) -> "RequirementItem":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Requirement evidence IDs must be unique.")
        if self.origin != "user_request" and not self.evidence_ids:
            raise ValueError(
                "Evidence and ontology requirements require provenance IDs."
            )
        return self


class RequirementGap(StrictRecommendationModel):
    gap_code: str = Field(min_length=1)
    severity: Literal["critical", "non_critical"]
    requirement_ids: list[UUID] = Field(default_factory=list)
    blocking: bool
    source: Literal["request_validation", "evidence_coverage", "ontology_rule"]


class RequirementConflictRef(StrictRecommendationModel):
    conflict_id: UUID
    requirement_ids: list[UUID] = Field(min_length=2)
    severity: Literal["hard", "soft"]
    reason_code: str = Field(min_length=1)
    evidence_ids: list[UUID] = Field(default_factory=list)

    @field_validator("requirement_ids")
    @classmethod
    def requirement_ids_are_unique(
        cls,
        value: list[UUID],
    ) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("Conflict requirement IDs must be unique.")
        return value


def requirement_profile_checksum(value: dict[str, Any]) -> str:
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


class RequirementProfile(StrictRecommendationModel):
    profile_id: UUID
    profile_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    request_id: UUID
    ontology_version: str = Field(min_length=1)
    compound_identity: CompoundRef
    requirements: list[RequirementItem] = Field(min_length=1)
    critical_gaps: list[RequirementGap] = Field(default_factory=list)
    conflicts: list[RequirementConflictRef] = Field(default_factory=list)
    source_evidence_run_id: UUID
    ranking_eligibility: Literal["ready", "blocked"]
    created_at: datetime
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("profile_version")
    @classmethod
    def profile_version_is_semver(cls, value: str) -> str:
        if re.fullmatch(r"\d+\.\d+\.\d+", value) is None:
            raise ValueError("profile_version must be semantic version.")
        return value

    @field_validator("created_at")
    @classmethod
    def created_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_profile(self) -> "RequirementProfile":
        requirement_ids = [
            requirement.requirement_id for requirement in self.requirements
        ]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("Requirement IDs must be unique.")
        known_ids = set(requirement_ids)
        referenced_ids = {
            requirement_id
            for conflict in self.conflicts
            for requirement_id in conflict.requirement_ids
        } | {
            requirement_id
            for gap in self.critical_gaps
            for requirement_id in gap.requirement_ids
        }
        if not referenced_ids.issubset(known_ids):
            raise ValueError("Gap or conflict references an unknown requirement.")
        expected_eligibility = (
            "blocked"
            if any(
                gap.blocking for gap in self.critical_gaps
            )
            or any(conflict.severity == "hard" for conflict in self.conflicts)
            else "ready"
        )
        if self.ranking_eligibility != expected_eligibility:
            raise ValueError(
                "ranking_eligibility does not match gaps and conflicts."
            )
        expected_checksum = requirement_profile_checksum(
            self.model_dump(mode="json")
        )
        if self.checksum != expected_checksum:
            raise ValueError("RequirementProfile checksum mismatch.")
        return self


ConstraintSeverity = Literal["hard", "soft"]
ConstraintResult = Literal[
    "pass",
    "fail",
    "unknown",
    "not_applicable",
]
ModelCandidateClass = Literal[
    "exact_compound_model",
    "same_compound_adaptation",
    "mechanistic_template",
]
ModelCandidateEligibility = Literal[
    "eligible",
    "ineligible",
    "template_only",
]
ModelDecision = Literal["reuse", "adapt", "abstain"]
DecisionReasonCode = Literal[
    "REQUIREMENT_SATISFIED",
    "NOT_APPLICABLE",
    "ASSET_MISSING",
    "CHECKSUM_FAILED",
    "LICENSE_UNRESOLVED",
    "PARSER_FAILED",
    "EXECUTION_PROBE_FAILED",
    "COMPOUND_MISMATCH",
    "SPECIES_MISMATCH",
    "ROUTE_UNSUPPORTED",
    "FORMULATION_UNSUPPORTED",
    "REQUIRED_MECHANISM_MISSING",
    "ENDPOINT_UNSUPPORTED",
    "SOFTWARE_UNAVAILABLE",
    "VALIDATION_NOT_TRANSFERABLE",
    "CRITICAL_EVIDENCE_GAP",
    "UNRESOLVED_CONFLICT",
    "AMBIGUOUS_TOP_CANDIDATES",
    "SCORE_BELOW_THRESHOLD",
    "NO_ELIGIBLE_MODEL",
    "NO_EXECUTABLE_TEMPLATE",
    "VALIDATION_CRITERIA_MISSING",
    "VALIDATION_FAILED",
    "TRAINING_LABELS_INSUFFICIENT",
]


class ConstraintCheck(StrictRecommendationModel):
    constraint_id: str = Field(min_length=1)
    model_version_id: UUID
    source_model_version_key: str = Field(min_length=1)
    requirement_id: UUID
    severity: ConstraintSeverity
    result: ConstraintResult
    observed_value: (
        str
        | float
        | int
        | RequirementRange
        | list[str]
        | dict[str, Any]
        | None
    ) = None
    reason_code: DecisionReasonCode
    evidence_ids: list[UUID] = Field(default_factory=list)
    model_asset_locations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_constraint_provenance(self) -> "ConstraintCheck":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Constraint evidence IDs must be unique.")
        if len(self.model_asset_locations) != len(
            set(self.model_asset_locations)
        ):
            raise ValueError(
                "Constraint model asset locations must be unique."
            )
        if (
            self.result != "not_applicable"
            and not self.evidence_ids
            and not self.model_asset_locations
        ):
            raise ValueError(
                "Applicable constraints require evidence or model assets."
            )
        if (
            self.result == "pass"
            and self.reason_code != "REQUIREMENT_SATISFIED"
        ):
            raise ValueError(
                "Passing constraints require REQUIREMENT_SATISFIED."
            )
        if (
            self.result == "not_applicable"
            and self.reason_code != "NOT_APPLICABLE"
        ):
            raise ValueError(
                "Not-applicable constraints require NOT_APPLICABLE."
            )
        return self


class ModelScoreFeature(StrictRecommendationModel):
    feature_name: Literal[
        "compound_identity_match",
        "mechanism_coverage",
        "route_formulation_match",
        "population_match",
        "structure_match",
        "endpoint_coverage",
        "validation_quality",
        "validation_transferability",
        "source_quality",
        "reproducibility",
        "parameter_completeness",
        "evidence_conflict_penalty",
        "adaptation_cost",
        "uncertainty",
    ]
    raw_value: float
    normalized_value: float = Field(ge=0, le=1)
    weight: float
    weighted_value: float
    method_version: str = Field(min_length=1)


class RankedModelCandidate(StrictRecommendationModel):
    model_version_id: UUID
    source_model_version_key: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    candidate_class: ModelCandidateClass
    eligibility: ModelCandidateEligibility
    constraint_checks: list[ConstraintCheck] = Field(min_length=1)
    score_features: list[ModelScoreFeature] = Field(default_factory=list)
    deterministic_compatibility: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    learned_graph_compatibility: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    final_score: float | None = Field(default=None, ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    validation_run_id: str | None = None
    run_label_grade: int = Field(default=0, ge=0, le=3)
    ranking_label: Literal[
        "reuse_positive",
        "abstain_validation_failed",
        "abstain_execution_failed",
    ] | None = None
    matched_requirement_ids: list[UUID] = Field(default_factory=list)
    unmet_requirement_ids: list[UUID] = Field(default_factory=list)
    exclusion_reason_codes: list[DecisionReasonCode] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_candidate_eligibility(self) -> "RankedModelCandidate":
        if any(
            check.model_version_id != self.model_version_id
            or check.source_model_version_key
            != self.source_model_version_key
            for check in self.constraint_checks
        ):
            raise ValueError(
                "Constraint checks must reference their candidate."
            )
        hard_blocked = any(
            check.severity == "hard"
            and check.result in {"fail", "unknown"}
            for check in self.constraint_checks
        )
        expected = (
            "template_only"
            if self.candidate_class == "mechanistic_template"
            else "ineligible"
            if hard_blocked
            else "eligible"
        )
        if self.eligibility != expected:
            raise ValueError(
                "Candidate eligibility does not match class/constraints."
            )
        if len(self.matched_requirement_ids) != len(
            set(self.matched_requirement_ids)
        ):
            raise ValueError("Matched requirement IDs must be unique.")
        if len(self.unmet_requirement_ids) != len(
            set(self.unmet_requirement_ids)
        ):
            raise ValueError("Unmet requirement IDs must be unique.")
        if set(self.matched_requirement_ids) & set(
            self.unmet_requirement_ids
        ):
            raise ValueError(
                "A requirement cannot be both matched and unmet."
            )
        if len(self.exclusion_reason_codes) != len(
            set(self.exclusion_reason_codes)
        ):
            raise ValueError("Exclusion reason codes must be unique.")
        if (
            self.eligibility != "eligible"
            and not self.exclusion_reason_codes
        ):
            raise ValueError(
                "Ineligible/template candidates need reason codes."
            )
        return self


def ranked_model_recommendation_checksum(
    value: dict[str, Any],
) -> str:
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


class RankedModelRecommendation(StrictRecommendationModel):
    ranking_run_id: UUID
    ranking_schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    profile_id: UUID
    decision: ModelDecision
    candidates: list[RankedModelCandidate] = Field(min_length=1)
    selected_model_version_id: UUID | None = None
    adaptation_plan_id: UUID | None = None
    abstention_reason_codes: list[DecisionReasonCode] = Field(
        default_factory=list
    )
    decision_certainty: float = Field(ge=0, le=1)
    confidence_semantics: Literal[
        "deterministic_gate",
        "probability_calibrated",
    ]
    model_run_manifest_id: UUID
    auto_reuse_enabled: bool
    created_at: datetime
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("created_at")
    @classmethod
    def ranking_time_is_timezone_aware(
        cls,
        value: datetime,
    ) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> "RankedModelRecommendation":
        if (
            self.confidence_semantics == "deterministic_gate"
            and self.decision_certainty != 1.0
        ):
            raise ValueError(
                "Deterministic gate certainty must be 1.0 and must not be "
                "interpreted as a calibrated success probability."
            )
        candidate_ids = [
            candidate.model_version_id
            for candidate in self.candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Candidate model version IDs must be unique.")
        selected = next(
            (
                candidate
                for candidate in self.candidates
                if candidate.model_version_id
                == self.selected_model_version_id
            ),
            None,
        )
        if self.decision == "reuse":
            if (
                selected is None
                or selected.eligibility != "eligible"
                or self.adaptation_plan_id is not None
            ):
                raise ValueError(
                    "reuse requires one eligible selected model."
                )
            if any(
                check.severity == "hard"
                and check.result in {"fail", "unknown"}
                for check in selected.constraint_checks
            ):
                raise ValueError(
                    "reuse requires every hard constraint to pass."
                )
            if self.abstention_reason_codes:
                raise ValueError("reuse cannot carry abstention reasons.")
        elif self.decision == "adapt":
            if (
                selected is None
                or selected.eligibility != "eligible"
                or self.adaptation_plan_id is None
            ):
                raise ValueError(
                    "adapt requires an eligible model and adaptation plan."
                )
            if self.abstention_reason_codes:
                raise ValueError("adapt cannot carry abstention reasons.")
        else:
            if (
                self.selected_model_version_id is not None
                or self.adaptation_plan_id is not None
                or not self.abstention_reason_codes
            ):
                raise ValueError(
                    "abstain requires reasons and no selected model."
                )
        if len(self.abstention_reason_codes) != len(
            set(self.abstention_reason_codes)
        ):
            raise ValueError("Abstention reason codes must be unique.")
        expected_checksum = ranked_model_recommendation_checksum(
            self.model_dump(mode="json")
        )
        if self.checksum != expected_checksum:
            raise ValueError(
                "RankedModelRecommendation checksum mismatch."
            )
        return self
