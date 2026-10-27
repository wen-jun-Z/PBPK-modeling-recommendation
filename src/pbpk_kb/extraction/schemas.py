from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class CurationStatus(StrEnum):
    machine_extracted = "machine_extracted"
    needs_review = "needs_review"
    reviewed_accepted = "reviewed_accepted"
    reviewed_rejected = "reviewed_rejected"
    approved_for_modeling = "approved_for_modeling"
    deprecated = "deprecated"
    duplicate = "duplicate"


class ExtractionMethod(StrEnum):
    machine_extracted = "machine_extracted"
    rule_based = "rule_based"
    section_candidate = "section_candidate"
    connector_metadata = "connector_metadata"
    human_reviewed = "human_reviewed"


class ValueOrigin(StrEnum):
    measured = "measured"
    predicted = "predicted"
    fitted = "fitted"
    assumed = "assumed"
    reported = "reported"
    unknown = "unknown"


class EvidenceOrigin(StrEnum):
    human = "human"
    animal = "animal"
    in_vitro = "in_vitro"
    in_silico = "in_silico"
    mixed = "mixed"
    unknown = "unknown"


class EvidenceBase(BaseModel):
    source_id: str
    source_location: str
    evidence_text: str | None = None
    value_origin: ValueOrigin = ValueOrigin.unknown
    evidence_origin: EvidenceOrigin = EvidenceOrigin.unknown
    extraction_method: ExtractionMethod = ExtractionMethod.machine_extracted
    extractor_version: str = "0.1.0"
    confidence_score: float | None = Field(default=None, ge=0, le=1)
    curation_status: CurationStatus = CurationStatus.machine_extracted
    raw_document_id: UUID | None = None
    parsed_document_id: UUID | None = None
    created_at: datetime | None = None

    @field_validator("source_location")
    @classmethod
    def source_location_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_location is required for traceability")
        return value


class CompoundIdentityEvidence(EvidenceBase):
    compound_id: UUID | None = None
    preferred_name: str
    synonym: str | None = None
    pubchem_cid: str | None = None
    chembl_id: str | None = None
    drugcentral_id: str | None = None
    rxnorm_rxcui: str | None = None
    inchi_key: str | None = None
    cas_number: str | None = None
    approval_status: str | None = None


class PhysicochemicalPropertyEvidence(EvidenceBase):
    compound_id: UUID | None = None
    property_name: str
    property_value: float | None = None
    property_value_text: str | None = None
    unit: str | None = None
    canonical_value: float | None = None
    canonical_unit: str | None = None
    unit_dimension: str | None = None
    method: str | None = None
    pH: float | None = None
    temperature: float | None = None


class ADMEParameterEvidence(EvidenceBase):
    compound_id: UUID | None = None
    parameter_name: str
    value: float | None = None
    value_text: str | None = None
    unit: str | None = None
    canonical_value: float | None = None
    canonical_unit: str | None = None
    unit_dimension: str | None = None
    matrix: str | None = None
    species: str | None = None
    system: str | None = None
    assay_method: str | None = None
    enzyme: str | None = None
    transporter: str | None = None
    measured_or_predicted: str | None = None


class ClinicalPKEvidence(EvidenceBase):
    compound_id: UUID | None = None
    study_context: str | None = None
    population: str | None = None
    dose: str | None = None
    route: str | None = None
    regimen: str | None = None
    matrix: str | None = None
    analyte: str | None = None
    pk_parameter: str
    value: float | None = None
    value_text: str | None = None
    unit: str | None = None
    canonical_value: float | None = None
    canonical_unit: str | None = None
    unit_dimension: str | None = None
    variability: str | None = None
    food_effect_flag: bool | None = None
    renal_impairment_flag: bool | None = None
    hepatic_impairment_flag: bool | None = None


class DDIEvidence(EvidenceBase):
    victim_compound_id: UUID | None = None
    perpetrator_compound_id: UUID | None = None
    enzyme_or_transporter: str | None = None
    interaction_type: str
    study_type: str | None = None
    auc_ratio: float | None = None
    cmax_ratio: float | None = None
    dose_victim: str | None = None
    dose_perpetrator: str | None = None
    population: str | None = None


class SpecialPopulationEvidence(EvidenceBase):
    compound_id: UUID | None = None
    population_type: str
    population_definition: str | None = None
    covariate: str | None = None
    pk_parameter: str | None = None
    value: float | None = None
    value_text: str | None = None
    unit: str | None = None
    canonical_value: float | None = None
    canonical_unit: str | None = None
    unit_dimension: str | None = None
    ratio_to_reference: float | None = None


class TissueExpressionEvidence(EvidenceBase):
    protein_target_id: UUID | None = None
    tissue: str
    expression_value: float | None = None
    expression_value_text: str | None = None
    unit: str | None = None
    canonical_value: float | None = None
    canonical_unit: str | None = None
    unit_dimension: str | None = None
    species: str | None = None
    assay_method: str | None = None


class PublicPBPKModelMetadata(EvidenceBase):
    model_id: UUID | None = None
    model_name: str
    compound_id: UUID | None = None
    model_type: str | None = None
    software: str | None = None
    species: str | None = None
    population: str | None = None
    route: str | None = None
    model_source: str | None = None
    model_file_path: str | None = None
    parameter_file_path: str | None = None
    evaluation_report_path: str | None = None
    publication_id: str | None = None
    validation_status: str | None = None
    notes: str | None = None
