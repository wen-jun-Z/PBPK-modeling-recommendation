from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pbpk_kb.recommendation.ontology import ONTOLOGY_VERSION, validate_node_type, validate_relation


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(namespace: str, *parts: Any) -> str:
    body = "\x1f".join(str(part or "").strip() for part in parts)
    digest = hashlib.sha256(f"{namespace}\x1e{body}".encode("utf-8")).hexdigest()
    return f"{namespace}:{digest[:24]}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(StrictModel):
    evidence_id: str
    source_id: str
    source_location: str
    supporting_text: str = Field(min_length=1)
    source_text_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    extraction_method: str
    extractor_version: str
    confidence: float = Field(ge=0.0, le=1.0)
    quality_status: Literal["auto_validated"]
    negation_status: Literal["affirmed", "negated", "uncertain"] = "affirmed"
    speculation_status: Literal["asserted", "speculative", "uncertain"] = "asserted"
    context_id: str | None = None


class EntityMention(StrictModel):
    mention_id: str
    evidence_id: str
    source_id: str
    source_location: str
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    entity_type: str
    normalized_text: str
    linked_node_id: str | None = None
    standard_identifier: str | None = None
    link_method: str
    recognition_method: str
    recognition_model: str
    confidence: float = Field(ge=0.0, le=1.0)
    role: str | None = None

    @model_validator(mode="after")
    def validate_span(self) -> "EntityMention":
        if self.end_char <= self.start_char:
            raise ValueError("Entity mention end_char must be greater than start_char.")
        return self


class UnresolvedEntity(StrictModel):
    mention_id: str
    evidence_id: str
    entity_type: str
    mention_text: str
    normalized_text: str
    candidate_identifiers: list[str] = Field(default_factory=list)
    rejection_reason: str
    recognition_confidence: float = Field(ge=0.0, le=1.0)


class KGNode(StrictModel):
    node_id: str
    node_type: str
    preferred_name: str
    standard_identifiers: dict[str, str] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)
    ontology_version: str = ONTOLOGY_VERSION

    @model_validator(mode="after")
    def validate_type(self) -> "KGNode":
        error = validate_node_type(self.node_type)
        if error:
            raise ValueError(error)
        return self


class KGEdge(StrictModel):
    edge_id: str
    subject_id: str
    subject_type: str
    predicate: str
    object_id: str
    object_type: str
    provenance: Provenance
    properties: dict[str, Any] = Field(default_factory=dict)
    ontology_version: str = ONTOLOGY_VERSION

    @model_validator(mode="after")
    def validate_domain_range(self) -> "KGEdge":
        error = validate_relation(self.predicate, self.subject_type, self.object_type)
        if error:
            raise ValueError(error)
        return self


class KGEvent(StrictModel):
    event_id: str
    event_type: str
    evidence_id: str
    trigger_text: str
    argument_edge_ids: list[str] = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    extraction_method: str
    extractor_version: str
    confidence: float = Field(ge=0.0, le=1.0)
    validation_status: Literal["validated", "rejected", "unresolved"]
    validation_reasons: list[str] = Field(default_factory=list)


class RejectedRelation(StrictModel):
    rejection_id: str
    evidence_id: str
    predicate: str | None = None
    subject_mention: str | None = None
    object_mention: str | None = None
    extraction_method: str
    reasons: list[str] = Field(min_length=1)
    supporting_text: str
    created_at: str = Field(default_factory=utc_now)


class ExtractionManifest(StrictModel):
    run_id: str
    input_path: str
    input_sha256: str
    external_catalog_path: str | None = None
    external_catalog_sha256: str | None = None
    external_catalog_count: int = 0
    ddi_relation_decisions_path: str | None = None
    ddi_relation_decisions_sha256: str | None = None
    ddi_relation_decision_count: int = 0
    ontology_version: str = ONTOLOGY_VERSION
    ner_backend: str
    ner_model: str
    ner_threshold: float
    relation_models: list[str]
    llm_model: str | None = None
    evidence_count: int
    mention_count: int
    linked_mention_count: int
    unresolved_mention_count: int
    node_count: int
    edge_count: int
    event_count: int
    rejected_relation_count: int
    quality_status_counts: dict[str, int]
    generated_at: str = Field(default_factory=utc_now)
