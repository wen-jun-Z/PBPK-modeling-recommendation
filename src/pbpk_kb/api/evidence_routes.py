from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.storage.activity_log import record_user_activity
from pbpk_kb.storage.db import get_db_session
from pbpk_kb.storage.final_evidence_query import reviewable_evidence_sql, serving_evidence_sql


router = APIRouter(tags=["evidence"])


class CurationAction(StrEnum):
    accept = "accept"
    reject = "reject"
    approve_for_modeling = "approve_for_modeling"
    mark_duplicate = "mark_duplicate"
    flag_for_review = "flag_for_review"
    edit = "edit"
    note = "note"


class CurationRequest(BaseModel):
    action: CurationAction
    curator: str = Field(min_length=1)
    comment: str | None = None
    field_updates: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceTable:
    evidence_type: str
    table_name: str
    id_column: str
    parameter_expression: str
    value_expression: str
    value_text_expression: str
    unit_expression: str
    canonical_value_expression: str
    canonical_unit_expression: str
    unit_dimension_expression: str
    context_expression: str
    compound_id_expression: str
    compound_join_expression: str
    editable_fields: tuple[str, ...]


COMMON_EDITABLE_FIELDS = (
    "evidence_text",
    "value_origin",
    "evidence_origin",
    "confidence_score",
)


EVIDENCE_TABLES: dict[str, EvidenceTable] = {
    "physicochemical_property": EvidenceTable(
        evidence_type="physicochemical_property",
        table_name="physicochemical_property_evidence",
        id_column="evidence_id",
        parameter_expression="e.property_name",
        value_expression="e.property_value::float",
        value_text_expression="e.property_value_text",
        unit_expression="e.unit",
        canonical_value_expression="e.canonical_value::float",
        canonical_unit_expression="e.canonical_unit",
        unit_dimension_expression="e.unit_dimension",
        context_expression="e.method",
        compound_id_expression="e.compound_id",
        compound_join_expression="compound.compound_id = e.compound_id",
        editable_fields=("property_name", "property_value", "property_value_text", "unit", "method", "ph", "temperature", *COMMON_EDITABLE_FIELDS),
    ),
    "adme_parameter": EvidenceTable(
        evidence_type="adme_parameter",
        table_name="adme_parameter_evidence",
        id_column="evidence_id",
        parameter_expression="e.parameter_name",
        value_expression="e.value::float",
        value_text_expression="e.value_text",
        unit_expression="e.unit",
        canonical_value_expression="e.canonical_value::float",
        canonical_unit_expression="e.canonical_unit",
        unit_dimension_expression="e.unit_dimension",
        context_expression="CONCAT_WS(' / ', e.matrix, e.species, e.system, e.assay_method, e.enzyme, e.transporter)",
        compound_id_expression="e.compound_id",
        compound_join_expression="compound.compound_id = e.compound_id",
        editable_fields=(
            "parameter_name",
            "value",
            "value_text",
            "unit",
            "matrix",
            "species",
            "system",
            "assay_method",
            "enzyme",
            "transporter",
            "measured_or_predicted",
            *COMMON_EDITABLE_FIELDS,
        ),
    ),
    "clinical_pk": EvidenceTable(
        evidence_type="clinical_pk",
        table_name="clinical_pk_evidence",
        id_column="evidence_id",
        parameter_expression="e.pk_parameter",
        value_expression="e.value::float",
        value_text_expression="e.value_text",
        unit_expression="e.unit",
        canonical_value_expression="e.canonical_value::float",
        canonical_unit_expression="e.canonical_unit",
        unit_dimension_expression="e.unit_dimension",
        context_expression="e.study_context",
        compound_id_expression="e.compound_id",
        compound_join_expression="compound.compound_id = e.compound_id",
        editable_fields=(
            "study_context",
            "population",
            "dose",
            "route",
            "regimen",
            "matrix",
            "analyte",
            "pk_parameter",
            "value",
            "value_text",
            "unit",
            "variability",
            "food_effect_flag",
            "renal_impairment_flag",
            "hepatic_impairment_flag",
            *COMMON_EDITABLE_FIELDS,
        ),
    ),
    "ddi": EvidenceTable(
        evidence_type="ddi",
        table_name="ddi_evidence",
        id_column="evidence_id",
        parameter_expression="e.interaction_type",
        value_expression="e.auc_ratio::float",
        value_text_expression="COALESCE(e.auc_ratio::text, e.cmax_ratio::text)",
        unit_expression="NULL::text",
        canonical_value_expression="NULL::float",
        canonical_unit_expression="NULL::text",
        unit_dimension_expression="NULL::text",
        context_expression="CONCAT_WS(' / ', e.enzyme_or_transporter, e.study_type)",
        compound_id_expression="e.victim_compound_id",
        compound_join_expression="compound.compound_id = e.victim_compound_id",
        editable_fields=(
            "enzyme_or_transporter",
            "interaction_type",
            "study_type",
            "auc_ratio",
            "cmax_ratio",
            "dose_victim",
            "dose_perpetrator",
            "population",
            *COMMON_EDITABLE_FIELDS,
        ),
    ),
    "special_population": EvidenceTable(
        evidence_type="special_population",
        table_name="special_population_evidence",
        id_column="evidence_id",
        parameter_expression="e.population_type",
        value_expression="e.value::float",
        value_text_expression="e.value_text",
        unit_expression="e.unit",
        canonical_value_expression="e.canonical_value::float",
        canonical_unit_expression="e.canonical_unit",
        unit_dimension_expression="e.unit_dimension",
        context_expression="CONCAT_WS(' / ', e.population_definition, e.covariate, e.pk_parameter)",
        compound_id_expression="e.compound_id",
        compound_join_expression="compound.compound_id = e.compound_id",
        editable_fields=(
            "population_type",
            "population_definition",
            "covariate",
            "pk_parameter",
            "value",
            "value_text",
            "unit",
            "ratio_to_reference",
            *COMMON_EDITABLE_FIELDS,
        ),
    ),
    "tissue_expression": EvidenceTable(
        evidence_type="tissue_expression",
        table_name="tissue_expression_evidence",
        id_column="evidence_id",
        parameter_expression=(
            "CONCAT_WS(' in ', "
            "(SELECT COALESCE(pt.gene_symbol, pt.preferred_name) "
            " FROM protein_target_identity pt "
            " WHERE pt.protein_target_id = e.protein_target_id), e.tissue)"
        ),
        value_expression="e.expression_value::float",
        value_text_expression="e.expression_value_text",
        unit_expression="e.unit",
        canonical_value_expression="e.canonical_value::float",
        canonical_unit_expression="e.canonical_unit",
        unit_dimension_expression="e.unit_dimension",
        context_expression="CONCAT_WS(' / ', e.species, e.assay_method)",
        compound_id_expression="NULL",
        compound_join_expression="FALSE",
        editable_fields=(
            "tissue",
            "expression_value",
            "expression_value_text",
            "unit",
            "species",
            "assay_method",
            *COMMON_EDITABLE_FIELDS,
        ),
    ),
    "model_parameter": EvidenceTable(
        evidence_type="model_parameter",
        table_name="model_parameter_evidence",
        id_column="model_parameter_evidence_id",
        parameter_expression="e.parameter_name",
        value_expression="e.parameter_value::float",
        value_text_expression="e.parameter_value_text",
        unit_expression="e.unit",
        canonical_value_expression="e.canonical_value::float",
        canonical_unit_expression="e.canonical_unit",
        unit_dimension_expression="e.unit_dimension",
        context_expression="e.parameter_context",
        compound_id_expression="e.compound_id",
        compound_join_expression="compound.compound_id = e.compound_id",
        editable_fields=(
            "parameter_name",
            "parameter_value",
            "parameter_value_text",
            "unit",
            "parameter_context",
            *COMMON_EDITABLE_FIELDS,
        ),
    ),
}

TABLE_ALIASES = {config.table_name: config.evidence_type for config in EVIDENCE_TABLES.values()}

FINAL_EVIDENCE_TYPES = {
    "physicochemical",
    "adme",
    "clinical_pk",
    "ddi",
    "special_population",
    "tissue_expression",
    "public_pbpk_model",
}

PROCESSED_EVIDENCE_EDITABLE_FIELDS = (
    "source_text",
    "parameter",
    "value",
    "value_text",
    "unit",
    "canonical_value",
    "canonical_unit",
    "condition",
    "population",
    "route",
    "dose",
    "extraction_confidence",
)


def serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        number = float(value)
        return int(number) if number.is_integer() else number
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    return value


def serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: serialize_value(value) for key, value in row.items()}


def resolve_evidence_table(evidence_type_or_table: str) -> EvidenceTable:
    key = evidence_type_or_table.strip()
    if key in EVIDENCE_TABLES:
        return EVIDENCE_TABLES[key]
    if key in TABLE_ALIASES:
        return EVIDENCE_TABLES[TABLE_ALIASES[key]]
    raise HTTPException(status_code=404, detail=f"Unsupported evidence type: {evidence_type_or_table}")


def is_final_evidence_type(evidence_type: str) -> bool:
    return evidence_type.strip() in FINAL_EVIDENCE_TYPES


def target_status_for_action(action: CurationAction, current_status: str) -> str:
    if action == CurationAction.accept:
        return "reviewed_accepted"
    if action == CurationAction.reject:
        return "reviewed_rejected"
    if action == CurationAction.approve_for_modeling:
        return "approved_for_modeling"
    if action == CurationAction.mark_duplicate:
        return "duplicate"
    if action == CurationAction.flag_for_review:
        return "needs_review"
    return current_status


VALID_TRANSITIONS: dict[str, set[str]] = {
    "machine_extracted": {"needs_review", "duplicate"},
    "needs_review": {"reviewed_accepted", "reviewed_rejected", "duplicate", "needs_review"},
    "reviewed_accepted": {"approved_for_modeling", "needs_review", "duplicate", "reviewed_accepted"},
    "reviewed_rejected": {"needs_review", "duplicate", "reviewed_rejected"},
    "approved_for_modeling": {"needs_review", "deprecated", "approved_for_modeling"},
    "duplicate": {"needs_review", "duplicate"},
    "deprecated": {"needs_review", "deprecated"},
}


def validate_transition(current_status: str, new_status: str, action: CurationAction) -> None:
    if action in {CurationAction.edit, CurationAction.note} and current_status == new_status:
        return
    if new_status not in VALID_TRANSITIONS.get(current_status, set()):
        raise HTTPException(
            status_code=409,
            detail=f"Invalid curation transition from {current_status} to {new_status}",
        )


VALUE_FIELDS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "physicochemical_property_evidence": ("property_value", "property_value_text"),
    "adme_parameter_evidence": ("value", "value_text"),
    "clinical_pk_evidence": ("value", "value_text"),
    "ddi_evidence": ("auc_ratio", "cmax_ratio"),
    "special_population_evidence": ("value", "value_text", "ratio_to_reference"),
    "tissue_expression_evidence": ("expression_value", "expression_value_text"),
    "model_parameter_evidence": ("parameter_value", "parameter_value_text"),
}

CONTEXT_FIELDS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "physicochemical_property_evidence": ("method",),
    "adme_parameter_evidence": ("matrix", "species", "system", "assay_method", "enzyme", "transporter"),
    "clinical_pk_evidence": ("study_context", "population", "dose", "route", "matrix", "analyte"),
    "ddi_evidence": ("enzyme_or_transporter", "study_type", "population"),
    "special_population_evidence": ("population_definition", "covariate", "pk_parameter"),
    "tissue_expression_evidence": ("tissue", "species", "assay_method"),
    "model_parameter_evidence": ("parameter_context",),
}


def _has_text(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _has_modeling_value(config: EvidenceTable, row: dict[str, Any]) -> bool:
    return any(_has_text(row.get(field_name)) for field_name in VALUE_FIELDS_BY_TABLE.get(config.table_name, ()))


def _has_numeric_like_value_without_unit(config: EvidenceTable, row: dict[str, Any]) -> bool:
    if config.table_name == "ddi_evidence":
        return False
    if not any(field_name in row for field_name in ("unit",)):
        return False
    if _has_text(row.get("unit")):
        return False
    numeric_fields = {
        "physicochemical_property_evidence": ("property_value", "property_value_text"),
        "adme_parameter_evidence": ("value", "value_text"),
        "clinical_pk_evidence": ("value", "value_text"),
        "special_population_evidence": ("value", "value_text"),
        "tissue_expression_evidence": ("expression_value", "expression_value_text"),
    }.get(config.table_name, ())
    return any(_has_text(row.get(field_name)) for field_name in numeric_fields)


def _has_modeling_context(config: EvidenceTable, row: dict[str, Any]) -> bool:
    context_fields = CONTEXT_FIELDS_BY_TABLE.get(config.table_name, ())
    if any(_has_text(row.get(field_name)) for field_name in context_fields):
        return True
    return _has_text(row.get("evidence_text")) and _has_text(row.get("source_location"))


def modeling_approval_quality_issues(config: EvidenceTable, row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if not _has_modeling_value(config, row):
        issues.append("approved_for_modeling requires a numeric value, value text, ratio, or equivalent model-ready value")
    if _has_numeric_like_value_without_unit(config, row):
        issues.append("numeric evidence requires a unit before approved_for_modeling")
    if not _has_modeling_context(config, row):
        issues.append("approved_for_modeling requires study, assay, population, route, or source-text context")
    return issues


def validate_modeling_approval_quality(config: EvidenceTable, row: dict[str, Any]) -> None:
    issues = modeling_approval_quality_issues(config, row)
    if issues:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Evidence is not ready for approved_for_modeling",
                "quality_issues": issues,
            },
        )


def evidence_union_sql() -> str:
    selects = []
    for config in EVIDENCE_TABLES.values():
        selects.append(
            f"""
            SELECT
                '{config.evidence_type}' AS evidence_type,
                '{config.table_name}' AS evidence_table,
                e.{config.id_column}::text AS evidence_id,
                {config.compound_id_expression}::text AS compound_id,
                compound.preferred_name AS compound_name,
                {config.parameter_expression} AS parameter_name,
                {config.value_expression} AS numeric_value,
                {config.value_text_expression} AS value_text,
                {config.unit_expression} AS unit,
                {config.canonical_value_expression} AS canonical_value,
                {config.canonical_unit_expression} AS canonical_unit,
                {config.unit_dimension_expression} AS unit_dimension,
                {config.context_expression} AS context,
                e.source_id,
                source.source_name,
                e.source_location,
                e.evidence_text,
                e.value_origin,
                e.evidence_origin,
                e.extraction_method,
                e.confidence_score::float,
                e.curation_status::text,
                e.created_at
            FROM {config.table_name} e
            LEFT JOIN compound_identity compound ON {config.compound_join_expression}
            LEFT JOIN source_registry source ON source.source_id = e.source_id
            """
        )
    return "\nUNION ALL\n".join(selects)


def fetch_evidence_row(db: Session, config: EvidenceTable, evidence_id: str) -> dict[str, Any]:
    row = db.execute(
        text(f"SELECT * FROM {config.table_name} WHERE {config.id_column} = CAST(:evidence_id AS uuid)"),
        {"evidence_id": evidence_id},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Evidence record not found")
    return serialize_row(dict(row))


def normalize_field_updates(config: EvidenceTable, updates: dict[str, Any]) -> dict[str, Any]:
    invalid_fields = sorted(set(updates) - set(config.editable_fields))
    if invalid_fields:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported editable field(s)",
                "invalid_fields": invalid_fields,
                "editable_fields": list(config.editable_fields),
            },
        )
    return {key: (None if value == "" else value) for key, value in updates.items()}


def apply_field_updates(db: Session, config: EvidenceTable, evidence_id: str, updates: dict[str, Any], previous_row: dict[str, Any]) -> dict[str, Any]:
    if not updates:
        return {}
    assignments = []
    params: dict[str, Any] = {"evidence_id": evidence_id}
    field_changes: dict[str, dict[str, Any]] = {}
    for index, (field_name, value) in enumerate(updates.items()):
        param_name = f"field_value_{index}"
        assignments.append(f"{field_name} = :{param_name}")
        params[param_name] = value
        field_changes[field_name] = {
            "previous": previous_row.get(field_name),
            "new": serialize_value(value),
        }
    assignments.append("updated_at = now()")
    db.execute(
        text(
            f"""
            UPDATE {config.table_name}
            SET {", ".join(assignments)}
            WHERE {config.id_column} = CAST(:evidence_id AS uuid)
            """
        ),
        params,
    )
    return field_changes


def update_status(db: Session, config: EvidenceTable, evidence_id: str, new_status: str) -> None:
    db.execute(
        text(
            f"""
            UPDATE {config.table_name}
            SET curation_status = CAST(:new_status AS curation_status),
                updated_at = now()
            WHERE {config.id_column} = CAST(:evidence_id AS uuid)
            """
        ),
        {"evidence_id": evidence_id, "new_status": new_status},
    )


def insert_curation_event(
    db: Session,
    *,
    config: EvidenceTable,
    evidence_id: str,
    previous_status: str,
    new_status: str,
    curator: str,
    comment: str | None,
    field_changes: dict[str, Any],
) -> dict[str, Any]:
    row = db.execute(
        text(
            """
            INSERT INTO curation_event (
                evidence_table, evidence_id, previous_status, new_status,
                curator, curator_comment, field_changes
            )
            VALUES (
                :evidence_table, CAST(:evidence_id AS uuid),
                CAST(:previous_status AS curation_status),
                CAST(:new_status AS curation_status),
                :curator, :curator_comment, CAST(:field_changes AS jsonb)
            )
            RETURNING curation_event_id::text, curation_timestamp
            """
        ),
        {
            "evidence_table": config.table_name,
            "evidence_id": evidence_id,
            "previous_status": previous_status,
            "new_status": new_status,
            "curator": curator,
            "curator_comment": comment,
            "field_changes": json.dumps(field_changes),
        },
    ).mappings().one()
    return serialize_row(dict(row))


@router.get("/evidence/review-queue")
def review_queue(
    status: str = Query(default="needs_review"),
    evidence_type: str = "",
    source_id: str = "",
    compound: str = "",
    q: str = "",
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    created_from: date | None = None,
    created_to: date | None = None,
    duplicate_risk: str = Query(default="all"),
    missing_unit: bool | None = None,
    missing_context: bool | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            f"""
            WITH evidence AS (
                {reviewable_evidence_sql()}
            ),
            enriched AS (
                SELECT
                    *,
                    CASE
                        WHEN COUNT(*) OVER (
                            PARTITION BY LOWER(COALESCE(compound_name, '')),
                                         LOWER(COALESCE(parameter_name, '')),
                                         source_id
                        ) > 1
                        THEN 'high'
                        ELSE 'low'
                    END AS duplicate_risk,
                    COALESCE(NULLIF(BTRIM(COALESCE(unit, '')), ''), NULL) IS NULL AS missing_unit,
                    COALESCE(NULLIF(BTRIM(COALESCE(context, '')), ''), NULL) IS NULL AS missing_context
                FROM evidence
            )
            SELECT *
            FROM enriched
            WHERE (CAST(:status AS text) = 'all' OR curation_status = CAST(:status AS text))
              AND (CAST(:evidence_type AS text) = '' OR evidence_type = CAST(:evidence_type AS text))
              AND (CAST(:source_id AS text) = '' OR source_id = CAST(:source_id AS text))
              AND (CAST(:compound AS text) = '' OR compound_name ILIKE :compound_pattern)
              AND (
                  CAST(:q AS text) = ''
                  OR compound_name ILIKE :q_pattern
                  OR parameter_name ILIKE :q_pattern
                  OR evidence_text ILIKE :q_pattern
              )
              AND (
                  CAST(:min_confidence AS double precision) IS NULL
                  OR confidence_score >= CAST(:min_confidence AS double precision)
              )
              AND (
                  NULLIF(CAST(:created_from AS text), '') IS NULL
                  OR created_at::date >= CAST(NULLIF(CAST(:created_from AS text), '') AS date)
              )
              AND (
                  NULLIF(CAST(:created_to AS text), '') IS NULL
                  OR created_at::date <= CAST(NULLIF(CAST(:created_to AS text), '') AS date)
              )
              AND (CAST(:duplicate_risk AS text) = 'all' OR duplicate_risk = CAST(:duplicate_risk AS text))
              AND (
                  CAST(:missing_unit AS boolean) IS NULL
                  OR missing_unit = CAST(:missing_unit AS boolean)
              )
              AND (
                  CAST(:missing_context AS boolean) IS NULL
                  OR missing_context = CAST(:missing_context AS boolean)
              )
            ORDER BY
                CASE duplicate_risk WHEN 'high' THEN 0 ELSE 1 END,
                created_at ASC,
                confidence_score DESC NULLS LAST
            LIMIT :limit
            """
        ),
        {
            "status": status,
            "evidence_type": evidence_type,
            "source_id": source_id,
            "compound": compound,
            "compound_pattern": f"%{compound}%",
            "q": q,
            "q_pattern": f"%{q}%",
            "min_confidence": min_confidence,
            "created_from": created_from,
            "created_to": created_to,
            "duplicate_risk": duplicate_risk,
            "missing_unit": missing_unit,
            "missing_context": missing_context,
            "limit": limit,
        },
    ).mappings()
    return [serialize_row(dict(row)) for row in rows]


def fetch_processed_evidence_row(db: Session, evidence_id: str) -> dict[str, Any]:
    row = db.execute(
        text(
            """
            SELECT *
            FROM processed_evidence_candidate
            WHERE evidence_id = CAST(:evidence_id AS uuid)
            """
        ),
        {"evidence_id": evidence_id},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Evidence record not found")
    return serialize_row(dict(row))


def final_evidence_summary(db: Session, evidence_type: str, evidence_id: str) -> dict[str, Any]:
    row = db.execute(
        text(
            f"""
            WITH evidence AS (
                {serving_evidence_sql()}
            )
            SELECT *
            FROM evidence
            WHERE evidence_type = :evidence_type
              AND evidence_id = :evidence_id
            """
        ),
        {"evidence_type": evidence_type, "evidence_id": evidence_id},
    ).mappings().first()
    if row is None:
        fallback = db.execute(
            text(
                """
                SELECT
                    pec.evidence_type,
                    pec.evidence_id::text,
                    pec.compound_id::text,
                    compound.preferred_name AS compound_name,
                    pec.parameter AS parameter_name,
                    pec.value::float AS numeric_value,
                    pec.value_text,
                    pec.unit,
                    pec.canonical_value::float,
                    pec.canonical_unit,
                    NULL::text AS unit_dimension,
                    CONCAT_WS(' / ', pec.condition, pec.population, pec.route, pec.dose) AS context,
                    pec.source_id,
                    source.source_name,
                    pec.source_location,
                    pec.source_text AS evidence_text,
                    pec.extraction_method,
                    pec.extraction_confidence::float AS confidence_score,
                    pec.curation_status::text,
                    pec.created_at,
                    'candidate'::text AS evidence_layer
                FROM processed_evidence_candidate pec
                LEFT JOIN compound_identity compound ON compound.compound_id = pec.compound_id
                LEFT JOIN source_registry source ON source.source_id = pec.source_id
                WHERE pec.evidence_type = :evidence_type
                  AND pec.evidence_id = CAST(:evidence_id AS uuid)
                """
            ),
            {"evidence_type": evidence_type, "evidence_id": evidence_id},
        ).mappings().first()
        if fallback is None:
            raise HTTPException(status_code=404, detail="Evidence record not found")
        return serialize_row(dict(fallback))
    return serialize_row(dict(row))


def source_document_for_processed_row(db: Session, processed_row: dict[str, Any]) -> dict[str, Any] | None:
    parsed_document_id = processed_row.get("parsed_document_id")
    if not parsed_document_id:
        # Structured API fields and recovered table cells can have complete raw
        # provenance without an article-level parsed_document. Resolve their
        # actual imported source instead of inventing a parsed document.
        raw_document_id = processed_row.get("raw_document_id")
        raw_asset_id = processed_row.get("raw_asset_id")
        source_id = processed_row.get("source_id")
        if not raw_document_id or not raw_asset_id or not source_id:
            return None
        raw = db.execute(
            text("""
                SELECT raw_document_id::text, source_id, source_record_id,
                       source_url, raw_storage_path, raw_checksum, raw_mime_type
                FROM raw_document
                WHERE raw_document_id = CAST(:raw_document_id AS uuid)
                  AND source_record_id = :raw_asset_id AND source_id = :source_id
            """),
            {"raw_document_id": raw_document_id, "raw_asset_id": raw_asset_id, "source_id": source_id},
        ).mappings().first()
        if raw is None:
            return None
        return serialize_row({**dict(raw), "source_kind": "raw_asset",
                              "parsed_document_id": None,
                              "source_location": processed_row.get("source_location")})
    row = db.execute(
        text(
            """
            SELECT
                parsed.parsed_document_id::text,
                parsed.source_id,
                parsed.document_type,
                parsed.title,
                parsed.source_location,
                parsed.parsed_storage_path,
                parsed.parser_name,
                parsed.parser_version,
                raw.raw_document_id::text,
                raw.source_url,
                raw.raw_storage_path,
                raw.raw_checksum
            FROM parsed_document parsed
            LEFT JOIN raw_document raw ON raw.raw_document_id = parsed.raw_document_id
            WHERE parsed.parsed_document_id = CAST(:parsed_document_id AS uuid)
            """
        ),
        {"parsed_document_id": parsed_document_id},
    ).mappings().first()
    return serialize_row(dict(row)) if row else None


def curation_history_for_processed_evidence(db: Session, evidence_id: str) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT
                curation_event_id::text,
                evidence_table,
                evidence_id::text,
                previous_status::text,
                new_status::text,
                curator,
                curation_timestamp,
                curator_comment,
                field_changes
            FROM curation_event
            WHERE evidence_table = 'processed_evidence_candidate'
              AND evidence_id = CAST(:evidence_id AS uuid)
            ORDER BY curation_timestamp DESC
            """
        ),
        {"evidence_id": evidence_id},
    ).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


def related_final_evidence(db: Session, summary: dict[str, Any], evidence_id: str, limit: int = 10) -> list[dict[str, Any]]:
    compound_id = summary.get("compound_id")
    if not compound_id:
        return []
    rows = db.execute(
        text(
            f"""
            WITH evidence AS (
                {serving_evidence_sql()}
            )
            SELECT *
            FROM evidence
            WHERE compound_id = :compound_id
              AND evidence_id <> :evidence_id
            ORDER BY confidence_score DESC NULLS LAST, created_at DESC
            LIMIT :limit
            """
        ),
        {"compound_id": compound_id, "evidence_id": evidence_id, "limit": limit},
    ).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


def normalized_compound_for_summary(db: Session, summary: dict[str, Any]) -> dict[str, Any] | None:
    compound_id = summary.get("compound_id")
    if not compound_id:
        return None
    row = db.execute(
        text(
            """
            SELECT
                compound_id::text,
                preferred_name,
                normalized_name::text,
                synonyms,
                pubchem_cid,
                chembl_id,
                drugcentral_id,
                rxnorm_rxcui,
                inchi_key,
                cas_number,
                approval_status,
                source_id,
                source_location,
                confidence_score::float,
                curation_status::text,
                created_at,
                updated_at
            FROM compound_identity
            WHERE compound_id = CAST(:compound_id AS uuid)
            """
        ),
        {"compound_id": compound_id},
    ).mappings().first()
    return serialize_row(dict(row)) if row else None


def final_evidence_detail_response(db: Session, evidence_type: str, evidence_id: str) -> dict[str, Any]:
    processed_row = fetch_processed_evidence_row(db, evidence_id)
    if processed_row.get("evidence_type") != evidence_type:
        raise HTTPException(status_code=404, detail="Evidence record not found for requested type")
    summary = final_evidence_summary(db, evidence_type, evidence_id)
    return {
        "summary": summary,
        "extracted_fields": processed_row,
        "normalized_fields": normalized_compound_for_summary(db, summary),
        "source_document": source_document_for_processed_row(db, processed_row),
        "related_evidence": related_final_evidence(db, summary, evidence_id),
        "curation_history": curation_history_for_processed_evidence(db, evidence_id),
        "editable_fields": list(PROCESSED_EVIDENCE_EDITABLE_FIELDS),
        "allowed_actions": [action.value for action in CurationAction],
    }


def normalize_processed_field_updates(updates: dict[str, Any]) -> dict[str, Any]:
    invalid_fields = sorted(set(updates) - set(PROCESSED_EVIDENCE_EDITABLE_FIELDS))
    if invalid_fields:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported editable field(s)",
                "invalid_fields": invalid_fields,
                "editable_fields": list(PROCESSED_EVIDENCE_EDITABLE_FIELDS),
            },
        )
    return {key: (None if value == "" else value) for key, value in updates.items()}


def apply_processed_field_updates(db: Session, evidence_id: str, updates: dict[str, Any], previous_row: dict[str, Any]) -> dict[str, Any]:
    if not updates:
        return {}
    assignments = []
    params: dict[str, Any] = {"evidence_id": evidence_id}
    field_changes: dict[str, dict[str, Any]] = {}
    for index, (field_name, value) in enumerate(updates.items()):
        param_name = f"field_value_{index}"
        assignments.append(f"{field_name} = :{param_name}")
        params[param_name] = value
        field_changes[field_name] = {
            "previous": previous_row.get(field_name),
            "new": serialize_value(value),
        }
    assignments.append("updated_at = now()")
    db.execute(
        text(
            f"""
            UPDATE processed_evidence_candidate
            SET {", ".join(assignments)}
            WHERE evidence_id = CAST(:evidence_id AS uuid)
            """
        ),
        params,
    )
    return field_changes


def update_processed_status(db: Session, evidence_id: str, new_status: str) -> None:
    db.execute(
        text(
            """
            UPDATE processed_evidence_candidate
            SET curation_status = CAST(:new_status AS curation_status),
                updated_at = now()
            WHERE evidence_id = CAST(:evidence_id AS uuid)
            """
        ),
        {"evidence_id": evidence_id, "new_status": new_status},
    )


def insert_processed_curation_event(
    db: Session,
    *,
    evidence_id: str,
    previous_status: str,
    new_status: str,
    curator: str,
    comment: str | None,
    field_changes: dict[str, Any],
) -> dict[str, Any]:
    row = db.execute(
        text(
            """
            INSERT INTO curation_event (
                evidence_table, evidence_id, previous_status, new_status,
                curator, curator_comment, field_changes
            )
            VALUES (
                'processed_evidence_candidate', CAST(:evidence_id AS uuid),
                CAST(:previous_status AS curation_status),
                CAST(:new_status AS curation_status),
                :curator, :curator_comment, CAST(:field_changes AS jsonb)
            )
            RETURNING curation_event_id::text, curation_timestamp
            """
        ),
        {
            "evidence_id": evidence_id,
            "previous_status": previous_status,
            "new_status": new_status,
            "curator": curator,
            "curator_comment": comment,
            "field_changes": json.dumps(field_changes),
        },
    ).mappings().one()
    return serialize_row(dict(row))


def sync_curated_layer_memberships(db: Session, evidence_id: str, new_status: str) -> None:
    if new_status in {"reviewed_accepted", "approved_for_modeling"}:
        db.execute(
            text(
                """
                INSERT INTO evidence_layer_membership (
                    evidence_id, run_id, evidence_layer,
                    exclusion_reasons, review_blockers, semantic_group_ids,
                    conflict_group_ids, metadata_json, updated_at
                )
                SELECT
                    candidate.evidence_id,
                    candidate.run_id,
                    layer_name,
                    '[]'::jsonb,
                    '[]'::jsonb,
                    COALESCE(reviewable.semantic_group_ids, '[]'::jsonb),
                    COALESCE(reviewable.conflict_group_ids, '[]'::jsonb),
                    jsonb_build_object('created_by', 'curation_event'),
                    now()
                FROM processed_evidence_candidate candidate
                CROSS JOIN unnest(CAST(:layers AS text[])) AS layer_name
                LEFT JOIN evidence_layer_membership reviewable
                  ON reviewable.evidence_id = candidate.evidence_id
                 AND reviewable.evidence_layer = 'reviewable'
                WHERE candidate.evidence_id = CAST(:evidence_id AS uuid)
                ON CONFLICT (evidence_id, evidence_layer) DO UPDATE SET
                    updated_at = now(),
                    metadata_json = evidence_layer_membership.metadata_json || EXCLUDED.metadata_json
                """
            ),
            {
                "evidence_id": evidence_id,
                "layers": ["accepted", "modeling"] if new_status == "approved_for_modeling" else ["accepted"],
            },
        )
    if new_status != "approved_for_modeling":
        db.execute(
            text(
                """
                DELETE FROM evidence_layer_membership
                WHERE evidence_id = CAST(:evidence_id AS uuid)
                  AND evidence_layer = 'modeling'
                """
            ),
            {"evidence_id": evidence_id},
        )
    if new_status not in {"reviewed_accepted", "approved_for_modeling"}:
        db.execute(
            text(
                """
                DELETE FROM evidence_layer_membership
                WHERE evidence_id = CAST(:evidence_id AS uuid)
                  AND evidence_layer = 'accepted'
                """
            ),
            {"evidence_id": evidence_id},
        )


def validate_processed_modeling_approval(row: dict[str, Any]) -> None:
    has_value = _has_text(row.get("value")) or _has_text(row.get("value_text")) or _has_text(row.get("canonical_value"))
    has_context = any(_has_text(row.get(field_name)) for field_name in ("condition", "population", "route", "dose", "source_text"))
    issues: list[str] = []
    if not has_value:
        issues.append("approved_for_modeling requires a numeric value, value text, or canonical value")
    if (_has_text(row.get("value")) or _has_text(row.get("value_text"))) and not _has_text(row.get("unit")):
        issues.append("numeric or value-text evidence requires a unit before approved_for_modeling")
    if not has_context:
        issues.append("approved_for_modeling requires condition, population, route, dose, or source-text context")
    if issues:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Evidence is not ready for approved_for_modeling",
                "quality_issues": issues,
            },
        )


def curate_final_evidence(
    *,
    db: Session,
    evidence_type: str,
    evidence_id: str,
    request: CurationRequest,
) -> dict[str, Any]:
    previous_row = fetch_processed_evidence_row(db, evidence_id)
    if previous_row.get("evidence_type") != evidence_type:
        raise HTTPException(status_code=404, detail="Evidence record not found for requested type")
    previous_status = str(previous_row["curation_status"])
    normalized_updates = normalize_processed_field_updates(request.field_updates)
    new_status = target_status_for_action(request.action, previous_status)
    validate_transition(previous_status, new_status, request.action)

    with db.begin_nested():
        field_changes = apply_processed_field_updates(db, evidence_id, normalized_updates, previous_row)
        candidate_row = fetch_processed_evidence_row(db, evidence_id)
        if new_status == "approved_for_modeling":
            validate_processed_modeling_approval(candidate_row)
        if new_status != previous_status:
            update_processed_status(db, evidence_id, new_status)
        event = insert_processed_curation_event(
            db,
            evidence_id=evidence_id,
            previous_status=previous_status,
            new_status=new_status,
            curator=request.curator.strip(),
            comment=request.comment,
            field_changes={
                "action": request.action.value,
                "fields": field_changes,
            },
        )
        sync_curated_layer_memberships(db, evidence_id, new_status)
        record_user_activity(
            db,
            user_identifier=request.curator.strip(),
            activity_type="evidence_curated",
            entity_type="processed_evidence_candidate",
            entity_id=evidence_id,
            details={
                "action": request.action.value,
                "previous_status": previous_status,
                "new_status": new_status,
                "field_changes": field_changes,
            },
        )
    db.commit()

    updated_row = fetch_processed_evidence_row(db, evidence_id)
    return {
        "evidence_type": evidence_type,
        "evidence_table": "processed_evidence_candidate",
        "evidence_id": evidence_id,
        "previous_status": previous_status,
        "new_status": updated_row["curation_status"],
        "curation_event": event,
        "field_changes": field_changes,
    }


@router.get("/evidence/{evidence_type}/{evidence_id}")
def evidence_detail(
    evidence_type: str,
    evidence_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    if is_final_evidence_type(evidence_type):
        return final_evidence_detail_response(db, evidence_type, evidence_id)

    config = resolve_evidence_table(evidence_type)
    extracted_fields = fetch_evidence_row(db, config, evidence_id)
    summary = db.execute(
        text(
            f"""
            WITH evidence AS (
                {evidence_union_sql()}
            )
            SELECT *
            FROM evidence
            WHERE evidence_type = :evidence_type
              AND evidence_id = :evidence_id
            """
        ),
        {"evidence_type": config.evidence_type, "evidence_id": evidence_id},
    ).mappings().first()
    if summary is None:
        raise HTTPException(status_code=404, detail="Evidence record not found")

    document = None
    parsed_document_id = extracted_fields.get("parsed_document_id")
    if parsed_document_id:
        document = db.execute(
            text(
                """
                SELECT
                    parsed.parsed_document_id::text,
                    parsed.source_id,
                    parsed.document_type,
                    parsed.title,
                    parsed.source_location,
                    parsed.parsed_storage_path,
                    parsed.parser_name,
                    parsed.parser_version,
                    raw.raw_document_id::text,
                    raw.source_url,
                    raw.raw_storage_path,
                    raw.raw_checksum
                FROM parsed_document parsed
                LEFT JOIN raw_document raw ON raw.raw_document_id = parsed.raw_document_id
                WHERE parsed.parsed_document_id = CAST(:parsed_document_id AS uuid)
                """
            ),
            {"parsed_document_id": parsed_document_id},
        ).mappings().first()

    history = db.execute(
        text(
            """
            SELECT
                curation_event_id::text,
                evidence_table,
                evidence_id::text,
                previous_status::text,
                new_status::text,
                curator,
                curation_timestamp,
                curator_comment,
                field_changes
            FROM curation_event
            WHERE evidence_table = :evidence_table
              AND evidence_id = CAST(:evidence_id AS uuid)
            ORDER BY curation_timestamp DESC
            """
        ),
        {"evidence_table": config.table_name, "evidence_id": evidence_id},
    ).mappings().all()

    related: list[dict[str, Any]] = []
    normalized_fields = None
    compound_id = dict(summary).get("compound_id")
    if compound_id:
        normalized_fields_row = db.execute(
            text(
                """
                SELECT
                    compound_id::text,
                    preferred_name,
                    normalized_name::text,
                    synonyms,
                    pubchem_cid,
                    chembl_id,
                    drugcentral_id,
                    rxnorm_rxcui,
                    inchi_key,
                    cas_number,
                    approval_status,
                    source_id,
                    source_location,
                    confidence_score::float,
                    curation_status::text,
                    created_at,
                    updated_at
                FROM compound_identity
                WHERE compound_id = CAST(:compound_id AS uuid)
                """
            ),
            {"compound_id": compound_id},
        ).mappings().first()
        normalized_fields = serialize_row(dict(normalized_fields_row)) if normalized_fields_row else None
        related_rows = db.execute(
            text(
                f"""
                WITH evidence AS (
                    {evidence_union_sql()}
                )
                SELECT *
                FROM evidence
                WHERE compound_id = :compound_id
                  AND NOT (evidence_type = :evidence_type AND evidence_id = :evidence_id)
                ORDER BY confidence_score DESC NULLS LAST, created_at DESC
                LIMIT 10
                """
            ),
            {"compound_id": compound_id, "evidence_type": config.evidence_type, "evidence_id": evidence_id},
        ).mappings().all()
        related = [serialize_row(dict(row)) for row in related_rows]

    return {
        "summary": serialize_row(dict(summary)),
        "extracted_fields": extracted_fields,
        "normalized_fields": normalized_fields,
        "source_document": serialize_row(dict(document)) if document else None,
        "related_evidence": related,
        "curation_history": [serialize_row(dict(row)) for row in history],
        "editable_fields": list(config.editable_fields),
        "allowed_actions": [action.value for action in CurationAction],
    }


@router.post("/evidence/{evidence_type}/{evidence_id}/curate")
def curate_evidence(
    evidence_type: str,
    evidence_id: str,
    request: CurationRequest,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    if is_final_evidence_type(evidence_type):
        return curate_final_evidence(db=db, evidence_type=evidence_type, evidence_id=evidence_id, request=request)

    config = resolve_evidence_table(evidence_type)
    previous_row = fetch_evidence_row(db, config, evidence_id)
    previous_status = str(previous_row["curation_status"])
    normalized_updates = normalize_field_updates(config, request.field_updates)
    new_status = target_status_for_action(request.action, previous_status)
    validate_transition(previous_status, new_status, request.action)

    with db.begin_nested():
        field_changes = apply_field_updates(db, config, evidence_id, normalized_updates, previous_row)
        candidate_row = fetch_evidence_row(db, config, evidence_id)
        if new_status == "approved_for_modeling":
            validate_modeling_approval_quality(config, candidate_row)
        if new_status != previous_status:
            update_status(db, config, evidence_id, new_status)
        event = insert_curation_event(
            db,
            config=config,
            evidence_id=evidence_id,
            previous_status=previous_status,
            new_status=new_status,
            curator=request.curator.strip(),
            comment=request.comment,
            field_changes={
                "action": request.action.value,
                "fields": field_changes,
            },
        )
        record_user_activity(
            db,
            user_identifier=request.curator.strip(),
            activity_type="evidence_curated",
            entity_type=config.table_name,
            entity_id=evidence_id,
            details={
                "action": request.action.value,
                "previous_status": previous_status,
                "new_status": new_status,
                "field_changes": field_changes,
            },
        )
    db.commit()

    updated_row = fetch_evidence_row(db, config, evidence_id)
    return {
        "evidence_type": config.evidence_type,
        "evidence_table": config.table_name,
        "evidence_id": evidence_id,
        "previous_status": previous_status,
        "new_status": updated_row["curation_status"],
        "curation_event": event,
        "field_changes": field_changes,
    }
