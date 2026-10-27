from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from pbpk_kb.config.stage1_sources import STAGE1_IMPLEMENTED_SOURCE_IDS
from pbpk_kb.config.settings import get_settings
from pbpk_kb.ui import api_client


# Streamlit keeps direct SQL only for dashboard-style aggregates that do not yet
# have equivalent API routes. User-facing search/profile paths go through FastAPI
# so the UI cannot drift from the API evidence schema.
EVIDENCE_TYPES = [
    "physicochemical",
    "adme",
    "clinical_pk",
    "ddi",
    "special_population",
    "tissue_expression",
    "public_pbpk_model",
]

STAGE1_SOURCE_IDS_SQL = ", ".join(f"'{source_id}'" for source_id in STAGE1_IMPLEMENTED_SOURCE_IDS)


@lru_cache(maxsize=4)
def get_engine(database_url: str | None = None) -> Engine:
    return create_engine(database_url or get_settings().database_url, pool_pre_ping=True)


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
    return value


def serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: serialize_value(value) for key, value in row.items()}


def fetch_all(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    with get_engine().connect() as connection:
        rows = connection.execute(text(sql), params or {}).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


def fetch_one(sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


def dashboard_metrics() -> dict[str, Any]:
    return fetch_one(
        f"""
        SELECT
            (SELECT COUNT(*) FROM source_registry WHERE source_id IN ({STAGE1_SOURCE_IDS_SQL})) AS source_count,
            (SELECT COUNT(*) FROM raw_document) AS raw_document_count,
            (SELECT COUNT(*) FROM parsed_document) AS parsed_document_count,
            (SELECT COUNT(*) FROM rag_retrieval_record
             WHERE evidence_layer = 'auto_validated') AS rag_record_count,
            (SELECT COUNT(*) FROM compound_identity) AS compound_count,
            (SELECT COUNT(*) FROM auto_validated_evidence) AS evidence_count,
            (SELECT COUNT(*) FROM public_pbpk_model_registry WHERE COALESCE(validation_status, '') <> 'model_mention_only') AS public_model_count,
            (SELECT COUNT(*) FROM auto_validated_evidence) AS auto_validated_count
        """
    ) or {}


def evidence_status_counts() -> list[dict[str, Any]]:
    return fetch_all(
        f"""
        SELECT curation_status::text AS curation_status, COUNT(*) AS record_count
        FROM auto_validated_evidence
        GROUP BY curation_status
        ORDER BY record_count DESC, curation_status
        """
    )


def latest_ingestion_runs(limit: int = 10) -> list[dict[str, Any]]:
    return fetch_all(
        f"""
        SELECT
            run.ingestion_run_id::text,
            run.external_run_id,
            run.source_id,
            source.source_name,
            run.status::text,
            run.started_at,
            run.completed_at,
            run.source_version,
            run.raw_record_count,
            run.parsed_record_count,
            run.evidence_record_count,
            run.log_path
        FROM ingestion_run run
        LEFT JOIN source_registry source ON source.source_id = run.source_id
        WHERE run.source_id IN ({STAGE1_SOURCE_IDS_SQL})
        ORDER BY run.completed_at DESC NULLS LAST, run.created_at DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )


def latest_export_jobs(limit: int = 10) -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT
            export_job_id::text,
            export_type,
            requested_by,
            status,
            output_path,
            included_curation_statuses::text[] AS included_curation_statuses,
            created_at,
            completed_at
        FROM export_job
        ORDER BY completed_at DESC NULLS LAST, created_at DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )


def source_registry_rows(family: str = "", enabled: str = "all") -> list[dict[str, Any]]:
    enabled_value = None
    if enabled == "enabled":
        enabled_value = True
    elif enabled == "disabled":
        enabled_value = False
    return api_client.list_sources(family=family, enabled=enabled_value)


def source_families() -> list[str]:
    return api_client.source_families()


def source_ids() -> list[str]:
    return api_client.source_ids()


def search_compounds(query: str, limit: int = 25) -> list[dict[str, Any]]:
    return api_client.search_compounds(query.strip(), limit=limit)


def get_compound(compound_id: str) -> dict[str, Any] | None:
    rows = search_compounds("", limit=200)
    return next((row for row in rows if row["compound_id"] == compound_id), None)


def evidence_search(
    *,
    compound: str = "",
    query: str = "",
    evidence_type: str = "",
    source_id: str = "",
    source_family: str = "",
    species: str = "",
    population: str = "",
    route: str = "",
    dose: str = "",
    enzyme_or_transporter: str = "",
    source_date_from: str = "",
    source_date_to: str = "",
    curation_status: str = "",
    min_confidence: float = 0.0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    return api_client.search_evidence(
        compound=compound.strip(),
        q=query.strip(),
        evidence_type=evidence_type,
        source_id=source_id,
        source_family=source_family,
        species=species.strip(),
        population=population.strip(),
        route=route.strip(),
        dose=dose.strip(),
        enzyme_or_transporter=enzyme_or_transporter.strip(),
        source_date_from=source_date_from,
        source_date_to=source_date_to,
        curation_status=curation_status,
        min_confidence=min_confidence,
        limit=limit,
    )


def compound_evidence(compound_id: str, evidence_type: str = "", limit: int = 200) -> list[dict[str, Any]]:
    return api_client.search_evidence(compound_id=compound_id, evidence_type=evidence_type, limit=limit)


def semantic_search(query: str, source_id: str = "", limit: int = 25) -> list[dict[str, Any]]:
    return api_client.semantic_search(query, source_id=source_id, limit=limit)


def compound_source_documents(compound_id: str) -> list[dict[str, Any]]:
    return fetch_all(
        """
        WITH linked_documents AS (
            SELECT parsed_document_id
            FROM auto_validated_evidence
            WHERE attributed_compound_id::text = :compound_id
        )
        SELECT DISTINCT
            parsed.parsed_document_id::text,
            parsed.source_id,
            source.source_name,
            parsed.document_type,
            parsed.title,
            parsed.source_location,
            parsed.parser_name,
            parsed.parser_version,
            parsed.created_at
        FROM linked_documents linked
        JOIN parsed_document parsed ON parsed.parsed_document_id = linked.parsed_document_id
        LEFT JOIN source_registry source ON source.source_id = parsed.source_id
        WHERE linked.parsed_document_id IS NOT NULL
        ORDER BY parsed.source_id, parsed.title
        """,
        {"compound_id": compound_id},
    )


def model_registry(
    *,
    compound: str = "",
    source_id: str = "",
    software: str = "",
    validation_status: str = "",
    reusable_only: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    return api_client.search_models(
        compound=compound.strip(),
        source_id=source_id,
        software=software.strip(),
        validation_status=validation_status,
        reusable_only=reusable_only,
        limit=limit,
    )


def validation_statuses() -> list[str]:
    rows = fetch_all(
        """
        SELECT DISTINCT validation_status
        FROM public_pbpk_model_registry
        WHERE validation_status IS NOT NULL
        ORDER BY validation_status
        """
    )
    return [row["validation_status"] for row in rows]


def curation_statuses() -> list[str]:
    rows = fetch_all(
        """
        SELECT DISTINCT curation_status::text AS curation_status
        FROM auto_validated_evidence
        ORDER BY curation_status
        """
    )
    return [row["curation_status"] for row in rows]


def format_list(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return str(value)


def evidence_value(row: dict[str, Any]) -> str:
    value = row.get("numeric_value")
    value_text = row.get("value_text")
    unit = row.get("unit")
    if value is not None:
        return f"{value} {unit}".strip() if unit else str(value)
    if value_text:
        return f"{value_text} {unit}".strip() if unit else str(value_text)
    return ""


def snippet(text_value: str | None, length: int = 220) -> str:
    text_value = " ".join((text_value or "").split())
    if len(text_value) <= length:
        return text_value
    return f"{text_value[: length - 3]}..."
