from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.api.evidence_routes import serialize_row
from pbpk_kb.config.stage1_sources import STAGE1_IMPLEMENTED_SOURCE_IDS
from pbpk_kb.connectors.registry import CONNECTOR_REGISTRY
from pbpk_kb.connectors.run_connectors import run_sources
from pbpk_kb.storage.activity_log import record_user_activity
from pbpk_kb.storage.db import get_db_session


router = APIRouter(tags=["sources"])


class SourceUpdateRequest(BaseModel):
    enabled: bool | None = None
    owner_curator: str | None = None
    requested_by: str = "stage1_admin"


class IngestionRetryRequest(BaseModel):
    query: str = "midazolam"
    retmax: int = 1
    allow_network: bool = False
    requested_by: str = "stage1_admin"


SOURCE_SELECT_SQL = """
WITH source_aliases(child_source_id, parent_source_id) AS (
    VALUES
        ('pmc_oa'::text, 'pubmed'::text),
        ('openfda_drug_label'::text, 'drugs_fda'::text)
),
raw_counts AS (
    SELECT COALESCE(alias.parent_source_id, raw.source_id) AS source_id,
           COUNT(*) AS raw_document_count
    FROM raw_document raw
    LEFT JOIN source_aliases alias ON alias.child_source_id = raw.source_id
    GROUP BY COALESCE(alias.parent_source_id, raw.source_id)
),
parsed_counts AS (
    SELECT COALESCE(alias.parent_source_id, parsed.source_id) AS source_id,
           COUNT(*) AS parsed_document_count
    FROM parsed_document parsed
    LEFT JOIN source_aliases alias ON alias.child_source_id = parsed.source_id
    GROUP BY COALESCE(alias.parent_source_id, parsed.source_id)
),
chunk_counts AS (
    SELECT COALESCE(alias.parent_source_id, record.source_id) AS source_id,
           COUNT(*) AS chunk_count
    FROM rag_retrieval_record record
    LEFT JOIN source_aliases alias ON alias.child_source_id = record.source_id
    WHERE record.evidence_layer = 'auto_validated'
    GROUP BY COALESCE(alias.parent_source_id, record.source_id)
),
evidence_counts AS (
    SELECT COALESCE(alias.parent_source_id, evidence.source_id) AS source_id,
           COUNT(*) AS evidence_count
    FROM auto_validated_evidence evidence
    LEFT JOIN source_aliases alias ON alias.child_source_id = evidence.source_id
    GROUP BY COALESCE(alias.parent_source_id, evidence.source_id)
),
model_counts AS (
    SELECT source_id, COUNT(*) AS model_count
    FROM public_pbpk_model_registry
    WHERE COALESCE(validation_status, '') <> 'model_mention_only'
    GROUP BY source_id
),
latest_ingestion AS (
    SELECT DISTINCT ON (source_id)
        source_id,
        ingestion_run_id::text,
        external_run_id,
        status::text AS latest_ingestion_status,
        started_at AS latest_ingestion_started_at,
        completed_at AS latest_ingestion_completed_at,
        raw_record_count AS latest_ingestion_raw_record_count,
        parsed_record_count AS latest_ingestion_parsed_record_count,
        evidence_record_count AS latest_ingestion_evidence_record_count,
        log_path AS latest_ingestion_log_path
    FROM ingestion_run
    ORDER BY source_id, completed_at DESC NULLS LAST, created_at DESC
)
SELECT
    s.source_id,
    s.source_name,
    s.source_family,
    s.access_url,
    s.api_url,
    s.access_method,
    s.expected_format,
    s.refresh_frequency,
    s.license_notes,
    s.connector,
    s.parser,
    s.entities,
    s.enabled,
    s.source_version,
    s.checksum,
    s.owner_curator,
    s.created_at,
    s.updated_at,
    COALESCE(r.raw_document_count, 0) AS raw_document_count,
    COALESCE(p.parsed_document_count, 0) AS parsed_document_count,
    COALESCE(c.chunk_count, 0) AS chunk_count,
    COALESCE(e.evidence_count, 0) AS evidence_count,
    COALESCE(m.model_count, 0) AS model_count,
    latest.ingestion_run_id AS latest_ingestion_run_id,
    latest.external_run_id AS latest_external_run_id,
    latest.latest_ingestion_status,
    latest.latest_ingestion_started_at,
    latest.latest_ingestion_completed_at,
    latest.latest_ingestion_raw_record_count,
    latest.latest_ingestion_parsed_record_count,
    latest.latest_ingestion_evidence_record_count,
    latest.latest_ingestion_log_path
FROM source_registry s
LEFT JOIN raw_counts r ON r.source_id = s.source_id
LEFT JOIN parsed_counts p ON p.source_id = s.source_id
LEFT JOIN chunk_counts c ON c.source_id = s.source_id
LEFT JOIN evidence_counts e ON e.source_id = s.source_id
LEFT JOIN model_counts m ON m.source_id = s.source_id
LEFT JOIN latest_ingestion latest ON latest.source_id = s.source_id
"""


@router.get("/sources")
def list_sources(
    family: str | None = None,
    enabled: bool | None = None,
    include_planned: bool = False,
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            SOURCE_SELECT_SQL
            + """
            WHERE (CAST(:family AS text) IS NULL OR s.source_family = CAST(:family AS text))
              AND (CAST(:enabled AS boolean) IS NULL OR s.enabled = CAST(:enabled AS boolean))
              AND (
                  CAST(:include_planned AS boolean)
                  OR s.source_id = ANY(CAST(:stage1_source_ids AS text[]))
              )
            ORDER BY s.source_family, s.source_id
            """
        ),
        {
            "family": family,
            "enabled": enabled,
            "include_planned": include_planned,
            "stage1_source_ids": list(STAGE1_IMPLEMENTED_SOURCE_IDS),
        },
    ).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


@router.get("/sources/{source_id}")
def get_source(source_id: str, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    row = db.execute(
        text(SOURCE_SELECT_SQL + "WHERE s.source_id = :source_id"),
        {"source_id": source_id},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return serialize_row(dict(row))


@router.patch("/sources/{source_id}")
def update_source(
    source_id: str,
    request: SourceUpdateRequest,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    current = db.execute(
        text("SELECT source_id FROM source_registry WHERE source_id = :source_id"),
        {"source_id": source_id},
    ).mappings().first()
    if current is None:
        raise HTTPException(status_code=404, detail="Source not found")

    db.execute(
        text(
            """
            UPDATE source_registry
            SET enabled = COALESCE(CAST(:enabled AS boolean), enabled),
                owner_curator = COALESCE(NULLIF(CAST(:owner_curator AS text), ''), owner_curator),
                updated_at = now()
            WHERE source_id = :source_id
            """
        ),
        {
            "source_id": source_id,
            "enabled": request.enabled,
            "owner_curator": request.owner_curator,
        },
    )
    record_user_activity(
        db,
        user_identifier=request.requested_by,
        activity_type="source_registry_updated",
        entity_type="source_registry",
        details={"source_id": source_id, "enabled": request.enabled, "owner_curator": request.owner_curator},
    )
    db.commit()
    return get_source(source_id, db)


@router.get("/ingestion-runs")
def list_ingestion_runs(
    source_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT
                ingestion_run_id::text,
                external_run_id,
                source_id,
                status::text,
                started_at,
                completed_at,
                source_version,
                raw_record_count,
                parsed_record_count,
                evidence_record_count,
                raw_checksum,
                log_path,
                error_summary,
                created_at
            FROM ingestion_run
            WHERE (CAST(:source_id AS text) IS NULL OR source_id = CAST(:source_id AS text))
              AND (CAST(:status AS text) IS NULL OR status::text = CAST(:status AS text))
            ORDER BY completed_at DESC NULLS LAST, created_at DESC
            LIMIT :limit
            """
        ),
        {"source_id": source_id, "status": status, "limit": limit},
    ).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


@router.get("/ingestion-runs/{ingestion_run_id}/log")
def ingestion_run_log(
    ingestion_run_id: str,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    row = db.execute(
        text(
            """
            SELECT ingestion_run_id::text, source_id, log_path, error_summary
            FROM ingestion_run
            WHERE ingestion_run_id = CAST(:ingestion_run_id AS uuid)
            """
        ),
        {"ingestion_run_id": ingestion_run_id},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Ingestion run not found")
    payload = serialize_row(dict(row))
    log_path = payload.get("log_path")
    if log_path:
        path = Path(log_path)
        if path.exists():
            try:
                payload["log"] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload["log"] = {"raw_text": path.read_text(encoding="utf-8")}
        else:
            payload["log_missing"] = True
    return payload


@router.post("/ingestion-runs/{ingestion_run_id}/retry")
def retry_ingestion_run(
    ingestion_run_id: str,
    request: IngestionRetryRequest,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    row = db.execute(
        text(
            """
            SELECT source_id
            FROM ingestion_run
            WHERE ingestion_run_id = CAST(:ingestion_run_id AS uuid)
            """
        ),
        {"ingestion_run_id": ingestion_run_id},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Ingestion run not found")

    source_id = row["source_id"]
    if source_id not in CONNECTOR_REGISTRY:
        raise HTTPException(status_code=404, detail=f"No connector registered for source: {source_id}")
    results = run_sources(
        sources=[source_id],
        query=request.query,
        retmax=request.retmax,
        output_root=".",
        allow_network=request.allow_network,
        sync_source_registry=True,
    )
    record_user_activity(
        db,
        user_identifier=request.requested_by,
        activity_type="source_ingestion_retried",
        entity_type="ingestion_run",
        entity_id=ingestion_run_id,
        details={"source_id": source_id, "query": request.query, "retmax": request.retmax},
    )
    db.commit()
    return {"source_id": source_id, "retried_from_ingestion_run_id": ingestion_run_id, "results": results}


@router.post("/sources/{source_id}/ingest")
def ingest_source(
    source_id: str,
    query: str = "midazolam",
    retmax: int = Query(default=1, ge=1, le=20),
    allow_network: bool = False,
    requested_by: str = "stage1_admin",
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    if source_id not in CONNECTOR_REGISTRY:
        raise HTTPException(status_code=404, detail=f"No connector registered for source: {source_id}")
    results = run_sources(
        sources=[source_id],
        query=query,
        retmax=retmax,
        output_root=".",
        allow_network=allow_network,
        sync_source_registry=True,
    )
    record_user_activity(
        db,
        user_identifier=requested_by,
        activity_type="source_ingestion_triggered",
        entity_type="source_registry",
        details={"source_id": source_id, "query": query, "retmax": retmax, "allow_network": allow_network},
    )
    db.commit()
    return {"source_id": source_id, "results": results}
