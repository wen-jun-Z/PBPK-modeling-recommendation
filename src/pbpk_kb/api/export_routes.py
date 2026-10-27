from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.api.evidence_routes import serialize_row
from pbpk_kb.exports.compound_profile_export import export_compound_profile
from pbpk_kb.exports.evidence_csv_export import export_evidence_csv
from pbpk_kb.exports.model_registry_export import export_model_registry_csv
from pbpk_kb.exports.stage1_report_export import export_stage1_summary
from pbpk_kb.storage.db import get_db_session


router = APIRouter(tags=["exports"])


class EvidenceExportRequest(BaseModel):
    compound: str = ""
    evidence_type: str = ""
    source_id: str = ""
    curation_status: str = ""
    evidence_layer: str = Field(
        default="auto_validated",
        pattern="^auto_validated$",
    )
    q: str = ""
    requested_by: str | None = None
    limit: int = Field(default=5000, ge=1, le=50000)


class CompoundProfileExportRequest(BaseModel):
    compound: str = Field(min_length=1)
    curation_status: str = ""
    evidence_layer: str = Field(
        default="auto_validated",
        pattern="^auto_validated$",
    )
    requested_by: str | None = None
    limit: int = Field(default=5000, ge=1, le=50000)


class ModelRegistryExportRequest(BaseModel):
    compound: str = ""
    source_id: str = ""
    validation_status: str = ""
    requested_by: str | None = None
    limit: int = Field(default=5000, ge=1, le=50000)


class Stage1SummaryExportRequest(BaseModel):
    requested_by: str | None = None


@router.post("/exports/evidence")
def create_evidence_export(request: EvidenceExportRequest, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    return export_evidence_csv(db, **request.model_dump())


@router.post("/exports/compound-profile")
def create_compound_profile_export(request: CompoundProfileExportRequest, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    return export_compound_profile(db, **request.model_dump())


@router.post("/exports/model-registry")
def create_model_registry_export(request: ModelRegistryExportRequest, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    return export_model_registry_csv(db, **request.model_dump())


@router.post("/exports/stage1-summary")
def create_stage1_summary_export(request: Stage1SummaryExportRequest, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    return export_stage1_summary(db, **request.model_dump())


@router.get("/exports/jobs")
def export_jobs(limit: int = Query(default=25, ge=1, le=200), db: Session = Depends(get_db_session)) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT
                export_job_id::text,
                export_type,
                filters,
                requested_by,
                status,
                output_path,
                source_versions,
                included_curation_statuses::text[] AS included_curation_statuses,
                created_at,
                completed_at
            FROM export_job
            ORDER BY created_at DESC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    ).mappings().all()
    return [serialize_row(dict(row)) for row in rows]
