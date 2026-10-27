from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.api.evidence_routes import serialize_row
from pbpk_kb.blueprints.schemas import BlueprintTemplate, PBPKBlueprint
from pbpk_kb.blueprints.service import (
    BlueprintServiceError,
    create_blueprint_for_request,
    get_blueprint_detail,
    list_blueprints,
)
from pbpk_kb.blueprints.validators import validate_blueprint
from pbpk_kb.recommendation.schemas import RecommendationRequest, RequirementProfile
from pbpk_kb.recommendation.public_model_service import (
    PublicModelRecommendationError,
)
from pbpk_kb.storage.db import get_db_session


blueprint_router = APIRouter(prefix="/blueprints", tags=["blueprints"])
job_router = APIRouter(prefix="/jobs", tags=["runtime-jobs"])
simulation_router = APIRouter(
    prefix="/simulation-runs",
    tags=["simulation-runs"],
)


def _require_blueprint(session: Session, blueprint_id: UUID) -> dict[str, Any]:
    row = get_blueprint_detail(session, str(blueprint_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Blueprint not found.")
    return row


@blueprint_router.post("")
def create_blueprint(
    request: RecommendationRequest,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    try:
        return create_blueprint_for_request(session=session, request=request)
    except (BlueprintServiceError, PublicModelRecommendationError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=422,
            detail={"status": "blueprint_blocked", "errors": [str(exc)]},
        ) from exc


@blueprint_router.get("")
def blueprint_list(
    compound: str = Query(default=""),
    session: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    return list_blueprints(session, compound=compound)


@blueprint_router.get("/{blueprint_id}")
def blueprint_detail(
    blueprint_id: UUID,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    return _require_blueprint(session, blueprint_id)


@blueprint_router.get("/{blueprint_id}/versions")
def blueprint_versions(
    blueprint_id: UUID,
    session: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    _require_blueprint(session, blueprint_id)
    rows = session.execute(
        text(
            """
            SELECT blueprint_version_id, blueprint_version,
                   parent_blueprint_checksum, status, blueprint_checksum,
                   source_location, created_at
            FROM blueprint_version
            WHERE blueprint_id = :blueprint_id
            ORDER BY created_at, blueprint_version
            """
        ),
        {"blueprint_id": str(blueprint_id)},
    ).mappings()
    return [serialize_row(dict(row)) for row in rows]


@blueprint_router.get("/{blueprint_id}/parameters")
def blueprint_parameters(
    blueprint_id: UUID,
    session: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    blueprint = _require_blueprint(session, blueprint_id)
    rows = session.execute(
        text(
            """
            SELECT parameter_id, semantic_id, parameter_name, origin,
                   value, initial_value, canonical_unit, blocking,
                   calibration_run_id, provenance_json, payload_json
            FROM blueprint_parameter
            WHERE blueprint_version_id = :version_id
            ORDER BY origin, parameter_name, parameter_id
            """
        ),
        {"version_id": blueprint["blueprint_version_id"]},
    ).mappings()
    return [serialize_row(dict(row)) for row in rows]


@blueprint_router.get("/{blueprint_id}/components")
def blueprint_components(
    blueprint_id: UUID,
    component_kind: Literal[
        "compartment",
        "process",
        "dosing_regimen",
        "simulation_scenario",
    ] | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=2000),
    session: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    blueprint = _require_blueprint(session, blueprint_id)
    rows = session.execute(
        text(
            """
            SELECT component_id, component_kind, semantic_id, payload_json,
                   source_location, source_checksum
            FROM blueprint_component
            WHERE blueprint_version_id = :version_id
              AND (
                CAST(:component_kind AS text) IS NULL
                OR component_kind = CAST(:component_kind AS text)
              )
            ORDER BY component_kind, component_id
            LIMIT :limit
            """
        ),
        {
            "version_id": blueprint["blueprint_version_id"],
            "component_kind": component_kind,
            "limit": limit,
        },
    ).mappings()
    return [serialize_row(dict(row)) for row in rows]


@blueprint_router.get("/{blueprint_id}/artifacts")
def blueprint_artifacts(
    blueprint_id: UUID,
    session: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    blueprint = _require_blueprint(session, blueprint_id)
    rows = session.execute(
        text(
            """
            SELECT blueprint_artifact_id, artifact_kind, relative_path,
                   sha256, byte_size, source_location, metadata_json
            FROM blueprint_artifact
            WHERE blueprint_version_id = :version_id
            ORDER BY artifact_kind, relative_path
            """
        ),
        {"version_id": blueprint["blueprint_version_id"]},
    ).mappings()
    return [serialize_row(dict(row)) for row in rows]


@blueprint_router.post("/{blueprint_id}/validate")
def validate_registered_blueprint(
    blueprint_id: UUID,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    row = _require_blueprint(session, blueprint_id)
    report = validate_blueprint(
        blueprint=PBPKBlueprint.model_validate(row["blueprint_json"]),
        template=BlueprintTemplate.model_validate(row["template_json"]),
        profile=RequirementProfile.model_validate(row["profile_json"]),
    )
    return {
        "status": "deterministic_validation_completed",
        "accepted": report.accepted,
        "compile_eligible": report.compile_eligible,
        "scientific_registration_allowed": bool(
            row["registration_allowed"]
        ),
        "scientific_status": row["scientific_status"],
        "scientific_blockers": row["blocker_codes"],
        "checks": report.checks,
        "counts": report.counts,
        "errors": report.errors,
        "stored_validation_report": row["validation_report_json"],
    }


def _blueprint_job(
    session: Session,
    *,
    blueprint_id: UUID,
    job_type: str,
) -> dict[str, Any] | None:
    if job_type == "compile":
        row = session.execute(
            text(
                """
                SELECT * FROM runtime_job
                WHERE job_type = 'compile'
                  AND output_json ->> 'input_blueprint_id' = :blueprint_id
                ORDER BY finished_at DESC NULLS LAST
                LIMIT 1
                """
            ),
            {"blueprint_id": str(blueprint_id)},
        ).mappings().first()
    else:
        row = session.execute(
            text(
                """
                SELECT DISTINCT job.*
                FROM runtime_job job
                JOIN simulation_run run ON run.job_id = job.job_id
                JOIN blueprint_version version
                  ON version.blueprint_version_id = run.blueprint_version_id
                WHERE version.blueprint_id = :blueprint_id
                  AND job.job_type = 'simulate'
                ORDER BY job.finished_at DESC NULLS LAST
                LIMIT 1
                """
            ),
            {"blueprint_id": str(blueprint_id)},
        ).mappings().first()
    return serialize_row(dict(row)) if row else None


@blueprint_router.post("/{blueprint_id}/compile")
def compile_registered_blueprint(
    blueprint_id: UUID,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    blueprint = _require_blueprint(session, blueprint_id)
    job = _blueprint_job(
        session,
        blueprint_id=blueprint_id,
        job_type="compile",
    )
    if job is None or job["status"] != "succeeded":
        raise HTTPException(
            status_code=409,
            detail="No checksum-locked successful compilation is registered.",
        )
    return {
        "status": "existing_compilation_returned",
        "blueprint_status": blueprint["current_status"],
        "job": job,
    }


@blueprint_router.post("/{blueprint_id}/run")
def run_registered_blueprint(
    blueprint_id: UUID,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    blueprint = _require_blueprint(session, blueprint_id)
    job = _blueprint_job(
        session,
        blueprint_id=blueprint_id,
        job_type="simulate",
    )
    if job is None or job["status"] != "succeeded":
        raise HTTPException(
            status_code=409,
            detail="No checksum-locked successful simulation is registered.",
        )
    runs = simulation_runs_for_blueprint(
        blueprint_id=blueprint_id,
        session=session,
    )
    return {
        "status": "existing_simulation_runs_returned",
        "blueprint_status": blueprint["current_status"],
        "job": job,
        "runs": runs,
    }


@blueprint_router.get("/{blueprint_id}/simulation-runs")
def simulation_runs_for_blueprint(
    blueprint_id: UUID,
    session: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    blueprint = _require_blueprint(session, blueprint_id)
    rows = session.execute(
        text(
            """
            SELECT run.simulation_run_id, run.scenario_id, run.run_ordinal,
                   run.status, run.runtime_json, run.result_manifest_sha256,
                   run.source_location, run.started_at, run.finished_at,
                   COUNT(output.simulation_output_id) AS output_count
            FROM simulation_run run
            LEFT JOIN simulation_output output
              ON output.simulation_run_id = run.simulation_run_id
            WHERE run.blueprint_version_id = :version_id
            GROUP BY run.simulation_run_id
            ORDER BY run.run_ordinal, run.scenario_id
            """
        ),
        {"version_id": blueprint["blueprint_version_id"]},
    ).mappings()
    return [serialize_row(dict(row)) for row in rows]


@blueprint_router.get("/{blueprint_id}/package")
def download_blueprint_package(
    blueprint_id: UUID,
    session: Session = Depends(get_db_session),
) -> FileResponse:
    blueprint = _require_blueprint(session, blueprint_id)
    root = Path(blueprint["asset_root"]).resolve()
    relative_path = (blueprint.get("package") or {}).get("relative_path")
    if not relative_path:
        raise HTTPException(status_code=404, detail="Blueprint package not found.")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="Unsafe package path.") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Blueprint package not found.")
    return FileResponse(
        path,
        media_type="application/zip",
        filename=path.name,
    )


@job_router.get("")
def list_runtime_jobs(
    status: str = Query(default=""),
    job_type: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=500),
    session: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT * FROM runtime_job
            WHERE (:status = '' OR status = :status)
              AND (:job_type = '' OR job_type = :job_type)
            ORDER BY created_at DESC, job_id
            LIMIT :limit
            """
        ),
        {"status": status, "job_type": job_type, "limit": limit},
    ).mappings()
    return [serialize_row(dict(row)) for row in rows]


@job_router.get("/{job_id}")
def runtime_job_detail(
    job_id: UUID,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    row = session.execute(
        text("SELECT * FROM runtime_job WHERE job_id = :job_id"),
        {"job_id": str(job_id)},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Runtime job not found.")
    value = serialize_row(dict(row))
    value["events"] = [
        serialize_row(dict(event))
        for event in session.execute(
            text(
                "SELECT * FROM runtime_job_event WHERE job_id = :job_id "
                "ORDER BY event_id"
            ),
            {"job_id": str(job_id)},
        ).mappings()
    ]
    return value


@job_router.post("/{job_id}/cancel")
def cancel_runtime_job(
    job_id: UUID,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    row = session.execute(
        text(
            """
            UPDATE runtime_job
            SET status = 'cancelled', finished_at = now()
            WHERE job_id = :job_id AND status IN ('queued', 'running')
            RETURNING *
            """
        ),
        {"job_id": str(job_id)},
    ).mappings().first()
    if row is None:
        raise HTTPException(
            status_code=409,
            detail="Only queued or running jobs may be cancelled.",
        )
    session.execute(
        text(
            """
            INSERT INTO runtime_job_event (
                job_id, event_type, status, message
            ) VALUES (:job_id, 'cancelled', 'cancelled', 'Cancelled by API request.')
            """
        ),
        {"job_id": str(job_id)},
    )
    session.commit()
    return serialize_row(dict(row))


@job_router.post("/{job_id}/retry")
def retry_runtime_job(
    job_id: UUID,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    row = session.execute(
        text(
            """
            UPDATE runtime_job
            SET status = 'queued', attempt = attempt + 1,
                progress_current = 0, started_at = NULL,
                heartbeat_at = NULL, finished_at = NULL,
                error_code = NULL, error_log_location = NULL
            WHERE job_id = :job_id
              AND status IN ('failed', 'timed_out')
              AND attempt < 2
            RETURNING *
            """
        ),
        {"job_id": str(job_id)},
    ).mappings().first()
    if row is None:
        raise HTTPException(
            status_code=409,
            detail="Only failed/timed-out first attempts may be retried.",
        )
    session.execute(
        text(
            """
            INSERT INTO runtime_job_event (
                job_id, event_type, status, message
            ) VALUES (:job_id, 'retry_queued', 'queued', 'Retry queued by API request.')
            """
        ),
        {"job_id": str(job_id)},
    )
    session.commit()
    return serialize_row(dict(row))


@simulation_router.get("/{simulation_run_id}")
def simulation_run_detail(
    simulation_run_id: UUID,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    row = session.execute(
        text(
            "SELECT * FROM simulation_run WHERE simulation_run_id = :run_id"
        ),
        {"run_id": str(simulation_run_id)},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Simulation run not found.")
    value = serialize_row(dict(row))
    value["outputs"] = [
        serialize_row(dict(output))
        for output in session.execute(
            text(
                """
                SELECT * FROM simulation_output
                WHERE simulation_run_id = :run_id
                ORDER BY output_kind, relative_path
                """
            ),
            {"run_id": str(simulation_run_id)},
        ).mappings()
    ]
    return value
