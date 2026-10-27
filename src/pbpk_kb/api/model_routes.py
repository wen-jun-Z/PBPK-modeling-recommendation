from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.api.evidence_routes import serialize_row
from pbpk_kb.storage.db import get_db_session


router = APIRouter(tags=["models"])


MODEL_SELECT_SQL = """
SELECT
    model.model_id::text,
    model.external_model_id,
    model.model_name,
    model.model_type,
    model.software,
    model.species,
    model.population,
    model.route,
    model.compound_id::text,
    compound.preferred_name AS compound_name,
    model.model_source,
    model.source_id,
    model.source_variant_id,
    source.source_name,
    model.repository_url,
    model.repository_commit,
    model.release_tag,
    model.license_spdx_id,
    model.model_file_path,
    model.parameter_file_path,
    model.evaluation_report_path,
    model.publication_id,
    model.validation_status,
    model.validation_grade,
    model.run_label_grade,
    model.ranking_label,
    model.reusable_candidate,
    model.model_checksum,
    model.source_snapshot_checksum,
    model.static_parse_status,
    model.execution_probe_status,
    model.latest_validation_run_id,
    model.simulation_count,
    model.parameter_record_count,
    model.component_record_count,
    model.process_record_count,
    model.curation_status::text,
    model.source_location,
    model.notes,
    model.metadata_json,
    model.created_at,
    model.updated_at
FROM public_pbpk_model_registry model
LEFT JOIN compound_identity compound ON compound.compound_id = model.compound_id
LEFT JOIN source_registry source ON source.source_id = model.source_id
"""


@router.get("/models/search")
def search_models(
    compound: str | None = None,
    source_id: str | None = None,
    software: str | None = None,
    validation_status: str | None = None,
    reusable_only: bool = False,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            MODEL_SELECT_SQL
            + """
            WHERE (
                CAST(:compound AS text) IS NULL
                OR compound.preferred_name ILIKE :compound_pattern
                OR model.model_name ILIKE :compound_pattern
            )
              AND (CAST(:source_id AS text) IS NULL OR model.source_id = CAST(:source_id AS text))
              AND (CAST(:software AS text) IS NULL OR model.software ILIKE :software_pattern)
              AND (
                CAST(:validation_status AS text) IS NOT NULL
                OR COALESCE(model.validation_status, '') <> 'model_mention_only'
              )
              AND (
                CAST(:validation_status AS text) IS NULL
                OR model.validation_status = CAST(:validation_status AS text)
              )
              AND (:reusable_only IS FALSE OR model.reusable_candidate IS TRUE)
            ORDER BY model.reusable_candidate DESC,
                     model.run_label_grade DESC,
                     model.model_name
            LIMIT :limit
            """
        ),
        {
            "compound": compound,
            "compound_pattern": f"%{compound or ''}%",
            "source_id": source_id,
            "software": software,
            "software_pattern": f"%{software or ''}%",
            "validation_status": validation_status,
            "reusable_only": reusable_only,
            "limit": limit,
        },
    ).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


@router.get("/models/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    model = db.execute(
        text(MODEL_SELECT_SQL + "WHERE model.model_id = CAST(:model_id AS uuid)"),
        {"model_id": model_id},
    ).mappings().first()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    parameters = db.execute(
        text(
            """
            SELECT
                model_parameter_evidence_id::text,
                model_id::text,
                compound_id::text,
                parameter_name,
                parameter_value::float,
                parameter_value_text,
                unit,
                parameter_context,
                value_origin,
                evidence_origin,
                source_id,
                source_location,
                evidence_text,
                extraction_method,
                extractor_version,
                confidence_score::float,
                curation_status::text,
                created_at,
                updated_at
            FROM model_parameter_evidence
            WHERE model_id = CAST(:model_id AS uuid)
            ORDER BY parameter_name
            """
        ),
        {"model_id": model_id},
    ).mappings().all()

    versions = db.execute(
        text(
            """
            SELECT
                model_version_id::text,
                external_model_version_id,
                version_label,
                repository_url,
                repository_commit,
                project_relative_path,
                project_sha256,
                source_snapshot_sha256,
                software,
                software_version,
                license_status,
                license_spdx_id,
                license_source_location,
                license_source_checksum,
                static_parse_status,
                execution_probe_status,
                pkml_parse_status,
                reusable_candidate,
                compounds,
                species,
                populations,
                routes,
                formulations,
                enzymes,
                transporters,
                compartments,
                process_types,
                output_names,
                simulation_count,
                parameter_record_count,
                component_record_count,
                process_record_count,
                report_count,
                dataset_reference_count,
                source_id,
                source_variant_id,
                source_location,
                metadata_json,
                created_at,
                updated_at
            FROM public_pbpk_model_version
            WHERE model_id = CAST(:model_id AS uuid)
            ORDER BY repository_commit, model_version_id
            """
        ),
        {"model_id": model_id},
    ).mappings().all()
    assets = db.execute(
        text(
            """
            SELECT
                asset.model_asset_id::text,
                asset.model_version_id::text,
                asset.asset_kind,
                asset.relative_path,
                asset.sha256,
                asset.source_location,
                asset.metadata_json
            FROM public_pbpk_model_asset asset
            JOIN public_pbpk_model_version version
              ON version.model_version_id = asset.model_version_id
            WHERE version.model_id = CAST(:model_id AS uuid)
            ORDER BY asset.asset_kind, asset.relative_path
            """
        ),
        {"model_id": model_id},
    ).mappings().all()
    simulations = db.execute(
        text(
            """
            SELECT
                simulation.model_simulation_id::text,
                simulation.model_version_id::text,
                simulation.external_simulation_id,
                simulation.protocol_sequence,
                simulation.simulation_name,
                simulation.protocol_role,
                simulation.solver,
                simulation.species,
                simulation.populations,
                simulation.routes,
                simulation.formulations,
                simulation.validation_dataset_name,
                simulation.pkml_relative_path,
                simulation.pkml_sha256,
                simulation.result_relative_path,
                simulation.result_sha256,
                simulation.parameter_count,
                simulation.component_count,
                simulation.process_count,
                simulation.administration_count,
                simulation.source_location,
                simulation.metadata_json
            FROM public_pbpk_model_simulation simulation
            JOIN public_pbpk_model_version version
              ON version.model_version_id = simulation.model_version_id
            WHERE version.model_id = CAST(:model_id AS uuid)
            ORDER BY simulation.protocol_sequence
            """
        ),
        {"model_id": model_id},
    ).mappings().all()
    validation_runs = db.execute(
        text(
            """
            SELECT
                run.validation_run_id::text,
                run.external_validation_run_id,
                run.model_version_id::text,
                run.protocol_id,
                run.protocol_path,
                run.protocol_sha256,
                run.windows_manifest_path,
                run.windows_manifest_sha256,
                run.audit_report_path,
                run.audit_report_sha256,
                run.validation_dataset_metrics_path,
                run.source_snapshot_sha256,
                run.validation_grade,
                run.run_label_grade,
                run.ranking_label,
                run.reusable_candidate,
                run.engineering_accepted,
                run.scientific_accepted,
                run.verification_dataset_count,
                run.pooled_observation_count,
                run.pooled_gmfe,
                run.pooled_geometric_bias,
                run.pooled_two_fold_fraction,
                run.endpoint_scenario_pass_fraction,
                run.validated_at,
                run.source_location,
                run.metadata_json
            FROM public_pbpk_model_validation_run run
            JOIN public_pbpk_model_version version
              ON version.model_version_id = run.model_version_id
            WHERE version.model_id = CAST(:model_id AS uuid)
            ORDER BY run.run_label_grade DESC, run.validated_at DESC
            """
        ),
        {"model_id": model_id},
    ).mappings().all()
    datasets = db.execute(
        text(
            """
            SELECT
                dataset.validation_dataset_id::text,
                dataset.validation_run_id::text,
                dataset.simulation_sequence,
                dataset.dataset_name,
                dataset.simulation_name,
                dataset.reference,
                dataset.source_location,
                dataset.prediction_unit,
                dataset.observation_count,
                dataset.positive_pair_count,
                dataset.excluded_nonpositive_observation_count,
                dataset.auc0_last_ratio,
                dataset.cmax_ratio,
                dataset.gmfe,
                dataset.geometric_bias,
                dataset.log_rmse,
                dataset.two_fold_fraction,
                dataset.dataset_guard_passed,
                dataset.endpoint_evaluable,
                dataset.endpoint_passed,
                dataset.metadata_json
            FROM public_pbpk_model_validation_dataset dataset
            JOIN public_pbpk_model_validation_run run
              ON run.validation_run_id = dataset.validation_run_id
            JOIN public_pbpk_model_version version
              ON version.model_version_id = run.model_version_id
            WHERE version.model_id = CAST(:model_id AS uuid)
            ORDER BY dataset.simulation_sequence, dataset.dataset_name
            """
        ),
        {"model_id": model_id},
    ).mappings().all()
    metrics = db.execute(
        text(
            """
            SELECT
                metric.validation_metric_id::text,
                metric.validation_run_id::text,
                metric.validation_dataset_id::text,
                dataset.dataset_name,
                metric.metric_scope,
                metric.metric_name,
                metric.metric_value,
                metric.passed,
                metric.source_location,
                metric.metadata_json
            FROM public_pbpk_model_validation_metric metric
            JOIN public_pbpk_model_validation_run run
              ON run.validation_run_id = metric.validation_run_id
            JOIN public_pbpk_model_version version
              ON version.model_version_id = run.model_version_id
            LEFT JOIN public_pbpk_model_validation_dataset dataset
              ON dataset.validation_dataset_id = metric.validation_dataset_id
            WHERE version.model_id = CAST(:model_id AS uuid)
            ORDER BY metric.metric_scope DESC,
                     dataset.simulation_sequence NULLS FIRST,
                     metric.metric_name
            """
        ),
        {"model_id": model_id},
    ).mappings().all()

    serialized_validation_runs = []
    for row in validation_runs:
        value = serialize_row(dict(row))
        metadata = value.get("metadata_json") or {}
        value["validation_independence"] = metadata.get(
            "validation_independence",
            "repository_preregistered_holdout",
        )
        serialized_validation_runs.append(value)

    return {
        "model": serialize_row(dict(model)),
        "parameters": [serialize_row(dict(row)) for row in parameters],
        "versions": [serialize_row(dict(row)) for row in versions],
        "assets": [serialize_row(dict(row)) for row in assets],
        "simulations": [serialize_row(dict(row)) for row in simulations],
        "validation_runs": serialized_validation_runs,
        "validation_datasets": [
            serialize_row(dict(row)) for row in datasets
        ],
        "validation_metrics": [
            serialize_row(dict(row)) for row in metrics
        ],
    }
