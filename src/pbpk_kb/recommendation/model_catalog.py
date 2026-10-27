from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.model_assets.models import (
    ExecutionProbeStatus,
    LicenseDecision,
    LicenseStatus,
    ModelValidationRunLabel,
    OSPModelVersionAssetReference,
    OSPModelVersionSnapshot,
    ParseStatus,
)
from pbpk_kb.model_assets.validation_scope import build_model_validation_scope


MODEL_CATALOG_LOADER_VERSION = "pbpk-model-catalog-loader-1.1.0"

VALIDATION_SCOPE_SQL = text(
    """
    WITH selected_validation AS (
        SELECT *
        FROM (
            SELECT
                run.*,
                ROW_NUMBER() OVER (
                    PARTITION BY run.model_version_id
                    ORDER BY run.run_label_grade DESC,
                             run.validated_at DESC,
                             run.validation_run_id
                ) AS selection_rank
            FROM public_pbpk_model_validation_run run
        ) ranked
        WHERE ranked.selection_rank = 1
    )
    SELECT
        simulation.model_version_id::text AS model_version_id,
        simulation.external_simulation_id,
        simulation.simulation_name,
        simulation.protocol_role,
        simulation.species,
        simulation.populations,
        simulation.routes,
        simulation.formulations,
        simulation.validation_dataset_name,
        simulation.source_location,
        simulation.metadata_json,
        jsonb_build_object(
            'auc0_last_ratio', dataset.auc0_last_ratio,
            'cmax_ratio', dataset.cmax_ratio
        ) AS validation_metrics
    FROM public_pbpk_model_simulation simulation
    LEFT JOIN selected_validation selected
      ON selected.model_version_id = simulation.model_version_id
    LEFT JOIN public_pbpk_model_validation_dataset dataset
      ON dataset.validation_run_id = selected.validation_run_id
     AND dataset.simulation_sequence = simulation.protocol_sequence
    WHERE simulation.protocol_role = 'model_verification'
    ORDER BY simulation.model_version_id,
             simulation.protocol_sequence,
             simulation.model_simulation_id
    """
)

ASSET_SQL = text(
    """
    SELECT model_version_id::text AS model_version_id,
           asset_kind, relative_path, sha256, source_location
    FROM public_pbpk_model_asset
    ORDER BY model_version_id, asset_kind, relative_path, sha256
    """
)

LATEST_VALIDATION_SQL = text(
    """
    SELECT *
    FROM (
        SELECT
            run.*,
            ROW_NUMBER() OVER (
                PARTITION BY run.model_version_id
                ORDER BY run.run_label_grade DESC,
                         run.validated_at DESC,
                         run.validation_run_id
            ) AS selection_rank
        FROM public_pbpk_model_validation_run run
    ) ranked
    WHERE ranked.selection_rank = 1
    ORDER BY ranked.model_version_id
    """
)

VALIDATION_METRIC_COUNT_SQL = text(
    """
    WITH selected_validation AS (
        SELECT *
        FROM (
            SELECT
                run.validation_run_id,
                run.model_version_id,
                ROW_NUMBER() OVER (
                    PARTITION BY run.model_version_id
                    ORDER BY run.run_label_grade DESC,
                             run.validated_at DESC,
                             run.validation_run_id
                ) AS selection_rank
            FROM public_pbpk_model_validation_run run
        ) ranked
        WHERE ranked.selection_rank = 1
    )
    SELECT selected.model_version_id::text AS model_version_id,
           COUNT(metric.validation_metric_id)::integer AS metric_count
    FROM selected_validation selected
    LEFT JOIN public_pbpk_model_validation_metric metric
      ON metric.validation_run_id = selected.validation_run_id
    GROUP BY selected.model_version_id
    ORDER BY selected.model_version_id
    """
)


class ModelCatalogError(RuntimeError):
    pass


def _value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def load_registered_model_snapshots(
    session: Session,
    *,
    reusable_only: bool = False,
) -> list[OSPModelVersionSnapshot]:
    version_rows = session.execute(
        text(
            """
            SELECT
                version.*,
                registry.model_name,
                registry.external_model_id,
                registry.metadata_json AS registry_metadata
            FROM public_pbpk_model_version version
            JOIN public_pbpk_model_registry registry
              ON registry.model_id = version.model_id
            WHERE (:reusable_only IS FALSE OR version.reusable_candidate IS TRUE)
            ORDER BY lower(registry.model_name), version.repository_commit,
                     version.model_version_id
            """
        ),
        {"reusable_only": reusable_only},
    ).mappings().all()
    selected_version_ids = {
        str(row["model_version_id"]) for row in version_rows
    }
    assets_by_version: dict[str, list[OSPModelVersionAssetReference]] = (
        defaultdict(list)
    )
    for asset in session.execute(ASSET_SQL).mappings():
        version_id = str(asset["model_version_id"])
        if version_id not in selected_version_ids:
            continue
        assets_by_version[version_id].append(
            OSPModelVersionAssetReference(
                asset_kind=asset["asset_kind"],
                relative_path=asset["relative_path"],
                sha256=asset["sha256"],
                source_location=asset["source_location"],
            )
        )
    validation_by_version = {
        str(row["model_version_id"]): row
        for row in session.execute(LATEST_VALIDATION_SQL).mappings()
        if str(row["model_version_id"]) in selected_version_ids
    }
    metric_count_by_version = {
        str(row["model_version_id"]): int(row["metric_count"])
        for row in session.execute(VALIDATION_METRIC_COUNT_SQL).mappings()
        if str(row["model_version_id"]) in selected_version_ids
    }
    scope_rows_by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for simulation in session.execute(VALIDATION_SCOPE_SQL).mappings():
        version_id = str(simulation["model_version_id"])
        if version_id in selected_version_ids:
            scope_rows_by_version[version_id].append(dict(simulation))

    models: list[OSPModelVersionSnapshot] = []
    for row in version_rows:
        version_id = str(row["model_version_id"])
        metadata = row.get("metadata_json") or {}
        assets = assets_by_version.get(version_id, [])
        validation = validation_by_version.get(version_id)
        label = _validation_label(validation) if validation is not None else None
        validation_metric_count = metric_count_by_version.get(version_id, 0)
        validation_scope_rows = scope_rows_by_version.get(version_id, [])
        model_family = str(
            metadata.get("model_family")
            or (row.get("compounds") or [row["model_name"]])[0]
        )
        license_status = LicenseStatus(str(row["license_status"]))
        license = LicenseDecision(
            status=license_status,
            spdx_id=row.get("license_spdx_id"),
            source_location=row.get("license_source_location"),
            source_checksum=row.get("license_source_checksum"),
            reason_code=(
                "LICENSE_ALLOWED_BY_SPDX_POLICY"
                if license_status == LicenseStatus.ALLOWED
                else "LICENSE_STATUS_IMPORTED_FROM_CATALOG"
            ),
        )
        models.append(
            OSPModelVersionSnapshot(
                model_id=str(row["external_model_id"]),
                model_version_id=str(row["external_model_version_id"]),
                model_family=model_family,
                project_key=str(metadata["project_key"]),
                project_relative_path=str(row["project_relative_path"]),
                project_sha256=str(row["project_sha256"]),
                source_id=str(row["source_variant_id"]),
                repository_url=str(row["repository_url"]),
                repository_commit=str(row["repository_commit"]),
                software=str(row["software"]),
                software_version=str(row.get("software_version") or ""),
                license=license,
                static_parse_status=ParseStatus(
                    _value(row["static_parse_status"])
                ),
                execution_probe_status=ExecutionProbeStatus(
                    _value(row["execution_probe_status"])
                ),
                pkml_parse_status=ParseStatus(
                    _value(row["pkml_parse_status"])
                ),
                reusable_candidate=bool(row["reusable_candidate"]),
                compounds=list(row.get("compounds") or []),
                metabolites=list(metadata.get("metabolites") or []),
                species=list(row.get("species") or []),
                populations=list(row.get("populations") or []),
                routes=list(row.get("routes") or []),
                formulations=list(row.get("formulations") or []),
                enzymes=list(row.get("enzymes") or []),
                transporters=list(row.get("transporters") or []),
                compartments=list(row.get("compartments") or []),
                process_types=list(row.get("process_types") or []),
                output_names=list(row.get("output_names") or []),
                simulation_count=int(row["simulation_count"]),
                parameter_record_count=int(row["parameter_record_count"]),
                component_record_count=int(row["component_record_count"]),
                process_record_count=int(row["process_record_count"]),
                report_count=int(row["report_count"]),
                dataset_reference_count=int(row["dataset_reference_count"]),
                validation_metric_count=validation_metric_count,
                assets=assets,
                report_paths=list(metadata.get("report_paths") or []),
                validation_run_label=label,
                validation_scope=build_model_validation_scope(
                    validation_scope_rows
                ),
                blocking_reasons=list(metadata.get("blocking_reasons") or []),
            )
        )
    return models


def _validation_label(row: Any) -> ModelValidationRunLabel:
    metrics_path = row.get("validation_dataset_metrics_path")
    if not metrics_path:
        metrics_path = str(row["audit_report_path"]).replace(
            "snapshot_reuse_audit.json",
            "verification_dataset_metrics.jsonl",
        )
    return ModelValidationRunLabel(
        validation_run_id=str(row["external_validation_run_id"]),
        protocol_id=str(row["protocol_id"]),
        protocol_sha256=str(row["protocol_sha256"]),
        windows_manifest_sha256=str(row["windows_manifest_sha256"]),
        audit_report_sha256=str(row["audit_report_sha256"]),
        source_snapshot_sha256=str(row["source_snapshot_sha256"]),
        validation_grade=str(row["validation_grade"]),
        run_label_grade=int(row["run_label_grade"]),
        ranking_label=str(row["ranking_label"]),
        engineering_accepted=bool(row["engineering_accepted"]),
        scientific_accepted=bool(row["scientific_accepted"]),
        reusable_candidate=bool(row["reusable_candidate"]),
        validation_independence=str(
            (row.get("metadata_json") or {}).get(
                "validation_independence",
                "repository_preregistered_holdout",
            )
        ),
        verification_dataset_count=int(row["verification_dataset_count"]),
        pooled_observation_count=int(row["pooled_observation_count"]),
        pooled_gmfe=row.get("pooled_gmfe"),
        pooled_geometric_bias=row.get("pooled_geometric_bias"),
        pooled_two_fold_fraction=row.get("pooled_two_fold_fraction"),
        endpoint_scenario_pass_fraction=row.get(
            "endpoint_scenario_pass_fraction"
        ),
        protocol_path=str(row["protocol_path"]),
        windows_manifest_path=str(row["windows_manifest_path"]),
        audit_report_path=str(row["audit_report_path"]),
        validation_dataset_metrics_path=str(metrics_path),
        validated_at=row["validated_at"],
    )
