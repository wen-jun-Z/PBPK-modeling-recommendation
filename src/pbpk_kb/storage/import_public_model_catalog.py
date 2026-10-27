from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import Connection, text

from pbpk_kb.model_assets.models import OSPModelVersionSnapshot
from pbpk_kb.storage.db import create_db_engine


IMPORTER_VERSION = "public-model-catalog-importer-1.0.0"


class PublicModelCatalogImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublicModelCatalogBundle:
    root: Path
    summary: dict[str, Any]
    gate1_audit: dict[str, Any]
    source_snapshot_sha256: str
    models: tuple[OSPModelVersionSnapshot, ...]
    simulations: tuple[dict[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_uuid(kind: str, *values: object) -> UUID:
    identity = "|".join([kind, *(str(value) for value in values)])
    return uuid5(NAMESPACE_URL, f"pbpk-stage1:{identity}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicModelCatalogImportError(
            f"Expected JSON object: {path}"
        )
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise PublicModelCatalogImportError(
            f"Expected JSONL objects: {path}"
        )
    return rows


def _require_equal(
    observed: object,
    expected: object,
    message: str,
) -> None:
    if observed != expected:
        raise PublicModelCatalogImportError(
            f"{message}: expected {expected!r}, observed {observed!r}"
        )


def load_public_model_catalog_bundle(
    *,
    bundle_directory: Path,
    gate1_audit_path: Path,
) -> PublicModelCatalogBundle:
    root = bundle_directory.resolve()
    summary_path = root / "summary.json"
    model_path = root / "model_versions.jsonl"
    simulation_path = root / "simulations.jsonl"
    for path in (summary_path, model_path, simulation_path, gate1_audit_path):
        if not path.is_file():
            raise PublicModelCatalogImportError(
                f"Required catalog artifact is missing: {path}"
            )

    summary = _read_json(summary_path)
    gate1_audit = _read_json(gate1_audit_path)
    _require_equal(
        summary.get("gate1_model_version_acceptance"),
        True,
        "Model-version build acceptance",
    )
    _require_equal(
        gate1_audit.get("accepted"),
        True,
        "Gate 1 audit acceptance",
    )
    _require_equal(
        gate1_audit.get("input_checksums", {}).get(
            "model_version_summary"
        ),
        sha256_file(summary_path),
        "Gate 1 model-version summary checksum",
    )
    output_checksums = summary.get("output_checksums") or {}
    for filename, path in (
        ("model_versions.jsonl", model_path),
        ("simulations.jsonl", simulation_path),
    ):
        expected = output_checksums.get(filename)
        if not expected:
            raise PublicModelCatalogImportError(
                f"Catalog checksum is absent for {filename}."
            )
        _require_equal(
            sha256_file(path),
            expected,
            f"Catalog checksum for {filename}",
        )

    model_rows = _read_jsonl(model_path)
    simulation_rows = _read_jsonl(simulation_path)
    models = tuple(
        OSPModelVersionSnapshot.model_validate(row)
        for row in model_rows
    )
    _require_equal(
        len(models),
        int(summary.get("model_version_count") or -1),
        "Model-version count",
    )
    _require_equal(
        len(simulation_rows),
        int(summary.get("simulation_count") or -1),
        "Simulation count",
    )
    if len({model.model_version_id for model in models}) != len(models):
        raise PublicModelCatalogImportError(
            "External model-version IDs are not unique."
        )
    if len({model.project_key for model in models}) != len(models):
        raise PublicModelCatalogImportError(
            "Project keys are not unique across model versions."
        )

    simulation_counts: dict[str, int] = defaultdict(int)
    for row in simulation_rows:
        project_key = str(row.get("project_key") or "")
        simulation_counts[project_key] += 1
    expected_project_keys = {model.project_key for model in models}
    if set(simulation_counts) != expected_project_keys:
        raise PublicModelCatalogImportError(
            "Simulation and model-version project-key sets differ."
        )
    for model in models:
        _require_equal(
            simulation_counts[model.project_key],
            model.simulation_count,
            f"Simulation count for {model.project_relative_path}",
        )
        if model.repository_commit != summary.get("repository_commit"):
            raise PublicModelCatalogImportError(
                "A model version uses a different repository commit."
            )

    return PublicModelCatalogBundle(
        root=root,
        summary=summary,
        gate1_audit=gate1_audit,
        source_snapshot_sha256=sha256_file(summary_path),
        models=models,
        simulations=tuple(simulation_rows),
    )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _normalized_name(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def _compound_ids(connection: Connection) -> dict[str, UUID]:
    rows = connection.execute(
        text(
            """
            SELECT compound_id, preferred_name::text, normalized_name::text,
                   synonyms
            FROM compound_identity
            ORDER BY compound_id
            """
        )
    ).mappings()
    candidates: dict[str, set[UUID]] = defaultdict(set)
    for row in rows:
        names = [row.get("preferred_name"), row.get("normalized_name")]
        names.extend(row.get("synonyms") or [])
        for name in names:
            if name:
                candidates[_normalized_name(str(name))].add(
                    UUID(str(row["compound_id"]))
                )
    return {
        name: next(iter(values))
        for name, values in candidates.items()
        if len(values) == 1
    }


def _require_source(connection: Connection, source_id: str) -> None:
    value = connection.execute(
        text(
            "SELECT source_id FROM source_registry "
            "WHERE source_id = :source_id"
        ),
        {"source_id": source_id},
    ).scalar_one_or_none()
    if value is None:
        raise PublicModelCatalogImportError(
            f"Required source registry entry is absent: {source_id}"
        )


def _delete_stale_catalog_stubs(
    connection: Connection,
    *,
    source_id: str,
) -> int:
    result = connection.execute(
        text(
            """
            DELETE FROM public_pbpk_model_registry registry
            WHERE registry.source_id = :source_id
              AND registry.external_model_id IS NULL
              AND registry.validation_status IN (
                  'downloadable_public_model',
                  'metadata_indexed',
                  'model_mention_only'
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM public_pbpk_model_version version
                  WHERE version.model_id = registry.model_id
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM model_parameter_evidence parameter
                  WHERE parameter.model_id = registry.model_id
              )
            """
        ),
        {"source_id": source_id},
    )
    return int(result.rowcount or 0)


def _aggregate_checksum(values: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(values)).encode("utf-8")
    ).hexdigest()


def _existing_or_stable_model_id(
    connection: Connection,
    *,
    external_model_id: str,
    model_family: str,
    source_id: str,
) -> UUID:
    existing = connection.execute(
        text(
            """
            SELECT model_id
            FROM public_pbpk_model_registry
            WHERE external_model_id = :external_model_id
               OR (
                    source_id = :source_id
                    AND lower(model_name) IN (
                        lower(:model_family),
                        lower(:registry_name)
                    )
               )
            ORDER BY (external_model_id = :external_model_id) DESC,
                     model_id
            LIMIT 1
            """
        ),
        {
            "external_model_id": external_model_id,
            "source_id": source_id,
            "model_family": model_family,
            "registry_name": f"{model_family} PBPK model",
        },
    ).scalar_one_or_none()
    return (
        UUID(str(existing))
        if existing is not None
        else stable_uuid("public-pbpk-model", external_model_id)
    )


def _upsert_registry(
    connection: Connection,
    *,
    models: list[OSPModelVersionSnapshot],
    compound_ids: dict[str, UUID],
    source_snapshot_sha256: str,
    release_tag: str | None,
) -> UUID:
    first = models[0]
    source_id = first.source_id
    model_id = _existing_or_stable_model_id(
        connection,
        external_model_id=first.model_id,
        model_family=first.model_family,
        source_id=source_id,
    )
    compound_id = compound_ids.get(_normalized_name(first.model_family))
    species = sorted({value for model in models for value in model.species})
    populations = sorted(
        {value for model in models for value in model.populations}
    )
    routes = sorted({value for model in models for value in model.routes})
    compounds = sorted(
        {value for model in models for value in model.compounds}
    )
    license_decisions = [model.license for model in models]
    allowed_licenses = [
        decision for decision in license_decisions
        if str(decision.status.value) == "allowed"
    ]
    license_spdx_id = (
        allowed_licenses[0].spdx_id if allowed_licenses else None
    )
    reports = sorted(
        {path for model in models for path in model.report_paths}
    )
    repository_location = (
        f"{first.repository_url.removesuffix('.git')}/tree/"
        f"{first.repository_commit}/{first.model_family}"
    )
    reusable = any(model.reusable_candidate for model in models)
    ranking_label = "reuse_positive" if reusable else "retrieval_only"
    run_label_grade = 3 if reusable else 1
    validation_status = (
        "external_study_validated"
        if reusable
        else "execution_verified_retrieval_only"
    )
    connection.execute(
        text(
            """
            INSERT INTO public_pbpk_model_registry (
                model_id, external_model_id, model_name, compound_id,
                model_type, software, species, population, route,
                model_source, source_id, source_variant_id,
                repository_url, repository_commit, release_tag,
                license_spdx_id, model_file_path, evaluation_report_path,
                validation_status, validation_grade, run_label_grade,
                ranking_label, reusable_candidate, model_checksum,
                source_snapshot_checksum, static_parse_status,
                execution_probe_status, simulation_count,
                parameter_record_count, component_record_count,
                process_record_count, curation_status, source_location,
                notes, metadata_json, updated_at
            ) VALUES (
                CAST(:model_id AS uuid), :external_model_id, :model_name,
                CAST(:compound_id AS uuid), :model_type, :software,
                :species, :population, :route, :model_source, :source_id,
                :source_variant_id, :repository_url, :repository_commit,
                :release_tag, :license_spdx_id, :model_file_path,
                :evaluation_report_path, :validation_status,
                :validation_grade, :run_label_grade, :ranking_label,
                :reusable_candidate, :model_checksum,
                :source_snapshot_checksum, :static_parse_status,
                :execution_probe_status, :simulation_count,
                :parameter_record_count, :component_record_count,
                :process_record_count,
                CAST(:curation_status AS curation_status),
                :source_location, :notes, CAST(:metadata_json AS jsonb),
                now()
            )
            ON CONFLICT (model_id) DO UPDATE SET
                external_model_id = EXCLUDED.external_model_id,
                model_name = EXCLUDED.model_name,
                compound_id = EXCLUDED.compound_id,
                model_type = EXCLUDED.model_type,
                software = EXCLUDED.software,
                species = EXCLUDED.species,
                population = EXCLUDED.population,
                route = EXCLUDED.route,
                model_source = EXCLUDED.model_source,
                source_id = EXCLUDED.source_id,
                source_variant_id = EXCLUDED.source_variant_id,
                repository_url = EXCLUDED.repository_url,
                repository_commit = EXCLUDED.repository_commit,
                release_tag = EXCLUDED.release_tag,
                license_spdx_id = EXCLUDED.license_spdx_id,
                model_file_path = EXCLUDED.model_file_path,
                evaluation_report_path = EXCLUDED.evaluation_report_path,
                validation_status = EXCLUDED.validation_status,
                validation_grade = EXCLUDED.validation_grade,
                run_label_grade = EXCLUDED.run_label_grade,
                ranking_label = EXCLUDED.ranking_label,
                reusable_candidate = EXCLUDED.reusable_candidate,
                model_checksum = EXCLUDED.model_checksum,
                source_snapshot_checksum = EXCLUDED.source_snapshot_checksum,
                static_parse_status = EXCLUDED.static_parse_status,
                execution_probe_status = EXCLUDED.execution_probe_status,
                simulation_count = EXCLUDED.simulation_count,
                parameter_record_count = EXCLUDED.parameter_record_count,
                component_record_count = EXCLUDED.component_record_count,
                process_record_count = EXCLUDED.process_record_count,
                curation_status = EXCLUDED.curation_status,
                source_location = EXCLUDED.source_location,
                notes = EXCLUDED.notes,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = now()
            """
        ),
        {
            "model_id": str(model_id),
            "external_model_id": first.model_id,
            "model_name": f"{first.model_family} PBPK model",
            "compound_id": str(compound_id) if compound_id else None,
            "model_type": "whole-body PBPK",
            "software": f"{first.software} {first.software_version}".strip(),
            "species": ", ".join(species),
            "population": ", ".join(populations),
            "route": ", ".join(routes),
            "model_source": first.repository_url,
            "source_id": source_id,
            "source_variant_id": source_id,
            "repository_url": first.repository_url,
            "repository_commit": first.repository_commit,
            "release_tag": release_tag,
            "license_spdx_id": license_spdx_id,
            "model_file_path": first.project_relative_path,
            "evaluation_report_path": reports[0] if reports else None,
            "validation_status": validation_status,
            "validation_grade": validation_status,
            "run_label_grade": run_label_grade,
            "ranking_label": ranking_label,
            "reusable_candidate": reusable,
            "model_checksum": _aggregate_checksum(
                [model.project_sha256 for model in models]
            ),
            "source_snapshot_checksum": source_snapshot_sha256,
            "static_parse_status": "parsed",
            "execution_probe_status": "passed",
            "simulation_count": sum(
                model.simulation_count for model in models
            ),
            "parameter_record_count": sum(
                model.parameter_record_count for model in models
            ),
            "component_record_count": sum(
                model.component_record_count for model in models
            ),
            "process_record_count": sum(
                model.process_record_count for model in models
            ),
            "curation_status": "auto_validated",
            "source_location": repository_location,
            "notes": (
                "Downloaded public model assets passed checksum, static parse, "
                "PKML parse, and locked runtime execution checks. Automatic "
                "reuse remains blocked unless license and independent "
                "validation gates also pass."
            ),
            "metadata_json": _json(
                {
                    "importer_version": IMPORTER_VERSION,
                    "model_family": first.model_family,
                    "model_version_count": len(models),
                    "compounds": compounds,
                    "project_paths": [
                        model.project_relative_path for model in models
                    ],
                    "license_statuses": sorted(
                        {
                            str(model.license.status.value)
                            for model in models
                        }
                    ),
                }
            ),
        },
    )
    return model_id


def _upsert_version(
    connection: Connection,
    *,
    model: OSPModelVersionSnapshot,
    model_id: UUID,
    source_snapshot_sha256: str,
    release_tag: str | None,
) -> UUID:
    version_id = stable_uuid(
        "public-pbpk-model-version", model.model_version_id
    )
    license_decision = model.license
    source_location = (
        f"{model.repository_url.removesuffix('.git')}/blob/"
        f"{model.repository_commit}/{model.project_relative_path}"
    )
    connection.execute(
        text(
            """
            INSERT INTO public_pbpk_model_version (
                model_version_id, model_id, external_model_version_id,
                version_label, repository_url, repository_commit,
                project_relative_path, project_sha256,
                source_snapshot_sha256, software, software_version,
                license_status, license_spdx_id, license_source_location,
                license_source_checksum, static_parse_status,
                execution_probe_status, pkml_parse_status,
                reusable_candidate, compounds, species, populations, routes,
                formulations, enzymes, transporters, compartments,
                process_types, output_names, simulation_count,
                parameter_record_count, component_record_count,
                process_record_count, report_count,
                dataset_reference_count, source_id, source_variant_id,
                source_location, metadata_json, updated_at
            ) VALUES (
                CAST(:model_version_id AS uuid), CAST(:model_id AS uuid),
                :external_model_version_id, :version_label,
                :repository_url, :repository_commit, :project_relative_path,
                :project_sha256, :source_snapshot_sha256, :software,
                :software_version, :license_status, :license_spdx_id,
                :license_source_location, :license_source_checksum,
                :static_parse_status, :execution_probe_status,
                :pkml_parse_status, :reusable_candidate, :compounds,
                :species, :populations, :routes, :formulations, :enzymes,
                :transporters, :compartments, :process_types, :output_names,
                :simulation_count, :parameter_record_count,
                :component_record_count, :process_record_count,
                :report_count, :dataset_reference_count, :source_id,
                :source_variant_id, :source_location,
                CAST(:metadata_json AS jsonb), now()
            )
            ON CONFLICT (model_version_id) DO UPDATE SET
                model_id = EXCLUDED.model_id,
                external_model_version_id = EXCLUDED.external_model_version_id,
                version_label = EXCLUDED.version_label,
                repository_url = EXCLUDED.repository_url,
                repository_commit = EXCLUDED.repository_commit,
                project_relative_path = EXCLUDED.project_relative_path,
                project_sha256 = EXCLUDED.project_sha256,
                source_snapshot_sha256 = EXCLUDED.source_snapshot_sha256,
                software = EXCLUDED.software,
                software_version = EXCLUDED.software_version,
                license_status = EXCLUDED.license_status,
                license_spdx_id = EXCLUDED.license_spdx_id,
                license_source_location = EXCLUDED.license_source_location,
                license_source_checksum = EXCLUDED.license_source_checksum,
                static_parse_status = EXCLUDED.static_parse_status,
                execution_probe_status = EXCLUDED.execution_probe_status,
                pkml_parse_status = EXCLUDED.pkml_parse_status,
                reusable_candidate = EXCLUDED.reusable_candidate,
                compounds = EXCLUDED.compounds,
                species = EXCLUDED.species,
                populations = EXCLUDED.populations,
                routes = EXCLUDED.routes,
                formulations = EXCLUDED.formulations,
                enzymes = EXCLUDED.enzymes,
                transporters = EXCLUDED.transporters,
                compartments = EXCLUDED.compartments,
                process_types = EXCLUDED.process_types,
                output_names = EXCLUDED.output_names,
                simulation_count = EXCLUDED.simulation_count,
                parameter_record_count = EXCLUDED.parameter_record_count,
                component_record_count = EXCLUDED.component_record_count,
                process_record_count = EXCLUDED.process_record_count,
                report_count = EXCLUDED.report_count,
                dataset_reference_count = EXCLUDED.dataset_reference_count,
                source_id = EXCLUDED.source_id,
                source_variant_id = EXCLUDED.source_variant_id,
                source_location = EXCLUDED.source_location,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = now()
            """
        ),
        {
            "model_version_id": str(version_id),
            "model_id": str(model_id),
            "external_model_version_id": model.model_version_id,
            "version_label": release_tag,
            "repository_url": model.repository_url,
            "repository_commit": model.repository_commit,
            "project_relative_path": model.project_relative_path,
            "project_sha256": model.project_sha256,
            "source_snapshot_sha256": source_snapshot_sha256,
            "software": model.software,
            "software_version": model.software_version,
            "license_status": str(license_decision.status.value),
            "license_spdx_id": license_decision.spdx_id,
            "license_source_location": license_decision.source_location,
            "license_source_checksum": license_decision.source_checksum,
            "static_parse_status": str(model.static_parse_status.value),
            "execution_probe_status": str(
                model.execution_probe_status.value
            ),
            "pkml_parse_status": str(model.pkml_parse_status.value),
            "reusable_candidate": model.reusable_candidate,
            "compounds": model.compounds,
            "species": model.species,
            "populations": model.populations,
            "routes": model.routes,
            "formulations": model.formulations,
            "enzymes": model.enzymes,
            "transporters": model.transporters,
            "compartments": model.compartments,
            "process_types": model.process_types,
            "output_names": model.output_names,
            "simulation_count": model.simulation_count,
            "parameter_record_count": model.parameter_record_count,
            "component_record_count": model.component_record_count,
            "process_record_count": model.process_record_count,
            "report_count": model.report_count,
            "dataset_reference_count": model.dataset_reference_count,
            "source_id": model.source_id,
            "source_variant_id": model.source_id,
            "source_location": source_location,
            "metadata_json": _json(
                {
                    "importer_version": IMPORTER_VERSION,
                    "model_family": model.model_family,
                    "project_key": model.project_key,
                    "blocking_reasons": model.blocking_reasons,
                    "metabolites": model.metabolites,
                    "report_paths": model.report_paths,
                    "reported_validation_metric_count": (
                        model.validation_metric_count
                    ),
                }
            ),
        },
    )
    return version_id


def _upsert_assets(
    connection: Connection,
    *,
    model: OSPModelVersionSnapshot,
    model_version_id: UUID,
) -> int:
    for asset in model.assets:
        asset_id = stable_uuid(
            "public-pbpk-model-asset",
            model.model_version_id,
            asset.asset_kind,
            asset.relative_path,
            asset.sha256,
        )
        connection.execute(
            text(
                """
                INSERT INTO public_pbpk_model_asset (
                    model_asset_id, model_version_id, asset_kind,
                    relative_path, sha256, source_location, metadata_json
                ) VALUES (
                    CAST(:model_asset_id AS uuid),
                    CAST(:model_version_id AS uuid), :asset_kind,
                    :relative_path, :sha256, :source_location,
                    CAST(:metadata_json AS jsonb)
                )
                ON CONFLICT (model_asset_id) DO UPDATE SET
                    source_location = EXCLUDED.source_location,
                    metadata_json = EXCLUDED.metadata_json
                """
            ),
            {
                "model_asset_id": str(asset_id),
                "model_version_id": str(model_version_id),
                "asset_kind": asset.asset_kind,
                "relative_path": asset.relative_path,
                "sha256": asset.sha256,
                "source_location": asset.source_location,
                "metadata_json": _json(
                    {"importer_version": IMPORTER_VERSION}
                ),
            },
        )
    return len(model.assets)


def _upsert_simulations(
    connection: Connection,
    *,
    model: OSPModelVersionSnapshot,
    model_version_id: UUID,
    simulations: list[dict[str, Any]],
) -> int:
    for sequence, row in enumerate(simulations, 1):
        external_id = str(row["simulation_id"])
        pkml = row["pkml_source"]
        csv_results = row.get("csv_results") or []
        primary_result = csv_results[0] if csv_results else None
        simulation_id = stable_uuid(
            "public-pbpk-model-simulation",
            model.model_version_id,
            external_id,
        )
        connection.execute(
            text(
                """
                INSERT INTO public_pbpk_model_simulation (
                    model_simulation_id, model_version_id,
                    external_simulation_id, protocol_sequence,
                    simulation_name, protocol_role, solver, species,
                    populations, routes, formulations,
                    validation_dataset_name, pkml_relative_path,
                    pkml_sha256, result_relative_path, result_sha256,
                    parameter_count, component_count, process_count,
                    administration_count, source_location, metadata_json
                ) VALUES (
                    CAST(:model_simulation_id AS uuid),
                    CAST(:model_version_id AS uuid),
                    :external_simulation_id, :protocol_sequence,
                    :simulation_name, :protocol_role, :solver, :species,
                    :populations, :routes, :formulations,
                    :validation_dataset_name, :pkml_relative_path,
                    :pkml_sha256, :result_relative_path, :result_sha256,
                    :parameter_count, :component_count, :process_count,
                    :administration_count, :source_location,
                    CAST(:metadata_json AS jsonb)
                )
                ON CONFLICT (model_simulation_id) DO UPDATE SET
                    protocol_sequence = EXCLUDED.protocol_sequence,
                    simulation_name = EXCLUDED.simulation_name,
                    protocol_role = EXCLUDED.protocol_role,
                    solver = EXCLUDED.solver,
                    species = EXCLUDED.species,
                    populations = EXCLUDED.populations,
                    routes = EXCLUDED.routes,
                    formulations = EXCLUDED.formulations,
                    pkml_relative_path = EXCLUDED.pkml_relative_path,
                    pkml_sha256 = EXCLUDED.pkml_sha256,
                    result_relative_path = EXCLUDED.result_relative_path,
                    result_sha256 = EXCLUDED.result_sha256,
                    parameter_count = EXCLUDED.parameter_count,
                    component_count = EXCLUDED.component_count,
                    process_count = EXCLUDED.process_count,
                    administration_count = EXCLUDED.administration_count,
                    source_location = EXCLUDED.source_location,
                    metadata_json = EXCLUDED.metadata_json
                """
            ),
            {
                "model_simulation_id": str(simulation_id),
                "model_version_id": str(model_version_id),
                "external_simulation_id": external_id,
                "protocol_sequence": sequence,
                "simulation_name": str(row["simulation_name"]),
                "protocol_role": "catalog_runtime_probe",
                "solver": row.get("solver"),
                "species": row.get("species") or [],
                "populations": row.get("populations") or [],
                "routes": row.get("routes") or [],
                "formulations": row.get("formulations") or [],
                "validation_dataset_name": None,
                "pkml_relative_path": str(pkml["relative_path"]),
                "pkml_sha256": str(pkml["sha256"]),
                "result_relative_path": (
                    str(primary_result["relative_path"])
                    if primary_result else None
                ),
                "result_sha256": (
                    str(primary_result["sha256"])
                    if primary_result else None
                ),
                "parameter_count": int(row.get("parameter_count") or 0),
                "component_count": int(row.get("component_count") or 0),
                "process_count": int(row.get("process_count") or 0),
                "administration_count": int(
                    row.get("administration_count") or 0
                ),
                "source_location": str(pkml["source_location"]),
                "metadata_json": _json(
                    {
                        "importer_version": IMPORTER_VERSION,
                        "parser_name": row.get("parser_name"),
                        "parser_version": row.get("parser_version"),
                        "parse_status": row.get("parse_status"),
                        "compounds": row.get("compounds") or [],
                        "enzymes": row.get("enzymes") or [],
                        "transporters": row.get("transporters") or [],
                        "result_csv_count": len(csv_results),
                        "warnings": row.get("warnings") or [],
                    }
                ),
            },
        )
    return len(simulations)


def import_public_model_catalog(
    *,
    bundle_directory: Path,
    gate1_audit_path: Path,
    database_url: str,
    release_tag: str | None = None,
) -> dict[str, Any]:
    bundle = load_public_model_catalog_bundle(
        bundle_directory=bundle_directory,
        gate1_audit_path=gate1_audit_path,
    )
    source_ids = {model.source_id for model in bundle.models}
    if len(source_ids) != 1:
        raise PublicModelCatalogImportError(
            "A catalog import must contain exactly one source ID."
        )
    source_id = next(iter(source_ids))
    models_by_external_id: dict[str, list[OSPModelVersionSnapshot]] = (
        defaultdict(list)
    )
    simulations_by_project: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for model in bundle.models:
        models_by_external_id[model.model_id].append(model)
    for simulation in bundle.simulations:
        simulations_by_project[str(simulation["project_key"])].append(
            simulation
        )

    counts = {
        "models": 0,
        "model_versions": 0,
        "assets": 0,
        "simulations": 0,
        "compound_links": 0,
        "superseded_metadata_only_models": 0,
    }
    engine = create_db_engine(database_url)
    with engine.begin() as connection:
        _require_source(connection, source_id)
        counts["superseded_metadata_only_models"] = (
            _delete_stale_catalog_stubs(
                connection,
                source_id=source_id,
            )
        )
        compound_ids = _compound_ids(connection)
        for external_model_id in sorted(models_by_external_id):
            family_models = sorted(
                models_by_external_id[external_model_id],
                key=lambda item: item.project_relative_path.casefold(),
            )
            model_id = _upsert_registry(
                connection,
                models=family_models,
                compound_ids=compound_ids,
                source_snapshot_sha256=bundle.source_snapshot_sha256,
                release_tag=release_tag,
            )
            counts["models"] += 1
            if _normalized_name(family_models[0].model_family) in compound_ids:
                counts["compound_links"] += 1
            for model in family_models:
                model_version_id = _upsert_version(
                    connection,
                    model=model,
                    model_id=model_id,
                    source_snapshot_sha256=bundle.source_snapshot_sha256,
                    release_tag=release_tag,
                )
                counts["model_versions"] += 1
                counts["assets"] += _upsert_assets(
                    connection,
                    model=model,
                    model_version_id=model_version_id,
                )
                counts["simulations"] += _upsert_simulations(
                    connection,
                    model=model,
                    model_version_id=model_version_id,
                    simulations=simulations_by_project[model.project_key],
                )

    return {
        "accepted": True,
        "importer_version": IMPORTER_VERSION,
        "source_id": source_id,
        "repository_commit": bundle.summary["repository_commit"],
        "source_snapshot_sha256": bundle.source_snapshot_sha256,
        "release_tag": release_tag,
        "counts": counts,
        "automatic_reuse_enabled": False,
        "automatic_reuse_blocker": (
            "Imported OSP versions are retrieval-only until license and "
            "independent validation gates pass."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and import an audited public PBPK model catalog without "
            "promoting retrieval-only models to automatic reuse."
        )
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--gate1-audit", type=Path, required=True)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--release-tag")
    args = parser.parse_args()
    result = import_public_model_catalog(
        bundle_directory=args.bundle,
        gate1_audit_path=args.gate1_audit,
        database_url=args.database_url,
        release_tag=args.release_tag,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
