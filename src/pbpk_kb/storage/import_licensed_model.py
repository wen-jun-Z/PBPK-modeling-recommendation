from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import Connection, text

from pbpk_kb.model_assets.models import (
    ModelValidationRunLabel,
    OSPModelVersionSnapshot,
)
from pbpk_kb.model_assets.validation_scope import build_model_validation_scope
from pbpk_kb.storage.db import create_db_engine


IMPORTER_VERSION = "licensed-model-catalog-importer-1.0.0"
DATABASE_SOURCE_ID = "osp_pbpk_model_library"
REQUIRED_FILES = (
    "adapter_manifest.json",
    "gate1_audit.json",
    "summary.json",
    "model_versions.jsonl.gz",
    "simulations.jsonl.gz",
    "validation_run_labels.jsonl.gz",
    "validation_runs.jsonl.gz",
    "provenance/registration_summary.json",
    "provenance/snapshot_reuse_audit.json",
    "provenance/snapshot_reuse_protocol.json",
    "provenance/verification_dataset_metrics.jsonl",
)
DATASET_METRIC_NAMES = (
    "auc0_last_ratio",
    "cmax_ratio",
    "gmfe",
    "geometric_bias",
    "log_rmse",
    "two_fold_fraction",
)
POOLED_METRIC_NAMES = (
    "pooled_gmfe",
    "pooled_geometric_bias",
    "pooled_two_fold_fraction",
    "endpoint_scenario_pass_fraction",
)


class LicensedModelImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class LicensedModelBundle:
    root: Path
    adapter_manifest: dict[str, Any]
    gate1_audit: dict[str, Any]
    build_summary: dict[str, Any]
    registration_summary: dict[str, Any]
    protocol: dict[str, Any]
    scientific_audit: dict[str, Any]
    model: OSPModelVersionSnapshot
    validation_label: ModelValidationRunLabel
    validation_run: dict[str, Any]
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
        raise LicensedModelImportError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"mode": "rt", "encoding": "utf-8"}
    with opener(path, **kwargs) as handle:  # type: ignore[arg-type]
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise LicensedModelImportError(f"Expected JSONL objects: {path}")
    return rows


def _require_equal(observed: object, expected: object, message: str) -> None:
    if observed != expected:
        raise LicensedModelImportError(
            f"{message}: expected {expected!r}, observed {observed!r}"
        )


def load_licensed_model_bundle(root: Path) -> LicensedModelBundle:
    root = root.resolve()
    for relative_path in REQUIRED_FILES:
        path = root / relative_path
        if not path.is_file():
            raise LicensedModelImportError(f"Required bundle file is missing: {path}")

    adapter = _read_json(root / "adapter_manifest.json")
    gate1 = _read_json(root / "gate1_audit.json")
    summary = _read_json(root / "summary.json")
    registration = _read_json(root / "provenance/registration_summary.json")
    protocol = _read_json(root / "provenance/snapshot_reuse_protocol.json")
    scientific_audit = _read_json(
        root / "provenance/snapshot_reuse_audit.json"
    )

    _require_equal(adapter.get("accepted"), True, "Adapter acceptance")
    _require_equal(gate1.get("accepted"), True, "Gate 1 acceptance")
    _require_equal(
        summary.get("gate1_model_version_acceptance"),
        True,
        "Build acceptance",
    )
    _require_equal(registration.get("accepted"), True, "Registration acceptance")
    _require_equal(
        scientific_audit.get("engineering_accepted"),
        True,
        "Engineering audit acceptance",
    )
    _require_equal(
        scientific_audit.get("scientific_accepted"),
        True,
        "Scientific audit acceptance",
    )

    output_checksums = adapter.get("output_checksums") or {}
    if not isinstance(output_checksums, dict) or not output_checksums:
        raise LicensedModelImportError("Adapter output checksum manifest is empty.")
    for relative_path, expected in output_checksums.items():
        path = root / str(relative_path)
        if not path.is_file():
            raise LicensedModelImportError(
                f"Adapter output is missing: {relative_path}"
            )
        _require_equal(
            sha256_file(path), expected, f"Checksum for {relative_path}"
        )
    _require_equal(
        sha256_file(root / "gate1_audit.json"),
        adapter.get("gate1_audit_sha256"),
        "Gate 1 audit checksum",
    )
    _require_equal(
        sha256_file(root / "summary.json"),
        adapter.get("summary_sha256"),
        "Build summary checksum",
    )
    _require_equal(
        sha256_file(root / "provenance/registration_summary.json"),
        adapter.get("source_registration_summary_sha256"),
        "Registration summary checksum",
    )
    provenance_checksums = gate1.get("input_checksums") or {}
    for key, relative_path in {
        "protocol": "provenance/snapshot_reuse_protocol.json",
        "validation_audit": "provenance/snapshot_reuse_audit.json",
        "validation_dataset_metrics": (
            "provenance/verification_dataset_metrics.jsonl"
        ),
    }.items():
        _require_equal(
            sha256_file(root / relative_path),
            provenance_checksums.get(key),
            f"Provenance checksum for {key}",
        )

    model_rows = _read_jsonl(root / "model_versions.jsonl.gz")
    label_rows = _read_jsonl(root / "validation_run_labels.jsonl.gz")
    validation_rows = _read_jsonl(root / "validation_runs.jsonl.gz")
    simulations = tuple(_read_jsonl(root / "simulations.jsonl.gz"))
    _require_equal(len(model_rows), 1, "Registered model version count")
    _require_equal(len(label_rows), 1, "Validation label count")
    _require_equal(len(validation_rows), 1, "Validation run count")

    model_payload = dict(model_rows[0])
    if model_payload.get("validation_scope") is None:
        metrics_by_sequence = {
            int(row["simulation_sequence"]): row
            for row in (validation_rows[0].get("datasets") or [])
        }
        scope = build_model_validation_scope(
            {
                **row,
                "validation_metrics": metrics_by_sequence.get(
                    int(row["protocol_sequence"]), {}
                ),
            }
            for row in simulations
        )
        model_payload["validation_scope"] = (
            scope.model_dump(mode="json") if scope is not None else None
        )
    model = OSPModelVersionSnapshot.model_validate(model_payload)
    label = ModelValidationRunLabel.model_validate(label_rows[0])
    validation_run = validation_rows[0]
    if model.validation_run_label is None:
        raise LicensedModelImportError("Model has no validation run label.")
    _require_equal(
        model.validation_run_label.model_dump(mode="json"),
        label.model_dump(mode="json"),
        "Model and standalone validation labels",
    )
    _require_equal(model.model_id, gate1.get("model_id"), "Gate 1 model ID")
    _require_equal(
        model.model_version_id,
        gate1.get("model_version_id"),
        "Gate 1 model version ID",
    )
    _require_equal(
        label.validation_run_id,
        gate1.get("validation_run_id"),
        "Gate 1 validation run ID",
    )
    for observed, expected, message in (
        (model.reusable_candidate, True, "Model reusable flag"),
        (label.validation_grade, "external_study_validated", "Validation grade"),
        (label.run_label_grade, 3, "Validation run label grade"),
        (label.ranking_label, "reuse_positive", "Ranking label"),
        (label.scientific_accepted, True, "Scientific label acceptance"),
        (len(simulations), model.simulation_count, "Simulation count"),
        (
            len(validation_run.get("datasets") or []),
            label.verification_dataset_count,
            "Validation dataset count",
        ),
    ):
        _require_equal(observed, expected, message)
    _require_equal(
        ModelValidationRunLabel.model_validate(
            validation_run.get("label")
        ).model_dump(mode="json"),
        label.model_dump(mode="json"),
        "Validation run embedded label",
    )
    _require_equal(
        registration.get("model_version_id"),
        model.model_version_id,
        "Registration model version identity",
    )
    _require_equal(
        protocol.get("source", {}).get("commit_sha"),
        model.repository_commit,
        "Protocol repository commit",
    )
    _require_equal(
        protocol.get("source", {}).get("source_id"),
        model.source_id,
        "Protocol source variant",
    )
    if (
        str(model.license.status.value) != "allowed"
        or not model.license.spdx_id
        or not model.license.source_location
        or not model.license.source_checksum
    ):
        raise LicensedModelImportError(
            "A reusable model requires a complete allowed-license decision."
        )
    simulation_sequences = [row.get("protocol_sequence") for row in simulations]
    if simulation_sequences != list(range(1, len(simulations) + 1)):
        raise LicensedModelImportError(
            "Simulation protocol sequences must be complete and ordered."
        )
    return LicensedModelBundle(
        root=root,
        adapter_manifest=adapter,
        gate1_audit=gate1,
        build_summary=summary,
        registration_summary=registration,
        protocol=protocol,
        scientific_audit=scientific_audit,
        model=model,
        validation_label=label,
        validation_run=validation_run,
        simulations=simulations,
    )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _require_compound(
    connection: Connection,
    *,
    compound_name: str,
    expected_pubchem_cid: str,
) -> UUID:
    rows = connection.execute(
        text(
            """
            SELECT compound_id, preferred_name::text, pubchem_cid
            FROM compound_identity
            WHERE (
                lower(preferred_name::text) = lower(:compound_name)
                OR lower(COALESCE(normalized_name::text, '')) = lower(:compound_name)
                OR EXISTS (
                    SELECT 1 FROM unnest(synonyms) AS synonym
                    WHERE lower(synonym) = lower(:compound_name)
                )
            )
            ORDER BY
                (lower(preferred_name::text) = lower(:compound_name)) DESC,
                compound_id
            """
        ),
        {"compound_name": compound_name},
    ).mappings().all()
    exact = [
        row
        for row in rows
        if str(row.get("pubchem_cid") or "") == expected_pubchem_cid
    ]
    if len(exact) != 1:
        raise LicensedModelImportError(
            "Model compound resolution requires exactly one name and PubChem "
            f"match for {compound_name!r}/{expected_pubchem_cid}; found {len(exact)}."
        )
    return UUID(str(exact[0]["compound_id"]))


def _require_source(connection: Connection) -> None:
    source = connection.execute(
        text("SELECT source_id FROM source_registry WHERE source_id = :source_id"),
        {"source_id": DATABASE_SOURCE_ID},
    ).scalar_one_or_none()
    if source is None:
        raise LicensedModelImportError(
            f"Required source registry entry is absent: {DATABASE_SOURCE_ID}"
        )


def _upsert_registry(
    connection: Connection,
    *,
    bundle: LicensedModelBundle,
    compound_id: UUID,
) -> UUID:
    model = bundle.model
    label = bundle.validation_label
    source = bundle.protocol["source"]
    registry_name = f"{model.model_family} PBPK model"
    existing = connection.execute(
        text(
            """
            SELECT model_id
            FROM public_pbpk_model_registry
            WHERE external_model_id = :external_model_id
               OR (
                    lower(model_name) IN (
                        lower(:model_name), lower(:model_family)
                    )
                    AND compound_id = CAST(:compound_id AS uuid)
                    AND source_id = :source_id
                    AND (
                        repository_url IS NULL
                        OR repository_url = :repository_url
                    )
               )
            ORDER BY (external_model_id = :external_model_id) DESC, model_id
            LIMIT 1
            """
        ),
        {
            "external_model_id": model.model_id,
            "model_name": registry_name,
            "model_family": model.model_family,
            "compound_id": str(compound_id),
            "source_id": DATABASE_SOURCE_ID,
            "repository_url": model.repository_url,
        },
    ).scalar_one_or_none()
    model_id = (
        UUID(str(existing))
        if existing is not None
        else stable_uuid("public-pbpk-model", model.model_id)
    )
    repository_location = (
        f"{model.repository_url.removesuffix('.git')}/tree/"
        f"{model.repository_commit}"
    )
    connection.execute(
        text(
            """
            INSERT INTO public_pbpk_model_registry (
                model_id, external_model_id, model_name, compound_id,
                model_type, software, species, population, route,
                model_source, source_id, source_variant_id,
                repository_url, repository_commit, release_tag,
                license_spdx_id, model_file_path, parameter_file_path,
                evaluation_report_path, validation_status,
                validation_grade, run_label_grade, ranking_label,
                reusable_candidate, model_checksum,
                source_snapshot_checksum, static_parse_status,
                execution_probe_status, latest_validation_run_id,
                simulation_count, parameter_record_count,
                component_record_count, process_record_count,
                curation_status, source_location, notes, metadata_json,
                updated_at
            ) VALUES (
                CAST(:model_id AS uuid), :external_model_id, :model_name,
                CAST(:compound_id AS uuid), :model_type, :software,
                :species, :population, :route, :model_source, :source_id,
                :source_variant_id, :repository_url, :repository_commit,
                :release_tag, :license_spdx_id, :model_file_path,
                :parameter_file_path, :evaluation_report_path,
                :validation_status, :validation_grade, :run_label_grade,
                :ranking_label, :reusable_candidate, :model_checksum,
                :source_snapshot_checksum, :static_parse_status,
                :execution_probe_status, :latest_validation_run_id,
                :simulation_count, :parameter_record_count,
                :component_record_count, :process_record_count,
                CAST(:curation_status AS curation_status), :source_location,
                :notes, CAST(:metadata_json AS jsonb), now()
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
                parameter_file_path = EXCLUDED.parameter_file_path,
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
                latest_validation_run_id = EXCLUDED.latest_validation_run_id,
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
            "external_model_id": model.model_id,
            "model_name": registry_name,
            "compound_id": str(compound_id),
            "model_type": "whole-body PBPK",
            "software": f"{model.software} {model.software_version}",
            "species": ", ".join(model.species),
            "population": (
                f"{label.verification_dataset_count} repository-frozen "
                "model-verification datasets"
            ),
            "route": ", ".join(model.routes),
            "model_source": model.repository_url,
            "source_id": DATABASE_SOURCE_ID,
            "source_variant_id": model.source_id,
            "repository_url": model.repository_url,
            "repository_commit": model.repository_commit,
            "release_tag": source.get("release_tag"),
            "license_spdx_id": model.license.spdx_id,
            "model_file_path": model.project_relative_path,
            "parameter_file_path": str(bundle.root / "parameters.jsonl.gz"),
            "evaluation_report_path": str(
                bundle.root / "provenance/snapshot_reuse_audit.json"
            ),
            "validation_status": label.validation_grade,
            "validation_grade": label.validation_grade,
            "run_label_grade": label.run_label_grade,
            "ranking_label": label.ranking_label,
            "reusable_candidate": label.reusable_candidate,
            "model_checksum": model.project_sha256,
            "source_snapshot_checksum": label.source_snapshot_sha256,
            "static_parse_status": str(model.static_parse_status.value),
            "execution_probe_status": str(model.execution_probe_status.value),
            "latest_validation_run_id": label.validation_run_id,
            "simulation_count": model.simulation_count,
            "parameter_record_count": model.parameter_record_count,
            "component_record_count": model.component_record_count,
            "process_record_count": model.process_record_count,
            "curation_status": "auto_validated",
            "source_location": repository_location,
            "notes": (
                "Official OSP release; executed twice in a locked PK-Sim "
                "runtime and independently code-audited against the public "
                "repository's preregistered model-verification split."
            ),
            "metadata_json": _json(
                {
                    "importer_version": IMPORTER_VERSION,
                    "bundle_path": str(bundle.root),
                    "bundle_manifest_sha256": sha256_file(
                        bundle.root / "adapter_manifest.json"
                    ),
                    "compounds": model.compounds,
                    "enzymes": model.enzymes,
                    "transporters": model.transporters,
                    "formulations": model.formulations,
                    "compartments": model.compartments,
                    "process_types": model.process_types,
                    "output_names": model.output_names,
                }
            ),
        },
    )
    return model_id


def _upsert_version(
    connection: Connection,
    *,
    bundle: LicensedModelBundle,
    model_id: UUID,
) -> UUID:
    model = bundle.model
    source = bundle.protocol["source"]
    model_version_id = stable_uuid(
        "public-pbpk-model-version", model.model_version_id
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
            "model_version_id": str(model_version_id),
            "model_id": str(model_id),
            "external_model_version_id": model.model_version_id,
            "version_label": source.get("release_tag"),
            "repository_url": model.repository_url,
            "repository_commit": model.repository_commit,
            "project_relative_path": model.project_relative_path,
            "project_sha256": model.project_sha256,
            "source_snapshot_sha256": bundle.validation_label.source_snapshot_sha256,
            "software": model.software,
            "software_version": model.software_version,
            "license_status": str(model.license.status.value),
            "license_spdx_id": model.license.spdx_id,
            "license_source_location": model.license.source_location,
            "license_source_checksum": model.license.source_checksum,
            "static_parse_status": str(model.static_parse_status.value),
            "execution_probe_status": str(model.execution_probe_status.value),
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
            "source_id": DATABASE_SOURCE_ID,
            "source_variant_id": model.source_id,
            "source_location": (
                f"{model.repository_url.removesuffix('.git')}/tree/"
                f"{model.repository_commit}"
            ),
            "metadata_json": _json(
                {
                    "model_family": model.model_family,
                    "blocking_reasons": model.blocking_reasons,
                    "metabolites": model.metabolites,
                    "report_paths": model.report_paths,
                    "project_key": model.project_key,
                }
            ),
        },
    )
    return model_version_id


def _upsert_assets(
    connection: Connection,
    *,
    bundle: LicensedModelBundle,
    model_version_id: UUID,
) -> int:
    for asset in bundle.model.assets:
        asset_id = stable_uuid(
            "public-pbpk-model-asset",
            bundle.model.model_version_id,
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
                    CAST(:asset_id AS uuid), CAST(:model_version_id AS uuid),
                    :asset_kind, :relative_path, :sha256, :source_location,
                    '{}'::jsonb
                )
                ON CONFLICT (model_asset_id) DO UPDATE SET
                    source_location = EXCLUDED.source_location
                """
            ),
            {
                "asset_id": str(asset_id),
                "model_version_id": str(model_version_id),
                "asset_kind": asset.asset_kind,
                "relative_path": asset.relative_path,
                "sha256": asset.sha256,
                "source_location": asset.source_location,
            },
        )
    return len(bundle.model.assets)


def _upsert_simulations(
    connection: Connection,
    *,
    bundle: LicensedModelBundle,
    model_version_id: UUID,
) -> int:
    for row in bundle.simulations:
        simulation_id = stable_uuid(
            "public-pbpk-model-simulation",
            bundle.model.model_version_id,
            row["simulation_id"],
        )
        csv_rows = row.get("csv_results") or []
        result = csv_rows[0] if csv_rows else {}
        pkml = row.get("pkml_source") or {}
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
                    CAST(:simulation_id AS uuid),
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
                    simulation_name = EXCLUDED.simulation_name,
                    protocol_role = EXCLUDED.protocol_role,
                    validation_dataset_name = EXCLUDED.validation_dataset_name,
                    result_relative_path = EXCLUDED.result_relative_path,
                    result_sha256 = EXCLUDED.result_sha256,
                    metadata_json = EXCLUDED.metadata_json
                """
            ),
            {
                "simulation_id": str(simulation_id),
                "model_version_id": str(model_version_id),
                "external_simulation_id": row["simulation_id"],
                "protocol_sequence": row["protocol_sequence"],
                "simulation_name": row["simulation_name"],
                "protocol_role": row["protocol_role"],
                "solver": row.get("solver"),
                "species": row.get("species") or [],
                "populations": row.get("populations") or [],
                "routes": row.get("routes") or [],
                "formulations": row.get("formulations") or [],
                "validation_dataset_name": row.get("validation_dataset_name"),
                "pkml_relative_path": pkml["relative_path"],
                "pkml_sha256": pkml["sha256"],
                "result_relative_path": result.get("relative_path"),
                "result_sha256": result.get("sha256"),
                "parameter_count": row.get("parameter_count") or 0,
                "component_count": row.get("component_count") or 0,
                "process_count": row.get("process_count") or 0,
                "administration_count": row.get("administration_count") or 0,
                "source_location": pkml["source_location"],
                "metadata_json": _json(
                    {
                        "compounds": row.get("compounds") or [],
                        "metabolites": row.get("metabolites") or [],
                        "enzymes": row.get("enzymes") or [],
                        "transporters": row.get("transporters") or [],
                        "observed_dataset_names": (
                            row.get("observed_dataset_names") or []
                        ),
                        "outputs": result.get("outputs") or [],
                        "time_min": result.get("time_min"),
                        "time_max": result.get("time_max"),
                        "time_unit": result.get("time_unit"),
                    }
                ),
            },
        )
    return len(bundle.simulations)


def _upsert_validation(
    connection: Connection,
    *,
    bundle: LicensedModelBundle,
    model_version_id: UUID,
) -> tuple[UUID, int, int]:
    label = bundle.validation_label
    run = bundle.validation_run
    validation_run_id = stable_uuid(
        "public-pbpk-validation-run", label.validation_run_id
    )
    connection.execute(
        text(
            """
            INSERT INTO public_pbpk_model_validation_run (
                validation_run_id, external_validation_run_id,
                model_version_id, protocol_id, protocol_path,
                protocol_sha256, windows_manifest_path,
                windows_manifest_sha256, audit_report_path,
                audit_report_sha256, validation_dataset_metrics_path,
                source_snapshot_sha256,
                validation_grade, run_label_grade, ranking_label,
                reusable_candidate, engineering_accepted,
                scientific_accepted, verification_dataset_count,
                pooled_observation_count, pooled_gmfe,
                pooled_geometric_bias, pooled_two_fold_fraction,
                endpoint_scenario_pass_fraction, validated_at,
                source_location, metadata_json, updated_at
            ) VALUES (
                CAST(:validation_run_id AS uuid),
                :external_validation_run_id,
                CAST(:model_version_id AS uuid), :protocol_id,
                :protocol_path, :protocol_sha256, :windows_manifest_path,
                :windows_manifest_sha256, :audit_report_path,
                :audit_report_sha256, :validation_dataset_metrics_path,
                :source_snapshot_sha256,
                :validation_grade, :run_label_grade, :ranking_label,
                :reusable_candidate, :engineering_accepted,
                :scientific_accepted, :verification_dataset_count,
                :pooled_observation_count, :pooled_gmfe,
                :pooled_geometric_bias, :pooled_two_fold_fraction,
                :endpoint_scenario_pass_fraction, :validated_at,
                :source_location, CAST(:metadata_json AS jsonb), now()
            )
            ON CONFLICT (validation_run_id) DO UPDATE SET
                model_version_id = EXCLUDED.model_version_id,
                validation_grade = EXCLUDED.validation_grade,
                run_label_grade = EXCLUDED.run_label_grade,
                ranking_label = EXCLUDED.ranking_label,
                reusable_candidate = EXCLUDED.reusable_candidate,
                engineering_accepted = EXCLUDED.engineering_accepted,
                scientific_accepted = EXCLUDED.scientific_accepted,
                validation_dataset_metrics_path = EXCLUDED.validation_dataset_metrics_path,
                verification_dataset_count = EXCLUDED.verification_dataset_count,
                pooled_observation_count = EXCLUDED.pooled_observation_count,
                pooled_gmfe = EXCLUDED.pooled_gmfe,
                pooled_geometric_bias = EXCLUDED.pooled_geometric_bias,
                pooled_two_fold_fraction = EXCLUDED.pooled_two_fold_fraction,
                endpoint_scenario_pass_fraction = EXCLUDED.endpoint_scenario_pass_fraction,
                validated_at = EXCLUDED.validated_at,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = now()
            """
        ),
        {
            "validation_run_id": str(validation_run_id),
            "external_validation_run_id": label.validation_run_id,
            "model_version_id": str(model_version_id),
            "protocol_id": label.protocol_id,
            "protocol_path": str(
                bundle.root / "provenance/snapshot_reuse_protocol.json"
            ),
            "protocol_sha256": label.protocol_sha256,
            "windows_manifest_path": label.windows_manifest_path,
            "windows_manifest_sha256": label.windows_manifest_sha256,
            "audit_report_path": str(
                bundle.root / "provenance/snapshot_reuse_audit.json"
            ),
            "audit_report_sha256": label.audit_report_sha256,
            "validation_dataset_metrics_path": str(
                bundle.root
                / "provenance/verification_dataset_metrics.jsonl"
            ),
            "source_snapshot_sha256": label.source_snapshot_sha256,
            "validation_grade": label.validation_grade,
            "run_label_grade": label.run_label_grade,
            "ranking_label": label.ranking_label,
            "reusable_candidate": label.reusable_candidate,
            "engineering_accepted": label.engineering_accepted,
            "scientific_accepted": label.scientific_accepted,
            "verification_dataset_count": label.verification_dataset_count,
            "pooled_observation_count": label.pooled_observation_count,
            "pooled_gmfe": label.pooled_gmfe,
            "pooled_geometric_bias": label.pooled_geometric_bias,
            "pooled_two_fold_fraction": label.pooled_two_fold_fraction,
            "endpoint_scenario_pass_fraction": (
                label.endpoint_scenario_pass_fraction
            ),
            "validated_at": label.validated_at,
            "source_location": f"VALIDATION_AUDIT:{label.validation_run_id}",
            "metadata_json": _json(
                {
                    "auditor_version": bundle.scientific_audit.get(
                        "auditor_version"
                    ),
                    "engineering_checks": bundle.scientific_audit.get(
                        "engineering_checks"
                    ),
                    "scientific_checks": bundle.scientific_audit.get(
                        "scientific_checks"
                    ),
                    "roundtrip": bundle.scientific_audit.get("roundtrip"),
                    "validation_independence": (
                        label.validation_independence
                    ),
                }
            ),
        },
    )

    dataset_count = 0
    metric_count = 0
    for row in run.get("datasets") or []:
        dataset_id = stable_uuid(
            "public-pbpk-validation-dataset",
            label.validation_run_id,
            row["simulation_sequence"],
            row["dataset_name"],
        )
        connection.execute(
            text(
                """
                INSERT INTO public_pbpk_model_validation_dataset (
                    validation_dataset_id, validation_run_id,
                    simulation_sequence, dataset_name, simulation_name,
                    reference, source_location, prediction_unit,
                    observation_count, positive_pair_count,
                    excluded_nonpositive_observation_count,
                    auc0_last_ratio, cmax_ratio, gmfe, geometric_bias,
                    log_rmse, two_fold_fraction, dataset_guard_passed,
                    endpoint_evaluable, endpoint_passed, metadata_json
                ) VALUES (
                    CAST(:dataset_id AS uuid),
                    CAST(:validation_run_id AS uuid), :simulation_sequence,
                    :dataset_name, :simulation_name, :reference,
                    :source_location, :prediction_unit, :observation_count,
                    :positive_pair_count,
                    :excluded_nonpositive_observation_count,
                    :auc0_last_ratio, :cmax_ratio, :gmfe,
                    :geometric_bias, :log_rmse, :two_fold_fraction,
                    :dataset_guard_passed, :endpoint_evaluable,
                    :endpoint_passed, CAST(:metadata_json AS jsonb)
                )
                ON CONFLICT (validation_dataset_id) DO UPDATE SET
                    reference = EXCLUDED.reference,
                    observation_count = EXCLUDED.observation_count,
                    positive_pair_count = EXCLUDED.positive_pair_count,
                    auc0_last_ratio = EXCLUDED.auc0_last_ratio,
                    cmax_ratio = EXCLUDED.cmax_ratio,
                    gmfe = EXCLUDED.gmfe,
                    geometric_bias = EXCLUDED.geometric_bias,
                    log_rmse = EXCLUDED.log_rmse,
                    two_fold_fraction = EXCLUDED.two_fold_fraction,
                    dataset_guard_passed = EXCLUDED.dataset_guard_passed,
                    endpoint_evaluable = EXCLUDED.endpoint_evaluable,
                    endpoint_passed = EXCLUDED.endpoint_passed,
                    metadata_json = EXCLUDED.metadata_json
                """
            ),
            {
                "dataset_id": str(dataset_id),
                "validation_run_id": str(validation_run_id),
                "simulation_sequence": row["simulation_sequence"],
                "dataset_name": row["dataset_name"],
                "simulation_name": row["simulation_name"],
                "reference": row["reference"],
                "source_location": row["source_location"],
                "prediction_unit": row.get("prediction_unit"),
                "observation_count": row["observation_count"],
                "positive_pair_count": row["positive_pair_count"],
                "excluded_nonpositive_observation_count": row.get(
                    "excluded_nonpositive_observation_count", 0
                ),
                "auc0_last_ratio": row.get("auc0_last_ratio"),
                "cmax_ratio": row.get("cmax_ratio"),
                "gmfe": row.get("gmfe"),
                "geometric_bias": row.get("geometric_bias"),
                "log_rmse": row.get("log_rmse"),
                "two_fold_fraction": row.get("two_fold_fraction"),
                "dataset_guard_passed": row["dataset_guard_passed"],
                "endpoint_evaluable": row["endpoint_evaluable"],
                "endpoint_passed": row["endpoint_passed"],
                "metadata_json": _json(
                    {"prediction_csv": row.get("prediction_csv")}
                ),
            },
        )
        dataset_count += 1
        for metric_name in DATASET_METRIC_NAMES:
            value = row.get(metric_name)
            if value is None:
                raise LicensedModelImportError(
                    f"Dataset {row['dataset_name']} lacks {metric_name}."
                )
            metric_id = stable_uuid(
                "public-pbpk-validation-metric",
                label.validation_run_id,
                str(dataset_id),
                metric_name,
            )
            _upsert_metric(
                connection,
                metric_id=metric_id,
                validation_run_id=validation_run_id,
                validation_dataset_id=dataset_id,
                scope="dataset",
                name=metric_name,
                value=float(value),
                passed=(
                    bool(row["endpoint_passed"])
                    if metric_name in {"auc0_last_ratio", "cmax_ratio"}
                    else bool(row["dataset_guard_passed"])
                ),
                source_location=row["source_location"],
            )
            metric_count += 1

    for metric_name in POOLED_METRIC_NAMES:
        value = getattr(label, metric_name)
        if value is None:
            raise LicensedModelImportError(
                f"Validation label lacks pooled metric {metric_name}."
            )
        metric_id = stable_uuid(
            "public-pbpk-validation-metric",
            label.validation_run_id,
            "pooled",
            metric_name,
        )
        _upsert_metric(
            connection,
            metric_id=metric_id,
            validation_run_id=validation_run_id,
            validation_dataset_id=None,
            scope="pooled",
            name=metric_name,
            value=float(value),
            passed=True,
            source_location=f"VALIDATION_AUDIT:{label.validation_run_id}",
        )
        metric_count += 1
    return validation_run_id, dataset_count, metric_count


def _upsert_metric(
    connection: Connection,
    *,
    metric_id: UUID,
    validation_run_id: UUID,
    validation_dataset_id: UUID | None,
    scope: str,
    name: str,
    value: float,
    passed: bool,
    source_location: str,
) -> None:
    connection.execute(
        text(
            """
            INSERT INTO public_pbpk_model_validation_metric (
                validation_metric_id, validation_run_id,
                validation_dataset_id, metric_scope, metric_name,
                metric_value, passed, source_location, metadata_json
            ) VALUES (
                CAST(:metric_id AS uuid), CAST(:validation_run_id AS uuid),
                CAST(:validation_dataset_id AS uuid), :scope, :name,
                :value, :passed, :source_location, '{}'::jsonb
            )
            ON CONFLICT (validation_metric_id) DO UPDATE SET
                metric_value = EXCLUDED.metric_value,
                passed = EXCLUDED.passed,
                source_location = EXCLUDED.source_location
            """
        ),
        {
            "metric_id": str(metric_id),
            "validation_run_id": str(validation_run_id),
            "validation_dataset_id": (
                str(validation_dataset_id)
                if validation_dataset_id is not None
                else None
            ),
            "scope": scope,
            "name": name,
            "value": value,
            "passed": passed,
            "source_location": source_location,
        },
    )


def import_licensed_model(
    *,
    bundle_directory: Path,
    database_url: str,
    compound_name: str,
    expected_pubchem_cid: str,
) -> dict[str, Any]:
    bundle = load_licensed_model_bundle(bundle_directory)
    if compound_name.casefold() not in {
        value.casefold() for value in bundle.model.compounds
    }:
        raise LicensedModelImportError(
            f"Requested compound {compound_name!r} is absent from the model."
        )
    engine = create_db_engine(database_url)
    with engine.begin() as connection:
        _require_source(connection)
        compound_id = _require_compound(
            connection,
            compound_name=compound_name,
            expected_pubchem_cid=expected_pubchem_cid,
        )
        model_id = _upsert_registry(
            connection, bundle=bundle, compound_id=compound_id
        )
        model_version_id = _upsert_version(
            connection, bundle=bundle, model_id=model_id
        )
        asset_count = _upsert_assets(
            connection, bundle=bundle, model_version_id=model_version_id
        )
        simulation_count = _upsert_simulations(
            connection, bundle=bundle, model_version_id=model_version_id
        )
        validation_run_id, dataset_count, metric_count = _upsert_validation(
            connection, bundle=bundle, model_version_id=model_version_id
        )
        _require_equal(
            metric_count,
            len(DATASET_METRIC_NAMES) * dataset_count
            + len(POOLED_METRIC_NAMES),
            "Imported validation metric count",
        )
    return {
        "accepted": True,
        "importer_version": IMPORTER_VERSION,
        "model_id": str(model_id),
        "model_version_id": str(model_version_id),
        "validation_run_id": str(validation_run_id),
        "external_model_id": bundle.model.model_id,
        "external_model_version_id": bundle.model.model_version_id,
        "external_validation_run_id": bundle.validation_label.validation_run_id,
        "compound_name": compound_name,
        "compound_pubchem_cid": expected_pubchem_cid,
        "counts": {
            "models": 1,
            "model_versions": 1,
            "assets": asset_count,
            "simulations": simulation_count,
            "validation_runs": 1,
            "validation_datasets": dataset_count,
            "validation_metrics": metric_count,
            "parameter_facts_external": bundle.model.parameter_record_count,
            "component_facts_external": bundle.model.component_record_count,
            "process_facts_external": bundle.model.process_record_count,
        },
        "bundle_path": str(bundle.root),
        "bundle_manifest_sha256": sha256_file(
            bundle.root / "adapter_manifest.json"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and import one audited grade-3 licensed PBPK model into "
            "the public model catalog."
        )
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--compound-name", required=True)
    parser.add_argument("--pubchem-cid", required=True)
    args = parser.parse_args()
    result = import_licensed_model(
        bundle_directory=args.bundle,
        database_url=args.database_url,
        compound_name=args.compound_name,
        expected_pubchem_cid=args.pubchem_cid,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
