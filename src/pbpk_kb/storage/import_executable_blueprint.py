from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from pbpk_kb.blueprints.schemas import (
    BlueprintCompilationReport,
    BlueprintTemplate,
    BlueprintValidationReport,
    PBPKBlueprint,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.recommendation.schemas import ModelingRequest, RequirementProfile


IMPORTER_VERSION = "executable-blueprint-importer-1.0.0"
TEMPLATE_NAMESPACE = uuid.UUID("69e2d962-46c5-47d9-a2b4-438230df0eb0")
VERSION_NAMESPACE = uuid.UUID("4cbd7de3-bc0b-4377-85a1-033789fd88c2")
ARTIFACT_NAMESPACE = uuid.UUID("881c7b22-4be4-4f54-ae20-09bd2b75bb72")
JOB_NAMESPACE = uuid.UUID("0555add8-0e04-4cd7-b739-091f79852ae0")
SIMULATION_NAMESPACE = uuid.UUID("bd991594-8812-46b1-8b93-94841009ae73")
OUTPUT_NAMESPACE = uuid.UUID("fe714947-85f3-40f1-a8aa-fe554dc97d02")


class ExecutableBlueprintImportError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _model_json(value: Any) -> str:
    return _canonical_json(value.model_dump(mode="json"))


def _validate_bundle(bundle_root: Path) -> dict[str, Any]:
    manifest_path = bundle_root / "bundle_manifest.json"
    manifest = _read_json(manifest_path)
    checksum = str(manifest.pop("checksum"))
    if _sha256_json(manifest) != checksum:
        raise ExecutableBlueprintImportError("Bundle manifest checksum mismatch.")
    manifest["checksum"] = checksum
    for item in manifest["files"]:
        path = bundle_root / item["relative_path"]
        if not path.is_file():
            raise ExecutableBlueprintImportError(
                f"Bundle asset is missing: {item['relative_path']}"
            )
        if path.stat().st_size != int(item["byte_size"]):
            raise ExecutableBlueprintImportError(
                f"Bundle asset size drift: {item['relative_path']}"
            )
        if sha256_file(path) != item["sha256"]:
            raise ExecutableBlueprintImportError(
                f"Bundle asset checksum drift: {item['relative_path']}"
            )
    if manifest["engineering_status"] != "executable":
        raise ExecutableBlueprintImportError("Bundle is not executable.")
    if manifest["scientific_status"] != "absorption_structure_failed":
        raise ExecutableBlueprintImportError(
            "Unexpected caffeine scientific status."
        )
    if manifest["registration_allowed"] is not False:
        raise ExecutableBlueprintImportError(
            "Scientifically failed bundle cannot be registration-allowed."
        )
    return manifest


def _artifact_kind(relative_path: str) -> str:
    path = relative_path.casefold()
    if path.endswith(".pksim5"):
        return "compiled_project"
    if path.endswith(".pkml"):
        return "pkml_simulation"
    if path.endswith("-results.csv"):
        return "simulation_results"
    if path.endswith(".xml"):
        return "simulation_structure_xml"
    if "manifest" in path or "audit" in path:
        return "audit_manifest"
    if path.endswith(".json"):
        return "structured_contract"
    if path.endswith(".pdf"):
        return "license"
    if path.endswith(".log"):
        return "runtime_log"
    return "supporting_asset"


def _version_id(blueprint: PBPKBlueprint) -> uuid.UUID:
    return uuid.uuid5(
        VERSION_NAMESPACE,
        f"{blueprint.blueprint_id}\n{blueprint.checksum}",
    )


def _insert_requirement_profile(
    connection: Any,
    *,
    profile: RequirementProfile,
    request: ModelingRequest,
) -> None:
    connection.execute(
        text(
            """
            INSERT INTO requirement_profile (
                profile_id, request_id, compound_id, profile_version,
                ontology_version, ranking_eligibility, request_sha256,
                request_json, profile_checksum, profile_json,
                source_evidence_run_id, created_at
            ) VALUES (
                :profile_id, :request_id, :compound_id, :profile_version,
                :ontology_version, :ranking_eligibility, :request_sha256,
                CAST(:request_json AS jsonb), :profile_checksum,
                CAST(:profile_json AS jsonb), :source_evidence_run_id,
                :created_at
            )
            ON CONFLICT (profile_id) DO UPDATE SET
                request_sha256 = EXCLUDED.request_sha256,
                request_json = EXCLUDED.request_json,
                profile_checksum = EXCLUDED.profile_checksum,
                profile_json = EXCLUDED.profile_json,
                ranking_eligibility = EXCLUDED.ranking_eligibility
            """
        ),
        {
            "profile_id": str(profile.profile_id),
            "request_id": str(profile.request_id),
            "compound_id": str(profile.compound_identity.compound_id),
            "profile_version": profile.profile_version,
            "ontology_version": profile.ontology_version,
            "ranking_eligibility": profile.ranking_eligibility,
            "request_sha256": _sha256_json(request.model_dump(mode="json")),
            "request_json": _model_json(request),
            "profile_checksum": profile.checksum,
            "profile_json": _model_json(profile),
            "source_evidence_run_id": str(profile.source_evidence_run_id),
            "created_at": profile.created_at,
        },
    )


def _insert_template(
    connection: Any,
    *,
    template: BlueprintTemplate,
    manifest: dict[str, Any],
    bundle_root: Path,
) -> uuid.UUID:
    template_checksum = _sha256_json(template.model_dump(mode="json"))
    template_id = uuid.uuid5(
        TEMPLATE_NAMESPACE,
        f"{template.template_id}\n{template.template_version}\n{template_checksum}",
    )
    runtime_version = (
        template.target_runtime.pksim_cli_version
        or template.target_runtime.ospsuite_r_version
        or "unknown"
    )
    persisted_template_id = connection.execute(
        text(
            """
            INSERT INTO pbpk_blueprint_template (
                template_id, external_template_id, template_version,
                compound_id, target_runtime_adapter, runtime_version,
                source_repository, source_commit, source_manifest_sha256,
                source_project_path, source_project_sha256,
                source_skeleton_path, source_skeleton_sha256,
                license_spdx_id, license_source_location, license_sha256,
                structural_status, scientific_status, template_eligible,
                scientific_registration_allowed, template_checksum,
                template_json, asset_root, source_id, source_location
            ) VALUES (
                :template_id, :external_template_id, :template_version,
                :compound_id, :target_runtime_adapter, :runtime_version,
                :source_repository, :source_commit, :source_manifest_sha256,
                :source_project_path, :source_project_sha256,
                :source_skeleton_path, :source_skeleton_sha256,
                :license_spdx_id, :license_source_location, :license_sha256,
                :structural_status, :scientific_status, TRUE, FALSE,
                :template_checksum, CAST(:template_json AS jsonb),
                :asset_root, :source_id, :source_location
            )
            ON CONFLICT (external_template_id, template_version) DO UPDATE SET
                target_runtime_adapter = EXCLUDED.target_runtime_adapter,
                runtime_version = EXCLUDED.runtime_version,
                source_manifest_sha256 = EXCLUDED.source_manifest_sha256,
                source_project_sha256 = EXCLUDED.source_project_sha256,
                source_skeleton_sha256 = EXCLUDED.source_skeleton_sha256,
                structural_status = EXCLUDED.structural_status,
                scientific_status = EXCLUDED.scientific_status,
                template_eligible = EXCLUDED.template_eligible,
                scientific_registration_allowed =
                    EXCLUDED.scientific_registration_allowed,
                template_checksum = EXCLUDED.template_checksum,
                template_json = EXCLUDED.template_json,
                asset_root = EXCLUDED.asset_root,
                source_location = EXCLUDED.source_location,
                updated_at = now()
            RETURNING template_id
            """
        ),
        {
            "template_id": str(template_id),
            "external_template_id": template.template_id,
            "template_version": template.template_version,
            "compound_id": manifest["compound_id"],
            "target_runtime_adapter": template.target_runtime.adapter,
            "runtime_version": runtime_version,
            "source_repository": template.source_repository,
            "source_commit": template.source_commit,
            "source_manifest_sha256": template.source_manifest_sha256,
            "source_project_path": template.source_project_relative_path,
            "source_project_sha256": template.source_project_sha256,
            "source_skeleton_path": template.source_skeleton_relative_path,
            "source_skeleton_sha256": template.source_skeleton_sha256,
            "license_spdx_id": template.license_id,
            "license_source_location": template.license_relative_path,
            "license_sha256": template.license_sha256,
            "structural_status": template.validation_status,
            "scientific_status": manifest["scientific_status"],
            "template_checksum": template_checksum,
            "template_json": _model_json(template),
            "asset_root": str(bundle_root),
            "source_id": "osp_pbpk_model_library",
            "source_location": (
                f"BUNDLE:{manifest['checksum']}#generated/template.json"
            ),
        },
    ).scalar_one()
    return uuid.UUID(str(persisted_template_id))


def _insert_blueprint_versions(
    connection: Any,
    *,
    versions: list[PBPKBlueprint],
    profile: RequirementProfile,
    template_id: uuid.UUID,
    validation: BlueprintValidationReport,
    compilation: BlueprintCompilationReport,
    manifest: dict[str, Any],
) -> dict[str, uuid.UUID]:
    current = versions[-1]
    connection.execute(
        text(
            """
            INSERT INTO pbpk_blueprint (
                blueprint_id, compound_id, profile_id, template_id,
                ranking_run_id, objective, current_status, scientific_status,
                registration_allowed, blocker_codes, created_at, updated_at
            ) VALUES (
                :blueprint_id, :compound_id, :profile_id, :template_id,
                NULL, :objective, :current_status, :scientific_status,
                FALSE, :blocker_codes, :created_at, :updated_at
            )
            ON CONFLICT (blueprint_id) DO UPDATE SET
                current_status = EXCLUDED.current_status,
                scientific_status = EXCLUDED.scientific_status,
                registration_allowed = EXCLUDED.registration_allowed,
                blocker_codes = EXCLUDED.blocker_codes,
                updated_at = EXCLUDED.updated_at
            """
        ),
        {
            "blueprint_id": str(current.blueprint_id),
            "compound_id": manifest["compound_id"],
            "profile_id": str(profile.profile_id),
            "template_id": str(template_id),
            "objective": current.objective,
            "current_status": current.status,
            "scientific_status": manifest["scientific_status"],
            "blocker_codes": manifest["blocker_codes"],
            "created_at": versions[0].created_at,
            "updated_at": current.created_at,
        },
    )
    version_ids: dict[str, uuid.UUID] = {}
    for blueprint in versions:
        version_id = _version_id(blueprint)
        version_ids[blueprint.status] = version_id
        validation_json = (
            _model_json(validation)
            if blueprint.status in {
                "structurally_validated",
                "compiled",
                "executable",
            }
            else None
        )
        compilation_json = (
            _model_json(compilation)
            if blueprint.status in {"compiled", "executable"}
            else None
        )
        connection.execute(
            text(
                """
                INSERT INTO blueprint_version (
                    blueprint_version_id, blueprint_id, blueprint_version,
                    parent_blueprint_checksum, status, blueprint_checksum,
                    blueprint_json, validation_report_json,
                    compilation_report_json, source_location, created_at
                ) VALUES (
                    :version_id, :blueprint_id, :blueprint_version,
                    :parent_checksum, :status, :checksum,
                    CAST(:blueprint_json AS jsonb),
                    CAST(:validation_json AS jsonb),
                    CAST(:compilation_json AS jsonb),
                    :source_location, :created_at
                )
                ON CONFLICT (blueprint_version_id) DO UPDATE SET
                    blueprint_json = EXCLUDED.blueprint_json,
                    validation_report_json = EXCLUDED.validation_report_json,
                    compilation_report_json = EXCLUDED.compilation_report_json,
                    source_location = EXCLUDED.source_location
                """
            ),
            {
                "version_id": str(version_id),
                "blueprint_id": str(blueprint.blueprint_id),
                "blueprint_version": blueprint.blueprint_version,
                "parent_checksum": blueprint.parent_blueprint_checksum,
                "status": blueprint.status,
                "checksum": blueprint.checksum,
                "blueprint_json": _model_json(blueprint),
                "validation_json": validation_json,
                "compilation_json": compilation_json,
                "source_location": (
                    f"BUNDLE:{manifest['checksum']}#generated/"
                    + {
                        "draft_grounded": "draft_blueprint.json",
                        "structurally_validated": (
                            "structurally_validated_blueprint.json"
                        ),
                        "compiled": "compiled_blueprint.json",
                        "executable": "executable_blueprint.json",
                    }[blueprint.status]
                ),
                "created_at": blueprint.created_at,
            },
        )
        for kind, components in (
            ("compartment", blueprint.compartments),
            ("process", blueprint.processes),
            ("dosing_regimen", blueprint.dosing_regimens),
            ("simulation_scenario", blueprint.simulation_scenarios),
        ):
            for component in components:
                payload = component.model_dump(mode="json")
                component_id = str(
                    payload.get("compartment_id")
                    or payload.get("process_id")
                    or payload.get("regimen_id")
                    or payload.get("scenario_id")
                )
                provenance = payload.get("provenance") or payload.get("evidence") or []
                source = provenance[0] if provenance else {}
                source_checksum = str(
                    source.get("source_checksum")
                    or manifest["checksum"]
                )
                source_location = str(
                    source.get("source_location")
                    or f"BLUEPRINT:{blueprint.checksum}#{kind}/{component_id}"
                )
                connection.execute(
                    text(
                        """
                        INSERT INTO blueprint_component (
                            blueprint_version_id, component_id,
                            component_kind, semantic_id, payload_json,
                            source_location, source_checksum
                        ) VALUES (
                            :version_id, :component_id, :kind, :semantic_id,
                            CAST(:payload AS jsonb), :source_location,
                            :source_checksum
                        )
                        ON CONFLICT (
                            blueprint_version_id, component_kind, component_id
                        ) DO UPDATE SET
                            payload_json = EXCLUDED.payload_json,
                            source_location = EXCLUDED.source_location,
                            source_checksum = EXCLUDED.source_checksum
                        """
                    ),
                    {
                        "version_id": str(version_id),
                        "component_id": component_id,
                        "kind": kind,
                        "semantic_id": payload.get("semantic_id"),
                        "payload": _canonical_json(payload),
                        "source_location": source_location,
                        "source_checksum": source_checksum,
                    },
                )
        for parameter in blueprint.parameters:
            payload = parameter.model_dump(mode="json")
            connection.execute(
                text(
                    """
                    INSERT INTO blueprint_parameter (
                        blueprint_version_id, parameter_id, semantic_id,
                        parameter_name, origin, value, initial_value,
                        canonical_unit, blocking, calibration_run_id,
                        provenance_json, payload_json
                    ) VALUES (
                        :version_id, :parameter_id, :semantic_id,
                        :parameter_name, :origin, :value, :initial_value,
                        :canonical_unit, :blocking, :calibration_run_id,
                        CAST(:provenance_json AS jsonb),
                        CAST(:payload_json AS jsonb)
                    )
                    ON CONFLICT (blueprint_version_id, parameter_id) DO UPDATE SET
                        origin = EXCLUDED.origin,
                        value = EXCLUDED.value,
                        initial_value = EXCLUDED.initial_value,
                        provenance_json = EXCLUDED.provenance_json,
                        payload_json = EXCLUDED.payload_json
                    """
                ),
                {
                    "version_id": str(version_id),
                    "parameter_id": parameter.parameter_id,
                    "semantic_id": parameter.semantic_id,
                    "parameter_name": parameter.name,
                    "origin": parameter.origin,
                    "value": getattr(parameter, "value", None),
                    "initial_value": getattr(parameter, "initial_value", None),
                    "canonical_unit": parameter.canonical_unit,
                    "blocking": parameter.blocking,
                    "calibration_run_id": str(
                        getattr(parameter, "calibration_run_id", "") or ""
                    ) or None,
                    "provenance_json": _canonical_json(
                        [item.model_dump(mode="json") for item in parameter.provenance]
                    ),
                    "payload_json": _canonical_json(payload),
                },
            )
    return version_ids


def _insert_artifacts(
    connection: Any,
    *,
    version_id: uuid.UUID,
    bundle_root: Path,
    manifest: dict[str, Any],
) -> int:
    rows = list(manifest["files"])
    for relative_path in (
        "bundle_manifest.json",
        "caffeine_v5_blueprint_bundle.zip",
    ):
        path = bundle_root / relative_path
        if path.is_file():
            rows.append(
                {
                    "relative_path": relative_path,
                    "sha256": sha256_file(path),
                    "byte_size": path.stat().st_size,
                }
            )
    for item in rows:
        artifact_id = uuid.uuid5(
            ARTIFACT_NAMESPACE,
            f"{version_id}\n{item['relative_path']}\n{item['sha256']}",
        )
        connection.execute(
            text(
                """
                INSERT INTO blueprint_artifact (
                    blueprint_artifact_id, blueprint_version_id,
                    artifact_kind, relative_path, sha256, byte_size,
                    source_location, metadata_json
                ) VALUES (
                    :artifact_id, :version_id, :kind, :relative_path,
                    :sha256, :byte_size, :source_location,
                    CAST(:metadata AS jsonb)
                )
                ON CONFLICT (blueprint_artifact_id) DO UPDATE SET
                    sha256 = EXCLUDED.sha256,
                    byte_size = EXCLUDED.byte_size,
                    metadata_json = EXCLUDED.metadata_json
                """
            ),
            {
                "artifact_id": str(artifact_id),
                "version_id": str(version_id),
                "kind": _artifact_kind(item["relative_path"]),
                "relative_path": item["relative_path"],
                "sha256": item["sha256"],
                "byte_size": int(item["byte_size"]),
                "source_location": (
                    f"BUNDLE:{manifest['checksum']}#{item['relative_path']}"
                ),
                "metadata": _canonical_json(
                    {"importer_version": IMPORTER_VERSION}
                ),
            },
        )
    return len(rows)


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _insert_runtime_audit(
    connection: Any,
    *,
    version_id: uuid.UUID,
    bundle_root: Path,
    manifest: dict[str, Any],
    compilation: BlueprintCompilationReport,
) -> tuple[int, int, int]:
    completed_at = _read_json(
        bundle_root / "runtime/windows_run_manifest.json"
    ).get("CompletedAtUtc")
    compile_job_id = uuid.uuid5(JOB_NAMESPACE, compilation.checksum)
    simulation_manifest_sha256 = sha256_file(
        bundle_root / "runtime/files.sha256.csv"
    )
    simulate_job_id = uuid.uuid5(JOB_NAMESPACE, simulation_manifest_sha256)
    jobs = [
        {
            "job_id": compile_job_id,
            "job_type": "compile",
            "idempotency_key": _sha256_json(
                {"job": "compile", "checksum": compilation.checksum}
            ),
            "progress_current": 1,
            "progress_total": 1,
            "input_manifest_sha256": compilation.input_blueprint_checksum,
            "output_manifest_sha256": compilation.checksum,
            "input_json": {
                "blueprint_checksum": compilation.input_blueprint_checksum
            },
            "output_json": compilation.model_dump(mode="json"),
        },
        {
            "job_id": simulate_job_id,
            "job_type": "simulate",
            "idempotency_key": _sha256_json(
                {"job": "simulate", "checksum": simulation_manifest_sha256}
            ),
            "progress_current": 14,
            "progress_total": 14,
            "input_manifest_sha256": compilation.checksum,
            "output_manifest_sha256": simulation_manifest_sha256,
            "input_json": {"runs": 2, "scenarios": 7},
            "output_json": {
                "runtime": manifest["runtime"],
                "source_location": (
                    f"BUNDLE:{manifest['checksum']}#simulation_outputs"
                ),
            },
        },
    ]
    for job in jobs:
        connection.execute(
            text(
                """
                INSERT INTO runtime_job (
                    job_id, job_type, idempotency_key, status, attempt,
                    progress_current, progress_total, worker_id, started_at,
                    heartbeat_at, finished_at, input_manifest_sha256,
                    output_manifest_sha256, input_json, output_json
                ) VALUES (
                    :job_id, :job_type, :idempotency_key, 'succeeded', 1,
                    :progress_current, :progress_total, 'windows:WENJUN',
                    :finished_at, :finished_at, :finished_at,
                    :input_manifest_sha256, :output_manifest_sha256,
                    CAST(:input_json AS jsonb), CAST(:output_json AS jsonb)
                )
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    progress_current = EXCLUDED.progress_current,
                    progress_total = EXCLUDED.progress_total,
                    heartbeat_at = EXCLUDED.heartbeat_at,
                    finished_at = EXCLUDED.finished_at,
                    output_manifest_sha256 = EXCLUDED.output_manifest_sha256,
                    output_json = EXCLUDED.output_json
                """
            ),
            {
                **{key: value for key, value in job.items() if key not in {"input_json", "output_json"}},
                "job_id": str(job["job_id"]),
                "finished_at": completed_at or datetime.now(UTC),
                "input_json": _canonical_json(job["input_json"]),
                "output_json": _canonical_json(job["output_json"]),
            },
        )
        connection.execute(
            text("DELETE FROM runtime_job_event WHERE job_id = :job_id"),
            {"job_id": str(job["job_id"])},
        )
        connection.execute(
            text(
                """
                INSERT INTO runtime_job_event (
                    job_id, event_type, status, message, payload_json
                ) VALUES (
                    :job_id, 'imported_completed_run', 'succeeded',
                    :message, CAST(:payload AS jsonb)
                )
                """
            ),
            {
                "job_id": str(job["job_id"]),
                "message": "Imported a completed checksum-locked Windows run.",
                "payload": _canonical_json(
                    {"bundle_checksum": manifest["checksum"]}
                ),
            },
        )

    output_count = 0
    simulation_count = 0
    for run_ordinal in (1, 2):
        run_root = bundle_root / f"simulation_outputs/run_{run_ordinal}"
        for scenario_root in sorted(path for path in run_root.iterdir() if path.is_dir()):
            scenario_id = scenario_root.name
            files = sorted(path for path in scenario_root.iterdir() if path.is_file())
            result_manifest_sha256 = _sha256_json(
                {path.name: sha256_file(path) for path in files}
            )
            simulation_run_id = uuid.uuid5(
                SIMULATION_NAMESPACE,
                f"{version_id}\n{scenario_id}\n{run_ordinal}",
            )
            connection.execute(
                text(
                    """
                    INSERT INTO simulation_run (
                        simulation_run_id, blueprint_version_id, job_id,
                        scenario_id, run_ordinal, status, runtime_json,
                        result_manifest_sha256, source_location, finished_at
                    ) VALUES (
                        :run_id, :version_id, :job_id, :scenario_id,
                        :run_ordinal, 'succeeded', CAST(:runtime_json AS jsonb),
                        :result_manifest_sha256, :source_location, :finished_at
                    )
                    ON CONFLICT (
                        blueprint_version_id, scenario_id, run_ordinal
                    ) DO UPDATE SET
                        status = EXCLUDED.status,
                        runtime_json = EXCLUDED.runtime_json,
                        result_manifest_sha256 =
                            EXCLUDED.result_manifest_sha256,
                        source_location = EXCLUDED.source_location,
                        finished_at = EXCLUDED.finished_at
                    """
                ),
                {
                    "run_id": str(simulation_run_id),
                    "version_id": str(version_id),
                    "job_id": str(simulate_job_id),
                    "scenario_id": scenario_id,
                    "run_ordinal": run_ordinal,
                    "runtime_json": _canonical_json(manifest["runtime"]),
                    "result_manifest_sha256": result_manifest_sha256,
                    "source_location": (
                        f"BUNDLE:{manifest['checksum']}#simulation_outputs/"
                        f"run_{run_ordinal}/{scenario_id}"
                    ),
                    "finished_at": completed_at or datetime.now(UTC),
                },
            )
            simulation_count += 1
            for path in files:
                relative_path = path.relative_to(bundle_root).as_posix()
                suffix = path.suffix.casefold()
                output_kind = {
                    ".csv": "simulation_results",
                    ".pkml": "pkml_simulation",
                    ".xml": "simulation_structure_xml",
                }.get(suffix, "supporting_output")
                output_id = uuid.uuid5(
                    OUTPUT_NAMESPACE,
                    f"{simulation_run_id}\n{relative_path}",
                )
                connection.execute(
                    text(
                        """
                        INSERT INTO simulation_output (
                            simulation_output_id, simulation_run_id,
                            output_kind, relative_path, sha256, byte_size,
                            row_count, source_location, metadata_json
                        ) VALUES (
                            :output_id, :run_id, :output_kind,
                            :relative_path, :sha256, :byte_size, :row_count,
                            :source_location, CAST(:metadata AS jsonb)
                        )
                        ON CONFLICT (simulation_output_id) DO UPDATE SET
                            sha256 = EXCLUDED.sha256,
                            byte_size = EXCLUDED.byte_size,
                            row_count = EXCLUDED.row_count,
                            metadata_json = EXCLUDED.metadata_json
                        """
                    ),
                    {
                        "output_id": str(output_id),
                        "run_id": str(simulation_run_id),
                        "output_kind": output_kind,
                        "relative_path": relative_path,
                        "sha256": sha256_file(path),
                        "byte_size": path.stat().st_size,
                        "row_count": _csv_row_count(path) if suffix == ".csv" else None,
                        "source_location": (
                            f"BUNDLE:{manifest['checksum']}#{relative_path}"
                        ),
                        "metadata": _canonical_json(
                            {"scenario_id": scenario_id, "run_ordinal": run_ordinal}
                        ),
                    },
                )
                output_count += 1
    return len(jobs), simulation_count, output_count


def import_executable_blueprint(
    *,
    bundle_root: Path,
    database_url: str,
) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    manifest = _validate_bundle(bundle_root)
    generated = bundle_root / "generated"
    request = ModelingRequest.model_validate(
        _read_json(generated / "modeling_request.json")
    )
    profile = RequirementProfile.model_validate(
        _read_json(generated / "requirement_profile.json")
    )
    template = BlueprintTemplate.model_validate(
        _read_json(generated / "template.json")
    )
    versions = [
        PBPKBlueprint.model_validate(
            _read_json(generated / filename)
        )
        for filename in (
            "draft_blueprint.json",
            "structurally_validated_blueprint.json",
            "compiled_blueprint.json",
            "executable_blueprint.json",
        )
    ]
    validation = BlueprintValidationReport.model_validate(
        _read_json(generated / "validation_report.json")
    )
    compilation = BlueprintCompilationReport.model_validate(
        _read_json(generated / "compilation_report.json")
    )
    if [item.status for item in versions] != [
        "draft_grounded",
        "structurally_validated",
        "compiled",
        "executable",
    ]:
        raise ExecutableBlueprintImportError(
            "Blueprint status lineage is incomplete or out of order."
        )
    for parent, child in zip(versions[:-1], versions[1:], strict=True):
        if child.parent_blueprint_checksum != parent.checksum:
            raise ExecutableBlueprintImportError(
                "Blueprint parent checksum lineage is broken."
            )

    engine = create_engine(database_url, pool_pre_ping=True)
    with engine.begin() as connection:
        compound_exists = connection.execute(
            text(
                "SELECT 1 FROM compound_identity WHERE compound_id = :compound_id"
            ),
            {"compound_id": manifest["compound_id"]},
        ).scalar_one_or_none()
        if compound_exists is None:
            raise ExecutableBlueprintImportError(
                "Caffeine compound identity must be imported first."
            )
        _insert_requirement_profile(
            connection,
            profile=profile,
            request=request,
        )
        template_id = _insert_template(
            connection,
            template=template,
            manifest=manifest,
            bundle_root=bundle_root,
        )
        version_ids = _insert_blueprint_versions(
            connection,
            versions=versions,
            profile=profile,
            template_id=template_id,
            validation=validation,
            compilation=compilation,
            manifest=manifest,
        )
        artifact_count = _insert_artifacts(
            connection,
            version_id=version_ids["executable"],
            bundle_root=bundle_root,
            manifest=manifest,
        )
        job_count, simulation_count, output_count = _insert_runtime_audit(
            connection,
            version_id=version_ids["executable"],
            bundle_root=bundle_root,
            manifest=manifest,
            compilation=compilation,
        )
    return {
        "status": "imported",
        "importer_version": IMPORTER_VERSION,
        "bundle_checksum": manifest["checksum"],
        "template_id": str(template_id),
        "blueprint_id": str(versions[-1].blueprint_id),
        "blueprint_status": versions[-1].status,
        "scientific_status": manifest["scientific_status"],
        "registration_allowed": False,
        "version_count": len(version_ids),
        "artifact_count": artifact_count,
        "job_count": job_count,
        "simulation_run_count": simulation_count,
        "simulation_output_count": output_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import an audited executable blueprint bundle."
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--database-url", required=True)
    args = parser.parse_args()
    result = import_executable_blueprint(
        bundle_root=args.bundle_root,
        database_url=args.database_url,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
