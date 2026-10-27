from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import Field

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    ModelAsset,
    OSPExecutionPlan,
    OSPExecutionProjectSpec,
    OSPExecutionSimulationSpec,
    OSPProjectParseResult,
    OSPRuntimeLock,
    RepositoryInventorySummary,
    RuntimeDownloadLock,
    StrictModel,
)


class ModelRepositoryConfig(StrictModel):
    source_id: str
    repository_url: str
    release_tag: str
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")


class SuiteConfig(StrictModel):
    release_name: str
    version: str
    installer_name: str
    installer_url: str
    installer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PKSimConfig(StrictModel):
    version: str
    portable_name: str
    portable_url: str
    portable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cli_relative_path: str
    export_flags: list[Literal["-k", "-x", "-c", "-r"]]


class OSPSuiteRConfig(StrictModel):
    version: str
    linux_binary_name: str
    linux_binary_url: str
    linux_binary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutionConfig(StrictModel):
    worker_platform: Literal["Windows"]
    command_timeout_seconds: int = Field(gt=0)
    require_all_projects: Literal[True]
    require_all_simulations: Literal[True]
    immutable_outputs: Literal[True]
    required_export_extensions: list[Literal[".pkml", ".xml", ".csv"]]


class ModelRuntimeConfig(StrictModel):
    schema_version: Literal["1.0.0"]
    model_repository: ModelRepositoryConfig
    osp_suite: SuiteConfig
    pksim: PKSimConfig
    ospsuite_r: OSPSuiteRConfig
    execution: ExecutionConfig


class ExecutionPlanError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")

    def json_default(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if isinstance(item, datetime):
            return item.isoformat()
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json_atomic(path: Path, value: Any, *, canonical: bool = False) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if canonical:
        payload = canonical_json_bytes(value) + b"\n"
    else:
        payload = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def read_runtime_config(path: Path) -> ModelRuntimeConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelRuntimeConfig.model_validate(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ExecutionPlanError(
                    f"{path}:{line_number} is not valid JSON: {exc}"
                ) from exc
    return records


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ExecutionPlanError(f"Unsafe repository-relative path: {value!r}")
    return path.as_posix()


def _runtime_lock(config: ModelRuntimeConfig) -> OSPRuntimeLock:
    return OSPRuntimeLock(
        suite_release_name=config.osp_suite.release_name,
        suite_version=config.osp_suite.version,
        suite_installer=RuntimeDownloadLock(
            name=config.osp_suite.installer_name,
            url=config.osp_suite.installer_url,
            sha256=config.osp_suite.installer_sha256,
        ),
        pksim_version=config.pksim.version,
        pksim_portable=RuntimeDownloadLock(
            name=config.pksim.portable_name,
            url=config.pksim.portable_url,
            sha256=config.pksim.portable_sha256,
        ),
        pksim_cli_relative_path=config.pksim.cli_relative_path,
        ospsuite_r_version=config.ospsuite_r.version,
        ospsuite_r_linux_binary=RuntimeDownloadLock(
            name=config.ospsuite_r.linux_binary_name,
            url=config.ospsuite_r.linux_binary_url,
            sha256=config.ospsuite_r.linux_binary_sha256,
        ),
    )


def build_execution_plan(
    *,
    inventory_directory: Path,
    repository_root: Path,
    runtime_config_path: Path,
) -> OSPExecutionPlan:
    inventory_directory = inventory_directory.resolve()
    repository_root = repository_root.resolve()
    config = read_runtime_config(runtime_config_path)
    summary_path = inventory_directory / "summary.json"
    summary = RepositoryInventorySummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    if not summary.gate1_static_acceptance:
        raise ExecutionPlanError(
            "Static inventory acceptance failed; an execution plan cannot be created."
        )
    if summary.repository_commit != config.model_repository.commit_sha:
        raise ExecutionPlanError(
            "Inventory commit does not match the locked runtime configuration."
        )
    if summary.source_id != config.model_repository.source_id:
        raise ExecutionPlanError(
            "Inventory source does not match the locked runtime configuration."
        )
    if config.pksim.export_flags != ["-k", "-x", "-c", "-r"]:
        raise ExecutionPlanError(
            "Runtime configuration must export PKML, XML, CSV and run simulations."
        )
    if config.execution.required_export_extensions != [".pkml", ".xml", ".csv"]:
        raise ExecutionPlanError(
            "Runtime configuration must require PKML, XML and CSV outputs."
        )

    assets = [
        ModelAsset.model_validate(record)
        for record in _read_jsonl(inventory_directory / "assets.jsonl")
    ]
    project_assets = {
        asset.relative_path: asset
        for asset in assets
        if asset.asset_kind.value == "pksim_project"
    }
    parsed_projects = [
        OSPProjectParseResult.model_validate(record)
        for record in _read_jsonl(inventory_directory / "projects.jsonl")
    ]
    projects: list[OSPExecutionProjectSpec] = []
    for parsed in parsed_projects:
        try:
            relative_path = Path(parsed.project_path).resolve().relative_to(
                repository_root
            ).as_posix()
        except ValueError as exc:
            raise ExecutionPlanError(
                f"Parsed project is outside the locked repository: {parsed.project_path}"
            ) from exc
        relative_path = _safe_relative_path(relative_path)
        asset = project_assets.get(relative_path)
        if asset is None:
            raise ExecutionPlanError(
                f"Parsed project has no matching inventory asset: {relative_path}"
            )
        actual_path = repository_root / relative_path
        actual_sha256 = sha256_file(actual_path)
        if actual_sha256 != parsed.project_sha256 or actual_sha256 != asset.sha256:
            raise ExecutionPlanError(
                f"Project checksum drift detected for {relative_path}."
            )
        simulations = [
            OSPExecutionSimulationSpec(
                simulation_id=item.simulation_id,
                name=item.name,
                simulation_mode=item.simulation_mode,
            )
            for item in parsed.simulations
        ]
        if not simulations:
            raise ExecutionPlanError(
                f"Project has no simulations and cannot be executed: {relative_path}"
            )
        project_key = hashlib.sha256(
            f"{summary.repository_commit}\n{relative_path}\n{actual_sha256}".encode()
        ).hexdigest()[:32]
        projects.append(
            OSPExecutionProjectSpec(
                project_key=project_key,
                model_family=PurePosixPath(relative_path).parts[0],
                relative_path=relative_path,
                project_sha256=actual_sha256,
                project_name=parsed.project_name,
                project_version=parsed.project_version,
                simulations=simulations,
            )
        )

    projects.sort(key=lambda item: item.relative_path.casefold())
    if len(projects) != summary.model_project_count:
        raise ExecutionPlanError(
            f"Expected {summary.model_project_count} projects; built {len(projects)}."
        )
    if set(project_assets) != {item.relative_path for item in projects}:
        raise ExecutionPlanError(
            "Project asset and parsed project path sets are not identical."
        )
    identity = {
        "source_id": summary.source_id,
        "repository_commit": summary.repository_commit,
        "inventory_summary_sha256": sha256_file(summary_path),
        "runtime": _runtime_lock(config),
        "projects": projects,
    }
    job_id = f"osp-exec-{canonical_sha256(identity)[:32]}"
    return OSPExecutionPlan(
        job_id=job_id,
        created_at=summary.generated_at,
        source_id=summary.source_id,
        repository_url=config.model_repository.repository_url,
        repository_release_tag=config.model_repository.release_tag,
        repository_commit=summary.repository_commit,
        inventory_run_id=inventory_directory.name,
        inventory_summary_sha256=sha256_file(summary_path),
        runtime=_runtime_lock(config),
        export_flags=config.pksim.export_flags,
        command_timeout_seconds=config.execution.command_timeout_seconds,
        required_export_extensions=config.execution.required_export_extensions,
        projects=projects,
    )


def write_execution_plan(output_directory: Path, plan: OSPExecutionPlan) -> tuple[Path, str]:
    output_directory.mkdir(parents=True, exist_ok=False)
    plan_path = output_directory / "execution_plan.json"
    plan_sha256 = write_json_atomic(plan_path, plan, canonical=True)
    sidecar = output_directory / "execution_plan.sha256"
    write_json_atomic(
        sidecar,
        {"file": plan_path.name, "sha256": plan_sha256},
        canonical=True,
    )
    return plan_path, plan_sha256


def load_execution_plan(path: Path) -> tuple[OSPExecutionPlan, str]:
    plan = OSPExecutionPlan.model_validate_json(path.read_text(encoding="utf-8"))
    plan_sha256 = sha256_file(path)
    sidecar_path = path.with_name("execution_plan.sha256")
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("file") != path.name or sidecar.get("sha256") != plan_sha256:
            raise ExecutionPlanError("Execution plan checksum sidecar does not match.")
    return plan, plan_sha256
