from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbpk_kb.model_assets.execution_plan import (
    load_execution_plan,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file


class WorkerBundleError(RuntimeError):
    pass


WORKER_SOURCE_FILES = (
    "pbpk_kb/__init__.py",
    "pbpk_kb/model_assets/__init__.py",
    "pbpk_kb/model_assets/models.py",
    "pbpk_kb/model_assets/license_policy.py",
    "pbpk_kb/model_assets/execution_plan.py",
    "pbpk_kb/model_assets/windows_extract.py",
    "pbpk_kb/model_assets/windows_worker.py",
    "scripts/windows/requirements-osp-worker.txt",
    "scripts/windows/run_osp_execution_worker.ps1",
)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _git_archive(
    *,
    repository_root: Path,
    commit: str,
    destination: Path,
    prefix: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "archive",
            "--format=zip",
            f"--prefix={prefix}/",
            f"--output={temporary}",
            commit,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise WorkerBundleError(result.stderr.strip() or result.stdout.strip())
    temporary.replace(destination)


def _filesystem_archive(
    *,
    repository_root: Path,
    destination: Path,
    prefix: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="x",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for path in sorted(
            (
                path
                for path in repository_root.rglob("*")
                if path.is_file()
                and ".git" not in path.relative_to(repository_root).parts
            ),
            key=lambda path: (
                path.relative_to(repository_root).as_posix().casefold()
            ),
        ):
            archive.write(
                path,
                arcname=(
                    f"{prefix}/"
                    f"{path.relative_to(repository_root).as_posix()}"
                ),
            )
    with zipfile.ZipFile(temporary) as archive:
        broken = archive.testzip()
        if broken is not None:
            temporary.unlink(missing_ok=True)
            raise WorkerBundleError(
                f"Repository snapshot ZIP validation failed at {broken}."
            )
    temporary.replace(destination)


def _repository_root_name(source_id: str, release_tag: str) -> str:
    if source_id == "osp_pbpk_model_library":
        return f"OSP-PBPK-Model-Library-{release_tag}"
    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        f"{source_id}-{release_tag}",
    ).strip("-.")
    if not value:
        raise WorkerBundleError("Repository archive root name is empty.")
    return value


def _verify_project_inputs(
    *,
    repository_root: Path,
    plan: Any,
) -> None:
    for project in plan.projects:
        path = (repository_root / project.relative_path).resolve()
        try:
            path.relative_to(repository_root)
        except ValueError as exc:
            raise WorkerBundleError(
                f"Project escapes repository root: {project.relative_path}"
            ) from exc
        if not path.is_file():
            raise WorkerBundleError(
                f"Planned project is missing: {project.relative_path}"
            )
        if sha256_file(path) != project.project_sha256:
            raise WorkerBundleError(
                f"Planned project checksum drift: {project.relative_path}"
            )


def _bundle_file_records(bundle_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(
        (path for path in bundle_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(bundle_root).as_posix().casefold(),
    ):
        if path.name == "bundle_manifest.json":
            continue
        records.append(
            {
                "relative_path": path.relative_to(bundle_root).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _zip_bundle(bundle_root: Path, archive_path: Path) -> str:
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="x",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for path in sorted(
            (path for path in bundle_root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(bundle_root).as_posix().casefold(),
        ):
            archive.write(
                path,
                arcname=(
                    f"{bundle_root.name}/"
                    f"{path.relative_to(bundle_root).as_posix()}"
                ),
            )
    with zipfile.ZipFile(temporary) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise WorkerBundleError(f"Worker bundle ZIP validation failed at {broken}.")
    temporary.replace(archive_path)
    return sha256_file(archive_path)


def build_windows_worker_bundle(
    *,
    project_root: Path,
    plan_path: Path,
    repository_root: Path,
    runtime_archive: Path,
    output_directory: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    repository_root = repository_root.resolve()
    runtime_archive = runtime_archive.resolve()
    plan, plan_sha256 = load_execution_plan(plan_path)
    if sha256_file(runtime_archive) != plan.runtime.pksim_portable.sha256:
        raise WorkerBundleError("Runtime archive checksum does not match the plan.")
    _verify_project_inputs(repository_root=repository_root, plan=plan)
    is_git_checkout = (repository_root / ".git").is_dir()
    if is_git_checkout:
        resolved_commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            resolved_commit.returncode
            or resolved_commit.stdout.strip() != plan.repository_commit
        ):
            raise WorkerBundleError(
                "Repository commit does not match the execution plan."
            )
    if output_directory.exists():
        raise WorkerBundleError(f"Bundle output already exists: {output_directory}")
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        raise WorkerBundleError(f"Stale bundle temporary directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for relative_path in WORKER_SOURCE_FILES:
            source = project_root / relative_path
            if not source.is_file():
                raise WorkerBundleError(f"Worker source file is missing: {source}")
            destination_relative = (
                Path(relative_path).name
                if relative_path.startswith("scripts/windows/")
                else Path(relative_path)
            )
            _copy_file(source, temporary / destination_relative)
        _copy_file(plan_path, temporary / "execution_plan.json")
        _copy_file(
            plan_path.with_name("execution_plan.sha256"),
            temporary / "execution_plan.sha256",
        )
        _copy_file(
            runtime_archive,
            temporary / "runtime" / plan.runtime.pksim_portable.name,
        )
        repository_root_name = _repository_root_name(
            plan.source_id,
            plan.repository_release_tag,
        )
        repository_archive_relative_path = (
            f"repository/{repository_root_name}.zip"
        )
        repository_archive = (
            temporary / repository_archive_relative_path
        )
        if is_git_checkout:
            _git_archive(
                repository_root=repository_root,
                commit=plan.repository_commit,
                destination=repository_archive,
                prefix=repository_root_name,
            )
            repository_snapshot_mode = "git_checkout"
        else:
            _filesystem_archive(
                repository_root=repository_root,
                destination=repository_archive,
                prefix=repository_root_name,
            )
            repository_snapshot_mode = "pinned_files"
        files = _bundle_file_records(temporary)
        manifest = {
            "bundle_schema_version": "1.1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "job_id": plan.job_id,
            "plan_sha256": plan_sha256,
            "repository_release_tag": plan.repository_release_tag,
            "repository_commit": plan.repository_commit,
            "repository_snapshot_mode": repository_snapshot_mode,
            "repository_root_name": repository_root_name,
            "repository_archive_relative_path": (
                repository_archive_relative_path
            ),
            "runtime_archive_sha256": plan.runtime.pksim_portable.sha256,
            "file_count": len(files),
            "files": files,
        }
        write_json_atomic(
            temporary / "bundle_manifest.json",
            manifest,
            canonical=True,
        )
        temporary.replace(output_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    archive_path = output_directory.with_suffix(".zip")
    if archive_path.exists():
        raise WorkerBundleError(f"Bundle archive already exists: {archive_path}")
    archive_sha256 = _zip_bundle(output_directory, archive_path)
    archive_sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    payload = (
        f"{archive_sha256}  {archive_path.name}\n"
    ).encode("ascii")
    with archive_sidecar.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "bundle_directory": str(output_directory),
        "bundle_archive": str(archive_path),
        "bundle_archive_sha256": archive_sha256,
        "bundle_archive_bytes": archive_path.stat().st_size,
        "job_id": plan.job_id,
        "file_count": len(files),
    }
