from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from pbpk_kb.model_assets.execution_plan import (
    ExecutionPlanError,
    load_execution_plan,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    OSPExecutionArtifact,
    OSPExecutionPlan,
    OSPExecutionProjectSpec,
    OSPExecutionResult,
    OSPProjectExecutionResult,
    OSPSimulationExecutionResult,
    OSPWorkerRuntimeIdentity,
)


class OSPWorkerError(RuntimeError):
    pass


def _safe_archive_member(member: str) -> None:
    path = PurePosixPath(member.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise OSPWorkerError(f"Unsafe path in PK-Sim runtime archive: {member!r}")


def _windows_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.removeprefix("\\\\")
    return "\\\\?\\" + resolved


def _runtime_manifest_path(runtime_directory: Path) -> Path:
    return runtime_directory / "runtime_manifest.json"


def prepare_portable_runtime(
    *,
    archive_path: Path,
    runtime_root: Path,
    plan: OSPExecutionPlan,
) -> tuple[Path, dict[str, Any]]:
    archive_path = archive_path.resolve()
    actual_archive_sha256 = sha256_file(archive_path)
    if actual_archive_sha256 != plan.runtime.pksim_portable.sha256:
        raise OSPWorkerError(
            "PK-Sim portable archive checksum does not match the execution plan."
        )
    runtime_directory = (
        runtime_root.resolve()
        / f"pksim-{plan.runtime.pksim_version}-{actual_archive_sha256[:12]}"
    )
    manifest_path = _runtime_manifest_path(runtime_directory)
    if runtime_directory.exists():
        if not manifest_path.is_file():
            raise OSPWorkerError(
                f"Existing runtime has no manifest: {runtime_directory}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("archive_sha256") != actual_archive_sha256:
            raise OSPWorkerError("Existing runtime archive checksum does not match.")
        cli_path = runtime_directory / manifest["cli_relative_path"]
        if not cli_path.is_file():
            raise OSPWorkerError("Existing runtime CLI executable is missing.")
        if sha256_file(cli_path) != manifest.get("cli_sha256"):
            raise OSPWorkerError("Existing runtime CLI executable checksum drifted.")
        return cli_path, manifest

    temporary = runtime_directory.with_name(runtime_directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                _safe_archive_member(info.filename)
            archive.extractall(temporary)
        expected_name = PurePosixPath(
            plan.runtime.pksim_cli_relative_path.replace("\\", "/")
        ).name.casefold()
        candidates = [
            path
            for path in temporary.rglob("*")
            if path.is_file() and path.name.casefold() == expected_name
        ]
        if len(candidates) != 1:
            raise OSPWorkerError(
                f"Expected one {expected_name} in runtime archive; found {len(candidates)}."
            )
        cli_path = candidates[0]
        relative_cli = cli_path.relative_to(temporary).as_posix()
        manifest = {
            "runtime_manifest_version": "1.0.0",
            "pksim_version": plan.runtime.pksim_version,
            "archive_name": archive_path.name,
            "archive_sha256": actual_archive_sha256,
            "cli_relative_path": relative_cli,
            "cli_sha256": sha256_file(cli_path),
            "prepared_at": datetime.now(UTC).isoformat(),
        }
        write_json_atomic(
            _runtime_manifest_path(temporary),
            manifest,
            canonical=True,
        )
        temporary.replace(runtime_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return runtime_directory / relative_cli, manifest


def build_export_command(
    *,
    cli_path: Path,
    project_path: Path,
    output_directory: Path,
    plan: OSPExecutionPlan,
    simulation_names: list[str] | None = None,
) -> list[str]:
    command = [
        str(cli_path),
        "export",
        "-p",
        str(project_path),
        "-o",
        str(output_directory),
    ]
    if simulation_names:
        command.extend(["-s", *simulation_names])
    command.extend(plan.export_flags)
    return command


def _write_log(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        if value and not value.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _artifact(
    path: Path,
    *,
    result_root: Path,
    source_relative_path: str | None = None,
    simulation_id: str | None = None,
    simulation_name: str | None = None,
) -> OSPExecutionArtifact:
    resolved = path.resolve()
    try:
        relative_path = resolved.relative_to(result_root.resolve()).as_posix()
    except ValueError as exc:
        raise OSPWorkerError(f"Artifact is outside result root: {resolved}") from exc
    return OSPExecutionArtifact(
        relative_path=relative_path,
        byte_size=resolved.stat().st_size,
        sha256=sha256_file(resolved),
        extension=resolved.suffix.casefold(),
        source_relative_path=source_relative_path,
        simulation_id=simulation_id,
        simulation_name=simulation_name,
    )


def _run_probe(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _runtime_identity(
    *,
    cli_path: Path,
    runtime_manifest: dict[str, Any],
) -> OSPWorkerRuntimeIdentity:
    version = _run_probe([str(cli_path), "--version"])
    help_result = _run_probe([str(cli_path), "export", "--help"])
    if version.returncode not in {0, 1}:
        raise OSPWorkerError(
            "PKSim.CLI --version returned an unsupported exit code."
        )
    if help_result.returncode not in {0, 1}:
        raise OSPWorkerError(
            "PKSim.CLI export --help returned an unsupported exit code."
        )
    version_output = (version.stdout + "\n" + version.stderr).strip()
    expected_version = f"PKSim.CLI {runtime_manifest['pksim_version']}"
    if not version_output.startswith(expected_version):
        raise OSPWorkerError(
            "PKSim.CLI version output does not match the locked runtime."
        )
    help_text = (help_result.stdout + "\n" + help_result.stderr).strip()
    required_tokens = {
        "--project",
        "--output",
        "--simulations",
        "--csv",
        "--xml",
        "--pkml",
        "--run",
    }
    missing = sorted(token for token in required_tokens if token not in help_text)
    if missing:
        raise OSPWorkerError(
            "PKSim.CLI help is missing required export options: " + ", ".join(missing)
        )
    return OSPWorkerRuntimeIdentity(
        platform=platform.system(),
        platform_release=platform.release(),
        python_version=platform.python_version(),
        hostname=socket.gethostname(),
        runtime_archive_sha256=runtime_manifest["archive_sha256"],
        cli_executable_sha256=sha256_file(cli_path),
        cli_relative_path=runtime_manifest["cli_relative_path"],
        cli_version_output=version_output,
        cli_version_return_code=version.returncode,
        cli_help_output=help_text,
        cli_help_return_code=help_result.returncode,
        cli_help_sha256=hashlib.sha256(help_text.encode("utf-8")).hexdigest(),
    )


def _project_attempt_number(project_root: Path) -> int:
    existing = []
    for path in project_root.glob("attempt_*"):
        try:
            existing.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            continue
    return max(existing, default=0) + 1


def _scratch_project_key(project: OSPExecutionProjectSpec) -> str:
    return project.project_key[:12]


def _validate_scratch_path_budget(
    *,
    plan: OSPExecutionPlan,
    scratch_root: Path,
    maximum_path_length: int = 240,
) -> int:
    maximum = 0
    for project in plan.projects:
        attempt_root = (
            scratch_root
            / _scratch_project_key(project)
            / "a999999"
            / "exports"
        )
        for simulation in project.simulations:
            candidates = [
                attempt_root
                / simulation.name
                / f"{simulation.name}-Results.csv",
                attempt_root / simulation.name / f"{simulation.name}.pkml",
                attempt_root / simulation.name / f"{simulation.name}.xml",
            ]
            maximum = max(maximum, *(len(str(path)) for path in candidates))
    if maximum > maximum_path_length:
        raise OSPWorkerError(
            "PK-Sim scratch path budget exceeds "
            f"{maximum_path_length} characters (worst case: {maximum})."
        )
    return maximum


_INVALID_WINDOWS_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')


def _pksim_output_name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    sanitized = "".join(
        "_"
        if character in _INVALID_WINDOWS_FILENAME_CHARACTERS
        or ord(character) < 32
        else character
        for character in normalized
    )
    return sanitized.rstrip(" .").casefold()


def _canonical_export_filename(
    *,
    extension: str,
    extension_index: int,
    extension_count: int,
    artifact_index: int,
) -> str:
    if extension == ".pkml":
        stem = "simulation"
    elif extension == ".xml":
        stem = "simulation"
    elif extension == ".csv":
        stem = "results"
    else:
        stem = f"artifact_{artifact_index:03d}"
    if extension_count > 1 and extension in {".pkml", ".xml", ".csv"}:
        stem += f"_{extension_index:03d}"
    return stem + extension


def _ingest_scratch_exports(
    *,
    project: OSPExecutionProjectSpec,
    scratch_export_root: Path,
    export_root: Path,
    result_root: Path,
) -> tuple[list[OSPExecutionArtifact], int, list[str]]:
    if export_root.exists():
        raise OSPWorkerError(
            f"Canonical export directory already exists: {export_root}"
        )
    export_root.mkdir(parents=True)
    expected_by_key: dict[str, tuple[int, Any]] = {}
    for index, simulation in enumerate(project.simulations, start=1):
        key = _pksim_output_name_key(simulation.name)
        if key in expected_by_key:
            raise OSPWorkerError(
                f"Duplicate normalized simulation name in {project.relative_path}."
            )
        expected_by_key[key] = (index, simulation)

    output_directories = sorted(
        (path for path in scratch_export_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    reason_codes: list[str] = []
    mapped_simulation_ids: set[str] = set()
    artifacts: list[OSPExecutionArtifact] = []
    unmatched_index = 0
    for output_directory in output_directories:
        matched = expected_by_key.get(_pksim_output_name_key(output_directory.name))
        if matched is None:
            unmatched_index += 1
            canonical_directory = export_root / f"u{unmatched_index:04d}"
            simulation_id = None
            simulation_name = None
            reason_codes.append("UNMAPPED_SIMULATION_OUTPUT")
        else:
            simulation_index, simulation = matched
            canonical_directory = export_root / f"s{simulation_index:04d}"
            simulation_id = simulation.simulation_id
            simulation_name = simulation.name
            mapped_simulation_ids.add(simulation_id)
        canonical_directory.mkdir()
        source_files = sorted(
            (path for path in output_directory.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(output_directory).as_posix().casefold(),
        )
        extension_counts: dict[str, int] = {}
        for source in source_files:
            extension = source.suffix.casefold()
            extension_counts[extension] = extension_counts.get(extension, 0) + 1
        extension_indexes: dict[str, int] = {}
        for artifact_index, source in enumerate(source_files, start=1):
            extension = source.suffix.casefold()
            extension_indexes[extension] = extension_indexes.get(extension, 0) + 1
            destination = canonical_directory / _canonical_export_filename(
                extension=extension,
                extension_index=extension_indexes[extension],
                extension_count=extension_counts[extension],
                artifact_index=artifact_index,
            )
            os.replace(
                _windows_long_path(source),
                _windows_long_path(destination),
            )
            artifacts.append(
                _artifact(
                    destination,
                    result_root=result_root,
                    source_relative_path=source.relative_to(
                        scratch_export_root
                    ).as_posix(),
                    simulation_id=simulation_id,
                    simulation_name=simulation_name,
                )
            )

    loose_files = sorted(
        (path for path in scratch_export_root.iterdir() if path.is_file()),
        key=lambda path: path.name.casefold(),
    )
    if loose_files:
        reason_codes.append("UNMAPPED_ROOT_EXPORT_ARTIFACT")
        root_directory = export_root / "u_root"
        root_directory.mkdir()
        for index, source in enumerate(loose_files, start=1):
            destination = root_directory / (
                f"artifact_{index:03d}{source.suffix.casefold()}"
            )
            os.replace(
                _windows_long_path(source),
                _windows_long_path(destination),
            )
            artifacts.append(
                _artifact(
                    destination,
                    result_root=result_root,
                    source_relative_path=source.relative_to(
                        scratch_export_root
                    ).as_posix(),
                )
            )

    expected_ids = {
        simulation.simulation_id for simulation in project.simulations
    }
    if mapped_simulation_ids != expected_ids:
        reason_codes.append("SIMULATION_OUTPUT_MAPPING_MISMATCH")
    return artifacts, len(output_directories), sorted(set(reason_codes))


def _per_simulation_export_reason_codes(
    *,
    project: OSPExecutionProjectSpec,
    artifacts: list[OSPExecutionArtifact],
) -> list[str]:
    extensions_by_simulation: dict[str, list[str]] = {}
    for artifact in artifacts:
        if artifact.simulation_id is None:
            continue
        extensions_by_simulation.setdefault(artifact.simulation_id, []).append(
            artifact.extension
        )
    reason_codes: list[str] = []
    for simulation in project.simulations:
        extensions = extensions_by_simulation.get(simulation.simulation_id, [])
        if ".pkml" not in extensions:
            reason_codes.append("SIMULATION_PKML_COVERAGE_MISMATCH")
        if ".xml" not in extensions:
            reason_codes.append("SIMULATION_XML_COVERAGE_MISMATCH")
        if ".csv" not in extensions:
            reason_codes.append("SIMULATION_CSV_COVERAGE_MISMATCH")
    return sorted(set(reason_codes))


def _valid_completed_attempt(
    *,
    project: OSPExecutionProjectSpec,
    result_root: Path,
) -> OSPProjectExecutionResult | None:
    project_root = result_root / "projects" / project.project_key
    checkpoints = sorted(
        project_root.glob("attempt_*/project_result.json"),
        reverse=True,
    )
    for checkpoint in checkpoints:
        try:
            result = OSPProjectExecutionResult.model_validate_json(
                checkpoint.read_text(encoding="utf-8")
            )
        except Exception:
            continue
        if (
            result.status != "passed"
            or result.execution_mode != "per_simulation"
            or result.relative_path != project.relative_path
            or result.input_sha256 != project.project_sha256
            or result.expected_simulation_count != len(project.simulations)
        ):
            continue
        expected_ids = {
            simulation.simulation_id for simulation in project.simulations
        }
        artifact_ids = {
            artifact.simulation_id
            for artifact in result.artifacts
            if artifact.simulation_id is not None
        }
        if (
            artifact_ids != expected_ids
            or any(
                artifact.source_relative_path is None
                or artifact.simulation_id is None
                or artifact.simulation_name is None
                for artifact in result.artifacts
            )
        ):
            continue
        if _per_simulation_export_reason_codes(
            project=project,
            artifacts=result.artifacts,
        ):
            continue
        simulation_results_by_id = {
            item.simulation_id: item for item in result.simulation_results
        }
        if (
            len(simulation_results_by_id) != len(result.simulation_results)
            or set(simulation_results_by_id) != expected_ids
            or any(
                item.status != "passed"
                or item.return_code != 0
                or item.simulation_name != simulation.name
                or item.reason_codes
                for simulation in project.simulations
                for item in [simulation_results_by_id[simulation.simulation_id]]
            )
        ):
            continue
        artifacts = [
            *result.artifacts,
            result.stdout_artifact,
            result.stderr_artifact,
            *(
                artifact
                for simulation_result in result.simulation_results
                for artifact in (
                    simulation_result.stdout_artifact,
                    simulation_result.stderr_artifact,
                )
            ),
        ]
        valid = True
        for artifact in artifacts:
            relative = PurePosixPath(artifact.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                valid = False
                break
            path = result_root.joinpath(*relative.parts)
            if (
                not path.is_file()
                or path.stat().st_size != artifact.byte_size
                or sha256_file(path) != artifact.sha256
            ):
                valid = False
                break
        if valid:
            return result
    return None


def _execute_project(
    *,
    project: OSPExecutionProjectSpec,
    plan: OSPExecutionPlan,
    repository_root: Path,
    cli_path: Path,
    result_root: Path,
    scratch_root: Path,
) -> OSPProjectExecutionResult:
    project_path = (repository_root / project.relative_path).resolve()
    try:
        project_path.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise OSPWorkerError(
            f"Project path escapes repository root: {project.relative_path}"
        ) from exc
    if sha256_file(project_path) != project.project_sha256:
        raise OSPWorkerError(
            f"Project checksum does not match plan: {project.relative_path}"
        )
    project_root = result_root / "projects" / project.project_key
    attempt_number = _project_attempt_number(project_root)
    attempt_root = project_root / f"attempt_{attempt_number:03d}"
    export_root = attempt_root / "exports"
    attempt_root.mkdir(parents=True, exist_ok=False)
    scratch_project_root = scratch_root / _scratch_project_key(project)
    if scratch_project_root.exists():
        shutil.rmtree(_windows_long_path(scratch_project_root))
    scratch_export_root = (
        scratch_project_root / f"a{attempt_number:03d}" / "exports"
    )
    scratch_export_root.mkdir(parents=True, exist_ok=False)
    stdout_path = attempt_root / "stdout.log"
    stderr_path = attempt_root / "stderr.log"
    started = datetime.now(UTC)
    started_clock = time.monotonic()
    simulation_attempts: list[dict[str, Any]] = []
    combined_stdout: list[str] = []
    combined_stderr: list[str] = []
    for sequence, simulation in enumerate(project.simulations, start=1):
        command = build_export_command(
            cli_path=cli_path,
            project_path=project_path,
            output_directory=scratch_export_root,
            plan=plan,
            simulation_names=[simulation.name],
        )
        simulation_log_root = (
            attempt_root / "simulation_runs" / f"s{sequence:04d}"
        )
        simulation_stdout_path = simulation_log_root / "stdout.log"
        simulation_stderr_path = simulation_log_root / "stderr.log"
        simulation_started = datetime.now(UTC)
        simulation_started_clock = time.monotonic()
        simulation_return_code: int | None = None
        simulation_reasons: list[str] = []
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=plan.command_timeout_seconds,
            )
            simulation_return_code = completed.returncode
            simulation_stdout = completed.stdout
            simulation_stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            simulation_stdout = exc.stdout or ""
            simulation_stderr = exc.stderr or ""
            if isinstance(simulation_stdout, bytes):
                simulation_stdout = simulation_stdout.decode("utf-8", "replace")
            if isinstance(simulation_stderr, bytes):
                simulation_stderr = simulation_stderr.decode("utf-8", "replace")
            simulation_reasons.append("CLI_TIMEOUT")
        if simulation_return_code != 0:
            simulation_reasons.append("CLI_NONZERO_EXIT")
        _write_log(simulation_stdout_path, simulation_stdout)
        _write_log(simulation_stderr_path, simulation_stderr)
        combined_stdout.extend(
            [
                f"=== s{sequence:04d} {simulation.name} ===",
                f"COMMAND={json.dumps(command, ensure_ascii=False)}",
                f"RETURN_CODE={simulation_return_code}",
                simulation_stdout,
            ]
        )
        combined_stderr.extend(
            [
                f"=== s{sequence:04d} {simulation.name} ===",
                simulation_stderr,
            ]
        )
        simulation_attempts.append(
            {
                "sequence": sequence,
                "simulation": simulation,
                "started_at": simulation_started,
                "completed_at": datetime.now(UTC),
                "duration_seconds": time.monotonic() - simulation_started_clock,
                "command": command,
                "return_code": simulation_return_code,
                "stdout_path": simulation_stdout_path,
                "stderr_path": simulation_stderr_path,
                "reason_codes": simulation_reasons,
            }
        )
    _write_log(stdout_path, "\n".join(combined_stdout))
    _write_log(stderr_path, "\n".join(combined_stderr))
    artifacts, output_directory_count, mapping_reasons = _ingest_scratch_exports(
        project=project,
        scratch_export_root=scratch_export_root,
        export_root=export_root,
        result_root=result_root,
    )
    reason_codes = list(mapping_reasons)
    reason_codes.extend(
        _per_simulation_export_reason_codes(
            project=project,
            artifacts=artifacts,
        )
    )
    shutil.rmtree(
        _windows_long_path(scratch_project_root),
        ignore_errors=True,
    )
    completed_at = datetime.now(UTC)
    export_counts: dict[str, int] = {}
    for artifact in artifacts:
        export_counts[artifact.extension] = export_counts.get(artifact.extension, 0) + 1
    simulation_results: list[OSPSimulationExecutionResult] = []
    for attempt in simulation_attempts:
        simulation = attempt["simulation"]
        simulation_artifacts = [
            artifact
            for artifact in artifacts
            if artifact.simulation_id == simulation.simulation_id
        ]
        simulation_export_counts: dict[str, int] = {}
        for artifact in simulation_artifacts:
            simulation_export_counts[artifact.extension] = (
                simulation_export_counts.get(artifact.extension, 0) + 1
            )
        simulation_reasons = list(attempt["reason_codes"])
        if simulation_export_counts.get(".pkml", 0) != 1:
            simulation_reasons.append("PKML_COVERAGE_MISMATCH")
        if simulation_export_counts.get(".xml", 0) != 1:
            simulation_reasons.append("XML_COVERAGE_MISMATCH")
        if simulation_export_counts.get(".csv", 0) < 1:
            simulation_reasons.append("CSV_RESULT_COVERAGE_MISMATCH")
        if any(artifact.byte_size == 0 for artifact in simulation_artifacts):
            simulation_reasons.append("EMPTY_EXPORT_ARTIFACT")
        simulation_results.append(
            OSPSimulationExecutionResult(
                sequence=attempt["sequence"],
                simulation_id=simulation.simulation_id,
                simulation_name=simulation.name,
                status="passed" if not simulation_reasons else "failed",
                started_at=attempt["started_at"],
                completed_at=attempt["completed_at"],
                duration_seconds=attempt["duration_seconds"],
                command=attempt["command"],
                return_code=attempt["return_code"],
                export_counts=dict(sorted(simulation_export_counts.items())),
                stdout_artifact=_artifact(
                    attempt["stdout_path"],
                    result_root=result_root,
                    source_relative_path=(
                        f"simulation_runs/s{attempt['sequence']:04d}/stdout.log"
                    ),
                    simulation_id=simulation.simulation_id,
                    simulation_name=simulation.name,
                ),
                stderr_artifact=_artifact(
                    attempt["stderr_path"],
                    result_root=result_root,
                    source_relative_path=(
                        f"simulation_runs/s{attempt['sequence']:04d}/stderr.log"
                    ),
                    simulation_id=simulation.simulation_id,
                    simulation_name=simulation.name,
                ),
                reason_codes=sorted(set(simulation_reasons)),
            )
        )
    expected_count = len(project.simulations)
    failed_return_codes = [
        result.return_code
        for result in simulation_results
        if result.return_code != 0
    ]
    return_code = failed_return_codes[0] if failed_return_codes else 0
    if any(result.status != "passed" for result in simulation_results):
        reason_codes.append("SIMULATION_EXECUTION_FAILURES")
    if output_directory_count != expected_count:
        reason_codes.append("SIMULATION_DIRECTORY_COVERAGE_MISMATCH")
    if export_counts.get(".pkml", 0) != expected_count:
        reason_codes.append("PKML_COVERAGE_MISMATCH")
    if export_counts.get(".xml", 0) != expected_count:
        reason_codes.append("XML_COVERAGE_MISMATCH")
    if export_counts.get(".csv", 0) < expected_count:
        reason_codes.append("CSV_RESULT_COVERAGE_MISMATCH")
    if any(artifact.byte_size == 0 for artifact in artifacts):
        reason_codes.append("EMPTY_EXPORT_ARTIFACT")
    result = OSPProjectExecutionResult(
        project_key=project.project_key,
        relative_path=project.relative_path,
        input_sha256=project.project_sha256,
        attempt=attempt_number,
        status="passed" if not reason_codes else "failed",
        execution_mode="per_simulation",
        started_at=started,
        completed_at=completed_at,
        duration_seconds=time.monotonic() - started_clock,
        command=[],
        return_code=return_code,
        expected_simulation_count=expected_count,
        output_directory_count=output_directory_count,
        export_counts=dict(sorted(export_counts.items())),
        artifacts=artifacts,
        stdout_artifact=_artifact(stdout_path, result_root=result_root),
        stderr_artifact=_artifact(stderr_path, result_root=result_root),
        simulation_results=simulation_results,
        reason_codes=sorted(set(reason_codes)),
    )
    checkpoint_path = attempt_root / "project_result.json"
    write_json_atomic(checkpoint_path, result, canonical=True)
    return result


def execute_plan(
    *,
    plan_path: Path,
    repository_root: Path,
    runtime_archive: Path,
    runtime_root: Path,
    output_root: Path,
    scratch_root: Path,
) -> tuple[Path, OSPExecutionResult]:
    if platform.system() != "Windows":
        raise OSPWorkerError(
            "OSP execution worker must run on Windows with the locked PK-Sim runtime."
        )
    plan, plan_sha256 = load_execution_plan(plan_path)
    repository_root = repository_root.resolve()
    if not repository_root.is_dir():
        raise OSPWorkerError(f"Repository root does not exist: {repository_root}")
    cli_path, runtime_manifest = prepare_portable_runtime(
        archive_path=runtime_archive,
        runtime_root=runtime_root,
        plan=plan,
    )
    worker = _runtime_identity(cli_path=cli_path, runtime_manifest=runtime_manifest)
    result_root = output_root.resolve() / plan.job_id
    scratch_root = scratch_root.resolve()
    if result_root.drive.casefold() != scratch_root.drive.casefold():
        raise OSPWorkerError(
            "PK-Sim scratch and canonical output must be on the same drive."
        )
    scratch_keys = [_scratch_project_key(project) for project in plan.projects]
    if len(scratch_keys) != len(set(scratch_keys)):
        raise OSPWorkerError("Truncated scratch project keys are not unique.")
    _validate_scratch_path_budget(plan=plan, scratch_root=scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    if (result_root / "execution_result.json").exists():
        raise OSPWorkerError("Immutable execution result already exists for this job.")
    result_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    results: list[OSPProjectExecutionResult] = []
    for project in plan.projects:
        completed_attempt = _valid_completed_attempt(
            project=project,
            result_root=result_root,
        )
        if completed_attempt is not None:
            results.append(completed_attempt)
        else:
            results.append(
                _execute_project(
                    project=project,
                    plan=plan,
                    repository_root=repository_root,
                    cli_path=cli_path,
                    result_root=result_root,
                    scratch_root=scratch_root,
                )
            )
    reason_codes: list[str] = []
    if len(results) != len(plan.projects):
        reason_codes.append("PROJECT_COVERAGE_MISMATCH")
    if any(result.status != "passed" for result in results):
        reason_codes.append("PROJECT_EXECUTION_FAILURES")
    execution_result = OSPExecutionResult(
        job_id=plan.job_id,
        plan_sha256=plan_sha256,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        worker=worker,
        status="passed" if not reason_codes else "failed",
        projects=results,
        reason_codes=reason_codes,
    )
    result_path = result_root / "execution_result.json"
    result_sha256 = write_json_atomic(result_path, execution_result, canonical=True)
    write_json_atomic(
        result_root / "execution_result.sha256",
        {"file": result_path.name, "sha256": result_sha256},
        canonical=True,
    )
    return result_path, execution_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run locked OSP PK-Sim projects on a Windows worker."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result_path, result = execute_plan(
            plan_path=args.plan,
            repository_root=args.repository,
            runtime_archive=args.runtime_archive,
            runtime_root=args.runtime_root,
            output_root=args.output_root,
            scratch_root=args.scratch_root,
        )
    except (OSPWorkerError, ExecutionPlanError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "result_path": str(result_path),
                "job_id": result.job_id,
                "status": result.status,
                "project_count": len(result.projects),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result.status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
