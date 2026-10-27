from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pbpk_kb.model_assets.execution_plan import (
    ExecutionPlanError,
    load_execution_plan,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    OSPExecutionArtifact,
    OSPExecutionResult,
    OSPExecutionValidationReport,
)


class ExecutionValidationError(RuntimeError):
    pass


def _safe_artifact_path(result_root: Path, artifact: OSPExecutionArtifact) -> Path:
    relative = PurePosixPath(artifact.relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ExecutionValidationError(
            f"Unsafe result artifact path: {artifact.relative_path!r}"
        )
    path = (result_root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(result_root.resolve())
    except ValueError as exc:
        raise ExecutionValidationError(
            f"Result artifact escapes result root: {artifact.relative_path}"
        ) from exc
    return path


def _verify_artifact(
    result_root: Path,
    artifact: OSPExecutionArtifact,
    reason_codes: list[str],
) -> bool:
    try:
        path = _safe_artifact_path(result_root, artifact)
    except ExecutionValidationError:
        reason_codes.append("UNSAFE_ARTIFACT_PATH")
        return False
    if not path.is_file():
        reason_codes.append("ARTIFACT_MISSING")
        return False
    if path.stat().st_size != artifact.byte_size:
        reason_codes.append("ARTIFACT_SIZE_MISMATCH")
        return False
    if sha256_file(path) != artifact.sha256:
        reason_codes.append("ARTIFACT_CHECKSUM_MISMATCH")
        return False
    return True


def _verify_result_sidecar(result_path: Path, result_sha256: str) -> None:
    sidecar_path = result_path.with_name("execution_result.sha256")
    if not sidecar_path.is_file():
        raise ExecutionValidationError("Execution result checksum sidecar is missing.")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("file") != result_path.name
        or sidecar.get("sha256") != result_sha256
    ):
        raise ExecutionValidationError("Execution result checksum sidecar does not match.")


def validate_execution_result(
    *,
    plan_path: Path,
    result_path: Path,
) -> OSPExecutionValidationReport:
    plan, plan_sha256 = load_execution_plan(plan_path)
    result_path = result_path.resolve()
    result_root = result_path.parent
    result_sha256 = sha256_file(result_path)
    _verify_result_sidecar(result_path, result_sha256)
    result = OSPExecutionResult.model_validate_json(
        result_path.read_text(encoding="utf-8")
    )
    reason_codes: list[str] = []
    if result.job_id != plan.job_id:
        reason_codes.append("JOB_ID_MISMATCH")
    if result.plan_sha256 != plan_sha256:
        reason_codes.append("PLAN_CHECKSUM_MISMATCH")
    if result.worker.platform != "Windows":
        reason_codes.append("WORKER_PLATFORM_MISMATCH")
    if (
        result.worker.runtime_archive_sha256
        != plan.runtime.pksim_portable.sha256
    ):
        reason_codes.append("RUNTIME_ARCHIVE_MISMATCH")
    expected_version = f"PKSim.CLI {plan.runtime.pksim_version}"
    if not result.worker.cli_version_output.startswith(expected_version):
        reason_codes.append("RUNTIME_VERSION_MISMATCH")
    if result.worker.cli_version_return_code not in {0, 1}:
        reason_codes.append("RUNTIME_VERSION_PROBE_EXIT_CODE")
    if result.worker.cli_help_return_code not in {0, 1}:
        reason_codes.append("RUNTIME_HELP_PROBE_EXIT_CODE")
    required_help_tokens = {
        "--project",
        "--output",
        "--simulations",
        "--csv",
        "--xml",
        "--pkml",
        "--run",
    }
    if any(
        token not in result.worker.cli_help_output
        for token in required_help_tokens
    ):
        reason_codes.append("RUNTIME_HELP_CONTRACT_MISMATCH")
    if (
        hashlib.sha256(
            result.worker.cli_help_output.encode("utf-8")
        ).hexdigest()
        != result.worker.cli_help_sha256
    ):
        reason_codes.append("RUNTIME_HELP_CHECKSUM_MISMATCH")
    if result.status != "passed":
        reason_codes.append("WORKER_RESULT_FAILED")

    expected_by_key = {project.project_key: project for project in plan.projects}
    result_by_key = {project.project_key: project for project in result.projects}
    if len(result_by_key) != len(result.projects):
        reason_codes.append("DUPLICATE_PROJECT_RESULT")
    if set(expected_by_key) != set(result_by_key):
        reason_codes.append("PROJECT_COVERAGE_MISMATCH")

    verified_artifacts = 0
    passed_projects = 0
    seen_artifact_paths: set[str] = set()
    for project_key, expected in expected_by_key.items():
        actual = result_by_key.get(project_key)
        if actual is None:
            continue
        project_reason_count = len(reason_codes)
        if actual.relative_path != expected.relative_path:
            reason_codes.append("PROJECT_PATH_MISMATCH")
        if actual.input_sha256 != expected.project_sha256:
            reason_codes.append("PROJECT_INPUT_CHECKSUM_MISMATCH")
        if actual.status != "passed" or actual.return_code != 0:
            reason_codes.append("PROJECT_EXECUTION_FAILED")
        if actual.execution_mode != "per_simulation":
            reason_codes.append("EXECUTION_MODE_MISMATCH")
        expected_count = len(expected.simulations)
        if actual.expected_simulation_count != expected_count:
            reason_codes.append("EXPECTED_SIMULATION_COUNT_MISMATCH")
        if actual.output_directory_count != expected_count:
            reason_codes.append("SIMULATION_DIRECTORY_COVERAGE_MISMATCH")
        if actual.export_counts.get(".pkml", 0) != expected_count:
            reason_codes.append("PKML_COVERAGE_MISMATCH")
        if actual.export_counts.get(".xml", 0) != expected_count:
            reason_codes.append("XML_COVERAGE_MISMATCH")
        if actual.export_counts.get(".csv", 0) < expected_count:
            reason_codes.append("CSV_RESULT_COVERAGE_MISMATCH")
        expected_command_tail = ["-k", "-x", "-c", "-r"]
        expected_simulations = {
            simulation.simulation_id: simulation
            for simulation in expected.simulations
        }
        actual_simulations = {
            simulation.simulation_id: simulation
            for simulation in actual.simulation_results
        }
        if (
            len(actual_simulations) != len(actual.simulation_results)
            or set(actual_simulations) != set(expected_simulations)
        ):
            reason_codes.append("SIMULATION_EXECUTION_COVERAGE_MISMATCH")
        for simulation_id, expected_simulation in expected_simulations.items():
            simulation_result = actual_simulations.get(simulation_id)
            if simulation_result is None:
                continue
            if (
                simulation_result.simulation_name != expected_simulation.name
                or simulation_result.status != "passed"
                or simulation_result.return_code != 0
                or simulation_result.reason_codes
            ):
                reason_codes.append("SIMULATION_EXECUTION_FAILED")
            normalized_command = [
                value.replace("\\", "/")
                for value in simulation_result.command
            ]
            if len(normalized_command) < 10 or normalized_command[1] != "export":
                reason_codes.append("CLI_COMMAND_MISMATCH")
                continue
            try:
                project_arg = normalized_command[
                    normalized_command.index("-p") + 1
                ]
                output_arg = normalized_command[
                    normalized_command.index("-o") + 1
                ]
                simulation_index = normalized_command.index("-s")
            except (ValueError, IndexError):
                reason_codes.append("CLI_COMMAND_MISMATCH")
                continue
            if not project_arg.endswith(expected.relative_path):
                reason_codes.append("CLI_PROJECT_ARGUMENT_MISMATCH")
            if not output_arg:
                reason_codes.append("CLI_OUTPUT_ARGUMENT_MISSING")
            if normalized_command[simulation_index + 1 : -4] != [
                expected_simulation.name
            ]:
                reason_codes.append("CLI_SIMULATION_ARGUMENT_MISMATCH")
            if normalized_command[-4:] != expected_command_tail:
                reason_codes.append("CLI_EXPORT_FLAGS_MISMATCH")
            if simulation_result.export_counts.get(".pkml", 0) != 1:
                reason_codes.append("SIMULATION_PKML_COVERAGE_MISMATCH")
            if simulation_result.export_counts.get(".xml", 0) != 1:
                reason_codes.append("SIMULATION_XML_COVERAGE_MISMATCH")
            if simulation_result.export_counts.get(".csv", 0) < 1:
                reason_codes.append("SIMULATION_CSV_COVERAGE_MISMATCH")

        all_artifacts = [
            *actual.artifacts,
            actual.stdout_artifact,
            actual.stderr_artifact,
            *(
                artifact
                for simulation_result in actual.simulation_results
                for artifact in (
                    simulation_result.stdout_artifact,
                    simulation_result.stderr_artifact,
                )
            ),
        ]
        for artifact in all_artifacts:
            if artifact.relative_path in seen_artifact_paths:
                reason_codes.append("DUPLICATE_ARTIFACT_PATH")
                continue
            seen_artifact_paths.add(artifact.relative_path)
            if _verify_artifact(result_root, artifact, reason_codes):
                verified_artifacts += 1
        export_paths = {
            artifact.relative_path for artifact in actual.artifacts
        }
        expected_simulation_ids = {
            simulation.simulation_id for simulation in expected.simulations
        }
        artifact_simulation_ids = {
            artifact.simulation_id
            for artifact in actual.artifacts
            if artifact.simulation_id is not None
        }
        if (
            artifact_simulation_ids != expected_simulation_ids
            or any(
                artifact.source_relative_path is None
                or artifact.simulation_id is None
                or artifact.simulation_name is None
                for artifact in actual.artifacts
            )
        ):
            reason_codes.append("SIMULATION_ARTIFACT_MAPPING_MISMATCH")
        extensions_by_simulation: dict[str, list[str]] = {}
        for artifact in actual.artifacts:
            if artifact.simulation_id is None:
                continue
            extensions_by_simulation.setdefault(
                artifact.simulation_id,
                [],
            ).append(artifact.extension)
        for simulation in expected.simulations:
            extensions = extensions_by_simulation.get(
                simulation.simulation_id,
                [],
            )
            if ".pkml" not in extensions:
                reason_codes.append("SIMULATION_PKML_COVERAGE_MISMATCH")
            if ".xml" not in extensions:
                reason_codes.append("SIMULATION_XML_COVERAGE_MISMATCH")
            if ".csv" not in extensions:
                reason_codes.append("SIMULATION_CSV_COVERAGE_MISMATCH")
        attempt_root = (
            result_root
            / "projects"
            / actual.project_key
            / f"attempt_{actual.attempt:03d}"
        )
        actual_export_paths = {
            path.resolve().relative_to(result_root).as_posix()
            for path in (attempt_root / "exports").rglob("*")
            if path.is_file()
        }
        if export_paths != actual_export_paths:
            reason_codes.append("UNLISTED_OR_MISSING_EXPORT_ARTIFACT")
        if len(reason_codes) == project_reason_count:
            passed_projects += 1

    unique_reasons = sorted(set(reason_codes))
    return OSPExecutionValidationReport(
        validated_at=datetime.now(UTC),
        job_id=plan.job_id,
        plan_sha256=plan_sha256,
        result_sha256=result_sha256,
        project_count=len(plan.projects),
        expected_simulation_count=sum(
            len(project.simulations) for project in plan.projects
        ),
        passed_project_count=passed_projects,
        verified_artifact_count=verified_artifacts,
        execution_probe_complete=not unique_reasons,
        reason_codes=unique_reasons,
    )


def write_validation_report(
    output_path: Path,
    report: OSPExecutionValidationReport,
) -> str:
    if output_path.exists():
        raise ExecutionValidationError(
            f"Validation report is immutable and already exists: {output_path}"
        )
    return write_json_atomic(output_path, report, canonical=True)
