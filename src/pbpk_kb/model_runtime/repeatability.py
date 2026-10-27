from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbpk_kb.model_assets.execution_plan import (
    load_execution_plan,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    OSPExecutionResult,
    OSPExecutionValidationReport,
)
from pbpk_kb.model_runtime.jobs import (
    build_repeatability_control_plan,
)
from pbpk_kb.model_runtime.manifests import (
    RepeatabilityProtocol,
    RepeatabilityReport,
    RepeatabilityRunInput,
)
from pbpk_kb.model_runtime.result_parser import compare_csv_files


AUDITOR_VERSION = "pbpk-runtime-repeatability-auditor-1.0.0"


class RepeatabilityAuditError(RuntimeError):
    pass


def _load_validation(path: Path) -> OSPExecutionValidationReport:
    return OSPExecutionValidationReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _load_result(path: Path) -> OSPExecutionResult:
    return OSPExecutionResult.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _csv_artifacts(
    result: OSPExecutionResult,
) -> dict[tuple[str, str], list[Any]]:
    by_simulation: dict[tuple[str, str], list[Any]] = {}
    for project in result.projects:
        for artifact in project.artifacts:
            if artifact.extension != ".csv" or artifact.simulation_id is None:
                continue
            key = (project.project_key, artifact.simulation_id)
            by_simulation.setdefault(key, []).append(artifact)
    for artifacts in by_simulation.values():
        artifacts.sort(
            key=lambda item: (
                item.source_relative_path or "",
                item.relative_path,
            )
        )
    return by_simulation


def _artifact_inventory(
    result: OSPExecutionResult,
) -> set[tuple[str, str | None, str, str | None]]:
    return {
        (
            project.project_key,
            artifact.simulation_id,
            artifact.extension,
            artifact.source_relative_path,
        )
        for project in result.projects
        for artifact in project.artifacts
    }


def _verify_csv_artifact(result_root: Path, artifact: Any) -> Path:
    path = (result_root / artifact.relative_path).resolve()
    try:
        path.relative_to(result_root.resolve())
    except ValueError as exc:
        raise RepeatabilityAuditError(
            f"CSV artifact escapes result root: {artifact.relative_path}"
        ) from exc
    if not path.is_file():
        raise RepeatabilityAuditError(
            f"CSV artifact is missing: {artifact.relative_path}"
        )
    if (
        path.stat().st_size != artifact.byte_size
        or sha256_file(path) != artifact.sha256
    ):
        raise RepeatabilityAuditError(
            f"CSV artifact checksum mismatch: {artifact.relative_path}"
        )
    return path


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return digest.hexdigest()


def audit_repeatability(
    *,
    protocol_path: Path,
    control_plan_path: Path,
    run_a_result_path: Path,
    run_a_validation_path: Path,
    run_b_result_path: Path,
    run_b_validation_path: Path,
    output_directory: Path,
) -> RepeatabilityReport:
    protocol_sha256 = sha256_file(protocol_path)
    protocol = RepeatabilityProtocol.model_validate_json(
        protocol_path.read_text(encoding="utf-8")
    )
    control_plan, control_plan_sha256 = load_execution_plan(
        control_plan_path
    )
    validation_a = _load_validation(run_a_validation_path)
    validation_b = _load_validation(run_b_validation_path)
    result_a = _load_result(run_a_result_path)
    result_b = _load_result(run_b_result_path)
    result_a_sha256 = sha256_file(run_a_result_path)
    result_b_sha256 = sha256_file(run_b_result_path)
    validation_a_sha256 = sha256_file(run_a_validation_path)
    validation_b_sha256 = sha256_file(run_b_validation_path)
    result_root_a = run_a_result_path.resolve().parent
    result_root_b = run_b_result_path.resolve().parent

    checks = {
        "protocol_matches_control_plan": (
            protocol.control_plan_sha256 == control_plan_sha256
            and protocol.control_job_id == control_plan.job_id
        ),
        "output_roots_distinct": result_root_a != result_root_b,
        "run_a_validation_passed": (
            validation_a.execution_probe_complete
            and validation_a.result_sha256 == result_a_sha256
        ),
        "run_b_validation_passed": (
            validation_b.execution_probe_complete
            and validation_b.result_sha256 == result_b_sha256
        ),
        "same_plan_executed_twice": (
            result_a.plan_sha256 == control_plan_sha256
            and result_b.plan_sha256 == control_plan_sha256
            and validation_a.plan_sha256 == control_plan_sha256
            and validation_b.plan_sha256 == control_plan_sha256
        ),
        "job_identity_matches": (
            result_a.job_id == control_plan.job_id
            and result_b.job_id == control_plan.job_id
            and validation_a.job_id == control_plan.job_id
            and validation_b.job_id == control_plan.job_id
        ),
        "runtime_identity_matches": (
            result_a.worker.runtime_archive_sha256
            == result_b.worker.runtime_archive_sha256
            == control_plan.runtime.pksim_portable.sha256
            and result_a.worker.cli_executable_sha256
            == result_b.worker.cli_executable_sha256
            and result_a.worker.cli_version_output
            == result_b.worker.cli_version_output
            and result_a.worker.cli_help_sha256
            == result_b.worker.cli_help_sha256
        ),
        "artifact_semantic_inventory_matches": (
            _artifact_inventory(result_a)
            == _artifact_inventory(result_b)
        ),
        "csv_mapping_matches": True,
        "csv_structure_matches": True,
        "time_axis_matches": True,
        "non_numeric_cells_match": True,
        "numeric_values_within_tolerance": True,
    }
    csv_a = _csv_artifacts(result_a)
    csv_b = _csv_artifacts(result_b)
    if set(csv_a) != set(csv_b):
        checks["csv_mapping_matches"] = False

    mismatches: list[dict[str, Any]] = []
    compared_files = 0
    compared_numeric_cells = 0
    compared_non_numeric_cells = 0
    compared_time_cells = 0
    maximum_absolute_difference = 0.0
    maximum_relative_difference = 0.0
    worst_numeric_cell: dict[str, Any] | None = None
    for key in sorted(set(csv_a) & set(csv_b)):
        artifacts_a = csv_a[key]
        artifacts_b = csv_b[key]
        if len(artifacts_a) != len(artifacts_b):
            checks["csv_mapping_matches"] = False
            mismatches.append(
                {
                    "reason_code": "CSV_ARTIFACT_COUNT_MISMATCH",
                    "project_key": key[0],
                    "simulation_id": key[1],
                    "run_a_count": len(artifacts_a),
                    "run_b_count": len(artifacts_b),
                }
            )
            continue
        for ordinal, (artifact_a, artifact_b) in enumerate(
            zip(artifacts_a, artifacts_b, strict=True),
            start=1,
        ):
            compared_files += 1
            path_a = _verify_csv_artifact(result_root_a, artifact_a)
            path_b = _verify_csv_artifact(result_root_b, artifact_b)
            comparison = compare_csv_files(
                run_a_path=path_a,
                run_b_path=path_b,
                simulation_id=key[1],
                csv_ordinal=ordinal,
                absolute_tolerance=protocol.absolute_tolerance,
                relative_tolerance=protocol.relative_tolerance,
            )
            compared_numeric_cells += comparison.numeric_cell_count
            compared_non_numeric_cells += comparison.non_numeric_cell_count
            compared_time_cells += comparison.time_axis_cell_count
            mismatches.extend(
                {
                    "project_key": key[0],
                    "simulation_id": key[1],
                    "csv_ordinal": ordinal,
                    **item,
                }
                for item in comparison.mismatches
            )
            reason_codes = {
                item["reason_code"] for item in comparison.mismatches
            }
            if reason_codes & {
                "CSV_EMPTY",
                "CSV_HEADER_MISMATCH",
                "CSV_ROW_COUNT_MISMATCH",
                "CSV_COLUMN_COUNT_MISMATCH",
                "CELL_TYPE_MISMATCH",
            }:
                checks["csv_structure_matches"] = False
            if "TIME_AXIS_MISMATCH" in reason_codes:
                checks["time_axis_matches"] = False
            if "NON_NUMERIC_CELL_MISMATCH" in reason_codes:
                checks["non_numeric_cells_match"] = False
            if reason_codes & {
                "NUMERIC_TOLERANCE_EXCEEDED",
                "NON_FINITE_CELL_MISMATCH",
            }:
                checks["numeric_values_within_tolerance"] = False
            if (
                comparison.maximum_absolute_difference
                > maximum_absolute_difference
            ):
                maximum_absolute_difference = (
                    comparison.maximum_absolute_difference
                )
                worst_numeric_cell = comparison.worst_numeric_cell
            maximum_relative_difference = max(
                maximum_relative_difference,
                comparison.maximum_relative_difference,
            )

    if compared_files == 0:
        checks["csv_mapping_matches"] = False
        mismatches.append({"reason_code": "NO_CSV_RESULTS_COMPARED"})
    accepted = all(checks.values()) and not mismatches
    output_directory.mkdir(parents=True, exist_ok=False)
    mismatch_path = output_directory / "mismatches.jsonl"
    mismatch_sha256 = _write_jsonl_atomic(mismatch_path, mismatches)
    return RepeatabilityReport(
        auditor_version=AUDITOR_VERSION,
        audited_at=datetime.now(UTC),
        accepted=accepted,
        protocol_id=protocol.protocol_id,
        protocol_sha256=protocol_sha256,
        control_plan_sha256=control_plan_sha256,
        runtime_reproducibility_status=(
            "runtime_reproducibility_control_passed"
            if accepted
            else "runtime_reproducibility_control_failed"
        ),
        license_status=protocol.license_status,
        runs=[
            RepeatabilityRunInput(
                run_name="run_a",
                result_path=str(run_a_result_path.resolve()),
                result_sha256=result_a_sha256,
                validation_path=str(run_a_validation_path.resolve()),
                validation_sha256=validation_a_sha256,
                result_root=str(result_root_a),
            ),
            RepeatabilityRunInput(
                run_name="run_b",
                result_path=str(run_b_result_path.resolve()),
                result_sha256=result_b_sha256,
                validation_path=str(run_b_validation_path.resolve()),
                validation_sha256=validation_b_sha256,
                result_root=str(result_root_b),
            ),
        ],
        tolerances={
            "absolute": protocol.absolute_tolerance,
            "relative": protocol.relative_tolerance,
        },
        checks=checks,
        counts={
            "projects": len(control_plan.projects),
            "simulations": sum(
                len(project.simulations)
                for project in control_plan.projects
            ),
            "csv_files_compared": compared_files,
            "numeric_cells_compared": compared_numeric_cells,
            "non_numeric_cells_compared": compared_non_numeric_cells,
            "time_axis_cells_compared": compared_time_cells,
            "mismatches": len(mismatches),
        },
        maximum_absolute_difference=maximum_absolute_difference,
        maximum_relative_difference=maximum_relative_difference,
        worst_numeric_cell=worst_numeric_cell,
        mismatch_manifest_path=str(mismatch_path.resolve()),
        mismatch_manifest_sha256=mismatch_sha256,
        mismatch_examples=mismatches[:20],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and audit locked OSP repeatability controls."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-plan", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--absolute-tolerance",
        type=float,
        default=1e-12,
    )
    prepare.add_argument(
        "--relative-tolerance",
        type=float,
        default=1e-10,
    )

    audit = subparsers.add_parser("audit")
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--run-a-result", type=Path, required=True)
    audit.add_argument("--run-a-validation", type=Path, required=True)
    audit.add_argument("--run-b-result", type=Path, required=True)
    audit.add_argument("--run-b-validation", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        plan_path, protocol_path, protocol = (
            build_repeatability_control_plan(
                source_plan_path=args.source_plan,
                output_directory=args.output,
                absolute_tolerance=args.absolute_tolerance,
                relative_tolerance=args.relative_tolerance,
            )
        )
        print(
            json.dumps(
                {
                    "control_job_id": protocol.control_job_id,
                    "control_plan_path": str(plan_path),
                    "protocol_path": str(protocol_path),
                    "project": protocol.control_project_relative_path,
                    "simulation_count": protocol.simulation_count,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    report = audit_repeatability(
        protocol_path=args.protocol,
        control_plan_path=args.plan,
        run_a_result_path=args.run_a_result,
        run_a_validation_path=args.run_a_validation,
        run_b_result_path=args.run_b_result,
        run_b_validation_path=args.run_b_validation,
        output_directory=args.output,
    )
    report_path = args.output / "repeatability_audit.json"
    report_sha256 = write_json_atomic(
        report_path,
        report,
        canonical=True,
    )
    print(
        json.dumps(
            {
                **report.model_dump(mode="json"),
                "report_path": str(report_path),
                "report_sha256": report_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report.accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
