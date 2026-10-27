from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from unittest.mock import patch

import pytest

from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_execution_plan,
    write_json_atomic,
)
from pbpk_kb.model_assets.execution_validation import validate_execution_result
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    OSPExecutionArtifact,
    OSPExecutionPlan,
    OSPExecutionProjectSpec,
    OSPExecutionResult,
    OSPExecutionSimulationSpec,
    OSPProjectExecutionResult,
    OSPRuntimeLock,
    OSPSimulationExecutionResult,
    OSPWorkerRuntimeIdentity,
    RuntimeDownloadLock,
)
from pbpk_kb.model_assets.windows_worker import (
    OSPWorkerError,
    _execute_project,
    _ingest_scratch_exports,
    _runtime_identity,
    _validate_scratch_path_budget,
    build_export_command,
)
from pbpk_kb.model_assets.result_bundle import build_result_bundle
from pbpk_kb.model_assets.worker_bundle import build_windows_worker_bundle


def _download_lock(name: str, checksum: str = "a" * 64) -> RuntimeDownloadLock:
    return RuntimeDownloadLock(
        name=name,
        url=f"https://example.org/{name}",
        sha256=checksum,
    )


def _plan(
    *,
    repository_commit: str = "b" * 40,
    runtime_archive_sha256: str = "a" * 64,
    project_sha256: str = "d" * 64,
) -> OSPExecutionPlan:
    return OSPExecutionPlan(
        job_id="osp-exec-" + "1" * 32,
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
        source_id="osp_pbpk_model_library",
        repository_url="https://example.org/models.git",
        repository_release_tag="v12.3.1",
        repository_commit=repository_commit,
        inventory_run_id="inventory-run",
        inventory_summary_sha256="c" * 64,
        runtime=OSPRuntimeLock(
            suite_release_name="OSP Suite 12 Update 3 Hotfix 1",
            suite_version="12.3.263",
            suite_installer=_download_lock("suite.exe"),
            pksim_version="12.3.173",
            pksim_portable=_download_lock(
                "PKSim-Portable.12.3.173.zip",
                runtime_archive_sha256,
            ),
            pksim_cli_relative_path="PKSim.CLI.exe",
            ospsuite_r_version="12.4.3",
            ospsuite_r_linux_binary=_download_lock("ospsuite-r.tar.gz"),
        ),
        export_flags=["-k", "-x", "-c", "-r"],
        command_timeout_seconds=60,
        required_export_extensions=[".pkml", ".xml", ".csv"],
        projects=[
            OSPExecutionProjectSpec(
                project_key="2" * 32,
                model_family="Midazolam",
                relative_path="Midazolam/Midazolam-Model.pksim5",
                project_sha256=project_sha256,
                project_name="Midazolam-Model",
                project_version=80,
                simulations=[
                    OSPExecutionSimulationSpec(
                        simulation_id="simulation-1",
                        name="po 5 mg",
                        simulation_mode="Individual",
                    )
                ],
            )
        ],
    )


def _artifact(path: Path, root: Path) -> OSPExecutionArtifact:
    return OSPExecutionArtifact(
        relative_path=path.relative_to(root).as_posix(),
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
        extension=path.suffix.casefold(),
        source_relative_path=path.name,
        simulation_id="simulation-1",
        simulation_name="po 5 mg",
    )


def test_export_command_uses_locked_official_flags(tmp_path: Path) -> None:
    command = build_export_command(
        cli_path=Path("C:/osp/PKSim.CLI.exe"),
        project_path=Path("C:/models/Midazolam/Midazolam-Model.pksim5"),
        output_directory=tmp_path / "output",
        plan=_plan(),
        simulation_names=["po 5 mg"],
    )

    assert command[1:2] == ["export"]
    assert command[-4:] == ["-k", "-x", "-c", "-r"]
    assert command[command.index("-p") + 1].endswith("Midazolam-Model.pksim5")
    assert command[command.index("-s") + 1 : -4] == ["po 5 mg"]


def test_canonical_hash_serializes_nested_pydantic_models() -> None:
    first = canonical_sha256({"plan": _plan(), "projects": _plan().projects})
    second = canonical_sha256({"projects": _plan().projects, "plan": _plan()})

    assert first == second


def test_runtime_identity_accepts_pksim_informational_exit_code_one(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "PKSim.CLI.exe"
    cli.write_bytes(b"fixture")
    help_text = " ".join(
        [
            "--project",
            "--output",
            "--simulations",
            "--csv",
            "--xml",
            "--pkml",
            "--run",
        ]
    )
    probes = [
        subprocess.CompletedProcess(
            [str(cli), "--version"],
            1,
            stdout="",
            stderr="PKSim.CLI 12.3.173+commit\n",
        ),
        subprocess.CompletedProcess(
            [str(cli), "export", "--help"],
            1,
            stdout="",
            stderr=help_text,
        ),
    ]

    with patch(
        "pbpk_kb.model_assets.windows_worker._run_probe",
        side_effect=probes,
    ):
        identity = _runtime_identity(
            cli_path=cli,
            runtime_manifest={
                "pksim_version": "12.3.173",
                "archive_sha256": "a" * 64,
                "cli_relative_path": "PKSim.CLI.exe",
            },
        )

    assert identity.cli_version_return_code == 1
    assert identity.cli_help_return_code == 1
    assert identity.cli_version_output.startswith("PKSim.CLI 12.3.173")
    assert identity.cli_help_output == help_text


def test_scratch_path_budget_accepts_short_root_and_rejects_long_root() -> None:
    plan = _plan()
    plan.projects[0].simulations[0].name = "simulation " + ("x" * 70)

    maximum = _validate_scratch_path_budget(
        plan=plan,
        scratch_root=Path("D:/PKX/job"),
    )

    assert maximum < 240
    with pytest.raises(OSPWorkerError, match="scratch path budget"):
        _validate_scratch_path_budget(
            plan=plan,
            scratch_root=Path("D:/" + ("long-root/" * 20)),
        )


def test_scratch_exports_are_mapped_and_archived_with_short_names(
    tmp_path: Path,
) -> None:
    project = _plan().projects[0]
    project.simulations[0].name = "dose 1 mg/kg"
    scratch = tmp_path / "scratch"
    source_directory = scratch / "dose 1 mg_kg"
    source_directory.mkdir(parents=True)
    for name, value in [
        ("dose 1 mg_kg.pkml", "pkml"),
        ("dose 1 mg_kg.xml", "xml"),
        ("dose 1 mg_kg-Results.csv", "csv"),
    ]:
        (source_directory / name).write_text(value, encoding="utf-8")
    result_root = tmp_path / "result"
    export_root = result_root / "projects" / "project" / "attempt_001" / "exports"

    artifacts, directory_count, reasons = _ingest_scratch_exports(
        project=project,
        scratch_export_root=scratch,
        export_root=export_root,
        result_root=result_root,
    )

    assert directory_count == 1
    assert reasons == []
    assert {artifact.extension for artifact in artifacts} == {
        ".csv",
        ".pkml",
        ".xml",
    }
    assert {artifact.simulation_id for artifact in artifacts} == {
        "simulation-1"
    }
    assert all(artifact.source_relative_path for artifact in artifacts)
    assert (export_root / "s0001" / "simulation.pkml").is_file()
    assert (export_root / "s0001" / "simulation.xml").is_file()
    assert (export_root / "s0001" / "results.csv").is_file()


def test_project_execution_isolates_simulation_failures(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    project_path = repository / "Midazolam" / "Midazolam-Model.pksim5"
    project_path.parent.mkdir(parents=True)
    project_path.write_bytes(b"project")
    plan = _plan(project_sha256=sha256_file(project_path))
    plan.projects[0].simulations.append(
        OSPExecutionSimulationSpec(
            simulation_id="simulation-2",
            name="po 10 mg",
            simulation_mode="Individual",
        )
    )
    calls: list[list[str]] = []

    def run_simulation(command: list[str], **_kwargs: object):
        calls.append(command)
        simulation_name = command[command.index("-s") + 1]
        if simulation_name == "po 5 mg":
            output_root = Path(command[command.index("-o") + 1])
            simulation_root = output_root / simulation_name
            simulation_root.mkdir(parents=True)
            for name, content in [
                (f"{simulation_name}.pkml", "pkml"),
                (f"{simulation_name}.xml", "xml"),
                (f"{simulation_name}-Results.csv", "Time,Value\n0,0\n"),
            ]:
                (simulation_root / name).write_text(content, encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "success", "")
        return subprocess.CompletedProcess(command, 1, "failed", "conversion error")

    result_root = tmp_path / "results"
    result_root.mkdir()
    with patch(
        "pbpk_kb.model_assets.windows_worker.subprocess.run",
        side_effect=run_simulation,
    ):
        result = _execute_project(
            project=plan.projects[0],
            plan=plan,
            repository_root=repository,
            cli_path=tmp_path / "PKSim.CLI.exe",
            result_root=result_root,
            scratch_root=tmp_path / "scratch",
        )

    assert len(calls) == 2
    assert [call[call.index("-s") + 1] for call in calls] == [
        "po 5 mg",
        "po 10 mg",
    ]
    assert result.status == "failed"
    assert [item.status for item in result.simulation_results] == [
        "passed",
        "failed",
    ]
    assert "SIMULATION_EXECUTION_FAILURES" in result.reason_codes
    assert result.simulation_results[1].stderr_artifact.byte_size > 0


def test_windows_worker_bundle_is_checksum_locked(tmp_path: Path) -> None:
    repository = tmp_path / "models"
    subprocess.run(["git", "init", "-b", "master", str(repository)], check=True)
    (repository / "README.md").write_text("models\n", encoding="utf-8")
    project_path = repository / "Midazolam" / "Midazolam-Model.pksim5"
    project_path.parent.mkdir()
    project_path.write_bytes(b"project")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-m",
            "models",
        ],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_archive = tmp_path / "PKSim-Portable.12.3.173.zip"
    with zipfile.ZipFile(runtime_archive, "w") as archive:
        archive.writestr("PKSim.CLI.exe", b"fixture")
    plan_directory = tmp_path / "plan"
    plan_path, _plan_sha = write_execution_plan(
        plan_directory,
            _plan(
                repository_commit=commit,
                runtime_archive_sha256=sha256_file(runtime_archive),
                project_sha256=sha256_file(project_path),
            ),
    )
    project_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "worker-bundle"

    summary = build_windows_worker_bundle(
        project_root=project_root,
        plan_path=plan_path,
        repository_root=repository,
        runtime_archive=runtime_archive,
        output_directory=output,
    )

    assert Path(summary["bundle_archive"]).is_file()
    assert sha256_file(Path(summary["bundle_archive"])) == summary[
        "bundle_archive_sha256"
    ]
    assert (output / "bundle_manifest.json").is_file()
    assert (
        output
        / "repository"
        / "OSP-PBPK-Model-Library-v12.3.1.zip"
    ).is_file()


def test_windows_worker_bundle_accepts_pinned_file_snapshot(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "snapshot"
    project_path = repository / "extracted" / "Caffeine.pksim5"
    project_path.parent.mkdir(parents=True)
    project_path.write_bytes(b"pinned-project")
    (repository / "LICENSE").write_text(
        "GNU General Public License Version 2",
        encoding="utf-8",
    )
    runtime_archive = tmp_path / "PKSim-Portable.12.3.173.zip"
    with zipfile.ZipFile(runtime_archive, "w") as archive:
        archive.writestr("PKSim.CLI.exe", b"fixture")
    plan = _plan(
        repository_commit="a" * 40,
        runtime_archive_sha256=sha256_file(runtime_archive),
    ).model_copy(
        update={
            "source_id": "lactation_pbpk",
            "repository_release_tag": "a7c89f3",
            "projects": [
                OSPExecutionProjectSpec(
                    project_key="2" * 32,
                    model_family="Caffeine",
                    relative_path="extracted/Caffeine.pksim5",
                    project_sha256=sha256_file(project_path),
                    project_name="Caffeine",
                    project_version=77,
                    simulations=[
                        OSPExecutionSimulationSpec(
                            simulation_id="simulation-1",
                            name="PO 100 mg",
                            simulation_mode="Individual",
                        )
                    ],
                )
            ],
        }
    )
    plan_directory = tmp_path / "plan"
    plan_path, _plan_sha = write_execution_plan(plan_directory, plan)
    output = tmp_path / "worker-bundle"

    summary = build_windows_worker_bundle(
        project_root=Path(__file__).resolve().parents[2],
        plan_path=plan_path,
        repository_root=repository,
        runtime_archive=runtime_archive,
        output_directory=output,
    )

    archive_path = (
        output
        / "repository"
        / "lactation_pbpk-a7c89f3.zip"
    )
    manifest = json.loads(
        (output / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert archive_path.is_file()
    assert manifest["repository_snapshot_mode"] == "pinned_files"
    assert manifest["repository_root_name"] == (
        "lactation_pbpk-a7c89f3"
    )
    assert Path(summary["bundle_archive"]).is_file()
    with zipfile.ZipFile(archive_path) as archive:
        assert (
            "lactation_pbpk-a7c89f3/extracted/Caffeine.pksim5"
            in archive.namelist()
        )


def test_execution_result_validation_checks_all_exported_artifacts(
    tmp_path: Path,
) -> None:
    plan_directory = tmp_path / "plan"
    plan_path, plan_sha256 = write_execution_plan(plan_directory, _plan())
    result_root = tmp_path / "result"
    attempt = result_root / "projects" / ("2" * 32) / "attempt_001"
    exports = attempt / "exports" / "po 5 mg"
    exports.mkdir(parents=True)
    pkml = exports / "po 5 mg.pkml"
    xml = exports / "po 5 mg.xml"
    csv = exports / "po 5 mg.csv"
    stdout = attempt / "stdout.log"
    stderr = attempt / "stderr.log"
    simulation_stdout = attempt / "simulation_runs" / "s0001" / "stdout.log"
    simulation_stderr = attempt / "simulation_runs" / "s0001" / "stderr.log"
    for path, value in [
        (pkml, "<Simulation/>"),
        (xml, "<SimModel/>"),
        (csv, "Time,Value\n0,0\n"),
        (stdout, "success\n"),
        (stderr, ""),
        (simulation_stdout, "success\n"),
        (simulation_stderr, ""),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    result = OSPExecutionResult(
        job_id=_plan().job_id,
        plan_sha256=plan_sha256,
        started_at=datetime(2026, 7, 30, 1, tzinfo=UTC),
        completed_at=datetime(2026, 7, 30, 1, 1, tzinfo=UTC),
        worker=OSPWorkerRuntimeIdentity(
            platform="Windows",
            platform_release="11",
            python_version="3.12.3",
            hostname="worker",
            runtime_archive_sha256="a" * 64,
            cli_executable_sha256="e" * 64,
            cli_relative_path="PKSim.CLI.exe",
            cli_version_output="PKSim.CLI 12.3.173",
            cli_version_return_code=1,
            cli_help_output=(
                "--project --output --simulations --csv --xml --pkml --run"
            ),
            cli_help_return_code=1,
            cli_help_sha256=hashlib.sha256(
                b"--project --output --simulations --csv --xml --pkml --run"
            ).hexdigest(),
        ),
        status="passed",
        projects=[
            OSPProjectExecutionResult(
                project_key="2" * 32,
                relative_path="Midazolam/Midazolam-Model.pksim5",
                input_sha256="d" * 64,
                attempt=1,
                status="passed",
                execution_mode="per_simulation",
                started_at=datetime(2026, 7, 30, 1, tzinfo=UTC),
                completed_at=datetime(2026, 7, 30, 1, 1, tzinfo=UTC),
                duration_seconds=60,
                command=[],
                return_code=0,
                expected_simulation_count=1,
                output_directory_count=1,
                export_counts={".pkml": 1, ".xml": 1, ".csv": 1},
                artifacts=[
                    _artifact(pkml, result_root),
                    _artifact(xml, result_root),
                    _artifact(csv, result_root),
                ],
                stdout_artifact=_artifact(stdout, result_root),
                stderr_artifact=_artifact(stderr, result_root),
                simulation_results=[
                    OSPSimulationExecutionResult(
                        sequence=1,
                        simulation_id="simulation-1",
                        simulation_name="po 5 mg",
                        status="passed",
                        started_at=datetime(2026, 7, 30, 1, tzinfo=UTC),
                        completed_at=datetime(2026, 7, 30, 1, 1, tzinfo=UTC),
                        duration_seconds=60,
                        command=[
                            "C:/osp/PKSim.CLI.exe",
                            "export",
                            "-p",
                            "C:/models/Midazolam/Midazolam-Model.pksim5",
                            "-o",
                            str(attempt / "exports"),
                            "-s",
                            "po 5 mg",
                            "-k",
                            "-x",
                            "-c",
                            "-r",
                        ],
                        return_code=0,
                        export_counts={".pkml": 1, ".xml": 1, ".csv": 1},
                        stdout_artifact=_artifact(
                            simulation_stdout,
                            result_root,
                        ),
                        stderr_artifact=_artifact(
                            simulation_stderr,
                            result_root,
                        ),
                    )
                ],
            )
        ],
    )
    result_path = result_root / "execution_result.json"
    result_sha256 = write_json_atomic(result_path, result, canonical=True)
    write_json_atomic(
        result_root / "execution_result.sha256",
        {"file": result_path.name, "sha256": result_sha256},
        canonical=True,
    )

    report = validate_execution_result(
        plan_path=plan_path,
        result_path=result_path,
    )

    assert report.execution_probe_complete is True
    assert report.passed_project_count == 1
    assert report.verified_artifact_count == 7

    csv.write_text("Time,Value\n9,9\n", encoding="utf-8")
    failed_report = validate_execution_result(
        plan_path=plan_path,
        result_path=result_path,
    )
    assert failed_report.execution_probe_complete is False
    assert "ARTIFACT_CHECKSUM_MISMATCH" in failed_report.reason_codes


def test_result_bundle_contains_only_checksum_referenced_outputs(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    exports = result_root / "projects" / "project" / "attempt_005" / "exports" / "s0001"
    exports.mkdir(parents=True)
    referenced: list[dict[str, object]] = []
    for name, content in [
        ("simulation.pkml", "<Simulation/>"),
        ("simulation.xml", "<SimModel/>"),
        ("results.csv", "Time,Value\n0,0\n"),
    ]:
        path = exports / name
        path.write_text(content, encoding="utf-8")
        referenced.append(
            {
                "relative_path": path.relative_to(result_root).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
                "extension": path.suffix,
            }
        )
    attempt = exports.parents[1]
    stdout = attempt / "stdout.log"
    stderr = attempt / "stderr.log"
    stdout.write_text("ok\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    old_attempt = result_root / "projects" / "project" / "attempt_001" / "old.pkml"
    old_attempt.parent.mkdir(parents=True)
    old_attempt.write_text("obsolete", encoding="utf-8")

    def artifact(path: Path) -> dict[str, object]:
        return {
            "relative_path": path.relative_to(result_root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
            "extension": path.suffix,
        }

    result = {
        "job_id": "osp-exec-" + "a" * 32,
        "status": "passed",
        "projects": [
            {
                "artifacts": referenced,
                "stdout_artifact": artifact(stdout),
                "stderr_artifact": artifact(stderr),
            }
        ],
    }
    result_path = result_root / "execution_result.json"
    result_sha256 = write_json_atomic(result_path, result, canonical=True)
    write_json_atomic(
        result_root / "execution_result.sha256",
        {"file": result_path.name, "sha256": result_sha256},
        canonical=True,
    )
    output = tmp_path / "bundle.zip"

    summary = build_result_bundle(
        result_path=result_path,
        output_archive=output,
    )

    assert summary["file_count"] == 7
    assert sha256_file(output) == summary["sha256"]
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert not any("attempt_001" in name for name in names)
        manifest_name = (
            f"{result['job_id']}/bundle_manifest.json"
        )
        manifest = json.loads(archive.read(manifest_name))
        assert manifest["file_count"] == 7
        assert len(names) == 8
