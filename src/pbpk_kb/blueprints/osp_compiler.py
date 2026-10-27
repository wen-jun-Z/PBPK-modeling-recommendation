from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbpk_kb.blueprints.schemas import (
    BlueprintCompilationReport,
    BlueprintTemplate,
    BlueprintValidationReport,
    CompilationArtifact,
    EstimatedParameter,
    OSPCompilerRuntimeIdentity,
    PBPKBlueprint,
    RuntimeDependency,
    advance_blueprint_status,
    compilation_report_checksum,
)
from pbpk_kb.blueprints.template_catalog import (
    instantiated_processes,
    resolve_template_asset_path,
)
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.osp_pkml_parser import (
    parse_pkml_simulation,
    parse_result_csv,
)


COMPILER_VERSION = "osp-blueprint-compiler-1.1.0"
COMPILATION_NAMESPACE = uuid.UUID(
    "74aa4680-76da-4db0-b978-a1fa6bf0aa1c"
)


class OSPCompilationError(RuntimeError):
    pass


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(value.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _artifact(
    path: Path,
    *,
    output_directory: Path,
    artifact_kind: str,
) -> CompilationArtifact:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(
            output_directory.resolve()
        ).as_posix()
    except ValueError as exc:
        raise OSPCompilationError(
            f"Compilation artifact is outside output root: {resolved}"
        ) from exc
    return CompilationArtifact(
        artifact_kind=artifact_kind,
        relative_path=relative,
        sha256=sha256_file(resolved),
        byte_size=resolved.stat().st_size,
    )


def _r_vector(values: list[str]) -> str:
    return "c(" + ",".join(json.dumps(value) for value in values) + ")"


def _r_number_vector(values: list[float]) -> str:
    return "c(" + ",".join(format(value, ".17g") for value in values) + ")"


def _compiler_script(
    *,
    r_library: Path,
    skeleton_path: Path,
    compiled_path: Path,
    results_path: Path,
    session_path: Path,
    parameter_paths: list[str],
    parameter_values: list[float],
    parameter_units: list[str],
    output_paths: list[str],
) -> str:
    return "\n".join(
        [
            "options(warn = 1)",
            (
                ".libPaths(c("
                + json.dumps(str(r_library.resolve()))
                + ", .libPaths()))"
            ),
            "suppressPackageStartupMessages(library(ospsuite))",
            (
                "simulation <- loadSimulation("
                + json.dumps(str(skeleton_path.resolve()))
                + ", loadFromCache = FALSE, addToCache = FALSE, "
                "resetIds = FALSE)"
            ),
            f"output_paths <- {_r_vector(output_paths)}",
            (
                "setOutputs(quantitiesOrPaths = output_paths, "
                "simulation = simulation, stopIfNotFound = TRUE)"
            ),
            f"parameter_paths <- {_r_vector(parameter_paths)}",
            f"parameter_values <- {_r_number_vector(parameter_values)}",
            f"parameter_units <- {_r_vector(parameter_units)}",
            (
                "setParameterValuesByPath("
                "parameterPaths = parameter_paths, "
                "values = parameter_values, "
                "simulation = simulation, units = parameter_units, "
                "stopIfNotFound = TRUE)"
            ),
            (
                "saveSimulation(simulation, "
                + json.dumps(str(compiled_path.resolve()))
                + ")"
            ),
            (
                "results <- runSimulations("
                "simulation, silentMode = FALSE, stopIfFails = TRUE)[[1]]"
            ),
            (
                "exportResultsToCSV(results, "
                + json.dumps(str(results_path.resolve()))
                + ")"
            ),
            f"sink({json.dumps(str(session_path.resolve()))})",
            "print(sessionInfo())",
            "sink()",
            "",
        ]
    )


def _runtime_environment(
    *,
    r_library: Path,
    dotnet_root: Path,
) -> dict[str, str]:
    return {
        **os.environ,
        "R_LIBS_USER": str(r_library.resolve()),
        "DOTNET_ROOT": str(dotnet_root.resolve()),
        "PATH": (
            str(dotnet_root.resolve())
            + os.pathsep
            + os.environ.get("PATH", "")
        ),
    }


def _runtime_identity(
    *,
    r_executable: Path,
    r_library: Path,
    ospsuite_package_archive: Path,
    dotnet_root: Path,
    dotnet_release_metadata_path: Path,
    template: BlueprintTemplate,
) -> tuple[OSPCompilerRuntimeIdentity, dict[str, Any]]:
    rscript = r_executable.with_name("Rscript")
    for executable in (r_executable, rscript):
        if not executable.is_file():
            raise OSPCompilationError(
                f"R runtime executable is missing: {executable}"
            )
    dotnet_executable = dotnet_root / "dotnet"
    if not dotnet_executable.is_file():
        raise OSPCompilationError(
            f".NET runtime executable is missing: {dotnet_executable}"
        )
    if not dotnet_release_metadata_path.is_file():
        raise OSPCompilationError(
            ".NET release metadata file is missing: "
            f"{dotnet_release_metadata_path}"
        )
    dotnet_release_metadata_sha256 = sha256_file(
        dotnet_release_metadata_path
    )
    if dotnet_release_metadata_sha256 != (
        template.target_runtime.dotnet_release_metadata_sha256
    ):
        raise OSPCompilationError(
            ".NET release metadata checksum does not match the template lock."
        )
    runtime_environment = _runtime_environment(
        r_library=r_library,
        dotnet_root=dotnet_root,
    )
    dotnet_completed = subprocess.run(
        [str(dotnet_executable), "--list-runtimes"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        env=runtime_environment,
    )
    if dotnet_completed.returncode:
        raise OSPCompilationError(
            "Unable to inspect locked .NET runtime: "
            + (
                dotnet_completed.stderr.strip()
                or dotnet_completed.stdout.strip()
            )
        )
    expected_dotnet_line = (
        "Microsoft.NETCore.App "
        + template.target_runtime.dotnet_runtime_version
        + " "
    )
    if not any(
        line.startswith(expected_dotnet_line)
        for line in dotnet_completed.stdout.splitlines()
    ):
        raise OSPCompilationError(
            "Installed .NET runtime version does not match the template."
        )
    package_sha256 = sha256_file(ospsuite_package_archive)
    if package_sha256 != template.target_runtime.package_sha256:
        raise OSPCompilationError(
            "OSPSuite-R package checksum does not match the template lock."
        )
    expression = "\n".join(
        [
            (
                ".libPaths(c("
                + json.dumps(str(r_library.resolve()))
                + ", .libPaths()))"
            ),
            "library(ospsuite)",
            (
                "deps <- as.data.frame(installed.packages()"
                "[, c('Package', 'Version')], stringsAsFactors = FALSE)"
            ),
            "deps <- deps[order(deps$Package), ]",
            "cat(paste(R.version$major, R.version$minor, sep='.'), '\\n')",
            (
                "write.table(deps, file=stdout(), sep='\\t', "
                "row.names=FALSE, col.names=FALSE, quote=FALSE)"
            ),
        ]
    )
    completed = subprocess.run(
        [str(rscript), "--vanilla", "-e", expression],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=runtime_environment,
    )
    if completed.returncode:
        raise OSPCompilationError(
            "Unable to inspect locked OSPSuite-R environment: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    lines = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]
    if not lines:
        raise OSPCompilationError("R runtime inspection returned no output.")
    r_version = lines[0].strip()
    dependencies: list[RuntimeDependency] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 2:
            continue
        dependencies.append(
            RuntimeDependency(name=fields[0], version=fields[1])
        )
    dependency_versions = {
        item.name: item.version for item in dependencies
    }
    if dependency_versions.get("ospsuite") != (
        template.target_runtime.ospsuite_r_version
    ):
        raise OSPCompilationError(
            "Installed OSPSuite-R version does not match the template."
        )
    if r_version != template.target_runtime.r_version:
        raise OSPCompilationError(
            "Installed R version does not match the template."
        )
    if platform.system() != "Linux" or platform.machine() not in {
        "x86_64",
        "AMD64",
    }:
        raise OSPCompilationError(
            "OSP compiler requires the locked linux-x86_64 platform."
        )
    environment_payload = {
        "platform": platform.platform(),
        "r_version": r_version,
        "r_executable_sha256": sha256_file(r_executable),
        "rscript_executable_sha256": sha256_file(rscript),
        "ospsuite_r_package_sha256": package_sha256,
        "dotnet_runtime_version": (
            template.target_runtime.dotnet_runtime_version
        ),
        "dotnet_executable_sha256": sha256_file(dotnet_executable),
        "dotnet_release_metadata_sha256": (
            dotnet_release_metadata_sha256
        ),
        "dotnet_runtime_inventory": dotnet_completed.stdout.splitlines(),
        "dependencies": [
            item.model_dump(mode="json") for item in dependencies
        ],
    }
    identity = OSPCompilerRuntimeIdentity(
        platform="linux-x86_64",
        r_version=r_version,
        r_executable_sha256=environment_payload[
            "r_executable_sha256"
        ],
        rscript_executable_sha256=environment_payload[
            "rscript_executable_sha256"
        ],
        ospsuite_r_version=dependency_versions["ospsuite"],
        ospsuite_r_package_sha256=package_sha256,
        dotnet_runtime_version=(
            template.target_runtime.dotnet_runtime_version
        ),
        dotnet_executable_sha256=environment_payload[
            "dotnet_executable_sha256"
        ],
        dotnet_release_metadata_sha256=(
            dotnet_release_metadata_sha256
        ),
        dependencies=dependencies,
        environment_digest=canonical_sha256(environment_payload),
    )
    return identity, environment_payload


def _parameter_updates(
    *,
    blueprint: PBPKBlueprint,
    template: BlueprintTemplate,
) -> tuple[list[str], list[float], list[str]]:
    slots = {item.model_path: item for item in template.parameter_slots}
    paths: list[str] = []
    values: list[float] = []
    units: list[str] = []
    for parameter in sorted(
        blueprint.parameters,
        key=lambda item: item.model_path.casefold(),
    ):
        value = getattr(parameter, "value", None)
        if value is None and isinstance(parameter, EstimatedParameter):
            value = parameter.initial_value
        if value is None:
            if parameter.blocking:
                raise OSPCompilationError(
                    f"Blocking parameter has no value: "
                    f"{parameter.parameter_id}"
                )
            continue
        if not math.isfinite(value):
            raise OSPCompilationError(
                f"Parameter is non-finite: {parameter.parameter_id}"
            )
        slot = slots.get(parameter.model_path)
        if slot is None:
            raise OSPCompilationError(
                f"Parameter path is not whitelisted: "
                f"{parameter.model_path}"
            )
        if parameter.canonical_unit != slot.canonical_unit:
            raise OSPCompilationError(
                f"Parameter unit differs from the template slot: "
                f"{parameter.model_path}"
            )
        if not slot.apply_to_runtime:
            continue
        paths.append(parameter.model_path)
        values.append(value)
        units.append(slot.canonical_unit)
    if not paths:
        raise OSPCompilationError(
            "The blueprint has no resolved parameter updates."
        )
    return paths, values, units


def _structure_signature(records: Any) -> dict[str, Any]:
    return {
        "compounds": sorted(records.summary.compounds, key=str.casefold),
        "species": sorted(records.summary.species, key=str.casefold),
        "routes": sorted(records.summary.routes, key=str.casefold),
        "compartments": sorted(
            records.summary.compartments,
            key=str.casefold,
        ),
        "processes": sorted(
            (
                process.process_id,
                process.name,
                process.process_type,
                tuple(process.educt_identifiers),
                tuple(process.product_identifiers),
                tuple(process.modifier_identifiers),
            )
            for process in records.processes
        ),
    }


def _report(
    *,
    payload: dict[str, Any],
) -> BlueprintCompilationReport:
    draft = BlueprintCompilationReport.model_construct(
        **payload,
        checksum="0" * 64,
    )
    payload["checksum"] = compilation_report_checksum(
        draft.model_dump(mode="json")
    )
    return BlueprintCompilationReport.model_validate(payload)


def compile_osp_blueprint(
    *,
    blueprint: PBPKBlueprint,
    validation_report: BlueprintValidationReport,
    template: BlueprintTemplate,
    asset_root: Path,
    output_directory: Path,
    r_executable: Path,
    r_library: Path,
    ospsuite_package_archive: Path,
    dotnet_root: Path,
    dotnet_release_metadata_path: Path,
    timeout_seconds: int = 3600,
) -> tuple[Path, BlueprintCompilationReport]:
    if blueprint.status != "structurally_validated":
        raise OSPCompilationError(
            "Only a structurally_validated blueprint may compile."
        )
    if (
        not validation_report.accepted
        or not validation_report.compile_eligible
        or validation_report.blueprint_id != blueprint.blueprint_id
        or validation_report.blueprint_checksum != blueprint.checksum
    ):
        raise OSPCompilationError(
            "Blueprint validation report does not authorize compilation."
        )
    if blueprint.template.template_id != template.template_id:
        raise OSPCompilationError("Blueprint template identity mismatch.")
    skeleton_source = resolve_template_asset_path(
        asset_root,
        template.source_skeleton_relative_path,
    )
    if sha256_file(skeleton_source) != template.source_skeleton_sha256:
        raise OSPCompilationError("Template skeleton checksum drift.")
    if output_directory.exists():
        raise OSPCompilationError(
            f"Immutable compilation output exists: {output_directory}"
        )
    runtime, environment_payload = _runtime_identity(
        r_executable=r_executable,
        r_library=r_library,
        ospsuite_package_archive=ospsuite_package_archive,
        dotnet_root=dotnet_root,
        dotnet_release_metadata_path=dotnet_release_metadata_path,
        template=template,
    )
    parameter_paths, parameter_values, parameter_units = (
        _parameter_updates(blueprint=blueprint, template=template)
    )
    scenario = blueprint.simulation_scenarios[0]
    temporary = output_directory.with_name(
        output_directory.name + ".tmp"
    )
    if temporary.exists():
        raise OSPCompilationError(
            f"Stale compilation temporary directory exists: {temporary}"
        )
    temporary.mkdir(parents=True)
    input_directory = temporary / "input"
    output_root = temporary / "output"
    audit_directory = temporary / "audit"
    input_blueprint_path = input_directory / "blueprint.json"
    validation_path = input_directory / "validation_report.json"
    skeleton_path = input_directory / "template_skeleton.pkml"
    script_path = audit_directory / "compile_and_run.R"
    stdout_path = audit_directory / "stdout.log"
    stderr_path = audit_directory / "stderr.log"
    session_path = audit_directory / "runtime_session.txt"
    dependency_path = audit_directory / "dependency_lock.json"
    dotnet_metadata_copy = audit_directory / "dotnet_releases.json"
    compiled_path = output_root / "simulation.pkml"
    results_path = output_root / "simulation_results.csv"
    executable_blueprint_path = output_root / "blueprint.json"

    write_json_atomic(input_blueprint_path, blueprint, canonical=True)
    write_json_atomic(
        validation_path,
        validation_report,
        canonical=True,
    )
    _copy_atomic(skeleton_source, skeleton_path)
    _copy_atomic(dotnet_release_metadata_path, dotnet_metadata_copy)
    validation_sha256 = sha256_file(validation_path)
    compilation_id = uuid.uuid5(
        COMPILATION_NAMESPACE,
        (
            f"{blueprint.checksum}\n{validation_sha256}\n"
            f"{runtime.environment_digest}"
        ),
    )
    write_json_atomic(
        dependency_path,
        {
            "compiler_version": COMPILER_VERSION,
            "runtime": runtime,
            "environment": environment_payload,
            "parameter_update_count": len(parameter_paths),
        },
        canonical=True,
    )
    _write_text_atomic(
        script_path,
        _compiler_script(
            r_library=r_library,
            skeleton_path=skeleton_path,
            compiled_path=compiled_path,
            results_path=results_path,
            session_path=session_path,
            parameter_paths=parameter_paths,
            parameter_values=parameter_values,
            parameter_units=parameter_units,
            output_paths=scenario.output_paths,
        ),
    )
    command = [
        str(r_executable.with_name("Rscript")),
        "--vanilla",
        str(script_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        env=_runtime_environment(
            r_library=r_library,
            dotnet_root=dotnet_root,
        ),
    )
    _write_text_atomic(stdout_path, completed.stdout)
    _write_text_atomic(stderr_path, completed.stderr)

    checks: dict[str, bool] = {
        "runtime_exit_zero": completed.returncode == 0,
        "compiled_pkml_exists": compiled_path.is_file(),
        "results_csv_exists": results_path.is_file(),
        "runtime_session_exists": session_path.is_file(),
        "compiled_pkml_parse": False,
        "structure_preserved": False,
        "compound_identity_preserved": False,
        "species_preserved": False,
        "process_coverage_preserved": False,
        "result_csv_parse": False,
        "result_rows_present": False,
        "result_values_finite": False,
        "result_output_paths_match": False,
        "artifact_checksum_coverage": False,
    }
    errors: list[dict[str, Any]] = []
    source_records = None
    compiled_records = None
    csv_summary = None
    project_key = hashlib.sha256(
        template.template_id.encode("utf-8")
    ).hexdigest()[:32]
    if compiled_path.is_file():
        try:
            source_records = parse_pkml_simulation(
                skeleton_path,
                artifact_relative_path="input/template_skeleton.pkml",
                artifact_sha256=sha256_file(skeleton_path),
                job_id=f"blueprint-compile:{compilation_id}",
                project_key=project_key,
                simulation_id=scenario.scenario_id,
                simulation_name=scenario.name,
            )
            compiled_records = parse_pkml_simulation(
                compiled_path,
                artifact_relative_path="output/simulation.pkml",
                artifact_sha256=sha256_file(compiled_path),
                job_id=f"blueprint-compile:{compilation_id}",
                project_key=project_key,
                simulation_id=scenario.scenario_id,
                simulation_name=scenario.name,
            )
            checks["compiled_pkml_parse"] = (
                compiled_records.summary.parse_status == "parsed"
            )
            checks["structure_preserved"] = (
                _structure_signature(source_records)
                == _structure_signature(compiled_records)
            )
            checks["compound_identity_preserved"] = (
                blueprint.compound_name.casefold()
                in {
                    item.casefold()
                    for item in compiled_records.summary.compounds
                }
            )
            checks["species_preserved"] = (
                blueprint.species.casefold()
                in {
                    item.casefold()
                    for item in compiled_records.summary.species
                }
            )
            source_instantiated_processes = instantiated_processes(
                source_records
            )
            compiled_instantiated_processes = instantiated_processes(
                compiled_records
            )
            source_process_types = Counter(
                item.process_type for item in source_instantiated_processes
            )
            compiled_process_types = Counter(
                item.process_type for item in compiled_instantiated_processes
            )
            required_process_types = {
                item.process_type for item in blueprint.processes
            }
            checks["process_coverage_preserved"] = (
                source_process_types == compiled_process_types
                and required_process_types.issubset(
                    set(compiled_process_types)
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "reason_code": "COMPILED_PKML_VALIDATION_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if results_path.is_file():
        csv_summary = parse_result_csv(
            results_path,
            artifact_relative_path="output/simulation_results.csv",
            source_relative_path="simulation_results.csv",
        )
        checks["result_csv_parse"] = (
            csv_summary.parse_status == "parsed"
            and csv_summary.result_kind == "timeseries"
        )
        checks["result_rows_present"] = csv_summary.row_count > 0
        checks["result_values_finite"] = (
            csv_summary.non_finite_value_count == 0
        )
        checks["result_output_paths_match"] = [
            item.name for item in csv_summary.outputs
        ] == scenario.output_paths
        if not checks["result_output_paths_match"]:
            errors.append(
                {
                    "reason_code": "RESULT_OUTPUT_PATHS_MISMATCH",
                    "expected": scenario.output_paths,
                    "actual": [
                        item.name for item in csv_summary.outputs
                    ],
                }
            )
        if csv_summary.parse_status == "failed":
            errors.append(
                {
                    "reason_code": "RESULT_CSV_VALIDATION_FAILED",
                    "error": csv_summary.error,
                }
            )
    if completed.returncode:
        errors.append(
            {
                "reason_code": "OSPSUITE_R_NONZERO_EXIT",
                "return_code": completed.returncode,
            }
        )

    compile_checks = {
        "runtime_exit_zero",
        "compiled_pkml_exists",
        "compiled_pkml_parse",
        "structure_preserved",
        "compound_identity_preserved",
        "species_preserved",
        "process_coverage_preserved",
    }
    execution_checks = compile_checks | {
        "results_csv_exists",
        "runtime_session_exists",
        "result_csv_parse",
        "result_rows_present",
        "result_values_finite",
        "result_output_paths_match",
    }
    compile_accepted = all(checks[name] for name in compile_checks)
    preliminary_execution_accepted = all(
        checks[name] for name in execution_checks
    )
    executable_blueprint = None

    artifact_specs: list[tuple[Path, str]] = [
        (input_blueprint_path, "input_blueprint"),
        (validation_path, "validation_report"),
        (skeleton_path, "template_skeleton"),
        (script_path, "compiler_script"),
        (stdout_path, "stdout_log"),
        (stderr_path, "stderr_log"),
        (dependency_path, "dependency_lock"),
        (dotnet_metadata_copy, "dotnet_release_metadata"),
    ]
    if session_path.is_file():
        artifact_specs.append((session_path, "runtime_session"))
    if compiled_path.is_file():
        artifact_specs.append((compiled_path, "compiled_simulation"))
    if results_path.is_file():
        artifact_specs.append((results_path, "simulation_results"))
    artifacts = [
        _artifact(
            path,
            output_directory=temporary,
            artifact_kind=kind,
        )
        for path, kind in artifact_specs
    ]
    checks["artifact_checksum_coverage"] = all(
        (
            temporary / artifact.relative_path
        ).is_file()
        and sha256_file(temporary / artifact.relative_path)
        == artifact.sha256
        for artifact in artifacts
    )
    execution_accepted = (
        preliminary_execution_accepted
        and checks["artifact_checksum_coverage"]
    )
    if preliminary_execution_accepted and not checks[
        "artifact_checksum_coverage"
    ]:
        errors.append(
            {"reason_code": "ARTIFACT_CHECKSUM_COVERAGE_FAILED"}
        )
    if execution_accepted:
        compiled_blueprint = advance_blueprint_status(
            blueprint,
            status="compiled",
            created_at=datetime.now(UTC),
        )
        executable_blueprint = advance_blueprint_status(
            compiled_blueprint,
            status="executable",
            created_at=datetime.now(UTC),
        )
        write_json_atomic(
            executable_blueprint_path,
            executable_blueprint,
            canonical=True,
        )
        executable_artifact = _artifact(
            executable_blueprint_path,
            output_directory=temporary,
            artifact_kind="executable_blueprint",
        )
        artifacts.append(executable_artifact)
        checks["artifact_checksum_coverage"] = (
            executable_blueprint_path.is_file()
            and sha256_file(executable_blueprint_path)
            == executable_artifact.sha256
        )
        if not checks["artifact_checksum_coverage"]:
            executable_blueprint_path.unlink(missing_ok=True)
            executable_blueprint = None
            execution_accepted = False
            errors.append(
                {"reason_code": "EXECUTABLE_BLUEPRINT_CHECKSUM_FAILED"}
            )
    status = (
        "executable"
        if execution_accepted
        else "execution_failed"
        if compile_accepted
        else "compile_failed"
    )
    report_payload: dict[str, Any] = {
        "compiler_version": COMPILER_VERSION,
        "compilation_id": compilation_id,
        "compiled_at": datetime.now(UTC),
        "input_blueprint_id": blueprint.blueprint_id,
        "input_blueprint_checksum": blueprint.checksum,
        "output_blueprint_checksum": (
            executable_blueprint.checksum
            if executable_blueprint is not None
            else None
        ),
        "validation_report_sha256": validation_sha256,
        "template_id": template.template_id,
        "template_skeleton_sha256": template.source_skeleton_sha256,
        "status": status,
        "compile_accepted": compile_accepted,
        "execution_accepted": execution_accepted,
        "command": command,
        "runtime": runtime,
        "artifacts": artifacts,
        "checks": checks,
        "counts": {
            "parameter_updates": len(parameter_paths),
            "requested_outputs": len(scenario.output_paths),
            "source_processes": (
                len(source_records.processes)
                if source_records is not None
                else 0
            ),
            "compiled_processes": (
                len(compiled_records.processes)
                if compiled_records is not None
                else 0
            ),
            "result_rows": (
                csv_summary.row_count
                if csv_summary is not None
                else 0
            ),
            "result_numeric_values": (
                csv_summary.numeric_value_count
                if csv_summary is not None
                else 0
            ),
            "artifacts": len(artifacts),
            "errors": len(errors),
        },
        "errors": errors,
    }
    report = _report(payload=report_payload)
    report_path = audit_directory / "compilation_report.json"
    write_json_atomic(report_path, report, canonical=True)
    write_json_atomic(
        audit_directory / "compilation_report.sha256",
        {
            "file": report_path.name,
            "sha256": sha256_file(report_path),
        },
        canonical=True,
    )
    temporary.replace(output_directory)
    return (
        output_directory / "audit" / "compilation_report.json",
        report,
    )
