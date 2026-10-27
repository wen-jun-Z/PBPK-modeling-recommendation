from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any

from pbpk_kb.model_assets.license_policy import sha256_file


LINUX_REPEAT_VERSION = "pbpk-linux-ospsuite-repeat-1.0.0"
DEFAULT_ABSOLUTE_TOLERANCE = 1e-7
DEFAULT_RELATIVE_TOLERANCE = 1e-6

R_RUNNER = r'''suppressPackageStartupMessages(library(ospsuite))
args <- commandArgs(trailingOnly = TRUE)
input <- normalizePath(args[[1]], mustWork = TRUE)
windows_csv <- normalizePath(args[[2]], mustWork = TRUE)
output_csv <- args[[3]]
runtime_manifest <- args[[4]]

headers <- names(read.csv(
  windows_csv,
  nrows = 1,
  check.names = FALSE,
  fileEncoding = "UTF-8-BOM"
))
if (length(headers) < 3) {
  stop("Windows CSV has no simulation output columns.")
}
output_paths <- sub(" \\[[^][]+\\]$", "", headers[-c(1, 2)])
simulation <- loadSimulation(
  input,
  loadFromCache = FALSE,
  addToCache = FALSE,
  resetIds = FALSE
)
setOutputs(output_paths, simulation, stopIfNotFound = TRUE)
result <- runSimulations(list(simulation), stopIfFails = TRUE)[[1]]
exportResultsToCSV(result, output_csv)
jsonlite::write_json(
  list(
    simulation_id = simulation$id,
    simulation_name = simulation$name,
    input_file = input,
    output_paths = output_paths,
    output_count = length(output_paths),
    ospsuite_version = as.character(packageVersion("ospsuite")),
    rsharp_version = as.character(packageVersion("rSharp")),
    r_version = R.version.string
  ),
  runtime_manifest,
  auto_unbox = TRUE,
  pretty = TRUE
)
'''


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        for row in rows:
            handle.write(
                (
                    json.dumps(row, ensure_ascii=True, sort_keys=True, default=str)
                    + "\n"
                ).encode("utf-8")
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _artifact_path(
    *,
    result_root: Path,
    simulation: Mapping[str, Any],
    extension: str,
    relative_path_suffix: str | None = None,
) -> Path:
    matches = [
        row
        for row in simulation.get("artifacts") or []
        if str(row.get("extension") or "").casefold() == extension.casefold()
        and (
            relative_path_suffix is None
            or str(row.get("relative_path") or "")
            .replace("\\", "/")
            .casefold()
            .endswith(relative_path_suffix.casefold())
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {extension} artifact for simulation "
            f"{simulation.get('sequence')}; found {len(matches)}."
        )
    row = matches[0]
    relative = str(row["relative_path"]).replace("\\", "/")
    path = result_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(row["sha256"]):
        raise ValueError(f"Windows simulation artifact checksum drifted: {path}")
    return path


def compare_result_csvs(
    *,
    windows_csv: Path,
    linux_csv: Path,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    windows_row_count = 0
    linux_row_count = 0
    mismatch_count = 0
    numeric_cell_count = 0
    max_absolute_difference = 0.0
    max_relative_difference = 0.0
    mismatch_examples: list[dict[str, Any]] = []
    with (
        windows_csv.open("r", encoding="utf-8-sig", newline="") as windows_handle,
        linux_csv.open("r", encoding="utf-8-sig", newline="") as linux_handle,
    ):
        windows_rows = csv.reader(windows_handle)
        linux_rows = csv.reader(linux_handle)
        windows_header = next(windows_rows, None)
        linux_header = next(linux_rows, None)
        if windows_header is None:
            raise ValueError(f"CSV is empty: {windows_csv}")
        if linux_header is None:
            raise ValueError(f"CSV is empty: {linux_csv}")
        header_order_match = windows_header == linux_header
        header_inventory_match = (
            len(set(windows_header)) == len(windows_header)
            and len(set(linux_header)) == len(linux_header)
            and set(windows_header) == set(linux_header)
        )
        linux_column_by_name = {
            name: index for index, name in enumerate(linux_header)
        }
        column_alignment = (
            [linux_column_by_name[name] for name in windows_header]
            if header_inventory_match
            else []
        )
        shape_match = header_inventory_match
        for row_index, (windows_row, linux_row) in enumerate(
            zip_longest(windows_rows, linux_rows), start=2
        ):
            if windows_row is not None:
                windows_row_count += 1
            if linux_row is not None:
                linux_row_count += 1
            if windows_row is None or linux_row is None:
                shape_match = False
                continue
            if (
                len(windows_row) != len(windows_header)
                or len(linux_row) != len(linux_header)
            ):
                shape_match = False
                continue
            if not header_inventory_match:
                continue
            for column_index, windows_value in enumerate(windows_row):
                linux_value = linux_row[column_alignment[column_index]]
                if windows_value == "" or linux_value == "":
                    equal = windows_value == linux_value
                    absolute = relative = math.inf if not equal else 0.0
                else:
                    try:
                        expected = float(windows_value)
                        actual = float(linux_value)
                    except ValueError:
                        equal = windows_value == linux_value
                        absolute = relative = math.inf if not equal else 0.0
                    else:
                        numeric_cell_count += 1
                        if math.isnan(expected) or math.isnan(actual):
                            equal = math.isnan(expected) and math.isnan(actual)
                            absolute = relative = math.inf if not equal else 0.0
                        else:
                            absolute = abs(actual - expected)
                            scale = max(abs(expected), abs(actual))
                            relative = absolute / scale if scale else 0.0
                            equal = math.isclose(
                                actual,
                                expected,
                                abs_tol=absolute_tolerance,
                                rel_tol=relative_tolerance,
                            )
                if math.isfinite(absolute):
                    max_absolute_difference = max(
                        max_absolute_difference, absolute
                    )
                if math.isfinite(relative):
                    max_relative_difference = max(
                        max_relative_difference, relative
                    )
                if not equal:
                    mismatch_count += 1
                    if len(mismatch_examples) < 10:
                        mismatch_examples.append(
                            {
                                "row": row_index,
                                "column": windows_header[column_index],
                                "windows_value": windows_value,
                                "linux_value": linux_value,
                                "absolute_difference": absolute,
                                "relative_difference": relative,
                            }
                        )
        if windows_row_count != linux_row_count:
            shape_match = False
    return {
        "accepted": (
            header_inventory_match and shape_match and mismatch_count == 0
        ),
        "headers_match": header_inventory_match,
        "header_inventory_match": header_inventory_match,
        "header_order_match": header_order_match,
        "shape_match": shape_match,
        "windows_row_count": windows_row_count,
        "linux_row_count": linux_row_count,
        "column_count": len(windows_header),
        "numeric_cell_count": numeric_cell_count,
        "mismatch_count": mismatch_count,
        "maximum_absolute_difference": max_absolute_difference,
        "maximum_relative_difference": max_relative_difference,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "mismatch_examples": mismatch_examples,
    }


def _run_simulation(
    *,
    record: Mapping[str, Any],
    runner_path: Path,
    output_root: Path,
    rscript: Path,
    r_library: Path,
    dotnet_root: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    protocol_key = str(record["protocol_key"])
    sequence = int(record["sequence"])
    simulation_root = output_root / "simulations" / protocol_key / f"s{sequence:04d}"
    simulation_root.mkdir(parents=True, exist_ok=True)
    linux_csv = simulation_root / "linux-results.csv"
    runtime_manifest = simulation_root / "runtime_manifest.json"
    stdout_path = simulation_root / "stdout.log"
    stderr_path = simulation_root / "stderr.log"
    result_path = simulation_root / "result.json"
    previous: dict[str, Any] | None = None
    if result_path.is_file():
        try:
            loaded = json.loads(result_path.read_text(encoding="utf-8-sig"))
            previous = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            previous = None
    if previous is not None:
        comparison = previous.get("comparison") or {}
        if (
            previous.get("status") in {"passed", "executed_negative"}
            and previous.get("windows_pkml_sha256") == record["pkml_sha256"]
            and previous.get("windows_csv_sha256")
            == record["windows_csv_sha256"]
            and linux_csv.is_file()
            and previous.get("linux_csv_sha256") == sha256_file(linux_csv)
            and runtime_manifest.is_file()
            and previous.get("runtime_manifest_sha256")
            == sha256_file(runtime_manifest)
            and comparison.get("absolute_tolerance") == absolute_tolerance
            and comparison.get("relative_tolerance") == relative_tolerance
        ):
            previous["resumed"] = True
            return previous

    stale_paths = (
        linux_csv,
        runtime_manifest,
        stdout_path,
        stderr_path,
        result_path,
        linux_csv.with_suffix(linux_csv.suffix + ".tmp"),
        runtime_manifest.with_suffix(runtime_manifest.suffix + ".tmp"),
        stdout_path.with_suffix(stdout_path.suffix + ".tmp"),
        stderr_path.with_suffix(stderr_path.suffix + ".tmp"),
        result_path.with_suffix(result_path.suffix + ".tmp"),
    )
    if result_path.is_file():
        attempts_root = simulation_root / "attempts"
        attempts_root.mkdir(exist_ok=True)
        attempt_number = 1
        while (attempts_root / f"attempt_{attempt_number:04d}").exists():
            attempt_number += 1
        attempt_root = attempts_root / f"attempt_{attempt_number:04d}"
        attempt_root.mkdir()
        archived = []
        for stale in stale_paths:
            if stale.is_file():
                destination = attempt_root / stale.name
                shutil.move(str(stale), destination)
                archived.append(destination.name)
        _atomic_json(
            attempt_root / "archive_manifest.json",
            {
                "archived_at": datetime.now(UTC).isoformat(),
                "archived_files": sorted(archived),
                "previous_status": previous.get("status") if previous else None,
                "previous_reason_codes": (
                    previous.get("reason_codes") if previous else []
                ),
                "retry_timeout_seconds": timeout_seconds,
            },
        )
    for stale in stale_paths:
        stale.unlink(missing_ok=True)
    started_at = datetime.now(UTC)
    environment = os.environ.copy()
    environment.update(
        {
            "R_LIBS_USER": str(r_library),
            "DOTNET_ROOT": str(dotnet_root),
            "PATH": f"{dotnet_root}:{environment.get('PATH', '')}",
        }
    )
    command = [
        str(rscript),
        str(runner_path),
        str(record["pkml_path"]),
        str(record["windows_csv_path"]),
        str(linux_csv),
        str(runtime_manifest),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout_seconds,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        return_code = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        timed_out = True
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    result: dict[str, Any] = {
        "linux_repeat_version": LINUX_REPEAT_VERSION,
        "protocol_key": protocol_key,
        "protocol_id": record["protocol_id"],
        "sequence": sequence,
        "simulation_name": record["simulation_name"],
        "status": "runner_failed",
        "return_code": return_code,
        "timeout_seconds": timeout_seconds,
        "duration_seconds": (
            datetime.now(UTC) - started_at
        ).total_seconds(),
        "windows_pkml_sha256": record["pkml_sha256"],
        "windows_csv_sha256": record["windows_csv_sha256"],
        "windows_csv_role": record["windows_csv_role"],
        "windows_csv_selection_rule": record["windows_csv_selection_rule"],
        "auxiliary_csv_count": record["auxiliary_csv_count"],
        "linux_csv_sha256": sha256_file(linux_csv) if linux_csv.is_file() else None,
        "runtime_manifest_sha256": (
            sha256_file(runtime_manifest) if runtime_manifest.is_file() else None
        ),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "reason_codes": [],
    }
    if timed_out:
        result["reason_codes"].append("LINUX_OSPSUITE_R_EXECUTION_TIMEOUT")
        _atomic_json(result_path, result)
        return result
    if return_code != 0:
        result["reason_codes"].append("LINUX_OSPSUITE_R_EXECUTION_FAILED")
        _atomic_json(result_path, result)
        return result
    if not linux_csv.is_file() or not runtime_manifest.is_file():
        result["reason_codes"].append("LINUX_OUTPUT_INCOMPLETE")
        _atomic_json(result_path, result)
        return result
    try:
        comparison = compare_result_csvs(
            windows_csv=Path(str(record["windows_csv_path"])),
            linux_csv=linux_csv,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    except (OSError, ValueError) as exc:
        result["reason_codes"].append("CROSS_PLATFORM_COMPARISON_FAILED")
        result["failure_type"] = type(exc).__name__
        result["failure_detail"] = str(exc)
        _atomic_json(result_path, result)
        return result
    result["comparison"] = comparison
    if comparison["accepted"]:
        result["status"] = "passed"
    else:
        result["status"] = "executed_negative"
        result["reason_codes"].append("WINDOWS_LINUX_NUMERIC_MISMATCH")
    _atomic_json(result_path, result)
    return result


def run_linux_ospsuite_repeat(
    *,
    batch_directory: Path,
    windows_results_root: Path,
    output_directory: Path,
    rscript: Path,
    r_library: Path,
    dotnet_root: Path,
    workers: int = 8,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    batch_directory = batch_directory.resolve()
    windows_results_root = windows_results_root.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if workers < 1:
        raise ValueError("workers must be positive.")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive.")
    for required in (rscript, r_library, dotnet_root):
        if not required.exists():
            raise FileNotFoundError(required)
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        jobs_path = batch_directory / "jobs.jsonl"
        batch_manifest_path = batch_directory / "batch_manifest.json"
        batch_manifest = json.loads(
            batch_manifest_path.read_text(encoding="utf-8-sig")
        )
        if sha256_file(jobs_path) != str(batch_manifest["jobs_sha256"]):
            raise ValueError("Windows batch jobs checksum drifted.")
        jobs = _read_jsonl(jobs_path)
        if len(jobs) != int(batch_manifest["job_count"]):
            raise ValueError("Windows batch job count does not match its manifest.")
        runner_path = temporary / "run_simulation_ospsuite_r.R"
        if runner_path.is_file():
            if runner_path.read_text(encoding="utf-8") != R_RUNNER:
                raise ValueError("Linux OSPSuite-R runner changed during resume.")
        else:
            runner_path.write_text(R_RUNNER, encoding="utf-8")
        records: list[dict[str, Any]] = []
        for job in jobs:
            key = str(job["protocol_key"])
            result_root = windows_results_root / key
            manifest_path = result_root / "windows_run_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if str(manifest.get("status")) != "passed":
                raise ValueError(f"Windows protocol did not pass: {key}")
            runs = manifest.get("runs") or []
            if len(runs) != 2 or any(run.get("status") != "passed" for run in runs):
                raise ValueError(f"Windows double-run is incomplete: {key}")
            for simulation in runs[0].get("simulations") or []:
                if simulation.get("status") != "passed":
                    raise ValueError(
                        f"Windows simulation did not pass: {key}/"
                        f"{simulation.get('sequence')}"
                    )
                pkml = _artifact_path(
                    result_root=result_root,
                    simulation=simulation,
                    extension=".pkml",
                )
                windows_csv = _artifact_path(
                    result_root=result_root,
                    simulation=simulation,
                    extension=".csv",
                    relative_path_suffix="-Results.csv",
                )
                csv_artifacts = [
                    row
                    for row in simulation.get("artifacts") or []
                    if str(row.get("extension") or "").casefold() == ".csv"
                ]
                records.append(
                    {
                        "protocol_key": key,
                        "protocol_id": job["protocol_id"],
                        "sequence": int(simulation["sequence"]),
                        "simulation_name": str(simulation["name"]),
                        "pkml_path": pkml,
                        "pkml_sha256": sha256_file(pkml),
                        "windows_csv_path": windows_csv,
                        "windows_csv_sha256": sha256_file(windows_csv),
                        "windows_csv_role": "simulation_results",
                        "windows_csv_selection_rule": "unique_-Results.csv",
                        "auxiliary_csv_count": len(csv_artifacts) - 1,
                    }
                )
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_simulation,
                    record=record,
                    runner_path=runner_path,
                    output_root=temporary,
                    rscript=rscript.resolve(),
                    r_library=r_library.resolve(),
                    dotnet_root=dotnet_root.resolve(),
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                    timeout_seconds=timeout_seconds,
                )
                for record in records
            ]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(
            key=lambda row: (str(row["protocol_key"]), int(row["sequence"]))
        )
        protocol_rows = []
        for job in jobs:
            key = str(job["protocol_key"])
            rows = [row for row in results if row["protocol_key"] == key]
            protocol_rows.append(
                {
                    "protocol_key": key,
                    "protocol_id": job["protocol_id"],
                    "simulation_count": len(rows),
                    "passed_count": sum(row["status"] == "passed" for row in rows),
                    "executed_negative_count": sum(
                        row["status"] == "executed_negative" for row in rows
                    ),
                    "runner_failed_count": sum(
                        row["status"] == "runner_failed" for row in rows
                    ),
                    "status": (
                        "passed"
                        if rows and all(row["status"] == "passed" for row in rows)
                        else "runner_failed"
                        if any(row["status"] == "runner_failed" for row in rows)
                        else "executed_negative"
                    ),
                }
            )
        protocol_rows.sort(key=lambda row: str(row["protocol_key"]))
        failures = [row for row in results if row["status"] != "passed"]
        checksums = {
            "simulation_results.jsonl": _atomic_jsonl(
                temporary / "simulation_results.jsonl", results
            ),
            "protocol_results.jsonl": _atomic_jsonl(
                temporary / "protocol_results.jsonl", protocol_rows
            ),
            "failures.jsonl": _atomic_jsonl(
                temporary / "failures.jsonl", failures
            ),
            "run_simulation_ospsuite_r.R": sha256_file(runner_path),
        }
        counts = {
            "protocol_count": len(protocol_rows),
            "protocol_passed_count": sum(
                row["status"] == "passed" for row in protocol_rows
            ),
            "simulation_count": len(results),
            "simulation_passed_count": sum(
                row["status"] == "passed" for row in results
            ),
            "executed_negative_count": sum(
                row["status"] == "executed_negative" for row in results
            ),
            "runner_failed_count": sum(
                row["status"] == "runner_failed" for row in results
            ),
        }
        execution_complete = (
            counts["protocol_count"] == int(batch_manifest["job_count"])
            and counts["simulation_count"] == len(records)
            and counts["runner_failed_count"] == 0
        )
        cross_platform_all_passed = (
            execution_complete
            and counts["protocol_passed_count"] == counts["protocol_count"]
            and counts["executed_negative_count"] == 0
        )
        summary = {
            "linux_repeat_version": LINUX_REPEAT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "batch_directory": str(batch_directory),
            "batch_manifest_sha256": sha256_file(batch_manifest_path),
            "jobs_sha256": sha256_file(jobs_path),
            "windows_results_root": str(windows_results_root),
            "rscript": str(rscript.resolve()),
            "rscript_sha256": sha256_file(rscript.resolve()),
            "r_library": str(r_library.resolve()),
            "dotnet_root": str(dotnet_root.resolve()),
            "workers": workers,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "timeout_seconds": timeout_seconds,
            "counts": counts,
            "complete": execution_complete,
            "execution_complete": execution_complete,
            "cross_platform_all_passed": cross_platform_all_passed,
            "output_checksums": checksums,
        }
        (temporary / "orchestrator_failure.json").unlink(missing_ok=True)
        _atomic_json(temporary / "summary.json", summary)
        os.replace(temporary, output_directory)
        return summary
    except Exception as exc:
        try:
            _atomic_json(
                temporary / "orchestrator_failure.json",
                {
                    "linux_repeat_version": LINUX_REPEAT_VERSION,
                    "failed_at": datetime.now(UTC).isoformat(),
                    "failure_type": type(exc).__name__,
                    "failure_detail": str(exc),
                },
            )
        except Exception:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat Windows-exported PK-Sim simulations with Linux OSPSuite-R."
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--windows-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--dotnet-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--absolute-tolerance", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE
    )
    parser.add_argument(
        "--relative-tolerance", type=float, default=DEFAULT_RELATIVE_TOLERANCE
    )
    args = parser.parse_args()
    result = run_linux_ospsuite_repeat(
        batch_directory=args.batch,
        windows_results_root=args.windows_results,
        output_directory=args.output,
        rscript=args.rscript,
        r_library=args.r_library,
        dotnet_root=args.dotnet_root,
        workers=args.workers,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
