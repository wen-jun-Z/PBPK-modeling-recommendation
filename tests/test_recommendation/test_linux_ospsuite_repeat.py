from __future__ import annotations

import json
from pathlib import Path

import pytest

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.recommendation.model_matching.linux_ospsuite_repeat import (
    compare_result_csvs,
    run_linux_ospsuite_repeat,
)
from pbpk_kb.recommendation.model_matching.execution_benchmark import (
    _read_linux_repeat_index,
)


def _write_batch(root: Path) -> tuple[Path, Path]:
    batch = root / "batch"
    results = root / "windows"
    batch.mkdir()
    protocol_key = "a" * 24
    job = {
        "protocol_key": protocol_key,
        "protocol_id": "protocol:test",
    }
    jobs = batch / "jobs.jsonl"
    jobs.write_text(json.dumps(job) + "\n", encoding="utf-8")
    (batch / "batch_manifest.json").write_text(
        json.dumps({"job_count": 1, "jobs_sha256": sha256_file(jobs)}),
        encoding="utf-8",
    )
    simulation_root = results / protocol_key / "runs" / "run_1" / "s0001"
    simulation_root.mkdir(parents=True)
    pkml = simulation_root / "simulation.pkml"
    windows_csv = simulation_root / "simulation-Results.csv"
    pkml.write_text("locked simulation\n", encoding="utf-8")
    windows_csv.write_text(
        '"IndividualId","Time [min]","Organism|Plasma|Drug [mg/L]"\n'
        "0,0,0\n0,1,1.25\n",
        encoding="utf-8",
    )
    pk_analysis_csv = simulation_root / "simulation-PK-Analyses.csv"
    population_csv = simulation_root / "simulation-Population.csv"
    pk_analysis_csv.write_text("Metric,Value\nAUC,1\n", encoding="utf-8")
    population_csv.write_text("IndividualId,Age\n0,30\n", encoding="utf-8")
    artifacts = [
        {
            "relative_path": str(pkml.relative_to(results / protocol_key)),
            "extension": ".pkml",
            "sha256": sha256_file(pkml),
        },
        {
            "relative_path": str(
                windows_csv.relative_to(results / protocol_key)
            ),
            "extension": ".csv",
            "sha256": sha256_file(windows_csv),
        },
        {
            "relative_path": str(
                pk_analysis_csv.relative_to(results / protocol_key)
            ),
            "extension": ".csv",
            "sha256": sha256_file(pk_analysis_csv),
        },
        {
            "relative_path": str(
                population_csv.relative_to(results / protocol_key)
            ),
            "extension": ".csv",
            "sha256": sha256_file(population_csv),
        },
    ]
    simulation = {
        "sequence": 1,
        "name": "verification scenario",
        "status": "passed",
        "artifacts": artifacts,
    }
    manifest = {
        "status": "passed",
        "runs": [
            {"run_id": "run_1", "status": "passed", "simulations": [simulation]},
            {"run_id": "run_2", "status": "passed", "simulations": [simulation]},
        ],
    }
    (results / protocol_key / "windows_run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return batch, results


def test_csv_comparison_accepts_locked_rounding_tolerance(tmp_path: Path) -> None:
    windows = tmp_path / "windows.csv"
    linux = tmp_path / "linux.csv"
    header = '"IndividualId","Time [min]","A [mg/L]"\n'
    windows.write_text(header + "0,0,0.6376957\n", encoding="utf-8")
    linux.write_text(header + "0,0,0.63769567\n", encoding="utf-8")

    result = compare_result_csvs(windows_csv=windows, linux_csv=linux)

    assert result["accepted"] is True
    assert result["mismatch_count"] == 0


def test_csv_comparison_detects_streamed_shape_mismatch(tmp_path: Path) -> None:
    windows = tmp_path / "windows.csv"
    linux = tmp_path / "linux.csv"
    header = '"IndividualId","Time [min]","A [mg/L]"\n'
    windows.write_text(header + "0,0,1\n0,1,2\n", encoding="utf-8")
    linux.write_text(header + "0,0,1\n", encoding="utf-8")

    result = compare_result_csvs(windows_csv=windows, linux_csv=linux)

    assert result["accepted"] is False
    assert result["headers_match"] is True
    assert result["shape_match"] is False
    assert result["windows_row_count"] == 2
    assert result["linux_row_count"] == 1


def test_csv_comparison_aligns_unique_output_columns_by_path(
    tmp_path: Path,
) -> None:
    windows = tmp_path / "windows.csv"
    linux = tmp_path / "linux.csv"
    windows.write_text(
        '"IndividualId","Time [min]","A [mg/L]","B [mg/L]"\n'
        "0,0,1,2\n",
        encoding="utf-8",
    )
    linux.write_text(
        '"IndividualId","Time [min]","B [mg/L]","A [mg/L]"\n'
        "0,0,2,1\n",
        encoding="utf-8",
    )

    result = compare_result_csvs(windows_csv=windows, linux_csv=linux)

    assert result["accepted"] is True
    assert result["header_inventory_match"] is True
    assert result["header_order_match"] is False
    assert result["mismatch_count"] == 0


@pytest.mark.parametrize("stale_state", ["orphan", "runner_failed"])
def test_linux_repeat_resumes_incomplete_simulation_directory(
    tmp_path: Path, stale_state: str
) -> None:
    batch, windows = _write_batch(tmp_path)
    rscript = tmp_path / "Rscript"
    rscript.write_text(
        "#!/bin/sh\n"
        'cp "$3" "$4"\n'
        "printf '%s\\n' '{\"ospsuite_version\":\"test\"}' > \"$5\"\n",
        encoding="utf-8",
    )
    rscript.chmod(0o755)
    r_library = tmp_path / "r-library"
    dotnet_root = tmp_path / "dotnet"
    r_library.mkdir()
    dotnet_root.mkdir()

    output = tmp_path / "linux-repeat"
    stale_root = (
        output.with_name(output.name + ".tmp")
        / "simulations"
        / ("a" * 24)
        / "s0001"
    )
    stale_root.mkdir(parents=True)
    (stale_root / "stdout.log").write_text("incomplete attempt\n", encoding="utf-8")
    if stale_state == "orphan":
        (stale_root / "result.json.tmp").write_text("{", encoding="utf-8")
    else:
        (stale_root / "result.json").write_text(
            json.dumps(
                {
                    "status": "runner_failed",
                    "reason_codes": ["LINUX_OSPSUITE_R_EXECUTION_TIMEOUT"],
                }
            ),
            encoding="utf-8",
        )
    summary = run_linux_ospsuite_repeat(
        batch_directory=batch,
        windows_results_root=windows,
        output_directory=output,
        rscript=rscript,
        r_library=r_library,
        dotnet_root=dotnet_root,
        workers=1,
    )

    assert summary["complete"] is True
    assert summary["execution_complete"] is True
    assert summary["cross_platform_all_passed"] is True
    assert summary["counts"] == {
        "protocol_count": 1,
        "protocol_passed_count": 1,
        "simulation_count": 1,
        "simulation_passed_count": 1,
        "executed_negative_count": 0,
        "runner_failed_count": 0,
    }
    protocol = json.loads(
        (output / "protocol_results.jsonl").read_text().splitlines()[0]
    )
    assert protocol["status"] == "passed"
    assert (
        sha256_file(output / "protocol_results.jsonl")
        == summary["output_checksums"]["protocol_results.jsonl"]
    )
    assert not (output / "simulations" / ("a" * 24) / "s0001" / "result.json.tmp").exists()
    if stale_state == "runner_failed":
        archived = (
            output
            / "simulations"
            / ("a" * 24)
            / "s0001"
            / "attempts"
            / "attempt_0001"
        )
        assert json.loads((archived / "result.json").read_text())["status"] == "runner_failed"
        assert json.loads((archived / "archive_manifest.json").read_text())[
            "previous_reason_codes"
        ] == ["LINUX_OSPSUITE_R_EXECUTION_TIMEOUT"]


def test_linux_repeat_keeps_numeric_negative_as_completed_benchmark_input(
    tmp_path: Path,
) -> None:
    batch, windows = _write_batch(tmp_path)
    rscript = tmp_path / "Rscript"
    rscript.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' "
        "'\"IndividualId\",\"Time [min]\",\"Organism|Plasma|Drug [mg/L]\"' "
        "'0,0,0' '0,1,2' > \"$4\"\n"
        "printf '%s\\n' '{\"ospsuite_version\":\"test\"}' > \"$5\"\n",
        encoding="utf-8",
    )
    rscript.chmod(0o755)
    r_library = tmp_path / "r-library"
    dotnet_root = tmp_path / "dotnet"
    r_library.mkdir()
    dotnet_root.mkdir()

    output = tmp_path / "linux-negative"
    summary = run_linux_ospsuite_repeat(
        batch_directory=batch,
        windows_results_root=windows,
        output_directory=output,
        rscript=rscript,
        r_library=r_library,
        dotnet_root=dotnet_root,
        workers=1,
    )
    _, protocols = _read_linux_repeat_index(output)

    assert summary["execution_complete"] is True
    assert summary["complete"] is True
    assert summary["cross_platform_all_passed"] is False
    assert summary["counts"]["executed_negative_count"] == 1
    assert next(iter(protocols.values()))["status"] == "executed_negative"
