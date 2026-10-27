from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pbpk_kb.model_assets.execution_plan import (
    write_execution_plan,
)
from pbpk_kb.model_assets.models import (
    OSPExecutionPlan,
    OSPExecutionProjectSpec,
    OSPExecutionSimulationSpec,
    OSPRuntimeLock,
    RuntimeDownloadLock,
)
from pbpk_kb.model_runtime.jobs import (
    build_repeatability_control_plan,
)
from pbpk_kb.model_runtime.result_parser import compare_csv_files


def _download(name: str, character: str) -> RuntimeDownloadLock:
    return RuntimeDownloadLock(
        name=name,
        url=f"https://example.test/{name}",
        sha256=character * 64,
    )


def _source_plan() -> OSPExecutionPlan:
    runtime = OSPRuntimeLock(
        suite_release_name="OSP Suite 12.3.0",
        suite_version="12.3.0",
        suite_installer=_download("suite.exe", "1"),
        pksim_version="12.3.0",
        pksim_portable=_download("pksim.zip", "2"),
        pksim_cli_relative_path="PKSim.CLI.exe",
        ospsuite_r_version="12.3.1",
        ospsuite_r_linux_binary=_download("ospsuite", "3"),
    )
    simulation = OSPExecutionSimulationSpec(
        simulation_id="simulation-1",
        name="Simulation 1",
        simulation_mode="Individual",
    )
    return OSPExecutionPlan(
        job_id="osp-exec-" + "a" * 32,
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
        source_id="osp_pbpk_model_library",
        repository_url="https://example.test/repository.git",
        repository_release_tag="v12.3.1",
        repository_commit="b" * 40,
        inventory_run_id="inventory",
        inventory_summary_sha256="c" * 64,
        runtime=runtime,
        export_flags=["-k", "-x", "-c", "-r"],
        command_timeout_seconds=3600,
        required_export_extensions=[".pkml", ".xml", ".csv"],
        projects=[
            OSPExecutionProjectSpec(
                project_key="d" * 32,
                model_family="Large",
                relative_path="Large/Large.pksim5",
                project_sha256="e" * 64,
                project_name="Large",
                project_version=80,
                simulations=[simulation, simulation.model_copy()],
            ),
            OSPExecutionProjectSpec(
                project_key="f" * 32,
                model_family="Small",
                relative_path="Small/Small.pksim5",
                project_sha256="0" * 64,
                project_name="Small",
                project_version=80,
                simulations=[simulation],
            ),
        ],
    )


def test_repeatability_control_selects_smallest_project(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source_plan, _ = write_execution_plan(source, _source_plan())
    output = tmp_path / "control"

    plan_path, protocol_path, protocol = (
        build_repeatability_control_plan(
            source_plan_path=source_plan,
            output_directory=output,
        )
    )

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [item["relative_path"] for item in payload["projects"]] == [
        "Small/Small.pksim5"
    ]
    assert protocol.control_project_relative_path == "Small/Small.pksim5"
    assert protocol.simulation_count == 1
    assert protocol.ranking_label_permitted is False
    assert protocol_path.is_file()


def test_csv_comparison_enforces_tolerance_and_exact_time_axis(
    tmp_path: Path,
) -> None:
    run_a = tmp_path / "a.csv"
    run_b = tmp_path / "b.csv"
    run_a.write_text(
        '"IndividualId","Time [min]","C [mg/l]"\n0,0,1\n0,1,2\n',
        encoding="utf-8",
    )
    run_b.write_text(
        '"IndividualId","Time [min]","C [mg/l]"\n'
        "0,0,1.000000000001\n0,1,2.000000000001\n",
        encoding="utf-8",
    )

    accepted = compare_csv_files(
        run_a_path=run_a,
        run_b_path=run_b,
        simulation_id="simulation-1",
        csv_ordinal=1,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
    )
    assert accepted.accepted
    assert accepted.numeric_cell_count == 4

    run_b.write_text(
        '"IndividualId","Time [min]","C [mg/l]"\n'
        "0,0,1.000000000001\n0,1.1,2.000000000001\n",
        encoding="utf-8",
    )
    rejected = compare_csv_files(
        run_a_path=run_a,
        run_b_path=run_b,
        simulation_id="simulation-1",
        csv_ordinal=1,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
    )
    assert not rejected.accepted
    assert {
        item["reason_code"] for item in rejected.mismatches
    } == {"TIME_AXIS_MISMATCH"}
