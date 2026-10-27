from __future__ import annotations

import json
import gzip
import subprocess
from pathlib import Path

import pytest

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.licensed_model_registration import (
    register_licensed_model,
)
from pbpk_kb.model_assets.licensed_snapshot_reuse import (
    ExecutionArtifact,
    FrozenObservedSeries,
    FrozenSimulation,
    LicensedSnapshotReuseProtocol,
    ScientificCriteria,
    _concentration_in_mol_per_litre,
    _dataset_metrics,
    _geometric_metrics,
    _interpolate,
    _time_in_hours,
    audit_snapshot_reuse_execution,
    build_snapshot_reuse_protocol,
    compare_roundtrip_snapshots,
)


def _run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _snapshot() -> dict:
    observed_name = "Verification plasma"
    output_path = "verification|Organism|Blood|Drug|Plasma"
    simulations = [
        {
            "Name": "development",
            "Individual": "Adult A",
            "ObservedData": [],
            "OutputMappings": [],
        },
        {
            "Name": "verification",
            "Individual": "Adult B",
            "ObservedData": [observed_name],
            "Parameters": [
                {
                    "Path": (
                        "Applications|po 10 mg|Application_1|"
                        "ProtocolSchemaItem|Volume of water/body weight"
                    ),
                    "Value": 3.0,
                    "Unit": "ml/kg",
                }
            ],
            "OutputMappings": [
                {
                    "Path": output_path,
                    "ObservedData": observed_name,
                    "Scaling": "Log",
                }
            ],
        },
        {
            "Name": "execution only",
            "Individual": "Adult C",
            "ObservedData": [],
            "OutputMappings": [],
        },
    ]
    return {
        "Version": 79,
        "ExpressionProfiles": [],
        "Individuals": [{"Name": f"Adult {letter}"} for letter in "ABC"],
        "Compounds": [{"Name": "Drug"}],
        "Formulations": [{"Name": "Tablet"}],
        "Protocols": [{"Name": "po 10 mg"}],
        "Simulations": simulations,
        "ObservedData": [
            {
                "Name": observed_name,
                "ExtendedProperties": [
                    {
                        "Name": "Reference",
                        "Value": "https://example.test/study",
                    },
                    {"Name": "Description", "Value": ""},
                ],
                "BaseGrid": {
                    "Name": "Time",
                    "Values": [1.0, 2.0],
                    "Unit": "h",
                },
                "Columns": [
                    {
                        "Name": "Avg",
                        "Values": [10.0, 5.0],
                        "Dimension": "Concentration (mass)",
                        "Unit": "ng/ml",
                        "DataInfo": {
                            "Origin": "Observation",
                            "MolWeight": 500.0,
                        },
                    }
                ],
            }
        ],
        "SimulationClassifications": [{"Name": "po"}],
    }


def _prepare_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repository)], check=True)
    _run_git(repository, "remote", "add", "origin", "https://example.test/model.git")
    (repository / "LICENSE").write_text(
        "GNU GENERAL PUBLIC LICENSE Version 2, June 1991\n",
        encoding="utf-8",
    )
    snapshot_path = repository / "Model.json"
    snapshot_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    plan_path = repository / "Evaluation" / "Input" / "evaluation_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps(
            {
                "Plots": {
                    "AllPlots": [
                        {"Simulation": "development", "SectionId": 331},
                        {"Simulation": "verification", "SectionId": 332},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    _run_git(repository, "add", ".")
    _run_git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-m",
        "fixture",
    )
    commit = _run_git(repository, "rev-parse", "HEAD")
    _run_git(repository, "tag", "v1.0")
    archive = tmp_path / "Model-v1.0.tar.gz"
    archive.write_bytes(b"locked source archive")
    config = {
        "schema_version": "1.0.0",
        "source": {
            "source_id": "licensed_model",
            "model_id": "drug",
            "model_name": "Drug",
            "repository_url": "https://example.test/model.git",
            "release_tag": "v1.0",
            "commit_sha": commit,
            "license_spdx_id": "GPL-2.0-only",
            "license_relative_path": "LICENSE",
            "license_sha256": sha256_file(repository / "LICENSE"),
            "snapshot_relative_path": "Model.json",
            "snapshot_sha256": sha256_file(snapshot_path),
            "snapshot_schema_version": 79,
            "evaluation_plan_relative_path": "Evaluation/Input/evaluation_plan.json",
            "evaluation_plan_sha256": sha256_file(plan_path),
            "source_archive_name": archive.name,
            "source_archive_sha256": sha256_file(archive),
        },
        "expected_counts": {
            "compounds": 1,
            "expression_profiles": 0,
            "formulations": 1,
            "individuals": 3,
            "protocols": 1,
            "simulations": 3,
            "observed_data": 1,
            "simulation_classifications": 1,
            "model_building": 1,
            "model_verification": 1,
            "execution_only": 1,
        },
        "partition": {
            "all_plots_path": "Plots.AllPlots",
            "development_section_id": 331,
            "verification_section_id": 332,
        },
        "runtime": {
            "pksim_version": "12.3.173",
            "pksim_portable_sha256": "a" * 64,
            "cli_relative_path": "PKSim.CLI.exe",
            "command_timeout_seconds": 3600,
            "required_export_extensions": [".pkml", ".xml", ".csv"],
        },
        "criteria": {
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-10,
            "pooled_gmfe_max": 2.0,
            "pooled_bias_min": 0.5,
            "pooled_bias_max": 2.0,
            "pooled_two_fold_fraction_min": 0.8,
            "dataset_gmfe_max": 3.0,
            "dataset_bias_min": 1 / 3,
            "dataset_bias_max": 3.0,
            "endpoint_ratio_min": 0.5,
            "endpoint_ratio_max": 2.0,
            "endpoint_scenario_fraction_min": 0.8,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return repository, archive, config_path


def test_protocol_builder_accepts_git_worktree(tmp_path: Path) -> None:
    repository, archive, config = _prepare_source(tmp_path)
    checkout = tmp_path / "worktree"
    _run_git(repository, "worktree", "add", "--detach", str(checkout), "HEAD")

    protocol = build_snapshot_reuse_protocol(
        config_path=config,
        repository_root=checkout,
        source_archive=archive,
        output_directory=tmp_path / "protocol-worktree",
    )

    assert protocol.source.commit_sha == _run_git(checkout, "rev-parse", "HEAD")


def test_build_protocol_uses_official_partition_and_freezes_observation(
    tmp_path: Path,
) -> None:
    repository, archive, config = _prepare_source(tmp_path)
    protocol = build_snapshot_reuse_protocol(
        config_path=config,
        repository_root=repository,
        source_archive=archive,
        output_directory=tmp_path / "protocol",
    )

    assert [item.role for item in protocol.simulations] == [
        "model_building",
        "model_verification",
        "execution_only",
    ]
    frozen = protocol.simulations[1].evaluation_series
    assert frozen is not None
    assert frozen.dataset_name == "Verification plasma"
    assert frozen.time_values == [1.0, 2.0]
    assert frozen.observed_values == [10.0, 5.0]
    assert protocol.protocol_id.startswith("snapshot-reuse-")


def test_build_protocol_supports_section_reference_partition(tmp_path: Path) -> None:
    repository, archive, config_path = _prepare_source(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["partition"] = {
        "all_plots_path": "Plots.AllPlots",
        "development_section_references": ["model-building"],
        "verification_section_references": ["model-verification"],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan_path = repository / "Evaluation" / "Input" / "evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for plot in plan["Plots"]["AllPlots"]:
        plot["SectionReference"] = (
            "model-building"
            if plot.pop("SectionId") == 331
            else "model-verification"
        )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    _run_git(repository, "add", ".")
    _run_git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "--amend",
        "--no-edit",
    )
    commit = _run_git(repository, "rev-parse", "HEAD")
    _run_git(repository, "tag", "-f", "v1.0")
    config["source"]["commit_sha"] = commit
    config["source"]["evaluation_plan_sha256"] = sha256_file(plan_path)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    protocol = build_snapshot_reuse_protocol(
        config_path=config_path,
        repository_root=repository,
        source_archive=archive,
        output_directory=tmp_path / "protocol-reference",
    )

    assert [item.role for item in protocol.simulations] == [
        "model_building",
        "model_verification",
        "execution_only",
    ]


def _artifact(root: Path, simulation_root: Path, path: Path) -> dict:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "simulation_relative_path": path.relative_to(simulation_root).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "extension": path.suffix,
    }


def _build_windows_result(
    root: Path,
    protocol_path: Path,
    protocol: LicensedSnapshotReuseProtocol,
    source_snapshot: Path,
) -> None:
    (root / "input").mkdir(parents=True)
    (root / "project").mkdir()
    (root / "roundtrip").mkdir()
    input_snapshot = root / "input" / "Model.json"
    roundtrip = root / "roundtrip" / "Model.json"
    project = root / "project" / "Model.pksim5"
    input_snapshot.write_bytes(source_snapshot.read_bytes())
    roundtrip.write_bytes(source_snapshot.read_bytes())
    project.write_bytes(b"project")
    runs = []
    for run_id in ("run_1", "run_2"):
        simulation_results = []
        for simulation in protocol.simulations:
            simulation_root = root / "runs" / run_id / f"s{simulation.sequence:04d}"
            simulation_root.mkdir(parents=True)
            csv_path = simulation_root / "results.csv"
            output_path = (
                simulation.evaluation_series.output_path
                if simulation.evaluation_series is not None
                else f"{simulation.name}|Organism|Blood|Drug|Plasma"
            )
            csv_path.write_text(
                f"Time [h],{output_path} [ng/ml]\n1,10\n2,5\n",
                encoding="utf-8",
            )
            pkml = simulation_root / "simulation.pkml"
            xml = simulation_root / "simulation.xml"
            pkml.write_text(
                f'<Simulation name="{simulation.name}">'
                '<MoleculeBuilder id="drug" name="Drug" '
                'quantityType="Drug"/></Simulation>',
                encoding="utf-8",
            )
            xml.write_text("<Simulation/>", encoding="utf-8")
            stdout = simulation_root / "stdout.log"
            stderr = simulation_root / "stderr.log"
            stdout.write_text("ok\n", encoding="utf-8")
            stderr.write_text("", encoding="utf-8")
            simulation_results.append(
                {
                    "sequence": simulation.sequence,
                    "name": simulation.name,
                    "status": "passed",
                    "return_code": 0,
                    "duration_seconds": 1.0,
                    "command": ["PKSim.CLI.exe", "export"],
                    "stdout_relative_path": stdout.relative_to(root).as_posix(),
                    "stderr_relative_path": stderr.relative_to(root).as_posix(),
                    "artifacts": [
                        _artifact(root, simulation_root, path)
                        for path in (csv_path, pkml, xml)
                    ],
                    "reason_codes": [],
                }
            )
        runs.append(
            {"run_id": run_id, "status": "passed", "simulations": simulation_results}
        )
    worker_script = root / "input" / "run_licensed_snapshot_reuse.ps1"
    worker_script.write_text("Write-Host test\n", encoding="utf-8")
    manifest = {
        "manifest_schema_version": "1.1.0",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_text_encoding": "utf-8",
        "worker_script_relative_path": worker_script.relative_to(root).as_posix(),
        "worker_script_sha256": sha256_file(worker_script),
        "status": "passed",
        "cli_version_output": "PKSim.CLI 12.3.173",
        "cli_sha256": "b" * 64,
        "runtime_archive_sha256": "a" * 64,
        "input_snapshot_relative_path": input_snapshot.relative_to(root).as_posix(),
        "input_snapshot_sha256": sha256_file(input_snapshot),
        "reconstructed_project_relative_path": project.relative_to(root).as_posix(),
        "reconstructed_project_sha256": sha256_file(project),
        "roundtrip_snapshot_relative_path": roundtrip.relative_to(root).as_posix(),
        "roundtrip_snapshot_sha256": sha256_file(roundtrip),
        "runs": runs,
        "completed_at_utc": "2026-08-01T00:00:00Z",
    }
    (root / "windows_run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_audit_recomputes_repeatability_and_scientific_metrics(
    tmp_path: Path,
) -> None:
    repository, archive, config = _prepare_source(tmp_path)
    protocol_directory = tmp_path / "protocol"
    protocol = build_snapshot_reuse_protocol(
        config_path=config,
        repository_root=repository,
        source_archive=archive,
        output_directory=protocol_directory,
    )
    protocol_path = protocol_directory / "snapshot_reuse_protocol.json"
    result_root = tmp_path / "windows"
    _build_windows_result(
        result_root,
        protocol_path,
        protocol,
        repository / "Model.json",
    )

    report = audit_snapshot_reuse_execution(
        protocol_path=protocol_path,
        windows_result_root=result_root,
        output_directory=tmp_path / "audit",
    )

    assert report["engineering_accepted"] is True
    assert report["scientific_accepted"] is True
    assert report["validation_grade"] == "external_study_validated"
    assert report["ranking_label"] == "reuse_positive"
    assert report["counts"]["evaluated_datasets"] == 1
    assert report["pooled_metrics"]["gmfe"] == 1.0


def test_audit_accepts_windows_manifest_relative_paths(
    tmp_path: Path,
) -> None:
    repository, archive, config = _prepare_source(tmp_path)
    protocol_directory = tmp_path / "protocol"
    protocol = build_snapshot_reuse_protocol(
        config_path=config,
        repository_root=repository,
        source_archive=archive,
        output_directory=protocol_directory,
    )
    protocol_path = protocol_directory / "snapshot_reuse_protocol.json"
    result_root = tmp_path / "windows"
    _build_windows_result(
        result_root,
        protocol_path,
        protocol,
        repository / "Model.json",
    )
    manifest_path = result_root / "windows_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "input_snapshot_relative_path",
        "reconstructed_project_relative_path",
        "roundtrip_snapshot_relative_path",
    ):
        manifest[key] = manifest[key].replace("/", "\\")
    for run in manifest["runs"]:
        for simulation in run["simulations"]:
            simulation["stdout_relative_path"] = simulation[
                "stdout_relative_path"
            ].replace("/", "\\")
            simulation["stderr_relative_path"] = simulation[
                "stderr_relative_path"
            ].replace("/", "\\")
            for artifact in simulation["artifacts"]:
                artifact["relative_path"] = artifact["relative_path"].replace(
                    "/", "\\"
                )
                artifact["simulation_relative_path"] = artifact[
                    "simulation_relative_path"
                ].replace("/", "\\")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_snapshot_reuse_execution(
        protocol_path=protocol_path,
        windows_result_root=result_root,
        output_directory=tmp_path / "audit",
    )

    assert report["engineering_accepted"] is True
    assert report["scientific_accepted"] is True


def test_roundtrip_accepts_only_locked_schema_and_unit_migrations() -> None:
    source = _snapshot()
    roundtrip = json.loads(json.dumps(source))
    roundtrip["Version"] = 80
    column = roundtrip["ObservedData"][0]["Columns"][0]
    column["Unit"] = "mg/l"
    column["Values"] = [value * 0.001 for value in column["Values"]]
    parameter = roundtrip["Simulations"][1]["Parameters"][0]
    parameter["Path"] = parameter["Path"].replace(
        "Applications|", "Events|", 1
    )
    roundtrip["ObservedData"][0]["ExtendedProperties"][1].pop("Value")

    comparison = compare_roundtrip_snapshots(
        input_snapshot=source,
        roundtrip_snapshot=roundtrip,
        expected_input_version=79,
    )

    assert comparison["accepted"] is True
    assert comparison["normalized_difference_count"] == 0
    assert comparison["roundtrip_schema_version"] == 80


def test_roundtrip_accepts_mg_per_litre_to_microgram_per_litre() -> None:
    source = _snapshot()
    source_column = source["ObservedData"][0]["Columns"][0]
    source_column["Unit"] = "mg/l"
    source_column["Values"] = [0.3107233, 0.08481341]
    roundtrip = json.loads(json.dumps(source))
    roundtrip_column = roundtrip["ObservedData"][0]["Columns"][0]
    roundtrip_column["Unit"] = "µg/l"
    roundtrip_column["Values"] = [310.7233, 84.81341]

    comparison = compare_roundtrip_snapshots(
        input_snapshot=source,
        roundtrip_snapshot=roundtrip,
        expected_input_version=79,
    )

    assert comparison["accepted"] is True
    assert comparison["normalized_difference_count"] == 0


def test_roundtrip_rejects_unit_change_without_value_conversion() -> None:
    source = _snapshot()
    source_column = source["ObservedData"][0]["Columns"][0]
    source_column["Unit"] = "mg/l"
    source_column["Values"] = [0.3107233, 0.08481341]
    roundtrip = json.loads(json.dumps(source))
    roundtrip["ObservedData"][0]["Columns"][0]["Unit"] = "µg/l"

    comparison = compare_roundtrip_snapshots(
        input_snapshot=source,
        roundtrip_snapshot=roundtrip,
        expected_input_version=79,
    )

    assert comparison["accepted"] is False
    assert comparison["normalized_difference_count"] == 2


def test_roundtrip_rejects_model_parameter_change() -> None:
    source = _snapshot()
    changed = json.loads(json.dumps(source))
    changed["Simulations"][1]["Parameters"][0]["Value"] = 4.0

    comparison = compare_roundtrip_snapshots(
        input_snapshot=source,
        roundtrip_snapshot=changed,
        expected_input_version=79,
    )

    assert comparison["accepted"] is False
    assert comparison["normalized_difference_count"] == 1
    assert "Parameters" in comparison["difference_examples"][0]["path"]


def test_roundtrip_observed_mass_units_require_equal_physical_values() -> None:
    source = _snapshot()
    column = source["ObservedData"][0]["Columns"][0]
    column.update(Dimension="Mass", Unit="kg", Values=[0.000274761, 0.000365411])
    converted = json.loads(json.dumps(source))
    converted["ObservedData"][0]["Columns"][0].update(Unit="mg", Values=[274.761, 365.41098])
    assert compare_roundtrip_snapshots(input_snapshot=source, roundtrip_snapshot=converted, expected_input_version=79)["accepted"]
    converted["ObservedData"][0]["Columns"][0]["Values"] = column["Values"]
    assert not compare_roundtrip_snapshots(input_snapshot=source, roundtrip_snapshot=converted, expected_input_version=79)["accepted"]


def test_roundtrip_dimensionless_default_is_limited_to_chart_axes() -> None:
    source = _snapshot()
    source["SimulationComparisons"] = [{"Name": "Comparison", "IndividualComparison": {
        "Axes": [{"Type": "Y", "Unit": "", "Dimension": "Dimensionless"}]}}]
    changed = json.loads(json.dumps(source))
    changed["SimulationComparisons"][0]["IndividualComparison"]["Axes"][0].pop("Dimension")
    assert compare_roundtrip_snapshots(input_snapshot=source, roundtrip_snapshot=changed, expected_input_version=79)["accepted"]
    source["Simulations"][1]["Parameters"][0]["Dimension"] = "Dimensionless"
    changed = json.loads(json.dumps(source))
    changed["Simulations"][1]["Parameters"][0].pop("Dimension")
    assert not compare_roundtrip_snapshots(input_snapshot=source, roundtrip_snapshot=changed, expected_input_version=79)["accepted"]


def _residual_chart_fixture():
    source = _snapshot()
    source["ParameterIdentifications"] = [{"Name": "Fit", "Analyses": [{
        "Name": "Residuals vs. Time", "Type": "ParameterIdentificationResidualVsTimeChart",
        "Chart": {"Axes": [{"Type": "Y", "Unit": "", "Dimension": "Dimensionless"}],
                  "Curves": [{"Name": "display", "X": "", "Y": "sim|observed|output",
                              "CurveOptions": {"Color": "#FF0000", "Symbol": "Circle"}}]}}]}]
    return source


def test_roundtrip_cached_residual_display_elision_is_recorded() -> None:
    source = _residual_chart_fixture()
    changed = json.loads(json.dumps(source))
    chart = changed["ParameterIdentifications"][0]["Analyses"][0]["Chart"]
    chart.pop("Curves")
    chart["Axes"][0].pop("Dimension")
    result = compare_roundtrip_snapshots(input_snapshot=source, roundtrip_snapshot=changed, expected_input_version=79)
    assert result["accepted"]
    assert result["display_only_elisions"][0]["omitted_display_curve_count"] == 1
    assert "Curves" in source["ParameterIdentifications"][0]["Analyses"][0]["Chart"]


@pytest.mark.parametrize("case", ["retained_modified_curve", "nonempty_x", "numerical_curve", "other_chart", "altered_model"])
def test_roundtrip_residual_elision_does_not_hide_semantic_changes(case) -> None:
    source = _residual_chart_fixture()
    analysis = source["ParameterIdentifications"][0]["Analyses"][0]
    if case == "nonempty_x":
        analysis["Chart"]["Curves"][0]["X"] = "real_observed_output"
    elif case == "numerical_curve":
        analysis["Chart"]["Curves"][0]["Values"] = [1, 2]
    elif case == "other_chart":
        analysis["Type"] = "ParameterIdentificationPredictedVsObservedChart"
    changed = json.loads(json.dumps(source))
    chart = changed["ParameterIdentifications"][0]["Analyses"][0]["Chart"]
    if case == "retained_modified_curve":
        chart["Curves"][0]["Y"] = "changed_mapping"
    else:
        chart.pop("Curves")
    if case == "altered_model":
        changed["Simulations"][1]["Parameters"][0]["Value"] = 4.0
    assert not compare_roundtrip_snapshots(input_snapshot=source, roundtrip_snapshot=changed, expected_input_version=79)["accepted"]


def test_roundtrip_rejects_compound_alternative_and_output_mapping_loss() -> None:
    source = _snapshot()
    source["Simulations"][0]["Compounds"] = [{"Name": "Drug", "Alternatives": [
        {"GroupName": "COMPOUND_LIPOPHILICITY", "Name": "Default"},
        {"GroupName": "COMPOUND_INTESTINAL_PERMEABILITY", "Name": "Default"}]}]
    source["Simulations"][0]["OutputMappings"] = [{"Path": "study2017|Organism|Drug"}]
    changed = json.loads(json.dumps(source))
    changed["Simulations"][0]["Compounds"][0]["Alternatives"].pop(0)
    changed["Simulations"][0]["OutputMappings"][0]["Path"] = "study2007|Organism|Drug"
    result = compare_roundtrip_snapshots(input_snapshot=source, roundtrip_snapshot=changed, expected_input_version=79)
    assert not result["accepted"]
    assert any("OutputMappings" in row["path"] for row in result["difference_examples"])
    assert any("Alternatives" in row["path"] for row in result["difference_examples"])


def test_grade_3_audit_registers_a_traceable_model_package(
    tmp_path: Path,
) -> None:
    repository, archive, config = _prepare_source(tmp_path)
    protocol_directory = tmp_path / "protocol"
    protocol = build_snapshot_reuse_protocol(
        config_path=config,
        repository_root=repository,
        source_archive=archive,
        output_directory=protocol_directory,
    )
    protocol_path = protocol_directory / "snapshot_reuse_protocol.json"
    result_root = tmp_path / "windows"
    _build_windows_result(
        result_root,
        protocol_path,
        protocol,
        repository / "Model.json",
    )
    audit_directory = tmp_path / "audit"
    audit_snapshot_reuse_execution(
        protocol_path=protocol_path,
        windows_result_root=result_root,
        output_directory=audit_directory,
    )

    summary = register_licensed_model(
        protocol_path=protocol_path,
        audit_path=audit_directory / "snapshot_reuse_audit.json",
        windows_result_root=result_root,
        output_directory=tmp_path / "registered",
    )

    assert summary["accepted"] is True
    assert summary["run_label_grade"] == 3
    assert summary["counts"]["simulations"] == 3
    with gzip.open(
        tmp_path / "registered" / "model_versions.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as handle:
        model = json.loads(next(handle))
    assert model["compounds"] == ["Drug"]
    assert model["validation_run_label"]["ranking_label"] == "reuse_positive"
    assert model["reusable_candidate"] is True
    assert len(model["validation_scope"]["scenarios"]) == 1
    assert model["validation_scope"]["dataset_names"] == [
        "Verification plasma"
    ]


def test_unit_conversion_log_interpolation_and_geometric_metrics() -> None:
    assert _time_in_hours(1.5, "day(s)") == 36.0
    assert _concentration_in_mol_per_litre(
        1.0, "ng/ml", 500.0
    ) == pytest.approx(2e-9)
    assert _concentration_in_mol_per_litre(
        1.0, "kg/l", 500.0
    ) == pytest.approx(2.0)
    assert _interpolate(
        [0.0, 2.0], [1.0, 4.0], 1.0, logarithmic=True
    ) == pytest.approx(2.0)
    metrics = _geometric_metrics([1.0, 2.0], [2.0, 1.0])
    assert metrics["gmfe"] == pytest.approx(2.0)
    assert metrics["geometric_bias"] == pytest.approx(1.0)
    assert metrics["two_fold_fraction"] == pytest.approx(1.0)


def test_dataset_metrics_excludes_zero_prediction_with_coverage_guard(
    tmp_path: Path,
) -> None:
    output_path = "Organism|Blood|Drug|Plasma"
    prediction = tmp_path / "prediction.csv"
    prediction.write_text(
        f"Time [h],{output_path} [ng/ml]\n"
        "0,0\n1,10\n2,8\n3,6\n4,4\n",
        encoding="utf-8",
    )
    observed = FrozenObservedSeries(
        dataset_name="Positive baseline fixture",
        dataset_index=0,
        column_index=0,
        source_location="fixture#/ObservedData/0/Columns/0",
        output_path=f"verification|{output_path}",
        scaling="Log",
        time_values=[0.0, 1.0, 2.0, 3.0, 4.0],
        time_unit="h",
        observed_values=[1.0, 10.0, 8.0, 6.0, 4.0],
        observed_unit="ng/ml",
        molecular_weight_g_per_mol=500.0,
    )
    simulation = FrozenSimulation(
        sequence=1,
        name="verification",
        role="model_verification",
        observed_dataset_names=[observed.dataset_name],
        evaluation_series=observed,
    )
    criteria = ScientificCriteria(
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
        pooled_gmfe_max=2.0,
        pooled_bias_min=0.5,
        pooled_bias_max=2.0,
        pooled_two_fold_fraction_min=0.8,
        dataset_gmfe_max=3.0,
        dataset_bias_min=1 / 3,
        dataset_bias_max=3.0,
        endpoint_ratio_min=0.5,
        endpoint_ratio_max=2.0,
        endpoint_scenario_fraction_min=0.8,
    )
    metrics = _dataset_metrics(
        result_root=tmp_path,
        artifacts=[
            ExecutionArtifact(
                relative_path=prediction.name,
                simulation_relative_path=prediction.name,
                byte_size=prediction.stat().st_size,
                sha256=sha256_file(prediction),
                extension=".csv",
            )
        ],
        simulation=simulation,
        observed=observed,
        criteria=criteria,
    )

    assert metrics["positive_pair_count"] == 4
    assert metrics["positive_pair_fraction"] == pytest.approx(0.8)
    assert metrics["excluded_nonpositive_prediction_count"] == 1
    assert metrics["dataset_guard_passed"] is True
