from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.licensed_snapshot_reuse import snapshot_semantic_sha256
from pbpk_kb.recommendation.model_matching.adaptation_planner import (
    AdaptationPlannerError,
    _convert_dose,
    _extend_output_horizon,
    plan_controlled_adaptations,
)


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _snapshot(*, dose: float, simulation_name: str) -> dict[str, object]:
    output = "Organism|PeripheralVenousBlood|Drug|Plasma (Peripheral Venous Blood)"
    return {
        "Version": 79,
        "ExpressionProfiles": [],
        "Individuals": [
            {
                "Name": "Standard adult",
                "OriginData": {
                    "Species": "Human",
                    "Population": "European_ICRP_2002",
                    "Gender": "MALE",
                },
            }
        ],
        "Compounds": [{"Name": "Drug", "Processes": []}],
        "Formulations": [{"Name": "Tablet"}],
        "Protocols": [
            {
                "Name": f"PO {dose:g} mg",
                "ApplicationType": "Oral",
                "DosingInterval": "Single",
                "Parameters": [
                    {"Name": "Start time", "Value": 0.0, "Unit": "h"},
                    {"Name": "InputDose", "Value": dose, "Unit": "mg"},
                    {
                        "Name": "Volume of water/body weight",
                        "Value": 3.5,
                        "Unit": "ml/kg",
                    },
                ],
            }
        ],
        "Simulations": [
            {
                "Name": simulation_name,
                "Individual": "Standard adult",
                "Compounds": [
                    {
                        "Name": "Drug",
                        "Protocol": {
                            "Name": f"PO {dose:g} mg",
                            "Formulations": [
                                {"Name": "Tablet", "Key": "Formulation"}
                            ],
                        },
                    }
                ],
                "OutputSchema": [
                    {
                        "Parameters": [
                            {"Name": "Start time", "Value": 0.0, "Unit": "h"},
                            {"Name": "End time", "Value": 24.0, "Unit": "h"},
                            {"Name": "Resolution", "Value": 10.0, "Unit": "pts/h"},
                        ]
                    }
                ],
                "Parameters": [
                    {
                        "Path": (
                            f"Applications|PO {dose:g} mg|Tablet|Application_1|"
                            "ProtocolSchemaItem|Volume of water/body weight"
                        ),
                        "Value": 3.5,
                        "Unit": "ml/kg",
                    }
                ],
                "OutputSelections": [output],
                "ObservedData": [],
                "OutputMappings": [],
                "HasResults": True,
            }
        ],
        "ObservedData": [],
        "SimulationClassifications": [],
    }


def _source_lock(snapshot_sha: str) -> dict[str, object]:
    return {
        "source_id": "osp_model_library",
        "model_id": "drug-model",
        "model_name": "Drug",
        "repository_url": "https://example.org/drug-model",
        "release_tag": "commit",
        "commit_sha": "a" * 40,
        "license_spdx_id": "GPL-2.0-only",
        "license_relative_path": "LICENSE",
        "license_sha256": "b" * 64,
        "snapshot_relative_path": "Drug-Model.json",
        "snapshot_declared_path": "Drug-Model.json",
        "snapshot_resolution_mode": "declared_relative_path",
        "snapshot_sha256": snapshot_sha,
        "snapshot_schema_version": 79,
        "evaluation_plan_relative_path": "evaluation-plan.yml",
        "evaluation_plan_sha256": "c" * 64,
        "source_archive_name": "drug.zip",
        "source_archive_sha256": "d" * 64,
    }


def _protocol(
    *, snapshot: dict[str, object], snapshot_sha: str, simulation_name: str
) -> dict[str, object]:
    output = (
        f"{simulation_name}|Organism|PeripheralVenousBlood|Drug|"
        "Plasma (Peripheral Venous Blood)"
    )
    return {
        "protocol_schema_version": "1.0.0",
        "protocol_builder_version": "test",
        "protocol_id": "snapshot-reuse-" + "e" * 32,
        "generated_at": "2026-08-23T00:00:00Z",
        "source": _source_lock(snapshot_sha),
        "source_archive_path": "/locked/drug.zip",
        "source_snapshot_semantic_sha256": snapshot_semantic_sha256(snapshot),
        "counts": {
            "compounds": 1,
            "expression_profiles": 0,
            "formulations": 1,
            "individuals": 1,
            "protocols": 1,
            "simulations": 1,
            "observed_data": 0,
            "simulation_classifications": 0,
            "model_building": 0,
            "model_verification": 1,
            "execution_only": 0,
        },
        "runtime": {
            "pksim_version": "12.3.173",
            "pksim_portable_sha256": "f" * 64,
            "cli_relative_path": "PKSim.CLI.exe",
            "command_timeout_seconds": 21600,
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
        "simulations": [
            {
                "sequence": 1,
                "name": simulation_name,
                "role": "model_verification",
                "individual": "Standard adult",
                "observed_dataset_names": ["Study"],
                "evaluation_series": {
                    "dataset_name": "Study",
                    "dataset_index": 0,
                    "column_index": 0,
                    "source_location": "Drug.json#/ObservedData/0/Columns/0",
                    "reference": "https://example.org/study",
                    "output_path": output,
                    "scaling": "Log",
                    "time_values": [1.0, 2.0],
                    "time_unit": "h",
                    "observed_values": [1.0, 0.5],
                    "observed_unit": "mg/l",
                    "molecular_weight_g_per_mol": 100.0,
                },
                "additional_evaluation_series": [],
            }
        ],
    }


def test_convert_dose_preserves_mass_basis() -> None:
    assert _convert_dose(50.0, "µg", "mg") == pytest.approx(0.05)
    assert _convert_dose(0.05, "mg/kg", "µg/kg") == pytest.approx(50.0)
    with pytest.raises(AdaptationPlannerError, match="cannot be interconverted"):
        _convert_dose(40.0, "mg", "mg/kg")


def test_output_horizon_uses_stable_decimal_value() -> None:
    simulation = _snapshot(
        dose=40.0, simulation_name="Stable output horizon"
    )["Simulations"][0]
    end_time = next(
        row
        for row in simulation["OutputSchema"][0]["Parameters"]
        if row["Name"] == "End time"
    )
    end_time["Value"] = 48.0
    resolution = next(
        row
        for row in simulation["OutputSchema"][0]["Parameters"]
        if row["Name"] == "Resolution"
    )
    resolution["Value"] = 20.0

    adjustment = _extend_output_horizon(
        simulation, required_end_time_hours=48.0025864
    )

    assert adjustment is not None
    assert adjustment["new_value"] == 48.0525864
    assert repr(adjustment["new_value"]) == "48.0525864"


def test_planner_builds_checksum_locked_dose_adaptation(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    benchmark = tmp_path / "benchmark"
    candidate_snapshot = _snapshot(dose=40.0, simulation_name="Candidate 40 mg PO")
    target_scenario = _snapshot(dose=50.0, simulation_name="Source 50 mg PO")
    target_end_time = next(
        row
        for row in target_scenario["Simulations"][0]["OutputSchema"][0][
            "Parameters"
        ]
        if row["Name"] == "End time"
    )
    target_end_time["Value"] = 1.0
    candidate_snapshot["Protocols"].extend(target_scenario["Protocols"])
    candidate_snapshot["Simulations"].extend(target_scenario["Simulations"])
    candidate_snapshot["Compounds"].append(
        {"Name": "Drug metabolite", "Processes": []}
    )
    candidate_snapshot["ObservedData"] = [{"Name": "Withheld target data"}]
    candidate_snapshot["ObservedDataClassifications"] = [
        {"Name": "Withheld target classification"}
    ]
    candidate_snapshot["ParameterIdentifications"] = [
        {"Name": "Analysis referencing removed simulations"}
    ]
    candidate_snapshot["SensitivityAnalyses"] = [
        {"Name": "Sensitivity analysis referencing removed simulations"}
    ]
    for simulation in candidate_snapshot["Simulations"]:
        simulation["Compounds"].append(
            {"Name": "Drug metabolite"}
        )
    source_snapshot = copy.deepcopy(candidate_snapshot)
    candidate_snapshot_path = batch / "snapshots/candidate/Drug-Model.json"
    source_snapshot_path = batch / "snapshots/source/Drug-Model.json"
    candidate_snapshot_sha = _write_json(candidate_snapshot_path, candidate_snapshot)
    source_snapshot_sha = _write_json(source_snapshot_path, source_snapshot)
    candidate_protocol = _protocol(
        snapshot=candidate_snapshot,
        snapshot_sha=candidate_snapshot_sha,
        simulation_name="Candidate 40 mg PO",
    )
    target_protocol = _protocol(
        snapshot=source_snapshot,
        snapshot_sha=source_snapshot_sha,
        simulation_name="Source 50 mg PO",
    )
    target_frozen = target_protocol["simulations"][0]
    target_frozen["sequence"] = 2
    candidate_protocol["simulations"].append(target_frozen)
    candidate_protocol["counts"]["simulations"] = 2
    candidate_protocol["counts"]["model_verification"] = 2
    source_protocol = copy.deepcopy(candidate_protocol)
    source_protocol["protocol_id"] = "snapshot-reuse-" + "a" * 32
    candidate_protocol_path = batch / "protocols/candidate/snapshot_reuse_protocol.json"
    source_protocol_path = batch / "protocols/source/snapshot_reuse_protocol.json"
    candidate_protocol_sha = _write_json(candidate_protocol_path, candidate_protocol)
    source_protocol_sha = _write_json(source_protocol_path, source_protocol)
    jobs = [
        {
            "protocol_key": "candidate",
            "protocol_relative_path": candidate_protocol_path.relative_to(batch).as_posix(),
            "protocol_sha256": candidate_protocol_sha,
            "snapshot_relative_path": candidate_snapshot_path.relative_to(batch).as_posix(),
            "snapshot_sha256": candidate_snapshot_sha,
            "repository_commit": "a" * 40,
        },
        {
            "protocol_key": "source",
            "protocol_relative_path": source_protocol_path.relative_to(batch).as_posix(),
            "protocol_sha256": source_protocol_sha,
            "snapshot_relative_path": source_snapshot_path.relative_to(batch).as_posix(),
            "snapshot_sha256": source_snapshot_sha,
            "repository_commit": "b" * 40,
        },
    ]
    jobs_sha = _write_jsonl(batch / "jobs.jsonl", jobs)
    _write_json(batch / "batch_manifest.json", {"jobs_sha256": jobs_sha})

    task_id = "task-1"
    candidate_id = "candidate-1"
    tasks = [
        {
            "task_id": task_id,
            "active_moieties": ["drug"],
        }
    ]
    truth = [
        {
            "task_id": task_id,
            "source_protocol_keys": ["source"],
            "source_simulation_sequence": 2,
            "dataset_name": "Study",
            "source_location": "Drug.json#/ObservedData/0/Columns/0",
            "reference": "https://example.org/study",
            "output_path": (
                "Source 50 mg PO|Organism|PeripheralVenousBlood|Drug|"
                "Plasma (Peripheral Venous Blood)"
            ),
            "time_values": [1.0, 2.0],
            "time_unit": "h",
            "observed_values": [1.0, 0.5],
            "observed_unit": "mg/l",
            "molecular_weight_g_per_mol": 100.0,
            "criteria": source_protocol["criteria"],
        }
    ]
    candidates = [
        {
            "candidate_id": candidate_id,
            "model_family": "Drug",
            "version_aliases": [
                {
                    "protocol_key": "candidate",
                    "repository_commit": "a" * 40,
                }
            ],
        }
    ]
    pairs = [
        {
            "task_id": task_id,
            "candidate_id": candidate_id,
            "adaptation_required": True,
            "same_active_moiety": True,
        }
    ]
    _write_jsonl(benchmark / "benchmark_tasks.jsonl", tasks)
    _write_jsonl(benchmark / "benchmark_truth.jsonl", truth)
    _write_jsonl(benchmark / "candidates.jsonl", candidates)
    pair_sha = _write_jsonl(benchmark / "candidate_pairs.jsonl", pairs)
    _write_json(
        benchmark / "manifest.json",
        {"output_checksums": {"candidate_pairs.jsonl": pair_sha}},
    )

    result = plan_controlled_adaptations(
        batch_directory=batch,
        benchmark_directory=benchmark,
        output_directory=tmp_path / "adaptations",
    )

    assert result["eligible_pair_count"] == 1
    assert result["planned_count"] == 1
    assert result["blocked_count"] == 0
    plan = json.loads(
        (tmp_path / "adaptations/adaptation_plans.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert plan["adaptation_operation_multilabel"] == ["dose"]
    assert plan["adaptation_operation_count"] == 1
    assert plan["template_simulation_name"] == "Candidate 40 mg PO"
    assert plan["source_simulation_name"] == "Source 50 mg PO"
    assert plan["withheld_target_simulation"] is True
    assert plan["execution_adjustment_count"] == 1
    assert plan["execution_adjustments"][0]["adjustment_type"] == (
        "output_horizon"
    )
    adapted_snapshot_path = (
        tmp_path / "adaptations" / plan["adapted_snapshot_relative_path"]
    )
    assert sha256_file(adapted_snapshot_path) == plan["adapted_snapshot_sha256"]
    adapted_snapshot = json.loads(adapted_snapshot_path.read_text(encoding="utf-8"))
    adapted_protocol = adapted_snapshot["Protocols"][-1]
    dose = next(
        row
        for row in adapted_protocol["Parameters"]
        if row["Name"] == "InputDose"
    )
    assert dose["Value"] == pytest.approx(50.0)
    assert [row["Name"] for row in adapted_snapshot["Simulations"]] == [
        f"ADAPT-{plan['protocol_key']}"
    ]
    assert "SimulationClassifications" not in adapted_snapshot
    assert "ObservedData" not in adapted_snapshot
    assert "ObservedDataClassifications" not in adapted_snapshot
    assert "ParameterIdentifications" not in adapted_snapshot
    assert "SensitivityAnalyses" not in adapted_snapshot
    assert "ObservedData" not in adapted_snapshot["Simulations"][0]
    assert "OutputMappings" not in adapted_snapshot["Simulations"][0]
    assert adapted_snapshot["Simulations"][0]["Parameters"][0]["Path"] == (
        f"Applications|Adaptation ADAPT-{plan['protocol_key']}|Tablet|"
        "Application_1|ProtocolSchemaItem|Volume of water/body weight"
    )
    adapted_end_time = next(
        row
        for row in adapted_snapshot["Simulations"][0]["OutputSchema"][0][
            "Parameters"
        ]
        if row["Name"] == "End time"
    )
    assert adapted_end_time["Value"] == pytest.approx(2.1)
    assert plan["execution_adjustments"][0]["grid_step_hours"] == (
        pytest.approx(0.1)
    )
    assert [
        row["Name"] for row in adapted_snapshot["Simulations"][0]["Compounds"]
    ] == ["Drug", "Drug metabolite"]
    assert "Protocol" not in adapted_snapshot["Simulations"][0]["Compounds"][1]
    protocol_payload = json.loads(
        (
            tmp_path
            / "adaptations"
            / plan["protocol_relative_path"]
        ).read_text(encoding="utf-8")
    )
    assert protocol_payload["counts"]["simulations"] == 1
    assert protocol_payload["counts"]["model_verification"] == 1
