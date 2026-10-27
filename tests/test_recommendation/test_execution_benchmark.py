from __future__ import annotations

import gzip
import json
from pathlib import Path

from pbpk_kb.recommendation.model_matching.execution_benchmark import (
    _assign_compound_and_family_disjoint_splits,
    _read_linux_repeat_index,
    _series_compound,
    _structured_simulation_requirement,
    _structured_series_requirement,
    build_automatic_labels,
    build_automatic_labels_from_audits,
    build_execution_benchmark,
    build_execution_benchmark_from_audits,
    import_locked_validation_results,
    parse_simulation_requirement,
)
from pbpk_kb.recommendation.model_matching.execution_experiment import (
    run_execution_experiment,
)


def test_grouped_split_stratifies_reusable_candidate_components() -> None:
    compounds = [f"compound-{index}" for index in range(6)]
    tasks = [
        {
            "task_id": f"task-{index}",
            "required_model_compounds": [compound],
        }
        for index, compound in enumerate(compounds)
    ]
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "model_family": f"family-{index}",
            "active_moieties": [compound],
            "reusable_candidate": index < 3,
        }
        for index, compound in enumerate(compounds)
    ]

    audit = _assign_compound_and_family_disjoint_splits(
        tasks=tasks,
        candidates=candidates,
    )

    assert audit["passed"] is True
    assert audit["reusable_candidate_split_counts"] == {
        "test": 1,
        "train": 1,
        "validation": 1,
    }
    assert audit["compound_split_violations"] == {}
    assert audit["model_family_split_violations"] == {}


def _write_registration(root: Path) -> Path:
    registration = root / "registration"
    provenance = registration / "provenance"
    provenance.mkdir(parents=True)
    criteria = {
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
    }
    simulations = []
    for sequence, dose in ((1, 10), (2, 20)):
        simulations.append(
            {
                "sequence": sequence,
                "name": f"po {dose} mg, independent study, tab, n=12",
                "role": "model_verification",
                "evaluation_series": {
                    "dataset_name": f"hidden dataset {sequence}",
                    "source_location": f"source.json#/ObservedData/{sequence}",
                    "reference": f"https://example.org/{sequence}",
                    "output_path": "Organism|Plasma|Test drug",
                    "time_values": [0.0, 1.0, 2.0],
                    "time_unit": "h",
                    "observed_values": [1.0, 2.0, 1.0],
                    "observed_unit": "µmol/l",
                    "molecular_weight_g_per_mol": 300.0,
                },
            }
        )
    protocol = {
        "source_snapshot_semantic_sha256": "c" * 64,
        "criteria": criteria,
        "simulations": simulations,
    }
    (provenance / "snapshot_reuse_protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    model = {
        "model_version_id": "model-version:test",
        "model_family": "Test drug",
        "compounds": ["Test drug"],
        "species": ["Human"],
        "routes": ["Oral"],
        "enzymes": ["CYP3A4"],
        "transporters": [],
        "process_types": ["absorption", "metabolism"],
        "software": "PK-Sim",
        "software_version": "12.3.173",
        "license": {"status": "accepted", "spdx_id": "GPL-2.0"},
        "static_parse_status": "parsed",
        "execution_probe_status": "passed",
        "reusable_candidate": True,
        "repository_commit": "d" * 40,
        "project_sha256": "e" * 64,
    }
    with gzip.open(registration / "model_versions.jsonl.gz", "wt") as handle:
        handle.write(json.dumps(model) + "\n")
    (registration / "summary.json").write_text("{}\n", encoding="utf-8")
    return registration


def test_requirement_parser_preserves_continuous_dose() -> None:
    parsed = parse_simulation_requirement("po 39.9 mg QD, tablet, female")
    assert parsed == {
        "route": "oral",
        "dose_value": 39.9,
        "dose_unit": "mg",
        "frequency": "once_daily",
        "sex": "female",
        "formulation": "tablet",
    }


def test_structured_requirement_reads_protocol_instead_of_simulation_name() -> None:
    snapshot = {
        "Individuals": [
            {
                "Name": "Adult",
                "OriginData": {
                    "Species": "Human",
                    "Population": "European_ICRP_2002",
                    "Gender": "FEMALE",
                },
            }
        ],
        "Protocols": [
            {
                "Name": "Protocol A",
                "ApplicationType": "Oral",
                "DosingInterval": "DI_24",
                "Parameters": [
                    {"Name": "InputDose", "Value": 39.9, "Unit": "mg"},
                    {"Name": "Start time", "Value": 2.0, "Unit": "h"},
                    {"Name": "End time", "Value": 50.0, "Unit": "h"},
                ],
            }
        ],
        "Simulations": [
            {
                "Name": "opaque scenario name",
                "Individual": "Adult",
                "Compounds": [
                    {
                        "Name": "Test drug",
                        "Protocol": {
                            "Name": "Protocol A",
                            "Formulations": [{"Name": "Tablet"}],
                        },
                    }
                ],
            }
        ],
    }

    scenario = _structured_simulation_requirement(
        snapshot=snapshot,
        simulation_name="opaque scenario name",
        primary_compound="Test drug",
    )

    assert scenario["dose_value"] == 39.9
    assert scenario["dose_unit"] == "mg"
    assert scenario["route"] == "oral"
    assert scenario["frequency"] == "once_daily"
    assert scenario["formulations"] == ["Tablet"]
    assert scenario["scenario_source"] == "structured_pksim_snapshot"


def test_multi_compound_requirement_separates_analyte_and_regimens() -> None:
    snapshot = {
        "Individuals": [
            {
                "Name": "Adult",
                "OriginData": {
                    "Species": "Human",
                    "Population": "European_ICRP_2002",
                    "Gender": "MALE",
                },
            }
        ],
        "Protocols": [
            {
                "Name": "Victim protocol",
                "ApplicationType": "Oral",
                "DosingInterval": "Single",
                "Parameters": [{"Name": "InputDose", "Value": 5, "Unit": "mg"}],
            },
            {
                "Name": "Perpetrator protocol",
                "ApplicationType": "Oral",
                "DosingInterval": "DI_24",
                "Parameters": [
                    {"Name": "InputDose", "Value": 200, "Unit": "mg"}
                ],
            },
        ],
        "Simulations": [
            {
                "Name": "DDI scenario",
                "Individual": "Adult",
                "Compounds": [
                    {"Name": "Victim", "Protocol": {"Name": "Victim protocol"}},
                    {
                        "Name": "Perpetrator",
                        "Protocol": {"Name": "Perpetrator protocol"},
                    },
                ],
            }
        ],
    }
    series = {
        "output_path": "DDI scenario|Organism|Plasma|Victim|Concentration"
    }

    observed = _series_compound(
        compounds=["Victim", "Perpetrator"], series=series
    )
    scenario = _structured_series_requirement(
        snapshot=snapshot,
        simulation_name="DDI scenario",
        observed_compound=observed,
        declared_compounds=["Victim", "Perpetrator"],
    )

    assert observed == "Victim"
    assert scenario["observed_compound"] == "Victim"
    assert scenario["administered_compounds"] == ["Perpetrator", "Victim"]
    assert len(scenario["regimens"]) == 2
    assert scenario["primary_regimen_role"] == "observed_compound_regimen"
    assert scenario["dose_value"] == 5


def test_structured_requirement_parses_multi_event_population_protocol() -> None:
    snapshot = {
        "Populations": [
            {
                "Name": "European population",
                "Settings": {
                    "Individual": {
                        "OriginData": {
                            "Species": "Human",
                            "Population": "European_ICRP_2002",
                            "Gender": "MALE",
                        }
                    }
                },
            }
        ],
        "Protocols": [
            {
                "Name": "Split dose",
                "DosingInterval": "Single",
                "Schemas": [
                    {
                        "Name": "First",
                        "SchemaItems": [
                            {
                                "ApplicationType": "Oral",
                                "Parameters": [
                                    {"Name": "InputDose", "Value": 120, "Unit": "mg"},
                                    {"Name": "Start time", "Value": 0, "Unit": "h"},
                                ],
                            }
                        ],
                        "Parameters": [
                            {"Name": "Start time", "Value": 0, "Unit": "h"},
                            {"Name": "NumberOfRepetitions", "Value": 1},
                        ],
                    },
                    {
                        "Name": "Second",
                        "SchemaItems": [
                            {
                                "ApplicationType": "Oral",
                                "Parameters": [
                                    {"Name": "InputDose", "Value": 80, "Unit": "mg"},
                                    {"Name": "Start time", "Value": 0, "Unit": "h"},
                                ],
                            }
                        ],
                        "Parameters": [
                            {"Name": "Start time", "Value": 2, "Unit": "h"},
                            {"Name": "NumberOfRepetitions", "Value": 1},
                        ],
                    },
                ],
            }
        ],
        "Simulations": [
            {
                "Name": "population split-dose study",
                "Population": "European population",
                "Compounds": [
                    {
                        "Name": "Test drug",
                        "Protocol": {
                            "Name": "Split dose",
                            "Formulations": [{"Name": "Tablet"}],
                        },
                    }
                ],
            }
        ],
    }

    scenario = _structured_simulation_requirement(
        snapshot=snapshot,
        simulation_name="population split-dose study",
        primary_compound="Test drug",
    )

    assert scenario["route"] == "oral"
    assert scenario["dose_value"] == 200
    assert scenario["dose_unit"] == "mg"
    assert scenario["frequency"] == "single_dose_multi_event"
    assert scenario["schedule_kind"] == "multi_event"
    assert [row["start_time_value"] for row in scenario["dosing_events"]] == [
        0,
        2,
    ]
    assert scenario["population_name"] == "European population"
    assert scenario["population"] == "European_ICRP_2002"


def test_execution_labels_require_complete_pooled_validation(tmp_path: Path) -> None:
    registration = _write_registration(tmp_path)
    benchmark = tmp_path / "benchmark"
    manifest = build_execution_benchmark(
        registrations=[registration], output_directory=benchmark
    )
    assert manifest["task_count"] == 2
    task_rows = [
        json.loads(line)
        for line in (benchmark / "benchmark_tasks.jsonl").read_text().splitlines()
    ]
    serialized_tasks = json.dumps(task_rows)
    assert "model-version:test" not in serialized_tasks
    assert "hidden dataset" not in serialized_tasks

    results_path = tmp_path / "execution_results.jsonl"
    rows = []
    for task in task_rows:
        rows.append(
            {
                "task_id": task["task_id"],
                "candidate_id": "model-version:test",
                "compile_success": True,
                "execution_success": True,
                "dimensional_check_passed": True,
                "reproducibility_passed": True,
                "mass_balance_passed": None,
                "predicted_time_values": [0.0, 1.0, 2.0],
                "predicted_time_unit": "h",
                "predicted_values": [1.0, 2.0, 1.0],
                "predicted_unit": "µmol/l",
                "adaptation_operations": [],
                "model_structure_changed": False,
                "compound_parameters_changed": False,
                "execution_manifest_sha256": "f" * 64,
            }
        )
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    labels = tmp_path / "labels"
    result = build_automatic_labels(
        benchmark_directory=benchmark,
        execution_results_path=results_path,
        output_directory=labels,
    )
    assert result["label_counts"]["3"] == 2
    group = json.loads(
        (labels / "candidate_group_labels.jsonl").read_text().splitlines()[0]
    )
    assert group["decision"] == "direct_reuse"
    assert group["pooled_scientific_passed"] is True


def test_locked_validation_import_is_marked_as_not_new_execution(tmp_path: Path) -> None:
    registration = _write_registration(tmp_path)
    benchmark = tmp_path / "benchmark"
    build_execution_benchmark(
        registrations=[registration], output_directory=benchmark
    )
    metrics = []
    for sequence in (1, 2):
        metrics.append(
            {
                "dataset_name": f"hidden dataset {sequence}",
                "positive_observed_mol_per_litre": [1e-6, 2e-6, 1e-6],
                "positive_predicted_mol_per_litre": [1e-6, 2e-6, 1e-6],
                "dataset_guard_passed": True,
            }
        )
    provenance = registration / "provenance"
    (provenance / "verification_dataset_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8"
    )
    (provenance / "snapshot_reuse_audit.json").write_text(
        json.dumps(
            {
                "engineering_checks": {
                    "all_artifacts_verified": True,
                    "all_simulations_passed": True,
                    "csv_repeatability_passed": True,
                },
                "windows_manifest_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )

    result = import_locked_validation_results(
        benchmark_directory=benchmark,
        registration=registration,
        output_path=tmp_path / "locked_results.jsonl",
    )

    assert result["result_count"] == 2
    assert result["new_execution_performed"] is False
    assert result["scientific_claim_eligible"] is False


def test_execution_baselines_consume_only_complete_automatic_labels(
    tmp_path: Path,
) -> None:
    registration = _write_registration(tmp_path)
    benchmark = tmp_path / "benchmark"
    build_execution_benchmark(
        registrations=[registration], output_directory=benchmark
    )
    tasks = [
        json.loads(line)
        for line in (benchmark / "benchmark_tasks.jsonl").read_text().splitlines()
    ]
    results_path = tmp_path / "execution_results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task["task_id"],
                    "candidate_id": "model-version:test",
                    "compile_success": True,
                    "execution_success": True,
                    "dimensional_check_passed": True,
                    "reproducibility_passed": True,
                    "mass_balance_passed": None,
                    "predicted_time_values": [0.0, 1.0, 2.0],
                    "predicted_time_unit": "h",
                    "predicted_values": [1.0, 2.0, 1.0],
                    "predicted_unit": "µmol/l",
                    "adaptation_operations": [],
                    "model_structure_changed": False,
                    "compound_parameters_changed": False,
                }
            )
            + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels"
    build_automatic_labels(
        benchmark_directory=benchmark,
        execution_results_path=results_path,
        output_directory=labels,
    )
    perturbed_tasks_path = tmp_path / "perturbed_tasks.jsonl"
    perturbed_tasks_path.write_text(
        "".join(
            json.dumps(
                {
                    **task,
                    "task_id": f"perturbed-{task['task_id']}",
                    "base_task_id": task["task_id"],
                    "perturbation_group_id": "perturbation-group:test",
                    "perturbation_type": "relative_plus_1pct",
                    "domain_status": "local",
                    "scenario": {
                        **task["scenario"],
                        "dose_value": float(task["scenario"]["dose_value"]) * 1.01,
                    },
                    "scientific_grade_generated": False,
                }
            )
            + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )

    experiment = run_execution_experiment(
        benchmark_directory=benchmark,
        labels_directory=labels,
        output_directory=tmp_path / "experiment",
        perturbed_tasks_path=perturbed_tasks_path,
    )

    assert experiment["benchmark_level"] == "execution_derived"
    assert experiment["label_coverage"] == 1.0
    assert experiment["pair_count_per_method"] == 2
    assert experiment["perturbed_query_count"] == 2
    assert experiment["perturbed_pair_count_per_method"] == 2
    assert experiment["perturbation_comparison"]["hard_gate_only"]["pair_count"] == 2
    perturbation_prediction = json.loads(
        (tmp_path / "experiment" / "perturbation_predictions_hard_gate_only.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert perturbation_prediction["relevance_grade"] is None
    assert experiment["scientific_claim_eligible"] is False


def test_audit_derived_benchmark_keeps_all_series_and_semantic_versions(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    audits = tmp_path / "audits"
    (batch / "protocols").mkdir(parents=True)
    (batch / "snapshots").mkdir()
    criteria = {
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
    }
    first_series = {
        "dataset_name": "study plasma",
        "source_location": "snapshot.json#/ObservedData/0/Columns/1",
        "reference": "https://example.org/study",
        "output_path": "po 10 mg|Organism|Plasma|Test drug",
        "time_values": [0.0, 1.0, 2.0],
        "time_unit": "h",
        "observed_values": [1.0, 2.0, 1.0],
        "observed_unit": "µmol/l",
        "molecular_weight_g_per_mol": 300.0,
    }
    second_series = {
        **first_series,
        "dataset_name": "study whole blood",
        "source_location": "snapshot.json#/ObservedData/1/Columns/1",
        "output_path": "po 10 mg|Organism|WholeBlood|Test drug",
        "observed_values": [0.8, 1.6, 0.8],
    }
    jobs = []
    audit_rows = []
    for index, semantic in enumerate(("a" * 64, "b" * 64), 1):
        key = f"key{index}"
        protocol = {
            "source_snapshot_semantic_sha256": semantic,
            "source": {
                "model_name": "Test drug",
                "license_spdx_id": "GPL-2.0",
            },
            "runtime": {"pksim_version": "12.3.173"},
            "counts": {"simulations": 1},
            "criteria": criteria,
            "simulations": [
                {
                    "sequence": 1,
                    "name": "po 10 mg independent study tablet",
                    "role": "model_verification",
                    "evaluation_series": first_series,
                    "additional_evaluation_series": [second_series],
                }
            ],
        }
        protocol_path = batch / "protocols" / f"{key}.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        snapshot = {
            "Compounds": [
                {
                    "Name": "Test drug",
                    "Processes": [{"MoleculeName": "CYP3A4"}],
                }
            ],
            "Formulations": [{"Name": "tablet"}],
            "Individuals": [
                {
                    "Name": "Adult",
                    "OriginData": {
                        "Species": "Human",
                        "Population": "European_ICRP_2002",
                        "Gender": "MALE",
                    },
                }
            ],
            "Protocols": [
                {
                    "Name": "PO 10 mg",
                    "ApplicationType": "Oral",
                    "DosingInterval": "Single",
                    "Parameters": [
                        {"Name": "InputDose", "Value": 10.0, "Unit": "mg"},
                        {"Name": "Start time", "Value": 0.0, "Unit": "h"},
                    ],
                }
            ],
            "Simulations": [
                {
                    "Name": "po 10 mg independent study tablet",
                    "Individual": "Adult",
                    "Compounds": [
                        {
                            "Name": "Test drug",
                            "Protocol": {
                                "Name": "PO 10 mg",
                                "Formulations": [{"Name": "tablet"}],
                            },
                        }
                    ],
                    "OutputSelections": [
                        "Organism|Plasma|Test drug",
                        "Organism|WholeBlood|Test drug",
                    ],
                }
            ],
        }
        snapshot_path = batch / "snapshots" / f"{key}.json"
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
        jobs.append(
            {
                "protocol_key": key,
                "protocol_id": f"snapshot-reuse-{index:032x}",
                "protocol_sha256": _sha256(protocol_path),
                "protocol_relative_path": f"protocols/{key}.json",
                "snapshot_sha256": _sha256(snapshot_path),
                "snapshot_relative_path": f"snapshots/{key}.json",
                "full_name": "OSP/Test-drug-Model",
                "version_id": f"OSP/Test-drug-Model@{index}",
                "repository_commit": f"{index:040x}",
                "compounds": ["Test drug"],
            }
        )
        audit_root = audits / "audits" / key
        audit_root.mkdir(parents=True)
        metrics = []
        for series in (first_series, second_series):
            metrics.append(
                {
                    "simulation_sequence": 1,
                    "dataset_name": series["dataset_name"],
                    "source_location": series["source_location"],
                    "dataset_guard_passed": True,
                    "endpoint_passed": True,
                }
            )
        (audit_root / "verification_dataset_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in metrics),
            encoding="utf-8",
        )
        audit_report = {
            "engineering_accepted": True,
            "scientific_accepted": True,
            "run_label_grade": 3,
            "roundtrip": {"roundtrip_semantic_sha256": semantic},
        }
        audit_path = audit_root / "snapshot_reuse_audit.json"
        audit_path.write_text(json.dumps(audit_report), encoding="utf-8")
        audit_rows.append(
            {
                "protocol_key": key,
                "protocol_id": jobs[-1]["protocol_id"],
                "audit_status": "audited_passed",
                "audit_relative_path": f"audits/{key}/snapshot_reuse_audit.json",
                "audit_report_sha256": _sha256(audit_path),
            }
        )
    jobs_path = batch / "jobs.jsonl"
    jobs_path.write_text(
        "".join(json.dumps(row) + "\n" for row in jobs), encoding="utf-8"
    )
    (batch / "batch_manifest.json").write_text(
        json.dumps({"jobs_sha256": _sha256(jobs_path)}), encoding="utf-8"
    )
    audit_index = audits / "audit_index.jsonl"
    audit_index.write_text(
        "".join(json.dumps(row) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    (audits / "summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "output_checksums": {
                    "audit_index.jsonl": _sha256(audit_index)
                },
            }
        ),
        encoding="utf-8",
    )

    benchmark = tmp_path / "audit_benchmark"
    manifest = build_execution_benchmark_from_audits(
        batch_directory=batch,
        audit_directory=audits,
        output_directory=benchmark,
    )
    assert manifest["task_count"] == 2
    assert manifest["candidate_count"] == 2
    assert manifest["candidate_pair_count"] == 4
    assert manifest["non_degenerate_same_compound_groups"] == ["test drug"]
    assert manifest["scientific_claim_eligible"] is False
    assert "TASK_COUNT_LT_100" in manifest["scientific_claim_blockers"]
    assert (
        "INDEPENDENT_COMPOUND_COUNT_LT_30"
        in manifest["scientific_claim_blockers"]
    )
    assert "LINUX_OSPSUITE_R_REPEAT_REQUIRED" not in manifest["scientific_claim_blockers"]
    assert (
        "LINUX_OSPSUITE_R_REPEAT_REQUIRED"
        in manifest["cross_platform_claim_blockers"]
    )

    labels = build_automatic_labels_from_audits(
        benchmark_directory=benchmark,
        output_directory=tmp_path / "audit_labels",
    )
    assert labels["label_counts"]["3"] == 4
    assert labels["label_origin_counts"] == {"exact_executed_scenario": 4}
    experiment = run_execution_experiment(
        benchmark_directory=benchmark,
        labels_directory=tmp_path / "audit_labels",
        output_directory=tmp_path / "audit_experiment",
    )
    assert experiment["label_coverage"] == 1.0
    assert experiment["pair_count_per_method"] == 4
    assert experiment["non_degenerate_ranking_query_count"] == 2

    linux_repeat = tmp_path / "linux-repeat"
    linux_repeat.mkdir()
    linux_simulations = [
        {
            "protocol_key": job["protocol_key"],
            "sequence": 1,
            "status": "executed_negative" if index == 0 else "passed",
        }
        for index, job in enumerate(jobs)
    ]
    linux_protocols = [
        {
            "protocol_key": row["protocol_key"],
            "simulation_count": 1,
            "passed_count": int(row["status"] == "passed"),
            "executed_negative_count": int(row["status"] == "executed_negative"),
            "runner_failed_count": 0,
            "status": row["status"],
        }
        for row in linux_simulations
    ]
    linux_simulation_path = linux_repeat / "simulation_results.jsonl"
    linux_protocol_path = linux_repeat / "protocol_results.jsonl"
    linux_simulation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in linux_simulations),
        encoding="utf-8",
    )
    linux_protocol_path.write_text(
        "".join(json.dumps(row) + "\n" for row in linux_protocols),
        encoding="utf-8",
    )
    (linux_repeat / "summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "execution_complete": True,
                "cross_platform_all_passed": False,
                "jobs_sha256": _sha256(jobs_path),
                "counts": {
                    "protocol_count": 2,
                    "simulation_count": 2,
                    "simulation_passed_count": 1,
                    "executed_negative_count": 1,
                    "runner_failed_count": 0,
                },
                "output_checksums": {
                    "simulation_results.jsonl": _sha256(linux_simulation_path),
                    "protocol_results.jsonl": _sha256(linux_protocol_path),
                },
            }
        ),
        encoding="utf-8",
    )

    benchmark_with_linux = tmp_path / "audit_benchmark_with_linux"
    linux_manifest = build_execution_benchmark_from_audits(
        batch_directory=batch,
        audit_directory=audits,
        linux_repeat_directory=linux_repeat,
        output_directory=benchmark_with_linux,
    )
    linux_candidates = [
        json.loads(line)
        for line in (benchmark_with_linux / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(candidate["run_label_grade"] == 3 for candidate in linux_candidates)
    assert all(candidate["scientific_accepted"] for candidate in linux_candidates)
    assert any(
        candidate["linux_ospsuite_repeat_executed_negative"]
        and not candidate["cross_platform_scientific_accepted"]
        for candidate in linux_candidates
    )
    assert (
        "LINUX_OSPSUITE_R_CROSS_PLATFORM_MISMATCH_PRESENT"
        in linux_manifest["cross_platform_claim_blockers"]
    )
    linux_labels = build_automatic_labels_from_audits(
        benchmark_directory=benchmark_with_linux,
        output_directory=tmp_path / "audit_labels_with_linux",
    )
    assert linux_labels["label_counts"]["3"] == 4


def test_linux_repeat_index_keeps_terminal_runner_failures_for_blockers(
    tmp_path: Path,
) -> None:
    repeat = tmp_path / "linux-repeat"
    repeat.mkdir()
    simulations = [
        {
            "protocol_key": "protocol-one",
            "sequence": 1,
            "status": "runner_failed",
            "reason_codes": ["LINUX_OSPSUITE_R_EXECUTION_TIMEOUT"],
        }
    ]
    protocols = [
        {
            "protocol_key": "protocol-one",
            "simulation_count": 1,
            "passed_count": 0,
            "executed_negative_count": 0,
            "runner_failed_count": 1,
            "status": "runner_failed",
        }
    ]
    simulation_path = repeat / "simulation_results.jsonl"
    protocol_path = repeat / "protocol_results.jsonl"
    simulation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in simulations),
        encoding="utf-8",
    )
    protocol_path.write_text(
        "".join(json.dumps(row) + "\n" for row in protocols),
        encoding="utf-8",
    )
    (repeat / "summary.json").write_text(
        json.dumps(
            {
                "complete": False,
                "execution_complete": False,
                "cross_platform_all_passed": False,
                "counts": {
                    "protocol_count": 1,
                    "simulation_count": 1,
                    "simulation_passed_count": 0,
                    "executed_negative_count": 0,
                    "runner_failed_count": 1,
                },
                "output_checksums": {
                    "simulation_results.jsonl": _sha256(simulation_path),
                    "protocol_results.jsonl": _sha256(protocol_path),
                },
            }
        ),
        encoding="utf-8",
    )

    summary, indexed = _read_linux_repeat_index(repeat)

    assert summary["execution_complete"] is False
    assert indexed["protocol-one"]["status"] == "runner_failed"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
