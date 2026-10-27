from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

from pbpk_kb.graph.model_kg_audit import audit_model_kg
from pbpk_kb.graph.model_kg_builder import build_model_kg
from pbpk_kb.graph.model_kg_merge import merge_model_kgs
from pbpk_kb.graph.model_kg_schema import validate_relation


SHA = "a" * 64
COMMIT = "b" * 40
PROJECT_KEY = "c" * 32
JOB_ID = "osp-exec-" + "d" * 32


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    )
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        path.write_text(payload, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_kg_relation_schema_rejects_invalid_domain() -> None:
    assert (
        validate_relation(
            "HAS_VERSION",
            "ModelVersion",
            "PBPKModel",
        )
        is not None
    )
    assert (
        validate_relation(
            "HAS_PRODUCT",
            "ModelComponent",
            "Enzyme",
        )
        is None
    )


def test_model_kg_build_and_independent_audit(tmp_path: Path) -> None:
    model_versions = tmp_path / "model_versions"
    model_versions.mkdir()
    source = {
        "relative_path": "projects/p/exports/s0001/simulation.pkml",
        "sha256": SHA,
        "source_location": (
            f"OSP_EXECUTION_RESULT:{JOB_ID}/projects/p/"
            "exports/s0001/simulation.pkml#element_path=/Simulation[1]"
        ),
    }
    model_version = {
        "model_id": "osp-pbpk-model:test",
        "model_version_id": "osp-pbpk-model-version:test-v1",
        "model_family": "TestDrug",
        "project_key": PROJECT_KEY,
        "project_relative_path": "TestDrug/TestDrug.pksim5",
        "project_sha256": SHA,
        "source_id": "osp_pbpk_model_library",
        "repository_url": "https://example.test/models.git",
        "repository_commit": COMMIT,
        "software": "PK-Sim",
        "software_version": "12.3.173",
        "license": {
            "status": "unresolved",
            "spdx_id": None,
            "source_location": None,
            "source_checksum": None,
            "reason_code": "LICENSE_FILE_MISSING",
        },
        "static_parse_status": "registered_not_executable",
        "execution_probe_status": "passed",
        "pkml_parse_status": "parsed",
        "reusable_candidate": False,
        "compounds": ["TestDrug"],
        "metabolites": ["TestMetabolite"],
        "species": ["Human"],
        "populations": ["European adult"],
        "routes": ["Oral"],
        "formulations": ["Tablet"],
        "enzymes": ["CYP3A4"],
        "transporters": [],
        "compartments": ["Liver"],
        "process_types": ["metabolism"],
        "output_names": ["Plasma concentration"],
        "simulation_count": 1,
        "parameter_record_count": 2,
        "component_record_count": 1,
        "process_record_count": 1,
        "report_count": 0,
        "dataset_reference_count": 0,
        "validation_metric_count": 0,
        "assets": [
            {
                "asset_kind": "pksim_project",
                "relative_path": "TestDrug/TestDrug.pksim5",
                "sha256": SHA,
                "source_location": (
                    f"GITHUB_BLOB:{COMMIT}/TestDrug/TestDrug.pksim5"
                ),
            }
        ],
        "report_paths": [],
        "blocking_reasons": ["LICENSE_FILE_MISSING"],
    }
    _write_jsonl(
        model_versions / "model_versions.jsonl",
        [model_version],
    )
    _write_jsonl(
        model_versions / "simulations.jsonl",
        [
            {
                "project_key": PROJECT_KEY,
                "simulation_id": "simulation-1",
                "simulation_name": "Oral study",
                "embedded_simulation_id": "embedded-1",
                "pkml_source": source,
                "solver": "CVODES",
                "parse_status": "parsed",
                "csv_results": [
                    {
                        "result_kind": "timeseries",
                        "non_finite_value_count": 0,
                    }
                ],
                "parameter_count": 2,
                "component_count": 1,
                "process_count": 1,
                "administration_count": 1,
                "routes": ["Oral"],
                "formulations": ["Tablet"],
                "populations": ["European adult"],
                "species": ["Human"],
            }
        ],
    )
    _write_jsonl(
        model_versions / "components.jsonl.gz",
        [
            {
                "project_key": PROJECT_KEY,
                "simulation_id": "simulation-1",
                "component_type": "compound",
                "component_id": "compound-1",
                "name": "TestDrug",
                "element_type": "Compound",
                "description": None,
                "dimension": None,
                "display_unit": None,
                "source": source,
            }
        ],
    )
    _write_jsonl(
        model_versions / "processes.jsonl.gz",
        [
            {
                "project_key": PROJECT_KEY,
                "simulation_id": "simulation-1",
                "process_id": "process-1",
                "name": "CYP3A4 metabolism",
                "process_type": "metabolism",
                "element_type": "ReactionBuilder",
                "description": None,
                "dimension": "Amount per time",
                "formula_id": None,
                "educts": ["TestDrug"],
                "products": ["TestMetabolite"],
                "modifiers": ["CYP3A4"],
                "educt_identifiers": ["compound-1"],
                "product_identifiers": ["metabolite-1"],
                "modifier_identifiers": [
                    "enzyme-1",
                    "Hydroxy-Itraconazole",
                    "LslFaZsuYUqJvhMBiG-hSg",
                ],
                "educt_participants": [
                    {
                        "identifier": "compound-1",
                        "name": "TestDrug",
                        "component_type": "compound",
                        "resolution_status": "resolved_by_component_id",
                    }
                ],
                "product_participants": [
                    {
                        "identifier": "metabolite-1",
                        "name": "TestMetabolite",
                        "component_type": "metabolite",
                        "resolution_status": "resolved_by_component_id",
                    }
                ],
                "modifier_participants": [
                    {
                        "identifier": "enzyme-1",
                        "name": "CYP3A4",
                        "component_type": "enzyme",
                        "resolution_status": "resolved_by_component_id",
                    },
                    {
                        "identifier": "Hydroxy-Itraconazole",
                        "name": "Hydroxy-Itraconazole",
                        "component_type": "compound",
                        "resolution_status": "resolved_by_component_name",
                    },
                    {
                        "identifier": "LslFaZsuYUqJvhMBiG-hSg",
                        "name": None,
                        "component_type": None,
                        "resolution_status": "unresolved",
                    },
                ],
                "unresolved_partner_identifiers": [
                    "LslFaZsuYUqJvhMBiG-hSg"
                ],
                "source": source,
            }
        ],
    )
    _write_jsonl(
        model_versions / "parameters.jsonl.gz",
        [
            {
                "project_key": PROJECT_KEY,
                "simulation_id": "simulation-1",
                "parameter_id": "parameter-1",
                "name": "Molecular weight",
                "scope_path": "/Simulation[opaque123]/Compound[opaque456]",
                "value": 300.0,
                "display_unit": "g/mol",
                "dimension": "Molecular weight",
                "formula_id": None,
                "value_origin_source": "Publication",
                "value_origin_description": "Table 1",
                "element_type": "Parameter",
                "quantity_type": None,
                "is_default": False,
                "source": source,
            },
            {
                "project_key": PROJECT_KEY,
                "simulation_id": "simulation-1",
                "parameter_id": "parameter-2",
                "name": "Molecular weight",
                "scope_path": (
                    "/Simulation[different123]/Compound[different456]"
                ),
                "value": 301.0,
                "display_unit": "g/mol",
                "dimension": "Molecular weight",
                "formula_id": "formula-2",
                "value_origin_source": "Database",
                "value_origin_description": "Variant value",
                "element_type": "Parameter",
                "quantity_type": None,
                "is_default": True,
                "source": source,
            },
        ],
    )
    _write_jsonl(
        model_versions / "administrations.jsonl",
        [
            {
                "project_key": PROJECT_KEY,
                "simulation_id": "simulation-1",
                "administration_id": "administration-1",
                "name": "Oral dose",
                "molecule_name": "TestDrug",
                "route": "Oral",
                "formulation": "Tablet",
                "source": source,
            }
        ],
    )
    _write_jsonl(model_versions / "reports.jsonl.gz", [])
    summary = {
        "build_schema_version": "1.0.0",
        "generated_at": "2026-07-30T00:00:00Z",
        "job_id": JOB_ID,
        "source_id": "osp_pbpk_model_library",
        "repository_commit": COMMIT,
        "execution_result_sha256": SHA,
        "execution_validation_sha256": SHA,
        "project_count": 1,
        "simulation_count": 1,
        "parsed_simulation_count": 1,
        "failed_simulation_count": 0,
        "model_version_count": 1,
        "reusable_candidate_count": 0,
        "parameter_record_count": 2,
        "component_record_count": 1,
        "process_record_count": 1,
        "output_checksums": {},
        "gate1_model_version_acceptance": True,
        "blocking_reasons": ["LICENSE_FILE_MISSING"],
    }
    _write_json(model_versions / "summary.json", summary)

    plan = {
        "plan_schema_version": "1.0.0",
        "job_id": JOB_ID,
        "created_at": "2026-07-30T00:00:00Z",
        "source_id": "osp_pbpk_model_library",
        "repository_url": "https://example.test/models.git",
        "repository_release_tag": "v12.3.1",
        "repository_commit": COMMIT,
        "inventory_run_id": "test",
        "inventory_summary_sha256": SHA,
        "runtime": {
            "suite_release_name": "OSP Suite 12.3",
            "suite_version": "12.3",
            "suite_installer": {
                "name": "installer",
                "url": "https://example.test/installer.exe",
                "sha256": SHA,
            },
            "pksim_version": "12.3.173",
            "pksim_portable": {
                "name": "PKSim",
                "url": "https://example.test/pksim.zip",
                "sha256": SHA,
            },
            "pksim_cli_relative_path": "PKSim.CLI.exe",
            "ospsuite_r_version": "12.3",
            "ospsuite_r_linux_binary": {
                "name": "ospsuite-r",
                "url": "https://example.test/ospsuite-r.tar.gz",
                "sha256": SHA,
            },
        },
        "export_flags": ["-k", "-x", "-c", "-r"],
        "command_timeout_seconds": 3600,
        "required_export_extensions": [".pkml", ".xml", ".csv"],
        "projects": [
            {
                "project_key": PROJECT_KEY,
                "model_family": "TestDrug",
                "relative_path": "TestDrug/TestDrug.pksim5",
                "project_sha256": SHA,
                "project_name": "TestDrug",
                "project_version": 80,
                "simulations": [
                    {
                        "simulation_id": "simulation-1",
                        "name": "Oral study",
                        "simulation_mode": "Individual",
                    }
                ],
            }
        ],
    }
    plan_path = tmp_path / "execution_plan.json"
    _write_json(plan_path, plan)
    gate1_audit = tmp_path / "gate1_audit.json"
    _write_json(
        gate1_audit,
        {
            "accepted": True,
            "input_checksums": {
                "model_version_summary": _sha256(
                    model_versions / "summary.json"
                )
            },
        },
    )
    graph_directory = tmp_path / "model_kg"

    build_summary = build_model_kg(
        model_version_directory=model_versions,
        gate1_audit_path=gate1_audit,
        plan_path=plan_path,
        output_directory=graph_directory,
    )
    audit = audit_model_kg(graph_directory)

    assert build_summary["model_count"] == 1
    assert build_summary["model_version_count"] == 1
    assert build_summary["node_count"] > 10
    assert build_summary["edge_count"] > 10
    assert audit["accepted"] is True
    assert audit["counts"]["isolated_scientific_facts"] == 0
    assert audit["counts"]["unresolved_process_partner_unique"] == 1
    connection = sqlite3.connect(graph_directory / "model_kg.sqlite")
    parameter_count, parameter_occurrences = connection.execute(
        """
        SELECT COUNT(*), SUM(occurrence_count)
        FROM nodes WHERE node_type = 'ModelParameter'
        """
    ).fetchone()
    unresolved_entity_count = connection.execute(
        """
        SELECT COUNT(*) FROM nodes
        WHERE node_type IN ('Compound', 'Metabolite', 'Enzyme', 'Transporter')
          AND name = 'LslFaZsuYUqJvhMBiG-hSg'
        """
    ).fetchone()[0]
    scientific_name_count = connection.execute(
        """
        SELECT COUNT(*) FROM nodes
        WHERE node_type = 'Compound'
          AND name = 'Hydroxy-Itraconazole'
        """
    ).fetchone()[0]
    connection.close()
    assert parameter_count == 1
    assert parameter_occurrences == 2
    assert unresolved_entity_count == 0
    assert scientific_name_count == 1

    graph_audit_path = tmp_path / "model_kg_audit.json"
    _write_json(graph_audit_path, audit)
    merged_directory = tmp_path / "model_kg_merged"
    merged_summary = merge_model_kgs(
        graph_directories=[graph_directory, graph_directory],
        audit_paths=[graph_audit_path, graph_audit_path],
        output_directory=merged_directory,
    )
    merged_audit = audit_model_kg(merged_directory)

    assert merged_summary["model_count"] == 1
    assert merged_summary["merge_counts"]["duplicate_nodes_merged"] > 0
    assert merged_summary["input_record_counts"]["parameters"] == 4
    assert merged_audit["accepted"] is True
    assert merged_audit["counts"]["parameter_fact_occurrences"] == 4
