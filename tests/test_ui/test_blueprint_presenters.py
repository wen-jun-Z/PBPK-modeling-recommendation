from __future__ import annotations

from pbpk_kb.ui.blueprint_presenters import (
    artifact_rows,
    compartment_rows,
    parameter_rows,
    process_rows,
    process_summary,
    public_model_request_defaults,
    simulation_rows,
)


def test_public_model_request_defaults_are_compound_specific() -> None:
    rosuvastatin = public_model_request_defaults("Rosuvastatin")
    assert rosuvastatin["dose"] == 20.0
    assert rosuvastatin["dose_unit"] == "mg"
    assert rosuvastatin["regimen_mode"] == "multiple"
    assert rosuvastatin["formulation"] == "Tablet (Rosuvastatin)"

    caffeine = public_model_request_defaults("caffeine")
    assert caffeine["dose"] == 5.0
    assert caffeine["dose_unit"] == "mg/kg"
    assert caffeine["regimen_mode"] == "single"
    assert caffeine["frequency"] == "single dose"
    assert caffeine["formulation"] == "aqueous solution"
    assert caffeine["enzymes"] == "CYP1A2"
    assert caffeine["model_structure"] == "whole_body"

    unknown = public_model_request_defaults("warfarin")
    assert unknown["dose"] == 0.0
    assert unknown["dose_unit"] == ""
    assert unknown["route"] == ""
    assert unknown["endpoints"] == []


def test_parameter_rows_separate_fitted_values_and_initial_estimates() -> None:
    rows = parameter_rows(
        [
            {
                "parameter_name": "Molecular weight",
                "semantic_id": "parameter:mw",
                "origin": "observed",
                "value": 194.19,
                "initial_value": None,
                "canonical_unit": "g/mol",
                "blocking": True,
                "provenance_json": [
                    {
                        "source_id": "parameter_manifest",
                        "source_location": "parameters[mw]",
                        "source_checksum": "a" * 64,
                    }
                ],
            },
            {
                "parameter_name": "Intestinal permeability",
                "semantic_id": "parameter:permeability",
                "origin": "estimated",
                "value": None,
                "initial_value": 2.5,
                "canonical_unit": "cm/min",
                "blocking": True,
                "provenance_json": [],
            },
        ]
    )

    assert rows[0]["value"] == 194.19
    assert rows[0]["value_status"] == "fixed"
    assert rows[0]["source_location"] == "parameters[mw]"
    assert rows[1]["value"] == 2.5
    assert rows[1]["value_status"] == "initial estimate"


def test_component_presenters_preserve_type_and_provenance() -> None:
    process = {
        "source_location": "BUNDLE:model.pkml",
        "payload_json": {
            "name": "CYP1A2 metabolism",
            "process_type": "metabolism",
            "enzyme_semantic_id": "enzyme:cyp1a2",
            "input_compartment_id": "liver",
            "output_compartment_id": "liver",
            "equation_template_id": "osp-formula:1",
            "evidence": [
                {"source_location": "PKML:model#reaction=1"}
            ],
        },
    }
    compartment = {
        "source_location": "PKML:model#container=1",
        "payload_json": {
            "name": "Liver",
            "compartment_type": "organ",
            "semantic_id": "organ:liver",
            "parent_compartment_id": "organism",
        },
    }

    assert process_summary([process, process]) == [
        {"process_type": "metabolism", "process_count": 2}
    ]
    assert process_rows([process])[0]["source_location"] == (
        "PKML:model#reaction=1"
    )
    assert compartment_rows([compartment])[0]["name"] == "Liver"


def test_simulation_and_artifact_presenters_keep_checksums() -> None:
    simulation = simulation_rows(
        [
            {
                "scenario_id": "CAL_PO",
                "run_ordinal": 1,
                "status": "succeeded",
                "output_count": 3,
                "result_manifest_sha256": "b" * 64,
            }
        ]
    )[0]
    artifact = artifact_rows(
        [
            {
                "artifact_kind": "model",
                "relative_path": "model.pkml",
                "byte_size": 100,
                "sha256": "c" * 64,
                "source_location": "BUNDLE:model.pkml",
            }
        ]
    )[0]

    assert simulation["outputs"] == 3
    assert simulation["result_manifest_sha256"] == "b" * 64
    assert artifact["sha256"] == "c" * 64
