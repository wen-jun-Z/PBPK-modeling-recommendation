from __future__ import annotations

from copy import deepcopy

from pbpk_kb.blueprints.service import (
    _blueprint_package,
    _compatibility_blockers,
    _scientific_interpretation,
    _select_compatible_blueprint,
)
from pbpk_kb.recommendation.schemas import RecommendationRequest


def test_blueprint_package_requires_one_checksum_bundle(tmp_path) -> None:
    assert _blueprint_package(tmp_path) is None

    package = tmp_path / "compound_blueprint_bundle.zip"
    package.write_bytes(b"bundle")
    assert _blueprint_package(tmp_path) == package.resolve()

    (tmp_path / "second_blueprint_bundle.zip").write_bytes(b"other")
    assert _blueprint_package(tmp_path) is None


def _registered_row() -> dict:
    return {
        "template_json": {
            "supported_objectives": ["public_model_reuse", "base_pk"],
            "supported_routes": ["Oral"],
            "supported_process_types": [
                "absorption",
                "distribution",
                "metabolism",
                "renal_excretion",
            ],
            "model_structure": "whole_body",
        },
        "blueprint_json": {
            "species": "Human",
            "compartments": [
                {"name": "Liver", "semantic_id": "organ:liver"},
                {"name": "Kidney", "semantic_id": "organ:kidney"},
            ],
            "processes": [],
            "dosing_regimens": [
                {
                    "dose": 5.0,
                    "dose_unit": "mg/kg",
                    "frequency": "single dose",
                    "formulation": "aqueous dissolved solution",
                }
            ],
            "simulation_scenarios": [
                {"output_paths": ["PeripheralVenousBlood|Caffeine|Plasma"]}
            ],
            "validation_plan": {
                "endpoints": [
                    "concentration_time_profile",
                    "fraction_absorbed",
                ]
            },
        },
        "process_components": [
            {
                "payload_json": {
                    "process_type": process,
                    "enzyme_semantic_id": (
                        "enzyme:CYP1A2" if process == "metabolism" else None
                    ),
                }
            }
            for process in (
                "absorption",
                "distribution",
                "metabolism",
                "renal_excretion",
            )
        ],
    }


def test_registered_blueprint_accepts_only_exact_supported_contract() -> None:
    request = RecommendationRequest(
        compound="caffeine",
        objective="public_model_reuse",
        species="human",
        route="oral",
        dose=5,
        dose_unit="mg/kg",
        frequency="single dose",
        single_or_multiple="single",
        formulation="aqueous solution",
        required_enzymes=["CYP1A2"],
        prediction_endpoints=["concentration_time_profile"],
        required_organs=["liver"],
        whole_body_or_reduced_pbpk="whole_body",
    )

    assert _compatibility_blockers(request=request, row=_registered_row()) == []


def test_registered_blueprint_fails_closed_for_unrepresented_context() -> None:
    request = RecommendationRequest(
        compound="caffeine",
        objective="pediatric",
        population="children",
        route="oral",
        dose=10,
        dose_unit="mg",
        frequency="twice daily",
        required_organs=["placenta"],
        prediction_endpoints=["tissue_concentration"],
        whole_body_or_reduced_pbpk="reduced",
    )

    blockers = _compatibility_blockers(request=request, row=_registered_row())
    assert "TEMPLATE_OBJECTIVE_UNSUPPORTED" in blockers
    assert "REQUEST_REQUIRES_TEMPLATE_RECOMPILATION" in blockers
    assert "TEMPLATE_FREQUENCY_UNSUPPORTED" in blockers
    assert "TEMPLATE_POPULATION_UNSUPPORTED" in blockers
    assert "TEMPLATE_ENDPOINT_UNSUPPORTED:TISSUE_CONCENTRATION" in blockers
    assert "TEMPLATE_ORGAN_UNSUPPORTED:PLACENTA" in blockers
    assert "TEMPLATE_MODEL_STRUCTURE_UNSUPPORTED" in blockers


def test_registered_blueprint_requires_complete_explicit_regimen() -> None:
    request = RecommendationRequest(
        compound="caffeine",
        objective="public_model_reuse",
        species="human",
    )

    blockers = _compatibility_blockers(request=request, row=_registered_row())

    assert blockers == [
        "BLUEPRINT_REQUEST_DOSE_MISSING",
        "BLUEPRINT_REQUEST_DOSE_UNIT_MISSING",
        "BLUEPRINT_REQUEST_ENDPOINT_MISSING",
        "BLUEPRINT_REQUEST_FORMULATION_MISSING",
        "BLUEPRINT_REQUEST_FREQUENCY_MISSING",
        "BLUEPRINT_REQUEST_REGIMEN_MODE_MISSING",
        "BLUEPRINT_REQUEST_ROUTE_MISSING",
    ]


def test_blueprint_selection_requires_one_compatible_template() -> None:
    request = RecommendationRequest(
        compound="caffeine",
        objective="public_model_reuse",
        species="human",
        route="oral",
        dose=5,
        dose_unit="mg/kg",
        frequency="single dose",
        single_or_multiple="single",
        formulation="aqueous solution",
        required_enzymes=["CYP1A2"],
        prediction_endpoints=["concentration_time_profile"],
        required_organs=["liver"],
        whole_body_or_reduced_pbpk="whole_body",
    )
    first = _registered_row()
    first.update({"blueprint_id": "blueprint-1", "template_id": "template-1"})
    second = deepcopy(first)
    second.update({"blueprint_id": "blueprint-2", "template_id": "template-2"})

    selected, blockers, audits = _select_compatible_blueprint(
        request=request,
        rows=[first, second],
    )

    assert selected is None
    assert blockers == ["AMBIGUOUS_COMPATIBLE_BLUEPRINT_TEMPLATES"]
    assert len(audits) == 2
    assert all(row["compatible"] for row in audits)


def test_blueprint_selection_checks_all_candidates_before_selection() -> None:
    request = RecommendationRequest(
        compound="caffeine",
        objective="public_model_reuse",
        species="human",
        route="oral",
        dose=5,
        dose_unit="mg/kg",
        frequency="single dose",
        single_or_multiple="single",
        formulation="aqueous solution",
        required_enzymes=["CYP1A2"],
        prediction_endpoints=["concentration_time_profile"],
        required_organs=["liver"],
        whole_body_or_reduced_pbpk="whole_body",
    )
    compatible = _registered_row()
    compatible.update(
        {"blueprint_id": "blueprint-good", "template_id": "template-good"}
    )
    incompatible = deepcopy(compatible)
    incompatible.update(
        {"blueprint_id": "blueprint-bad", "template_id": "template-bad"}
    )
    incompatible["template_json"]["supported_routes"] = ["Intravenous"]

    selected, blockers, audits = _select_compatible_blueprint(
        request=request,
        rows=[incompatible, compatible],
    )

    assert selected is compatible
    assert blockers == []
    assert [row["compatible"] for row in audits] == [False, True]


def test_scientific_interpretation_uses_actual_registered_blockers() -> None:
    message = _scientific_interpretation(
        {
            "registration_allowed": False,
            "blocker_codes": [
                "EXTERNAL_VALIDATION_NOT_RUN",
                "CALIBRATION_NOT_ACCEPTED",
            ],
        }
    )

    assert "CALIBRATION_NOT_ACCEPTED" in message
    assert "EXTERNAL_VALIDATION_NOT_RUN" in message
    assert "absorption structure" not in message
    assert "must not be treated as calibrated" in message


def test_scientific_interpretation_limits_registered_scope() -> None:
    message = _scientific_interpretation(
        {"registration_allowed": True, "blocker_codes": []}
    )

    assert "exact blueprint scope" in message
    assert "does not authorize transfer" in message
