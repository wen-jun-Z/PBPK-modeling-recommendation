from __future__ import annotations

import json
from pathlib import Path

import pytest

from pbpk_kb.recommendation.model_matching.perturbation import (
    _absolute_delta_in_unit,
    _canonical_decimal_float,
    _derive_snapshot,
    _perturbation_values,
    _variable_eligibility,
)
from pbpk_kb.recommendation.model_matching.perturbation_evaluation import (
    _curve_metrics,
    evaluate_ranking_continuity,
)


def test_perturbation_values_are_positive_unique_and_domain_labeled() -> None:
    rows = _perturbation_values(
        base_value=40.0,
        unit="mg",
        compatible_doses=[40.0, 50.0],
    )
    by_type = {row["perturbation_type"]: row for row in rows}

    assert by_type["absolute_small_decrease"]["new_value"] == pytest.approx(39.9)
    assert by_type["relative_minus_1pct"]["new_value"] == pytest.approx(39.6)
    assert by_type["relative_plus_1pct"]["new_value"] == pytest.approx(40.4)
    assert by_type["validated_interval_midpoint"]["new_value"] == pytest.approx(45.0)
    assert by_type["validated_interval_midpoint"]["domain_status"] == "interpolation"
    assert len({row["new_value"] for row in rows}) == len(rows)


def test_absolute_delta_preserves_mass_basis() -> None:
    assert _absolute_delta_in_unit("mg") == pytest.approx(0.1)
    assert _absolute_delta_in_unit("ug") == pytest.approx(100.0)
    assert _absolute_delta_in_unit("mg/kg") == pytest.approx(0.001)


def test_generated_parameter_values_remove_binary_float_tails() -> None:
    assert _canonical_decimal_float(662.8149999999999) == 662.815
    assert _canonical_decimal_float(0.043429999999999996) == 0.04343
    assert _canonical_decimal_float(5.9399999999999995) == 5.94


def test_derived_snapshot_removes_project_dependencies_and_rewrites_paths() -> None:
    protocol = {
        "Name": "PO 40 mg",
        "Parameters": [{"Name": "InputDose", "Value": 40.0, "Unit": "mg"}],
    }
    simulation = {
        "Name": "verification",
        "ObservedData": ["dataset"],
        "OutputMappings": ["mapping"],
        "HasResults": True,
        "IndividualAnalyses": ["analysis"],
        "Compounds": [
            {"Name": "Test drug", "Protocol": {"Name": "PO 40 mg"}}
        ],
        "Parameters": [
            {"Path": "Events|PO 40 mg|Application_1|Dose", "Value": 40.0},
            {"Path": "Organism|Liver|Volume", "Value": 1.0},
        ],
    }
    snapshot = {
        "Protocols": [protocol],
        "Simulations": [simulation, {"Name": "unrelated"}],
        "ObservedData": [{"Name": "dataset"}],
        "ObservedDataClassifications": [{"Name": "class"}],
        "SimulationClassifications": [{"Name": "class"}],
        "ParameterIdentifications": [{"Name": "fit"}],
        "ParameterIdentificationClassifications": [{"Name": "fit-class"}],
        "SensitivityAnalyses": [{"Name": "sensitivity"}],
    }

    derived, records = _derive_snapshot(
        snapshot=snapshot,
        base_simulation=simulation,
        descriptor={"protocol": protocol},
        primary="Test drug",
        group_key="testgroup",
        perturbations=[
            {
                "perturbation_type": "relative_plus_1pct",
                "new_value": 40.4,
                "unit": "mg",
                "domain_status": "local",
            }
        ],
    )

    assert len(derived["Simulations"]) == 2
    assert records[0]["dose_ratio"] == pytest.approx(1.01)
    for key in (
        "ObservedData",
        "ObservedDataClassifications",
        "SimulationClassifications",
        "ParameterIdentifications",
        "ParameterIdentificationClassifications",
        "SensitivityAnalyses",
    ):
        assert key not in derived
    for row in derived["Simulations"]:
        assert "ObservedData" not in row
        assert "OutputMappings" not in row
        assert "IndividualAnalyses" not in row
        assert row["HasResults"] is False
    perturbed = derived["Simulations"][1]
    assert perturbed["Compounds"][0]["Protocol"]["Name"] == (
        "Perturbation testgroup-01"
    )
    assert perturbed["Parameters"][0]["Path"] == (
        "Events|Perturbation testgroup-01|Application_1|Dose"
    )
    assert perturbed["Parameters"][1]["Path"] == "Organism|Liver|Volume"


def test_module_declares_no_scientific_grade() -> None:
    source = Path(
        "pbpk_kb/recommendation/model_matching/perturbation.py"
    ).read_text(encoding="utf-8")
    assert '"scientific_grade_generated": False' in source
    assert "LicensedSnapshotReuseProtocol.model_validate" in source


def test_variable_eligibility_records_planned_and_blocked_variables() -> None:
    protocol = {
        "Name": "PO 40 mg",
        "Parameters": [
            {"Name": "InputDose", "Value": 40.0, "Unit": "mg"},
            {"Name": "Start time", "Value": 0.0, "Unit": "h"},
        ],
    }
    simulation = {"Name": "verification", "Individual": "Adult"}
    snapshot = {
        "Protocols": [protocol],
        "Individuals": [
            {
                "Name": "Adult",
                "OriginData": {
                    "Age": {"Value": 30.0, "Unit": "year(s)"}
                },
            }
        ],
    }
    rows = _variable_eligibility(
        snapshot=snapshot,
        simulation=simulation,
        descriptor={"protocol": protocol, "dose_unit": "mg"},
        perturbation_group_id="perturbation-group:test",
        active_moiety="Test drug",
        perturbations=[{"perturbation_type": "relative_plus_1pct"}],
    )
    by_variable = {row["variable"]: row for row in rows}

    assert set(by_variable) == {
        "dose",
        "age",
        "body_weight",
        "organ_function",
        "administration_time",
    }
    assert by_variable["dose"]["status"] == "planned"
    assert by_variable["age"]["reason_code"] == "PUBLIC_VALIDATION_BOUNDARY_UNAVAILABLE"
    assert by_variable["body_weight"]["reason_code"] == "UNIQUE_STRUCTURED_PARAMETER_PATH_UNAVAILABLE"
    assert by_variable["administration_time"]["structured_path"].endswith("/1/Value")


def test_curve_metrics_measure_dose_normalized_continuity() -> None:
    metrics = _curve_metrics(
        base_times=[0.0, 1.0, 2.0],
        base_values=[0.0, 2.0, 1.0],
        perturbed_times=[0.0, 1.0, 2.0],
        perturbed_values=[0.0, 2.02, 1.01],
        dose_ratio=1.01,
    )
    assert metrics["auc_ratio"] == pytest.approx(1.01)
    assert metrics["cmax_ratio"] == pytest.approx(1.01)
    assert metrics["dose_normalized_curve_nrmse"] == pytest.approx(0.0)


def test_ranking_continuity_reports_small_score_perturbation() -> None:
    base = [
        {"query_id": "base", "candidate_id": "a", "score": 0.9, "reuse_decision": True},
        {"query_id": "base", "candidate_id": "b", "score": 0.8, "reuse_decision": False},
        {"query_id": "base", "candidate_id": "c", "score": 0.2, "reuse_decision": False},
    ]
    perturbed = [
        {"query_id": "pert", "candidate_id": "a", "score": 0.89, "reuse_decision": True},
        {"query_id": "pert", "candidate_id": "b", "score": 0.81, "reuse_decision": False},
        {"query_id": "pert", "candidate_id": "c", "score": 0.21, "reuse_decision": False},
    ]
    result = evaluate_ranking_continuity(
        base_rows=base,
        perturbed_rows=perturbed,
        task_links={"pert": "base"},
        k=2,
    )
    assert result["top1_stability_rate"] == 1.0
    assert result["mean_topk_jaccard"] == 1.0
    assert result["mean_kendall_tau"] == 1.0
    assert result["scientific_grade_generated"] is False
