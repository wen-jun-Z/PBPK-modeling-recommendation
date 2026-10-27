"""Source/request tests with toy unit conversions, no external PK-value reading."""
from pathlib import Path
import copy
import json

import pytest

from pbpk_kb.blueprints import multidrug_cases as cases
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.human_model_snapshot import build_human_model_snapshot


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def manifest():
    # Already registered preparation output contains requests/references only.
    # Do not open evaluator observation JSON while models may still be running.
    path = ROOT / "experiments/blueprint_design/development_inputs/multidrug_fixed_v1/input_manifest.json"
    result = json.loads(path.read_text())
    for case in result["cases"]:
        for source in case["source_audit"]:
            cases._verify(ROOT, source["path"], source["sha256"])
        for scenario in case["scenarios"]:
            reference = scenario["holdout_reference"]
            cases._verify(ROOT, reference["path"], reference["sha256"])
    return result


def test_three_drugs_six_scenarios_have_explicit_qualification(manifest):
    registered = manifest["cases"]
    assert [case["compound_name"] for case in registered] == ["Amikacin", "Vancomycin", "Pregabalin"]
    assert [case["tier"] for case in registered] == ["bottom_up_external", "bottom_up_external", "reconstruction"]
    assert len({case["compound_id"] for case in registered}) == 3
    assert [s["observation_count"] for case in registered for s in case["scenarios"]] == [11, 11, 9, 9, 10, 10]
    for case in registered:
        assert not case["strict_main_eligible"]
        assert case["parameter_search"] == {"parameters": [], "fit_permitted": False, "selection_permitted": False}
        assert len({s["study_group_id"] for s in case["scenarios"]}) == 1
        assert {s["role"] for s in case["scenarios"]} == {"external"}
    assert any(item["compound"] == "Inulin" and "Rat" in item["reason"] for item in manifest["exclusions"])


def test_design_manifest_has_no_external_concentrations_or_old_results(manifest):
    def visit(value):
        if isinstance(value, dict):
            assert not {"observations", "ObservedData", "concentration_umol_per_l", "ParameterIdentifications", "Results"} & value.keys()
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(manifest)
    assert not manifest["external_observations_in_design_payload"]
    assert manifest["solver_calls"] == 0


def test_six_requests_statically_build_new_standard_models(manifest):
    for case in manifest["cases"]:
        for scenario in case["scenarios"]:
            request = BlueprintDesignRequest.model_validate(scenario["design_request"])
            assert request.studies == []
            assert request.human_model.registry_id == "pksim-human-standard-small-molecule"
            instance = request.instance.model_dump(mode="json")
            instance.pop("population_context")
            assert "ExpressionProfiles" not in instance["individual"]
            package = build_human_model_snapshot(**instance, expected_compound_name=request.requirement.name)
            simulation = package["snapshot"]["Simulations"][0]
            assert simulation["Model"] == "4Comp" and simulation["HasResults"] is False
            assert "Parameters" not in simulation and "ModelProperties" not in simulation
            assert simulation["OutputSelections"] == [scenario["target_output"]]
            assert package["manifest"]["execution"] == {"gpu_jobs": 0, "llm_calls": 0, "solver_calls": 0}


def test_source_drug_overrides_are_fixed_and_disclosed(manifest):
    amikacin, vancomycin, pregabalin = manifest["cases"]
    compounds = [case["design_request"]["instance"]["compound"] for case in manifest["cases"]]
    amikacin_parameters = {p["Name"]: p["Value"] for p in compounds[0]["Parameters"]}
    vancomycin_parameters = {p["Name"]: p["Value"] for p in compounds[1]["Parameters"]}
    assert amikacin_parameters["Molecular weight"] == 585.6
    assert amikacin_parameters["Cl"] == 0
    assert vancomycin_parameters["Cl"] == 2
    assert compounds[1]["Lipophilicity"][0]["Parameters"][0]["Value"] == 1.11
    assert compounds[1]["FractionUnbound"][0]["Parameters"][0]["Value"] == .48
    assert compounds[2]["Lipophilicity"][0]["Parameters"][0]["Value"] == -1.6
    permeability = compounds[2]["IntestinalPermeability"][0]["Parameters"][0]
    assert permeability["Value"] == pytest.approx(0.00096644681642 / 60)
    assert permeability["Unit"] == "cm/s"
    assert "fitted" in " ".join(pregabalin["limitations"]).lower()
    assert "588.6" in " ".join(amikacin["limitations"])
    for compound in compounds:
        assert compound["Processes"][0]["InternalName"] == "GlomerularFiltration"
        assert compound["Processes"][0]["Parameters"] == [{"Name": "GFR fraction", "Value": 1.0}]


def _toy_observed(unit="ug/l", times=None, values=None):
    return {"BaseGrid": {"Unit": "h", "Values": [1, 2] if times is None else times},
            "Columns": [{"Unit": unit, "Values": [1, 2] if values is None else values,
                         "DataInfo": {"Origin": "Observation"}}]}


@pytest.mark.parametrize("unit,expected", [("µg/l", .01), ("ng/ml", .01), ("mg/l", 10), ("ug/ml", 10), ("μmol/l", 1)])
def test_external_unit_conversion_uses_molar_identity(unit, expected):
    result = cases._external_observations(_toy_observed(unit), 100)
    assert result[0]["time_min"] == 60
    assert result[0]["concentration_umol_per_l"] == pytest.approx(expected)
    assert result[1]["concentration_umol_per_l"] == pytest.approx(2 * expected)


@pytest.mark.parametrize("observed", [
    _toy_observed("unknown"), _toy_observed(times=[2, 1]),
    _toy_observed(values=[1]), _toy_observed(values=[0, 1]),
    _toy_observed(values=[float("nan"), 1]),
])
def test_ambiguous_or_invalid_external_points_are_not_silently_dropped(observed):
    with pytest.raises(cases.DevelopmentCaseError):
        cases._external_observations(observed, 100)


def test_output_is_idempotent_but_rejects_changed_bytes(tmp_path):
    path = tmp_path / "one.json"
    checksum = cases._write_once(path, {"value": 1})
    assert cases._write_once(path, {"value": 1}) == checksum
    with pytest.raises(cases.DevelopmentCaseError, match="overwrite"):
        cases._write_once(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_outside_output_and_unknown_compound_fail_before_reading_sources(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(cases, "_verify", lambda *a, **k: pytest.fail("Invalid request must fail before source reading"))
    with pytest.raises(cases.DevelopmentCaseError, match="inside"):
        cases.prepare_multidrug_inputs(root, tmp_path / "outside")
    with pytest.raises(cases.DevelopmentCaseError, match="Unknown"):
        cases.prepare_multidrug_inputs(root, root / "inside", compounds=["unregistered"])


def test_changed_identity_source_is_rejected(tmp_path, monkeypatch):
    identity = tmp_path / cases._IDENTITY[0]
    identity.parent.mkdir(parents=True)
    identity.write_text('{"changed":true}')
    monkeypatch.setattr(cases, "load_standard_human_model", lambda root: pytest.fail("Must stop before loading the human model"))
    with pytest.raises(cases.DevelopmentCaseError, match="checksum mismatch"):
        cases.prepare_multidrug_inputs(tmp_path, tmp_path / "inputs")


@pytest.fixture(scope="module")
def alfentanil_package():
    # The package has calibration/selection values, but the final-study values
    # live in another file. Only checksum-read that file; do not deserialize it.
    path = ROOT / "experiments/blueprint_design/development_inputs/alfentanil_multistudy_v1/input_package.json"
    package = json.loads(path.read_text())
    for source in package["case"]["source_audit"]:
        cases._verify(ROOT, source["path"], source["sha256"])
    for scenario in package["external_scenarios"]:
        reference = scenario["holdout_reference"]
        cases._verify(ROOT, reference["path"], reference["sha256"])
    return package


def test_alfentanil_four_studies_have_disjoint_development_roles(alfentanil_package):
    case = alfentanil_package["case"]
    assert case["tier"] == "development"
    assert case["purpose"] == "historical_development_reconstruction"
    assert not case["strict_main_eligible"]
    assert [s["role"] for s in case["scenarios"]] == ["calibration", "calibration", "selection"]
    assert [s["observation_count"] for s in case["scenarios"]] == [14, 13, 16]
    external = alfentanil_package["external_scenarios"]
    assert len(external) == 1 and external[0]["observation_count"] == 15
    assert len({s["study_group_id"] for s in [*case["scenarios"], *external]}) == 4
    assert external[0]["study_group_id"] in case["forbidden_study_group_ids"]
    assert "observations" not in external[0]
    assert "alfentanil_kharasch_2011_iv:row_" not in json.dumps(alfentanil_package)


def test_alfentanil_restarts_from_public_search_config_not_final_fit(alfentanil_package):
    case = alfentanil_package["case"]
    clearance, logp = case["parameter_search"]["parameters"]
    assert (clearance["parameter_id"], clearance["initial_value"], clearance["lower_bound"], clearance["upper_bound"]) == (
        "cyp3a4_intrinsic_clearance", .504, .1, 2)
    assert (logp["parameter_id"], logp["initial_value"], logp["lower_bound"], logp["upper_bound"]) == (
        "lipophilicity", 1.75, 1.3, 2)
    assert (clearance["search_scale"], clearance["sensitivity_scale"]) == ("log", "log")
    assert (logp["search_scale"], logp["sensitivity_scale"]) == ("linear", "bounded_linear")
    assert all(p["bound_semantics"] == "historical_numerical_search_guards" for p in (clearance, logp))
    assert clearance["runtime_mappings"] == [{"runtime_path": "Alfentanil-CYP3A4-1st order CL|Intrinsic clearance", "multiplier": 1}]
    assert logp["runtime_mappings"] == [{"runtime_path": "Alfentanil|Lipophilicity", "multiplier": 1}]
    compound = case["design_request"]["instance"]["compound"]
    assert "Permeability" not in compound and "IntestinalPermeability" not in compound
    process = next(p for p in compound["Processes"] if p["InternalName"] == "MetabolizationIntrinsic_FirstOrder")
    assert process["Parameters"] == [{"Name": "Intrinsic clearance", "Value": .504, "Unit": "l/min"}]
    assert compound["Lipophilicity"][0]["Parameters"][0]["Value"] == 1.75
    assert "0.5272297928" not in json.dumps(case)
    assert "1.8463211883" not in json.dumps(case)


def test_alfentanil_inputs_use_only_unfitted_standard_expression(alfentanil_package):
    for scenario in [*alfentanil_package["case"]["scenarios"], *alfentanil_package["external_scenarios"]]:
        request = BlueprintDesignRequest.model_validate(scenario["design_request"])
        instance = request.instance.model_dump(mode="json")
        profile = instance["expression_profiles"][0]
        assert profile["Molecule"] == "CYP3A4" and profile["Category"] == "Standard European Male for DDI"
        assert next(p for p in profile["Parameters"] if p["Path"] == "CYP3A4|Reference concentration")["Value"] == 4.32
        assert instance["individual"]["ExpressionProfiles"] == ["CYP3A4|Human|Standard European Male for DDI"]
        assert instance["generated_demographics"] == ["Weight", "Height"]
        assert "historical model assumption" in instance["human_source"]["source_location"]
        assert instance["protocol"]["ApplicationType"] == "IntravenousBolus"
        assert instance["protocol"]["Name"] == "PROTO_" + scenario["scenario_id"]
        assert "µg/kg IV bolus" in instance["protocol_source"]["source_location"]
        assert "8.6130459586" not in json.dumps(instance)
