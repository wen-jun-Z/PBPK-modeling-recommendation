"""Small source-locked preparation checks; no solver or external score access."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pbpk_kb.blueprints.cohort_cases import (
    _formula_counts, _normalise_protocol, _parameter_specs, prepare_cohort_inputs,
    prepare_fluconazole_endpoint_inputs, qualify_sampled_endpoints,
)
from pbpk_kb.blueprints.development_case import DevelopmentCaseError
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.human_model_snapshot import build_human_model_snapshot


ROOT = Path(__file__).resolve().parents[2]


def test_sampled_endpoint_qualification_keeps_restricted_estimands():
    result = qualify_sampled_endpoints([1.0, 10.0, 100.0], source={
        "source_checksum": "a" * 64, "source_location": "BaseGrid"})
    assert result["auc_evaluable"] and result["cmax_evaluable"]
    assert result["sampling_qualification"]["observed_window_auc"]["start_min"] == 1
    assert not result["sampling_qualification"]["total_auc_qualified"]
    assert not result["sampling_qualification"]["continuous_cmax_qualified"]


@pytest.mark.parametrize("times", [[0., 1.], [1., 1.], [2., 1.], [1., float("nan")], [1.]])
def test_sampling_qualification_does_not_repair_ambiguous_times(times):
    with pytest.raises(DevelopmentCaseError):
        qualify_sampled_endpoints(times, source={"source_checksum": "a" * 64, "source_location": "BaseGrid"})


def test_fluconazole_endpoint_registration_changes_only_metadata(tmp_path):
    original_path = ROOT / "experiments/blueprint_design/development_inputs/cohort_reconstruction_v2/fluconazole/input_package.json"
    original_bytes = original_path.read_bytes()
    old = json.loads(original_bytes)
    info = prepare_fluconazole_endpoint_inputs(ROOT, original_path, tmp_path / "endpoint_inputs")
    new = json.loads((ROOT / info["bundle_path"]).read_text())
    assert info["scenarios_qualified"] == 5
    assert new.pop("endpoint_registration")["prospective_validation"] is False
    for before, after in zip(old["case"]["scenarios"] + old["external_scenarios"],
                             new["case"]["scenarios"] + new["external_scenarios"]):
        assert after.pop("auc_evaluable") is True
        assert after.pop("cmax_evaluable") is True
        after.pop("sampling_qualification")
        assert after == before
    assert new == old
    assert original_path.read_bytes() == original_bytes


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    output = tmp_path_factory.mktemp("cohort_inputs")
    manifest = prepare_cohort_inputs(ROOT, output)
    bundles = {c["compound_name"]: json.loads((ROOT / c["bundle_path"]).read_text()) for c in manifest["cases"]}
    return output, manifest, bundles


def test_real_registry_and_source_hashes(cohort):
    _, manifest, bundles = cohort
    assert manifest["summary"] == {"source_snapshots": 169, "multi_study_label_candidates": 19,
                                    "ready_distinct_drugs": 3, "strict_main_eligible_drugs": 0}
    assert set(bundles) == {"Sufentanil", "Fluconazole", "Moxifloxacin"}
    for entry in manifest["cases"]:
        assert hashlib.sha256((ROOT / entry["bundle_path"]).read_bytes()).hexdigest() == entry["bundle_sha256"]
        assert hashlib.sha256((ROOT / entry["source"]["path"]).read_bytes()).hexdigest() == entry["source"]["sha256"]
        assert entry["tier"] == "reconstruction" and not entry["strict_main_eligible"]


def test_external_values_never_enter_bundles_and_studies_disjoint(cohort):
    _, _, bundles = cohort
    for bundle in bundles.values():
        case = bundle["case"]
        by_role = {role: {s["study_group_id"] for s in case["scenarios"] if s["role"] == role}
                   for role in ("calibration", "selection")}
        external = {s["study_group_id"] for s in bundle["external_scenarios"]}
        assert not by_role["calibration"] & by_role["selection"]
        assert not external & (by_role["calibration"] | by_role["selection"])
        for scenario in bundle["external_scenarios"]:
            assert "observations" not in scenario
            assert "concentration_umol_per_l" not in json.dumps(scenario)
            assert scenario["design_request"]["studies"] == []
            # Hash verification does not deserialize held-out observations.
            ref = scenario["holdout_reference"]
            assert hashlib.sha256((ROOT / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]


def test_same_study_doses_and_source_assumptions_are_explicit(cohort):
    _, _, bundles = cohort
    flu = bundles["Fluconazole"]["case"]
    shiba = [s for s in flu["scenarios"] if s["source_study_id"] == "Shiba 1990"]
    assert len(shiba) == 2 and {s["role"] for s in shiba} == {"calibration"}
    assert len({s["study_group_id"] for s in shiba}) == 1
    processes = flu["design_request"]["instance"]["compound"]["Processes"]
    assert any(p["InternalName"] == "NoncompetitiveInhibition" and p["Molecule"] == "CYP3A4" for p in processes)
    suf = bundles["Sufentanil"]
    population = {s["source_study_id"]: s["design_request"]["requirement"]["population"]
                  for s in suf["case"]["scenarios"] + suf["external_scenarios"]}
    assert population["Bovill 1984"] == population["Taverne 1992"] == "adult_surgical_patients"
    assert population["Willsie 2015"] == "healthy_adult"


def test_all_requests_build_static_snapshot_without_old_structure(cohort):
    _, _, bundles = cohort
    total = 0
    for bundle in bundles.values():
        for scenario in bundle["case"]["scenarios"] + bundle["external_scenarios"]:
            request = BlueprintDesignRequest.model_validate(scenario["design_request"])
            instance = request.instance.model_dump(mode="json", exclude_none=True)
            instance.pop("population_context")
            assert set(instance["individual"]) <= {"Name", "Seed", "OriginData", "ExpressionProfiles"}
            assert "Permeability" not in instance["compound"]
            assert instance["compound"]["CalculationMethods"][0].endswith("PK-Sim Standard")
            build_human_model_snapshot(**instance, expected_compound_name=bundle["case"]["compound_name"])
            total += 1
    assert total == 11


def test_zero_source_bound_and_policy_semantics(cohort):
    _, _, bundles = cohort
    moxi = bundles["Moxifloxacin"]["case"]["parameter_search"]["parameters"]
    clearance = next(p for p in moxi if p["parameter_id"] == "single_process_clearance")
    assert clearance["lower_bound"] == 0
    assert clearance["search_scale"] == "linear"
    assert clearance["sensitivity_scale"] == "bounded_linear"
    suf = bundles["Sufentanil"]["case"]["parameter_search"]["parameters"]
    assert all(p["policy_id"] == "reconstruction-numerical-guards-v1" for p in suf)
    assert all(p["anchor_may_be_historically_fitted"] for p in suf)
    assert suf[0]["lower_bound"] == pytest.approx(.1 * suf[0]["initial_value"])


def test_output_idempotent_and_tampering_refused(cohort):
    output, manifest, _ = cohort
    assert prepare_cohort_inputs(ROOT, output) == manifest
    with pytest.raises(DevelopmentCaseError, match="unique registered"):
        prepare_cohort_inputs(ROOT, output, compounds=["sufentanil", "sufentanil"])
    with pytest.raises(DevelopmentCaseError, match="project-contained"):
        prepare_cohort_inputs(ROOT, ROOT.parent)


def test_unit_normalization_does_not_change_source_or_timing():
    protocol = {"Name": "5ug/kg IV", "DosingInterval": "Single", "ApplicationType": "IntravenousBolus",
                "Parameters": [{"Name": "InputDose", "Value": 5., "Unit": "µg/kg"},
                               {"Name": "Start time", "Value": 0., "Unit": "h"}]}
    source = copy.deepcopy(protocol)
    normal = _normalise_protocol(protocol, "fixture")
    assert protocol == source
    assert normal["Parameters"][0] == {"Name": "InputDose", "Value": .005, "Unit": "mg/kg"}
    assert normal["Parameters"][1] == source["Parameters"][1]
    assert _formula_counts("C13H12F2N6O") == {"F": 2, "Cl": 0, "Br": 0, "I": 0}
    with pytest.raises(DevelopmentCaseError, match="formula"):
        _formula_counts("C6H6.X")
