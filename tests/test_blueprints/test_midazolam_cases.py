"""Static source-contract checks; no solver or held-out concentration reads."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.development_case import DevelopmentCaseError
from pbpk_kb.blueprints.human_model_snapshot import build_human_model_snapshot
from pbpk_kb.blueprints.midazolam_cases import ZERO_TIME_STUDIES, prepare_midazolam_inputs
from pbpk_kb.blueprints.native_edits import apply_parameter_values

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    directory = tmp_path_factory.mktemp("midazolam_source_case")
    result = prepare_midazolam_inputs(ROOT, directory)
    return directory, result, json.loads((ROOT / result["bundle_path"]).read_text())


def test_global_publication_groups_roles_and_sample_counts(package):
    _, result, bundle = package
    assert result["study_counts"] == {"calibration": 2, "selection": 2, "external": 4}
    assert result["tier"] == "reconstruction" and not result["strict_main_eligible"]
    groups, totals = {}, {}
    scenarios = bundle["case"]["scenarios"] + bundle["external_scenarios"]
    assert len(scenarios) == 9
    for s in scenarios:
        assert s["study_group_id"] == s["global_study_id"]
        assert s["global_study_id"].startswith("pmid:")
        groups.setdefault(s["role"], set()).add(s["study_group_id"])
        totals[s["role"]] = totals.get(s["role"], 0) + s["observation_count"]
    assert totals == {"calibration": 38, "selection": 30, "external": 49}
    assert not groups["calibration"] & groups["selection"]
    assert not (groups["calibration"] | groups["selection"]) & groups["external"]
    assert set(bundle["case"]["forbidden_study_group_ids"]) == groups["external"]
    assert next(s for s in scenarios if s["source_study_id"] == "Phimmasone 2001")["global_study_id"] == "pmid:11753266"


def test_external_values_are_separate_and_outputs_immutable(package):
    directory, result, bundle = package
    for s in bundle["external_scenarios"]:
        assert "observations" not in s and s["design_request"]["studies"] == []
        assert "concentration_umol_per_l" not in json.dumps(s)
        ref = s["holdout_reference"]
        # Verify bytes only, without decoding held-out concentrations.
        assert hashlib.sha256((ROOT / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    assert prepare_midazolam_inputs(ROOT, directory) == result


def test_all_nine_requests_preserve_separate_mm_processes_and_regenerate_human(package):
    _, _, bundle = package
    for s in bundle["case"]["scenarios"] + bundle["external_scenarios"]:
        request = BlueprintDesignRequest.model_validate(s["design_request"])
        instance = request.instance.model_dump(mode="json", exclude_none=True)
        instance.pop("population_context")
        result = build_human_model_snapshot(**instance, expected_compound_name="Midazolam")
        assert result["manifest"]["execution"]["solver_calls"] == 0
        processes = instance["compound"]["Processes"]
        mm = [p for p in processes if p["InternalName"] == "MetabolizationLiverMicrosomes_MM"]
        assert len(mm) == 2
        assert {p["Molecule"] for p in mm} == {"CYP3A4", "UGT1A4"}
        assert "Parameters" not in instance["individual"]
        assert "Permeability" not in instance["compound"]
        assert "IntestinalPermeability" not in instance["compound"]
        assert s["auc_evaluable"] and s["cmax_evaluable"]
        assert not s["sampling_qualification"]["total_auc_qualified"]


def test_source_lipophilicity_parameter_does_not_change_mm_capacities(package):
    _, _, bundle = package
    specs = bundle["case"]["parameter_search"]["parameters"]
    assert len(specs) == 1
    spec = specs[0]
    assert spec["parameter_id"] == "lipophilicity"
    assert [spec[k] for k in ("initial_value", "lower_bound", "upper_bound")] == [3.1, 2.8, 4.0]
    assert spec["unit"] == "Log Units"
    original = bundle["case"]["scenarios"][0]["design_request"]
    unchanged = copy.deepcopy(original)
    changed = apply_parameter_values(original, specs, [3.2], provenance={
        "provenance_id": "6642b8be-a176-4947-b695-a61c8c7a277e", "provenance_kind": "calibration_run",
        "source_id": "test-lipophilicity", "source_location": "controlled logP3.2", "source_checksum": "a" * 64})
    compound = changed["instance"]["compound"]
    assert compound["Processes"] == original["instance"]["compound"]["Processes"]
    assert next(p for p in compound["Lipophilicity"][0]["Parameters"] if p["Name"] == "Lipophilicity")["Value"] == 3.2
    assert original == unchanged


def test_prespecified_zero_time_exclusions_and_exact_infusion_conversion(package):
    _, _, bundle = package
    qualification = json.loads((ROOT / bundle["source_qualification"]["path"]).read_text())
    assert not qualification["selection_performance_based"]
    assert not qualification["native_execution_performed"]
    assert not qualification["newly_added_to_running_171_jobs"]
    excluded = [r for r in qualification["screening"] if r["reason"] == "unresolved_source_zero_time_observation_semantics"]
    assert {r["source_study_id"] for r in excluded} == ZERO_TIME_STUDIES
    scenarios = bundle["case"]["scenarios"] + bundle["external_scenarios"]
    assert not {s["source_study_id"] for s in scenarios} & ZERO_TIME_STUDIES
    scenario = next(s for s in scenarios if s["source_study_id"] == "Schwagmeier 1998")
    infusion = next(p for p in scenario["design_request"]["instance"]["protocol"]["Parameters"] if p["Name"] == "Infusion time")
    assert infusion["Unit"] == "min" and infusion["Value"] == 0.5


def test_existing_bundle_is_not_overwritten(tmp_path):
    result = prepare_midazolam_inputs(ROOT, tmp_path / "changed")
    path = ROOT / result["bundle_path"]
    path.write_text('{"locally_changed":true}')
    with pytest.raises(DevelopmentCaseError, match="overwrite"):
        prepare_midazolam_inputs(ROOT, tmp_path / "changed")
    assert path.read_text() == '{"locally_changed":true}'
