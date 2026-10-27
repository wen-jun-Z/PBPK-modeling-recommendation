"""Source-bound nonlinear reconstruction checks without solving or held-out reads."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from pbpk_kb.blueprints.alprazolam_cases import ANCHORS, BRANCHES, prepare_alprazolam_inputs
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.development_case import DevelopmentCaseError
from pbpk_kb.blueprints.human_model_snapshot import build_human_model_snapshot
from pbpk_kb.blueprints.native_edits import apply_parameter_values

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    directory = tmp_path_factory.mktemp("alprazolam_source_case")
    result = prepare_alprazolam_inputs(ROOT, directory)
    path = ROOT / result["bundle_path"]
    return directory, result, json.loads(path.read_text())


def test_publication_roles_and_exact_sample_counts(package):
    _, result, bundle = package
    assert result["study_counts"] == {"calibration": 2, "selection": 1, "external": 3}
    assert not result["strict_main_eligible"] and result["tier"] == "reconstruction"
    groups = {}
    totals = {}
    for s in bundle["case"]["scenarios"] + bundle["external_scenarios"]:
        groups.setdefault(s["role"], set()).add(s["study_group_id"])
        totals[s["role"]] = totals.get(s["role"], 0) + s["observation_count"]
    assert totals == {"calibration": 35, "selection": 14, "external": 46}
    assert not groups["calibration"] & groups["selection"]
    assert not (groups["calibration"] | groups["selection"]) & groups["external"]
    assert set(bundle["case"]["forbidden_study_group_ids"]) == groups["external"]


def test_external_values_never_enter_designer_and_files_immutable(package):
    directory, result, bundle = package
    for s in bundle["external_scenarios"]:
        assert "observations" not in s and s["design_request"]["studies"] == []
        assert "concentration_umol_per_l" not in json.dumps(s)
        ref = s["holdout_reference"]
        # Hash only: no held-out observations are decoded by this test.
        assert hashlib.sha256((ROOT / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    assert prepare_alprazolam_inputs(ROOT, directory) == result


def test_all_seven_instances_regenerate_supported_nonlinear_human_model(package):
    _, _, bundle = package
    for s in bundle["case"]["scenarios"] + bundle["external_scenarios"]:
        request = BlueprintDesignRequest.model_validate(s["design_request"])
        instance = request.instance.model_dump(mode="json", exclude_none=True)
        instance.pop("population_context")
        result = build_human_model_snapshot(**instance, expected_compound_name="Alprazolam")
        manifest = result["manifest"]
        assert manifest["execution"]["solver_calls"] == 0
        processes = instance["compound"]["Processes"]
        assert sum(p["InternalName"] == "MetabolizationLiverMicrosomes_MM" for p in processes) == 2
        assert "Parameters" not in instance["individual"]
        assert "Permeability" not in instance["compound"]
        assert "IntestinalPermeability" not in instance["compound"]
        assert s["auc_evaluable"] and s["cmax_evaluable"]
        assert not s["sampling_qualification"]["total_auc_qualified"]


def test_factor_pi_preserves_unequal_branch_anchors_and_does_not_fit_external(package):
    _, _, bundle = package
    spec = bundle["case"]["parameter_search"]["parameters"][0]
    assert spec["source_anchor_values"] == list(ANCHORS)
    assert spec["parameterization"] == "shared_factor" and spec["unit"] == "dimensionless"
    assert [spec[k] for k in ("initial_value", "lower_bound", "upper_bound")] == [1., .1, 10.]
    assert spec["anchor_may_be_historically_fitted"]
    original = bundle["case"]["scenarios"][0]["design_request"]
    unchanged = copy.deepcopy(original)
    request = apply_parameter_values(original, [spec], [2.], provenance={
        "provenance_id": "ad8d0862-5c45-4f98-a7e6-e2baebbb1f22", "provenance_kind": "calibration_run",
        "source_id": "test-shared-factor", "source_location": "controlled factor2", "source_checksum": "a" * 64})
    for branch, anchor in zip(BRANCHES, ANCHORS):
        process = next(p for p in request["instance"]["compound"]["Processes"] if p["DataSource"] == branch)
        assert next(p for p in process["Parameters"] if p["Name"] == "kcat")["Value"] == 2 * anchor
    assert original == unchanged


def test_changed_existing_bundle_not_overwritten(tmp_path):
    result = prepare_alprazolam_inputs(ROOT, tmp_path / "changed")
    path = ROOT / result["bundle_path"]
    path.write_text('{"locally_changed":true}')
    with pytest.raises(DevelopmentCaseError, match="overwrite"):
        prepare_alprazolam_inputs(ROOT, tmp_path / "changed")
    assert path.read_text() == '{"locally_changed":true}'
