"""One source audit plus small checks; no compiler, fitting, solver or GPU."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pbpk_kb.blueprints import development_case as case_module
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.human_model_snapshot import build_human_model_snapshot


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def real_case() -> dict:
    # One complete audit against the registered project resources. Guard the
    # actual I/O, not just the returned labels, against opening validation data.
    opened: list[str] = []
    original_open = Path.open

    def checked_open(path: Path, *args, **kwargs):
        assert path.name not in {"observations.jsonl", "locked_validation_package.json"}
        assert path.suffix not in {".pkml", ".csv"}
        assert "-Results" not in path.name
        opened.append(str(path))
        return original_open(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", checked_open)
        result = case_module.make_caffeine_development_case(ROOT)
    assert any("identification_config.json" in path for path in opened)
    assert not any("locked_validation_package" in path for path in opened)
    return result


def test_real_case_exposes_only_the_sourced_oral_calibration(real_case: dict) -> None:
    assert real_case["purpose"] == "historical_development_reconstruction"
    assert len(real_case["source_audit"]) == 13
    assert all(source["verification"] == "passed" for source in real_case["source_audit"])
    calibration = real_case["calibration"]
    assert calibration["dataset_id"] == "blanchard_1983_oral_5mgkg"
    assert calibration["observation_count"] == len(calibration["observations"]) == 11
    assert [point["time_min"] for point in calibration["observations"]] == [15, 30, 45, 60, 120, 180, 240, 360, 540, 720, 900]
    assert calibration["observations"][0]["concentration_umol_per_l"] == pytest.approx(32.69693402058407)
    assert len({point["observation_id"] for point in calibration["observations"]}) == 11
    assert all(len(point["provenance"]) == 2 for point in calibration["observations"])
    assert all("CONTENTS.Id=274;" in point["provenance"][1]["source_location"] for point in calibration["observations"])
    assert {study["role"] for study in real_case["design_request"]["studies"]} == {"calibration"}
    assert len(real_case["design_request"]["studies"]) == 1
    encoded = json.dumps(real_case)
    for forbidden in ("wahllander", "eudract", "pmc5350996", "pmc5724581", "locked_validation_package", "pkml"):
        assert forbidden not in encoded.lower()
    assert real_case["compile_status"] == real_case["simulation_status"] == "not_run"


def test_real_request_builds_from_registered_human_model_and_native_inputs(real_case: dict) -> None:
    original = copy.deepcopy(real_case)
    request = BlueprintDesignRequest.model_validate(real_case["design_request"])
    assert request.base is None
    assert request.human_model.registry_id == "pksim-human-standard-small-molecule"
    instance = request.instance.model_dump(mode="json")
    instance.pop("population_context")
    package = build_human_model_snapshot(**instance, expected_compound_name="Caffeine")
    snapshot = package["snapshot"]
    assert len(snapshot["Simulations"]) == 1
    simulation = snapshot["Simulations"][0]
    assert simulation["Model"] == "4Comp"
    assert simulation["Name"] == "Caffeine_Blanchard_PO_Development"
    assert simulation["HasResults"] is False
    assert "Parameters" not in simulation and "ModelProperties" not in simulation
    assert "ObservedData" not in snapshot and "ParameterIdentifications" not in snapshot
    assert len(snapshot["ExpressionProfiles"]) == 1
    assert "Weight" not in snapshot["Individuals"][0]["OriginData"]
    assert "Height" not in snapshot["Individuals"][0]["OriginData"]
    assert real_case["design_request"]["instance"]["generated_demographics"] == ["Weight", "Height"]
    assert package["manifest"]["execution"] == {"solver_calls": 0, "llm_calls": 0, "gpu_jobs": 0}
    assert real_case == original


def test_units_search_starts_and_zero_clearance_boundary_are_explicit(real_case: dict) -> None:
    instance = real_case["design_request"]["instance"]
    compound = instance["compound"]
    parameters = {item["Name"]: item for item in compound["Parameters"]}
    assert parameters["Molecular weight"]["Value"] == 194.19
    assert all(parameters[atom]["Value"] == 0 for atom in ("F", "Cl", "Br", "I"))
    solubility = compound["Solubility"][0]["Parameters"][0]
    assert solubility["Value"] == 21.6 and solubility["Unit"] == "mg/ml"
    permeability = compound["IntestinalPermeability"][0]["Parameters"][0]
    assert permeability["Value"] == pytest.approx(1e-5 / 6)
    assert permeability["Unit"] == "cm/s"
    search = real_case["parameter_search"]
    clearance, permeability_search = search["parameters"]
    assert clearance["lower_bound"] == 0
    assert clearance["search_scale"] == "linear_nonnegative"
    assert clearance["upper_bound"] == pytest.approx(0.35848049857049)
    assert clearance["initial_value"] == pytest.approx(0.10754414957114701)
    assert [mapping["multiplier"] for mapping in clearance["runtime_mappings"]] == pytest.approx([0.839, 0.121, 0.04])
    assert permeability_search["unit"] == "dm/min"
    assert permeability_search["lower_bound"] == 1e-9
    assert permeability_search["upper_bound"] == 0.01
    assert len(search["clearance_starts"]) == 9
    assert len(search["permeability_candidates"]) == 71
    assert not search["initial_values_are_optimal"]
    assert search["runtime_units_must_be_verified"]
    assert "assumption" in instance["protocol_source"]["source_location"]
    assert all(reference["provenance_kind"] == "model_asset" for reference in instance["compound_sources"].values())


def test_changed_source_is_rejected_before_any_model_load(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "source.json"
    path.write_text('{"changed": true}')
    expected = hashlib.sha256(b'{"changed": false}').hexdigest()
    monkeypatch.setattr(case_module, "_INPUTS", {"snapshot": ("source.json", expected)})
    monkeypatch.setattr(case_module, "load_standard_human_model", lambda root: pytest.fail("Model must not load after source tampering"))
    with pytest.raises(case_module.DevelopmentCaseError, match="checksum mismatch"):
        case_module.make_caffeine_development_case(tmp_path)


@pytest.mark.parametrize("relative", ["../outside.json", "/outside.json", "nested\\source.json"])
def test_outside_source_paths_are_rejected(tmp_path: Path, relative: str) -> None:
    with pytest.raises(case_module.DevelopmentCaseError, match="project-relative"):
        case_module._verify(tmp_path, relative, "a" * 64)


def test_external_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "source.json").symlink_to(outside)
    with pytest.raises(case_module.DevelopmentCaseError, match="outside project"):
        case_module._verify(root, "source.json", hashlib.sha256(b"{}").hexdigest())


def test_duplicate_named_inputs_are_not_silently_selected() -> None:
    with pytest.raises(case_module.DevelopmentCaseError, match="Expected one"):
        case_module._one([{"Name": "Caffeine"}, {"Name": "Caffeine"}], "Name", "Caffeine")
