"""Small source/input checks only; no compiler, solver, fitting or GPU."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pbpk_kb.blueprints import experiment_cases as cases
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.human_model_snapshot import build_human_model_snapshot


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def real_case() -> dict:
    original_open, original_connect = Path.open, cases.sqlite3.connect
    queries = []
    opened = []

    def guarded_open(path: Path, *args, **kwargs):
        assert path.name not in {"observations.jsonl", "locked_validation_package.json"}
        assert path.suffix not in {".pkml", ".csv"}
        assert "-Results" not in path.name
        mode = args[0] if args else kwargs.get("mode", "r")
        assert not any(flag in mode for flag in "wax+")
        opened.append(str(path))
        return original_open(path, *args, **kwargs)

    class SelectedRepositoryOnly:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, query, parameters):
            assert "WHERE d.Id=?" in query
            assert parameters == ("6fUaQcC6TkG1PU6ROd4zQA",)
            queries.append(parameters)
            return self.connection.execute(query, parameters)

        def close(self):
            self.connection.close()

    def guarded_connect(database, **kwargs):
        if database.endswith("PKSimDB.sqlite?mode=ro"):
            # The registered standard-human provider separately verifies its
            # source DB, which contains rules/defaults rather than study curves.
            return original_connect(database, **kwargs)
        assert database.endswith("?mode=ro&immutable=1")
        assert kwargs == {"uri": True}
        return SelectedRepositoryOnly(original_connect(database, **kwargs))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", guarded_open)
        patch.setattr(cases.sqlite3, "connect", guarded_connect)
        result = cases.make_caffeine_experiment_case(ROOT)
    assert queries == [("6fUaQcC6TkG1PU6ROd4zQA",)]
    assert any("identification_config.json" in name for name in opened)
    return result


def test_study_split_is_joint_iv_po_not_separate_studies(real_case):
    assert real_case["purpose"] == "historical_development_reconstruction"
    scenarios = real_case["scenarios"]
    assert [s["observation_count"] for s in scenarios] == [17, 11, 7]
    assert [s["role"] for s in scenarios] == ["calibration", "calibration", "selection"]
    assert scenarios[0]["study_group_id"] == scenarios[1]["study_group_id"]
    assert scenarios[2]["study_group_id"] != scenarios[0]["study_group_id"]
    assert real_case["study_counts"] == {"calibration": 1, "selection": 1, "test_loaded": 0}
    assert real_case["observation_counts"] == {"calibration": 28, "selection": 7, "test_loaded": 0}
    cal1, cal2, selection = [s["source_audit"]["study_reference"] for s in scenarios]
    assert cal1 == cal2
    assert cal1["source_checksum"] != selection["source_checksum"]
    assert cal1["cohort_ids"] != selection["cohort_ids"]
    source_containers = []
    for scenario in scenarios:
        governance = scenario["source_audit"]["source_governance"]
        assert governance["original_container_shared"]
        assert not governance["derived_export_establishes_independence"]
        assert not governance["blinded_independent_test"]
        source_containers.append(governance["original_container"])
    assert source_containers[0] == source_containers[1] == source_containers[2]


def test_selection_and_test_are_not_calibration_inputs(real_case):
    assert [s["role"] for s in real_case["design_request"]["studies"]] == ["calibration"]
    assert real_case["parameter_search"]["dataset_ids"] == [
        "blanchard_1983_iv_5mgkg", "blanchard_1983_oral_5mgkg"]
    selection_ids = {p["observation_id"] for p in real_case["scenarios"][2]["observations"]}
    assert not any(identifier in json.dumps(real_case["design_request"]) for identifier in selection_ids)
    assert not any(identifier in json.dumps(real_case["parameter_search"]) for identifier in selection_ids)
    scenario_studies = {s["study_group_id"] for s in real_case["scenarios"]}
    assert scenario_studies.isdisjoint(real_case["forbidden_study_group_ids"])
    assert len(real_case["forbidden_study_group_ids"]) == 3


def test_objective_metadata_matches_study_and_curve_equal_runtime(real_case):
    search = real_case["parameter_search"]
    assert search["objective"] == "study_curve_equal_log_rmse"
    assert search["aggregation"] == "sqrt(mean_study(mean_curve(mean_point(log_error_squared))))"
    assert search["gmfe_aggregation"] == "exp(mean_study(mean_curve(mean_point(abs(log(Cpred/Cobs))))))"


def test_source_backed_dose_infusion_and_demographics(real_case):
    iv, po, selection = [s["design_request"]["instance"] for s in real_case["scenarios"]]
    for instance, dose, duration in ((iv, 5, 30), (selection, 3, 10)):
        protocol = instance["protocol"]
        params = {p["Name"]: p for p in protocol["Parameters"]}
        assert protocol["ApplicationType"] == "Intravenous"
        assert params["InputDose"] == {"Name": "InputDose", "Value": dose, "Unit": "mg/kg"}
        assert params["Infusion time"] == {"Name": "Infusion time", "Value": duration, "Unit": "min"}
        assert instance["formulation"] is None
        assert "Table S3" in instance["protocol_source"]["source_location"]
    assert iv["individual"]["Seed"] == po["individual"]["Seed"]
    assert iv["individual"]["OriginData"] == po["individual"]["OriginData"]
    assert iv["generated_demographics"] == po["generated_demographics"] == ["Weight", "Height"]
    assert selection["generated_demographics"] == ["Height"]
    assert selection["individual"]["OriginData"]["Weight"] == {"Value": 74, "Unit": "kg"}
    assert selection["individual"]["OriginData"]["Age"] == {"Value": 26, "Unit": "year(s)"}


def test_all_scenarios_build_static_native_snapshots_without_old_structure(real_case):
    before = copy.deepcopy(real_case)
    for scenario in real_case["scenarios"]:
        request = BlueprintDesignRequest.model_validate(scenario["design_request"])
        assert request.base is None
        assert request.human_model.registry_id == "pksim-human-standard-small-molecule"
        instance = request.instance.model_dump(mode="json")
        instance.pop("population_context")
        package = build_human_model_snapshot(**instance, expected_compound_name="Caffeine")
        snapshot = package["snapshot"]
        assert len(snapshot["Simulations"]) == 1
        simulation = snapshot["Simulations"][0]
        assert simulation["Model"] == "4Comp"
        assert simulation["HasResults"] is False
        assert simulation["OutputSelections"] == [scenario["target_output"]]
        assert "Parameters" not in simulation and "ModelProperties" not in simulation
        assert "ObservedData" not in snapshot and "ParameterIdentifications" not in snapshot
        assert package["manifest"]["execution"] == {"solver_calls": 0, "llm_calls": 0, "gpu_jobs": 0}
    assert real_case == before


def test_point_units_provenance_and_source_discrepancies_are_retained(real_case):
    for scenario in real_case["scenarios"]:
        points = scenario["observations"]
        assert scenario["unit"] == "umol/l" and scenario["time_unit"] == "min"
        assert len(points) == scenario["observation_count"]
        assert all(p["unit"] == "umol/l" and p["statistic"] == "arithmetic_mean" for p in points)
        assert all(p["provenance"] for p in points)
        assert scenario["checksum"] == cases._canonical_checksum(points)
        assert scenario["source_audit"]["discrepancy_codes"]
    points = real_case["scenarios"][2]["observations"]
    assert len(points) == 7
    assert all(p["extraction"]["sqlite_content_id"] == 282 for p in points)
    assert all(p["extraction"]["source_time_unit"] == "h" for p in points)
    assert all(p["extraction"]["source_concentration_unit"] == "umol/l" for p in points)
    assert {p["extraction"]["row_index"] for p in points} == set(range(7))


def test_no_old_fitted_optima_or_scientific_acceptance(real_case):
    clearance, permeability = real_case["parameter_search"]["parameters"]
    assert clearance["initial_value"] == pytest.approx(0.10754414957114701)
    assert permeability["initial_value"] == pytest.approx(1e-5)
    assert clearance["lower_bound"] == 0
    assert not real_case["parameter_search"]["initial_values_are_optimal"]
    assert not real_case["parameter_search"]["identifiability_verified"]
    assert real_case["compile_status"] == real_case["simulation_status"] == "not_run"
    assert real_case["scientific_status"] == "not_evaluated"


def test_registration_is_idempotent_and_refuses_tampering(tmp_path, monkeypatch):
    doc = {"study": "example", "observations": [1]}
    monkeypatch.setattr(cases, "_study_documents", lambda root: ({"one": doc}, {}, {}))
    first = cases.prepare_caffeine_experiment_inputs(tmp_path)
    assert cases.prepare_caffeine_experiment_inputs(tmp_path) == first
    relative = next(iter(first))
    path = tmp_path / relative
    path.write_text('{"changed":true}')
    with pytest.raises(cases.DevelopmentCaseError, match="checksum mismatch"):
        cases.prepare_caffeine_experiment_inputs(tmp_path)
    assert path.read_text() == '{"changed":true}'


def test_registration_rejects_external_directory_symlink(tmp_path, monkeypatch):
    root, outside = tmp_path / "project", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    parent = root / Path(cases._DIRECTORY).parent
    parent.mkdir(parents=True)
    (root / cases._DIRECTORY).symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(cases, "_study_documents", lambda root: ({"one": {}}, {}, {}))
    with pytest.raises(cases.DevelopmentCaseError, match="escapes"):
        cases.prepare_caffeine_experiment_inputs(root)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_invalid_point_cannot_be_silently_omitted(bad):
    point = {"observation_id": "one", "time_min": 1, "concentration_umol_per_l": bad}
    with pytest.raises(cases.DevelopmentCaseError, match="finite and positive"):
        cases._check_observations([point], {"observation_ids": ["one"]}, 1)
