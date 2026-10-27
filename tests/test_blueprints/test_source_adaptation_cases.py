"""Source-pinned pilot membership/reference checks without native execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("source_adaptation_cases", ROOT / "scripts/source_adaptation_cases.py")
cases_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cases_module)


def test_source_cases_preserve_independent_reference_and_repeat_time_origin():
    manifest = cases_module.build_case_manifest(ROOT)
    assert manifest["evaluation_contract"]["case_count"] == 8
    assert manifest["evaluation_contract"]["official_target_simulation_count"] == 8
    assert manifest["evaluation_contract"]["historical_curve_count"] == 9
    assert manifest["evaluation_contract"]["historical_study_count"] == 7
    for case in manifest["cases"]:
        source = cases_module._read_source(case["source"], ROOT)
        source_sim = cases_module._one(source["Simulations"], case["source"]["simulation_name"])
        official = cases_module._one(source["Simulations"], case["engineering_reference"]["simulation_name"])
        reference = cases_module.build_reference_snapshot(case, ROOT)
        target = reference["Simulations"][0]
        assert target["OutputSchema"] == source_sim["OutputSchema"]
        for key in ("Parameters", "Compounds", "Interactions", "Individual", "Events", "Solver", "Model"):
            assert target.get(key) == official.get(key), key
        for key in ("Compounds", "Individuals", "ExpressionProfiles", "Protocols", "Events", "Formulations"):
            assert reference.get(key) == source.get(key), key
        assert "ObservedData" not in reference
        assert "ParameterIdentifications" not in reference
        assert not case["independent_blind_evaluation"]
    repeated = manifest["cases"][-1]
    for arm in ("adapted", "official_reference"):
        expected = repeated["native_expectations"][arm]
        assert expected["expected_event_count"] == 7
        starts = [x["value"] for x in expected["expectations"] if x["path"].endswith("|Start time")]
        assert starts == [1440.0 * n for n in range(7)]
        doses = [x["value"] for x in expected["expectations"] if x["path"].endswith("|Dose")]
        assert doses == [450.0] * 7
    assert repeated["observation_refs"][1]["required_time_window_min"] == [8760.0, 9360.0]
    assert repeated["observation_refs"][1]["matrix_mapping"]["strict_matrix_validation"] is False


def test_case_metadata_never_inspects_concentration_values(monkeypatch):
    class NoConcentrationAccess(dict):
        def __getitem__(self, key):
            assert key != "Values", "Case selection inspected observed concentrations"
            return super().__getitem__(key)

        def get(self, key, default=None):
            assert key != "Values", "Case selection inspected observed concentrations"
            return super().get(key, default)

    original = cases_module._read_source

    def metadata_only_source(reference, project_root):
        source = original(reference, project_root)
        for observed in source["ObservedData"]:
            observed["Columns"] = [NoConcentrationAccess(column) for column in observed["Columns"]]
        return source

    monkeypatch.setattr(cases_module, "_read_source", metadata_only_source)
    first = cases_module.build_case_manifest(ROOT)
    second = cases_module.build_case_manifest(ROOT)
    assert first == second
    assert first["evaluation_contract"]["solver_calls"] == 0
    # Return values are independent objects, not shared mutable request state.
    before = deepcopy(cases_module.CASE_DEFINITIONS)
    first["cases"][1]["request"]["dose"]["value"] = 999
    assert cases_module.CASE_DEFINITIONS == before
