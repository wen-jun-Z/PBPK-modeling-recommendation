"""Protocol checks without ODE or evaluation concentration access."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("source_context_cases", ROOT / "scripts/source_context_cases.py")
cases = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cases)


def test_registered_roles_context_and_source_preservation():
    manifest = cases.build_case_manifest(ROOT)
    contract = manifest["split_contract"]
    assert contract["case_count"] == 17
    assert contract["historical_curve_count"] == 20
    assert contract["curve_count_by_role"] == {
        "calibration": 3, "selection": 2, "evaluation": 5, "diagnostic": 6, "sensitivity": 4}
    assert contract["calibration_eligible_curve_count"] == 2
    assert not contract["independent_blind_evaluation"]
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    assert sum(len(c["observation_refs"]) for c in by_id.values() if c.get("prior_pilot_curve")) == 9
    assert not by_id["rifampicin_cal_peloquin_600"]["calibration_eligible"]
    assert by_id["rifampicin_repeated_protocol_individual"]["context_request"]["human_sex"]["value"] == "FEMALE"
    assert by_id["midazolam_food_fasted"]["context_request"] == {}
    assert sum("solubility_alternative" in c["context_request"] for c in by_id.values()) == 3
    trough = by_id["rifampicin_eval_chattopadhyay_repeat"]
    assert trough["sampling_type"] == "longitudinal_predose_trough_sequence"
    assert trough["endpoint_eligibility"]["auc_evaluable"] is False
    assert trough["endpoint_eligibility"]["cmax_evaluable"] is False
    assert trough["context_request"] == {}
    after = by_id["midazolam_food_after"]
    assert after["observation_refs"][0]["required_time_window_min"] == [90.0, 660.0]
    starts = [x["value"] for x in after["native_expectations"]["official_reference"]["expectations"]
              if x["path"].endswith("|Start time")]
    assert starts == [60.0]
    repeated = by_id["rifampicin_eval_baneyx_repeat"]
    assert [r["required_time_window_min"] for r in repeated["observation_refs"]] == [[4380., 4800.], [8700., 9120.]]
    for case in by_id.values():
        snapshot = cases.build_reference_snapshot(case, ROOT)
        assert len(snapshot["Simulations"]) == 1
        assert "ObservedData" not in snapshot
        if case["compound_name"] == "Rifampicin":
            assert len(snapshot["Compounds"][0]["Processes"]) == 19
            assert len(snapshot["Simulations"][0]["Interactions"]) == 7
    spec = manifest["rifampicin_pint_spec"]
    assert spec["source_value"] == 1.24e-5
    assert spec["search_bounds"] == [3.1e-6, 4.96e-5]
    assert spec["source_pi_bounds"] == [3.48e-7, .00348]


def test_metadata_build_never_accesses_concentration_arrays(monkeypatch):
    class NoConcentrationAccess(dict):
        def __getitem__(self, key):
            assert key != "Values", "Concentrations accessed during protocol construction"
            return super().__getitem__(key)

        def get(self, key, default=None):
            assert key != "Values", "Concentrations accessed during protocol construction"
            return super().get(key, default)

    original = cases._legacy._read_source

    def source_metadata(reference, root):
        snapshot = original(reference, root)
        for observed in snapshot["ObservedData"]:
            observed["Columns"] = [NoConcentrationAccess(column) for column in observed["Columns"]]
        return snapshot

    monkeypatch.setattr(cases._legacy, "_read_source", source_metadata)
    assert cases.build_case_manifest(ROOT) == cases.build_case_manifest(ROOT)


def test_nonfitting_observations_require_explicit_role_and_lock(monkeypatch):
    case = next(c for c in cases.build_cases(ROOT) if c["role"] == "evaluation")

    def forbidden(*args, **kwargs):
        raise AssertionError("Observation decoder must not run before the lock")

    monkeypatch.setattr(cases._legacy, "load_observations", forbidden)
    with pytest.raises(ValueError, match="role"):
        cases.load_observations(case, ROOT, role="calibration")
    with pytest.raises(ValueError, match="locked"):
        cases.load_observations(case, ROOT, role="evaluation")


def test_calibration_loader_retains_registered_data_conflict():
    calibration = [c for c in cases.build_cases(ROOT) if c["role"] == "calibration"]
    rows = [r for c in calibration for r in cases.load_observations(c, ROOT, role="calibration")]
    assert len(rows) == 3
    assert all(r["status"] == "ready" for r in rows)
    assert sum(r["calibration_eligible"] for r in rows) == 2
    blocked = next(r for r in rows if not r["calibration_eligible"])
    assert blocked["data_quality_status"] == "initial_condition_conflict"
    assert blocked["observations"][0]["time_min"] == 0
    assert blocked["observations"][0]["concentration_umol_per_l"] == pytest.approx(.2 * 1000 / 822.94)


def test_trough_endpoint_correction_keeps_pointwise_data_and_existing_prediction_inputs(monkeypatch):
    case = next(c for c in cases.build_cases(ROOT)
                if c["case_id"] == "rifampicin_eval_chattopadhyay_repeat")
    reference = case["observation_refs"][0]
    # Synthetic decoder response tests semantics without reading held-out values.
    points = [{"time_min": 1440., "concentration_umol_per_l": .01, "unit": "umol/l"},
              {"time_min": 4320., "concentration_umol_per_l": .02, "unit": "umol/l"}]
    monkeypatch.setattr(cases._legacy, "load_observations", lambda *args: [{
        "source_observation_name": reference["observation_name"],
        "observations": points, "status": "ready", "auc_evaluable": True, "cmax_evaluable": True}])
    row = cases.load_observations(case, ROOT, role="evaluation", predictions_locked=True)[0]
    assert row["status"] == "ready"
    assert row["observations"] == points
    assert row["auc_evaluable"] is False and row["cmax_evaluable"] is False
    assert row["endpoint_protocol"]["pointwise_log_rmse_and_gmfe_evaluable"] is True
    assert row["role"] == row["study_role"] == "heldout_evaluation"
    assert case["required_times_min"] == [1440., 4320., 8640., 10800., 11520., 14400.]
    assert case["context_request"] == {}
