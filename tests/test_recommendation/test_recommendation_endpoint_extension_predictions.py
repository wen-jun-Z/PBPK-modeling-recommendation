"""Prediction-only endpoint extension: units, route and tail eligibility."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("extended_predictions", ROOT / "scripts/recommendation_endpoint_extension_predictions.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def sample(route="oral", schedule=None):
    times = [0., 1., 2., 4., 8., 12., 24.]
    values = [0., *[10 * math.exp(-.5 * (t - 1)) for t in times[1:]]]
    return m.clinical_nca(times, values, schedule or times, dose_mg=100., infusion_h=1. if route == "intravenous" else 0., route=route, lloq=1e-6)


def test_oral_terminal_nca_is_apparent_and_does_not_invent_vss():
    endpoints, diag = sample()
    assert endpoints["half_life"]["value"] == pytest.approx(math.log(2) / .5)
    assert diag["terminal_quantity"] == "apparent_terminal_half_life"
    assert endpoints["CL"]["value"] == pytest.approx(100 * 1000 / endpoints["AUC_infinity"]["value"])
    assert endpoints["Vz"]["value"] == pytest.approx(endpoints["CL"]["value"] / .5)
    assert endpoints["MRT"]["eligibility"]
    assert endpoints["Vss"]["value"] is None and not endpoints["Vss"]["eligibility"]


def test_iv_mrt_has_half_infusion_correction_and_vss_identity():
    oral, _ = sample()
    iv, _ = sample("intravenous")
    assert iv["MRT"]["value"] == pytest.approx(oral["MRT"]["value"] - .5)
    assert iv["Vss"]["value"] == pytest.approx(iv["CL"]["value"] * iv["MRT"]["value"])


def test_missing_dose_sample_may_add_verified_prediction_boundary_only():
    result, diagnostic = sample("intravenous", [1., 2., 4., 8., 12., 24.])
    added = diagnostic["added_dose_time_boundary"]
    assert added["concentration_ng_ml"] == 0 and not added["observed_sample"]
    assert not added["used_for_terminal_regression"]
    assert 0 not in diagnostic["selected_tail"]["time_h"]
    assert result["CL"]["eligibility"]


def test_no_lloq_remains_unavailable_without_invented_cutoff():
    endpoints, _ = m.clinical_nca([0.,1.,2.,4.], [0.,10.,5.,1.], [0.,1.,2.,4.], dose_mg=10., infusion_h=0., route="oral", lloq=None)
    assert all(r["value"] is None and not r["eligibility"] for r in endpoints.values())


def test_short_terminal_span_does_not_get_half_life_or_clearance():
    times = [0., 1., 2., 2.1, 2.2]
    values = [0.,10.,*[10*math.exp(-.5*(t-1)) for t in times[2:]]]
    endpoints, diagnostic = m.clinical_nca(times, values, times, dose_mg=100., infusion_h=0., route="oral", lloq=.1)
    assert endpoints["AUC_last"]["eligibility"]
    assert not endpoints["half_life"]["eligibility"] and not endpoints["CL"]["eligibility"]
    assert diagnostic["terminal_span_in_half_lives"] < 1


def test_lloq_units_and_conflicting_metadata_are_not_silently_mixed():
    cutoff, paths = m.assay_cutoff({"protocol": {"quantification_limit": {"unit": "ug/mL", "value": .25}}})
    assert cutoff == 250 and paths
    with pytest.raises(ValueError, match="Conflicting"):
        m.assay_cutoff({"lloq_ng_ml": 1., "other": {"lloq_ng_ml": 2.}})


def test_interval_clearance_requires_periodic_model_and_complete_dosing_history():
    t = list(np.linspace(-24, 12, 361))
    contract = {"administration_times_min": [0, 720, 1440, 2160], "observation_origin_min": 1440, "dose_mg": 100.}
    value, diagnostics = m.interval_clearance(t, [2.] * len(t), contract)
    assert value["steady_state"] and value["eligibility"]
    assert value["value"] == pytest.approx(100 * 1000 / (2 * 12))
    assert value["window_h"] == [0., 12.]
    bad, diag = m.interval_clearance(t, [2 + .05*(time+24) for time in t], contract)
    assert not bad["eligibility"] and diag["three_cycle_max_concentration_relative_change"] > .01
    missing, _ = m.interval_clearance(t, [2.] * len(t), {**contract, "administration_times_min": [0.]})
    assert not missing["eligibility"]


def test_scalar_schema_keeps_tmax_absolute_error_and_native_grid_separate():
    prediction = {"time_h": [0.,1.,2.,4.], "concentration_ng_ml": [0.,10.,5.,1.],
                  "native_curve": {"time_min": [0.,60.,120.,240.], "concentration": [0.,.1,.05,.01], "time_unit":"min", "unit":"umol/l"},
                  "molecular_weight_g_per_mol":100., "time_origin_h_after_first_dose":0., "matrix":"plasma", "binding":"total", "analyte":"parent", "compatibility":{}}
    plan = {"contract": {"route":"Oral", "sampling_times_min":[0.,120.,240.], "administration_times_min":[0.], "dose_mg":100.}}
    endpoints, _, _ = m.derive(prediction, plan, {}, {"nominal_times_h": [0.,2.,4.]})
    assert set(endpoints) == set(m.ENDPOINTS) and len(endpoints) == 11
    assert endpoints["Tmax"]["value"] == 2. and endpoints["Tmax"]["native_prediction"]["value"] == 1.
    assert endpoints["Tmax"]["error_metric"] == "absolute_error_h"
    assert endpoints["CL"]["quantity"] == "CL/F" and endpoints["Vz"]["quantity"] == "Vz/F"
    assert json.loads(json.dumps(endpoints, allow_nan=False)) == endpoints
    assert endpoints["AUC_window"] is not endpoints["AUC_window"]["native_prediction"]
    matched, diagnostic, trajectory = m.derive({**prediction, "sampling_grid_registration": None}, plan, {}, {"steady_state": True})
    assert matched["Cmax"]["steady_state"] and matched["Tmax"]["steady_state"]
    assert diagnostic["target_regimen_steady_state"] and trajectory["steady_state"]
    assert not trajectory["clinical_sampling_qualified"]
    assert matched["Tmax"]["prediction_kind"] == "native" and matched["Tmax"]["value"] == 1.
    assert not matched["half_life"]["eligibility"]
    assert "unreported" in matched["half_life"]["reason"]


def test_only_locked_model_weight_can_normalize_independent_definitions():
    assert m.source_body_weight({"contract": {"population": {"weight_mean_kg": 70}}}) == (None, None)
    weight, source = m.source_body_weight({"source_individual": {"OriginData": {"Weight": {"Value": 80., "Unit": "kg"}}}})
    assert weight == 80. and source.endswith("Weight")
    assert m.source_body_weight({"source_individual": [{"origin_raw_base_units": {"weight": 80, "weightUnit": "lb"}}]}) == (None, None)
    endpoints = {key: {"endpoint_key": key, "value": 10., "native_prediction": {"value": 20.}, "unit": unit} for key, unit in m.ENDPOINTS.items()}
    normalized = {r["endpoint_key"]: r for r in m.normalized_definitions(endpoints, weight, 40)}
    assert normalized["CL"]["value"] == .125 and normalized["CL"]["unit"] == "L/h/kg"
    assert normalized["Cmax"]["value"] == 20. and normalized["Cmax"]["native_prediction"]["value"] == 40.
    assert normalized["Cmax"]["dose_normalization"] == "dose_per_body_weight"
    assert endpoints["CL"]["value"] == 10. and endpoints["Cmax"]["native_prediction"]["value"] == 20.
