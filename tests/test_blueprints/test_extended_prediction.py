"""Analytic checks for model-only terminal endpoints and finite-window isolation."""
from copy import deepcopy
import math

import pytest

from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction


def curve(fn, end=48., step=.01):
    times = [i * step for i in range(round(end / step) + 1)]
    return {"time_min": [60 * time for time in times], "concentration": [fn(time) for time in times],
            "time_unit": "min", "unit": "umol/l"}


def request(**dosing):
    return {"dosing": {"route": "intravenous", "dose_mg": 100., "start_h": 0.,
                       "infusion_minutes": 0., **dosing},
            "evidence": [{"parameter": "molecular_weight", "value": 1000.}],
            "target": {"endpoints": ["auc"], "primary_endpoint": "auc",
                       "window_start_h": dosing.get("start_h", 0.), "window_end_h": 8.}}


def test_exponential_clips_only_numeric_suffix_and_keeps_requested_window():
    data = curve(lambda t: 10 * math.exp(-.5 * t))
    req = request()
    before = deepcopy((data, req))
    result = summarize_extended_prediction(data, req)
    endpoints = result["endpoints"]
    assert result["endpoint_counts"] == {"requested": 11, "completed": 11, "not_estimable": 0}
    assert endpoints["auc"]["value_mg_h_per_l"] == pytest.approx(20 * (1 - math.exp(-4)), rel=3e-6)
    assert endpoints["c_last"]["value_mg_per_l"] == pytest.approx(10 * math.exp(-4))
    assert endpoints["auc_inf"]["value_mg_h_per_l"] == pytest.approx(20., rel=3e-6)
    assert endpoints["clearance"]["value_l_per_h"] == pytest.approx(5., rel=3e-6)
    assert endpoints["terminal_half_life"]["value_h"] == pytest.approx(math.log(2) / .5)
    assert endpoints["vz"]["value_l"] == pytest.approx(10., rel=3e-6)
    assert endpoints["mrt"]["value_h"] == pytest.approx(2., rel=5e-6)
    assert endpoints["vss"]["value_l"] == pytest.approx(10., rel=8e-6)
    assert result["diagnostics"]["dropped_numeric_suffix_points"] > 0
    assert result["diagnostics"]["adjacent_slope_relative_difference"] < 1e-12
    assert result["extension"]["recommended"] is False
    assert (data, req) == before


def test_finite_infusion_uses_dose_elapsed_time_and_corrects_mrt():
    def infusion(t):
        elapsed = t - 3.
        if elapsed <= 0:
            return 0.
        return 20 * (1 - math.exp(-.5 * min(elapsed, 1.))) * math.exp(-.5 * max(0., elapsed - 1.))
    result = summarize_extended_prediction(curve(infusion), request(start_h=3., infusion_minutes=60.))
    endpoints = result["endpoints"]
    assert result["status"] == "completed"
    assert endpoints["tmax"]["value_h"] == 4.
    assert endpoints["mrt"]["value_h"] == pytest.approx(2., rel=2e-5)
    assert endpoints["vss"]["value_l"] == pytest.approx(10., rel=2e-5)
    assert endpoints["clearance"]["value_l_per_h"] == pytest.approx(5., rel=2e-5)


def test_biexponential_recovers_distinct_terminal_and_steady_state_volumes():
    result = summarize_extended_prediction(curve(lambda t: 8 * math.exp(-2 * t) + 2 * math.exp(-.2 * t), step=.005), request())
    ep = result["endpoints"]
    assert result["status"] == "completed"
    assert ep["auc_inf"]["value_mg_h_per_l"] == pytest.approx(14., rel=1e-5)
    assert ep["terminal_half_life"]["value_h"] == pytest.approx(math.log(2) / .2, rel=1e-6)
    assert ep["vz"]["value_l"] == pytest.approx(100 / (14 * .2), rel=1e-5)
    assert ep["vss"]["value_l"] == pytest.approx(100 * 52 / 14 ** 2, rel=2e-5)


def test_floor_noise_does_not_force_false_flat_terminal_half_life():
    data = curve(lambda t: 10 * math.exp(-.5 * t))
    data["concentration"] = [max(value, 2e-10) for value in data["concentration"]]
    result = summarize_extended_prediction(data, request())
    assert result["status"] == "completed"
    assert result["endpoints"]["terminal_half_life"]["value_h"] == pytest.approx(math.log(2) / .5)
    assert result["diagnostics"]["regression_candidate_count"] == 1


def test_late_above_floor_rebound_is_retained_and_fails_quality_gates():
    data = curve(lambda t: 10 * math.exp(-.5 * t))
    data["concentration"][-10:] = [.5] * 10
    result = summarize_extended_prediction(data, request())
    assert result["diagnostics"]["below_floor_rebound_detected"] is True
    assert result["diagnostics"]["dropped_numeric_suffix_points"] == 0
    assert result["endpoints"]["terminal_half_life"]["value"] is None
    assert result["endpoints"]["terminal_half_life"]["status"] == "not_estimable"


def test_short_horizon_suggests_extension_without_relaxing_gates():
    data = curve(lambda t: 10 * math.exp(-.03 * t), end=8.)
    result = summarize_extended_prediction(data, request())
    assert result["endpoints"]["auc"]["status"] == "completed"
    assert result["endpoints"]["terminal_half_life"]["value"] is None
    assert result["extension"]["recommended"] is True
    assert 8 < result["extension"]["suggested_end_h"] <= 168.
    assert result["diagnostics"]["quality_gates"]["min_span_half_lives"] == 1.


def test_oral_request_cannot_claim_absolute_volumes():
    with pytest.raises(ValueError, match="intravenous"):
        summarize_extended_prediction(curve(lambda t: math.exp(-t)), request(route="oral"))


def test_stricter_original_quality_gates_are_retained():
    req = request()
    req["target"]["terminal_fit"] = {"start_h": 4., "end_h": 8., "min_adjusted_r2": .999,
        "min_span_half_lives": 2., "max_auc_extrapolated_fraction": .1, "max_aumc_extrapolated_fraction": .1}
    result = summarize_extended_prediction(curve(lambda t: 10 * math.exp(-.5 * t)), req)
    assert result["status"] == "completed"
    assert result["diagnostics"]["quality_gates"]["min_adjusted_r2"] == .999
    assert result["diagnostics"]["quality_gates"]["min_span_half_lives"] == 2.
    assert result["diagnostics"]["quality_gates"]["max_auc_extrapolated_fraction"] == .1


def test_missing_original_dose_sample_prevents_total_exposure_fabrication():
    data = curve(lambda t: 10 * math.exp(-.5 * t))
    data["time_min"] = data["time_min"][1:]
    data["concentration"] = data["concentration"][1:]
    req = request()
    req["target"]["window_start_h"] = .02
    result = summarize_extended_prediction(data, req)
    assert result["endpoints"]["auc"]["status"] == "completed"
    assert result["endpoints"]["auc_inf"]["value"] is None


def test_all_zero_curve_returns_missing_terminal_values_not_zero_endpoints():
    result = summarize_extended_prediction(curve(lambda t: 0.), request())
    assert result["status"] == "completed_with_unavailable_endpoints"
    assert result["endpoints"]["auc"]["value_mg_h_per_l"] == 0.
    assert result["endpoints"]["clearance"]["value"] is None
