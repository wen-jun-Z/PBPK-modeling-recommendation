"""Endpoint math and scope checks on explicit, nonuniform synthetic trajectories."""
from copy import deepcopy
import math

import pytest

from pbpk_kb.blueprints.endpoint_prediction import predict_requested_endpoints


def _curve():
    return {"time_min": [0., 30., 90., 180.], "concentration": [0., 4., 2., 0.],
            "time_unit": "min", "unit": "umol/l"}


def _target():
    return {"endpoints": ["plasma_concentration_time", "auc", "cmax"], "primary_endpoint": "auc",
            "window_start_h": .25, "window_end_h": 2.}


def test_nonuniform_grid_and_both_interpolated_boundaries():
    curve, target = _curve(), _target()
    original = deepcopy((curve, target))
    out = predict_requested_endpoints(curve, target, 500.)
    clipped = out["endpoints"]["plasma_concentration_time"]
    assert clipped["time_h"] == [.25, .5, 1.5, 2.]
    assert clipped["concentration_umol_per_l"] == pytest.approx([2., 4., 2., 4/3])
    assert clipped["concentration_mg_per_l"] == pytest.approx([1., 2., 1., 2/3])
    assert out["endpoints"]["auc"]["value_umol_h_per_l"] == pytest.approx(55/12)
    assert out["endpoints"]["auc"]["value_mg_h_per_l"] == pytest.approx(55/24)
    assert out["endpoints"]["cmax"]["value_umol_per_l"] == 4.
    assert out["endpoints"]["cmax"]["value_mg_per_l"] == 2.
    assert out["endpoints"]["cmax"]["tmax_h"] == .5
    assert out["endpoints"]["cmax"]["source_grid_max_interval_h"] == 1.5
    assert out["warnings"] and out["extrapolation_performed"] is False
    assert (curve, target) == original


def test_endpoint_selection_uses_same_curve_without_changing_values():
    all_endpoints = predict_requested_endpoints(_curve(), _target(), 250.)
    for name, expected in all_endpoints["endpoints"].items():
        target = {**_target(), "endpoints": [name], "primary_endpoint": name}
        out = predict_requested_endpoints(_curve(), target, 250.)
        assert out["endpoints"] == {name: expected}


def test_exact_boundaries_are_not_duplicated_and_tmax_ties_choose_first():
    curve = {**_curve(), "concentration": [0., 4., 4., 0.]}
    target = {**_target(), "window_start_h": .5, "window_end_h": 1.5}
    out = predict_requested_endpoints(curve, target, 1.)
    assert out["window_curve_points"] == 2
    assert out["endpoints"]["cmax"]["tmax_h"] == .5
    assert out["endpoints"]["auc"]["value_umol_h_per_l"] == 4.


def test_interval_inside_one_source_segment():
    target = {**_target(), "window_start_h": .75, "window_end_h": 1.}
    out = predict_requested_endpoints(_curve(), target, 1000.)
    assert out["window_curve_points"] == 2
    assert out["endpoints"]["plasma_concentration_time"]["concentration_umol_per_l"] == [3.5, 3.]
    assert out["endpoints"]["auc"]["value_umol_h_per_l"] == .8125
    assert out["endpoints"]["cmax"]["value_umol_per_l"] == 3.5


@pytest.mark.parametrize("unit", ["umol/l", "µmol/l", "μmol/l"])
def test_micro_symbol_spellings_are_not_different_units(unit):
    assert predict_requested_endpoints({**_curve(), "unit": unit}, _target(), 300.)["status"] == "completed"


@pytest.mark.parametrize("unit", ["mg/l", "nmol/l", "", None])
def test_other_or_missing_units_are_rejected(unit):
    with pytest.raises(ValueError, match="umol/l"):
        predict_requested_endpoints({**_curve(), "unit": unit}, _target(), 300.)


@pytest.mark.parametrize("weight", [0., -1., float('nan'), float('inf'), True, "300"])
def test_invalid_molecular_weight(weight):
    with pytest.raises(ValueError):
        predict_requested_endpoints(_curve(), _target(), weight)


@pytest.mark.parametrize("start,end", [(-.1, 1.), (0., 3.01), (2., 1.), (1., 1.), (math.nan, 1.), (0., math.inf)])
def test_invalid_or_uncovered_window_is_rejected(start, end):
    with pytest.raises(ValueError):
        predict_requested_endpoints(_curve(), {**_target(), "window_start_h": start, "window_end_h": end}, 100.)


def test_no_left_extrapolation_even_for_nonnegative_start():
    curve = {**_curve(), "time_min": [10., 30., 90., 180.]}
    with pytest.raises(ValueError, match="extrapolation"):
        predict_requested_endpoints(curve, {**_target(), "window_start_h": 0.}, 100.)


@pytest.mark.parametrize("changes", [{"endpoints": ["auc", "auc"]}, {"endpoints": ["half_life"]},
                                     {"endpoints": []}, {"primary_endpoint": "unrequested"},
                                     {"endpoints": ["auc"], "primary_endpoint": "cmax"}])
def test_unsupported_duplicate_or_unselected_target(changes):
    with pytest.raises(ValueError):
        predict_requested_endpoints(_curve(), {**_target(), **changes}, 100.)


@pytest.mark.parametrize("field,value", [("concentration", [0., -1., 2., 0.]),
    ("concentration", [0., math.nan, 2., 0.]), ("concentration", [0., math.inf, 2., 0.]),
    ("time_min", [-1., 30., 90., 180.]), ("time_min", [0., 30., 30., 180.]),
    ("time_min", [0., 90., 30., 180.]), ("time_min", [0., 30., math.nan, 180.]),
    ("time_min", [0.]), ("time_unit", "h")])
def test_invalid_curve(field, value):
    with pytest.raises(ValueError):
        predict_requested_endpoints({**_curve(), field: value}, _target(), 100.)


def test_all_zero_trajectory_has_zero_endpoints_without_log_failure():
    out = predict_requested_endpoints({**_curve(), "concentration": [0., 0., 0., 0.]}, _target(), 100.)
    assert out["endpoints"]["auc"]["value_umol_h_per_l"] == 0.
    assert out["endpoints"]["cmax"]["value_umol_per_l"] == 0.


def test_finite_inputs_cannot_silently_return_nonfinite_endpoints():
    with pytest.raises(ValueError, match="finite numeric range"):
        predict_requested_endpoints({**_curve(), "concentration": [1e308] * 4}, _target(), 1e308)
