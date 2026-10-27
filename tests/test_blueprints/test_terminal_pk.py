"""Analytic NCA references and intentionally non-estimable terminal profiles."""
from copy import deepcopy
import math

import pytest

from pbpk_kb.blueprints.endpoint_prediction import SUPPORTED_ENDPOINTS, predict_requested_endpoints


def _curve(function=lambda t: 10 * math.exp(-.5 * t), end=24., step=.02, start=0.):
    times = [start + i * step for i in range(round((end - start) / step) + 1)]
    return {"time_min": [60 * t for t in times], "concentration": [function(t) for t in times],
            "unit": "umol/l", "time_unit": "min"}


def _target(end=24., start=0., fit_start=12., selected=None):
    selected = selected or ["auc", "cmax", "tmax", "c_last", "terminal_half_life", "auc_inf",
                            "clearance", "vz", "mrt", "vss"]
    return {"endpoints": selected, "primary_endpoint": selected[0], "window_start_h": start,
            "window_end_h": end, "terminal_fit": {"start_h": fit_start, "end_h": end}}


def _context(**changes):
    return {"route": "intravenous", "start_h": 0., "dose_mg": 100.,
            "infusion_minutes": 0., "single_dose": True, **changes}


def _run(curve=None, target=None, context=None):
    return predict_requested_endpoints(curve or _curve(), target or _target(), 1000., context or _context())


def test_catalog_has_thirteen_endpoints_and_legacy_alias_is_not_accepted():
    assert len(SUPPORTED_ENDPOINTS) == 13
    assert "half_life" not in SUPPORTED_ENDPOINTS


def test_iv_bolus_monoexponential_has_analytic_clearance_volumes_and_residence_time():
    out = _run()
    endpoints = out["endpoints"]
    assert endpoints["terminal_half_life"]["value_h"] == pytest.approx(math.log(2) / .5)
    assert endpoints["auc_inf"]["value_mg_h_per_l"] == pytest.approx(20., rel=1e-5)
    assert endpoints["clearance"]["value_l_per_h"] == pytest.approx(5., rel=1e-5)
    assert endpoints["vz"]["value_l"] == pytest.approx(10., rel=1e-5)
    assert endpoints["mrt"]["value_h"] == pytest.approx(2., rel=2e-5)
    assert endpoints["vss"]["value_l"] == pytest.approx(10., rel=3e-5)
    assert endpoints["tmax"]["value_h"] == 0.
    assert endpoints["c_last"]["value_mg_per_l"] == pytest.approx(10 * math.exp(-12))
    assert out["extrapolation_performed"] is True
    assert out["parameters_fitted"] is False
    assert out["terminal_log_linear_regression_performed"] is True
    assert out["endpoint_counts"] == {"requested": 10, "completed": 10, "not_estimable": 0}


def test_iv_biexponential_distinguishes_terminal_volume_from_steady_state_volume():
    curve = _curve(lambda t: 8 * math.exp(-2 * t) + 2 * math.exp(-.2 * t), end=36., step=.005)
    out = _run(curve, _target(end=36., fit_start=18.))["endpoints"]
    assert out["auc_inf"]["value_mg_h_per_l"] == pytest.approx(14., rel=1e-5)
    assert out["clearance"]["value_l_per_h"] == pytest.approx(100 / 14, rel=1e-5)
    assert out["vz"]["value_l"] == pytest.approx(100 / (14 * .2), rel=1e-5)
    assert out["mrt"]["value_h"] == pytest.approx(52 / 14, rel=1e-5)
    assert out["vss"]["value_l"] == pytest.approx(100 * 52 / 14 ** 2, rel=1e-5)
    assert out["vz"]["value_l"] != pytest.approx(out["vss"]["value_l"])


def test_constant_infusion_corrects_mrt_and_uses_time_since_dose_not_clock_time():
    def infusion(t):
        elapsed = t - 3.
        if elapsed <= 0:
            return 0.
        return 20. * (1 - math.exp(-.5 * min(elapsed, 1.))) * math.exp(-.5 * max(0., elapsed - 1.))
    out = _run(_curve(infusion, end=27., step=.01), _target(start=3., end=27., fit_start=15.),
               _context(start_h=3., infusion_minutes=60.))["endpoints"]
    assert out["mrt"]["value_h"] == pytest.approx(2., rel=2e-5)
    assert out["mrt"]["infusion_correction_h"] == .5
    assert out["mrt"]["includes_absorption_time"] is False
    assert out["vss"]["value_l"] == pytest.approx(10., rel=2e-5)
    assert out["tmax"]["value_h"] == 4.


def test_oral_returns_apparent_parameters_and_does_not_invent_absolute_volume():
    curve = _curve(lambda t: 5 / .75 * (math.exp(-.25 * t) - math.exp(-t)), end=48.)
    target = _target(end=48., fit_start=24., selected=["auc_inf", "apparent_clearance", "apparent_vz",
                                                     "mrt", "clearance", "vz", "vss"])
    out = _run(curve, target, _context(route="oral", absorption_complete_h=18.))
    endpoints = out["endpoints"]
    assert endpoints["apparent_clearance"]["value_l_per_h"] == pytest.approx(5., rel=1e-5)
    assert endpoints["apparent_vz"]["value_l"] == pytest.approx(20., rel=1e-5)
    assert endpoints["mrt"]["value_h"] == pytest.approx(5., rel=1e-5)
    assert endpoints["mrt"]["includes_absorption_time"] is True
    for name in ("clearance", "vz", "vss"):
        assert endpoints[name]["status"] == "not_estimable"
    assert out["endpoint_counts"] == {"requested": 7, "completed": 4, "not_estimable": 3}


def test_iv_does_not_label_parameters_as_apparent():
    out = _run(target=_target(selected=["apparent_clearance", "apparent_vz"]))
    assert out["endpoint_counts"]["not_estimable"] == 2


def test_half_life_survives_large_extrapolated_auc_while_other_endpoints_are_unavailable():
    out = _run(_curve(end=2.), _target(end=2., fit_start=.1))
    endpoints = out["endpoints"]
    assert endpoints["terminal_half_life"]["status"] == "completed"
    assert endpoints["auc"]["value_mg_h_per_l"] > 0
    assert endpoints["auc_inf"]["status"] == "not_estimable"
    assert "AUC tail" in endpoints["auc_inf"]["reason"]
    assert endpoints["auc_inf"]["diagnostics"]["auc_extrapolated_fraction"] > .2
    assert out["extrapolation_performed"] is False


def test_mrt_and_vss_have_separate_aumc_tail_gate():
    out = _run(_curve(end=4.), _target(end=4., fit_start=2.))["endpoints"]
    assert out["auc_inf"]["status"] == "completed"
    assert out["clearance"]["status"] == "completed"
    assert out["vz"]["status"] == "completed"
    assert out["mrt"]["status"] == "not_estimable"
    assert "AUMC tail" in out["vss"]["reason"]


@pytest.mark.parametrize("failure,changes", [
    ("predeclared", {"terminal_fit": None}),
    ("min_points", {"terminal_fit": {"start_h": 12., "end_h": 24., "min_points": True}}),
    ("valid range", {"terminal_fit": {"start_h": 12., "end_h": 24., "max_auc_extrapolated_fraction": 1.}}),
    ("end at its final", {"terminal_fit": {"start_h": 12., "end_h": 23.}}),
    ("predeclared", {"terminal_fit": {"start_h": 12., "end_h": 24., "auto_select": True}}),
    ("span gate", {"terminal_fit": {"start_h": 23.9, "end_h": 24.}}),
    ("insufficient", {"terminal_fit": {"start_h": 23.99, "end_h": 24.}}),
])
def test_invalid_or_uninformative_terminal_windows_keep_window_auc(failure, changes):
    out = _run(target={**_target(), **changes})["endpoints"]
    assert out["auc"]["value_mg_h_per_l"] > 0
    assert out["terminal_half_life"]["status"] == "not_estimable"
    assert failure in out["terminal_half_life"]["reason"]


@pytest.mark.parametrize("function,reason", [
    (lambda t: 0., "nonpositive"), (lambda t: 1., "decrease"),
    (lambda t: t + 1., "decrease"),
    (lambda t: math.exp(-.005 * t ** 2), "R-squared"),
])
def test_nonterminal_shapes_do_not_generate_half_life(function, reason):
    target = _target(fit_start=1.)
    out = _run(_curve(function), target)["endpoints"]
    assert out["auc"]["value_mg_h_per_l"] >= 0
    assert out["terminal_half_life"]["status"] == "not_estimable"
    assert reason in out["terminal_half_life"]["reason"]


@pytest.mark.parametrize("changes,reason", [
    ({"single_dose": False}, "single_dose"),
    ({"route": "intramuscular"}, "route"),
    ({"infusion_minutes": 1000.}, "administration"),
    ({"route": "oral"}, "absorption_complete"),
    ({"route": "oral", "absorption_complete_h": 20.}, "absorption completion"),
])
def test_unsupported_or_unfinished_administration_is_not_terminal(changes, reason):
    out = _run(context=_context(**changes))["endpoints"]
    assert out["terminal_half_life"]["status"] == "not_estimable"
    assert reason in out["terminal_half_life"]["reason"]


def test_absent_context_does_not_break_curve_auc_or_sampled_endpoints():
    out = predict_requested_endpoints(_curve(), _target(), 1000.)
    assert out["endpoints"]["auc"]["value_mg_h_per_l"] > 0
    assert out["endpoints"]["c_last"]["status"] == "completed"
    assert out["endpoints"]["terminal_half_life"]["status"] == "not_estimable"


@pytest.mark.parametrize("time_start", [1., .01])
def test_truncated_curve_never_becomes_total_exposure(time_start):
    out = _run(_curve(start=time_start, step=.01), _target(start=time_start))["endpoints"]
    assert out["terminal_half_life"]["status"] == "completed"
    assert out["auc"]["value_mg_h_per_l"] > 0
    for name in ("auc_inf", "clearance", "vz", "mrt", "vss"):
        assert out[name]["status"] == "not_estimable"
        assert "dose-time" in out[name]["reason"]


def test_nonzero_concentration_at_infusion_start_prevents_single_dose_total_exposure():
    out = _run(context=_context(infusion_minutes=30.))["endpoints"]
    assert out["terminal_half_life"]["status"] == "completed"
    assert out["auc_inf"]["status"] == "not_estimable"
    assert "zero predose" in out["auc_inf"]["reason"]


def test_unprovided_dose_only_blocks_dose_derived_endpoints():
    context = _context()
    del context["dose_mg"]
    out = _run(context=context)["endpoints"]
    assert out["auc_inf"]["status"] == "completed"
    assert out["mrt"]["status"] == "completed"
    for name in ("clearance", "vz", "vss"):
        assert out[name]["status"] == "not_estimable"
        assert "dose_mg" in out[name]["reason"]


def test_terminal_fit_uses_only_declared_original_samples_without_input_mutation():
    curve, target, context = _curve(step=1.), _target(fit_start=12.2), _context()
    original = deepcopy((curve, target, context))
    out = _run(curve, target, context)
    assert out["terminal_diagnostics"]["fit_times_h"] == list(range(13, 25))
    assert (curve, target, context) == original


def test_c_last_is_interpolated_window_end_and_tmax_uses_earliest_tie():
    curve = {"time_min": [0., 60., 120., 180.], "concentration": [0., 4., 4., 2.],
             "time_unit": "min", "unit": "umol/l"}
    target = {"endpoints": ["tmax", "c_last"], "primary_endpoint": "c_last",
              "window_start_h": .5, "window_end_h": 2.5}
    out = predict_requested_endpoints(curve, target, 500.)["endpoints"]
    assert out["tmax"]["value_h"] == 1.
    assert out["c_last"]["value_mg_per_l"] == 1.5
    assert "not a steady-state trough" in out["c_last"]["definition"]


def test_terminal_configuration_cannot_silently_ignore_new_unknown_target_fields():
    with pytest.raises(ValueError):
        _run(target={**_target(), "auto_fit": True})
