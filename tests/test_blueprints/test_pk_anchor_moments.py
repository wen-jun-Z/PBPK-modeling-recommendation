"""Analytic-model tests; these synthetic trajectories are not clinical data."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprints.pk_anchor_moments import model_integral_pk


def request(infusion_minutes=0):
    return {"dosing": {"route": "intravenous", "start_h": 0., "dose_mg": 10.,
                       "infusion_minutes": infusion_minutes},
            "evidence": [{"parameter": "molecular_weight", "value": 100.}]}


def curve(times, mg_l):
    return {"time_min": [float(value * 60) for value in times],
            "concentration": [float(value * 10) for value in mg_l],
            "time_unit": "min", "unit": "umol/l"}


def exponential(horizon=30):
    times = np.linspace(0, horizon, int(horizon * 100) + 1)
    return curve(times, np.exp(-times))


def test_bolus_integrals_recover_analytic_clearance_and_volume_without_half_life():
    native, req = exponential(), request()
    original = deepcopy((native, req))
    result = model_integral_pk(native, req)
    assert result["status"] == "ok"
    assert result["clearance_l_h"] == pytest.approx(10., rel=2e-5)
    assert result["vss_l"] == pytest.approx(10., rel=3e-5)
    assert result["diagnostics"]["dropped_numeric_suffix_points"] > 0
    assert result["diagnostics"]["half_life_reported"] is False
    assert "half_life" not in result
    assert (native, req) == original


def test_constant_rate_infusion_correction_recovers_analytic_volume():
    times = np.linspace(0, 30, 3001)
    concentration = np.where(times <= 1, 1 - np.exp(-times), (1 - math.exp(-1)) * np.exp(-(times - 1)))
    result = model_integral_pk(curve(times, concentration), request(60))
    assert result["status"] == "ok"
    assert result["clearance_l_h"] == pytest.approx(10., rel=2e-5)
    assert result["vss_l"] == pytest.approx(10., rel=3e-5)
    assert result["diagnostics"]["infusion_correction_h"] == .5


def test_converged_small_slow_component_does_not_need_a_half_life_fit_span():
    times = np.linspace(0, 1000, 100001)
    # The slow component has negligible mass and moment but a very long slope.
    # Its tail remains above the numerical floor and spans <1 half-life.
    concentration = np.exp(-times / 5) + 1e-7 * np.exp(-times / 1000)
    result = model_integral_pk(curve(times, concentration), request())
    assert result["status"] == "ok"
    assert result["diagnostics"]["policy"]["minimum_span_half_lives"] is None
    assert result["diagnostics"]["aumc_remainder_fraction"] <= .01


@pytest.mark.parametrize("horizon", [2, 5, 8])
def test_short_horizon_fails_even_when_exponential_tail_is_perfect(horizon):
    result = model_integral_pk(exponential(horizon), request())
    assert result["status"] == "not_estimable"
    assert result["clearance_l_h"] is None and result["vss_l"] is None
    assert "fraction" in result["reason"]


def test_meaningful_slow_tail_moment_is_not_excused_by_small_concentration():
    times = np.linspace(0, 1000, 100001)
    result = model_integral_pk(curve(times, np.exp(-times) + 1e-6 * np.exp(-times / 1000)), request())
    assert result["status"] == "not_estimable"
    assert result["clearance_l_h"] is None


@pytest.mark.parametrize("kind", ["flat", "rising", "rebound", "unstable"])
def test_untrustworthy_tails_are_rejected(kind):
    times = np.linspace(0, 30, 3001)
    concentration = np.exp(-times)
    if kind == "flat":
        concentration[times > 20] = 1e-5
    elif kind == "rising":
        concentration[times > 20] = 1e-5 * np.exp((times[times > 20] - 20) * .01)
    elif kind == "rebound":
        concentration[times > 25] = 1e-6
    else:
        concentration = np.exp(-times / 10 - (times / 10) ** 2)
    result = model_integral_pk(curve(times, concentration), request())
    assert result["status"] == "not_estimable"


@pytest.mark.parametrize("kind", ["oral", "late_dose", "multiple", "nonlinear", "observed", "units", "negative", "bool", "zero_peak"])
def test_invalid_context_or_curve_does_not_return_anchors(kind):
    native, req = exponential(), request()
    if kind == "oral": req["dosing"]["route"] = "oral"
    elif kind == "late_dose": req["dosing"]["start_h"] = 1.
    elif kind == "multiple": req["dosing"]["single_dose"] = False
    elif kind == "nonlinear": req["linear_kinetics"] = False
    elif kind == "observed": native["source"] = "observed"
    elif kind == "units": native["unit"] = "mg/l"
    elif kind == "negative": native["concentration"][-1] = -.1
    elif kind == "bool": req["dosing"]["dose_mg"] = True
    elif kind == "zero_peak": native["concentration"] = [0.] * len(native["concentration"])
    result = model_integral_pk(native, req)
    assert result["status"] == "not_estimable"
    assert result["reason"]


def test_nonzero_infusion_predose_and_missing_infusion_samples_are_rejected():
    result = model_integral_pk(exponential(), request(60))
    assert result["status"] == "not_estimable"
    assert "predose" in result["reason"]


def test_default_joint_gate_is_identical_to_explicit_require_vss_true():
    native, req = exponential(), request()
    assert model_integral_pk(native, req) == model_integral_pk(native, req, require_vss=True)


def test_cl_only_can_use_converged_auc_without_assigning_unconverged_vss_or_mrt():
    times = np.linspace(0, 1000, 100001)
    native = curve(times, np.exp(-times) + 1e-6 * np.exp(-times / 1000))
    original = deepcopy(native)
    joint = model_integral_pk(native, request())
    cl_only = model_integral_pk(native, request(), require_vss=False)
    assert joint["status"] == "not_estimable"
    assert joint["clearance_l_h"] is None and joint["vss_l"] is None
    assert cl_only["status"] == "ok"
    assert cl_only["clearance_l_h"] == pytest.approx(10. / (1 + 1e-3), rel=2e-5)
    assert cl_only["vss_l"] is None and cl_only["mrt_h"] is None
    assert cl_only["endpoint_statuses"] == {"clearance": "ok", "vss": "not_estimable", "mrt": "not_estimable"}
    assert "aumc" in cl_only["vss_unavailable_reason"]
    assert cl_only["diagnostics"]["auc_gates_passed"] is True
    assert cl_only["diagnostics"]["aumc_gates_passed"] is False
    assert cl_only["diagnostics"]["required_endpoints"] == ["clearance"]
    assert "half_life" not in cl_only
    assert native == original


def test_cl_only_keeps_the_original_moment_thresholds_and_fully_qualified_values():
    native, req = exponential(), request()
    joint = model_integral_pk(native, req)
    cl_only = model_integral_pk(native, req, require_vss=False)
    for endpoint in ("clearance_l_h", "vss_l", "mrt_h"):
        assert cl_only[endpoint] == joint[endpoint]
    assert cl_only["diagnostics"]["policy"] == joint["diagnostics"]["policy"]
    assert cl_only["diagnostics"]["policy"]["maximum_aumc_remainder_fraction"] == .01
    assert cl_only["diagnostics"]["policy"]["maximum_final_half_aumc_fraction"] == .01


@pytest.mark.parametrize("horizon", [2, 5, 8])
def test_cl_only_still_rejects_unconverged_auc(horizon):
    result = model_integral_pk(exponential(horizon), request(), require_vss=False)
    assert result["status"] == "not_estimable"
    assert result["clearance_l_h"] is None and result["vss_l"] is None
    assert "auc" in result["reason"] and "fraction" in result["reason"]


@pytest.mark.parametrize("kind", ["flat", "rising", "rebound", "unstable"])
def test_cl_only_does_not_bypass_common_tail_quality_checks(kind):
    times = np.linspace(0, 30, 3001)
    concentration = np.exp(-times)
    if kind == "flat":
        concentration[times > 20] = 1e-5
    elif kind == "rising":
        concentration[times > 20] = 1e-5 * np.exp((times[times > 20] - 20) * .01)
    elif kind == "rebound":
        concentration[times > 25] = 1e-6
    else:
        concentration = np.exp(-times / 10 - (times / 10) ** 2)
    result = model_integral_pk(curve(times, concentration), request(), require_vss=False)
    assert result["status"] == "not_estimable"
    assert result["clearance_l_h"] is None


@pytest.mark.parametrize("flag", [0, 1, None, "false"])
def test_requirement_flag_is_explicit_boolean(flag):
    result = model_integral_pk(exponential(), request(), require_vss=flag)
    assert result["status"] == "not_estimable"
    assert "boolean" in result["reason"]
