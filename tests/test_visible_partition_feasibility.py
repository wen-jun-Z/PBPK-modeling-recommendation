"""Nonnegative mass bound, observer identity, source protection and budget."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import blood_partition_calibration as blood
from pbpk_kb.blueprint_policy import distribution_calibration_optimizer as old
from pbpk_kb.blueprint_policy import visible_partition_feasibility as repair
from test_blood_partition_calibration import physiology, req, region, measured


def high_partition(phys):
    out = deepcopy(phys)
    kp = 10000.
    out.update(rbc_kp_plasma=kp, blood_to_plasma_ratio=.53 + .47 * kp,
               equilibrium_blood_to_plasma_ratio_check=.53 + .47 * kp)
    for kind, row in out["blood"].items():
        out["blood"][kind] = region(row["name"], row["vascular_volume_l"], 0., kp)
    out["organs"] = [{**region(o["name"], o["vascular_volume_l"], o["extravascular_volume_l"], kp),
                       "flow_l_h": o["flow_l_h"], "flow_l_min": o["flow_l_min"]} for o in out["organs"]]
    return out


def training(phys, request, *, absolute_rbc=1.):
    times = np.array([.05, .15, .5, 2.])
    prediction = blood.simulate(phys, request, times,
        [0., 0., 0., math.log(absolute_rbc / phys["rbc_kp_plasma"])])
    return {"compound_id": request["compound_id"], "human": deepcopy(request["human"]),
            "dosing": deepcopy(request["dosing"]), "matrix": "plasma", "time_unit": "min", "unit": "umol/l",
            "observations": [{"time_min": float(t * 60), "concentration_umol_per_l": float(c * 1000 / 400)}
                             for t, c in zip(times, prediction["concentration_mg_l"])]}


def test_proved_conflict_opens_neutral_rbc_and_preserves_inputs(physiology, req):
    phys = high_partition(physiology)
    visible = training(phys, req)
    before = deepcopy((phys, req, visible))
    audit = repair.feasibility(phys, req, visible)
    assert audit["repair_required"]
    assert audit["before"]["violations"] > 0
    assert audit["after"]["violations"] == 0
    assert audit["minimum_rbc_scale"] == pytest.approx(1e-4)
    assert audit["after"]["effective_rbc_partition"] == pytest.approx(1.)
    assert audit["after"]["blood_to_plasma_ratio"] == pytest.approx(1.)
    assert (phys, req, visible) == before
    assert audit["new_fitted_parameters"] == 0


@pytest.mark.parametrize("duration", [0., 30.])
@pytest.mark.parametrize("peripheral", [False, True])
def test_bound_uses_true_observer_and_delivered_dose(physiology, req, duration, peripheral):
    phys = high_partition(physiology)
    if not peripheral:
        phys["peripheral_venous_weights"] = {}
    req["dosing"]["infusion_minutes"] = duration
    visible = training(phys, req)
    audit = repair.feasibility(phys, req, visible)
    bound = audit["after"]
    maximum = bound["maximum_observer_coefficient_l_inverse"]
    if not peripheral:
        assert maximum == pytest.approx(1. / phys["blood"]["venous"]["vascular_volume_l"])
    else:
        expected = max(phys["peripheral_venous_weights"].get(o["name"], 0.) / o["vascular_volume_l"]
                       for o in phys["organs"])
        assert maximum == pytest.approx(expected)
    for row in bound["visible_points"]:
        delivered = 100. if duration == 0 else 100 * min(1., row["time_h"] * 60 / duration)
        assert row["concentration_upper_bound_mg_l"] == pytest.approx(delivered * maximum)


def test_measured_partition_blocks_repair(physiology, req):
    phys = high_partition(physiology)
    visible = training(phys, req)
    measured(req, "blood_to_plasma_ratio", phys["blood_to_plasma_ratio"])
    audit = repair.feasibility(phys, req, visible)
    assert audit["protected_measured_partition"]
    assert not audit["repair_required"]
    assert audit["minimum_rbc_scale"] == 1.
    assert audit["remaining_visible_mass_conflict"]


def test_measured_partition_refuses_free_axis_fit_before_any_prediction(physiology, req):
    visible = training(physiology, req)
    measured(req, "blood_to_plasma_ratio", physiology["blood_to_plasma_ratio"])
    calls = []
    def predict(x, times):
        calls.append(x)
        return blood.simulate(physiology, req, times, x)
    with pytest.raises(ValueError, match="Four-axis fitting cannot modify measured partition"):
        repair.fit(predict, req, visible, physiology, budget=32, starts=2)
    assert calls == []


def test_inconsistent_or_unqualified_measurement_fails_closed(physiology, req):
    visible = training(physiology, req)
    measured(req, "rbc_kp_plasma", physiology["rbc_kp_plasma"], basis="predicted")
    with pytest.raises(ValueError, match="explicit blood-partition"):
        repair.feasibility(physiology, req, visible)


def test_sane_native_family_keeps_every_bound_and_start(physiology, req):
    visible = training(physiology, req, absolute_rbc=physiology["rbc_kp_plasma"])
    audit = repair.feasibility(physiology, req, visible)
    assert not audit["repair_required"]
    original = old._initialization(req, physiology, "blood_partition", 16)
    new = repair.initialization(req, physiology, "blood_partition", 16, audit)
    for a, b in zip(original[:3], new[:3]):
        np.testing.assert_array_equal(a, b)
    assert original[3:] == new[3:]


def test_extended_bound_only_changes_one_coordinate_and_one_start(physiology, req):
    phys = high_partition(physiology)
    visible = training(phys, req)
    audit = repair.feasibility(phys, req, visible)
    a = old._initialization(req, phys, "blood_partition", 16)
    b = repair.initialization(req, phys, "blood_partition", 16, audit)
    np.testing.assert_array_equal(a[0][:3], b[0][:3])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2][:-1], b[2][:-1])
    np.testing.assert_array_equal(a[2][-1, :3], b[2][-1, :3])
    assert b[0][3] == pytest.approx(math.log(1e-4))
    assert b[2][-1, 3] == pytest.approx(math.log(1e-4))


def test_unaffected_fit_matches_frozen_and_does_not_mutate_globals(physiology, req):
    visible = training(physiology, req, absolute_rbc=physiology["rbc_kp_plasma"])
    original_initializer = old.fit.__globals__["_initialization"]
    predict = lambda x, times: blood.simulate(physiology, req, times, x)
    a = old.fit(predict, req, visible, physiology, variant="blood_partition", budget=32, starts=2)
    b = repair.fit(predict, req, visible, physiology, budget=32, starts=2)
    for key in ("best", "starts", "calls", "failures", "call_limit", "parameter_lower", "parameter_upper"):
        assert a[key] == b[key]
    assert old.fit.__globals__["_initialization"] is original_initializer


def test_repair_fit_keeps_fixed_budget_and_same_prediction_family(physiology, req):
    phys = high_partition(physiology)
    visible = training(phys, req)
    calls = []
    def predict(x, times):
        calls.append(x.copy())
        return blood.simulate(phys, req, times, x)
    result = repair.fit(predict, req, visible, phys, budget=32, starts=2)
    assert result["parameterization_changed"]
    assert result["calls"] == len(calls) <= 32
    assert len(result["starts"]) == 2
    assert result["parameter_lower"][3] == pytest.approx(math.log(1e-4))
    assert result["starts"][-1]["initial_log_parameters"][3] == pytest.approx(math.log(1e-4))
    assert result["target_observations_used"] is False


def test_no_reference_access_or_full_curve_initialization(physiology, req):
    phys = high_partition(physiology)
    visible = training(phys, req)
    expected = repair.feasibility(phys, req, visible)
    req["forbidden_sources"] = [{"path": "/must-not-read", "sha256": "0" * 64}]
    visible["unrevealed_observations"] = [{"time_min": 5., "concentration_umol_per_l": 1e12}]
    assert repair.feasibility(phys, req, visible) == expected


def test_still_impossible_observation_remains_flagged(physiology, req):
    phys = high_partition(physiology)
    visible = training(phys, req)
    for row in visible["observations"]:
        row["concentration_umol_per_l"] *= 1e10
    audit = repair.feasibility(phys, req, visible)
    assert audit["repair_required"] and audit["remaining_visible_mass_conflict"]


def test_visible_unit_mismatch_rejected(physiology, req):
    visible = training(physiology, req)
    visible["unit"] = "mg/l"
    with pytest.raises(ValueError, match="declare plasma"):
        repair.feasibility(physiology, req, visible)
