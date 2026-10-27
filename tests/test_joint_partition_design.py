"""Global partition mapping checks use synthetic physiology, not clinical data."""
from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pbpk_kb.blueprint_policy import joint_partition_design as joint
from pbpk_kb.blueprint_policy import capacity_guided_design as old_capacity
from pbpk_kb.blueprint_policy import learned_exchange_design as old_learned
from pbpk_kb.blueprint_policy import permeability_model as exchange
from pbpk_kb.blueprint_policy import perfusion_export as native
from tests.test_schmitt_physiology import _fixture
from tests.test_blueprints.test_evidence_decision import request as make_request, evidence
from tests.test_blueprints.test_capacity_guided_design import volume
from tests.test_blueprints.test_learned_exchange_design import half_life


def full_physiology(rbc=2.):
    plan, rows, processes = _fixture()
    values = {row["path"]: row for row in rows}
    values[plan["global_paths"]["rbc_kp_plasma"]]["value"] = rbc
    values[plan["global_paths"]["blood_to_plasma_ratio"]]["value"] = .6 + .4*rbc
    return {"version": native.VERSION, **native.assemble_export(plan, rows, processes)}


@pytest.fixture
def physiology():
    return full_physiology()


def test_alpha_one_is_identical_generator_and_native_input_is_immutable(physiology):
    original = deepcopy(physiology)
    old = exchange.build_system(physiology, total_clearance_l_h=21., renal_clearance_l_h=4.2)
    new = joint.build_system(physiology, total_clearance_l_h=21., renal_clearance_l_h=4.2)
    np.testing.assert_allclose(new.matrix, old.matrix, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(new.observation_vector, old.observation_vector, rtol=1e-13)
    np.testing.assert_allclose(new.blood_equivalent_capacities_l, old.blood_equivalent_capacities_l)
    assert physiology == original
    assert new.metadata["parameters"]["partition_scale"] == 1.


@pytest.mark.parametrize("alpha", [.0001, .2, 3., 1000.])
def test_common_scale_preserves_volumes_flows_fu_clearance_and_relative_kp(physiology, alpha):
    frozen = deepcopy(physiology)
    derived = joint.transform_physiology(physiology, alpha)
    assert derived["blood_to_plasma_ratio"] == pytest.approx(.6 + .4*alpha*2.)
    assert derived["rbc_kp_plasma"] == pytest.approx(alpha*2.)
    for key in ["fraction_unbound_plasma", "clearance", "circulation", "peripheral_venous_weights", "hematocrit"]:
        assert derived[key] == physiology[key]
    for before, after in zip([*physiology["organs"], *physiology["blood"].values()],
                             [*derived["organs"], *derived["blood"].values()]):
        for key in before:
            if "volume" in key or "flow" in key:
                assert after[key] == before[key]
        for old, new in zip(before["components"], after["components"]):
            assert new["volume_l"] == old["volume_l"]
            assert new["kp_plasma"] == pytest.approx(old["kp_plasma"] * (1 if old["kind"] == "Plasma" else alpha))
            assert new["capacity_l"] == pytest.approx(new["volume_l"]*new["kp_plasma"])
    assert physiology == frozen
    assert derived["joint_partition_mapping"]["clipping_applied"] is False


@pytest.mark.parametrize("alpha", [0., -1., True, math.nan, math.inf])
def test_invalid_scale_or_incomplete_native_components_cannot_be_guessed(physiology, alpha):
    with pytest.raises(ValueError):
        joint.transform_physiology(physiology, alpha)


def test_inconsistent_native_aggregate_and_double_mapping_are_rejected(physiology):
    bad = deepcopy(physiology)
    bad["organs"][0]["vascular_capacity_l"] += 1.
    with pytest.raises(ValueError, match="aggregate"):
        joint.transform_physiology(bad, 1.)
    with pytest.raises(ValueError, match="twice"):
        joint.transform_physiology(joint.transform_physiology(physiology, .2), .2)
    bad = deepcopy(physiology); bad["blood_to_plasma_ratio"] += 1.
    with pytest.raises(ValueError, match="equilibrium"):
        joint.transform_physiology(bad, .2)


@pytest.mark.parametrize("duration", [0., .5])
def test_mass_conservation_and_fixed_renal_component(physiology, duration):
    logs = {"partition_scale": math.log(.2), "bidirectional_exchange_scale": math.log(.1),
            "clearance_scale": math.log(2.)}
    p = joint.simulate(physiology, times_h=np.r_[0., np.geomspace(1e-5, 168., 120)],
        dose_mg=100., infusion_duration_h=duration, total_clearance_l_h=21., renal_clearance_l_h=4.2,
        log_parameters=logs, return_amounts=True)
    np.testing.assert_allclose(p["amounts_mg"].sum(axis=1)+p["eliminated_mg"], p["delivered_mg"], atol=1e-8)
    assert p["mass_balance"]["maximum_relative_error"] < 1e-8
    assert p["model"]["clearance_components_l_h"]["qualified_renal_fixed"] == 4.2
    assert p["effective_total_clearance_l_h"] == pytest.approx(4.2+(21.-4.2)*2.)
    assert p["parameters"]["partition_scale"] == pytest.approx(.2)
    assert p["amounts_mg"].min() >= 0.


@pytest.mark.parametrize("observer", ["venous", "native_peripheral_venous"])
def test_root_matches_exact_observer_vss_and_independent_integrals(physiology, observer):
    logs = {"partition_scale": math.log(.3), "bidirectional_exchange_scale": math.log(.2)}
    expected = joint.exact_moments(physiology, total_clearance_l_h=21., log_parameters=logs, observation=observer)
    target = expected["vss_l"]
    anchored = joint.anchor_partition(physiology, make_request(), [volume(target)],
        log_parameters={"bidirectional_exchange_scale": logs["bidirectional_exchange_scale"]}, observation=observer)
    assert anchored["status"] == "anchored"
    assert anchored["partition_scale"] == pytest.approx(.3, rel=1e-8)
    assert anchored["anchored_moments"]["vss_l"] == pytest.approx(target, rel=1e-8)
    if observer != "venous":
        assert target != pytest.approx(expected["equilibrium_plasma_capacity_l"], rel=1e-5)
    system = joint.build_system(physiology, total_clearance_l_h=21., log_parameters=logs, observation=observer)
    a, b, c = system.matrix[:-1, :-1], system.input_vector[:-1], system.observation_vector[:-1]
    n = len(b)
    def rhs(t, state):
        concentration = c @ state[:n]
        return np.r_[a @ state[:n], concentration, t*concentration]
    integrated = solve_ivp(rhs, (0., 100.), np.r_[b, 0., 0.], method="Radau", rtol=2e-9, atol=2e-12)
    assert integrated.success
    auc, aumc = integrated.y[n:, -1]
    assert aumc/auc**2 == pytest.approx(target, rel=2e-7)


@pytest.mark.parametrize("rbc,target", [(27.9, 20.), (1822.77, 32.)])
def test_high_rbc_capacity_conflict_can_be_resolved_without_changing_anatomy(rbc, target):
    physiology = full_physiology(rbc)
    old = old_capacity.anchor_partition(physiology, make_request(), [volume(target)])
    assert old["status"] == "blocked"
    assert "vss_target_at_or_below_immutable_vascular_observer_floor" in old["reasons"]
    new = joint.anchor_partition(physiology, make_request(), [volume(target)])
    assert new["status"] == "anchored"
    assert 0 < new["partition_scale"] < 1.
    assert new["anchored_moments"]["vss_l"] == pytest.approx(target, rel=1e-8)
    assert new["capacity_mapping"]["clipping_applied"] is False


def test_unreachable_below_plasma_volume_is_failure_not_clipped_solution(physiology):
    result = joint.anchor_partition(physiology, make_request(), [volume(.001)])
    assert result["status"] == "blocked"
    assert result["anchored_moments"] is None and result["reasons"]
    assert result["capacity_mapping"]["clipping_applied"] is False


@pytest.mark.parametrize("where", ["request", "vss", "half"])
def test_target_or_unbound_evidence_rejected_before_any_capacity_search(physiology, monkeypatch, where):
    request, volumes, halves = make_request(), [volume()], [half_life()]
    {"request": request["evidence"], "vss": volumes, "half": halves}[where][0].setdefault("context", {})["study_role"] = "evaluation"
    monkeypatch.setattr(joint, "anchor_partition", lambda *a, **k: pytest.fail("No capacity search permitted"))
    result = joint.simulate_joint_design(physiology, request, volumes, halves)
    assert result["status"] == "failed"
    assert result["joint_partition_audit"]["search_attempts"] == 0


def test_independent_bp_keeps_old_scheme_without_global_mutation(physiology, monkeypatch):
    request = make_request()
    request["evidence"].append(evidence("blood_to_plasma_ratio", 1.4, "fraction", context={
        "species": "human", "study_role": "independent_parameters", "independent_of_target": True}))
    sentinel = {"status": "failed", "reason": "old_scheme_sentinel"}
    calls = []
    def old(*args, **kwargs):
        calls.append((args, kwargs)); return deepcopy(sentinel)
    monkeypatch.setattr(old_learned, "simulate_learned_design", old)
    monkeypatch.setattr(joint, "anchor_partition", lambda *a, **k: pytest.fail("Keep measured blood partition"))
    result = joint.simulate_joint_design(physiology, request, [volume()], [half_life()])
    assert result["reason"] == sentinel["reason"] and len(calls) == 1
    assert result["joint_partition_audit"]["mode"] == "retained_independent_blood_partition"
    request["evidence"][-1]["value"] = 2.
    assert joint.simulate_joint_design(physiology, request, [volume()], [half_life()])["reason"] == "invalid_independent_blood_partition"
    assert len(calls) == 1


def test_full_design_budget_immutability_and_no_false_bp_protection(physiology):
    request = make_request()
    request["evidence"].append(evidence("renal_clearance", 1., "ml/min/kg", context={
        "species": "human", "matrix": "plasma", "route": "intravenous"}))
    vss = [volume(100.)]; halves = [half_life(10.)]
    before = deepcopy((physiology, request, vss, halves))
    frozen = (old_capacity.anchor_partition, old_capacity.simulate_design, old_learned.simulate_learned_design, exchange.build_system)
    result = joint.simulate_joint_design(physiology, request, vss, halves, max_trajectory_calls=5,
        times_h=np.r_[0., np.geomspace(.001, 168., 140)])
    audit = result["learned_exchange_audit"]
    assert result["status"] == "predicted"
    assert audit["search_attempts"] <= 5 and audit["linear_trajectory_calls"] <= 5
    assert result["exact_moments"]["vss_l"] == pytest.approx(100., rel=1e-8)
    assert result["exact_moments"]["clearance_components_l_h"]["qualified_renal_fixed"] == pytest.approx(4.2)
    assert result["mass_balance"]["maximum_relative_error"] < 1e-8
    assert "blood_to_plasma_ratio" not in audit["protected"]
    assert result["joint_partition_audit"]["new_learning_calls"] == 0
    assert result["joint_partition_audit"]["new_native_calls"] == 0
    assert (physiology, request, vss, halves) == before
    assert frozen == (old_capacity.anchor_partition, old_capacity.simulate_design, old_learned.simulate_learned_design, exchange.build_system)


def test_failed_capacity_attempts_consume_the_same_budget(physiology, monkeypatch):
    attempts = []
    def blocked(*args, **kwargs):
        attempts.append(kwargs["log_parameters"]["bidirectional_exchange_scale"])
        return {"status": "blocked", "reasons": ["synthetic_capacity_failure"]}
    monkeypatch.setattr(joint, "anchor_partition", blocked)
    result = joint.simulate_joint_design(physiology, make_request(), [volume()], [half_life()],
        max_trajectory_calls=5, times_h=[0., 1., 24.])
    assert result["status"] == "failed"
    assert result["learned_exchange_audit"]["search_attempts"] == len(attempts) == 5
    assert result["learned_exchange_audit"]["linear_trajectory_calls"] == 0
