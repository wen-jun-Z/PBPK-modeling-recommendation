"""Soft half-life rule, bounded attempts, frozen inputs and evidence barriers."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import uncertain_exchange_design as design
from pbpk_kb.blueprint_policy import capacity_guided_design as capacity
from pbpk_kb.blueprint_policy import learned_exchange_design as frozen
from pbpk_kb.blueprint_policy import permeability_model as exchange
from tests.test_blueprints.test_permeability_model import physiology
from tests.test_blueprints.test_evidence_decision import request as make_request, evidence
from tests.test_blueprints.test_capacity_guided_design import volume
from tests.test_blueprints.test_learned_exchange_design import half_life


def synthetic_solver(monkeypatch, half_function, *, failed_anchor=None, unassessed=False):
    attempts, solves = [], []
    def anchor(physiology, request, records, *, log_parameters, observation):
        ps = math.exp(log_parameters["bidirectional_exchange_scale"])
        attempts.append(ps)
        if failed_anchor and failed_anchor(ps):
            return {"status": "blocked", "reasons": ["synthetic_capacity_conflict"]}
        return {"status": "anchored", "partition_scale": .5,
                "anchored_moments": {"vss_l": 100.}, "log_parameters": deepcopy(log_parameters),
                "capacity_mapping": {"synthetic": True}}
    def simulate(physiology, request, *, log_parameters, times_h, observation):
        ps = math.exp(log_parameters["bidirectional_exchange_scale"])
        solves.append(ps)
        nca = {"status": "unassessed", "reason": "synthetic_short_tail"} if unassessed else {
            "status": "completed", "value_h": half_function(ps)}
        return {"status": "predicted", "nca_summary": {"endpoints": {"terminal_half_life": nca}},
                "exact_moments": {"vss_l": 100.}, "endpoints": {"vss": {"value": 100.}},
                "log_parameters": deepcopy(log_parameters), "full_trajectory": {"time_h": times_h.tolist()}}
    monkeypatch.setattr(capacity, "anchor_partition", anchor)
    monkeypatch.setattr(capacity, "simulate_design", simulate)
    def no_modal(*args, **kwargs):
        raise ArithmeticError("Synthetic test has no modal system")
    monkeypatch.setattr(exchange, "build_system", no_modal)
    return attempts, solves


def run(physiology, *, target=10., q=math.log(2.), **kwargs):
    return design.simulate_uncertain_design(physiology, make_request(), [volume()], [half_life(target)],
        half_log_tolerance=q, times_h=np.linspace(0, 48, 49), **kwargs)


def test_objective_matches_independent_formula_and_natural_log_units():
    q, ps, prediction, target = math.log(2.), 10., 40., 10.
    result = design.soft_half_objective(prediction, target, math.log(ps), q)
    assert result["outside_range_penalty"] == pytest.approx(1.)
    assert result["ps_penalty"] == pytest.approx(.25)
    assert result["objective"] == pytest.approx(1.25)
    assert not result["within_empirical_half_range"]
    assert design.soft_half_objective(15., 10., 0., q)["objective"] == 0.


@pytest.mark.parametrize("width", [True, False, 0., -1., math.inf, math.nan])
def test_width_must_be_finite_positive_not_bool_before_any_trials(physiology, monkeypatch, width):
    monkeypatch.setattr(capacity, "anchor_partition", lambda *a, **k: pytest.fail("No trial allowed"))
    with pytest.raises(ValueError):
        run(physiology, q=width)


def test_ps_one_inside_range_is_not_moved_toward_exact_half_match(physiology, monkeypatch):
    attempts, solves = synthetic_solver(monkeypatch, lambda ps: 15. / ps)
    result = run(physiology)
    audit = result["learned_exchange_audit"]
    assert result["version"] == design.VERSION
    assert audit["chosen_ps"] == 1.
    assert audit["achieved_nca_half_life_h"] == 15.
    assert audit["selected_objective"] == 0.
    assert audit["half_constraint_status"] == "within_empirical_range"
    assert not audit["nca_half_matches_input_at_declared_tolerance"]
    assert not audit["clinical_confidence_interval"]
    assert len(attempts) == len(solves) == audit["search_attempts"] == 33
    assert len(set(attempts)) == 33


def test_selected_objective_is_minimum_including_ps_penalty(physiology, monkeypatch):
    synthetic_solver(monkeypatch, lambda ps: 80. / ps)
    result = run(physiology)
    audit = result["learned_exchange_audit"]
    eligible = [r for r in audit["trials"] if r["status"] == "evaluated"]
    selected = min(eligible, key=lambda r: (r["objective"], abs(r["log_ps"]), r["log_ps"]))
    assert audit["chosen_ps"] == selected["ps_multiplier"]
    assert audit["selected_objective"] < audit["baseline_objective"]
    # Matching the centre exactly needs PS=8 and is not the regularized optimum.
    assert audit["chosen_ps"] < 8.
    for row in eligible:
        expected = max(abs(math.log(row["nca_half_life_h"] / 10.)) - math.log(2.), 0.) ** 2 / math.log(2.) ** 2
        expected += (math.log(row["ps_multiplier"]) / math.log(100.)) ** 2
        assert row["objective"] == pytest.approx(expected)


def test_failed_anchors_consume_all_budget_without_silent_success(physiology, monkeypatch):
    attempts, solves = synthetic_solver(monkeypatch, lambda ps: 10., failed_anchor=lambda ps: True)
    result = run(physiology)
    audit = result["learned_exchange_audit"]
    assert result["status"] == "failed"
    assert result["reason"] == "no_usable_sampled_NCA_model"
    assert len(attempts) == 33 and not solves
    assert audit["failed_search_attempts"] == audit["search_attempts"] == 33
    assert audit["linear_trajectory_calls"] == 0
    assert "synthetic_capacity_conflict" in audit["reasons"]


def test_mixed_failures_still_consume_budget_and_select_feasible_result(physiology, monkeypatch):
    attempts, solves = synthetic_solver(monkeypatch, lambda ps: 12., failed_anchor=lambda ps: ps < .5)
    result = run(physiology, max_trajectory_calls=9)
    audit = result["learned_exchange_audit"]
    assert len(attempts) == 9 and len(solves) < 9
    assert audit["search_attempts"] == 9
    assert audit["linear_trajectory_calls"] == len(solves)
    assert audit["failed_search_attempts"] == 9-len(solves)
    assert result["status"] == "predicted" and audit["chosen_ps"] == 1.


def test_unassessed_nca_is_not_filled_from_modal_value(physiology, monkeypatch):
    synthetic_solver(monkeypatch, lambda ps: 10., unassessed=True)
    result = run(physiology, max_trajectory_calls=5)
    audit = result["learned_exchange_audit"]
    assert result["status"] == "predicted"
    assert audit["chosen_ps"] == 1. and audit["fallback_used"]
    assert audit["achieved_nca_half_life_h"] is None
    assert audit["nca_half_within_empirical_range"] is None
    assert audit["half_constraint_status"] == "unassessed_sampled_nca"
    strict = run(physiology, fallback=False, max_trajectory_calls=5)
    assert strict["status"] == "failed"


def test_strict_mode_uses_empirical_range_not_exact_centre(physiology, monkeypatch):
    synthetic_solver(monkeypatch, lambda ps: 15.)
    assert run(physiology, fallback=False, max_trajectory_calls=3)["status"] == "predicted"
    strict = run(physiology, target=.001, fallback=False, max_trajectory_calls=3)
    assert strict["status"] == "failed"
    assert strict["reason"] == "half_life_outside_empirical_range"


def test_real_solver_preserves_cl_renal_vss_inputs_and_old_module(physiology):
    request = make_request()
    request["evidence"].append(evidence("renal_clearance", 1., "ml/min/kg", context={
        "species": "human", "matrix": "plasma", "route": "intravenous"}))
    logs = {"clearance_scale": math.log(2.)}
    records, halves = [volume()], [half_life(10.)]
    before = deepcopy((physiology, request, logs, records, halves))
    old_function = frozen.simulate_learned_design
    result = design.simulate_uncertain_design(physiology, request, records, halves,
        half_log_tolerance=math.log(10.), log_parameters=logs, max_trajectory_calls=5,
        times_h=np.unique(np.r_[0., np.geomspace(.01, 168., 150)]))
    assert result["status"] == "predicted"
    assert result["exact_moments"]["vss_l"] == pytest.approx(100., rel=1e-8)
    assert result["exact_moments"]["clearance_components_l_h"]["qualified_renal_fixed"] == pytest.approx(4.2)
    assert result["exact_moments"]["effective_total_clearance_l_h"] == pytest.approx(4.2 + (21.-4.2)*2.)
    assert result["log_parameters"]["clearance_scale"] == math.log(2.)
    assert result["mass_balance"]["maximum_relative_error"] < 1e-8
    assert (physiology, request, logs, records, halves) == before
    assert frozen.simulate_learned_design is old_function
    assert frozen.VERSION == "learned-exchange-design-2"
    assert all(t["anchored_vss_l"] == pytest.approx(100.) for t in result["learned_exchange_audit"]["trials"])


@pytest.mark.parametrize("where", ["half", "request", "vss"])
def test_target_evidence_blocks_before_any_simulation(physiology, monkeypatch, where):
    request, volumes, halves = make_request(), [volume()], [half_life()]
    rows = {"request": request["evidence"], "vss": volumes, "half": halves}
    rows[where][0].setdefault("context", {})["study_role"] = "evaluation"
    monkeypatch.setattr(capacity, "anchor_partition", lambda *a, **k: pytest.fail("No trial allowed"))
    result = design.simulate_uncertain_design(physiology, request, volumes, halves, half_log_tolerance=.5)
    assert result["status"] == "failed" and result["learned_exchange_audit"]["search_attempts"] == 0


def test_modal_requests_and_invalid_grids_are_rejected(physiology):
    with pytest.raises(ValueError, match="sampled_nca only"):
        run(physiology, target_basis="modal")
    with pytest.raises(ValueError, match="Prediction grid"):
        design.simulate_uncertain_design(physiology, make_request(), [volume()], [half_life()],
            half_log_tolerance=.5, times_h=[0., 1., .5])
