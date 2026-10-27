"""Independent modal identities, evidence barriers and real PBPK readback."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import learned_exchange_design as design
from pbpk_kb.blueprint_policy import capacity_guided_design as capacity
from pbpk_kb.blueprint_policy import permeability_model as exchange
from tests.test_blueprints.test_permeability_model import physiology
from tests.test_blueprints.test_evidence_decision import request as make_request, evidence
from tests.test_blueprints.test_capacity_guided_design import volume


def half_life(value=10., *, basis="predicted"):
    row = evidence("terminal_half_life", value, "h", basis, context={
        "species": "human", "matrix": "plasma", "route": "intravenous",
        "study_role": "independent_parameters", "independent_of_target": True})
    if basis == "predicted":
        row["prediction_method"] = {"name": "synthetic_predictor", "version": "1",
            "training_overlap": False, "applicability": {
                "target_structure_and_scaffold_excluded": True, "human_IV_parameter_model": True}}
    return row


def known_target(physiology, request, ps=.3, vss=100., logs=None):
    logs = {**(logs or {}), "bidirectional_exchange_scale": math.log(ps)}
    anchor = capacity.anchor_partition(physiology, request, [volume(vss)], log_parameters=logs)
    assert anchor["status"] == "anchored"
    cl = capacity.clearance_inputs(request)
    system = exchange.build_system(physiology, total_clearance_l_h=cl["total_clearance_l_h"],
        renal_clearance_l_h=cl["renal_clearance_l_h"], log_parameters=anchor["log_parameters"])
    mode = design.observable_terminal_mode(system.matrix[:-1, :-1], system.input_vector[:-1], system.observation_vector[:-1])
    return mode["half_life_h"]


@pytest.mark.parametrize("b,c", [([1., 1., 0.], [1., 1., 1.]), ([1., 1., 1.], [1., 1., 0.])])
def test_unexcited_or_unobserved_slow_pool_cannot_define_plasma_terminal_half_life(b, c):
    result = design.observable_terminal_mode(np.diag([-.5, -.05, -.001]), b, c)
    assert result["half_life_h"] == pytest.approx(math.log(2.) / .05)
    assert result["excluded_unexcited_or_unobserved_mode_count"] == 1
    # Independent late two-exponential concentration ratio approaches the mode.
    curve = lambda t: math.exp(-.5 * t) + math.exp(-.05 * t)
    observed_half = math.log(2.) * 20. / math.log(curve(200.) / curve(220.))
    assert result["half_life_h"] == pytest.approx(observed_half, rel=1e-10)


def test_two_compartment_analytic_slow_root_matches_transfer_mode():
    k10, k12, k21 = .2, .7, .03
    matrix = np.array([[-k10 - k12, k21], [k12, -k21]])
    trace_rate = k10 + k12 + k21
    beta = (trace_rate - math.sqrt(trace_rate ** 2 - 4 * k10 * k21)) / 2
    result = design.observable_terminal_mode(matrix, [1., 0.], [1., 0.])
    assert result["decay_rate_per_h"] == pytest.approx(beta, rel=2e-13)
    assert result["half_life_h"] == pytest.approx(math.log(2.) / beta)


def test_small_amplitude_long_lived_mode_is_retained_by_area_not_dropped_as_low_concentration():
    result = design.observable_terminal_mode(np.diag([-1., -1e-14]), [1., 1e-14], [1., 1.])
    assert result["half_life_h"] == pytest.approx(math.log(2.) * 1e14)
    assert result["relative_amplitude"] < 1e-12
    assert result["relative_absolute_modal_area"] == pytest.approx(.5)


@pytest.mark.parametrize("a,b,c", [([[0.]], [1.], [1.]), ([[1.]], [1.], [1.]),
    ([[-1., -1.], [0., -2.]], [1., 0.], [1., 0.]), ([[-1.]], [-1.], [1.]),
    ([[-1.]], [1.], [0.]), ([[-math.inf]], [1.], [1.])])
def test_invalid_or_unobservable_system_is_not_a_valid_tail(a, b, c):
    with pytest.raises((ValueError, ArithmeticError)):
        design.observable_terminal_mode(a, b, c)


def test_ps_is_recovered_without_native_runs_and_reanchors_vss_on_every_trial(physiology, monkeypatch):
    req = make_request(); value = known_target(physiology, req, ps=.3)
    original = deepcopy((physiology, req))
    calls = []
    simulate = capacity.simulate_design
    def record_call(*args, **kwargs):
        calls.append(kwargs)
        return simulate(*args, **kwargs)
    monkeypatch.setattr(capacity, "simulate_design", record_call)
    result = design.simulate_learned_design(physiology, req, [volume()], [half_life(value)])
    audit = result["learned_exchange_audit"]
    assert result["status"] == "predicted" and audit["half_constraint_status"] == "matched"
    assert audit["chosen_ps"] == pytest.approx(.3, rel=2e-5)
    assert audit["achieved_eigen_half_life_h"] == pytest.approx(value, rel=1e-5)
    assert len(calls) == 1 and audit["linear_trajectory_calls"] == 1 and audit["new_native_calls"] == 0
    assert all(row["anchored_vss_l"] == pytest.approx(100., rel=1e-8)
               for row in audit["trials"] if row["status"] == "evaluated")
    assert audit["achieved_model_vss_l"] == pytest.approx(100., rel=1e-8)
    assert audit["achieved_nca_half_life_h"] == result["endpoints"]["terminal_half_life"]["value"]
    assert result["mass_balance"]["maximum_relative_error"] < 1e-8
    assert not result["exact_moments_used_to_fill_NCA"]
    assert (physiology, req) == original
    json.dumps(result, allow_nan=False)


def test_clearance_renal_partition_and_human_inputs_are_protected(physiology):
    req = make_request()
    req["evidence"].append(evidence("renal_clearance", 1., "ml/min/kg", context={
        "species": "human", "matrix": "plasma", "route": "intravenous"}))
    logs = {"clearance_scale": math.log(2.), "bidirectional_exchange_scale": 0.}
    value = known_target(physiology, req, ps=.4, logs=logs)
    original = deepcopy((physiology, req, logs))
    result = design.simulate_learned_design(physiology, req, [volume()], [half_life(value)], log_parameters=logs)
    assert result["status"] == "predicted"
    moment = result["exact_moments"]
    assert moment["clearance_components_l_h"]["qualified_renal_fixed"] == pytest.approx(4.2)
    assert moment["effective_total_clearance_l_h"] == pytest.approx(4.2 + (21. - 4.2) * 2)
    assert moment["blood_to_plasma_ratio"] == physiology["blood_to_plasma_ratio"]
    assert result["log_parameters"]["clearance_scale"] == math.log(2.)
    assert (physiology, req, logs) == original


def test_unreachable_input_uses_declared_fixed_ps_fallback_without_fake_match(physiology):
    result = design.simulate_learned_design(physiology, make_request(), [volume()], [half_life(1e-5)])
    audit = result["learned_exchange_audit"]
    assert result["status"] == "predicted"
    assert audit["half_constraint_status"] == "unreachable_within_declared_search"
    assert audit["fallback_used"] and audit["chosen_ps"] == 1.
    assert audit["achieved_eigen_half_life_h"] != audit["target_half_life_h"]
    assert not audit["chosen_on_boundary"]
    strict = design.simulate_learned_design(physiology, make_request(), [volume()], [half_life(1e-5)], fallback=False)
    assert strict["status"] == "failed" and strict["reason"] == "half_life_unreachable"


def test_capacity_conflict_does_not_override_rbc_or_fake_a_valid_fallback(physiology):
    original = deepcopy(physiology)
    result = design.simulate_learned_design(physiology, make_request(), [volume(1e-3)], [half_life(5.)])
    assert result["status"] == "failed"
    assert result["reason"] == "fixed_PS_fallback_infeasible"
    assert result["learned_exchange_audit"]["linear_trajectory_calls"] == 0
    assert physiology == original


def test_modal_failure_does_not_erase_an_executable_fixed_ps_model(physiology, monkeypatch):
    def unresolved(*args, **kwargs):
        raise ArithmeticError("synthetic ill-conditioned modal decomposition")
    monkeypatch.setattr(design, "observable_terminal_mode", unresolved)
    result = design.simulate_learned_design(physiology, make_request(), [volume()], [half_life(5.)])
    assert result["status"] == "predicted"
    audit = result["learned_exchange_audit"]
    assert audit["fallback_used"] and audit["chosen_ps"] == 1.
    assert audit["half_constraint_status"] == "unassessed_modal_evaluation"
    assert audit["achieved_eigen_half_life_h"] is None
    assert result["exact_moments"]["vss_l"] == pytest.approx(100., rel=1e-8)


def test_held_out_clearance_cannot_be_smuggled_through_the_request(physiology, monkeypatch):
    req = make_request()
    next(row for row in req["evidence"] if row["parameter"] == "total_clearance")["context"]["study_role"] = "evaluation"
    monkeypatch.setattr(capacity, "anchor_partition", lambda *a, **k: pytest.fail("No model search allowed"))
    result = design.simulate_learned_design(physiology, req, [volume()], [half_life()])
    assert result["status"] == "failed"
    assert "request_uses_forbidden_outcome_evidence:total_clearance" in result["learned_exchange_audit"]["reasons"]


@pytest.mark.parametrize("problem", ["target_role", "forbidden_path", "forbidden_hash", "overlap", "scaffold", "species", "curve", "compound", "no_source", "no_independence"])
def test_half_evidence_conflicts_block_before_any_model_evaluation(physiology, monkeypatch, problem):
    req = make_request(); row = half_life()
    if problem == "target_role": row["context"]["study_role"] = "target"
    elif problem == "forbidden_path": req["forbidden_sources"] = [{"path": row["source"]["path"], "sha256": "b" * 64}]
    elif problem == "forbidden_hash": req["forbidden_sources"] = [{"path": "other", "sha256": row["source"]["sha256"]}]
    elif problem == "overlap": row["prediction_method"]["training_overlap"] = True
    elif problem == "scaffold": row["prediction_method"]["applicability"]["target_structure_and_scaffold_excluded"] = False
    elif problem == "species": row["context"]["species"] = "rat"
    elif problem == "curve": row["target_curve"] = [1., 2.]
    elif problem == "compound": row["compound_id"] = "other"
    elif problem == "no_source": row["source"]["sha256"] = "missing"
    elif problem == "no_independence": row["context"]["independent_of_target"] = False
    monkeypatch.setattr(capacity, "anchor_partition", lambda *a, **k: pytest.fail("No model search allowed"))
    result = design.simulate_learned_design(physiology, req, [volume()], [row])
    assert result["status"] == "failed" and result["reason"] == "invalid_or_missing_half_life_evidence"


@pytest.mark.parametrize("value,unit", [(True, "h"), (0., "h"), (-1., "h"), (math.nan, "h"), (math.inf, "h"), (60., "min")])
def test_half_units_and_positive_values_are_explicit(physiology, value, unit):
    row = half_life(value); row["unit"] = unit
    result = design.simulate_learned_design(physiology, make_request(), [volume()], [row])
    assert result["status"] == "failed"


def test_independent_half_has_priority_and_equal_priority_conflicts_block(physiology):
    req = make_request(); target = known_target(physiology, req)
    result = design.simulate_learned_design(physiology, req, [volume()], [half_life(2.), half_life(target, basis="measured")])
    assert result["learned_exchange_audit"]["selected_half_life_evidence"]["basis"] == "measured"
    assert result["half_constraint_status"] == "matched"
    conflict = design.simulate_learned_design(physiology, req, [volume()],
        [half_life(target, basis="measured"), half_life(target * 2, basis="derived")])
    assert conflict["status"] == "failed"
    assert "equal_priority_half_life_conflict" in conflict["learned_exchange_audit"]["reasons"]


def test_short_grid_does_not_fill_nca_from_matched_eigen_half_or_read_sources(physiology, monkeypatch):
    req = make_request(); req["target"]["window_end_h"] = 2.
    value = known_target(physiology, req)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("No input or evaluation reads"))
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **k: pytest.fail("No input or evaluation reads"))
    result = design.simulate_learned_design(physiology, req, [volume()], [half_life(value)], times_h=[0., .1, .5, 1., 2.])
    audit = result["learned_exchange_audit"]
    assert result["status"] == "predicted" and result["half_constraint_status"] == "matched"
    assert result["endpoints"]["terminal_half_life"]["value"] is None
    assert audit["achieved_nca_half_life_h"] is None
    assert audit["predictions_must_lock_before_scoring"] is True
    assert not audit["target_observations_used"]


@pytest.mark.parametrize("kwargs", [{"ps_bounds": (1., .1)}, {"ps_bounds": (0., 10.)},
    {"ps_bounds": (2., 10.)}, {"grid_points": 2}, {"fallback": "yes"}, {"half_relative_tolerance": 2.}])
def test_invalid_engineering_configuration_is_rejected(physiology, kwargs):
    with pytest.raises(ValueError):
        design.simulate_learned_design(physiology, make_request(), [volume()], [half_life()], **kwargs)


def sampled_request():
    req = make_request()
    req["dosing"]["start_h"] = 0.
    return req


def sampled_target(physiology, request, ps=.3, logs=None):
    anchor = capacity.anchor_partition(physiology, request, [volume()],
        log_parameters={**(logs or {}), "bidirectional_exchange_scale": math.log(ps)})
    return capacity.simulate_design(physiology, request, log_parameters=anchor["log_parameters"])


def test_sampled_nca_recovers_real_known_ps_and_reuses_one_whole_selected_trajectory(physiology, monkeypatch):
    req = sampled_request(); known = sampled_target(physiology, req)
    expected = known["endpoints"]["terminal_half_life"]["value"]
    assert expected is not None
    original = deepcopy((physiology, req)); calls, grids = [], []
    simulate, grid = capacity.simulate_design, capacity.prediction_grid
    def counted(*args, **kwargs):
        prediction = simulate(*args, **kwargs)
        calls.append((kwargs, prediction))
        return prediction
    def frozen_grid(*args):
        grids.append(True)
        return grid(*args)
    monkeypatch.setattr(capacity, "simulate_design", counted)
    monkeypatch.setattr(capacity, "prediction_grid", frozen_grid)
    # A completely different modal half must not rank NCA candidates.
    monkeypatch.setattr(design, "observable_terminal_mode", lambda *a, **k: {"half_life_h": 1000000.})
    result = design.simulate_learned_design(physiology, req, [volume()], [half_life(expected)], target_basis="sampled_nca")
    audit = result["learned_exchange_audit"]
    assert result["status"] == "predicted" and audit["half_constraint_status"] == "matched"
    assert audit["chosen_ps"] == pytest.approx(.3, rel=1e-4)
    assert result["endpoints"]["terminal_half_life"]["value"] == pytest.approx(expected, rel=1e-5)
    assert 17 <= len(calls) <= 33 and len(grids) == 1
    assert len(calls) == audit["linear_trajectory_calls"] == result["linear_solver_calls"]
    assert all(kwargs["times_h"] is calls[0][0]["times_h"] for kwargs, _ in calls)
    assert sum(result is prediction for _, prediction in calls) == 1
    assert audit["selected_trajectory_reused_without_rerun"] and not audit["modal_values_used_for_selection"]
    assert audit["achieved_eigen_half_life_h"] == 1000000.
    assert audit["selected_absolute_log_half_target_error"] < audit["baseline_absolute_log_half_target_error"]
    assert all(t["anchored_vss_l"] == pytest.approx(100., rel=1e-8) for t in audit["trials"] if "anchored_vss_l" in t)
    assert result["mass_balance"]["maximum_relative_error"] < 1e-8
    assert (physiology, req) == original
    # Recompute from the returned trajectory, without any target observations.
    trajectory = result["full_trajectory"]
    molar = {"time_min": (np.array(trajectory["time_h"])*60.).tolist(),
        "concentration": (np.array(trajectory["concentration_mg_l"])*1000./300.).tolist(),
        "time_unit": "min", "unit": "umol/l", "observation": result["observer"]}
    recomputed = capacity.summarize_extended_prediction(molar, req)
    assert recomputed["endpoints"]["terminal_half_life"]["value_h"] == result["endpoints"]["terminal_half_life"]["value"]
    assert not audit["clinical_validation"] and not audit["target_observations_used"]
    json.dumps(result, allow_nan=False)


def test_sampled_nca_baseline_and_ties_do_not_change_ps(physiology):
    req = sampled_request(); baseline = sampled_target(physiology, req, ps=1.)
    value = baseline["endpoints"]["terminal_half_life"]["value"]
    result = design.simulate_learned_design(physiology, req, [volume()], [half_life(value)],
        target_basis="sampled_nca", grid_points=5, max_trajectory_calls=9)
    audit = result["learned_exchange_audit"]
    assert audit["chosen_ps"] == 1. and not audit["improved_vs_fixed_ps_baseline"]
    assert audit["half_constraint_status"] == "matched"
    assert result["full_trajectory"] == baseline["full_trajectory"]


def test_sampled_nca_preserves_cl_renal_bp_and_handles_modal_diagnostic_failure(physiology, monkeypatch):
    req = sampled_request()
    req["evidence"].append(evidence("renal_clearance", 1., "ml/min/kg", context={
        "species": "human", "matrix": "plasma", "route": "intravenous"}))
    logs = {"clearance_scale": math.log(2.)}
    expected = sampled_target(physiology, req, ps=.3, logs=logs)["endpoints"]["terminal_half_life"]["value"]
    def unavailable(*a, **k): raise ArithmeticError("test unresolved modal roots")
    monkeypatch.setattr(design, "observable_terminal_mode", unavailable)
    result = design.simulate_learned_design(physiology, req, [volume()], [half_life(expected)],
        target_basis="sampled_nca", log_parameters=logs, grid_points=5, max_trajectory_calls=9)
    assert result["status"] == "predicted"
    assert result["exact_moments"]["clearance_components_l_h"]["qualified_renal_fixed"] == pytest.approx(4.2)
    assert result["effective_total_clearance_l_h"] == pytest.approx(4.2+(21.-4.2)*2)
    assert result["exact_moments"]["blood_to_plasma_ratio"] == physiology["blood_to_plasma_ratio"]
    assert result["exact_moments"]["vss_l"] == pytest.approx(100., rel=1e-8)
    assert result["learned_exchange_audit"]["achieved_eigen_half_life_h"] is None


def test_sampled_nca_short_grid_keeps_real_baseline_without_filling_half(physiology, monkeypatch):
    req = sampled_request(); req["target"]["window_end_h"] = 2.
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("No labels or input files may be opened"))
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **k: pytest.fail("No labels or input files may be opened"))
    result = design.simulate_learned_design(physiology, req, [volume()], [half_life(5.)],
        target_basis="sampled_nca", times_h=[0., .1, .5, 1., 2.], grid_points=5, max_trajectory_calls=9)
    audit = result["learned_exchange_audit"]
    assert result["status"] == "predicted" and result["endpoints"]["terminal_half_life"]["value"] is None
    assert audit["half_constraint_status"] == "unassessed_sampled_nca"
    assert audit["chosen_ps"] == 1. and audit["fallback_used"]
    assert audit["failed_search_attempts"] == audit["search_attempts"]
    assert audit["predictions_must_lock_before_scoring"]


def test_sampled_nca_counts_failed_trajectories_and_anchor_failures_in_budget(physiology, monkeypatch):
    req = sampled_request(); simulate = capacity.simulate_design; calls = []
    def fail_nonbaseline(*args, **kwargs):
        calls.append(kwargs)
        if abs(kwargs["log_parameters"]["bidirectional_exchange_scale"]) > 1e-12:
            raise ArithmeticError("synthetic propagation failure")
        return simulate(*args, **kwargs)
    monkeypatch.setattr(capacity, "simulate_design", fail_nonbaseline)
    result = design.simulate_learned_design(physiology, req, [volume()], [half_life(5.)],
        target_basis="sampled_nca", grid_points=5, max_trajectory_calls=9)
    audit = result["learned_exchange_audit"]
    assert result["status"] == "predicted" and audit["chosen_ps"] == 1.
    assert len(calls) == audit["linear_trajectory_calls"] == audit["search_attempts"] <= 9
    assert audit["failed_search_attempts"] == len(calls)-1
    blocked = design.simulate_learned_design(physiology, req, [volume(.001)], [half_life(5.)],
        target_basis="sampled_nca", grid_points=5, max_trajectory_calls=9)
    assert blocked["status"] == "failed"
    assert blocked["learned_exchange_audit"]["search_attempts"] == 5
    assert blocked["learned_exchange_audit"]["linear_trajectory_calls"] == 0


def test_sampled_nca_unreachable_half_is_approximate_without_fake_match(physiology):
    result = design.simulate_learned_design(physiology, sampled_request(), [volume()], [half_life(1e-5)],
        target_basis="sampled_nca", grid_points=5, max_trajectory_calls=9)
    audit = result["learned_exchange_audit"]
    assert result["status"] == "predicted" and audit["half_constraint_status"] in {"improved_not_matched", "baseline_not_matched"}
    assert not audit["nca_half_matches_input_at_declared_tolerance"]
    assert audit["selected_absolute_log_half_target_error"] <= audit["baseline_absolute_log_half_target_error"]
    assert result["endpoints"]["terminal_half_life"]["value"] != 1e-5


@pytest.mark.parametrize("kwargs", [{"target_basis": "clinical_truth"}, {"max_trajectory_calls": 34},
    {"max_trajectory_calls": True}, {"max_trajectory_calls": 2},
    {"target_basis": "sampled_nca", "ps_bounds": (.01, .5), "log_parameters": {"bidirectional_exchange_scale": math.log(.1)}}])
def test_sampled_nca_rejects_invalid_basis_budget_and_missing_baseline(physiology, kwargs):
    with pytest.raises(ValueError):
        design.simulate_learned_design(physiology, sampled_request(), [volume()], [half_life()], **kwargs)


def test_sampled_nca_forbidden_half_is_rejected_before_search(physiology, monkeypatch):
    row = half_life(); row["context"]["study_role"] = "target"
    monkeypatch.setattr(capacity, "simulate_design", lambda *a, **k: pytest.fail("No target-based search"))
    result = design.simulate_learned_design(physiology, sampled_request(), [volume()], [row], target_basis="sampled_nca")
    assert result["status"] == "failed"
    assert result["learned_exchange_audit"]["linear_trajectory_calls"] == 0
