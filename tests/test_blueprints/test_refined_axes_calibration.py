"""Apply the established isolation/restart contracts to the versioned adapter."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import refined_axes_calibration as cal
from pbpk_kb.blueprint_policy import native_axes_runtime


_spec = importlib.util.spec_from_file_location(
    "_refined_native_contracts", Path(__file__).with_name("test_axes_calibration.py"))
contracts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contracts)


@pytest.fixture
def case(tmp_path, monkeypatch):
    # The mock native runtime and the assertions are shared, while the real
    # versioned fit/predict/cleanup implementation is exercised in each test.
    monkeypatch.setattr(contracts, "cal", cal)
    return contracts.case.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("contract", [
    "test_base_and_all_trials_share_one_declared_budget",
    "test_native_predicted_status_and_loss_are_aligned_with_optimizer",
    "test_cached_replay_after_pause_is_deterministic",
    "test_infrastructure_retry_and_solver_entry_both_count_against_budget",
    "test_cleanup_preserves_selected_failed_base_and_compact_trials_and_allows_reuse",
    "test_forbidden_comparison_source_cannot_be_used_as_training_data",
    "test_development_conditions_must_match_fitting_request",
    "test_prediction_rejects_changed_drug_evidence_before_native_work",
])
def test_refined_adapter_contract(case, contract):
    getattr(contracts, contract)(case)


@pytest.mark.parametrize("change", ["code", "source", "budget", "active_parameters"])
def test_changed_inputs_cannot_resume(case, change):
    contracts.test_changed_inputs_or_code_cannot_reuse_locked_mapping(case, change)


def test_nested_initialization_is_declared_and_config_is_immutable(case):
    result = contracts.fit(case, max_evaluations=5)
    input_path = case[0] / "fit/input.json"
    fixed = cal._read(input_path)
    assert fixed["external_warm_start_used"] is False
    assert result["optimizer_config"] == fixed["optimizer_config"] == cal.CONFIG
    assert "same visible development subset" in fixed["initialization"]
    fixed["optimizer_config"]["loss_changed"] = True
    cal.save(input_path, fixed)
    with pytest.raises(ValueError, match="Resume sources, code, budget or parameters changed"):
        contracts.fit(case, max_evaluations=5)


def test_target_pause_after_base_does_not_submit_scaled_native_trial(case):
    contracts.fit(case, max_evaluations=3)
    root, request, _, _, state = case
    previous_trials = len(state["trial_calls"])
    previous_bases = state["base_calls"]
    checks = 0

    def pause():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise cal.InfrastructurePause("paused between target solves")

    with pytest.raises(cal.InfrastructurePause, match="between target solves"):
        cal.predict_mapping(request=request, mapping_path=root / "fit/mapping_locked.json",
            output_dir=root / "target", project_root=root, before_trial=pause)
    assert checks == 2
    assert state["base_calls"] == previous_bases + 1
    assert len(state["trial_calls"]) == previous_trials
    assert (root / "target/base/completion.json").exists()
    assert not (root / "target/execution").exists()


@pytest.fixture
def coupled_case(case, monkeypatch):
    """A curved synthetic valley exercises the entire 18 + 7 + 22 budget."""
    original_native = native_axes_runtime.run_axes_trial
    values = np.array([row["concentration_umol_per_l"] for row in case[3]["observations"]])

    def coupled_native(*, scales, **kwargs):
        result = original_native(scales=scales, **kwargs)
        if result["status"] == "predicted":
            x, y, w = np.log([scales[name] for name in cal.axes_optimizer.PARAMETERS]) / np.log(
                [cal.axes_optimizer.BOUNDS[name][1] for name in cal.axes_optimizer.PARAMETERS])
            delta = np.array([2 * (y - x*x), 1 - x, 2 * (w - y*y), 1 - y, 1 - w])
            result["curve"]["concentration"] = (values * np.exp(delta)).tolist()
        return result

    monkeypatch.setattr(native_axes_runtime, "run_axes_trial", coupled_native)
    return case


@pytest.mark.parametrize("pause_after", [None, 18, 25, 27])
def test_full_native_budget_and_stage_boundary_replay(coupled_case, pause_after):
    case = coupled_case
    root, _, _, _, state = case
    paused = False
    prior_records = []

    def stop_once():
        nonlocal paused
        if len(state["trial_calls"]) == pause_after and not paused:
            paused = True
            raise cal.InfrastructurePause("synthetic stage boundary pause")

    if pause_after is not None:
        with pytest.raises(cal.InfrastructurePause, match="stage boundary pause"):
            contracts.fit(case, max_evaluations=48, before_trial=stop_once)
        assert not (root / "fit/mapping_locked.json").exists()
        assert len(state["trial_calls"]) == pause_after
        prior_records = sorted((cal._read(path) for path in (root / "fit/trials").glob("*/trial.json")),
                               key=lambda row: row["evaluation"])
        assert len(prior_records) == pause_after

    result = contracts.fit(case, max_evaluations=48)
    calls = deepcopy(state["trial_calls"])
    assert result["evaluation_count"] == len(calls) == 47
    assert state["base_calls"] == 1
    assert result["total_native_calls"] == result["actual_solver_calls"] == 48
    assert result["actual_new_calls"] == (48 if pause_after is None else 47 - pause_after)
    assert result["budget_exhausted"] and not result["converged"]
    assert [stage["evaluation_count"] for stage in result["optimizer"]["stages"]] == [18, 7, 22]
    assert [stage["stage"] for stage in result["optimizer"]["stages"]] == [
        "nested_two_axis", "fixed_screening", "three_axis"]
    assert result["optimizer"]["three_axis_start_evaluation"] <= 25
    assert result["decision"]["objective"] <= result["optimizer"]["nested_incumbent"]["objective"]
    assert prior_records == result["evaluations"][:len(prior_records)]
    assert len({tuple(row["parameters"][name].hex() for name in cal.axes_optimizer.PARAMETERS)
                for row in calls}) == len(calls)
    # Every stage scores the same five supplied observations, never a larger
    # curve hidden behind a warm start or the nested initialization.
    for row in result["evaluations"]:
        assert row["curve_metrics"]["n"] == 5
        assert len(row["predicted_at_observed_times"]) == 5
        assert len(row["residual_vector"]) == 8

    fresh = contracts.fit(case, max_evaluations=48, output_dir=root / "fresh")
    assert [row["parameters"] for row in result["evaluations"]] == [
        row["parameters"] for row in fresh["evaluations"]]
    assert [row["objective"] for row in result["evaluations"]] == [
        row["objective"] for row in fresh["evaluations"]]
    assert [row["parameters"] for row in calls] == [
        row["parameters"] for row in state["trial_calls"][47:]]

    selected = root / result["decision"]["native_output_dir"]
    cleaned = cal.cleanup_unselected(output_dir=root / "fit", project_root=root)
    assert cleaned["removed_count"] == 46 and selected.is_dir()
    assert len(list((root / "fit/trials").glob("*/trial.json"))) == 47
    count = len(state["trial_calls"])
    reused = contracts.fit(case, max_evaluations=48)
    assert reused["actual_new_calls"] == 0 and len(state["trial_calls"]) == count
    assert reused["decision"] == result["decision"]
