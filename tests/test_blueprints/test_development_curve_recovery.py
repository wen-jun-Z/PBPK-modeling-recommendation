"""Failure recovery must use real coordinates and retain the original limits."""
import json
import math

import numpy as np
import pytest

from pbpk_kb.blueprints.development_curve_optimizer import optimize_development_curve


TIMES = [0., .5, 1., 2., 4., 8.]


def model(lip, scale):
    return {"status": "predicted", "clearance_l_h": 10.*scale, "vss_l": 50.,
        "curve": {"time_min": TIMES, "concentration": [math.exp(lip-.2*scale*t) for t in TIMES],
                  "time_unit": "min", "unit": "umol/l"}}


def run(evaluate=model, **changes):
    options = dict(observations={"observations": [
        {"time_min": t, "concentration_umol_per_l": math.exp(.7-.2*1.2*t)} for t in TIMES]},
        initial_lipophilicity=0., initial_clearance_scale=1., logp_prior=.7,
        target_clearance_l_h=12., max_evaluations=18)
    options.update(changes)
    return optimize_development_curve(evaluate, **options)


def fail_on(call_numbers, *, otherwise=model):
    calls = []
    def evaluate(lip, scale):
        calls.append((lip, scale))
        if len(calls) in call_numbers:
            return {"status": "native_failed", "reason": "controlled infeasible trial"}
        return otherwise(lip, scale)
    return evaluate, calls


def assert_only_real_candidates(result, calls):
    assert result["evaluation_count"] == len(calls) <= result["max_evaluations"]
    assert len(calls) == len(set(calls))
    for r, coordinates in zip(result["evaluations"], calls):
        assert (r["lipophilicity"], r["clearance_scale"]) == coordinates
        if r["status"] != "ok":
            assert "residuals" not in r and "objective" not in r
            assert not r["eligible_for_selection"]
        else:
            assert sum(v*v for v in r["residuals"]) == pytest.approx(r["objective"])
    selected = next(r for r in result["evaluations"] if r["evaluation"] == result["decision"]["evaluation"])
    assert selected["eligible_for_selection"]
    assert selected["objective"] == min(r["objective"] for r in result["evaluations"] if r["eligible_for_selection"])


def test_default_still_stops_on_fourth_native_call():
    evaluate, calls = fail_on({4})
    result = run(evaluate)
    assert result["status"] == "evaluation_failed"
    assert result["evaluation_count"] == 4
    assert result["failure_recovery"]["enabled"] is False
    assert result["failure_recovery"]["events"] == []
    assert_only_real_candidates(result, calls)


def test_recoverable_failure_backtracks_and_restarts_at_a_real_improvement():
    evaluate, calls = fail_on({4})
    result = run(evaluate, recover_failed_trials=True)
    policy = result["failure_recovery"]
    assert result["evaluation_count"] > 4
    assert 1 <= policy["restart_count"] <= 2
    event = policy["events"][0]
    assert event["status"] == "restarted"
    chosen = next(r for r in result["evaluations"] if r["evaluation"] == event["restart_evaluation"])
    assert chosen["objective"] < event["anchor_objective"]
    assert event["restart_coordinates"] == chosen["coordinates"]
    assert result["decision"]["objective"] < event["anchor_objective"]
    start, failed = np.array(event["anchor_coordinates"]), np.array(event["failed_coordinates"])
    for attempt in event["backtracks"]:
        assert attempt["coordinates"] == pytest.approx(start + attempt["fraction"]*(failed-start))
    # Every successful residual is based on this record's actual simulation.
    for r in result["evaluations"]:
        if r["status"] == "ok":
            assert r["predicted_at_observed_times"] == pytest.approx(
                model(r["lipophilicity"], r["clearance_scale"])["curve"]["concentration"])
    assert_only_real_candidates(result, calls)
    json.dumps(result, allow_nan=False)


def test_failed_half_step_can_use_quarter_step_with_same_anchor():
    evaluate, calls = fail_on({4, 5})
    result = run(evaluate, recover_failed_trials=True)
    event = result["failure_recovery"]["events"][0]
    assert event["status"] == "restarted"
    assert [a["fraction"] for a in event["backtracks"]] == [.5, .25]
    assert event["backtracks"][0]["status"] == "evaluation_failed"
    anchor = np.array(event["anchor_coordinates"])
    assert event["restart_coordinates"] == pytest.approx(anchor + .25*(np.array(event["failed_coordinates"])-anchor))
    assert_only_real_candidates(result, calls)


def test_three_failed_backtracks_stop_without_fake_residuals_or_extra_calls():
    evaluate, calls = fail_on(set(range(4, 100)))
    result = run(evaluate, recover_failed_trials=True)
    assert result["status"] == "evaluation_failed"
    assert result["evaluation_count"] == 7
    assert result["failed_evaluation_count"] == 4
    event = result["failure_recovery"]["events"][0]
    assert [a["fraction"] for a in event["backtracks"]] == [.5, .25, .125]
    assert event["status"] == "not_recovered"
    assert result["failure_recovery"]["restart_count"] == 0
    assert_only_real_candidates(result, calls)


def test_feasible_but_worse_backtracks_do_not_restart():
    calls = []
    def evaluate(lip, scale):
        calls.append((lip, scale))
        if len(calls) == 4:
            return {"status": "native_failed"}
        value = model(lip, scale)
        if len(calls) > 4:
            value["curve"]["concentration"] = [1000.]*len(TIMES)
        return value
    result = run(evaluate, recover_failed_trials=True)
    assert result["status"] == "evaluation_failed"
    assert result["evaluation_count"] == 7
    assert result["failure_recovery"]["restart_count"] == 0
    assert all(a["status"] == "ok" and not a["strictly_improves_anchor"]
               for a in result["failure_recovery"]["events"][0]["backtracks"])
    assert_only_real_candidates(result, calls)


def test_initial_failure_is_still_terminal_and_never_retried():
    evaluate, calls = fail_on({1})
    result = run(evaluate, recover_failed_trials=True)
    assert result["status"] == "initial_evaluation_failed"
    assert result["evaluation_count"] == len(calls) == 1
    assert result["decision"] is None
    assert result["failure_recovery"]["events"] == []


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 6])
def test_recovery_never_exceeds_original_budget_even_if_every_probe_fails(budget):
    evaluate, calls = fail_on(set(range(4, 100)))
    result = run(evaluate, recover_failed_trials=True, max_evaluations=budget)
    assert result["evaluation_count"] == len(calls) == budget
    assert result["status"] == "budget_exhausted"
    assert result["failure_recovery"]["restart_count"] == 0
    assert_only_real_candidates(result, calls)


def test_frozen_axis_and_excluded_sensitivity_point_remain_frozen_during_recovery():
    def cl_only(_lip, scale):
        return model(0., scale)
    evaluate, calls = fail_on({4}, otherwise=cl_only)
    result = run(evaluate, recover_failed_trials=True)
    assert result["sensitivity"]["active_parameters"] == ["log_clearance_scale"]
    assert result["failure_recovery"]["restart_count"] >= 1
    assert not result["evaluations"][1]["eligible_for_selection"]
    for r in result["evaluations"]:
        if r["purpose"] in {"optimization", "failure_backtrack"}:
            assert r["lipophilicity"] == 0.
    for event in result["failure_recovery"]["events"]:
        assert event["anchor_coordinates"][0] == 0.
        assert event["anchor_evaluation"] != 2
    assert result["decision"]["lipophilicity"] == 0.
    assert_only_real_candidates(result, calls)


def test_failed_initial_sensitivity_keeps_original_freezing_policy():
    evaluate, calls = fail_on({2})
    result = run(evaluate, recover_failed_trials=True)
    assert result["sensitivity"]["active_parameters"] == ["log_clearance_scale"]
    assert result["failure_recovery"]["events"] == []
    assert result["decision"]["lipophilicity"] == 0.
    assert_only_real_candidates(result, calls)


@pytest.mark.parametrize("repeated_failed_point", [False, True])
def test_two_restart_limit_stops_third_failure_before_any_more_backtracks(monkeypatch, repeated_failed_point):
    import scipy.optimize
    starts = []
    failures = [.8, .8, .8] if repeated_failed_point else [.8, 1.2, 1.6]
    calls = []
    def lip_only(lip, scale):
        calls.append((lip, scale))
        if any(abs(lip-f) < 1e-12 for f in failures):
            return {"status": "native_failed"}
        return model(lip, 1.)
    def directed_search(fun, x0, **_kwargs):
        starts.append(list(x0))
        fun(x0)  # real cached residual at the actual restart coordinate
        fun(np.array([failures[len(starts)-1]]))
        pytest.fail("The failed point must abort this solver invocation")
    monkeypatch.setattr(scipy.optimize, "least_squares", directed_search)
    result = run(lip_only, recover_failed_trials=True, logp_prior=0., target_clearance_l_h=10.,
        observations={"observations":[{"time_min":t,"concentration_umol_per_l":math.exp(1.4-.2*t)} for t in TIMES]})
    recovery = result["failure_recovery"]
    assert recovery["restart_count"] == 2
    assert len(starts) == 3
    assert recovery["events"][-1]["status"] == "restart_limit_reached"
    assert recovery["events"][-1]["backtracks"] == []
    for start, event in zip(starts[1:], recovery["events"]):
        assert start == [event["restart_coordinates"][0]]
        assert event["backtracks"][-1]["objective"] < event["anchor_objective"]
    assert result["evaluation_count"] == (6 if repeated_failed_point else 8)
    assert_only_real_candidates(result, calls)


def test_failed_jacobian_probe_aborts_and_restarts_at_its_own_feasible_coordinate(monkeypatch):
    import scipy.optimize
    from types import SimpleNamespace
    starts, calls = [], []
    def lip_only(lip, scale):
        calls.append((lip, scale))
        return {"status":"native_failed"} if abs(lip-.33)<1e-12 else model(lip, 1.)
    def directed_search(fun, x0, *, jac, **_kwargs):
        starts.append(list(x0))
        if len(starts) == 1:
            jac(np.array([.3]))
            pytest.fail("Failed derivative probe must abort the invocation")
        fun(x0)
        return SimpleNamespace(success=True, status=1, message="Controlled convergence after recovery")
    monkeypatch.setattr(scipy.optimize, "least_squares", directed_search)
    result = run(lip_only, recover_failed_trials=True)
    event = result["failure_recovery"]["events"][0]
    assert event["anchor_coordinates"] == pytest.approx([.3, 0.])
    assert event["failed_coordinates"] == pytest.approx([.33, 0.])
    assert starts[1] == pytest.approx([.315])
    assert result["failure_recovery"]["restart_count"] == 1
    assert_only_real_candidates(result, calls)


def test_backtracking_from_upper_bounds_stays_inside_original_search_box():
    evaluate, calls = fail_on({4})
    result = run(evaluate, recover_failed_trials=True,
                 initial_lipophilicity=3.2, initial_clearance_scale=4.)
    assert result["failure_recovery"]["events"]
    assert any(r["purpose"] == "failure_backtrack" for r in result["evaluations"])
    assert all(-1.8 <= lip <= 3.2 and .25 <= scale <= 4. for lip, scale in calls)
    assert_only_real_candidates(result, calls)


def test_no_failure_enabled_is_numerically_identical_to_disabled():
    disabled = run()
    enabled = run(recover_failed_trials=True)
    disabled.pop("failure_recovery")
    assert enabled.pop("failure_recovery")["events"] == []
    assert disabled == enabled


@pytest.mark.parametrize("invalid", [0, 1, None, "yes", []])
def test_non_boolean_recovery_switch_rejected_before_native(invalid):
    def never(*_):
        pytest.fail("Invalid recovery policy must not invoke native")
    with pytest.raises(ValueError, match="boolean"):
        run(never, recover_failed_trials=invalid)
