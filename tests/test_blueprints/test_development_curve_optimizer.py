"""Numerical behavior, scientific data boundaries and finite run budgets."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprints.development_curve_optimizer import optimize_development_curve


TIMES = [0., .5, 1., 2., 4., 8.]


def observations(values, times=TIMES):
    return {"observations": [{"time_min": t, "concentration_umol_per_l": float(v)}
                             for t, v in zip(times, values)]}


def output(values, *, cl=10., vss=50., times=TIMES):
    return {"status": "predicted", "clearance_l_h": cl, "vss_l": vss,
            "curve": {"time_min": list(times), "concentration": [float(v) for v in values],
                      "time_unit": "min", "unit": "umol/l"}}


def model(lip, scale):
    return output([math.exp(lip - .2 * scale * t) for t in TIMES], cl=10. * scale)


def run(evaluate=model, **overrides):
    options = {"observations": observations([math.exp(.7 - .2 * 1.2 * t) for t in TIMES]),
        "initial_lipophilicity": 0., "initial_clearance_scale": 1.,
        "logp_prior": .7, "target_clearance_l_h": 12., "max_evaluations": 18}
    options.update(overrides)
    return optimize_development_curve(evaluate, **options)


def test_two_identifiable_parameters_recover_known_curve_and_pk_target():
    result = run()
    chosen = result["decision"]
    assert chosen is not None
    assert chosen["objective"] < result["initial_objective"] * .001
    assert chosen["lipophilicity"] == pytest.approx(.7, abs=.015)
    assert chosen["clearance_scale"] == pytest.approx(1.2, abs=.015)
    assert chosen["curve_metrics"]["gmfe"] < 1.01
    assert result["sensitivity"]["rank"] == 2
    assert result["sensitivity"]["condition_number"] >= 1
    assert result["evaluation_count"] <= 18
    assert result["objective_spec"]["derived_endpoint_residuals"] is False


def test_exact_prespecified_loss_and_residual_normalization():
    result = run(lambda *_: output([2.] * len(TIMES), cl=10., vss=40.),
        observations=observations([1.] * len(TIMES)), initial_lipophilicity=.4,
        logp_prior=0., target_clearance_l_h=20., target_vss_l=20., max_evaluations=1)
    expected = math.log(2.) ** 2 + .1 + .1 + .01 * (.4 / 2.5) ** 2
    assert result["decision"]["objective"] == pytest.approx(expected)
    record = result["evaluations"][0]
    assert sum(r*r for r in record["residuals"]) == pytest.approx(expected)
    assert record["curve_metrics"]["gmfe"] == pytest.approx(2.)
    assert record["clearance_relative_error"] == pytest.approx(.5)
    assert record["vss_relative_error"] == pytest.approx(1.)
    assert result["converged"] is False
    assert result["status"] == "budget_exhausted"


def test_more_samples_do_not_artificially_increase_curve_loss_weight():
    losses = []
    for times in ([0., 8.], TIMES):
        result = run(lambda *_: output([2.] * len(TIMES)),
            observations=observations([1.] * len(times), times), logp_prior=0.,
            target_clearance_l_h=10., max_evaluations=1)
        losses.append(result["decision"]["objective"])
    assert losses[0] == pytest.approx(losses[1])


def test_real_calls_include_sensitivity_and_jacobian_and_cache_duplicates():
    calls = []
    def counted(lip, scale):
        calls.append((lip, scale))
        return model(lip, scale)
    result = run(counted, max_evaluations=7)
    assert result["status"] == "budget_exhausted"
    assert result["evaluation_count"] == len(calls) == 7
    assert len(set(calls)) == len(calls)
    assert result["cache_hits"] > 0
    feasible = [r for r in result["evaluations"] if r["eligible_for_selection"]]
    assert result["decision"]["objective"] == min(r["objective"] for r in feasible)


def test_failed_search_trial_is_retained_and_returns_only_real_feasible_best():
    calls = []
    def fail_search(lip, scale):
        calls.append((lip, scale))
        if len(calls) == 4:
            return {"status": "parameter_readback_failed", "reason": "clearance capacity"}
        return model(lip, scale)
    result = run(fail_search)
    assert result["status"] == "evaluation_failed"
    assert result["failed_evaluation_count"] == 1
    assert result["evaluation_count"] == 4
    assert result["converged"] is False
    assert result["decision"]["evaluation"] in [1, 2, 3]
    assert not result["evaluations"][-1]["eligible_for_selection"]
    assert "objective" not in result["evaluations"][-1]
    assert "clearance capacity" in result["reason"]


def test_exception_is_recorded_and_initial_failure_is_not_accepted():
    def fail(*_):
        raise RuntimeError("model failed")
    result = run(fail)
    assert result["status"] == "initial_evaluation_failed"
    assert result["decision"] is None
    assert result["evaluation_count"] == 1
    assert "RuntimeError: model failed" in result["reason"]


def test_zero_curve_information_freezes_both_parameters_and_does_not_optimize_priors():
    result = run(lambda *_: output([1.] * len(TIMES)),
        observations=observations([.8] * len(TIMES)))
    assert result["status"] == "no_identifiable_parameters"
    assert result["sensitivity"]["rank"] == 0
    assert len(result["sensitivity"]["frozen_parameters"]) == 2
    assert result["decision"]["evaluation"] == 1
    assert result["evaluation_count"] == 3
    assert result["converged"] is False
    assert not result["evaluations"][1]["eligible_for_selection"]


def test_only_informative_curve_parameter_is_active():
    def cl_only(_lip, scale):
        return output([math.exp(-.2 * scale * t) for t in TIMES], cl=10. * scale)
    result = run(cl_only, observations=observations([math.exp(-.2 * 1.2 * t) for t in TIMES]))
    assert result["sensitivity"]["rank"] == 1
    assert result["sensitivity"]["active_parameters"] == ["log_clearance_scale"]
    assert result["sensitivity"]["frozen_parameters"][0]["parameter"] == "effective_lipophilicity"
    assert result["decision"]["lipophilicity"] == 0.
    assert result["decision"]["clearance_scale"] == pytest.approx(1.2, abs=.015)
    assert all(r["coordinates"][0] == 0. for r in result["evaluations"] if r["purpose"] == "optimization")


def test_collinear_curve_effects_freeze_one_parameter_with_reason():
    def collinear(lip, scale):
        return output([math.exp(lip) * scale] * len(TIMES), cl=10. * scale)
    result = run(collinear, observations=observations([1.8] * len(TIMES)),
                 logp_prior=0., target_clearance_l_h=10.)
    assert result["sensitivity"]["rank"] == 1
    assert len(result["sensitivity"]["active_parameters"]) == 1
    frozen = result["sensitivity"]["frozen_parameters"]
    assert len(frozen) == 1 and "collinear" in frozen[0]["reason"]
    for r in result["evaluations"]:
        if r["purpose"] == "initial_sensitivity" and not r["eligible_for_selection"]:
            assert "frozen parameter" in r["selection_exclusion"]


def test_failed_sensitivity_trial_freezes_that_parameter_but_other_can_be_used():
    calls = 0
    def bad_lip_trial(lip, scale):
        nonlocal calls
        calls += 1
        return {"status": "native_failed"} if calls == 2 else model(lip, scale)
    result = run(bad_lip_trial)
    assert result["decision"] is not None
    assert result["failed_evaluation_count"] == 1
    assert result["sensitivity"]["active_parameters"] == ["log_clearance_scale"]
    assert result["decision"]["lipophilicity"] == 0.
    assert "not a feasible" in result["sensitivity"]["frozen_parameters"][0]["reason"]


def test_unit_correct_linear_interpolation_not_observation_substitution():
    curve_result = output([1., 5.], times=[0., 2.])
    result = run(lambda *_: curve_result, observations=observations([4., 4.], [1., 2.]), max_evaluations=1)
    assert result["evaluations"][0]["predicted_at_observed_times"] == [3., 5.]
    assert result["decision"]["curve_metrics"]["gmfe"] > 1.


@pytest.mark.parametrize("mutate,reason", [
    (lambda r: r["curve"].update(time_min=[1., 2., 3., 4., 5., 6.]), "extrapolation"),
    (lambda r: r["curve"].update(time_unit="h"), "time_unit"),
    (lambda r: r["curve"].update(unit="mg/l"), "unit"),
    (lambda r: r["curve"].update(concentration=[0.] * 6), "positive"),
    (lambda r: r["curve"].update(concentration=[-1.] * 6), "nonnegative"),
    (lambda r: r["curve"].update(time_min=[0., 1., 1., 2., 4., 8.]), "increase"),
    (lambda r: r.update(clearance_l_h=float("nan")), "finite"),
    (lambda r: r.update(clearance_l_h=True), "finite"),
    (lambda r: r["curve"].update(concentration_umol_l=r["curve"].pop("concentration")), "aligned"),
])
def test_invalid_simulations_never_become_candidates(mutate, reason):
    value = model(0., 1.)
    mutate(value)
    result = run(lambda *_: value)
    assert result["decision"] is None
    assert reason in result["reason"]
    assert result["evaluation_count"] == 1


def test_requested_vss_prior_requires_real_vss_but_absent_prior_does_not():
    missing_vss = model(0., 1.)
    missing_vss.pop("vss_l")
    assert run(lambda *_: missing_vss, target_vss_l=50.)["decision"] is None
    assert run(lambda *_: missing_vss, max_evaluations=1)["decision"]["vss_l"] is None


@pytest.mark.parametrize("override", [
    {"max_evaluations": 0}, {"max_evaluations": True}, {"max_evaluations": 1.5},
    {"initial_lipophilicity": 4.}, {"initial_clearance_scale": 0.},
    {"initial_clearance_scale": 4.1}, {"logp_prior": float("inf")},
    {"target_clearance_l_h": 0.}, {"target_vss_l": -1.},
    {"observations": observations([0., 1.], [0., 1.])},
    {"observations": observations([1., 2.], [1., 1.])},
    {"observations": observations([1., float("nan")], [0., 1.])},
    {"observations": {"observations": [{"time_min": 0., "concentration_umol_per_l": 1.}]}},
])
def test_bad_input_is_rejected_before_callback(override):
    calls = []
    def never(*args):
        calls.append(args)
        return model(*args)
    with pytest.raises(ValueError):
        run(never, **override)
    assert not calls


def test_search_bounds_center_on_evidence_prior_and_upper_bound_steps_inward():
    calls = []
    def counted(lip, scale):
        calls.append((lip, scale))
        return model(lip, scale)
    result = run(counted, initial_lipophilicity=2.5, initial_clearance_scale=4., logp_prior=0.)
    assert calls[1][0] < 2.5
    assert calls[2][1] < 4.
    assert all(-2.5 <= lip <= 2.5 and .25 <= scale <= 4. for lip, scale in calls)
    assert result["evaluation_count"] <= 18


def test_input_observations_and_callback_curve_are_immutable():
    obs = observations([1.] * len(TIMES))
    native = output([2.] * len(TIMES))
    obs_before, native_before = deepcopy(obs), deepcopy(native)
    run(lambda *_: native, observations=obs)
    assert obs == obs_before
    assert native == native_before


def test_output_can_be_json_serialized_without_nonfinite_values():
    import json
    result = run(lambda lip, scale: output([math.exp(lip) * scale] * len(TIMES)),
                 observations=observations([2.] * len(TIMES)))
    json.dumps(result, allow_nan=False)


def test_small_budgets_never_launch_hidden_extra_calls():
    for budget in (1, 2, 3):
        calls = []
        def counted(lip, scale):
            calls.append((lip, scale))
            return model(lip, scale)
        result = run(counted, max_evaluations=budget)
        assert len(calls) == result["evaluation_count"] == budget
        assert result["decision"] is not None
        assert result["status"] == "budget_exhausted"
        assert result["converged"] is False


def test_numpy_scalar_callback_endpoints_are_supported():
    native = model(0., 1.)
    native["clearance_l_h"] = np.float64(10.)
    result = run(lambda *_: native, max_evaluations=1)
    assert result["decision"]["clearance_l_h"] == 10.
