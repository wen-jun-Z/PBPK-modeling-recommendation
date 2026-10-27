"""No clinical data: isolation, frozen folds, selection and honest failures."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import development_prior_selection as policy


def request():
    return {"compound_id": "mock", "human": {"weight_kg": 70},
            "dosing": {"dose_mg": 100, "infusion_minutes": 30},
            "evidence": [{"parameter": "molecular_weight", "value": 500.}]}


def training(n=5):
    return {"compound_id": "mock", "human": {"weight_kg": 70},
            "dosing": {"dose_mg": 100, "infusion_minutes": 30},
            "matrix": "plasma", "time_unit": "min", "unit": "umol/l",
            "role": "calibration_data_reconstruction",
            "observations": [{"time_min": (i + 1) * 60., "concentration_umol_per_l": 2.}
                             for i in range(n)]}


def physiology():
    return {formula: {"marker": formula} for formula in policy.FORMULAS}


def fitter(error=0., *, hook=None):
    def fit(phys, req, data, budget, starts, guard):
        guard()
        if hook:
            hook(phys, req, data, budget, starts)
        return {"best": {"objective": 0., "log_parameters": [error, 0., 0.]},
                "calls": 2, "failures": 0}
    return fit


def simulate(phys, req, times, parameters):
    return {"concentration_mg_l": np.full(len(times), math.exp(parameters[0]))}


def run(fitters=None, data=None, sim=simulate, **kwargs):
    return policy.select(physiology(), request(), data or training(),
                         fitters or {name: fitter() for name in policy.METHODS}, sim, **kwargs)


def test_time_rank_folds_and_each_visible_point_held_out_exactly_once():
    data = training(7)
    data["observations"] = [data["observations"][i] for i in [6, 0, 4, 2, 1, 5, 3]]
    folds = policy.time_folds(data)
    assert folds[0]["heldout_indices"] == [1, 5]
    assert folds[1]["heldout_indices"] == [4, 0]
    assert sorted(i for fold in folds for i in fold["heldout_indices"]) == list(range(7))
    for fold in folds:
        assert set(fold["training_indices"]).isdisjoint(fold["heldout_indices"])
        assert len(fold["training_indices"]) >= 4
    changed = deepcopy(data)
    for row in changed["observations"]:
        row["concentration_umol_per_l"] *= row["time_min"]
    assert policy.time_folds(changed) == folds


def test_five_is_leave_one_and_smaller_training_or_duplicate_times_rejected():
    assert [len(f["training_indices"]) for f in policy.time_folds(training())] == [4] * 5
    with pytest.raises(ValueError):
        policy.time_folds(training(4), nfolds=4)
    duplicate = training()
    duplicate["observations"][1]["time_min"] = duplicate["observations"][0]["time_min"]
    with pytest.raises(ValueError, match="distinct"):
        policy.time_folds(duplicate)


def test_fitters_receive_only_visible_fold_and_no_hidden_sources_or_warm_start():
    data = training()
    data.update(full_source={"secret": "full curve"}, target_key={"secret": "target"},
                full_curve=training(20), fitted_parameters=[1, 2, 3])
    for row in data["observations"]:
        row["target_key"] = {"secret": "held targets"}
    original = deepcopy(data)
    calls = []

    def inspect(phys, req, subset, budget, starts):
        calls.append(deepcopy(subset))
        assert len(subset["observations"]) == 4
        assert set(subset) == set(policy._METADATA) | {"observations", "visible_observation_indices"}
        assert all(set(row) == {"time_min", "concentration_umol_per_l"} for row in subset["observations"])
        assert [row["time_min"] for row in subset["observations"]] == [
            original["observations"][i]["time_min"] for i in subset["visible_observation_indices"]]
        assert budget == 4096 and starts == 16
        # Mutation in an injected fitter must not affect any other fold/candidate.
        subset["observations"].clear()
        req["evidence"].clear()
        phys.clear()

    result = run({name: fitter(hook=inspect) for name in policy.METHODS}, data)
    assert data == original and len(calls) == 20
    assert result["selected_method"] == "fixed_prior"
    assert result["final_parameters"] is None and result["requires_final_refit"]
    assert result["target_observations_used"] is False


def test_conversion_and_costs_include_validation_simulations():
    result = run()
    assert result["methods"]["fixed_prior"]["score"] == pytest.approx(0.)
    assert result["calls"] == 50
    assert result["fitting_calls"] == 40 and result["validation_calls"] == 10
    assert result["failures"] == 0


def test_held_points_equal_weight_not_equal_weighted_fold_means():
    def pred(phys, req, times, parameters):
        errors = np.where(times == 1., 1., 0.) if parameters[0] == 0 else np.full(len(times), .13)
        return {"concentration_mg_l": np.exp(errors)}
    result = run({"fixed_prior": fitter(0), "direct_relaxed": fitter(1)}, training(7), pred)
    assert result["methods"]["fixed_prior"]["score"] == pytest.approx(1 / 7)
    assert result["methods"]["direct_relaxed"]["score"] == pytest.approx(.13)
    assert result["selected_method"] == "direct_relaxed"


def test_formula_selected_by_training_objective_before_heldout_prediction():
    seen = []

    def fit(phys, req, data, budget, starts, guard):
        standard = req["partition_method"] == "PKSimStandard"
        return {"best": {"objective": 0. if standard else 1.,
                         "log_parameters": [2. if standard else 0., 0., 0.]},
                "calls": 1, "failures": 0}

    def pred(phys, req, times, parameters):
        seen.append(req["partition_method"])
        return simulate(phys, req, times, parameters)

    result = run({name: fit for name in policy.METHODS}, sim=pred)
    assert seen == ["PKSimStandard"] * 10
    assert result["methods"]["fixed_prior"]["score"] == pytest.approx(2.)


@pytest.mark.parametrize("difference,choice", [(0., "fixed_prior"), (5e-13, "fixed_prior"), (2e-12, "direct_relaxed")])
def test_ties_prefer_fixed_with_frozen_tolerance(difference, choice):
    result = run({"fixed_prior": fitter(.1 + difference), "direct_relaxed": fitter(.1)})
    assert result["selected_method"] == choice


def test_one_failed_fold_disqualifies_candidate_without_penalty_average():
    def sometimes(phys, req, data, budget, starts, guard):
        if 60. not in [row["time_min"] for row in data["observations"]]:
            return {"best": None, "calls": 3, "failures": 3}
        return fitter()(phys, req, data, budget, starts, guard)
    result = run({"fixed_prior": sometimes, "direct_relaxed": fitter(1.)})
    assert result["selected_method"] == "direct_relaxed"
    fixed = result["methods"]["fixed_prior"]
    assert fixed["eligible"] is False and fixed["score"] is None
    assert fixed["scored_points"] == 4
    assert result["fitting_failures"] == 6


def test_both_failed_candidates_are_blocked():
    def failure(*args):
        return {"best": None, "calls": 1, "failures": 1}
    result = run({name: failure for name in policy.METHODS})
    assert result["status"] == "blocked" and result["selected_method"] is None
    assert result["calls"] == result["failures"] == 20
    assert result["validation_calls"] == 0


def test_invalid_heldout_prediction_is_failure_without_trying_other_formula():
    observed = []

    def invalid(phys, req, times, parameters):
        observed.append(req["partition_method"])
        return {"concentration_mg_l": np.zeros(len(times))}

    result = run(sim=invalid)
    assert result["status"] == "blocked"
    assert len(observed) == 10 and set(observed) == {"PKSimStandard"}
    assert result["validation_failures"] == 10
    assert all(row["score"] is None for row in result["methods"].values())


def test_integrity_exceptions_propagate_and_budget_overruns_raise():
    def broken_guard():
        raise ValueError("source integrity changed")
    with pytest.raises(ValueError, match="source integrity"):
        run(before_trial=broken_guard)

    def over_budget(*args):
        return {"best": None, "calls": 4097, "failures": 0}
    with pytest.raises(ValueError, match="within budget"):
        run({name: over_budget for name in policy.METHODS})
