"""Synthetic contracts for the two new four-axis distribution calibrations."""
from copy import deepcopy
import math
import types

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import distribution_calibration_optimizer as optimizer


def inputs(n=20):
    context = {"species": "human", "matrix": "plasma", "route": "intravenous"}
    request = {"compound_id": "synthetic", "human": {"weight_kg": 50.},
        "dosing": {"route": "intravenous", "dose_mg": 100., "infusion_minutes": 30., "start_h": 0.},
        "evidence": [{"parameter": "molecular_weight", "value": 500., "unit": "g/mol"},
            {"parameter": "total_clearance", "value": 4., "unit": "ml/min/kg", "context": context},
            {"parameter": "renal_clearance", "value": 1., "unit": "ml/min/kg", "context": context}]}
    times = np.linspace(.1, 5., n)
    training = {"compound_id": "synthetic", "human": deepcopy(request["human"]), "dosing": deepcopy(request["dosing"]),
        "matrix": "plasma", "time_unit": "min", "unit": "umol/l",
        "observations": [{"time_min": float(t * 60), "concentration_umol_per_l": 2.} for t in times]}
    physiology = {"circulation": {"cardiac_flow_l_h": 100.}}
    return request, training, physiology


def once(fun, x0, **kwargs):
    fun(x0)
    return types.SimpleNamespace(success=True)


@pytest.mark.parametrize("variant", optimizer.VARIANTS)
def test_known_parameters_recovered_with_the_declared_regularized_objective(variant):
    request, training, physiology = inputs()
    truth = np.array([.3, .5, -.4, .2])
    def basis(t):
        return np.column_stack([np.ones(len(t)), 2 * np.sin(t), 2 * np.cos(2*t), 2 * np.sin(3*t)])
    times = np.array([r["time_min"] / 60 for r in training["observations"]])
    design = basis(times)
    observed = np.exp(design @ truth)
    for row, value in zip(training["observations"], observed):
        row["concentration_umol_per_l"] = float(value * 1000 / 500)
    calls = []
    def predict(x, times_h):
        calls.append(1)
        return {"concentration_mg_l": np.exp(basis(times_h) @ x)}
    result = optimizer.fit(predict, request, training, physiology, variant=variant, starts=2, budget=2000)
    weights = np.diag([0., .01/math.log(10)**2, .01/math.log(100)**2,
                       .01/math.log(10)**2 if variant == "blood_partition" else .01])
    expected = np.linalg.solve(design.T @ design / len(times) + weights, design.T @ (design @ truth) / len(times))
    assert result["status"] == "fitted" and result["calls"] == len(calls)
    np.testing.assert_allclose(result["best"]["log_parameters"], expected, atol=2e-6)
    np.testing.assert_allclose(result["best"]["log_parameters"], truth, atol=.004)
    assert result["best"]["development_gmfe"] < 1.005


@pytest.mark.parametrize("variant", optimizer.VARIANTS)
def test_all_16_initializations_match_frozen_three_axes_and_independent_fourth(variant, monkeypatch):
    monkeypatch.setattr(optimizer, "least_squares", once)
    request, training, physiology = inputs()
    calls = []
    def predict(x, times_h):
        calls.append(x.copy())
        return {"concentration_mg_l": np.ones(len(times_h))}
    result = optimizer.fit(predict, request, training, physiology, variant=variant)
    lower = np.log([1e-4/9, .1, .01])
    upper = np.log([1000/9, 10., 100.])
    cl = np.r_[0., np.random.default_rng(20260930).uniform(lower[0], upper[0], 15)]
    shape = np.vstack([np.zeros(2), np.random.default_rng(20260930).uniform(lower[1:], upper[1:], size=(15, 2))])
    fourth_bounds = np.log([.05, 20.]) if variant == "blood_partition" else np.array([-1., 1.])
    fourth = np.r_[0., np.random.default_rng(20261008).uniform(*fourth_bounds, size=15)]
    np.testing.assert_array_equal(np.asarray(calls), np.column_stack([cl, shape, fourth]))
    assert len(result["starts"]) == result["calls"] == 16
    assert all(s["calls"] == 1 and s["call_limit"] == 1024 for s in result["starts"])
    assert result["clearance_input"]["renal_clearance_l_h"] == 3.


def test_actual_call_budget_includes_numerical_failures_and_difference_trials(monkeypatch):
    def hungry(fun, x0, **kwargs):
        for _ in range(10000):
            fun(x0)
        pytest.fail("Hard call quota failed")
    monkeypatch.setattr(optimizer, "least_squares", hungry)
    request, training, physiology = inputs()
    calls = []
    def predict(x, times_h):
        calls.append(1)
        if len(calls) % 3 == 0:
            raise ValueError("Synthetic invalid model")
        return {"concentration_mg_l": np.ones(len(times_h))}
    result = optimizer.fit(predict, request, training, physiology, variant="blood_partition", budget=17, starts=3)
    assert result["calls"] == len(calls) == 17 and result["failures"] == 5
    assert [s["calls"] for s in result["starts"]] == [6, 6, 5]
    assert all(s["status"] == "call_budget_exhausted" for s in result["starts"])


@pytest.mark.parametrize("variant", optimizer.VARIANTS)
def test_objective_has_no_cl_prior_and_only_the_declared_shape_penalties(variant, monkeypatch):
    request, training, physiology = inputs()
    x = np.array([1., .2, -.3, .4])
    captured = []
    def probe(fun, x0, **kwargs):
        captured.append(fun(x))
        return types.SimpleNamespace(success=True)
    monkeypatch.setattr(optimizer, "least_squares", probe)
    result = optimizer.fit(lambda x, t: {"concentration_mg_l": np.ones(len(t))}, request, training, physiology,
                           variant=variant, starts=1, budget=2)
    assert len(captured[0]) == len(training["observations"]) + 3
    np.testing.assert_array_equal(captured[0][:-3], 0.)
    fourth = .1*x[3]/math.log(10) if variant == "blood_partition" else .1*x[3]
    np.testing.assert_allclose(captured[0][-3:], [.1*x[1]/math.log(10), .1*x[2]/math.log(100), fourth])
    assert result["best"]["objective"] == pytest.approx(float(captured[0] @ captured[0]))


def test_mw_unit_conversion_visible_times_and_input_mutation_isolation(monkeypatch):
    monkeypatch.setattr(optimizer, "least_squares", once)
    request, training, physiology = inputs(5)
    original = deepcopy((request, training, physiology))
    training["hidden_full_curve"] = {"sentinel": "must never be consumed"}
    original[1]["hidden_full_curve"] = deepcopy(training["hidden_full_curve"])
    times = np.array([r["time_min"] / 60 for r in training["observations"]])
    def predict(x, supplied_times):
        np.testing.assert_array_equal(supplied_times, times)
        # Mutation by an injected predictor cannot overwrite optimizer arrays.
        supplied_times[:] = 999.
        x[:] = 999.
        return {"concentration_mg_l": np.ones(len(times))}
    result = optimizer.fit(predict, request, training, physiology, variant="exchange_shape", starts=1, budget=2)
    assert result["best"]["development_gmfe"] == 1.
    assert result["best"]["log_parameters"] == [0., 0., 0., 0.]
    assert (request, training, physiology) == original


@pytest.mark.parametrize("bad", [None, {}, {"concentration_mg_l": [0.]}, {"concentration_mg_l": [float("nan")]}, {"concentration_mg_l": [-1.]*20}])
def test_invalid_prediction_never_becomes_a_successful_fit(bad, monkeypatch):
    monkeypatch.setattr(optimizer, "least_squares", once)
    request, training, physiology = inputs()
    result = optimizer.fit(lambda *_: bad, request, training, physiology, variant="blood_partition", starts=2, budget=8)
    assert result["best"] is None and result["status"] == "failed"
    assert result["calls"] == result["failures"] == 2
    assert all(s["best"] is None and not s["eligible"] and s["status"] == "no_feasible_evaluation" for s in result["starts"])


def test_guard_valueerror_propagates_before_counting_model_calls(monkeypatch):
    monkeypatch.setattr(optimizer, "least_squares", once)
    request, training, physiology = inputs()
    calls = []
    def guard():
        raise ValueError("Source hash changed")
    with pytest.raises(ValueError, match="Source hash changed"):
        optimizer.fit(lambda *_: calls.append(1), request, training, physiology,
                      variant="blood_partition", before_trial=guard)
    assert calls == []


@pytest.mark.parametrize("argument,value", [("budget", True), ("budget", 3.5), ("budget", 0), ("budget", float("inf")),
    ("budget", 15), ("starts", True), ("starts", 0), ("starts", 17), ("starts", 3.)])
def test_strict_budget_and_start_validation(argument, value):
    request, training, physiology = inputs()
    with pytest.raises(ValueError):
        optimizer.fit(lambda *_: pytest.fail("Invalid configuration evaluated"), request, training, physiology,
                      variant="blood_partition", **{argument: value})


@pytest.mark.parametrize("field", ["human", "dosing", "route", "time_unit", "row_unit", "matrix", "compound", "clearance_unit", "duplicate_time"])
def test_rejects_wrong_development_conditions_and_units(field):
    request, training, physiology = inputs()
    if field == "human": training["human"]["weight_kg"] = 80.
    elif field == "dosing": training["dosing"]["dose_mg"] = 1.
    elif field == "route":
        request["dosing"]["route"] = training["dosing"]["route"] = "oral"
    elif field == "time_unit": training["time_unit"] = "h"
    elif field == "row_unit": training["observations"][0]["unit"] = "mg/L"
    elif field == "matrix": training["matrix"] = "whole_blood"
    elif field == "compound": training["compound_id"] = "wrong"
    elif field == "clearance_unit": request["evidence"][1]["unit"] = "L/h"
    elif field == "duplicate_time": training["observations"][1]["time_min"] = training["observations"][0]["time_min"]
    with pytest.raises(ValueError):
        optimizer.fit(lambda *_: pytest.fail("Invalid input evaluated"), request, training, physiology, variant="blood_partition")
