"""Select one existing CL calibration rule using visible development points.

This module reads no files and accepts no target observations. Cross-validation
is a development diagnostic, not independent clinical validation. Fold fits
must start independently; their parameters are not a final fitted mapping.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral

import numpy as np

VERSION = "development-time-fold-prior-selection-1"
METHODS = ("fixed_prior", "direct_relaxed")
FORMULAS = ("PKSimStandard", "RodgersRowland")
TIE_TOLERANCE = 1e-12
_METADATA = ("compound_id", "human", "dosing", "matrix", "time_unit", "unit", "role")
_OBSERVATION_FIELDS = ("time_min", "concentration_umol_per_l", "time_unit", "unit")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _visible_arrays(training):
    for key, expected in (("time_unit", "min"), ("unit", "umol/l"), ("matrix", "plasma")):
        if key in training and training[key] != expected:
            raise ValueError(f"Development {key} must be {expected}")
    rows = training.get("observations", [])
    times = np.asarray([row["time_min"] for row in rows], dtype=float)
    values = np.asarray([row["concentration_umol_per_l"] for row in rows], dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(times < 0) or len(np.unique(times)) != len(times)
            or values.shape != times.shape or not np.isfinite(values).all()
            or np.any(values <= 0)):
        raise ValueError("Visible development times must be distinct; concentrations finite and positive")
    for row in rows:
        if row.get("time_unit", "min") != "min" or row.get("unit", "umol/l") != "umol/l":
            raise ValueError("Observation units differ from development units")
    return times / 60., values


def time_folds(training, nfolds=5):
    """Assign sorted time rank modulo nfolds; never inspect concentrations to split.

    Returned indices address the original visible observation array. At least
    four training points are required for the existing three-parameter fits.
    """
    count = _positive_integer(nfolds, "nfolds")
    times, _ = _visible_arrays(training)
    if count < 2 or count > len(times):
        raise ValueError("Require two or more nonempty folds")
    order = np.argsort(times, kind="stable").tolist()
    folds = []
    for fold in range(count):
        held = [index for rank, index in enumerate(order) if rank % count == fold]
        train = [index for rank, index in enumerate(order) if rank % count != fold]
        if len(train) < 4:
            raise ValueError("Each fold requires at least four visible training points")
        folds.append({"fold": fold, "training_indices": train, "heldout_indices": held})
    return folds


def _fold_training(training, indices):
    # Explicit allowlists prevent alternate curves, source paths, cached peaks
    # and target labels from accompanying a fold into an injected fitter.
    result = {key: deepcopy(training[key]) for key in _METADATA if key in training}
    result["observations"] = [
        {key: deepcopy(training["observations"][index][key])
         for key in _OBSERVATION_FIELDS if key in training["observations"][index]}
        for index in indices]
    result["visible_observation_indices"] = list(indices)
    return result


def _best(fitted):
    row = fitted.get("best")
    if not isinstance(row, dict):
        return None
    try:
        value = float(row["objective"])
        parameters = np.asarray(row["log_parameters"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if (not math.isfinite(value) or value < 0 or value >= 1e10
            or parameters.shape != (3,) or not np.isfinite(parameters).all()):
        return None
    return deepcopy(row)


def _cost(fitted, budget):
    calls, failures = fitted.get("calls"), fitted.get("failures")
    if (isinstance(calls, bool) or not isinstance(calls, Integral)
            or not 0 <= calls <= budget or isinstance(failures, bool)
            or not isinstance(failures, Integral) or not 0 <= failures <= calls):
        raise ValueError("Fitter must report exact nonnegative calls and failures within budget")
    return int(calls), int(failures)


def select(physiology_by_formula, request, training, fitters, simulate,
           budget=4096, starts=16, before_trial=None):
    """Select fixed_prior/direct_relaxed by five development time folds.

    Fitter signature: (physiology, request, training, budget, starts, guard).
    Simulator signature: (physiology, request, times_h, log_parameters), returning
    a dictionary with concentration_mg_l. Each fold chooses its formula using
    the training objective, before evaluating held-out concentrations. All
    held-out points have equal weight regardless of unequal fold sizes.

    Source/guard/fitter exceptions propagate. Numerical validation failures
    make the candidate ineligible; a failed fold is never replaced by a penalty
    score or by a formula chosen after seeing the held-out error.
    """
    budget = _positive_integer(budget, "budget")
    starts = _positive_integer(starts, "starts")
    if starts > budget:
        raise ValueError("Budget must permit every declared start")
    if set(fitters) != set(METHODS) or not all(callable(fitters[m]) for m in METHODS):
        raise ValueError("Provide exactly fixed_prior and direct_relaxed fitters")
    if set(physiology_by_formula) != set(FORMULAS):
        raise ValueError("Provide exactly the two frozen partition formulas")
    if not callable(simulate) or (before_trial is not None and not callable(before_trial)):
        raise ValueError("Simulation and guard must be callable")
    guard = before_trial if before_trial is not None else lambda: None
    times, concentrations = _visible_arrays(training)
    weights = [float(row["value"]) for row in request.get("evidence", [])
               if row.get("parameter") == "molecular_weight"]
    if len(weights) != 1 or not math.isfinite(weights[0]) or weights[0] <= 0:
        raise ValueError("Require one positive finite molecular weight")
    observed = concentrations * weights[0] / 1000.
    folds = time_folds(training)
    methods = {}
    fit_calls = fit_failures = validation_calls = validation_failures = 0
    for method in METHODS:
        traces = []
        point_errors = []
        method_cost = {"fitting_calls": 0, "fitting_failures": 0,
                       "validation_calls": 0, "validation_failures": 0}
        for fold in folds:
            candidates = []
            for formula in FORMULAS:
                guard()
                req = deepcopy(request)
                req["partition_method"] = formula
                # Construct a fresh fold for every method/formula. No fitted
                # parameters or larger sampling condition enter this call.
                subset = _fold_training(training, fold["training_indices"])
                fitted = fitters[method](deepcopy(physiology_by_formula[formula]), req,
                                         subset, budget, starts, guard)
                calls, failures = _cost(fitted, budget)
                fit_calls += calls
                fit_failures += failures
                method_cost["fitting_calls"] += calls
                method_cost["fitting_failures"] += failures
                candidates.append({"formula": formula, "best": _best(fitted),
                                   "calls": calls, "failures": failures,
                                   "call_limit":budget,"no_warm_start":fitted.get("no_warm_start"),
                                   "starts_reported":len(fitted.get("starts",[])),
                                   "budget_exhausted_starts":sum(s.get("status")=="call_budget_exhausted" for s in fitted.get("starts",[]))})
            trace = {**deepcopy(fold), "formula_fits": candidates, "status": "failed",
                     "selected_formula": None, "log_parameters": None,
                     "heldout_abs_log_errors": None, "heldout_mean_abs_log_error": None}
            eligible = [candidate for candidate in candidates if candidate["best"] is not None]
            if not eligible:
                trace["reason"] = "No feasible formula from training fits"
                traces.append(trace)
                continue
            choice = min(eligible, key=lambda row: (row["best"]["objective"], row["formula"]))
            formula = choice["formula"]
            parameters = choice["best"]["log_parameters"]
            trace.update(selected_formula=formula, log_parameters=deepcopy(parameters))
            req = deepcopy(request)
            req["partition_method"] = formula
            indices = fold["heldout_indices"]
            guard()  # Integrity failures must not become a scientific failure.
            validation_calls += 1
            method_cost["validation_calls"] += 1
            try:
                prediction = simulate(deepcopy(physiology_by_formula[formula]), req,
                                      times[indices].copy(), deepcopy(parameters))
                values = np.asarray(prediction["concentration_mg_l"], dtype=float)
                if (values.shape != (len(indices),) or not np.isfinite(values).all()
                        or np.any(values <= 0)):
                    raise ValueError("Invalid held-out concentrations")
                errors = np.abs(np.log(values) - np.log(observed[indices]))
                if not np.isfinite(errors).all():
                    raise ValueError("Nonfinite held-out log errors")
                point_errors.extend(errors.tolist())
                trace.update(status="validated", heldout_abs_log_errors=errors.tolist(),
                             heldout_mean_abs_log_error=float(np.mean(errors)))
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                validation_failures += 1
                method_cost["validation_failures"] += 1
                trace["reason"] = str(exc)
            traces.append(trace)
        eligible = all(row["status"] == "validated" for row in traces)
        score = float(np.mean(point_errors)) if eligible else None
        methods[method] = {"eligible": eligible, "score": score, "folds": traces,
                           "scored_points": len(point_errors), **method_cost}
    candidates = [method for method in METHODS if methods[method]["eligible"]]
    chosen = None
    if candidates:
        minimum = min(methods[method]["score"] for method in candidates)
        chosen = next(method for method in METHODS
                      if method in candidates and methods[method]["score"] <= minimum + TIE_TOLERANCE)
    return {"version": VERSION, "status": "selected" if chosen else "blocked",
            "selected_method": chosen, "methods": methods, "folds": folds,
            "visible_points": len(times), "fitting_calls": fit_calls,
            "fitting_failures": fit_failures, "validation_calls": validation_calls,
            "validation_failures": validation_failures,
            "calls": fit_calls + validation_calls, "failures": fit_failures + validation_failures,
            "budget_per_formula_fit": budget, "starts_per_formula_fit": starts,
            "score_definition": "mean absolute log concentration error across all held-out points",
            "formula_selection": "training objective only, before held-out prediction",
            "tie_tolerance": TIE_TOLERANCE, "tie_preference": "fixed_prior",
            "final_parameters": None, "requires_final_refit": True,
            "no_warm_start": True, "target_observations_used": False}
