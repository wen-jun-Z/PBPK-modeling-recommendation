"""Analytic two-compartment controls for three-or-more observed development points.

This adapter leaves the frozen five-point comparator untouched. It uses the
identical starts, analytic solver and objective at prior_multiplier=1. Sparse
fits are allowed to be underidentified and this is reported, not concealed by
inventing measurements. Target outcomes are never decoded before prediction.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import least_squares

import blueprint_validation_classical as frozen
from blueprint_validation_classical import (
    classical, concentration_mg_l, dosing_kwargs, numerical_self_check,
    PARAMETERS, SEED, STARTS, MAX_NFEV, PRIOR_WEIGHT, HORIZON_H,
    _digest, _inside, _sha, _save, _ref, _verify, _embedded_pins,
    _clearance_prior, _initial_parameters, _target_grid,
)

VERSION = "sparse-classical-pk-1"
RANK_RELATIVE_THRESHOLD = 1e-6


def _valid_multiplier(value, with_cl_prior):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("prior_multiplier must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("prior_multiplier must be a finite positive number")
    if not with_cl_prior and value != 1.:
        raise ValueError("A non-default prior_multiplier requires with_cl_prior=True")
    return value


def _residual_vector(log_parameters, times, observed, kwargs, prior, prior_multiplier=1.):
    prediction = concentration_mg_l(np.exp(log_parameters), times, **kwargs)
    errors = np.log(np.maximum(prediction, np.finfo(float).tiny)) - np.log(observed)
    residual = errors / math.sqrt(len(errors))
    if prior is not None:
        penalty = math.sqrt(PRIOR_WEIGHT * prior_multiplier) * (log_parameters[0]-math.log(prior)) / math.log(2)
        residual = np.r_[residual, penalty]
    return residual


def _identifiability(log_parameters, times, observed, kwargs, prior, prior_multiplier, *, jacobian=None):
    """Local log-parameter sensitivity; convergence is not identifiability.

    Data-only numerical central differences are independent of the regularizer.
    Rank is diagnostic at this fitted point, not a global identifiability proof.
    """
    supplied = jacobian is not None
    if not supplied:
        h = 1e-5
        jacobian = np.empty((len(times), len(PARAMETERS)))
        for j in range(len(PARAMETERS)):
            delta = np.zeros(len(PARAMETERS)); delta[j] = h
            plus = _residual_vector(log_parameters+delta, times, observed, kwargs, None)
            minus = _residual_vector(log_parameters-delta, times, observed, kwargs, None)
            jacobian[:, j] = (plus-minus)/(2*h)
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.shape != (len(times), len(PARAMETERS)) or not np.isfinite(jacobian).all():
        raise ValueError("Invalid data-only sensitivity Jacobian")
    def diagnostic(matrix):
        singular = np.linalg.svd(matrix, compute_uv=False)
        cutoff = float(singular[0] * RANK_RELATIVE_THRESHOLD) if len(singular) else 0.
        rank = int(np.count_nonzero(singular > max(cutoff, np.finfo(float).eps)))
        return {"rank": rank, "singular_values": singular.tolist(),
                "relative_threshold": RANK_RELATIVE_THRESHOLD,
                "full_parameter_rank": rank == len(PARAMETERS),
                "condition_number": float(singular[0]/singular[-1]) if rank == len(PARAMETERS) else None}
    data = diagnostic(jacobian)
    augmented = None
    if prior is not None:
        prior_row = np.zeros(len(PARAMETERS))
        prior_row[0] = math.sqrt(PRIOR_WEIGHT * prior_multiplier)/math.log(2)
        augmented = diagnostic(np.vstack([jacobian, prior_row]))
    return {"parameters": list(PARAMETERS), "observations": len(times),
            "free_parameters": len(PARAMETERS), "data_only": data,
            "with_regularizer": augmented,
            "underdetermined_by_observation_count": len(times) < len(PARAMETERS),
            "data_rank_deficient": not data["full_parameter_rank"],
            "interpretation": "Local sensitivity of log concentrations to four log parameters; numerical convergence does not establish parameter identifiability; prior rows are not observations.",
            "jacobian_source": "selected scipy least_squares data rows" if supplied else "central finite differences",
            "diagnostic_analytic_calls": 0 if supplied else 2*len(PARAMETERS)}


def fit_predict(*, case: dict, request: dict, training_source: dict,
                output_dir: Path, project_root: Path, with_cl_prior: bool,
                prior_multiplier: float = 1.) -> dict:
    """Lock subset-only fit/prediction; optional prior strength is caller-frozen."""
    _valid_multiplier(prior_multiplier, with_cl_prior)
    root = project_root.resolve(strict=True)
    folder = _inside(output_dir, root, exists=False)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/".fit.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This sparse classical output already has an active writer") from exc
        return _fit_predict(case=case, request=request, training_source=training_source,
            output_dir=folder, project_root=root, with_cl_prior=with_cl_prior,
            prior_multiplier=prior_multiplier)


def _fit_predict(*, case: dict, request: dict, training_source: dict,
                output_dir: Path, project_root: Path, with_cl_prior: bool, prior_multiplier: float = 1.) -> dict:
    """Fit on this supplied subset only, lock a fresh target curve, or verify reuse."""
    multiplier = _valid_multiplier(prior_multiplier, with_cl_prior)
    root = project_root.resolve(strict=True)
    folder = _inside(output_dir, root, exists=False)
    request = deepcopy(request.model_dump(mode="json") if hasattr(request, "model_dump") else request)
    started = time.monotonic()
    pins = {}
    for document in (training_source, request, case):
        _embedded_pins(document, pins, root)
    for path in (Path(__file__).resolve(), Path(frozen.__file__).resolve(), Path(classical.__file__).resolve()):
        path = _inside(path, root)
        pins[str(path.relative_to(root))] = _sha(path)
    _verify(pins, root)
    source = _inside(training_source["path"], root)
    if _sha(source) != training_source["sha256"]:
        raise ValueError("Training subset checksum mismatch")
    development = json.loads(source.read_text())
    _embedded_pins(development, pins, root)
    # No reading case.development, any historical fitted parameters, or target O.
    physical_path = case["physchem"]["path"] if isinstance(case["physchem"], dict) else case["physchem"]
    physical_path = _inside(physical_path, root)
    physical_key = str(physical_path.relative_to(root))
    actual_physical_hash = _sha(physical_path)
    if physical_key in pins and pins[physical_key] != actual_physical_hash:
        raise ValueError("Physicochemical source checksum mismatch")
    pins[physical_key] = actual_physical_hash
    physical = json.loads(physical_path.read_text())
    _embedded_pins(physical, pins, root)
    _verify(pins, root)
    mw = float(physical["identity"]["molecular_weight_g_mol"])
    if (not math.isfinite(mw) or mw <= 0 or
        not math.isclose(mw, development["molecular_weight_g_mol"], rel_tol=1e-8)):
        raise ValueError("Invalid or mismatched molecular weight")
    if (development["time_unit"] != "min" or development["unit"] != "umol/l"
        or development["compound_id"] != case["compound_id"]
        or request["compound_id"] != case["compound_id"]
        or development["study_id"] == case["study_id"]):
        raise ValueError("Training units, identity or study separation mismatch")
    if (dosing_kwargs(development["dosing"]) != dosing_kwargs(request["dosing"])
        or any(request["human"].get(key) != value for key, value in development["human"].items())):
        raise ValueError("Training observations and development request conditions differ")
    times = np.asarray([row["time_min"]/60 for row in development["observations"]], dtype=float)
    observed = np.asarray([row["concentration_umol_per_l"]*mw/1000
                           for row in development["observations"]], dtype=float)
    kwargs = dosing_kwargs(request["dosing"])
    weight = float(request["human"]["weight_kg"])
    if (not math.isfinite(weight) or weight <= 0 or len(times) < 3
        or not np.isfinite(times).all() or not np.isfinite(observed).all()
        or not (observed > 0).all() or not (np.diff(times) > 0).all()
        or min(times) <= kwargs["start_h"]):
        raise ValueError("At least three finite positive ordered development observations are required")
    prior = _clearance_prior(request, weight) if with_cl_prior else None
    if prior is not None:
        prior.update(weight=PRIOR_WEIGHT * multiplier, prior_multiplier=multiplier,
            penalty=f"{PRIOR_WEIGHT * multiplier:g} * (ln(CL / CL_prior) / ln(2))^2",
            interpretation="Engineering regularization; not a calibrated confidence interval")
    method = "two_compartment_cl_prior" if with_cl_prior else "two_compartment"
    fingerprint = _digest({"case": case, "request": request, "training_source": training_source,
        "source_pins": pins, "method": method, "seed": SEED, "starts": STARTS,
        "max_nfev": MAX_NFEV, "prior_weight": PRIOR_WEIGHT, "prior_multiplier": multiplier,
        "version": VERSION, "rank_relative_threshold": RANK_RELATIVE_THRESHOLD, "horizon_h": HORIZON_H})
    completed = folder / "result.json"
    if completed.exists():
        previous = json.loads(completed.read_text())
        if previous["input_fingerprint"] != fingerprint or previous["status"] != "predicted":
            raise ValueError("Existing result belongs to different or incomplete inputs")
        _verify(previous["source_pins"], root)
        for key in ("fit", "prediction", "request"):
            _verify({previous[key]["path"]: previous[key]["sha256"]}, root)
        return {**previous, "reused": True, "this_invocation_residual_calls": 0}
    folder.mkdir(parents=True, exist_ok=True)
    history = []
    fit_path = folder / "fit.json"
    if fit_path.exists():
        previous = json.loads(fit_path.read_text())
        if previous["input_fingerprint"] != fingerprint:
            raise ValueError("Incomplete output belongs to different frozen inputs")
        history = previous.get("restart_history", []) + [{
            "previous_status": previous["status"], "completed_starts": len(previous.get("starts", [])),
            "known_residual_calls": sum(a["residual_calls_including_finite_differences"]
                                        for a in previous.get("starts", [])),
            "interrupted_active_start_calls_unknown": previous.get("active_start") is not None}]
    elif any(p.name != ".fit.lock" for p in folder.iterdir()):
        raise ValueError("Nonempty output has no auditable fit checkpoint")
    starts, lower, upper, initialization = _initial_parameters(times, observed, kwargs, weight, case["case_id"])
    fit = {"version": VERSION, "status": "fitting", "method": method, "case_id": case["case_id"],
        "input_fingerprint": fingerprint, "source_pins": pins, "training_source": training_source,
        "development_study_id": development["study_id"], "target_study_id": case["study_id"],
        "initialization": initialization, "clearance_prior": prior, "restart_history": history,
        "objective": "mean(log(C_pred/C_obs)^2)" + (f" + {PRIOR_WEIGHT * multiplier:g}*(ln(CL/CL_prior)/ln(2))^2" if prior else ""),
        "optimizer": {"method": "scipy least_squares TRF", "starts": STARTS, "max_nfev_per_start": MAX_NFEV,
                      "seed": SEED, "ftol": 1e-10, "xtol": 1e-10, "gtol": 1e-8},
        "parameter_bounds": {"lower": lower.tolist(), "upper": upper.tolist(), "units": list(PARAMETERS)},
        "starts": [], "target_outcomes_used": False, "native_pbpk_executed": False,
        "interpretation": "Exploratory clinical-data comparison, not newly blinded validation",
        "environment": {"numpy": np.__version__, "scipy": scipy.__version__},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    candidates = []
    for number, initial in enumerate(starts, 1):
        fit["active_start"] = number
        _save(fit_path, fit)
        calls = 0
        def residual(parameters):
            nonlocal calls
            calls += 1
            return _residual_vector(parameters, times, observed, kwargs,
                                    prior["value_L_h"] if prior else None, multiplier)
        try:
            optimized = least_squares(residual, np.log(initial), bounds=(np.log(lower), np.log(upper)),
                method="trf", max_nfev=MAX_NFEV, ftol=1e-10, xtol=1e-10, gtol=1e-8)
            fitted = np.exp(optimized.x)
            objective = float(np.sum(optimized.fun**2))
            if not math.isfinite(objective) or not np.isfinite(fitted).all():
                raise ArithmeticError("Nonfinite optimizer result")
            attempt = {"start": number, "initial": dict(zip(PARAMETERS, initial.tolist())),
                "converged": bool(optimized.success), "status": int(optimized.status),
                "message": str(optimized.message), "objective": objective, "nfev": int(optimized.nfev),
                "parameters": dict(zip(PARAMETERS, fitted.tolist())), "active_bounds": optimized.active_mask.tolist(),
                "residual_calls_including_finite_differences": calls}
            candidates.append((objective, number, fitted, optimized))
        except (ValueError, ArithmeticError, FloatingPointError) as exc:
            attempt = {"start": number, "converged": False, "error": str(exc),
                       "residual_calls_including_finite_differences": calls}
        fit["starts"].append(attempt)
        fit["active_start"] = None
        _save(fit_path, fit)
    if not candidates:
        fit["status"] = "failed"
        _save(fit_path, fit)
        raise RuntimeError("No finite subset-only classical fit; all attempts retained")
    objective, selected, fitted, optimized = min(candidates, key=lambda item: (item[0], item[1]))
    dev_prediction = concentration_mg_l(fitted, times, **kwargs)
    errors = np.log(np.maximum(dev_prediction, np.finfo(float).tiny)/observed)
    target = deepcopy(case["target_request_metadata"])
    ratio = float(target["human"]["weight_kg"])/weight
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid target weight")
    target_parameters = fitted*np.asarray([ratio**.75, ratio, ratio**.75, ratio])
    hours, target_kwargs = _target_grid(target, case["nca_policy"])
    prediction = concentration_mg_l(target_parameters, hours, **target_kwargs)
    native = {"time_min": (hours*60).tolist(), "concentration": (prediction*1000/mw).tolist(),
              "time_unit": "min", "unit": "umol/l"}
    if not (np.diff(np.asarray(native["time_min"])/60) > 0).all():
        raise ArithmeticError("Exported curve contains duplicate times")
    target_request = deepcopy(request)
    target_request.update(human=target["human"], dosing=target["dosing"])
    target_request["target"].update(window_start_h=0., window_end_h=float(target["window_end_h"]), terminal_fit=None)
    alpha, beta, a, b = classical.macro_parameters(target_parameters)
    analytic_auc = target_kwargs["dose_mg"]/target_parameters[1]*(a/alpha+b/beta)
    auc_error = abs(analytic_auc/(target_kwargs["dose_mg"]/target_parameters[0])-1)
    if auc_error > 1e-8:
        raise ArithmeticError("Analytic AUC mass balance failed")
    this_calls = sum(a["residual_calls_including_finite_differences"] for a in fit["starts"])
    resources = {"wall_seconds": time.monotonic()-started, "analytic_cpu_residual_calls": this_calls,
        "optimizer_nfev": sum(a.get("nfev", 0) for a in fit["starts"]), "native_execution_calls": 0,
        "previous_known_residual_calls": sum(h["known_residual_calls"] for h in history),
        "prior_interrupted_calls_unknown": any(h["interrupted_active_start_calls_unknown"] for h in history)}
    identifiability = _identifiability(np.log(fitted), times, observed, kwargs,
        prior["value_L_h"] if prior else None, multiplier, jacobian=optimized.jac[:len(times)])
    fit.update(status="predicted", identifiability=identifiability, selected_start=selected, converged=bool(optimized.success),
        selected_objective=objective, development_parameters=dict(zip(PARAMETERS, fitted.tolist())),
        target_parameters=dict(zip(PARAMETERS, target_parameters.tolist())),
        development_curve={"time_h": times.tolist(), "observed_mg_l": observed.tolist(), "predicted_mg_l": dev_prediction.tolist()},
        development_metrics={"observations": len(times), "log_mse": float(np.mean(errors**2)),
                             "gmfe": float(np.exp(np.mean(np.abs(errors))))},
        allometry={"body_weight_ratio": ratio, "CL_exponent": .75, "Q_exponent": .75, "Vc_exponent": 1., "Vp_exponent": 1.,
                   "other_covariates_used": False},
        resources=resources, quality_checks={"nonnegative_finite": bool(np.isfinite(prediction).all() and (prediction >= 0).all()),
        "grid_points": len(hours), "grid": "60 points/h through 48h; 4 points/h thereafter; exact protocol times included",
        "horizon_h": float(hours[-1]), "molecular_weight_g_mol": mw,
        "concentration_conversion": "mg/L * 1000 / molecular weight g/mol = umol/L",
        "analytic_auc_mass_balance_relative_error": auc_error})
    _verify(pins, root)
    _save(folder/"request.json", target_request)
    _save(folder/"prediction_locked.json", native)
    fit["prediction_lock"] = _ref(folder/"prediction_locked.json", root)
    fit["request_lock"] = _ref(folder/"request.json", root)
    _save(fit_path, fit)
    result = {"status": "predicted", "method": method, "case_id": case["case_id"],
        "input_fingerprint": fingerprint, "source_pins": pins, "fit": _ref(fit_path, root),
        "prediction": fit["prediction_lock"], "request": fit["request_lock"],
        "converged": bool(optimized.success), "development_metrics": fit["development_metrics"],
        "resources": resources, "reused": False, "this_invocation_residual_calls": this_calls,
        "target_outcomes_used": False, "native_pbpk_executed": False,
        "prediction_locked_before_scoring": True, "identifiability": identifiability,
        "prior_multiplier": multiplier, "version": VERSION}
    _save(completed, result)
    return result
