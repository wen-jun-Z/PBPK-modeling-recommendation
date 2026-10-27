"""Subset-only classical PK controls for the frozen blueprint validation protocol.

These analytic CPU fits are not native PBPK executions. Clinical target outcomes
are never decoded here; scoring belongs to the caller after prediction locking.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
import scipy
from scipy.optimize import least_squares

import run_classical_pk_comparator as classical
from run_classical_pk_comparator import concentration_mg_l, dosing_kwargs, numerical_self_check

PARAMETERS = classical.PARAMETERS
SEED, STARTS, MAX_NFEV = 20260929, 16, 400
PRIOR_WEIGHT = .1
HORIZON_H = 1344.


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _inside(path, root, *, exists=True):
    result = (root / path).resolve(strict=exists)
    if result == root or not result.is_relative_to(root):
        raise ValueError("All resources must remain in this project")
    return result


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if len(payload.encode()) > 16 * 1024**2:
        raise ValueError("Classical output exceeds the bounded artifact size")
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _ref(path, root):
    return {"path": str(path.relative_to(root)), "sha256": _sha(path)}


def _verify(pins, root):
    for path, expected in pins.items():
        if _sha(_inside(path, root)) != expected:
            raise ValueError("Frozen source changed: " + path)


def _embedded_pins(value, pins, root):
    # Hashing a primary source/forbidden target does not decode its outcomes.
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = str(_inside(value["path"], root).relative_to(root))
            if path in pins and pins[path] != value["sha256"]:
                raise ValueError("Conflicting source locks: " + path)
            pins[path] = value["sha256"]
        for item in value.values():
            _embedded_pins(item, pins, root)
    elif isinstance(value, list):
        for item in value:
            _embedded_pins(item, pins, root)


def _clearance_prior(request, weight):
    evidence = {item["parameter"]: item for item in request["evidence"]}
    if "total_clearance" in evidence:
        items = [evidence["total_clearance"]]
        rule = "total_clearance"
    elif {"hepatic_clearance", "renal_clearance"} <= evidence.keys():
        if evidence.get("elimination_scope", {}).get("value") != "hepatic_and_renal":
            raise ValueError("Summed clearance prior needs hepatic_and_renal scope")
        items = [evidence["hepatic_clearance"], evidence["renal_clearance"]]
        rule = "hepatic_clearance + renal_clearance"
    else:
        raise ValueError("No legal same-request total clearance prior")
    forbidden = {item["path"] for item in request.get("forbidden_sources", [])}
    for item in items:
        if item["unit"] != "ml/min/kg" or item.get("compound_id") != request["compound_id"]:
            raise ValueError("Clearance prior units or identity mismatch")
        if (isinstance(item["value"], bool) or not math.isfinite(item["value"])
                or item["value"] < 0 or item["source"]["path"] in forbidden):
            raise ValueError("Invalid or forbidden clinical clearance prior")
    per_kg = sum(float(item["value"]) for item in items)
    if per_kg <= 0:
        raise ValueError("Positive clearance prior is required")
    return {"value_L_h": per_kg * weight * .06,
            "value_ml_min_kg": per_kg, "development_weight_kg": weight,
            "conversion": "ml/min/kg * development weight kg * 0.06 = L/h",
            "rule": rule, "role": "same construction evidence used by Ours; not a target outcome",
            "evidence": deepcopy(items), "weight": PRIOR_WEIGHT,
            "penalty": "0.1 * (ln(CL / CL_prior) / ln(2))^2"}


def _initial_parameters(times, observed, kwargs, weight, case_id):
    """All data-derived starts use only the passed training rows, never old fits."""
    lower = weight * np.asarray([.0001, .001, .0001, .001])
    upper = weight * np.asarray([20., 30., 50., 100.])
    area = float(np.trapezoid(np.r_[0., observed], np.r_[kwargs["start_h"], times]))
    slope = float(np.polyfit(times[-3:], np.log(observed[-3:]), 1)[0])
    if slope < 0:
        area += observed[-1] / -slope
    cl = kwargs["dose_mg"] / area
    volume = kwargs["dose_mg"] / max(observed)
    starts = [np.asarray([cl, volume*.15, cl, volume]),
              np.asarray([cl*.5, volume*.5, cl*2, volume*2]),
              np.asarray([cl*2, volume*.2, cl*.5, volume*5]),
              weight*np.asarray([.2, .08, .5, .3])]
    seed = SEED + int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    center = np.asarray([cl, volume*.25, cl, volume])
    starts.extend(center*np.exp(rng.uniform(np.log(.08), np.log(12.), size=(STARTS-4, 4))))
    starts = [np.clip(value, lower*1.001, upper/1.001) for value in starts]
    return starts, lower, upper, {"training_observations": len(times), "case_seed": seed,
        "rough_training_auc_mg_h_l": area, "last_three_training_log_slope_per_h": slope,
        "initial_CL_L_h": cl, "initial_volume_L": volume,
        "clinical_endpoint_label": False, "old_fit_or_full_curve_read": False}


def _residual_vector(log_parameters, times, observed, kwargs, prior):
    prediction = concentration_mg_l(np.exp(log_parameters), times, **kwargs)
    errors = np.log(np.maximum(prediction, np.finfo(float).tiny)) - np.log(observed)
    residual = errors / math.sqrt(len(errors))
    if prior is not None:
        penalty = math.sqrt(PRIOR_WEIGHT) * (log_parameters[0]-math.log(prior)) / math.log(2)
        residual = np.r_[residual, penalty]
    return residual


def _target_grid(metadata, policy):
    kwargs = dosing_kwargs(metadata["dosing"])
    nominal = np.asarray(policy["sampling_time_h"], dtype=float)
    if nominal.ndim != 1 or not np.isfinite(nominal).all() or (nominal < 0).any():
        raise ValueError("Invalid nominal target schedule")
    end = kwargs["start_h"] + kwargs["infusion_h"]
    horizon = max(HORIZON_H, float(metadata["window_end_h"]), float(max(nominal)), end)
    # Build in minutes, round sub-picosecond float artifacts before union. The
    # extra uniqueness after conversion avoids duplicates in common NCA hours.
    times = np.round(np.r_[np.arange(0., 48.*60, 1.),
        np.arange(48.*60, horizon*60, 15.), horizon*60,
        kwargs["start_h"]*60, end*60, nominal*60], 10)
    hours = np.unique(times / 60)
    if not (np.diff(hours) > 0).all():
        raise ValueError("Non-increasing target grid")
    return hours, kwargs


def fit_predict(*, case: dict, request: dict, training_source: dict,
                output_dir: Path, project_root: Path, with_cl_prior: bool) -> dict:
    """Fit on this supplied subset only, lock a fresh target curve, or verify reuse."""
    root = project_root.resolve(strict=True)
    folder = _inside(output_dir, root, exists=False)
    request = deepcopy(request.model_dump(mode="json") if hasattr(request, "model_dump") else request)
    started = time.monotonic()
    pins = {}
    for document in (training_source, request, case):
        _embedded_pins(document, pins, root)
    for path in (Path(__file__).resolve(), Path(classical.__file__).resolve()):
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
    if (not math.isfinite(weight) or weight <= 0 or len(times) < 5
        or not np.isfinite(times).all() or not np.isfinite(observed).all()
        or not (observed > 0).all() or not (np.diff(times) > 0).all()
        or min(times) <= kwargs["start_h"]):
        raise ValueError("At least five finite positive ordered development observations are required")
    prior = _clearance_prior(request, weight) if with_cl_prior else None
    method = "two_compartment_cl_prior" if with_cl_prior else "two_compartment"
    fingerprint = _digest({"case": case, "request": request, "training_source": training_source,
        "source_pins": pins, "method": method, "seed": SEED, "starts": STARTS,
        "max_nfev": MAX_NFEV, "prior_weight": PRIOR_WEIGHT, "horizon_h": HORIZON_H})
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
    elif any(folder.iterdir()):
        raise ValueError("Nonempty output has no auditable fit checkpoint")
    starts, lower, upper, initialization = _initial_parameters(times, observed, kwargs, weight, case["case_id"])
    fit = {"status": "fitting", "method": method, "case_id": case["case_id"],
        "input_fingerprint": fingerprint, "source_pins": pins, "training_source": training_source,
        "development_study_id": development["study_id"], "target_study_id": case["study_id"],
        "initialization": initialization, "clearance_prior": prior, "restart_history": history,
        "objective": "mean(log(C_pred/C_obs)^2)" + (" + 0.1*(ln(CL/CL_prior)/ln(2))^2" if prior else ""),
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
                                    prior["value_L_h"] if prior else None)
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
    fit.update(status="predicted", selected_start=selected, converged=bool(optimized.success),
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
        "prediction_locked_before_scoring": True}
    _save(completed, result)
    return result
