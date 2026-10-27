#!/usr/bin/env python3
"""Fit classical two-compartment IV PK on frozen development curves only.

No target clinical-reference file is decoded. A deterministic multi-start fit
uses the same curated development observations supplied to Ours. Target dose,
infusion, weight and nominal sampling schedule are protocol inputs; prediction
and fit files are locked before downstream clinical scoring in another script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug"
SUMMARY = BASE / "optimized_run/summary.json"
OUTPUT = BASE / "additional_baselines/two_compartment"
PARAMETERS = ("CL_L_h", "Vc_L", "Q_L_h", "Vp_L")
SEED, STARTS, MAX_NFEV = 20260929, 16, 400


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def inside(value, *, exists=True):
    path = (ROOT / value).resolve(strict=exists)
    if not path.is_relative_to(ROOT):
        raise ValueError("All inputs and outputs must remain in this project")
    return path


def sha(path):
    return hashlib.sha256(inside(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(inside(path).read_text())


def save(path, value):
    path = inside(path, exists=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify(pins):
    for path, checksum in pins.items():
        if sha(path) != checksum:
            raise ValueError("Frozen input changed: " + path)


def macro_parameters(parameters):
    cl, vc, q, vp = np.asarray(parameters, dtype=float)
    if not np.isfinite(parameters).all() or min(cl, vc, q, vp) <= 0:
        raise ValueError("Two-compartment parameters must be finite and positive")
    k10, k12, k21 = cl/vc, q/vc, q/vp
    gap = math.sqrt((k10+k12-k21)**2 + 4*k12*k21)
    alpha = (k10+k12+k21+gap)/2
    beta = k10*k21/alpha
    a = (alpha-k21)/gap
    b = (k21-beta)/gap
    return alpha, beta, a, b


def concentration_mg_l(parameters, time_h, *, dose_mg, infusion_h, start_h=0.0):
    """Exact central concentration for a finite, constant-rate IV infusion.

    dAc/dt = rate - (CL+Q)*Ac/Vc + Q*Ap/Vp
    dAp/dt = Q*Ac/Vc - Q*Ap/Vp; C = Ac/Vc.
    Stable convolution of the positive biexponential bolus impulse response.
    """
    if not dose_mg > 0 or not infusion_h > 0 or not start_h >= 0:
        raise ValueError("A positive single IV infusion is required")
    t = np.asarray(time_h, dtype=float)-start_h
    if not np.isfinite(t).all():
        raise ValueError("Nonfinite prediction times")
    active = np.minimum(np.maximum(t, 0), infusion_h)
    post = np.maximum(t-infusion_h, 0)
    alpha, beta, a, b = macro_parameters(parameters)
    value = dose_mg/infusion_h/parameters[1] * (
        a*(-np.expm1(-alpha*active))/alpha*np.exp(-alpha*post)
        + b*(-np.expm1(-beta*active))/beta*np.exp(-beta*post))
    if not np.isfinite(value).all() or (value < 0).any():
        raise ArithmeticError("Invalid analytic concentration")
    return value


def dosing_kwargs(dosing):
    if dosing["route"] != "intravenous":
        raise ValueError("Only single intravenous infusions are supported")
    return {"dose_mg": float(dosing["dose_mg"]),
            "infusion_h": float(dosing["infusion_minutes"])/60,
            "start_h": float(dosing.get("start_h", 0))}


def numerical_self_check():
    """Independently verify infusion convolution against a matrix exponential."""
    maximum_relative_error, maximum_auc_relative_error = 0.0, 0.0
    points = 0
    for parameters in ([5., 10., 15., 30.], [100., 2., .2, 1000.],
                       [.01, 500., 30., .1], [20., 50., 1., 5.]):
        cl, vc, q, vp = parameters
        system = np.asarray([[-(cl+q)/vc, q/vp], [q/vc, -q/vp]])
        for infusion in (.05, .5, 2.5):
            dose, start = 7.5, .2
            grid = np.asarray([0, start, start+.001, start+infusion/2,
                               start+infusion, start+infusion+1e-6,
                               start+infusion+.5, start+infusion+5])
            augmented = np.zeros((3, 3))
            augmented[:2, :2] = system
            augmented[0, 2] = dose/infusion
            independent = []
            for time_h in grid:
                elapsed = max(time_h-start, 0)
                state = (expm(augmented*min(elapsed, infusion)) @ np.asarray([0., 0., 1.]))[:2]
                if elapsed > infusion:
                    state = expm(system*(elapsed-infusion)) @ state
                independent.append(state[0]/vc)
            actual = concentration_mg_l(parameters, grid, dose_mg=dose,
                                         infusion_h=infusion, start_h=start)
            np.testing.assert_allclose(actual, independent, rtol=3e-10, atol=1e-13)
            positive = np.asarray(independent) > 1e-12
            maximum_relative_error = max(maximum_relative_error,
                float(np.max(np.abs(actual[positive]/np.asarray(independent)[positive]-1))))
            alpha, beta, a, b = macro_parameters(parameters)
            total_auc = dose/vc*(a/alpha+b/beta)
            error = abs(total_auc/(dose/cl)-1)
            assert error < 3e-10
            maximum_auc_relative_error = max(maximum_auc_relative_error, error)
            points += len(grid)
    return {"status": "passed", "matrix_exponential_check_points": points,
            "maximum_relative_concentration_error": maximum_relative_error,
            "analytic_auc_equals_dose_over_CL_maximum_relative_error": maximum_auc_relative_error}


def prepare(summary_path, output, workers):
    if output.exists():
        raise FileExistsError("Existing comparator runs cannot be overwritten")
    base = read(summary_path)
    if base["status"] != "reported":
        raise ValueError("Use the frozen, reported ten-drug development comparison")
    source_pins = {str(summary_path.relative_to(ROOT)): sha(summary_path),
                   str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    cases, identities = [], set()
    for case in base["protocol"]["cases"]:
        cid = case["case_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", cid) or cid in identities:
            raise ValueError("Invalid or repeated case ID")
        identities.add(cid)
        development_ref = case["development_source"]
        pins = {development_ref["path"]: development_ref["sha256"],
                case["physchem"]: base["protocol"]["pins"][case["physchem"]]}
        rf_ref = base["locks"]["predictions"][cid]["rf"]
        pins[rf_ref["path"]] = rf_ref["sha256"]
        verify(pins)
        development = read(development_ref["path"])
        physical = read(case["physchem"])
        if development["study_id"] == case["study_id"] or development["compound_id"] != case["compound_id"]:
            raise ValueError("Development/target identity or study separation mismatch")
        for document in (development, physical):
            sources = list(document.get("source_files", []))
            if isinstance(document.get("source"), dict) and document["source"].get("path"):
                sources.append(document["source"])
            for source in sources:
                if source.get("sha256"):
                    pins[source["path"]] = source["sha256"]
        verify(pins)
        # RF concentrations and target clinical outcomes are not fitted inputs.
        # Only the already-fixed native horizon is used for comparable export.
        rf_prediction = read(rf_ref["path"])
        if rf_prediction["time_unit"] != "min":
            raise ValueError("Unexpected native time unit")
        metadata = case["target_request_metadata"]
        horizon_h = max(float(metadata["window_end_h"]), float(rf_prediction["time_min"][-1])/60)
        schedule = case["nca_policy"]["sampling_time_h"]
        if max(schedule) > horizon_h:
            raise ValueError("Target export horizon does not cover the nominal protocol")
        mw = float(physical["identity"]["molecular_weight_g_mol"])
        if not math.isclose(mw, development["molecular_weight_g_mol"], rel_tol=1e-8):
            raise ValueError("Development concentration conversion uses a different molecular weight")
        source_pins.update(pins)
        cases.append({"case_id": cid, "drug": case["drug"], "compound_id": case["compound_id"],
            "study_id": case["study_id"], "development_source": development_ref,
            "target_request_metadata": metadata, "molecular_weight_g_mol": mw,
            "horizon_h": horizon_h, "nominal_sampling_time_h": schedule,
            "source_pins": pins})
    if len(cases) != 10:
        raise ValueError("This prespecified comparator expects all ten frozen drugs")
    summary = {"status": "prepared", "method": "two_compartment", "created_at": now(),
        "protocol": {"source_summary": str(summary_path.relative_to(ROOT)), "cases": cases,
            "source_pins": source_pins, "seed": SEED, "starts_per_case": STARTS,
            "max_nfev_per_start": MAX_NFEV, "workers": workers,
            "parameters": list(PARAMETERS), "objective": "unweighted mean squared natural-log concentration residual on development observations only",
            "bounds_per_kg": {"lower": [.0001, .001, .0001, .001], "upper": [20., 30., 50., 100.]},
            "bounds_interpretation": "Broad prespecified numerical domains, not validated physiological parameter ranges",
            "optimizer": "scipy least_squares, TRF, bounded log parameters; ftol=xtol=1e-10, gtol=1e-8",
            "selection": "Lowest finite development MSE over all 16 starts; convergence separately reported; no target-based selection",
            "allometry": {"CL_exponent": .75, "Q_exponent": .75, "Vc_exponent": 1., "Vp_exponent": 1.,
                          "reference": "Same-drug development-study body weight; fixed exponents, never fitted on target observations"},
            "target_grid": "At most one minute, plus exact infusion start/end and nominal sampling times; through original frozen RF horizon or target window, whichever longer",
            "target_clinical_reference_decoded": False,
            "information": "Uses the same development curve as Ours; RF/SVR/XGB structure-prior methods have less clinical information",
            "interpretation": "Exploratory development comparison of previously inspected clinical data, not independent blinded validation",
            "limitations": ["Classical lumped compartments, not an organ PBPK model",
                "Only body weight is used for covariate transfer; age, sex, population, height and disease are not separately modeled",
                "Four fitted parameters can be weakly identified from sparse/group-summary curves",
                "Source digitization, population representativeness and possible study overlap limitations remain those of the frozen cohort"]},
        "environment": {"numpy": np.__version__, "scipy": scipy.__version__},
        "numerical_self_check": numerical_self_check(), "results": {}}
    output.mkdir(parents=True)
    save(output/"summary.json", summary)
    return summary


def fit_case(case, output):
    started = time.monotonic()
    verify(case["source_pins"])
    development = read(case["development_source"]["path"])
    if development["time_unit"] != "min" or development["unit"] != "umol/l":
        raise ValueError("Unexpected development units")
    mw = case["molecular_weight_g_mol"]
    times = np.asarray([row["time_min"]/60 for row in development["observations"]])
    observed = np.asarray([row["concentration_umol_per_l"]*mw/1000 for row in development["observations"]])
    kwargs = dosing_kwargs(development["dosing"])
    weight = float(development["human"]["weight_kg"])
    if weight <= 0 or len(times) < 5 or not (observed > 0).all() or not np.isfinite(observed).all() or not (np.diff(times) > 0).all():
        raise ValueError("Insufficient or invalid development observations")
    if min(times) <= kwargs["start_h"]:
        raise ValueError("Positive log-fit observations must follow dosing start")
    lower = weight*np.asarray([.0001, .001, .0001, .001])
    upper = weight*np.asarray([20., 30., 50., 100.])
    # Initialization uses development dose/concentrations only. A rough tail
    # correction helps starting CL, but is not an observation endpoint/label.
    area = float(np.trapezoid(np.r_[0., observed], np.r_[kwargs["start_h"], times]))
    slope = float(np.polyfit(times[-3:], np.log(observed[-3:]), 1)[0])
    if slope < 0:
        area += observed[-1]/(-slope)
    cl_initial = kwargs["dose_mg"]/area
    volume_initial = kwargs["dose_mg"]/max(observed)
    starts = [np.asarray([cl_initial, volume_initial*.15, cl_initial, volume_initial]),
              np.asarray([cl_initial*.5, volume_initial*.5, cl_initial*2, volume_initial*2]),
              np.asarray([cl_initial*2, volume_initial*.2, cl_initial*.5, volume_initial*5]),
              weight*np.asarray([.2, .08, .5, .3])]
    case_seed = SEED + int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(case_seed)
    center = np.asarray([cl_initial, volume_initial*.25, cl_initial, volume_initial])
    starts.extend(center*np.exp(rng.uniform(np.log(.08), np.log(12.), size=(STARTS-4, 4))))
    starts = [np.clip(value, lower*1.001, upper/1.001) for value in starts]
    log_observed = np.log(observed)
    attempts, candidates = [], []
    for number, initial in enumerate(starts, 1):
        calls = 0
        def residual(log_parameters):
            nonlocal calls
            calls += 1
            predicted = concentration_mg_l(np.exp(log_parameters), times, **kwargs)
            return np.log(np.maximum(predicted, np.finfo(float).tiny))-log_observed
        try:
            result = least_squares(residual, np.log(initial), bounds=(np.log(lower), np.log(upper)),
                method="trf", max_nfev=MAX_NFEV, ftol=1e-10, xtol=1e-10, gtol=1e-8)
            parameters = np.exp(result.x)
            mse = float(np.mean(result.fun**2))
            finite = math.isfinite(mse) and np.isfinite(parameters).all()
            attempt = {"start": number, "initial": dict(zip(PARAMETERS, initial.tolist())),
                "converged": bool(result.success), "status": int(result.status), "message": str(result.message),
                "nfev": int(result.nfev), "residual_calls_including_finite_differences": calls,
                "development_log_mse": mse if finite else None, "parameters": dict(zip(PARAMETERS, parameters.tolist())),
                "active_bounds": result.active_mask.tolist(), "optimality": float(result.optimality)}
            if finite:
                candidates.append((mse, number, parameters, result))
        except (ValueError, ArithmeticError, FloatingPointError) as exc:
            attempt = {"start": number, "converged": False, "error": str(exc),
                       "residual_calls_including_finite_differences": calls}
        attempts.append(attempt)
    if not candidates:
        raise RuntimeError("No finite development fit")
    mse, selected, fitted, result = min(candidates, key=lambda item: (item[0], item[1]))
    dev_prediction = concentration_mg_l(fitted, times, **kwargs)
    errors = np.log(dev_prediction/observed)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(result.jac))
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    target = case["target_request_metadata"]
    ratio = float(target["human"]["weight_kg"])/weight
    target_parameters = fitted*np.asarray([ratio**.75, ratio, ratio**.75, ratio])
    target_kwargs = dosing_kwargs(target["dosing"])
    end = target_kwargs["start_h"]+target_kwargs["infusion_h"]
    time_min = np.unique(np.r_[np.arange(0., case["horizon_h"]*60, 1.),
        case["horizon_h"]*60, target_kwargs["start_h"]*60, end*60,
        np.asarray(case["nominal_sampling_time_h"])*60])
    prediction = concentration_mg_l(target_parameters, time_min/60, **target_kwargs)
    assert np.diff(time_min).max() <= 1.000000001
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    assert prediction[time_min == 0][0] == 0 and end*60 in time_min
    alpha, beta, a, b = macro_parameters(target_parameters)
    analytic_auc = target_kwargs["dose_mg"]/target_parameters[1]*(a/alpha+b/beta)
    target_dose_over_cl = target_kwargs["dose_mg"]/target_parameters[0]
    assert math.isclose(analytic_auc, target_dose_over_cl, rel_tol=1e-8)
    folder = output/case["case_id"]
    folder.mkdir()
    native = {"time_min": time_min.tolist(), "concentration": (prediction*1000/mw).tolist(),
              "time_unit": "min", "unit": "umol/l"}
    save(folder/"prediction_locked.json", native)
    native_hash = sha(folder/"prediction_locked.json")
    fit = {"status": "predicted", "method": "two_compartment", "case_id": case["case_id"],
        "drug": case["drug"], "compound_id": case["compound_id"], "target_study_id": case["study_id"],
        "development_study_id": development["study_id"], "development_source": case["development_source"],
        "source_pins": case["source_pins"], "parameters_units": list(PARAMETERS),
        "development_parameters": dict(zip(PARAMETERS, fitted.tolist())),
        "target_parameters": dict(zip(PARAMETERS, target_parameters.tolist())),
        "development_human": development["human"], "target_human": target["human"],
        "development_dosing": development["dosing"], "target_dosing": target["dosing"],
        "body_weight_ratio": ratio, "target_age_sex_height_population_used_in_formula": False,
        "selected_start": selected, "converged": bool(result.success), "starts": attempts,
        "parameter_bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
        "development_curve": {"time_h": times.tolist(), "observed_mg_l": observed.tolist(),
                              "predicted_mg_l": dev_prediction.tolist()},
        "development_metrics": {"observations": len(times), "log_mse": mse,
            "ln_rmse": float(np.sqrt(np.mean(errors**2))), "gmfe": float(np.exp(np.mean(np.abs(errors))))},
        "identifiability": {"jacobian_rank": rank, "jacobian_condition_number": condition,
            "active_bounds": result.active_mask.tolist(),
            "interpretation": "Local diagnostics only; full parameter identification is not established by fit convergence"},
        "prediction_lock": {"path": str((folder/"prediction_locked.json").relative_to(ROOT)), "sha256": native_hash},
        "quality_checks": {"grid_points": len(time_min), "max_step_min": float(np.max(np.diff(time_min))),
            "time_zero_and_exact_infusion_end_present": True, "nonnegative_finite": True,
            "molecular_weight_g_mol": mw, "concentration_conversion": "mg/L * 1000 / molecular_weight_g_mol = umol/L",
            "analytic_auc_mg_h_l": analytic_auc, "dose_over_CL_mg_h_l": target_dose_over_cl,
            "analytic_auc_mass_balance_relative_error": abs(analytic_auc/target_dose_over_cl-1)},
        "target_outcomes_used": False, "created_at": now(), "elapsed_seconds": time.monotonic()-started}
    verify(case["source_pins"])
    save(folder/"fit.json", fit)
    return {"status": "predicted", "converged": bool(result.success), "drug": case["drug"],
        "development_metrics": fit["development_metrics"], "identifiability": fit["identifiability"],
        "fit": {"path": str((folder/"fit.json").relative_to(ROOT)), "sha256": sha(folder/"fit.json")},
        "prediction": fit["prediction_lock"], "elapsed_seconds": fit["elapsed_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(numerical_self_check(), indent=2)); return
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one through four project comparator workers")
    output = inside(args.output, exists=False)
    summary = prepare(inside(args.summary), output, args.workers)
    summary.update(status="running", started_at=now(), pid=os.getpid())
    save(output/"summary.json", summary)
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fit_case, case, output): case for case in summary["protocol"]["cases"]}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    summary["results"][case["case_id"]] = future.result()
                except Exception as exc:
                    summary["results"][case["case_id"]] = {"status": "failed", "error": str(exc)}
                summary.update(updated_at=now(), elapsed_seconds=time.monotonic()-started)
                save(output/"summary.json", summary)
                print(json.dumps({"case_id": case["case_id"], **summary["results"][case["case_id"]]}, allow_nan=False), flush=True)
        verify(summary["protocol"]["source_pins"])
        summary.update(status="completed" if all(v["status"] == "predicted" for v in summary["results"].values()) else "completed_with_failures",
            finished_at=now(), elapsed_seconds=time.monotonic()-started,
            predictions_locked_before_clinical_scoring=True, target_clinical_references_decoded=False)
        summary["counts"] = {"cases": len(summary["results"]),
            "predicted": sum(v["status"] == "predicted" for v in summary["results"].values()),
            "selected_fit_converged": sum(v.get("converged", False) for v in summary["results"].values())}
        save(output/"summary.json", summary)
        if summary["status"] != "completed":
            raise RuntimeError("At least one classical comparator failed; see retained failure records")
    except BaseException as exc:
        summary.update(status="failed", failure=str(exc), finished_at=now())
        save(output/"summary.json", summary)
        raise
    print(json.dumps({"status": summary["status"], "counts": summary["counts"], "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
