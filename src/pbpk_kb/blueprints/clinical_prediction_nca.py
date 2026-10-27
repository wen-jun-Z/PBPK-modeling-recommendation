"""NCA of a frozen predicted IV-infusion curve on a clinical sampling schedule.

This module accepts predictions and protocol constants only: no observed
concentrations, reference PK values, model parameters, files, or fitting target.
It never modifies a PBPK model. It approximates clinical NCA; it does not claim
to recover the original study's individual terminal-point selections.

Concentrations are ng/mL, time is hours, dose is the TOTAL administered ng, and
weight is kg. CL is returned in L/h/kg; Vz and Vss are returned in L/kg.
The supplied sampling schedule must start at time zero and be covered by the
prediction. Model concentration at time zero is retained, not replaced by zero.
Other BLQ samples are omitted from NCA; no LLOQ/2 or zero is imputed. Tail
regression uses quantified samples strictly after infusion end and the observed
sampled maximum. All suffixes of at least three such samples are considered.
Selection maximizes adjusted R-squared; within 0.0001 of the maximum, the longer
range is selected, as in the published Phoenix Best Fit rule. No clinical
reference or prediction-error criterion selects the range.
Rule source: https://onlinehelp.certara.com/phoenix/8.5/Phoenix_UserDocs/
NCA_Object/Lambda_Z_or_Slope_Estimation_settings.htm

Linear-up/log-down trapezoids integrate AUC; AUMC integrates t*C(t) analytically
under the same piecewise linear/exponential concentration assumption. All
infinity-dependent endpoints share the one selected lambda_z. No extra
adjusted-R-squared, tail-fraction, or span threshold is imposed silently; these
diagnostics are returned for the caller's frozen protocol. Missing tails and
nonpositive infusion-corrected MRT are retained as unavailable predictions.
"""
from __future__ import annotations

import bisect
import math
from numbers import Real
from typing import Sequence

UNITS = {
    "auc": "ng.h/mL", "cmax": "ng/mL", "tmax": "h", "c_last": "ng/mL",
    "terminal_half_life": "h", "auc_inf": "ng.h/mL", "clearance": "L/h/kg",
    "vz": "L/kg", "mrt": "h", "vss": "L/kg",
}


def _number(value: Real, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _sample(times: list[float], concentrations: list[float], time: float) -> float:
    index = bisect.bisect_left(times, time)
    if index < len(times) and times[index] == time:
        return concentrations[index]
    if index == 0 or index == len(times):
        raise ValueError("Prediction does not cover every nominal sampling time")
    weight = (time-times[index-1])/(times[index]-times[index-1])
    return concentrations[index-1] + weight*(concentrations[index]-concentrations[index-1])


def _integrals(times: list[float], concentrations: list[float]) -> tuple[float, float]:
    """Return AUC and AUMC for the same linear-up/log-down interpolant."""
    areas, moments = [], []
    for t0, t1, c0, c1 in zip(times, times[1:], concentrations, concentrations[1:]):
        dt = t1-t0
        if 0 < c1 < c0:
            z = math.log(c1)-math.log(c0)
            if abs(z) < 1e-4:
                # Stable series for integral_0^1 exp(z*u) du and u*exp(z*u) du.
                phi0 = math.fsum(z**n/(math.factorial(n)*(n+1)) for n in range(7))
                phi1 = math.fsum(z**n/(math.factorial(n)*(n+2)) for n in range(7))
            else:
                phi0 = math.expm1(z)/z
                phi1 = (z*math.exp(z)-math.expm1(z))/(z*z)
            area = c0*dt*phi0
            first_moment = c0*dt*dt*phi1
        else:
            area = dt*(c0+c1)/2
            first_moment = dt*dt*(c0+2*c1)/6
        areas.append(area)
        moments.append(t0*area+first_moment)
    return math.fsum(areas), math.fsum(moments)


def _tail_fit(times: list[float], concentrations: list[float]) -> dict | None:
    n = len(times)
    mean_t = math.fsum(times)/n
    logs = [math.log(x) for x in concentrations]
    mean_log = math.fsum(logs)/n
    sxx = math.fsum((t-mean_t)**2 for t in times)
    sst = math.fsum((y-mean_log)**2 for y in logs)
    if sxx == 0 or sst == 0:
        return None
    slope = math.fsum((t-mean_t)*(y-mean_log) for t, y in zip(times, logs))/sxx
    if not math.isfinite(slope) or slope >= 0:
        return None
    intercept = mean_log-slope*mean_t
    residuals = math.fsum((y-(intercept+slope*t))**2 for t, y in zip(times, logs))
    r_squared = min(1.0, max(0.0, 1-residuals/sst))
    adjusted = 1-(1-r_squared)*(n-1)/(n-2)
    return {"n": n, "start_h": times[0], "end_h": times[-1],
            "time_h": times, "lambda_z_per_h": -slope,
            "intercept_ln_ng_ml": intercept, "r_squared": r_squared,
            "adjusted_r_squared": adjusted}


def compute_clinical_prediction_nca(
    *, time_h: Sequence[Real], concentration_ng_ml: Sequence[Real], dose_ng: Real,
    infusion_h: Real, nominal_times_h: Sequence[Real], lloq_ng_ml: Real,
    weight_kg: Real,
) -> dict:
    """Return ten predicted endpoint records and transparent NCA diagnostics.

    status is completed, partial, or invalid_input. Each endpoint contains
    status=completed/unavailable, value, unit and (if unavailable) reason. No
    observed/reference endpoint is read or returned. Failure of a tail does
    not erase available finite-window endpoints.
    """
    endpoints = {key: {"status": "unavailable", "value": None, "unit": unit,
                       "reason": "Not yet computed"} for key, unit in UNITS.items()}
    diagnostics: dict = {
        "method": "predicted_curve_standard_NCA",
        "reference_values_used": False, "model_parameters_modified": False,
        "concentration_interpolation": "linear on frozen dense prediction",
        "integration": "linear-up/log-down concentration; matching analytic t*C integral",
        "blq_handling": "Retain actual model predose value; omit other samples below LLOQ; no imputation",
        "terminal_selection": "Post-infusion and post-sampled-Cmax quantified suffixes, >=3 points, negative slope; max adjusted R2; longest within 0.0001",
        "clinical_interpretation": "Standard NCA approximation; original individual terminal-fitting points are unavailable",
        "failures": [],
    }

    def assign(key: str, value: float) -> None:
        if not math.isfinite(value) or value < 0 or (key != "tmax" and value == 0):
            endpoints[key]["reason"] = f"Nonfinite/nonpositive predicted {key}"
            diagnostics["failures"].append(endpoints[key]["reason"])
        else:
            endpoints[key] = {"status": "completed", "value": value, "unit": UNITS[key]}

    def unavailable(keys: Sequence[str], reason: str) -> None:
        for key in keys:
            endpoints[key] = {"status": "unavailable", "value": None,
                              "unit": UNITS[key], "reason": reason}
        if reason not in diagnostics["failures"]:
            diagnostics["failures"].append(reason)

    def result() -> dict:
        complete = all(x["status"] == "completed" for x in endpoints.values())
        return {"status": "completed" if complete else "partial", "endpoints": endpoints,
                "diagnostics": diagnostics}

    try:
        times = [_number(x, "prediction time") for x in time_h]
        concentrations = [_number(x, "predicted concentration") for x in concentration_ng_ml]
        schedule = [_number(x, "nominal time") for x in nominal_times_h]
        dose = _number(dose_ng, "total dose", positive=True)
        infusion = _number(infusion_h, "infusion duration", positive=True)
        lloq = _number(lloq_ng_ml, "LLOQ", positive=True)
        weight = _number(weight_kg, "body weight", positive=True)
        if len(times) != len(concentrations) or len(times) < 2:
            raise ValueError("Prediction requires paired time/concentration arrays")
        if len(schedule) < 2 or schedule[0] != 0:
            raise ValueError("Nominal schedule must contain time zero and at least one later sample")
        if any(b <= a for a, b in zip(times, times[1:])) or any(b <= a for a, b in zip(schedule, schedule[1:])):
            raise ValueError("Prediction and nominal times must be strictly increasing")
        sampled = [_sample(times, concentrations, t) for t in schedule]
    except (ValueError, TypeError) as exc:
        unavailable(list(UNITS), str(exc))
        return {"status": "invalid_input", "endpoints": endpoints, "diagnostics": diagnostics}

    quantified = [value >= lloq for value in sampled]
    q_indices = [i for i, ok in enumerate(quantified) if ok and schedule[i] > 0]
    diagnostics.update(dose_ng=dose, infusion_h=infusion, lloq_ng_ml=lloq,
                       weight_kg=weight, sampled_time_h=schedule,
                       sampled_concentration_ng_ml=sampled, quantified_mask=quantified)
    if not q_indices:
        unavailable(list(UNITS), "No post-dose sample is quantifiable at the declared LLOQ")
        return result()
    last = q_indices[-1]
    retained = sorted({0, *q_indices})
    rt, rc = [schedule[i] for i in retained], [sampled[i] for i in retained]
    auc, aumc = _integrals(rt, rc)
    peak_index = max(q_indices, key=lambda i: sampled[i])
    t_last, c_last = schedule[last], sampled[last]
    for key, value in {"auc": auc, "cmax": sampled[peak_index], "tmax": schedule[peak_index], "c_last": c_last}.items():
        assign(key, value)
    diagnostics.update(nca_sample_time_h=rt, nca_sample_concentration_ng_ml=rc,
                       auc_last_ng_h_ml=auc, aumc_last_ng_h2_ml=aumc,
                       tlast_h=t_last, sampled_tmax_h=schedule[peak_index],
                       blq_samples_before_last_omitted=sum(not quantified[i] for i in range(1, last)),
                       internal_blq_samples_omitted=sum(not quantified[i] for i in range(q_indices[0]+1, last)))

    tail_indices = [i for i in q_indices if schedule[i] > infusion and schedule[i] > schedule[peak_index]]
    candidates = []
    for n in range(3, len(tail_indices)+1):
        subset = tail_indices[-n:]
        fit = _tail_fit([schedule[i] for i in subset], [sampled[i] for i in subset])
        if fit is not None:
            candidates.append(fit)
    diagnostics["terminal_candidates"] = candidates
    dependent = ("terminal_half_life", "auc_inf", "clearance", "vz", "mrt", "vss")
    if not candidates:
        unavailable(dependent, "No negative-slope terminal fit with >=3 quantified post-infusion/post-Cmax samples")
        return result()
    best_adjusted = max(c["adjusted_r_squared"] for c in candidates)
    chosen = max((c for c in candidates if c["adjusted_r_squared"] >= best_adjusted-0.0001), key=lambda c: c["n"])
    lam = chosen["lambda_z_per_h"]
    try:
        half_life = math.log(2)/lam
        auc_tail = c_last/lam
        aumc_tail = t_last*auc_tail + auc_tail/lam
        auc_inf, aumc_inf = auc+auc_tail, aumc+aumc_tail
        # ng / (ng*h/mL) gives mL/h: convert to L/h BEFORE normalizing once.
        cl_l_h = dose/auc_inf/1000
        mrt = aumc_inf/auc_inf-infusion/2
        diagnostics.update(selected_tail=chosen, lambda_z_per_h=lam,
                           selected_tail_times_h=chosen["time_h"],
                           adjusted_r_squared=chosen["adjusted_r_squared"],
                           auc_infinity_ng_h_ml=auc_inf, aumc_infinity_ng_h2_ml=aumc_inf,
                           auc_extrapolated_fraction=auc_tail/auc_inf,
                           aumc_extrapolated_fraction=aumc_tail/aumc_inf,
                           terminal_span_in_half_lives=(chosen["end_h"]-chosen["start_h"])/half_life,
                           absolute_clearance_l_h=cl_l_h, uncorrected_residence_time_h=aumc_inf/auc_inf)
        for key, value in {"terminal_half_life": half_life, "auc_inf": auc_inf,
                           "clearance": cl_l_h/weight, "vz": cl_l_h/lam/weight}.items():
            assign(key, value)
        if mrt <= 0 or not math.isfinite(mrt):
            unavailable(("mrt", "vss"), "Infusion-corrected MRT is not positive and finite")
        else:
            assign("mrt", mrt)
            assign("vss", cl_l_h*mrt/weight)
    except (OverflowError, ZeroDivisionError) as exc:
        unavailable(dependent, f"Terminal extrapolation is numerically undefined: {type(exc).__name__}")
    return result()
