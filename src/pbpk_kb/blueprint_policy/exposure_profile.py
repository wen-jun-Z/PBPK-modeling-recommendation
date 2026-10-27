"""Profile systemic clearance using only the visible development-window AUC.

The observed and simulated integrals use the same linear trapezoid weights.
No extrapolation, hidden full curve, target observation or fitted warm start is
accepted. This is constrained development calibration, not an AUC-infinity
estimate or proof of physiological/clinical validity.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import math
import time

import numpy as np
from scipy.optimize import brentq, least_squares

from .perfusion_calibration import clearance_inputs
from .permeability_model import simulate

VERSION = "development-window-exposure-profile-1"
SEED = 20260930
LOWER = np.log([.1,.01])
UPPER = np.log([10.,100.])
NONRENAL_MIN_L_H = 1e-4
UPPER_CARDIAC_MULTIPLIER = 10.
ROOT_XTOL = 1e-10
AUC_RELATIVE_TOLERANCE = 1e-8
ABSOLUTE_JACOBIAN_STEP = 1e-3


class BudgetExhausted(RuntimeError):
    """No additional mathematical model evaluations may be submitted."""


class ProfileInfeasible(ValueError):
    def __init__(self,reason):
        self.reason = reason
        super().__init__(reason)


def _positive(value,name,*,zero=False):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or (value<0 if zero else value<=0):
        raise ValueError(name+" must be finite and "+("nonnegative" if zero else "positive"))
    return float(value)


def visible_window(request,training):
    """Return visible hours, mg/L and positive linear-quadrature weights."""
    if not isinstance(training,dict) or not isinstance(training.get("observations"),list):
        raise ValueError("Only an explicit visible observation table is accepted")
    if training.get("time_unit","min")!="min" or training.get("unit","umol/l")!="umol/l":
        raise ValueError("Development observations must use minutes and umol/l")
    if training.get("compound_id",request.get("compound_id"))!=request.get("compound_id"):
        raise ValueError("Development compound does not match request")
    mw_rows = [r for r in request["evidence"] if r["parameter"]=="molecular_weight"]
    if len(mw_rows)!=1:
        raise ValueError("Exactly one molecular weight is required")
    mw = _positive(mw_rows[0]["value"],"molecular weight")
    rows = training["observations"]
    if len(rows)<3:
        raise ValueError("At least three visible development points are required")
    times = np.asarray([_positive(r["time_min"],"time",zero=True)/60. for r in rows])
    observed = np.asarray([_positive(r["concentration_umol_per_l"],"concentration")*mw/1000. for r in rows])
    if np.any(np.diff(times)<=0):
        raise ValueError("Visible times must be strictly increasing")
    if any(r.get("time_unit","min")!="min" or r.get("unit","umol/l")!="umol/l" for r in rows):
        raise ValueError("Per-row development units differ")
    dt = np.diff(times)
    weights = np.r_[dt[0]/2.,(dt[:-1]+dt[1:])/2.,dt[-1]/2.]
    area = float(weights@observed)
    if not math.isfinite(area) or area<=0:
        raise ValueError("Visible development AUC must be finite and positive")
    return times,observed,weights,area


def initial_starts(count=16):
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=16:
        raise ValueError("starts must be an integer in [1,16]")
    rng = np.random.default_rng(SEED)
    # Taking a prefix retains identical start locations for diagnostic budgets.
    return np.vstack([np.zeros(2),rng.uniform(LOWER,UPPER,size=(15,2))])[:count]


def fit(physiology,request,training,budget=16384,starts=16,before_trial=None):
    """Constrain exposure, then minimize log-shape error plus K/PS penalties.

    Every actual ``simulate`` attempt (including failed bracket/root/Jacobian
    evaluations) consumes the hard budget. Cache hits are reported separately.
    Source renal input stays fixed; the original request is never overwritten.
    ``best.log_parameters`` is compatible with the old three-axis predictor:
    [log(profiled_nonrenal/source_nonrenal), log(Kscale), log(PSscale)].
    """
    guesses = initial_starts(starts)
    if isinstance(budget,bool) or not isinstance(budget,int) or budget<starts:
        raise ValueError("budget must be an integer no smaller than starts")
    times,observed,weights,area = visible_window(request,training)
    cl = clearance_inputs(request)
    renal_argument = cl["renal_clearance_l_h"]
    renal = renal_argument or 0.
    source_nonrenal = cl["total_clearance_l_h"]-renal
    cardiac = _positive(physiology["circulation"]["cardiac_flow_l_h"],"cardiac flow")
    nonrenal_upper = UPPER_CARDIAC_MULTIPLIER*cardiac
    if nonrenal_upper<=NONRENAL_MIN_L_H:
        raise ValueError("Engineering nonrenal bracket is empty")
    lower_cl,upper_cl = math.log(NONRENAL_MIN_L_H),math.log(nonrenal_upper)
    dose = _positive(request["dosing"]["dose_mg"],"dose")
    duration = _positive(request["dosing"]["infusion_minutes"],"infusion minutes",zero=True)/60.
    if request["dosing"].get("start_h",0.)!=0.:
        raise ValueError("This calibration requires dose time zero")
    if request["dosing"].get("route","intravenous")!="intravenous":
        raise ValueError("This calibration is limited to a single IV dose")
    quotas = [budget//starts+int(i<budget%starts) for i in range(starts)]
    calls = failures = profile_requests = profile_cache_hits = root_callback_requests = root_cache_hits = 0
    reasons = Counter()
    all_starts = []
    best = None
    started = time.monotonic()

    for start_index,(x0,quota) in enumerate(zip(guesses,quotas)):
        used = local_failures = 0
        profile_count = cache_count = 0
        local_best = None
        cache = {}
        beginning = time.monotonic()

        def call_model(log_nr,shape):
            nonlocal used,calls,failures,local_failures
            if used>=quota or calls>=budget:
                raise BudgetExhausted()
            # Source-integrity/pause failures abort fitting, rather than being
            # misrepresented as a scientific infeasible model trial.
            if before_trial is not None:
                before_trial()
            used+=1;calls+=1
            try:
                prediction = simulate(physiology,times_h=times,dose_mg=dose,infusion_duration_h=duration,
                    total_clearance_l_h=renal+math.exp(log_nr),renal_clearance_l_h=renal_argument,
                    family="exchange",log_parameters={"clearance_scale":0.,"partition_scale":float(shape[0]),
                        "bidirectional_exchange_scale":float(shape[1])},observation="auto")
                values = np.asarray(prediction["concentration_mg_l"],dtype=float)
                if values.shape!=observed.shape or not np.isfinite(values).all() or np.any(values<=0):
                    raise ValueError("Simulation must provide a positive finite concentration at every visible point")
                predicted_auc = float(weights@values)
                if not math.isfinite(predicted_auc) or predicted_auc<=0:
                    raise ValueError("Invalid predicted sampling-window exposure")
                return values,predicted_auc
            except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                failures+=1;local_failures+=1
                reasons["model_failure:"+type(exc).__name__]+=1
                raise ProfileInfeasible("model_evaluation_failed") from exc

        def profile(shape):
            nonlocal profile_requests,profile_cache_hits,root_callback_requests,root_cache_hits
            nonlocal profile_count,cache_count,local_best,best
            shape = np.asarray(shape,dtype=float)
            if shape.shape!=(2,) or not np.isfinite(shape).all() or np.any(shape<LOWER) or np.any(shape>UPPER):
                raise ProfileInfeasible("shape_outside_fixed_bounds")
            profile_requests+=1
            key = tuple(shape.tolist())
            if key in cache:
                profile_cache_hits+=1;cache_count+=1
                value=cache[key]
                if isinstance(value,str):raise ProfileInfeasible(value)
                return value
            profile_count+=1
            at_begin = used
            trials = {}

            def exposure_error(log_nr):
                nonlocal root_callback_requests,root_cache_hits
                root_callback_requests+=1
                coordinate = float(log_nr)
                if coordinate in trials:
                    root_cache_hits+=1
                else:
                    trials[coordinate]=call_model(coordinate,shape)
                return math.log(trials[coordinate][1]/area)

            try:
                f_lower,f_upper = exposure_error(lower_cl),exposure_error(upper_cl)
                if f_lower+1e-9<f_upper:
                    raise ProfileInfeasible("nonmonotone_exposure_bracket")
                if f_lower<0.:
                    raise ProfileInfeasible("observed_auc_exceeds_min_clearance_exposure")
                if f_upper>0.:
                    raise ProfileInfeasible("observed_auc_below_max_clearance_exposure")
                # The bracket was checked above. Unexpected solver/callback
                # exceptions must propagate: a source guard may itself raise
                # ValueError, which must never become a scientific failure.
                log_nr,root = brentq(exposure_error,lower_cl,upper_cl,xtol=ROOT_XTOL,rtol=1e-12,
                    maxiter=64,full_output=True,disp=False)
                if not root.converged:
                    raise ProfileInfeasible("root_not_converged")
                # The returned root must be an actually simulated point; no
                # interpolation of two endpoint AUCs can count as a solution.
                exposure_error(log_nr)
                concentration,predicted_auc = trials[float(log_nr)]
                relative = abs(predicted_auc-area)/area
                if relative>AUC_RELATIVE_TOLERANCE:
                    raise ProfileInfeasible("root_auc_tolerance_not_met")
                data = (np.log(concentration)-np.log(observed))/math.sqrt(len(observed))
                penalties = np.array([.1*shape[0]/math.log(10.),.1*shape[1]/math.log(100.)])
                residual = np.r_[data,penalties]
                scale = math.exp(log_nr)/source_nonrenal
                row = {"objective":float(residual@residual),"development_ln_rmse":float(np.linalg.norm(data)),
                    "development_gmfe":math.exp(float(np.mean(abs(data))*math.sqrt(len(data)))),
                    "parameters":{"clearance_scale":scale,"partition_scale":math.exp(shape[0]),
                        "bidirectional_exchange_scale":math.exp(shape[1])},
                    "log_parameters":[math.log(scale),float(shape[0]),float(shape[1])],"start_index":start_index,
                    "profiled_nonrenal_clearance_l_h":math.exp(log_nr),
                    "profiled_total_clearance_l_h":renal+math.exp(log_nr),"fixed_renal_clearance_l_h":renal_argument,
                    "observed_window_auc_mg_h_l":area,"predicted_window_auc_mg_h_l":predicted_auc,
                    "relative_auc_error":relative,"root_model_evaluations":used-at_begin,
                    "objective_components":{"shape_log_mse":float(data@data),"K_PS_penalty":float(penalties@penalties),"CL_prior_penalty":0.},
                    "profile_status":"root_converged"}
                value=(residual,row)
                cache[key]=value
                if local_best is None or row["objective"]<local_best["objective"]:local_best=deepcopy(row)
                if best is None or row["objective"]<best["objective"]:best=deepcopy(row)
                return value
            except ProfileInfeasible as exc:
                reasons[exc.reason]+=1;cache[key]=exc.reason
                raise

        def residual(shape):
            try:
                return profile(shape)[0]
            except ProfileInfeasible:
                return np.full(len(observed)+2,1e6)

        def jacobian(shape):
            center = np.asarray(shape,dtype=float)
            baseline = residual(center)
            jac = np.empty((len(baseline),2))
            for column in range(2):
                shifted=center.copy()
                step=ABSOLUTE_JACOBIAN_STEP if center[column]+ABSOLUTE_JACOBIAN_STEP<=UPPER[column] else -ABSOLUTE_JACOBIAN_STEP
                shifted[column]+=step
                jac[:,column]=(residual(shifted)-baseline)/step
            return jac

        status="converged"
        message=""
        try:
            outcome=least_squares(residual,x0,jac=jacobian,bounds=(LOWER,UPPER),max_nfev=400,
                ftol=1e-8,xtol=1e-8,gtol=1e-8)
            message=str(outcome.message)
            if not outcome.success:status="optimizer_limit"
            if local_best is None:status="infeasible"
        except BudgetExhausted:
            status="call_budget_exhausted"
        all_starts.append({"index":start_index,"initial_log_shape_parameters":x0.tolist(),"status":status,
            "optimizer_message":message,"calls":used,"failures":local_failures,"call_limit":quota,
            "profiles_evaluated":profile_count,"profile_cache_hits":cache_count,"best":local_best,
            "wall_seconds":time.monotonic()-beginning})

    return {"version":VERSION,"status":"fitted" if best is not None else "failed", "best":best,"starts":all_starts,
        "calls":calls,"failures":failures,"call_limit":budget,"start_count":starts,
        "profile_requests":profile_requests,"profile_cache_hits":profile_cache_hits,
        "root_callback_requests":root_callback_requests,"root_cache_hits":root_cache_hits,
        "infeasible_reasons":dict(sorted(reasons.items())),"wall_seconds":time.monotonic()-started,
        "no_warm_start":True,"target_observations_used":False,"source_request_modified":False,
        "source_clearance":deepcopy(cl),"nonrenal_clearance_engineering_bounds_l_h":[NONRENAL_MIN_L_H,nonrenal_upper],
        "engineering_bounds_are_physiological_certification":False,
        "observed_window_auc_mg_h_l":area,"window_time_h":[float(times[0]),float(times[-1])],
        "visible_training_points":len(times),"integration":"linear trapezoid on visible development times only",
        "added_zero_time_or_extrapolated_tail":False,"quadrature_weights_h":weights.tolist(),
        "optimizer":{"outer_dimensions":2,"outer_parameters":["partition_scale","bidirectional_exchange_scale"],
            "CL_profile_equation":"visible_grid_trapezoid(P(CL_NR,K,PS)) = visible_grid_trapezoid(O)",
            "absolute_jacobian_step":ABSOLUTE_JACOBIAN_STEP,"seed":SEED,"CL_prior_penalty":0.},
        "limitations":["A development-window quadrature constraint is not AUC infinity or exact continuous exposure.",
            "Sparse sampling can bias quadrature and clearance; the full/five comparison must evaluate this risk.",
            "Clearance is calibrated with the same visible development observations, not independently measured.",
            "A fixed renal construction input does not validate renal exposure or renal clearance clinically.",
            "Conditional calibration and mathematical mass conservation do not establish clinical applicability."]}
