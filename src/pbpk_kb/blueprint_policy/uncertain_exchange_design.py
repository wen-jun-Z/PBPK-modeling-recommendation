"""Soft predicted-half-life constraints on the frozen human PBPK construction.

The caller supplies one frozen, training-only natural-log error width. No
clinical observations or source files are read here. Vss remains anchored and
CL/renal inputs remain fixed; only the common tissue exchange multiplier is
selected. This empirical input tolerance is not a clinical confidence interval.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np
from scipy.optimize import minimize_scalar

from . import learned_exchange_design as frozen
from . import capacity_guided_design as capacity
from . import permeability_model as exchange

VERSION = "uncertain-exchange-design-1"
PS_LOG_SCALE = math.log(100.)


def soft_half_objective(predicted_half, input_half, log_ps, half_log_tolerance):
    """Return the fixed dimensionless input-deviation/structure objective."""
    q = frozen._number(half_log_tolerance, "half_log_tolerance", positive=True)
    predicted = frozen._number(predicted_half, "predicted_half", positive=True)
    target = frozen._number(input_half, "input_half", positive=True)
    log_ps = frozen._number(log_ps, "log_ps")
    error = abs(math.log(predicted) - math.log(target))
    excess = max(error - q, 0.)
    deviation = (excess / max(q, 1e-6)) ** 2
    penalty = (log_ps / PS_LOG_SCALE) ** 2
    return {"objective": deviation + penalty, "absolute_log_half_target_error": error,
            "outside_range_penalty": deviation, "ps_penalty": penalty,
            "within_empirical_half_range": error <= q}


def simulate_uncertain_design(physiology, request, vss_evidence, half_life_evidence, *,
                              half_log_tolerance, log_parameters=None,
                              ps_bounds=(.01, 100.), observation="auto", times_h=None,
                              fallback=True, grid_points=17, max_root_iterations=60,
                              half_relative_tolerance=1e-5, residue_tolerance=1e-12,
                              target_basis="sampled_nca", max_trajectory_calls=33):
    """Return the selected real trajectory, using exactly the declared PS budget.

    Every distinct PS attempt, including failed capacity anchors and unusable
    NCA tails, consumes one budget slot. The default is 33 attempts, never more.
    No exact half-life match is required; strict mode requires range agreement.
    Existing frozen evidence validators and numerical solvers are reused without
    changing their modules or globals. Inputs are not mutated.
    """
    q = frozen._number(half_log_tolerance, "half_log_tolerance", positive=True)
    if target_basis != "sampled_nca":
        raise ValueError("Uncertain exchange design supports sampled_nca only")
    if (isinstance(max_trajectory_calls, bool) or not isinstance(max_trajectory_calls, int)
            or not 3 <= max_trajectory_calls <= 33):
        raise ValueError("max_trajectory_calls must be an integer in [3, 33]")
    if (not isinstance(fallback, bool) or isinstance(grid_points, bool)
            or not isinstance(grid_points, int) or grid_points < 3):
        raise ValueError("fallback must be bool and grid_points at least 3")
    if (isinstance(max_root_iterations, bool) or not isinstance(max_root_iterations, int)
            or max_root_iterations < 1):
        raise ValueError("max_root_iterations must be a positive integer")
    if not isinstance(ps_bounds, (list, tuple)) or len(ps_bounds) != 2:
        raise ValueError("ps_bounds must contain two positive ordered multipliers")
    low, high = [frozen._number(v, "PS bound", positive=True) for v in ps_bounds]
    tolerance = frozen._number(half_relative_tolerance, "half_relative_tolerance", positive=True)
    residue_tolerance = frozen._number(residue_tolerance, "residue_tolerance", positive=True)
    if high <= low or not low <= 1. <= high or tolerance >= 1:
        raise ValueError("PS bounds must include 1 and half-life tolerance must be below 1")
    logs = capacity._logs(log_parameters)
    initial_ps = math.exp(logs["bidirectional_exchange_scale"])
    if not low <= initial_ps <= high:
        raise ValueError("Initial PS must lie inside declared bounds")
    half, diagnostics, reasons = frozen._half_evidence(request, half_life_evidence)
    forbidden = request.get("forbidden_sources", [])
    paths = {r.get("path") for r in forbidden}
    hashes = {r.get("sha256") for r in forbidden}
    for row in request.get("evidence", []):
        context, source = row.get("context", {}), row.get("source", {})
        if (context.get("study_role") in {"target", "evaluation", "test", "validation", "heldout"}
                or source.get("path") in paths or source.get("sha256") in hashes):
            reasons.append("request_uses_forbidden_outcome_evidence:" + str(row.get("parameter")))
    audit = {"version": VERSION, "frozen_design_version": frozen.VERSION,
        "half_constraint_status": "blocked", "selected_half_life_evidence": deepcopy(half["record"]) if half else None,
        "target_half_life_h": half["value_h"] if half else None,
        "half_evidence_diagnostics": diagnostics, "half_log_tolerance": q,
        "tolerance_scale": "natural_log", "tolerance_source_contract": "caller_frozen_current_training_OOF_only",
        "clinical_confidence_interval": False, "clinical_validation": False,
        "target_observations_used": False, "predictions_must_lock_before_scoring": True,
        "source_verification": "caller_must_verify_hashes_extraction_and_independence",
        "half_constraint_role": "predicted_or_independent_input_not_evaluation_truth",
        "half_constraint_basis": "quality_gated_sampled_curve_NCA_on_frozen_prediction_grid",
        "target_basis": target_basis, "ps_bounds": [low, high], "initial_ps": initial_ps,
        "search_baseline_ps": 1., "grid_points": grid_points,
        "half_relative_tolerance": tolerance, "fallback_allowed": fallback,
        "fallback_used": False, "fallback_reason": None,
        "trajectory_budget": max_trajectory_calls, "search_attempts": 0,
        "search_objective": "(max(abs(ln(half/input_half))-q,0)/max(q,1e-6))^2+(ln(PS)/ln(100))^2",
        "ps_log_penalty_scale": PS_LOG_SCALE,
        "selection_rule": "minimum fixed penalized objective; ties prefer PS closest to 1, then lower PS",
        "budget_rule": "all distinct PS attempts including failures; deterministic largest-gap fill after local refinement",
        "protected": ["clearance_scale", "qualified_renal_clearance", "Vss_anchor", "blood_volume",
                      "blood_flow", "blood_to_plasma_ratio"],
        "trials": [], "reasons": reasons, "linear_trajectory_calls": 0, "new_native_calls": 0,
        "limitations": ["training_residual_range_is_not_a_clinical_confidence_interval",
            "input_range_agreement_does_not_guarantee_clinical_accuracy",
            "half_life_and_Vss_prior_errors_remain",
            "bounded_search_does_not_prove_global_optimality",
            "model_NCA_grid_is_not_an_original_clinical_assay_protocol"]}

    def failed(reason):
        return {"version": VERSION, "status": "failed", "reason": reason, "endpoints": {},
                "half_constraint_status": audit["half_constraint_status"], "learned_exchange_audit": audit,
                "clinical_validation": False, "curve_fitting_performed": False}

    if reasons:
        return failed("invalid_or_missing_half_life_evidence")
    vss, vd, vr = capacity._select_evidence(request, vss_evidence)
    audit["vss_evidence_diagnostics"] = vd
    if vr or vss is None:
        audit["reasons"].extend(vr or ["missing_qualified_vss_evidence"])
        return failed("invalid_or_missing_vss_evidence")
    audit.update(selected_vss_evidence=deepcopy(vss["record"]), target_vss_l=vss["target_vss_l"])
    clearance = capacity.clearance_inputs(request)
    times = np.array(capacity.prediction_grid(request) if times_h is None else times_h, dtype=float, copy=True)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Prediction grid must be finite, nonnegative and strictly increasing")
    times.setflags(write=False)
    audit["frozen_grid"] = {"samples": len(times), "start_h": float(times[0]), "end_h": float(times[-1])}
    started = time.monotonic()
    cache = {}

    def trial(log_ps):
        key = 0. if abs(float(log_ps)) < 1e-12 else float(log_ps)
        if key in cache:
            return cache[key]
        if len(cache) >= max_trajectory_calls:
            raise RuntimeError("sampled_NCA_search_budget_exhausted")
        row = {"ps_multiplier": math.exp(key), "log_ps": key, "status": "failed", "trajectory_attempted": False}
        anchor, prediction = None, None
        before = time.monotonic()
        try:
            anchor = capacity.anchor_partition(physiology, request, vss_evidence,
                log_parameters={**logs, "bidirectional_exchange_scale": key}, observation=observation)
            if anchor["status"] != "anchored":
                row["reason"] = ";".join(anchor["reasons"] or ["volume_not_anchored"])
            else:
                row.update(partition_scale=anchor["partition_scale"],
                    anchored_vss_l=anchor["anchored_moments"]["vss_l"], trajectory_attempted=True)
                audit["linear_trajectory_calls"] += 1
                prediction = capacity.simulate_design(physiology, request,
                    log_parameters=anchor["log_parameters"], times_h=times, observation=observation)
                nca = prediction["nca_summary"]["endpoints"]["terminal_half_life"]
                value = nca.get("value_h") if nca.get("status") == "completed" else None
                if value is not None and math.isfinite(value) and value > 0:
                    row.update(status="evaluated", nca_half_life_h=float(value),
                        log_half_target_ratio=math.log(value / half["value_h"]),
                        **soft_half_objective(value, half["value_h"], key, q))
                else:
                    row.update(status="nca_unassessed", reason=nca.get("reason") or "nonestimable_sampled_NCA_half",
                               nca_half_life_h=None)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            row["reason"] = str(error)
        row["elapsed_seconds"] = time.monotonic() - before
        audit["trials"].append(row)
        cache[key] = row, anchor, prediction
        audit["search_attempts"] = len(cache)
        return cache[key]

    def best_of():
        eligible = [r for r in cache.values() if r[0]["status"] == "evaluated"]
        # Exact objective ordering preserves the global zero-cost PS=1 optimum.
        return min(eligible, key=lambda r: (r[0]["objective"], abs(r[0]["log_ps"]), r[0]["log_ps"])) if eligible else None

    baseline = trial(0.)
    lo, hi = math.log(low), math.log(high)
    grid = sorted(set([*np.linspace(lo, hi, min(grid_points, max_trajectory_calls - 1)), 0.]))
    for point in grid:
        trial(point)
    best = best_of()
    if best is not None and len(cache) < max_trajectory_calls:
        position = grid.index(best[0]["log_ps"])
        left, right = grid[max(0, position - 1)], grid[min(len(grid) - 1, position + 1)]
        if left < right:
            try:
                fitted = minimize_scalar(lambda v: trial(v)[0].get("objective", 1e100),
                    bounds=(left, right), method="bounded",
                    options={"maxiter": max_trajectory_calls-len(cache), "xatol": 1e-6})
                audit["local_refinement"] = {"success": bool(fitted.success), "message": str(fitted.message)}
            except (ValueError, ArithmeticError, RuntimeError) as error:
                audit["local_refinement"] = {"success": False, "message": str(error)}
    while len(cache) < max_trajectory_calls:
        ordered = sorted(cache)
        left, right = max(zip(ordered[:-1], ordered[1:]), key=lambda pair: (pair[1]-pair[0], -pair[0]))
        point = (left + right) / 2.
        if point in cache or (abs(point) < 1e-12 and 0. in cache):
            raise ArithmeticError("PS budget cannot be filled with distinct finite candidates")
        trial(point)
    best = best_of()
    selected = best if best is not None else baseline
    row, anchor, prediction = selected
    audit.update(baseline_nca_half_life_h=baseline[0].get("nca_half_life_h"),
        baseline_objective=baseline[0].get("objective"), selected_objective=row.get("objective"),
        baseline_absolute_log_half_target_error=baseline[0].get("absolute_log_half_target_error"),
        selected_absolute_log_half_target_error=row.get("absolute_log_half_target_error"),
        failed_search_attempts=sum(r[0]["status"] != "evaluated" for r in cache.values()),
        selected_trajectory_reused_without_rerun=prediction is not None,
        budget_exhausted=len(cache) == max_trajectory_calls,
        improved_vs_fixed_ps_baseline=best is not None and (baseline[0].get("objective") is None
            or row["objective"] < baseline[0]["objective"]))
    if prediction is None:
        audit["reasons"].append(row.get("reason", "fixed_PS_fallback_infeasible"))
        return failed("no_usable_sampled_NCA_model")
    within = row.get("within_empirical_half_range", False)
    nca_half = row.get("nca_half_life_h")
    exact = nca_half is not None and abs(nca_half / half["value_h"] - 1.) <= tolerance
    audit["half_constraint_status"] = ("within_empirical_range" if within else
        "outside_empirical_range" if nca_half is not None else "unassessed_sampled_nca")
    if not within:
        audit["reasons"].append("half_life_outside_training_residual_range" if nca_half is not None else "half_life_NCA_unassessed")
        if not fallback:
            return failed("half_life_outside_empirical_range")
        audit.update(fallback_used=True, fallback_reason="Best penalized feasible prediction retained without claiming range agreement")
    mode = None
    try:
        system = exchange.build_system(physiology, total_clearance_l_h=clearance["total_clearance_l_h"],
            renal_clearance_l_h=clearance["renal_clearance_l_h"], log_parameters=anchor["log_parameters"], observation=observation)
        mode = frozen.observable_terminal_mode(system.matrix[:-1, :-1], system.input_vector[:-1],
            system.observation_vector[:-1], residue_tolerance=residue_tolerance)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        audit["modal_diagnostic_failure"] = str(error)
    audit.update(chosen_ps=row["ps_multiplier"], chosen_partition_scale=anchor["partition_scale"],
        chosen_on_boundary=any(math.isclose(row["ps_multiplier"], bound, rel_tol=1e-7) for bound in (low, high)),
        selected_capacity_mapping=anchor["capacity_mapping"], achieved_eigen_half_life_h=mode["half_life_h"] if mode else None,
        selected_observable_mode=mode, achieved_nca_half_life_h=nca_half,
        nca_half_target_fold_error=math.exp(row["absolute_log_half_target_error"]) if nca_half is not None else None,
        nca_half_matches_input_at_declared_tolerance=exact if nca_half is not None else None,
        nca_half_within_empirical_range=within if nca_half is not None else None,
        relative_nca_to_eigen_half_difference=abs(nca_half / mode["half_life_h"]-1.) if nca_half and mode else None,
        achieved_model_vss_l=prediction["exact_moments"]["vss_l"],
        achieved_nca_vss_l=prediction["endpoints"].get("vss", {}).get("value"),
        nca_half_life_not_forced_to_input=True, elapsed_seconds=time.monotonic()-started)
    prediction.update(version=VERSION, learned_exchange_audit=audit,
        half_constraint_status=audit["half_constraint_status"], decision=anchor,
        linear_solver_calls=audit["linear_trajectory_calls"], selected_trajectory_solver_calls=1)
    return prediction
