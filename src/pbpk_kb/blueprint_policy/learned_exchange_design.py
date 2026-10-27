"""Map independently predicted Vss and terminal half-life onto human PBPK.

CL, blood volumes, blood flow, blood partition and renal clearance remain fixed.
Only the shared extravascular capacity K and effective exchange multiplier PS
change. Every PS trial re-anchors K to the same qualified Vss input. A terminal
mode calculation remains available for compatibility. The sampled-NCA path
instead searches actual full trajectories with the same grid and NCA used for
reported predictions; an asymptotic root never substitutes for that endpoint.

No files, clinical target observations, GPU service or source labels are read.
Callers verify source contents/independence and lock all predictions before
scoring. An unreachable predicted half-life may use an explicitly requested
fixed-PS fallback, never a silently clipped or falsely matched boundary value.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
import time
import warnings

import numpy as np
from scipy.linalg import eig, solve, LinAlgWarning
from scipy.optimize import brentq, minimize_scalar

from . import capacity_guided_design as capacity
from . import permeability_model as exchange

VERSION = "learned-exchange-design-2"
PARAMETERS = exchange.PARAMETERS["exchange"]


def _number(value, name, *, positive=False):
    return capacity._number(value, name, positive=positive)


def observable_terminal_mode(matrix, input_vector, observation_vector, *, residue_tolerance=1e-12):
    """Return the slowest stable real mode that is both excited and observed.

    Arguments are the *transient* amount matrix and vectors (no cumulative
    eliminated state). Zero input/observer residues cannot define the plasma
    tail. A relative numerical threshold uses both instantaneous amplitude and
    integrated modal area, retaining small-amplitude but long-lived modes.
    It is not an assay detection limit; sampled-curve NCA is checked separately.
    """
    a, b, c = np.asarray(matrix, dtype=float), np.asarray(input_vector, dtype=float), np.asarray(observation_vector, dtype=float)
    tol = _number(residue_tolerance, "residue_tolerance", positive=True)
    if tol >= 1 or a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != (len(a),) or c.shape != b.shape:
        raise ValueError("Expected square transient matrix and matching vectors")
    if not all(np.isfinite(x).all() for x in (a, b, c)) or np.any(b < 0) or np.any(c < 0):
        raise ValueError("Finite nonnegative input and observation vectors are required")
    offdiag = a - np.diag(np.diag(a))
    if np.any(offdiag < -1e-12) or np.any(np.diag(a) >= 0):
        raise ValueError("Expected a stable positive compartment-system generator")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            roots, vectors = eig(a, check_finite=True)
            condition = float(np.linalg.cond(vectors))
            if not math.isfinite(condition) or condition > 1e10:
                raise ArithmeticError("Terminal eigensystem is ill-conditioned")
            coefficients = solve(vectors, b, assume_a="gen", check_finite=False)
        except LinAlgWarning as exc:
            raise ArithmeticError("Terminal modal decomposition is unresolved") from exc
    if np.any(roots.real >= 0) or not np.isfinite(roots).all():
        raise ArithmeticError("Transient system contains nondecaying modes")
    residues = (c @ vectors) * coefficients
    amplitude_total = float(np.abs(residues).sum())
    area_total = float(np.abs(residues / roots).sum())
    if amplitude_total <= 0 or area_total <= 0 or not math.isfinite(amplitude_total + area_total):
        raise ArithmeticError("Input has no observable concentration response")
    observable = [i for i in range(len(roots)) if max(abs(residues[i]) / amplitude_total,
                   abs(residues[i] / roots[i]) / area_total) > tol]
    if not observable:
        raise ArithmeticError("No numerically observable terminal mode")
    selected = max(observable, key=lambda i: roots[i].real)
    root, residue = roots[selected], residues[selected]
    if abs(root.imag) > 1e-8 * max(1., abs(root.real)):
        raise ArithmeticError("Leading observable terminal mode is not real")
    rate = float(-root.real)
    return {"half_life_h": math.log(2.) / rate, "decay_rate_per_h": rate,
        "selected_eigenvalue_real": float(root.real), "selected_eigenvalue_imag": float(root.imag),
        "modal_residue_real": float(residue.real), "modal_residue_imag": float(residue.imag),
        "relative_amplitude": float(abs(residue) / amplitude_total),
        "relative_absolute_modal_area": float(abs(residue / root) / area_total),
        "eigenvector_condition_number": condition, "observable_mode_count": len(observable),
        "excluded_unexcited_or_unobserved_mode_count": len(roots) - len(observable),
        "residue_tolerance": tol,
        "interpretation": "Asymptotic transfer-function half-life, not sampled clinical NCA or an assay-detectable tail"}


def _half_evidence(request, records):
    if not isinstance(records, list):
        raise ValueError("half_life_evidence must be a sidecar list")
    allowed = {"parameter", "compound_id", "value", "unit", "basis", "source", "context", "prediction_method", "notes"}
    forbidden = request.get("forbidden_sources", [])
    paths, hashes = ({row.get(key) for row in forbidden} for key in ("path", "sha256"))
    accepted, diagnostics, fatal = [], [], []
    for index, row in enumerate(records):
        issue = None
        if not isinstance(row, dict) or set(row) - allowed:
            issue = "invalid_half_life_evidence_schema"
        else:
            source, context = row.get("source"), row.get("context")
            if not isinstance(source, dict) or not isinstance(context, dict):
                issue = "invalid_half_life_source_or_context"
            elif (source.get("path") in paths or source.get("sha256") in hashes
                    or context.get("study_role") in {"target", "evaluation", "test", "validation", "heldout"}
                    or context.get("independent_of_target") is not True):
                issue = "held_out_or_nonindependent_half_life_evidence"
            elif row.get("compound_id") != request["compound_id"] or row.get("parameter") != "terminal_half_life":
                issue = "half_life_compound_or_parameter_mismatch"
            elif (not all(isinstance(source.get(k), str) and source[k].strip() for k in ("path", "location"))
                    or not isinstance(source.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None):
                issue = "unbound_half_life_source"
            elif (context.get("species") != "human" or context.get("matrix") != "plasma"
                    or context.get("route") not in {"iv", "intravenous"}
                    or context.get("study_role") not in {"independent_parameters", "independent_evidence", "construction_prior"}
                    or row.get("basis") not in {"measured", "derived", "predicted"}):
                issue = "unsupported_half_life_context"
            elif row["basis"] == "predicted":
                method = row.get("prediction_method") or {}
                applicability = method.get("applicability") if isinstance(method, dict) else None
                if (not isinstance(applicability, dict) or not method.get("name") or not method.get("version")
                        or method.get("training_overlap") is not False
                        or applicability.get("target_structure_and_scaffold_excluded") is not True
                        or applicability.get("human_IV_parameter_model") is not True):
                    issue = "unqualified_half_life_prediction"
        if issue is None:
            try:
                value = _number(row.get("value"), "half-life evidence", positive=True)
                if not isinstance(row.get("unit"), str) or row["unit"].lower() != "h":
                    raise ValueError("Half-life evidence must use h")
            except (ValueError, TypeError):
                issue = "invalid_half_life_value_or_unit"
        if issue:
            fatal.append(issue); diagnostics.append({"index": index, "status": "rejected", "reason": issue})
        else:
            accepted.append({"index": index, "record": deepcopy(row), "value_h": value,
                             "priority": 1 if row["basis"] == "predicted" else 0})
    if fatal:
        return None, diagnostics, list(dict.fromkeys(fatal))
    if not accepted:
        return None, diagnostics, ["missing_qualified_half_life_evidence"]
    best = [row for row in accepted if row["priority"] == min(x["priority"] for x in accepted)]
    if any(not math.isclose(row["value_h"], best[0]["value_h"], rel_tol=1e-9) for row in best[1:]):
        return None, diagnostics, ["equal_priority_half_life_conflict"]
    selected = best[0]
    diagnostics.extend({"index": row["index"], "status": "selected" if row is selected else "retained_not_selected"}
                       for row in accepted)
    return selected, diagnostics, []


def _sampled_nca_search(physiology, request, vss_evidence, half, logs, *,
                        clearance, low, high, observation, times_h, fallback,
                        grid_points, tolerance, residue_tolerance, budget,
                        audit, failed):
    """Bounded real-trajectory search; observations and modal roots do not rank trials."""
    started = time.monotonic()
    # Freeze exactly one time grid for every candidate, including PS=1.
    times = np.array(capacity.prediction_grid(request) if times_h is None else times_h,
                     dtype=float, copy=True)
    times.setflags(write=False)
    audit.update(half_constraint_basis="quality_gated_sampled_curve_NCA_on_frozen_prediction_grid",
        search_baseline_ps=1., supplied_initial_ps=math.exp(logs["bidirectional_exchange_scale"]),
        trajectory_budget=budget, search_attempts=0,
        search_objective="absolute_log_ratio_of_predicted_curve_NCA_half_to_independent_input_half",
        search_tie_rule="within 1e-12 objective: closest log(PS) to zero, then lower PS",
        selection_rule="change PS only for strictly lower input error than PS=1; any estimable candidate beats an unassessed baseline",
        approximation_rule="retain nearest evaluated feasible candidate only if it improves baseline; never label approximate agreement as matched",
        nca_policy="same capacity.simulate_design and summarize_extended_prediction as final output",
        numerical_floor_is_not_assay_lloq=True, modal_values_used_for_selection=False,
        clinical_scope="Agreement with a predicted or independent parameter input on a model-only sampling grid; not clinical validation",
        frozen_grid={"samples": len(times), "start_h": float(times[0]) if len(times) else None,
                     "end_h": float(times[-1]) if len(times) else None},
        limitations=["predicted_half_life_error_is_not_removed_by_matching_it",
            "model_NCA_grid_and_numerical_floor_are_not_an_original_clinical_assay_protocol",
            "bounded_budget_search_does_not_prove_global_unreachability",
            "input_agreement_improvement_does_not_guarantee_clinical_error_improvement",
            "RBC_partition_conflicts_are_not_silently_overridden"])
    cache = {}

    def trial(log_ps):
        key = 0. if abs(float(log_ps)) < 1e-12 else float(log_ps)
        if key in cache:
            return cache[key]
        if len(cache) >= budget:
            raise RuntimeError("sampled_NCA_search_budget_exhausted")
        row = {"ps_multiplier": math.exp(key), "log_ps": key, "status": "failed",
               "trajectory_attempted": False}
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
                        absolute_log_half_target_error=abs(math.log(value / half["value_h"])))
                else:
                    row.update(status="nca_unassessed", reason=nca.get("reason") or "nonestimable_sampled_NCA_half",
                               nca_half_life_h=None)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            row["reason"] = str(exc)
        row["elapsed_seconds"] = time.monotonic() - before
        audit["trials"].append(row)
        cache[key] = (row, anchor, prediction)
        audit["search_attempts"] = len(cache)
        return cache[key]

    def best_of(results):
        eligible = [r for r in results if r[0]["status"] == "evaluated"]
        if not eligible:
            return None
        minimum = min(r[0]["absolute_log_half_target_error"] for r in eligible)
        ties = [r for r in eligible if r[0]["absolute_log_half_target_error"] <= minimum + 1e-12]
        return min(ties, key=lambda r: (abs(r[0]["log_ps"]), r[0]["log_ps"]))

    baseline = trial(0.)
    # Candidate failures, including anchoring failures, consume the same budget.
    points = min(grid_points, budget - 1)
    grid = sorted(set(0. if abs(x) < 1e-12 else float(x)
                      for x in [*np.linspace(math.log(low), math.log(high), points), 0.]))
    for value in grid:
        trial(value)
    best = best_of(cache.values())
    if best is not None and len(cache) < budget:
        index = grid.index(best[0]["log_ps"])
        left, right = grid[max(0, index-1)], grid[min(len(grid)-1, index+1)]
        if left < right:
            def objective(value):
                row = trial(value)[0]
                return row.get("absolute_log_half_target_error", 1e100)
            try:
                optimized = minimize_scalar(objective, bounds=(left, right), method="bounded",
                    options={"maxiter": budget-len(cache), "xatol": 1e-6})
                audit["local_refinement"] = {"status": "converged" if optimized.success else "budget_or_tolerance_stop",
                    "message": str(optimized.message), "bounds_log_ps": [left, right]}
            except (ValueError, ArithmeticError, RuntimeError) as exc:
                audit["local_refinement"] = {"status": "stopped", "reason": str(exc)}
    best = best_of(cache.values())
    baseline_error = baseline[0].get("absolute_log_half_target_error")
    selected = baseline
    improved = best is not None and (baseline_error is None or
        best[0]["absolute_log_half_target_error"] < baseline_error - 1e-12)
    if improved:
        selected = best
    row, anchor, prediction = selected
    audit.update(baseline_nca_half_life_h=baseline[0].get("nca_half_life_h"),
        baseline_absolute_log_half_target_error=baseline_error,
        selected_absolute_log_half_target_error=row.get("absolute_log_half_target_error"),
        improved_vs_fixed_ps_baseline=bool(improved),
        failed_search_attempts=sum(r[0]["status"] != "evaluated" for r in cache.values()),
        selected_trajectory_reused_without_rerun=prediction is not None)
    if prediction is None:
        audit["reasons"].append(row.get("reason", "fixed_PS_fallback_infeasible"))
        return failed("no_usable_sampled_NCA_model")
    nca_half = row.get("nca_half_life_h")
    matched = nca_half is not None and abs(nca_half / half["value_h"] - 1.) <= tolerance
    if not matched and not fallback:
        audit["half_constraint_status"] = "not_matched_within_declared_search"
        return failed("half_life_unreachable")
    audit["half_constraint_status"] = ("matched" if matched else
        "improved_not_matched" if improved else "baseline_not_matched" if nca_half is not None else "unassessed_sampled_nca")
    if not matched:
        audit["reasons"].append("half_life_not_matched_within_declared_sampled_NCA_search")
    if not improved:
        audit.update(fallback_used=not matched, fallback_reason=(None if matched else
            "Retain PS=1: no strictly better estimable input agreement within the declared budget"))
    mode = None
    try:
        system = exchange.build_system(physiology, total_clearance_l_h=clearance["total_clearance_l_h"],
            renal_clearance_l_h=clearance["renal_clearance_l_h"],
            log_parameters=anchor["log_parameters"], observation=observation)
        mode = observable_terminal_mode(system.matrix[:-1, :-1], system.input_vector[:-1],
            system.observation_vector[:-1], residue_tolerance=residue_tolerance)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        audit["modal_diagnostic_failure"] = str(exc)
    audit.update(chosen_ps=row["ps_multiplier"], chosen_partition_scale=anchor["partition_scale"],
        chosen_on_boundary=any(math.isclose(row["ps_multiplier"], b, rel_tol=1e-7) for b in (low, high)),
        selected_capacity_mapping=anchor["capacity_mapping"],
        achieved_eigen_half_life_h=mode["half_life_h"] if mode else None,
        selected_observable_mode=mode, achieved_nca_half_life_h=nca_half,
        nca_half_target_fold_error=math.exp(row["absolute_log_half_target_error"]) if nca_half is not None else None,
        nca_half_matches_input_at_declared_tolerance=matched if nca_half is not None else None,
        relative_nca_to_eigen_half_difference=abs(nca_half / mode["half_life_h"]-1.) if nca_half and mode else None,
        achieved_model_vss_l=prediction["exact_moments"]["vss_l"],
        achieved_nca_vss_l=prediction["endpoints"].get("vss", {}).get("value"),
        nca_half_life_not_forced_to_input=True, elapsed_seconds=time.monotonic()-started)
    prediction.update(version=VERSION, learned_exchange_audit=audit,
        half_constraint_status=audit["half_constraint_status"], decision=anchor,
        linear_solver_calls=audit["linear_trajectory_calls"], selected_trajectory_solver_calls=1)
    return prediction


def simulate_learned_design(physiology, request, vss_evidence, half_life_evidence, *,
                           log_parameters=None, ps_bounds=(0.01, 100.), observation="auto",
                           times_h=None, fallback=True, grid_points=17, max_root_iterations=60,
                           half_relative_tolerance=1e-5, residue_tolerance=1e-12,
                           target_basis="modal", max_trajectory_calls=33):
    """Return one real trajectory plus the complete predicted-target audit.

    ``fallback=True`` retains the caller's original PS (default 1) when the
    predicted half-life is unreachable over the bounded search. It never
    weakens Vss feasibility, alters fixed physiology, or labels fallback as a
    matched half-life. Source/qualification conflicts always fail explicitly.
    Scalar PS is dimensionless; initial log parameters use natural logarithms.
    ``target_basis='sampled_nca'`` uses actual quality-gated NCA as the search
    objective, with a fixed PS=1 comparator and at most 33 attempted candidates.
    Only improved input agreement changes PS; this does not guarantee improved
    clinical accuracy. All failed attempts count against that search budget.
    """
    if target_basis not in {"modal", "sampled_nca"}:
        raise ValueError("target_basis must be modal or sampled_nca")
    if (isinstance(max_trajectory_calls, bool) or not isinstance(max_trajectory_calls, int)
            or not 3 <= max_trajectory_calls <= 33):
        raise ValueError("max_trajectory_calls must be an integer in [3, 33]")
    if not isinstance(fallback, bool) or not isinstance(grid_points, int) or isinstance(grid_points, bool) or grid_points < 3:
        raise ValueError("fallback must be bool and grid_points at least 3")
    if not isinstance(max_root_iterations, int) or isinstance(max_root_iterations, bool) or max_root_iterations < 1:
        raise ValueError("max_root_iterations must be a positive integer")
    if not isinstance(ps_bounds, (tuple, list)) or len(ps_bounds) != 2:
        raise ValueError("ps_bounds must contain two positive ordered multipliers")
    low, high = [_number(v, "PS bound", positive=True) for v in ps_bounds]
    tolerance = _number(half_relative_tolerance, "half_relative_tolerance", positive=True)
    if high <= low or tolerance >= 1:
        raise ValueError("Invalid PS search bounds or half-life tolerance")
    if target_basis == "sampled_nca" and not low <= 1. <= high:
        raise ValueError("Sampled NCA search must include its fixed PS=1 baseline")
    logs = capacity._logs(log_parameters)
    # Calibration of CL is explicitly outside this new construction operation.
    initial_ps = math.exp(logs["bidirectional_exchange_scale"])
    if not low <= initial_ps <= high:
        raise ValueError("Initial fallback PS must lie inside the declared search bounds")
    half, evidence_diagnostics, reasons = _half_evidence(request, half_life_evidence)
    forbidden = request.get("forbidden_sources", [])
    forbidden_paths = {row.get("path") for row in forbidden}
    forbidden_hashes = {row.get("sha256") for row in forbidden}
    request_issues = []
    for record in request.get("evidence", []):
        context, source = record.get("context", {}), record.get("source", {})
        if (context.get("study_role") in {"target", "evaluation", "test", "validation", "heldout"}
                or source.get("path") in forbidden_paths or source.get("sha256") in forbidden_hashes):
            request_issues.append("request_uses_forbidden_outcome_evidence:" + str(record.get("parameter")))
    reasons.extend(request_issues)
    audit = {"version": VERSION, "half_constraint_status": "blocked", "selected_half_life_evidence": half["record"] if half else None,
        "target_half_life_h": half["value_h"] if half else None, "half_evidence_diagnostics": evidence_diagnostics,
        "ps_bounds": [low, high], "initial_ps": initial_ps, "grid_points": grid_points,
        "half_relative_tolerance": tolerance, "fallback_allowed": fallback, "fallback_used": False,
        "fallback_reason": None, "source_verification": "caller_must_verify_hashes_extraction_and_independence",
        "half_constraint_role": "predicted_or_independent_input_not_evaluation_truth",
        "half_constraint_basis": "observable_asymptotic_transfer_mode_not_NCA",
        "target_basis": target_basis,
        "clinical_validation": False, "target_observations_used": False,
        "predictions_must_lock_before_scoring": True,
        "protected": ["clearance_scale", "qualified_renal_clearance", "blood_volume", "blood_flow", "blood_to_plasma_ratio"],
        "trials": [], "reasons": reasons, "linear_trajectory_calls": 0, "new_native_calls": 0,
        "limitations": ["predicted_half_life_error_is_not_removed_by_matching_it",
                        "observable_asymptotic_mode_may_differ_from_sampled_NCA_tail",
                        "bounded_PS_search_does_not_prove_global_unreachability",
                        "RBC_partition_conflicts_are_not_silently_overridden"]}

    def failed(reason):
        return {"version": VERSION, "status": "failed", "reason": reason, "endpoints": {},
                "half_constraint_status": audit["half_constraint_status"], "learned_exchange_audit": audit,
                "clinical_validation": False, "curve_fitting_performed": False}

    if reasons:
        return failed("invalid_or_missing_half_life_evidence")
    # Validate Vss before constructing any half-life search candidate.
    vss, diagnostics, vss_reasons = capacity._select_evidence(request, vss_evidence)
    audit["vss_evidence_diagnostics"] = diagnostics
    if vss_reasons or vss is None:
        audit["reasons"].extend(vss_reasons or ["missing_qualified_vss_evidence"])
        return failed("invalid_or_missing_vss_evidence")
    audit["selected_vss_evidence"] = deepcopy(vss["record"])
    audit["target_vss_l"] = vss["target_vss_l"]
    clearance = capacity.clearance_inputs(request)
    if target_basis == "sampled_nca":
        return _sampled_nca_search(physiology, request, vss_evidence, half, logs,
            clearance=clearance, low=low, high=high, observation=observation,
            times_h=times_h, fallback=fallback, grid_points=grid_points,
            tolerance=tolerance, residue_tolerance=residue_tolerance,
            budget=max_trajectory_calls, audit=audit, failed=failed)
    cache = {}

    def trial(log_ps):
        key = float(log_ps)
        if key in cache:
            return cache[key]
        ps = math.exp(key)
        row = {"ps_multiplier": ps, "log_ps": key, "status": "failed"}
        anchor, mode = None, None
        try:
            anchor = capacity.anchor_partition(physiology, request, vss_evidence,
                log_parameters={**logs, "bidirectional_exchange_scale": key}, observation=observation)
            if anchor["status"] != "anchored":
                row["reason"] = ";".join(anchor["reasons"] or ["volume_not_anchored"])
            else:
                system = exchange.build_system(physiology, total_clearance_l_h=clearance["total_clearance_l_h"],
                    renal_clearance_l_h=clearance["renal_clearance_l_h"],
                    log_parameters=anchor["log_parameters"], observation=observation)
                mode = observable_terminal_mode(system.matrix[:-1, :-1], system.input_vector[:-1],
                    system.observation_vector[:-1], residue_tolerance=residue_tolerance)
                row.update(status="evaluated", eigen_half_life_h=mode["half_life_h"],
                    log_half_target_ratio=math.log(mode["half_life_h"] / half["value_h"]),
                    partition_scale=anchor["partition_scale"], anchored_vss_l=anchor["anchored_moments"]["vss_l"],
                    observable_mode=mode)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            row["reason"] = str(exc)
        audit["trials"].append(row)
        cache[key] = (row, anchor, mode)
        return cache[key]

    log_grid = sorted(set(np.linspace(math.log(low), math.log(high), grid_points).tolist() + [math.log(initial_ps)]))
    grid = [(x, trial(x)) for x in log_grid]
    valid = [(x, result) for x, result in grid if result[0]["status"] == "evaluated"]
    matches = [result for _, result in valid if abs(math.expm1(result[0]["log_half_target_ratio"])) <= tolerance]
    for (left, ltrial), (right, rtrial) in zip(grid, grid[1:]):
        if ltrial[0]["status"] != "evaluated" or rtrial[0]["status"] != "evaluated":
            continue
        lv, rv = ltrial[0]["log_half_target_ratio"], rtrial[0]["log_half_target_ratio"]
        if lv * rv >= 0:
            continue
        try:
            def objective(value):
                row = trial(value)[0]
                if row["status"] != "evaluated":
                    raise ValueError("Intermediate PS trial is not feasible")
                return row["log_half_target_ratio"]
            solution = brentq(objective, left, right, xtol=1e-8, rtol=1e-10, maxiter=max_root_iterations)
            result = trial(solution)
            if abs(math.expm1(result[0]["log_half_target_ratio"])) <= tolerance:
                matches.append(result)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            audit.setdefault("root_search_failures", []).append(str(exc))
    if valid:
        audit["observed_grid_half_life_range_h"] = [min(x[1][0]["eigen_half_life_h"] for x in valid),
                                                   max(x[1][0]["eigen_half_life_h"] for x in valid)]
    if matches:
        selected = min(matches, key=lambda x: (abs(x[0]["log_ps"] - math.log(initial_ps)), x[0]["log_ps"]))
        audit["half_constraint_status"] = "matched"
    else:
        audit["half_constraint_status"] = "unreachable_within_declared_search"
        audit["reasons"].append("half_life_not_matched_within_declared_PS_bounds_and_feasible_Vss")
        if not fallback:
            return failed("half_life_unreachable")
        selected = trial(math.log(initial_ps))
        if selected[1] is None or selected[1]["status"] != "anchored":
            audit["half_constraint_status"] = "blocked"
            audit["reasons"].append(selected[0].get("reason", "fixed_PS_fallback_infeasible"))
            return failed("fixed_PS_fallback_infeasible")
        audit.update(fallback_used=True, fallback_reason="Retain original PS; predicted half-life is not a compulsory measured constraint")
        if selected[2] is None:
            audit["half_constraint_status"] = "unassessed_modal_evaluation"
            audit["fallback_reason"] = "Retain original feasible Vss model; asymptotic modal calculation was numerically unresolved"
    selected_trial, anchor, mode = selected
    audit.update(chosen_ps=selected_trial["ps_multiplier"], chosen_partition_scale=anchor["partition_scale"],
                 achieved_eigen_half_life_h=mode["half_life_h"] if mode else None, selected_observable_mode=mode,
                 relative_eigen_half_target_error=abs(mode["half_life_h"] / half["value_h"] - 1.) if mode else None,
                 chosen_on_boundary=any(math.isclose(selected_trial["ps_multiplier"], bound, rel_tol=1e-7) for bound in (low, high)),
                 selected_capacity_mapping=anchor["capacity_mapping"])
    try:
        prediction = capacity.simulate_design(physiology, request, log_parameters=anchor["log_parameters"],
                                              times_h=times_h, observation=observation)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        audit["reasons"].append(str(exc))
        return failed("selected_model_simulation_failed")
    audit["linear_trajectory_calls"] = 1
    nca_half = prediction["endpoints"].get("terminal_half_life", {}).get("value")
    audit.update(achieved_nca_half_life_h=nca_half, nca_half_target_fold_error=(max(nca_half / half["value_h"], half["value_h"] / nca_half)
        if nca_half is not None and nca_half > 0 else None),
        nca_half_matches_input_at_declared_tolerance=(abs(nca_half / half["value_h"] - 1.) <= tolerance
            if nca_half is not None and nca_half > 0 else None),
        relative_nca_to_eigen_half_difference=(abs(nca_half / mode["half_life_h"] - 1.)
            if nca_half is not None and nca_half > 0 and mode else None),
        achieved_model_vss_l=prediction["exact_moments"]["vss_l"],
        achieved_nca_vss_l=prediction["endpoints"].get("vss", {}).get("value"),
        nca_half_life_not_forced_to_input=True)
    prediction.update(version=VERSION, learned_exchange_audit=audit,
                      half_constraint_status=audit["half_constraint_status"], decision=anchor)
    return prediction
