"""Bounded development-curve fitting with priors on construction inputs.

This is a small coordinate pattern search, not an official HJKB implementation.
Only development concentrations define its data loss. Native physical audits
remain mandatory; inability to estimate a full-tail PK moment is a diagnostic,
not failure of an otherwise valid development concentration prediction.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
import math
from numbers import Integral
from pathlib import Path

from .development_curve_blueprint import VERSION, _annotate, _run_development, _training_document
from .development_curve_optimizer import _number, _observations, _sample_curve
from .evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint
from .pk_anchored_blueprint import _drug_signature, mapped_snapshot
from .source_workflow import inside, save, sha256


ALGORITHM_VERSION = "curve-pattern-input-prior-1.0"
INITIAL_STEP = .25
MINIMUM_STEP = 1e-3
_LOG4 = math.log(4.)
_NAMES = ["effective_lipophilicity", "log_clearance_scale"]


class _BudgetExhausted(Exception):
    pass


def _coordinates(point, prior):
    if not isinstance(point, Mapping):
        raise ValueError("Warm start decision must be a mapping")
    lip = _number(point.get("lipophilicity"), "warm start lipophilicity")
    scale = _number(point.get("clearance_scale"), "warm start clearance_scale", positive=True)
    if not prior - 2.5 <= lip <= prior + 2.5 or not .25 <= scale <= 4.:
        raise ValueError("Warm start lies outside the fixed construction bounds")
    return ((lip - prior) / 2.5, math.log(scale) / _LOG4)


def _diagnostic_number(value, name):
    if value is None:
        return None, "not_estimable"
    try:
        return _number(value, name, positive=True), "available"
    except ValueError:
        return None, "invalid_derived_diagnostic"


def optimize_pattern_curve(
    evaluate: Callable, *, observations, logp_prior,
    target_clearance_l_h=None, max_evaluations=24, warm_start=None,
):
    """Search two normalized axes without derivatives or permanent freezing.

    The callback must return an audited ``status='predicted'`` native curve in
    min and umol/l. Optional derived CL/Vss never enter the objective. Every
    distinct callback, including failed attempts and re-evaluated warm starts,
    consumes the shared budget; cached physical coordinates are never rerun.
    """
    if not callable(evaluate):
        raise ValueError("evaluate must be callable")
    times, observed = _observations(observations)
    prior = _number(logp_prior, "logp_prior")
    if not prior - 2.5 < prior < prior + 2.5:
        raise ValueError("LogP bounds must be representable")
    target_cl = None if target_clearance_l_h is None else _number(
        target_clearance_l_h, "target_clearance_l_h", positive=True)
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    max_evaluations = int(max_evaluations)
    warm = None if warm_start is None else _coordinates(warm_start, prior)
    import numpy as np

    observed_logs = np.log(np.asarray(observed, dtype=float))
    records, cache, polls = [], {}, []
    cache_hits = 0
    step = INITIAL_STEP
    converged = False
    warm_evaluation = None

    def trial(z, purpose):
        nonlocal cache_hits
        z = tuple(float(value) for value in z)
        if len(z) != 2 or any(not math.isfinite(v) or not -1. <= v <= 1. for v in z):
            raise ValueError("Pattern coordinates must lie in [-1, 1]^2")
        lip, scale = prior + 2.5 * z[0], math.exp(_LOG4 * z[1])
        key = (lip, scale)
        if key in cache:
            cache_hits += 1
            return cache[key]
        if len(records) >= max_evaluations:
            raise _BudgetExhausted
        record = {"evaluation": len(records) + 1, "purpose": purpose,
                  "normalized_coordinates": list(z), "lipophilicity": lip,
                  "clearance_scale": scale, "status": "evaluation_failed",
                  "eligible_for_selection": False}
        records.append(record)
        cache[key] = record
        try:
            result = evaluate(lip, scale)
            if not isinstance(result, Mapping):
                raise ValueError("evaluate must return a mapping")
            record["callback_status"] = result.get("status") if isinstance(result.get("status"), str) else None
            if result.get("status") != "predicted":
                raise ValueError("Unsuccessful audited simulation: " + str(result.get("status"))
                                 + ": " + str(result.get("reason", "")))
            predicted = _sample_curve(result.get("curve"), times)
            errors = np.log(predicted) - observed_logs
            components = {"mean_squared_log_curve_error": float(np.mean(errors ** 2)),
                          "clearance_input_prior": .1 * (math.log(scale) / math.log(2.)) ** 2,
                          "lipophilicity_prior": .01 * z[0] ** 2}
            objective = math.fsum(components.values())
            if not math.isfinite(objective):
                raise ValueError("Nonfinite development objective")
            cl, cl_status = _diagnostic_number(result.get("clearance_l_h"), "derived clearance_l_h")
            vss, vss_status = _diagnostic_number(result.get("vss_l"), "derived vss_l")
            log_gmfe = float(np.mean(np.abs(errors)))
            # A finite log loss may exceed the floating-point range of exp.
            # Preserve the real loss without manufacturing a finite GMFE.
            gmfe = math.exp(log_gmfe) if log_gmfe < math.log(float(np.finfo(float).max)) else None
            record.update(status="ok", eligible_for_selection=True, objective=objective,
                objective_components=components, clearance_l_h=cl, vss_l=vss,
                derived_pk_diagnostics={"clearance_status": cl_status, "vss_status": vss_status,
                    "pk_anchor_estimation_basis": result.get("pk_anchor_estimation_basis"),
                    "endpoint_statuses": result.get("endpoint_statuses", {}),
                    "used_in_objective": False},
                predicted_at_observed_times=predicted.tolist(), sampled_log_ratios=errors.tolist(),
                curve_metrics={"point_count": len(times), "gmfe": gmfe,
                    "log_rmse": math.sqrt(components["mean_squared_log_curve_error"]),
                    "mean_log_ratio": float(np.mean(errors)),
                    "within_twofold_fraction": float(np.mean(np.abs(errors) <= math.log(2.)))})
        except Exception as exc:
            record.update(reason=type(exc).__name__ + ": " + str(exc))
        return record

    def best_record():
        return min((r for r in records if r["eligible_for_selection"]),
                   key=lambda r: (r["objective"], r["evaluation"]), default=None)

    status, reason = "budget_exhausted", "Actual simulation call budget exhausted"
    try:
        trial((0., 0.), "rf_prior_initial")
        if warm is not None:
            warm_evaluation = trial(warm, "warm_start_reassessment")["evaluation"]
        center = best_record()
        center_z = center["normalized_coordinates"] if center else [0., 0.]
        while True:
            anchor = best_record()
            anchor_value = anchor["objective"] if anchor else None
            poll = {"round": len(polls) + 1, "center": list(center_z), "step": step,
                    "anchor_evaluation": anchor["evaluation"] if anchor else None,
                    "complete": False, "probes": []}
            polls.append(poll)
            for axis in range(2):
                for sign in (-1, 1):
                    candidate = list(center_z)
                    candidate[axis] = max(-1., min(1., candidate[axis] + sign * step))
                    before = len(records)
                    candidate_result = trial(candidate, "pattern_poll")
                    poll["probes"].append({"axis": axis, "direction": sign,
                        "evaluation": candidate_result["evaluation"],
                        "cached": before == len(records), "status": candidate_result["status"]})
            poll["complete"] = True
            selected = best_record()
            improved = selected is not None and (anchor_value is None or selected["objective"] < anchor_value)
            poll["improved"] = improved
            if improved:
                center_z = selected["normalized_coordinates"]
            elif step <= MINIMUM_STEP:
                failures = any(p["status"] != "ok" for p in poll["probes"])
                converged = selected is not None and not failures
                status = "poll_tolerance_reached" if converged else "no_feasible_improvement"
                reason = ("Complete feasible poll found no improvement at the declared step tolerance"
                          if converged else "Final poll contains failed directions; stationarity is not established")
                break
            else:
                step /= 2.
    except _BudgetExhausted:
        pass

    selected = best_record()
    decision = None if selected is None else {key: deepcopy(selected[key]) for key in (
        "evaluation", "lipophilicity", "clearance_scale", "objective", "objective_components",
        "clearance_l_h", "vss_l", "derived_pk_diagnostics", "curve_metrics")}
    initial = records[0].get("objective")
    return {"algorithm_version": ALGORITHM_VERSION, "status": status, "reason": reason,
        "converged": converged, "decision": decision, "eligible": decision is not None,
        "evaluations": records, "evaluation_count": len(records), "max_evaluations": max_evaluations,
        "failed_evaluation_count": sum(r["status"] != "ok" for r in records), "cache_hits": cache_hits,
        "initial_objective": initial, "warm_start_evaluation": warm_evaluation,
        "improved_over_initial": bool(selected is not None and initial is not None and selected["objective"] < initial),
        "optimizer": {"method": "bounded_coordinate_pattern_search", "success": converged,
            "numerical_convergence_only": True, "active_parameters": _NAMES,
            "initial_step": INITIAL_STEP, "minimum_step": MINIMUM_STEP, "final_step": step,
            "poll_order": [[0, -1], [0, 1], [1, -1], [1, 1]], "polls": polls},
        "sensitivity": {"status": "not_used_for_freezing", "active_parameters": _NAMES,
            "frozen_parameters": [], "rank": None, "identifiability_established": False},
        "failure_recovery": {"version": ALGORITHM_VERSION, "enabled": True,
            "strategy": "Reject failed probes and continue remaining bounded poll directions"},
        "targets": {"clearance_l_h": target_cl, "vss_l": None, "logp_prior": prior},
        "bounds": {"lipophilicity": [prior - 2.5, prior + 2.5], "clearance_scale": [.25, 4.],
            "normalized": [[-1., 1.], [-1., 1.]], "interpretation": "Computational bounds, not confidence intervals"},
        "objective_spec": {"formula": "mean(ln(P/O)^2)+0.1*(ln(clearance_scale)/ln(2))^2+0.01*((logMA-logP)/2.5)^2",
            "curve_weight": 1., "clearance_input_prior_weight": .1, "lipophilicity_prior_weight": .01,
            "derived_endpoint_residuals": False, "weights_are_fixed_engineering_regularization": True,
            "official_OSP_algorithm_reproduction": False,
            "sampling": "Linear interpolation at development observation times; extrapolation forbidden"},
        "interpretation": "Development fitting with construction-input regularization; derived output CL/Vss are diagnostics only"}


def _warm_start(source, request, training_source, root, logp):
    if source is None:
        return None
    if not isinstance(source, Mapping):
        raise ValueError("Warm start source must declare path and sha256")
    path = inside(source.get("path", ""), root, exists=True)
    if path.stat().st_size > 4 * 1024**2 or sha256(path) != source.get("sha256"):
        raise ValueError("Warm start source checksum or size is invalid")
    if any(path == inside(f["path"], root) or source["sha256"] == f["sha256"]
           for f in request.get("forbidden_sources", [])):
        raise ValueError("Comparison source cannot supply a warm start")
    mapping = json.loads(path.read_text())
    expected = {"compound_id": request["compound_id"], "drug_signature": _drug_signature(request),
        "construction_partition_method": request["partition_method"], "reference_human": request["human"],
        "reference_dosing": request["dosing"], "training_source": training_source,
        "target_vss_anchor": None, "target_vss_l": None}
    if (mapping.get("version") not in {"development-curve-pbpk-1.0", "development-curve-pbpk-1.1", VERSION}
            or mapping.get("eligible") is not True or not isinstance(mapping.get("decision"), Mapping)
            or any(key not in mapping or mapping[key] != value for key, value in expected.items())):
        raise ValueError("Warm start must be an eligible mapping for the identical drug, formula, human, dose and development source")
    _training_document(mapping["training_source"], request, root)
    _coordinates(mapping["decision"], logp)
    return {key: mapping["decision"][key] for key in ("lipophilicity", "clearance_scale")}


def fit_pattern_mapping(*, reference_request, training_source, output_dir,
                        project_root, max_evaluations=24, warm_start_source=None):
    """Create a new immutable mapping readable by the existing prediction API."""
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    if output.exists():
        raise FileExistsError("Existing fit output is immutable")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    req = EvidenceBlueprintRequest.model_validate(reference_request).model_dump(mode="json", exclude_none=True)
    if req["dosing"]["route"] != "intravenous":
        raise ValueError("Current pattern calibration requires single-dose IV")
    training_source, warm_start_source = deepcopy(training_source), deepcopy(warm_start_source)
    observed = _training_document(training_source, req, root)
    blueprint = assemble_evidence_blueprint(req, root)
    if blueprint["status"] != "assembled":
        raise ValueError("Required drug evidence is incomplete")
    allocation = blueprint["manifest"]["clearance_allocation"]
    target_cl = sum(allocation[k] for k in ("hepatic_ml_min_kg", "renal_ml_min_kg")) * req["human"]["weight_kg"] * .06
    logp = next(e["value"] for e in req["evidence"] if e["parameter"] == "logp")
    warm = _warm_start(warm_start_source, req, training_source, root, logp)
    output.mkdir(parents=True)
    save(output / "input.json", {"request": req, "training_source": training_source,
        "warm_start_source": warm_start_source, "target_clearance_l_h": target_cl,
        "target_vss_l": None, "algorithm_version": ALGORITHM_VERSION,
        "maximum_evaluations": int(max_evaluations)})
    calls = 0

    def evaluate(lipophilicity, clearance_scale):
        nonlocal calls
        calls += 1
        folder = output / f"attempt_{calls:02d}"
        folder.mkdir()
        try:
            native = mapped_snapshot(blueprint, lipophilicity, clearance_scale)
            result = _run_development(native, req, folder, root, audit=False, require_vss=False)
            _annotate(folder, training_source)
        except Exception as exc:
            result = {"status": "native_failed", "reason": type(exc).__name__ + ": " + str(exc)}
        save(folder / "result.json", result)
        save(output / "progress.json", {"status": "running", "native_evaluations": calls,
            "maximum_evaluations": int(max_evaluations), "latest": result})
        if result.get("status") == "predicted":
            result = {**result, "curve": json.loads((folder / "prediction_locked.json").read_text())}
        return result

    fit = optimize_pattern_curve(evaluate, observations=observed, logp_prior=logp,
        target_clearance_l_h=target_cl, max_evaluations=max_evaluations, warm_start=warm)
    _training_document(training_source, req, root)
    _warm_start(warm_start_source, req, training_source, root, logp)
    # Recheck all evidence source hashes after the bounded native calls.
    if assemble_evidence_blueprint(req, root)["status"] != "assembled":
        raise ValueError("Drug evidence changed during fitting")
    fit.update(version=VERSION, compound_id=req["compound_id"], drug_signature=_drug_signature(req),
        construction_partition_method=req["partition_method"], reference_human=req["human"],
        reference_dosing=req["dosing"], evidence_ledger=req["evidence_ledger"],
        target_vss_anchor=None, target_clearance_l_h=target_cl, target_vss_l=None,
        training_source=training_source, warm_start_source=warm_start_source,
        development_curve_used=True, comparison_curve_used=False, used_complete_drug_model=False,
        independent_validation=False,
        limitations=["Selected construction parameters are calibrated on an independent development study, not on comparison observations.",
            "Warm starts are re-evaluated using the new objective and consume the shared budget.",
            "Input-scale regularization does not constrain or replace the derived output clearance.",
            "Missing or unreliable full-tail PK moments remain unavailable diagnostics.",
            "A selected feasible point need not be numerically converged or clinically accurate.",
            "Engineering search bounds and regularization weights are not physiological uncertainty intervals.",
            "Local or global parameter identifiability is not established by this search."])
    save(output / "mapping_locked.json", fit)
    save(output / "progress.json", {"status": fit["status"], "native_evaluations": calls,
        "eligible": fit["eligible"], "converged": fit["converged"], "algorithm_version": ALGORITHM_VERSION})
    return fit
