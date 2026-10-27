"""Evidence-qualified tissue-capacity anchoring for the frozen exchange model.

No clinical curve, evaluation endpoint, file, native process or optimizer is
read here. One common extravascular multiplier preserves vascular capacities,
relative tissue capacities, blood flows and the specified exchange rule.

The observation-specific moment Vss is not assumed to equal total anatomical
volume or even total plasma-equivalent capacity. For the transient amount
matrix A, input b and plasma observer c, m0=c'(-A)^-1 b and
m1=c'(-A)^-2 b. Vss=m1/m0**2, CL=1/m0 and MRT=m1/m0. These infinite-horizon
model quantities are reported separately from the original sampled-curve NCA
endpoints; no exact moment fills a failed NCA result.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, solve

from . import permeability_model as model
from .perfusion_calibration import clearance_inputs, prediction_grid
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction

VERSION = "capacity-guided-design-1"
PARAMETERS = model.PARAMETERS["exchange"]
SCALAR_FIELDS = {
    "auc": ("value_mg_h_per_l", "mg*h/l"), "cmax": ("value_mg_per_l", "mg/l"),
    "tmax": ("value_h", "h"), "c_last": ("value_mg_per_l", "mg/l"),
    "terminal_half_life": ("value_h", "h"), "auc_inf": ("value_mg_h_per_l", "mg*h/l"),
    "clearance": ("value_l_per_h", "l/h"), "vz": ("value_l", "l"),
    "mrt": ("value_h", "h"), "vss": ("value_l", "l"),
}


def _number(value, name, *, positive=False):
    return model.perfusion._number(value, name, positive=positive)


def _logs(values):
    if values is None:
        values = {}
    if not isinstance(values, Mapping) or set(values) - set(PARAMETERS):
        raise ValueError("Expected the three declared CL/K/PS log multipliers")
    return {name: _number(values.get(name, 0.), name) for name in PARAMETERS}


def _solve(matrix, rhs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            result = solve(matrix, rhs, assume_a="gen", check_finite=True)
        except (LinAlgWarning, np.linalg.LinAlgError, ValueError) as error:
            raise ArithmeticError("Model moments are numerically unresolved") from error
    denominator = np.linalg.norm(matrix, np.inf) * np.linalg.norm(result, np.inf) + np.linalg.norm(rhs, np.inf)
    residual = float(np.linalg.norm(matrix @ result - rhs, np.inf) / denominator) if denominator else math.inf
    if not np.isfinite(result).all() or residual > 1e-10:
        raise ArithmeticError("Model moment solve failed its backward-residual check")
    return result, residual


def _system(physiology, total, renal, logs, observation):
    _number(total, "total_clearance_l_h", positive=True)
    return model.build_system(physiology, total_clearance_l_h=total,
        renal_clearance_l_h=renal, family="exchange", log_parameters=logs, observation=observation)


def _moment_result(system):
    if system.state_names[-1] != "Eliminated":
        raise ValueError("Expected the frozen model's final cumulative elimination state")
    transient = -system.matrix[:-1, :-1]
    input_vector = system.input_vector[:-1]
    integral, residual0 = _solve(transient, input_vector)
    first, residual1 = _solve(transient, integral)
    if np.any(integral < 0) or np.any(first < 0):
        raise ArithmeticError("Positive model has nonpositive moment solutions")

    def quantities(observer):
        m0 = _number(float(observer[:-1] @ integral), "zeroth concentration moment", positive=True)
        m1 = _number(float(observer[:-1] @ first), "first concentration moment", positive=True)
        return {"auc_per_unit_dose_h_l": m0, "aumc_per_unit_bolus_h2_l": m1,
                "clearance_l_h": _number(1. / m0, "moment clearance", positive=True),
                "mrt_h": _number(m1 / m0, "moment MRT", positive=True),
                "vss_l": _number((m1 / m0) / m0, "moment Vss", positive=True)}

    observed = quantities(system.observation_vector)
    central = quantities(system.venous_plasma_vector)
    bp = system.metadata["blood_to_plasma_ratio"]
    capacities = system.blood_equivalent_capacities_l * bp
    vascular = sum(float(value) for name, value in zip(system.state_names[:-1], capacities)
                   if not name.endswith("|Extravascular"))
    equilibrium = float(capacities.sum())
    effective = system.metadata["effective_total_clearance_l_h"]
    if (not math.isclose(central["vss_l"], equilibrium, rel_tol=1e-8)
            or not math.isclose(central["clearance_l_h"], effective, rel_tol=1e-8)
            or not math.isclose(observed["clearance_l_h"], effective, rel_tol=1e-8)):
        raise ArithmeticError("Moment/clearance/capacity identity failed for the specified model")
    return {"version": VERSION, "method": "infinite_horizon_linear_system_moments", **observed,
            "observation": system.metadata["observation"],
            "observation_weights": deepcopy(system.metadata["observation_weights"]),
            "central_venous": central, "equilibrium_plasma_capacity_l": equilibrium,
            "immutable_vascular_plasma_capacity_l": vascular,
            "observer_vss_offset_from_central_l": observed["vss_l"] - central["vss_l"],
            "effective_total_clearance_l_h": effective,
            "clearance_components_l_h": deepcopy(system.metadata["clearance_components_l_h"]),
            "blood_to_plasma_ratio": bp, "log_parameters": deepcopy(system.metadata["log_parameters"]),
            "parameters": deepcopy(system.metadata["parameters"]),
            "solve_backward_residuals": [residual0, residual1],
            "dose_and_infusion_invariant": True,
            "interpretation": "Model moments for the declared plasma observer; not measured anatomical volumes or quality-gated NCA",
            "clinical_validation": False}


def exact_moments(physiology, *, total_clearance_l_h, renal_clearance_l_h=None,
                  log_parameters=None, observation="auto") -> dict:
    """Return exact model CL/MRT/Vss without samples, fitting or propagation."""
    system = _system(physiology, total_clearance_l_h, renal_clearance_l_h,
                     _logs(log_parameters), observation)
    return _moment_result(system)


def _partition_affine(physiology, total, renal, logs, observation):
    """Obtain the K->0 limit without constructing a zero-capacity compartment.

    A=-F D(K)^-1, D(K)=D_vascular+K D_tissue. F has fixed conductances.
    The observer has only vascular entries, so c'D(K) is constant. Its first
    concentration moment and hence Vss are affine in K. Zero tissue capacity
    is used only in this limiting algebra, never as an executable model.
    """
    reference = _system(physiology, total, renal, {**logs, "partition_scale": 0.}, observation)
    capacities = reference.blood_equivalent_capacities_l
    conductance = -reference.matrix[:-1, :-1] * capacities[np.newaxis, :]
    state_concentration, r0 = _solve(conductance, reference.input_vector[:-1])
    observer = reference.observation_vector[:-1] * capacities
    tissue = np.array([name.endswith("|Extravascular") for name in reference.state_names[:-1]])
    if np.any(reference.observation_vector[:-1][tissue] != 0):
        raise ValueError("Capacity anchoring requires the frozen vascular plasma observer")
    zero = _number(float(observer @ state_concentration), "zeroth concentration moment", positive=True)
    fixed_moment, r1 = _solve(conductance, np.where(tissue, 0., capacities) * state_concentration)
    tissue_moment, r2 = _solve(conductance, np.where(tissue, capacities, 0.) * state_concentration)
    intercept = _number(float(observer @ fixed_moment) / zero**2, "observer vascular Vss floor", positive=True)
    slope = _number(float(observer @ tissue_moment) / zero**2, "Vss tissue-scale coefficient", positive=True)
    return {"observer_vss_floor_l": intercept, "vss_per_partition_scale_l": slope,
            "immutable_vascular_plasma_capacity_l": float(capacities[~tissue].sum()) * reference.metadata["blood_to_plasma_ratio"],
            "native_tissue_plasma_capacity_l": float(capacities[tissue].sum()) * reference.metadata["blood_to_plasma_ratio"],
            "formula": "Vss_observer(K)=observer_vss_floor_l+K*vss_per_partition_scale_l",
            "floor_is_zero_tissue_capacity_limit_not_an_executable_K_zero_model": True,
            "solve_backward_residuals": [r0, r1, r2]}


def _select_evidence(request, records):
    if records is None:
        records = []
    if not isinstance(records, list):
        raise ValueError("vss_evidence must be a sidecar list")
    forbidden = request.get("forbidden_sources", [])
    forbidden_paths = {row.get("path") for row in forbidden}
    forbidden_hashes = {row.get("sha256") for row in forbidden}
    accepted, diagnostics, fatal = [], [], []
    allowed = {"parameter", "compound_id", "value", "unit", "basis", "source", "context", "prediction_method", "notes"}
    for index, record in enumerate(records):
        issue = None
        if not isinstance(record, dict) or set(record) - allowed:
            issue = "invalid_vss_evidence_schema"
        else:
            source, context = record.get("source", {}), record.get("context", {})
            if not isinstance(source, dict) or not isinstance(context, dict):
                issue = "invalid_vss_source_or_context"
            elif (source.get("path") in forbidden_paths or source.get("sha256") in forbidden_hashes
                    or context.get("study_role") in {"target", "evaluation", "test", "validation", "heldout"}
                    or context.get("independent_of_target") is False):
                issue = "held_out_or_nonindependent_vss_evidence"
            elif record.get("compound_id") != request["compound_id"] or record.get("parameter") != "vss":
                issue = "vss_compound_or_parameter_mismatch"
            elif (not all(isinstance(source.get(key), str) and source[key].strip() for key in ("path", "location"))
                    or not isinstance(source.get("sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None):
                issue = "unbound_vss_source"
        if issue:
            fatal.append(issue); diagnostics.append({"index": index, "status": "rejected", "reason": issue})
            continue
        context = record.get("context", {})
        basis = record.get("basis")
        if (context.get("species") != "human" or context.get("matrix") != "plasma"
                or context.get("route") not in {"iv", "intravenous"}
                or context.get("independent_of_target") is not True
                or context.get("study_role") not in {"independent_parameters", "independent_evidence", "construction_prior"}
                or basis not in {"measured", "derived", "predicted"}):
            diagnostics.append({"index": index, "status": "unqualified", "reason": "unknown_or_unsupported_vss_context"})
            continue
        if basis == "predicted":
            method = record.get("prediction_method") or {}
            applicability = method.get("applicability") if isinstance(method, dict) else None
            if (not isinstance(applicability, dict) or not method.get("name") or not method.get("version")
                    or method.get("training_overlap") is not False
                    or applicability.get("target_structure_and_scaffold_excluded") is not True
                    or applicability.get("human_IV_parameter_model") is not True):
                diagnostics.append({"index": index, "status": "unqualified", "reason": "unqualified_structure_vss_prediction"})
                continue
        try:
            value = _number(record.get("value"), "Vss evidence", positive=True)
            unit = record.get("unit", "").lower()
            if unit not in {"l", "l/kg"}:
                raise ValueError("Vss evidence must use l or l/kg")
            target = value * (_number(request["human"]["weight_kg"], "body weight", positive=True) if unit == "l/kg" else 1.)
            target = _number(target, "converted Vss target", positive=True)
        except (ValueError, TypeError, AttributeError, OverflowError):
            fatal.append("invalid_vss_value_or_unit")
            diagnostics.append({"index": index, "status": "rejected", "reason": "invalid_vss_value_or_unit"})
            continue
        accepted.append({"record": deepcopy(record), "target_vss_l": target, "index": index,
                         "priority": 1 if basis == "predicted" else 0})
    if fatal:
        return None, diagnostics, list(dict.fromkeys(fatal))
    if not accepted:
        return None, diagnostics, []
    best = [row for row in accepted if row["priority"] == min(item["priority"] for item in accepted)]
    if any(not math.isclose(row["target_vss_l"], best[0]["target_vss_l"], rel_tol=1e-9) for row in best[1:]):
        return None, diagnostics, ["equal_priority_vss_conflict"]
    selected = best[0]
    for row in accepted:
        diagnostics.append({"index": row["index"], "status": "selected" if row is selected else "retained_not_selected",
                            "basis": row["record"]["basis"]})
    return selected, diagnostics, []


def anchor_partition(physiology, request, vss_evidence, *, log_parameters=None,
                     observation="auto") -> dict:
    """Analytically anchor tissue scale to independently qualified Vss evidence.

    Sources are validated structurally and compared with forbidden references;
    the caller must verify their bytes and their independent extraction. A
    predicted input remains predicted. Missing/unknown evidence retains the
    original model. Conflicts, forbidden inputs or an unattainable target block
    anchoring and retain the unanchored parameters for an explicit caller choice.
    No arbitrary scientific upper/lower clipping is imposed on a feasible K.
    """
    selected, evidence_diagnostics, reasons = _select_evidence(request, vss_evidence)
    logs = _logs(log_parameters)
    clearance = clearance_inputs(request)
    total, renal = clearance["total_clearance_l_h"], clearance["renal_clearance_l_h"]
    baseline = exact_moments(physiology, total_clearance_l_h=total, renal_clearance_l_h=renal,
                             log_parameters=logs, observation=observation)
    affine = _partition_affine(physiology, total, renal, logs, observation)
    result = {"version": VERSION, "status": "blocked" if reasons else "retain_unanchored",
        "selected_vss_evidence": deepcopy(selected["record"]) if selected else None,
        "vss_basis": selected["record"]["basis"] if selected else None,
        "target_vss_l": selected["target_vss_l"] if selected else None,
        "log_parameters": deepcopy(logs), "unanchored_log_parameters": deepcopy(logs),
        "partition_scale": math.exp(logs["partition_scale"]), "baseline_moments": baseline,
        "anchored_moments": None, "capacity_mapping": affine, "reasons": reasons,
        "evidence_diagnostics": evidence_diagnostics, "curve_fitting_performed": False,
        "source_verification": "caller_must_verify_hashes_extraction_and_independence",
        "protected": ["vascular_capacities", "blood_to_plasma_ratio", "blood_flow", "relative_extravascular_capacity_ratios",
                      "clearance_scale", "bidirectional_exchange_scale", "qualified_renal_clearance"],
        "clinical_validation": False,
        "interpretation": "Constraint to qualified Vss input, not measured tissue capacity or clinical validation"}
    if reasons or selected is None:
        return result
    target = selected["target_vss_l"]
    floor = affine["observer_vss_floor_l"]
    if target <= max(floor, affine["immutable_vascular_plasma_capacity_l"]):
        result.update(status="blocked", reasons=["vss_target_at_or_below_immutable_vascular_observer_floor"])
        return result
    scale = (target - floor) / affine["vss_per_partition_scale_l"]
    try:
        scale = _number(scale, "partition scale", positive=True)
        anchored_logs = {**logs, "partition_scale": math.log(scale)}
        anchored = exact_moments(physiology, total_clearance_l_h=total, renal_clearance_l_h=renal,
                                 log_parameters=anchored_logs, observation=observation)
        if not math.isclose(anchored["vss_l"], target, rel_tol=1e-8):
            raise ArithmeticError("Anchored model does not reproduce the Vss input")
    except (ValueError, ArithmeticError, OverflowError) as error:
        result.update(status="blocked", reasons=["numerically_unresolved_capacity_anchor"], numerical_reason=str(error))
        return result
    result.update(status="anchored", log_parameters=anchored_logs, partition_scale=scale,
                  anchored_moments=anchored, relative_target_error=abs(anchored["vss_l"] / target - 1.),
                  construction_target_role="predicted_prior" if result["vss_basis"] == "predicted" else "independent_parameter_evidence")
    return result


def simulate_design(physiology, request, *, log_parameters=None, times_h=None,
                    observation="auto") -> dict:
    """Run CL/K/PS on fixed physiology and retain the full sampled trajectory.

    ``endpoints`` retains the original NCA values/statuses as the primary
    comparison. ``exact_moments`` is a separately labelled model calculation.
    ``curve`` is the requested window; ``full_trajectory`` retains every sample
    used by NCA, including the tail. Compression/persistence belongs to callers.
    """
    if request["dosing"].get("route") != "intravenous" or request["dosing"].get("start_h", 0.) != 0:
        raise ValueError("The frozen exchange adapter requires single IV input at time zero")
    if request["dosing"].get("single_dose", True) is not True:
        raise ValueError("Repeated dosing is not supported")
    logs = _logs(log_parameters)
    clearance = clearance_inputs(request)
    dose = _number(request["dosing"]["dose_mg"], "dose_mg", positive=True)
    duration = _number(request["dosing"]["infusion_minutes"], "infusion_minutes") / 60.
    if duration < 0:
        raise ValueError("Infusion duration must be nonnegative")
    times = prediction_grid(request) if times_h is None else np.asarray(times_h)
    if (times.ndim != 1 or times.size < 2 or times.dtype.kind not in "iuf"
            or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("Full trajectory needs a strictly increasing finite grid starting at zero")
    prediction = model.simulate(physiology, times_h=times, dose_mg=dose, infusion_duration_h=duration,
        total_clearance_l_h=clearance["total_clearance_l_h"], renal_clearance_l_h=clearance["renal_clearance_l_h"],
        family="exchange", log_parameters=logs, observation=observation)
    mw = _number(next(row["value"] for row in request["evidence"] if row["parameter"] == "molecular_weight"),
                 "molecular weight", positive=True)
    molar = {"time_min": (times * 60.).tolist(), "concentration": (prediction["concentration_mg_l"] * 1000. / mw).tolist(),
             "time_unit": "min", "unit": "umol/l", "observation": prediction["observation"]}
    summary = summarize_extended_prediction(molar, request)
    endpoints = {}
    for name in request["target"]["endpoints"]:
        if name in SCALAR_FIELDS:
            field, unit = SCALAR_FIELDS[name]
            row = summary["endpoints"][name]
            endpoints[name] = {"value": row.get(field) if row["status"] == "completed" else None,
                               "unit": unit, "status": row["status"], "reason": row.get("reason")}
    start = _number(request["target"]["window_start_h"], "window_start_h")
    end = _number(request["target"]["window_end_h"], "window_end_h", positive=True)
    if start < 0 or end <= start or end > times[-1]:
        raise ValueError("The requested window must lie inside the complete trajectory")
    window = np.unique(np.r_[start, times[(times > start) & (times < end)], end])
    moments = exact_moments(physiology, total_clearance_l_h=clearance["total_clearance_l_h"],
        renal_clearance_l_h=clearance["renal_clearance_l_h"], log_parameters=logs, observation=observation)
    return {"version": VERSION, "status": "predicted", "log_parameters": logs,
        "parameters": deepcopy(prediction["parameters"]), "endpoints": endpoints,
        "endpoint_basis": "primary_original_quality_gated_sampled_curve_NCA",
        "curve": {"time_h": window.tolist(), "concentration_mg_l": np.interp(window, times, prediction["concentration_mg_l"]).tolist()},
        "full_trajectory": {"time_h": times.tolist(), "concentration_mg_l": prediction["concentration_mg_l"].tolist(),
                            "venous_plasma_mg_l": prediction["venous_plasma_mg_l"].tolist()},
        "nca_summary": summary, "terminal_diagnostics": deepcopy(summary["diagnostics"]),
        "exact_moments": moments, "exact_moments_used_to_fill_NCA": False,
        "effective_total_clearance_l_h": prediction["effective_total_clearance_l_h"],
        "mass_balance": deepcopy(prediction["mass_balance"]), "observer": prediction["observation"],
        "observation_weights": deepcopy(prediction["observation_weights"]),
        "clinical_validation": False, "curve_fitting_performed": False,
        "native_solver_calls": 0, "linear_solver_calls": 1}
