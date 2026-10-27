"""A shared, unmeasured RBC/tissue partition scale constrained by input Vss.

Anatomical volumes, flows, fu and clearance inputs stay fixed. One positive
alpha scales every non-plasma native Kp, including RBC Kp; B/P is then derived
from venous hematocrit. This is a new reduction hypothesis, not a measured RBC
correction or a clinical-curve fit. The frozen sampled-NCA PS search is reused.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq

from . import capacity_guided_design as capacity
from . import learned_exchange_design as learned
from . import permeability_model as exchange
from .schmitt_physiology import _clone, _digest

VERSION = "joint-partition-design-1"
ALPHA_BOUNDS = (1e-12, 1e12)
BLOOD_PARAMETERS = {"blood_to_plasma_ratio", "blood_plasma_ratio", "rbc_kp_plasma",
                    "rbc_partition_coefficient", "blood_cell_partition_coefficient"}
PROTECTED = ["anatomical_plasma_and_RBC_volumes", "anatomical_tissue_volumes", "blood_flows",
             "relative_nonplasma_partition_coefficients", "fraction_unbound", "clearance_scale",
             "qualified_renal_clearance", "native_readback_and_PKML"]


def _number(value, name, *, positive=False, nonnegative=False):
    return exchange.perfusion._number(value, name, positive=positive, nonnegative=nonnegative)


def _close(left, right, name):
    if not math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("Inconsistent exported compartment aggregate: " + name)


def transform_physiology(physiology, alpha):
    """Return a separate derived physiology; validate all native aggregates."""
    alpha = _number(alpha, "joint partition alpha", positive=True)
    if physiology.get("version") != exchange.PHYSIOLOGY_VERSION:
        raise ValueError("Unsupported native physiology schema")
    if physiology.get("joint_partition_mapping"):
        raise ValueError("Joint partition mapping cannot be applied twice")
    original_rbc = _number(physiology["rbc_kp_plasma"], "native RBC Kp", positive=True)
    original_bp = _number(physiology["blood_to_plasma_ratio"], "native B/P", positive=True)
    result = deepcopy(physiology)
    changes = []
    for aggregate in [*result["organs"], *result["blood"].values()]:
        components = aggregate.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError("Joint scaling requires explicit native compartment components")
        paths = [row["path"] for row in components]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate native compartment path")
        volumes = {kind: 0. for kind in ("Plasma", "BloodCells", "Interstitial", "Intracellular")}
        capacities = dict.fromkeys(volumes, 0.)
        for row in components:
            kind = row["kind"]
            if kind not in volumes:
                raise ValueError("Unsupported native compartment kind")
            volume = _number(row["volume_l"], "native volume", nonnegative=True)
            kp = _number(row["kp_plasma"], "native partition coefficient", positive=True)
            _close(_number(row["capacity_l"], "native capacity", nonnegative=True), volume * kp, row["path"])
            if kind == "Plasma":
                _close(kp, 1., "plasma Kp")
            elif kind == "BloodCells":
                _close(kp, original_rbc, "RBC Kp")
            volumes[kind] += volume
            capacities[kind] += volume * kp
        for field, expected in (
            ("volume_l", sum(volumes.values())), ("plasma_volume_l", volumes["Plasma"]),
            ("rbc_volume_l", volumes["BloodCells"]),
            ("interstitial_volume_l", volumes["Interstitial"]), ("intracellular_volume_l", volumes["Intracellular"]),
            ("vascular_volume_l", volumes["Plasma"] + volumes["BloodCells"]),
            ("extravascular_volume_l", volumes["Interstitial"] + volumes["Intracellular"]),
            ("vascular_capacity_l", capacities["Plasma"] + capacities["BloodCells"]),
            ("extravascular_capacity_l", capacities["Interstitial"] + capacities["Intracellular"]),
        ):
            _close(_number(aggregate[field], field, nonnegative=True), expected, aggregate["name"] + "." + field)
        for row in components:
            if row["kind"] == "Plasma":
                continue
            before = row["kp_plasma"]
            after = _number(before * alpha, "derived partition coefficient", positive=True)
            row["kp_plasma"] = after
            row["capacity_l"] = _number(row["volume_l"] * after, "derived capacity", nonnegative=True)
            changes.append({"path": row["path"], "kind": row["kind"],
                            "native_kp_plasma": before, "derived_kp_plasma": after})
        aggregate["vascular_capacity_l"] = sum(r["capacity_l"] for r in components if r["kind"] in {"Plasma", "BloodCells"})
        aggregate["extravascular_capacity_l"] = sum(r["capacity_l"] for r in components if r["kind"] in {"Interstitial", "Intracellular"})
        ev = aggregate["extravascular_volume_l"]
        aggregate["kp_extravascular_plasma"] = aggregate["extravascular_capacity_l"] / ev if ev else None
        aggregate["kp_total_plasma"] = (aggregate["vascular_capacity_l"] + aggregate["extravascular_capacity_l"]) / aggregate["volume_l"]
    venous = result["blood"]["venous"]
    hct = venous["rbc_volume_l"] / venous["volume_l"]
    if not 0 < hct < 1:
        raise ValueError("Joint partition mapping requires positive venous plasma and RBC volumes")
    _close(original_bp, 1. - hct + hct * original_rbc, "native RBC/BP equilibrium")
    if "hematocrit" in result:
        _close(result["hematocrit"], hct, "native hematocrit")
    result["rbc_kp_plasma"] = original_rbc * alpha
    result["blood_to_plasma_ratio"] = _number(1. - hct + hct * result["rbc_kp_plasma"], "derived B/P", positive=True)
    result["equilibrium_blood_to_plasma_ratio_check"] = result["blood_to_plasma_ratio"]
    result["mapping_version"] = VERSION
    result["joint_partition_mapping"] = {
        "version": VERSION, "alpha": alpha, "basis": "derived_not_measured",
        "formula": "K_nonplasma=alpha*K_native; BP=(1-Hct_venous)+Hct_venous*alpha*K_RBC_native",
        "native_rbc_kp_plasma": original_rbc, "derived_rbc_kp_plasma": result["rbc_kp_plasma"],
        "native_blood_to_plasma_ratio": original_bp, "derived_blood_to_plasma_ratio": result["blood_to_plasma_ratio"],
        "hematocrit": hct, "modified_partitions": changes, "protected": PROTECTED,
        "native_physiology_sha256": _digest(physiology), "native_model_modified": False,
        "clinical_observations_used": False, "clipping_applied": False,
        "interpretation": "One uncertain native binding-capacity scale, not known true RBC/tissue binding"}
    return result


def build_system(physiology, *, total_clearance_l_h, renal_clearance_l_h=None,
                 family="exchange", log_parameters=None, observation="auto"):
    logs = capacity._logs(log_parameters)
    try:
        alpha = _number(math.exp(logs["partition_scale"]), "joint partition alpha", positive=True)
    except OverflowError as error:
        raise ValueError("Joint partition scale overflow") from error
    derived = transform_physiology(physiology, alpha)
    system = exchange.build_system(derived, total_clearance_l_h=total_clearance_l_h,
        renal_clearance_l_h=renal_clearance_l_h, family=family,
        log_parameters={**logs, "partition_scale": 0.}, observation=observation)
    metadata = deepcopy(system.metadata)
    metadata.update(version=VERSION, joint_partition_mapping=derived["joint_partition_mapping"])
    metadata["parameters"]["partition_scale"] = alpha
    metadata["log_parameters"] = logs
    metadata["grouping_interpretation"] = "one_common_RBC_and_extravascular_partition_scale"
    return replace(system, metadata=metadata)


def _system(physiology, total, renal, logs, observation):
    return build_system(physiology, total_clearance_l_h=total, renal_clearance_l_h=renal,
                        log_parameters=logs, observation=observation)


def exact_moments(physiology, *, total_clearance_l_h, renal_clearance_l_h=None,
                  log_parameters=None, observation="auto"):
    return capacity._moment_result(build_system(physiology, total_clearance_l_h=total_clearance_l_h,
        renal_clearance_l_h=renal_clearance_l_h, log_parameters=log_parameters, observation=observation))


def simulate(physiology, **kwargs):
    return _clone(exchange.simulate, build_system=build_system, VERSION=VERSION)(physiology, **kwargs)


def simulate_design(physiology, request, **kwargs):
    model = SimpleNamespace(simulate=simulate, build_system=build_system)
    return _clone(capacity.simulate_design, model=model, exact_moments=exact_moments,
                  VERSION=VERSION)(physiology, request, **kwargs)


def anchor_partition(physiology, request, vss_evidence, *, log_parameters=None, observation="auto"):
    """Numerically solve the requested observer's Vss, not summed capacity."""
    selected, diagnostics, reasons = capacity._select_evidence(request, vss_evidence)
    logs = capacity._logs(log_parameters)
    result = {"version": VERSION, "status": "blocked" if reasons else "retain_unanchored",
        "selected_vss_evidence": deepcopy(selected["record"]) if selected else None,
        "vss_basis": selected["record"]["basis"] if selected else None,
        "target_vss_l": selected["target_vss_l"] if selected else None,
        "log_parameters": deepcopy(logs), "unanchored_log_parameters": deepcopy(logs),
        "partition_scale": math.exp(logs["partition_scale"]), "baseline_moments": None,
        "anchored_moments": None, "capacity_mapping": {}, "reasons": list(reasons),
        "evidence_diagnostics": diagnostics, "curve_fitting_performed": False,
        "source_verification": "caller_must_verify_hashes_extraction_and_independence",
        "protected": PROTECTED, "clinical_validation": False}
    if reasons or selected is None:
        return result
    clearance = capacity.clearance_inputs(request)
    target = selected["target_vss_l"]
    evaluations = {}
    def evaluate(log_alpha):
        key = float(log_alpha)
        if key not in evaluations:
            moments = exact_moments(physiology, total_clearance_l_h=clearance["total_clearance_l_h"],
                renal_clearance_l_h=clearance["renal_clearance_l_h"],
                log_parameters={**logs, "partition_scale": key}, observation=observation)
            evaluations[key] = moments
        return evaluations[key]["vss_l"] - target
    mapping = {"version": VERSION, "rule": "exact_observer_Vss_root_of_joint_RBC_and_tissue_scale",
               "alpha_numerical_bracket_limits": list(ALPHA_BOUNDS), "clipping_applied": False,
               "sum_capacity_not_substituted_for_requested_observer": True}
    result["capacity_mapping"] = mapping
    try:
        center = evaluate(0.)
        result["baseline_moments"] = evaluations[0.]
        left = right = 0.
        if not math.isclose(center + target, target, rel_tol=1e-10, abs_tol=1e-12):
            direction = -1. if center > 0 else 1.
            limit = math.log(ALPHA_BOUNDS[0] if direction < 0 else ALPHA_BOUNDS[1])
            while True:
                edge = max(limit, left - math.log(10.)) if direction < 0 else min(limit, right + math.log(10.))
                value = evaluate(edge)
                if direction < 0:
                    left = edge
                else:
                    right = edge
                if value == 0 or value * center < 0:
                    break
                if edge == limit:
                    result.update(status="blocked", reasons=["no_positive_joint_partition_root_in_numerical_bracket"])
                    return result
            root = brentq(evaluate, left, right, xtol=1e-11, rtol=1e-12, maxiter=100)
        else:
            root = 0.
        evaluate(root)
        moments = evaluations[root]
        if not math.isclose(moments["vss_l"], target, rel_tol=1e-8, abs_tol=1e-10):
            raise ArithmeticError("Joint partition root does not reproduce observer Vss")
        alpha = math.exp(root)
        mapping.update(selected_alpha=alpha, derived_mapping=transform_physiology(physiology, alpha)["joint_partition_mapping"])
        result.update(status="anchored", partition_scale=alpha,
                      log_parameters={**logs, "partition_scale": root}, anchored_moments=moments)
        return result
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        result.update(status="blocked", reasons=["joint_partition_capacity_root_numerically_unresolved"],
                      numerical_reason=str(error))
        return result
    finally:
        mapping["root_evaluations"] = len(evaluations)
        mapping["root_trials"] = [{"alpha": math.exp(k), "observer_vss_l": v["vss_l"]}
                                  for k, v in sorted(evaluations.items())]


def _blood_evidence(physiology, request):
    qualified, issues = [], []
    forbidden = request.get("forbidden_sources", [])
    paths, hashes = {r.get("path") for r in forbidden}, {r.get("sha256") for r in forbidden}
    for row in request.get("evidence", []):
        if row.get("parameter") not in BLOOD_PARAMETERS or row.get("basis") not in {"measured", "derived"}:
            continue
        context, source = row.get("context", {}), row.get("source", {})
        valid = (row.get("compound_id") == request["compound_id"] and row.get("unit") in {"fraction", "1"}
            and context.get("species") == "human" and context.get("independent_of_target") is True
            and context.get("study_role") in {"independent_parameters", "independent_measurement"}
            and bool(source.get("path")) and bool(source.get("location"))
            and isinstance(source.get("sha256"), str) and len(source["sha256"]) == 64
            and source["path"] not in paths and source["sha256"] not in hashes)
        try:
            actual = _number(row["value"], "independent blood partition", positive=True)
            key = "blood_to_plasma_ratio" if row["parameter"] in {"blood_to_plasma_ratio", "blood_plasma_ratio"} else "rbc_kp_plasma"
            valid = valid and math.isclose(actual, physiology[key], rel_tol=1e-6, abs_tol=1e-10)
        except (KeyError, ValueError):
            valid = False
        if valid:
            qualified.append(deepcopy(row))
        else:
            issues.append("invalid_or_unapplied_independent_blood_partition:" + str(row.get("parameter")))
    return qualified, issues


def simulate_joint_design(physiology, request, vss_evidence, half_life_evidence, *,
                         log_parameters=None, ps_bounds=(.01, 100.), observation="auto",
                         times_h=None, fallback=True, grid_points=17, max_root_iterations=60,
                         half_relative_tolerance=1e-5, residue_tolerance=1e-12,
                         target_basis="sampled_nca", max_trajectory_calls=33):
    """Same PS search, new global capacity mapping; no new native or ML calls."""
    if target_basis != "sampled_nca":
        raise ValueError("Joint partition design uses sampled_nca only")
    measurements, issues = _blood_evidence(physiology, request)
    audit = {"version": VERSION, "mode": "retained_independent_blood_partition" if measurements else "joint_RBC_tissue_scale",
        "independent_blood_partition_evidence": measurements, "protected": PROTECTED,
        "native_physiology_sha256": _digest(physiology), "target_observations_used": False,
        "clinical_curve_fitting": False, "new_native_calls": 0, "new_learning_calls": 0,
        "source_verification": "caller_must_verify_source_hashes_and_measurement_provenance",
        "limitations": ["Shared RBC/tissue scale is a construction hypothesis, not measured blood binding",
            "Matching predicted half-life does not correct its prediction error",
            "Positive numerical root limits do not guarantee all Vss inputs are physiologically feasible"]}
    if issues:
        return {"version": VERSION, "status": "failed", "reason": "invalid_independent_blood_partition",
                "endpoints": {}, "joint_partition_audit": {**audit, "reasons": issues},
                "learned_exchange_audit": {"reasons": issues, "search_attempts": 0, "linear_trajectory_calls": 0},
                "clinical_validation": False, "curve_fitting_performed": False}
    kwargs = dict(log_parameters=log_parameters, ps_bounds=ps_bounds, observation=observation,
        times_h=times_h, fallback=fallback, grid_points=grid_points, max_root_iterations=max_root_iterations,
        half_relative_tolerance=half_relative_tolerance, residue_tolerance=residue_tolerance,
        target_basis=target_basis, max_trajectory_calls=max_trajectory_calls)
    if measurements:
        result = learned.simulate_learned_design(physiology, request, vss_evidence, half_life_evidence, **kwargs)
        audit["retained_algorithm_version"] = result.get("version", learned.VERSION)
        result["version"] = VERSION
        result["joint_partition_audit"] = audit
        return result
    model = SimpleNamespace(build_system=build_system)
    adapted_capacity = SimpleNamespace(**{name: getattr(capacity, name) for name in
        ("_logs", "_select_evidence", "clearance_inputs", "prediction_grid")},
        anchor_partition=anchor_partition, simulate_design=simulate_design)
    search = _clone(learned._sampled_nca_search, capacity=adapted_capacity, exchange=model, VERSION=VERSION)
    runner = _clone(learned.simulate_learned_design, capacity=adapted_capacity, exchange=model,
                    _sampled_nca_search=search, VERSION=VERSION)
    result = runner(physiology, request, vss_evidence, half_life_evidence, **kwargs)
    design_audit = result["learned_exchange_audit"]
    design_audit["protected"] = PROTECTED
    design_audit["limitations"] = [x for x in design_audit.get("limitations", [])
                                    if x != "RBC_partition_conflicts_are_not_silently_overridden"]
    design_audit["limitations"].append("RBC_and_tissue_partitions_are_jointly_derived_from_Vss_not_independently_measured")
    audit.update(selected_alpha=design_audit.get("chosen_partition_scale"),
                 selected_capacity_mapping=design_audit.get("selected_capacity_mapping"),
                 search_attempts=design_audit.get("search_attempts", 0),
                 linear_trajectory_calls=design_audit.get("linear_trajectory_calls", 0))
    result["joint_partition_audit"] = audit
    return result
