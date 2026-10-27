"""One input-only fallback for an infeasible vascular distribution capacity.

The shared parameter predictor is already trained and frozen. Keep every usable
Standard blueprint exactly; only a proved Vss/vascular-capacity conflict may
try native Schmitt partitioning. No clinical curve, score or drug name chooses
the branch. A failed alternative remains a failure. This is a construction
policy, not a new fitted drug model or a guarantee of clinical accuracy.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from . import clinical_pbpk_comparison as original
from . import schmitt_physiology
from .learned_exchange_design import simulate_learned_design, VERSION as LEARNED_VERSION
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

VERSION = "capacity-qualified-schmitt-fallback-1"
FLOOR_CONFLICT = "vss_target_at_or_below_immutable_vascular_observer_floor"
CODE_PINS = {"src/pbpk_kb/blueprint_policy/schmitt_physiology.py": sha256(Path(schmitt_physiology.__file__))}
REFERENCES = (
    "https://doi.org/10.1016/j.tiv.2007.09.010",
    "https://docs.open-systems-pharmacology.org/working-with-pk-sim/pk-sim-documentation/pk-sim-compounds-definition-and-work-flow",
)


def eligible_capacity_failure(entry):
    """Only the complete model-only trial ledger can establish this conflict."""
    if entry.get("status") != "failed" or entry.get("reason") != "no_usable_sampled_NCA_model":
        return False
    audit = entry.get("design_audit") or {}
    trials = audit.get("trials") or []
    if (audit.get("version") != LEARNED_VERSION or audit.get("target_basis") != "sampled_nca"
            or audit.get("ps_bounds") != [.01, 100.] or audit.get("grid_points") != 17
            or audit.get("trajectory_budget") != 33 or audit.get("search_attempts") != 17
            or len(trials) != 17):
        return False
    expected = np.linspace(math.log(.01), math.log(100.), 17)
    expected[np.abs(expected) < 1e-12] = 0.
    try:
        actual = np.array(sorted(float(row["log_ps"]) for row in trials))
        if (not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=0., atol=1e-12)
                or any(not math.isclose(float(row["ps_multiplier"]), math.exp(float(row["log_ps"])),
                                        rel_tol=1e-12) for row in trials)):
            return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        audit.get("target_observations_used") is False
        and audit.get("reasons") == [FLOOR_CONFLICT]
        and audit.get("linear_trajectory_calls") == 0
        and all(row.get("status") == "failed"
                and row.get("trajectory_attempted") is False
                and row.get("reason") == FLOOR_CONFLICT for row in trials)
    )


def protected_blood_partition(request):
    """An independent B/P or RBC measurement must not be replaced by a formula."""
    names = {"blood_to_plasma_ratio", "blood_plasma_ratio", "blood_to_plasma",
             "rbc_kp_plasma", "rbc_partition_coefficient", "blood_cell_partition"}
    return [deepcopy(r) for r in request.get("evidence", [])
            if r.get("parameter") in names and r.get("basis") in {"measured", "derived"}]


def _complete(folder, root, result):
    save(folder / "result.json", result)
    return {**result, "result_ref": original._ref(folder / "result.json", root)}


def _check_source(request, vss, half, entry, root):
    original._check_request(request)
    original._check_sidecars(request, vss, root)
    original._check_sidecars(request, half, root)
    if (entry.get("request") != request
            or entry.get("target_observations_used") is not False
            or entry.get("status") not in {"predicted", "failed"}):
        raise ValueError("A locked source outcome for this exact request is required")
    if entry.get("status") == "predicted":
        if entry.get("endpoints", {}).get("observations_used") is not False:
            raise ValueError("Source endpoint estimation used observations")
        if not entry.get("curve"):
            raise ValueError("A successful source must contain its locked trajectory")


def run_design(request, vss_evidence, half_evidence, output_dir, project_root, *, source_entry):
    """Apply a fixed branch policy with zero training or clinical calibration.

    The caller verifies the full source-entry receipt and binds identical
    sidecars/core. All target observations must stay sealed until every
    prediction and failure in the experiment is locked.
    """
    root = Path(project_root).resolve(strict=True)
    folder = inside(output_dir, root)
    _check_source(request, vss_evidence, half_evidence, source_entry, root)
    if folder.exists():
        raise FileExistsError("Use the experiment runner to verify/resume existing results")
    audit = {
        "version": VERSION, "shared_predictor_retrained": False,
        "clinical_curve_fitting_performed": False, "target_observations_used": False,
        "selection_uses_clinical_scores": False, "formula_order": ["PKSimStandard", "Schmitt"],
        "trigger": FLOOR_CONFLICT, "source_status": source_entry["status"],
        "source_reason": source_entry.get("reason"), "fallback_attempted": False,
        "references": list(REFERENCES), "native_timeout_seconds": 180,
        "native_threads": 2, "maximum_alternative_native_calls": 1,
        "maximum_alternative_linear_trajectories": 33,
        "protected": ["shared_parameter_weights", "CL", "fu", "Vss", "half_life_input",
                      "renal_clearance", "human_anatomy", "blood_flow", "dose", "requested_window"],
        "limitations": ["Mechanistic alternative, not proof that Schmitt is more accurate",
                        "A predicted Vss or half-life remains an uncertain predicted input",
                        "Only construction feasibility selects the branch; recovery is not accuracy improvement",
                        "Schmitt equilibrium blood ratio is derived for the reduced framework; raw native readbacks remain unchanged"],
    }
    measured_blood = protected_blood_partition(request)
    eligible = eligible_capacity_failure(source_entry)
    if not eligible or measured_blood:
        folder.mkdir(parents=True)
        result = deepcopy(source_entry)
        result.pop("result_ref", None)
        audit.update(selected_formula="PKSimStandard", baseline_retained_exactly=True,
                     reason="qualified_blood_measurement_blocks_formula_override" if measured_blood else
                            "retain_valid_baseline" if source_entry["status"] == "predicted" else
                            "failure_is_not_a_proved_vascular_capacity_conflict")
        if measured_blood:
            audit["protected_blood_evidence"] = measured_blood
        result.update(version=VERSION, distribution_audit=audit, native_solver_calls=0,
                      linear_trajectory_calls=0, request=deepcopy(request), target_observations_used=False)
        return _complete(folder, root, result)

    source_native = original._read_ref(source_entry["native_source_ref"], root)
    if (source_native.get("request_sha256") != original._digest(request)
            or source_native.get("partition_method") != "PKSimStandard"
            or source_native.get("adapter_sha256") != sha256(Path(original.__file__))
            or source_native.get("status") != "predicted"
            or source_native.get("execution_ref") != source_entry.get("execution_ref")):
        raise ValueError("Capacity fallback must originate from a verified Standard native model")
    source_physiology = original._read_ref(source_entry["physiology_ref"], root)
    audit.update(fallback_attempted=True, baseline_retained_exactly=False,
                 original_failure_ref=deepcopy(source_entry.get("result_ref")),
                 original_physiology_ref=deepcopy(source_entry["physiology_ref"]),
                 original_blood_to_plasma_ratio=source_physiology["blood_to_plasma_ratio"],
                 original_rbc_kp_plasma=source_physiology["rbc_kp_plasma"],
                 selected_formula="Schmitt", reason="uniform_input_only_capacity_fallback")
    folder.mkdir(parents=True)
    native = original.run_native_partition(request, folder / "source", root, "Schmitt")
    result = {
        "version": VERSION, "status": "failed", "method": "ours_capacity_qualified_schmitt",
        "request": deepcopy(request), "request_sha256": original._digest(request),
        "observations_used": False, "target_observations_used": False,
        "curve": None, "endpoints": None, "distribution_audit": audit,
        "native_source_ref": native["result_ref"], "execution_ref": native["execution_ref"],
        "sidecar_sources": original._check_sidecars(request, vss_evidence, root)
                           + original._check_sidecars(request, half_evidence, root),
        "native_solver_calls": 1, "linear_trajectory_calls": 0,
        "prediction_horizon_h": original.HORIZON_H,
        "interpretation": "Native Schmitt equilibrium capacities with the unchanged human vascular/tissue exchange framework",
    }
    if native["status"] != "predicted":
        result["reason"] = "native_source_failed"
        audit["alternative_outcome"] = native.get("reason", "native_failed")
        return _complete(folder, root, result)
    if native.get("partition_method") != "Schmitt" or native.get("request_sha256") != original._digest(request):
        raise ValueError("Alternative native formula or exact construction inputs changed")
    physiology = schmitt_physiology.export_physiology(
        execution_manifest_path=root / native["execution_ref"]["path"],
        source_manifest_sha256=native["execution_ref"]["sha256"],
        output_dir=folder / "physiology", project_root=root)
    result["physiology_ref"] = physiology["manifest"]
    audit.update(alternative_blood_to_plasma_ratio=physiology["blood_to_plasma_ratio"],
                 alternative_rbc_kp_plasma=physiology["rbc_kp_plasma"],
                 physiology_mapping_version=schmitt_physiology.VERSION)
    model_request = deepcopy(request)
    canonicalized = []
    for row in model_request["evidence"]:
        if row["parameter"] in {"total_clearance", "hepatic_clearance", "renal_clearance"} and row.get("context", {}).get("route") == "iv":
            row["context"]["route"] = "intravenous"
            canonicalized.append(row["parameter"])
    result["iv_route_alias_canonicalized"] = canonicalized
    # Same grid, optimizer, bounds and input target as the frozen baseline.
    prediction = simulate_learned_design(physiology, model_request, deepcopy(vss_evidence),
        deepcopy(half_evidence), ps_bounds=(.01, 100.), fallback=True, target_basis="sampled_nca",
        max_trajectory_calls=33, times_h=np.asarray(native["curve"]["time_min"], dtype=float) / 60.)
    result["design_audit"] = prediction.get("learned_exchange_audit", {})
    result["linear_trajectory_calls"] = result["design_audit"].get("linear_trajectory_calls", 0)
    if prediction["status"] != "predicted":
        result["reason"] = prediction.get("reason", "learned_exchange_failed")
        audit["alternative_outcome"] = "still_infeasible"
        return _complete(folder, root, result)
    full = prediction["full_trajectory"]
    mw = next(r["value"] for r in request["evidence"] if r["parameter"] == "molecular_weight")
    curve = {"time_min": [float(t) * 60. for t in full["time_h"]],
             "concentration": [float(c) * 1000. / mw for c in full["concentration_mg_l"]],
             "time_unit": "min", "unit": "umol/l"}
    if not all(math.isfinite(c) and c >= 0 for c in curve["concentration"]):
        raise ArithmeticError("Alternative produced invalid plasma concentrations")
    result.update(status="predicted", curve=curve,
                  endpoints=original.summarize_extended_prediction(curve, request),
                  mass_balance=prediction["mass_balance"], construction_decision=prediction.get("decision"),
                  observer=prediction["observer"], exact_moments_used_to_fill_NCA=False,
                  runtime_binding=native.get("runtime_binding"))
    audit["alternative_outcome"] = "constructed"
    return _complete(folder, root, result)
