"""Reuse a frozen native human template for joint-capacity PBPK construction.

Only the reduced drug-distribution mapping changes. Native PKML/readbacks,
human anatomy, flows, dose, independent inputs and shared ML weights are not
modified. No native solver, training or target observation is needed here.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from . import clinical_pbpk_comparison as original
from . import joint_partition_design as joint
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

VERSION = "joint-partition-runtime-1"
CODE_PINS = {"src/pbpk_kb/blueprint_policy/" + name: sha256(Path(__file__).with_name(name))
             for name in ("joint_partition_design.py", "schmitt_physiology.py")}


def _complete(folder, root, result):
    save(folder / "result.json", result)
    return {**result, "result_ref": original._ref(folder / "result.json", root)}


def _source(request, vss, half, entry, root):
    original._check_request(request)
    sources = original._check_sidecars(request, vss, root) + original._check_sidecars(request, half, root)
    if (entry.get("request") != request or entry.get("target_observations_used") is not False
            or entry.get("observations_used", False) is not False
            or entry.get("status") not in {"predicted", "failed"}
            or entry.get("request_sha256") != original._digest(request)):
        raise ValueError("Exact-input, observation-free, locked native construction source required")
    native = original._read_ref(entry["native_source_ref"], root)
    if (native.get("request_sha256") != original._digest(request)
            or native.get("partition_method") != "PKSimStandard"
            or native.get("adapter_sha256") != sha256(Path(original.__file__))
            or native.get("status") != "predicted"
            or native.get("observations_used") is not False
            or native.get("execution_ref") != entry.get("execution_ref")):
        raise ValueError("The source must be a verified original Standard native build")
    physiology = original._read_ref(entry["physiology_ref"], root)
    if (physiology.get("version") != "native-perfusion-physiology-1"
            or physiology.get("observations_used") is not False
            or physiology.get("source_execution_manifest_sha256") != native["execution_ref"]["sha256"]
            or physiology.get("mapping_version") is not None):
        raise ValueError("Expected raw source physiology, not a different derived mapping")
    for name, digest in {**physiology["source_pins"], **physiology["artifact_pins"]}.items():
        if sha256(inside(name, root, exists=True)) != digest:
            raise ValueError("Frozen native readback or source changed: " + name)
    return native, physiology, sources


def run_design(request, vss_evidence, half_evidence, output_dir, project_root, *, source_entry):
    """Construct one drug with frozen weights and an input-only capacity rule.

    The experiment runner independently verifies the source entry and sidecar
    receipts, seals all clinical targets, and handles incomplete executions.
    This function has no fitting or clinical-scoring interface.
    """
    root = Path(project_root).resolve(strict=True)
    folder = inside(output_dir, root)
    if folder.exists():
        raise FileExistsError("Use the verified experiment runner to resume")
    native, physiology, sources = _source(request, vss_evidence, half_evidence, source_entry, root)
    source_digest = original._digest(physiology)
    request_digest = original._digest(request)
    sidecar_digest = original._digest([vss_evidence, half_evidence])
    folder.mkdir(parents=True)
    result = {
        "version": VERSION, "status": "failed", "method": "ours_joint_partition",
        "request": deepcopy(request), "request_sha256": request_digest,
        "observations_used": False, "target_observations_used": False,
        "curve": None, "endpoints": None, "native_solver_calls": 0,
        "linear_trajectory_calls": 0, "new_training_calls": 0,
        "native_source_ref": deepcopy(source_entry["native_source_ref"]),
        "execution_ref": deepcopy(native["execution_ref"]),
        "physiology_ref": deepcopy(source_entry["physiology_ref"]),
        "sidecar_sources": sources, "prediction_horizon_h": original.HORIZON_H,
        "model_physiology_basis": "Unmodified native source plus the recorded joint-partition transformation; no native PKML mutation",
        "distribution_audit": {
            "version": VERSION, "source_status": source_entry["status"],
            "shared_predictor_retrained": False, "clinical_curve_fitting_performed": False,
            "target_observations_used": False, "selection_uses_clinical_scores": False,
            "raw_physiology_sha256": source_digest,
            "requested_vss_source": deepcopy(vss_evidence),
            "requested_half_source": deepcopy(half_evidence),
            "maximum_linear_trajectories": 33,
            "protected": ["native_PKML", "native_readback", "human_volumes", "blood_flows",
                          "fu", "CL", "renal_clearance", "Vss_input", "half_life_input", "dose", "ML_weights"],
            "limitations": ["One empirical shared distribution correction is assumed across RBC and tissues",
                            "Vss does not independently identify true blood or tissue partition coefficients",
                            "This is a plasma-endpoint construction hypothesis, not tissue concentration validation",
                            "Clinical usefulness is measured after every prediction is locked"],
        },
    }
    model_request = deepcopy(request)
    canonicalized = []
    for row in model_request["evidence"]:
        if row["parameter"] in {"total_clearance", "hepatic_clearance", "renal_clearance"} and row.get("context", {}).get("route") == "iv":
            row["context"]["route"] = "intravenous"
            canonicalized.append(row["parameter"])
    result["iv_route_alias_canonicalized"] = canonicalized
    prediction = joint.simulate_joint_design(physiology, model_request, deepcopy(vss_evidence),
        deepcopy(half_evidence), ps_bounds=(.01, 100.), fallback=True, target_basis="sampled_nca",
        max_trajectory_calls=33, times_h=np.asarray(native["curve"]["time_min"], dtype=float) / 60.)
    if (original._digest(physiology) != source_digest or original._digest(request) != request_digest
            or original._digest([vss_evidence, half_evidence]) != sidecar_digest):
        raise ValueError("Joint construction mutated a frozen input")
    result["design_audit"] = prediction.get("learned_exchange_audit", {})
    result["joint_partition_audit"] = prediction.get("joint_partition_audit", {})
    result["linear_trajectory_calls"] = result["design_audit"].get("linear_trajectory_calls", 0)
    if prediction["status"] != "predicted":
        result["reason"] = prediction.get("reason", "joint_partition_failed")
        result["distribution_audit"]["construction_outcome"] = "failed"
        return _complete(folder, root, result)
    full = prediction["full_trajectory"]
    mw = next(r["value"] for r in request["evidence"] if r["parameter"] == "molecular_weight")
    curve = {"time_min": [float(t) * 60. for t in full["time_h"]],
             "concentration": [float(c) * 1000. / mw for c in full["concentration_mg_l"]],
             "time_unit": "min", "unit": "umol/l"}
    if not all(math.isfinite(c) and c >= 0 for c in curve["concentration"]):
        raise ArithmeticError("Joint construction produced invalid concentrations")
    result.update(status="predicted", curve=curve,
        endpoints=original.summarize_extended_prediction(curve, request),
        mass_balance=prediction["mass_balance"], construction_decision=prediction.get("decision"),
        observer=prediction["observer"], exact_moments=prediction["exact_moments"],
        exact_moments_used_to_fill_NCA=False, runtime_binding=native.get("runtime_binding"))
    result["distribution_audit"]["construction_outcome"] = "predicted"
    return _complete(folder, root, result)
