"""Translate out-of-compound ML PK predictions into a new PBPK construction.

The two numerical mapping targets are predictions, never clinical anchors. This
module has no observation input. Matching those predictions is not a clinical
validation result; a separate, frozen evaluation must assess clinical accuracy.
"""
from __future__ import annotations

import json
import math
from numbers import Integral
from pathlib import Path

from .evidence_blueprint import EvidenceBlueprintRequest, _digest, assemble_evidence_blueprint
from .pk_anchor_optimizer import optimize_pk_anchors
from .pk_anchored_blueprint import HORIZON_H, _run, mapped_snapshot
from .source_workflow import inside, save, sha256

VERSION = "ml-hybrid-blueprint-1.0"
CONSTRUCTION = "matching_ML_predictions_not_clinical_fitting"
SUPPORTED_METHODS = {"random_forest", "training_median"}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive number")
    return float(value)


def _verified_file(reference, root, request, *, location_required=False):
    if not isinstance(reference, dict) or (location_required and not reference.get("location")):
        raise ValueError("Prediction provenance requires a located project-local source")
    path = inside(reference["path"], root, exists=True)
    digest = reference.get("sha256")
    if not path.is_file() or not isinstance(digest, str) or sha256(path) != digest:
        raise ValueError("Prediction source or model/split provenance checksum changed")
    if any(digest == forbidden["sha256"] or path == inside(forbidden["path"], root)
           for forbidden in request.get("forbidden_sources", [])):
        raise ValueError("Prediction/model/split source belongs to forbidden clinical comparison data")
    return path


def _predicted_targets(request, source, root):
    path = _verified_file(source, root, request, location_required=True)
    if path.suffix.lower() != ".json" or path.stat().st_size > 2 * 1024**2:
        raise ValueError("ML prediction must be a bounded JSON document")
    record = json.loads(path.read_text())
    if record.get("compound_id") != request["compound_id"]:
        raise ValueError("ML prediction compound identity differs from requested drug")
    if record.get("method") not in SUPPORTED_METHODS:
        raise ValueError("ML method is outside the frozen small-pilot methods")
    if (record.get("training_overlap") is not False
            or record.get("same_scaffold_training_overlap") is not False):
        raise ValueError("Held-out compound and scaffold exclusions must be explicitly false for overlap")
    for name in ("split", "model"):
        _verified_file(record.get("provenance", {}).get(name), root, request)
    values = record.get("predictions", {})
    targets = {key: _positive(values.get(key), key) for key in
               ("clearance_ml_min_kg", "vss_l_kg", "fraction_unbound")}
    if targets["fraction_unbound"] > 1:
        raise ValueError("Predicted unbound fraction must not exceed one")
    evidence = {entry["parameter"]: entry for entry in request["evidence"]}
    for parameter, key in (("total_clearance", "clearance_ml_min_kg"), ("fraction_unbound", "fraction_unbound")):
        entry = evidence.get(parameter, {})
        ref = entry.get("source", {})
        if (entry.get("basis") != "predicted" or ref.get("sha256") != source["sha256"]
                or inside(ref.get("path", "."), root) != path
                or not math.isclose(_positive(entry.get("value"), parameter), targets[key], rel_tol=1e-12)):
            raise ValueError(parameter + " must remain explicitly predicted and match the locked ML source")
        if entry.get("prediction_method", {}).get("training_overlap") is not False:
            raise ValueError("Evidence prediction_method must preserve the held-out training status")
    return record, targets


def _run_predicted(snapshot, request, output, root, *, audit):
    result = _run(snapshot, request, output, root, audit=audit)
    # The shared execution helper describes evidence-anchor construction. Make
    # the distinct meaning of this caller explicit in every persisted endpoint.
    endpoint_path = output / "endpoints.json"
    if endpoint_path.exists():
        endpoints = json.loads(endpoint_path.read_text())
        endpoints.update(parameter_construction=CONSTRUCTION,
                         observations_used_for_parameter_construction=False,
                         construction_target_basis="predicted",
                         clinical_validation=False)
        save(endpoint_path, endpoints)
    return {**result, "parameter_construction": CONSTRUCTION,
            "construction_target_basis": "predicted", "observations_used": False}


def run_ml_hybrid_case(*, request, predicted_pk_source, output_dir, project_root, max_evaluations=12):
    """Fit native mapping to predicted CL/Vss, then execute with complete audits.

    At most ``max_evaluations`` construction trials and one final audited run are
    made. First-trial direct predictions and all unsuccessful attempts remain
    available. Failed mapping never falls back to an unaccepted best attempt.
    No clinical curve, observed endpoint, or source-derived fit window is read.
    """
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral)
            or not 1 <= max_evaluations <= 12):
        raise ValueError("max_evaluations must be an integer from 1 through 12")
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    if output.exists():
        raise FileExistsError("Existing ML hybrid outputs are immutable")
    req = EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
    if req["dosing"]["route"] != "intravenous" or req["partition_method"] != "PKSimStandard":
        raise ValueError("Small-pilot ML construction requires single IV input and fixed PKSimStandard")
    if req["target"].get("terminal_fit") is not None:
        raise ValueError("Clinical terminal-fit windows must not enter ML parameter construction")
    prediction, targets = _predicted_targets(req, predicted_pk_source, root)
    blueprint = assemble_evidence_blueprint(req, root)
    output.mkdir(parents=True)
    save(output / "request.json", req)
    save(output / "input.json", {"version": VERSION, "predicted_pk_source": predicted_pk_source,
         "method": prediction["method"], "predicted_targets": targets,
         "construction": CONSTRUCTION, "observations_used": False,
         "partition_method": "PKSimStandard", "fixed_model_horizon_h": HORIZON_H,
         "max_mapping_evaluations": int(max_evaluations), "max_total_native_runs": int(max_evaluations) + 1,
         "request_sha256": _digest(req), "source_integrity": blueprint["manifest"]})
    if blueprint["status"] != "assembled":
        result = {"status": "construction_failed", "gaps": blueprint.get("gaps", []),
                  "construction": CONSTRUCTION, "native_evaluations": 0}
        save(output / "result.json", result)
        save(output / "progress.json", result)
        return result
    weight = req["human"]["weight_kg"]
    target_cl = targets["clearance_ml_min_kg"] * weight * .06
    target_vss = targets["vss_l_kg"] * weight
    allocation = blueprint["manifest"]["clearance_allocation"]
    allocated_cl = (allocation["hepatic_ml_min_kg"] + allocation["renal_ml_min_kg"]) * weight * .06
    if not math.isclose(allocated_cl, target_cl, rel_tol=1e-12):
        raise ValueError("Organ input clearances do not preserve predicted total clearance")
    logp = next(entry["value"] for entry in req["evidence"] if entry["parameter"] == "logp")
    calls = 0

    def evaluate(lipophilicity, clearance_scale):
        nonlocal calls
        if calls >= max_evaluations:
            raise RuntimeError("Native mapping budget exceeded")
        _predicted_targets(req, predicted_pk_source, root)
        calls += 1
        folder = output / f"attempt_{calls:02d}"
        folder.mkdir()
        snapshot = mapped_snapshot(blueprint, lipophilicity, clearance_scale)
        result = _run_predicted(snapshot, req, folder, root, audit=False)
        save(folder / "result.json", result)
        save(output / "progress.json", {"status": "mapping", "native_evaluations": calls,
             "max_mapping_evaluations": int(max_evaluations), "latest": result})
        return result

    fit = optimize_pk_anchors(evaluate, initial_lipophilicity=logp, target_clearance_l_h=target_cl,
                             target_vss_l=target_vss, max_evaluations=int(max_evaluations))
    fit.update(version=VERSION, construction=CONSTRUCTION, compound_id=req["compound_id"],
               method=prediction["method"], predicted_pk_source=predicted_pk_source,
               target_basis="predicted", observations_used=False, clinical_validation=False,
               partition_method="PKSimStandard", request_sha256=_digest(req),
               interpretation="Numerical translation of held-out ML predictions; not agreement with clinical observations")
    fit["optimizer"]["objective"] = "sum_squared_log_simulated_over_ML_predicted_target"
    save(output / "mapping_locked.json", fit)
    if not fit["converged"] or fit["decision"] is None:
        result = {"status": "mapping_failed", "mapping_status": fit["status"], "reason": fit["reason"],
                  "native_evaluations": calls, "mapping_evaluations": calls, "construction": CONSTRUCTION,
                  "mapping_locked_path": str(output / "mapping_locked.json"), "prediction_path": None,
                  "endpoints_path": None, "clinical_validation": False}
    else:
        _predicted_targets(req, predicted_pk_source, root)
        decision = fit["decision"]
        snapshot = mapped_snapshot(blueprint, decision["lipophilicity"], decision["clearance_scale"])
        final = output / "final_prediction"
        final.mkdir()
        save(final / "construction.json", {"construction": CONSTRUCTION, "target_basis": "predicted",
             "mapping_path": str((output / "mapping_locked.json").relative_to(root)),
             "mapping_sha256": sha256(output / "mapping_locked.json"),
             "request_sha256": _digest(req), "snapshot_sha256": _digest(snapshot)})
        result = _run_predicted(snapshot, req, final, root, audit=True)
        _predicted_targets(req, predicted_pk_source, root)
        if result.get("status") == "predicted":
            cl_error = abs(_positive(result.get("clearance_l_h"), "Final model CL") / target_cl - 1)
            vss_error = abs(_positive(result.get("vss_l"), "Final model Vss") / target_vss - 1)
            if (not result.get("native_audits_passed") or cl_error > .05 or vss_error > .10):
                result.update(status="final_mapping_verification_failed",
                              reason="Audited final execution did not preserve accepted ML mapping")
            result.update(final_target_relative_errors={"clearance": cl_error, "vss": vss_error})
        result.update(mapping_status=fit["status"], native_evaluations=calls + 1, mapping_evaluations=calls,
                      construction=CONSTRUCTION, mapping_locked_path=str(output / "mapping_locked.json"),
                      prediction_path=str(final / "prediction_locked.json") if (final / "prediction_locked.json").exists() else None,
                      endpoints_path=str(final / "endpoints.json") if (final / "endpoints.json").exists() else None,
                      clinical_validation=False)
        save(final / "result.json", result)
    direct = output / "attempt_01" / "prediction_locked.json"
    result["direct_prediction_path"] = str(direct) if direct.exists() else None
    save(output / "result.json", result)
    save(output / "progress.json", result)
    return result


def run_ml_direct_case(*, request, predicted_pk_source, output_dir, project_root):
    """Run the same ML inputs directly, without numerical CL/Vss matching.

    Uses the identical model horizon and final audits as the mapped method.
    Predicted Vss is recorded but unused in this direct-input control.
    """
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    if output.exists():
        raise FileExistsError("Existing direct ML outputs are immutable")
    req = EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
    if req["dosing"]["route"] != "intravenous" or req["partition_method"] != "PKSimStandard":
        raise ValueError("Small-pilot ML construction requires single IV input and fixed PKSimStandard")
    if req["target"].get("terminal_fit") is not None:
        raise ValueError("Clinical terminal-fit windows must not enter ML parameter construction")
    prediction, targets = _predicted_targets(req, predicted_pk_source, root)
    blueprint = assemble_evidence_blueprint(req, root)
    output.mkdir(parents=True)
    save(output / "request.json", req)
    construction = "direct_ML_inputs_without_CL_Vss_mapping"
    save(output / "construction.json", {"version": VERSION, "construction": construction,
         "predicted_pk_source": predicted_pk_source, "method": prediction["method"],
         "predicted_inputs": targets, "predicted_vss_used": False,
         "partition_method": "PKSimStandard", "fixed_model_horizon_h": HORIZON_H,
         "request_sha256": _digest(req), "source_integrity": blueprint["manifest"],
         "observations_used": False, "construction_parameters_fitted": False})
    if blueprint["status"] != "assembled":
        result = {"status": "construction_failed", "gaps": blueprint.get("gaps", []),
                  "native_evaluations": 0, "prediction_path": None, "endpoints_path": None}
    else:
        logp = next(entry["value"] for entry in req["evidence"] if entry["parameter"] == "logp")
        snapshot = mapped_snapshot(blueprint, logp, 1.)
        result = _run(snapshot, req, output, root, audit=True)
        _predicted_targets(req, predicted_pk_source, root)
        endpoint_path = output / "endpoints.json"
        if endpoint_path.exists():
            endpoints = json.loads(endpoint_path.read_text())
            endpoints.update(parameters_fitted=False, construction_parameters_fitted=False,
                             parameter_construction=construction,
                             observations_used_for_parameter_construction=False,
                             construction_target_basis="predicted", clinical_validation=False)
            save(endpoint_path, endpoints)
        result.update(native_evaluations=1,
                      prediction_path=str(output / "prediction_locked.json") if (output / "prediction_locked.json").exists() else None,
                      endpoints_path=str(endpoint_path) if endpoint_path.exists() else None)
    result.update(construction=construction, construction_target_basis="predicted",
                  observations_used=False, clinical_validation=False,
                  construction_parameters_fitted=False)
    save(output / "result.json", result)
    save(output / "progress.json", result)
    return result
