"""Literature-informed middle-out fitting on explicitly allowed development data.

Keep human physiology and source evidence intact. Compare only registered
partition formulas and at most two numerical construction parameters. This is
regularized calibration, not a Bayesian posterior or an LLM implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

from .evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint, _digest
from .evidence_evaluation import observed_rows
from .pk_anchored_blueprint import _drug_signature, _target_vss, _run, mapped_snapshot
from .source_workflow import inside, save, sha256

VERSION = "development-curve-pbpk-1.2"


def _training_document(record, request, root):
    """Validate identity, declared study role, immutable source and exclusions."""
    path = inside(record["path"], root, exists=True)
    if record.get("role") != "calibration_data_reconstruction":
        raise ValueError("Only declared development curves may be fitted")
    if path.stat().st_size > 2 * 1024**2 or sha256(path) != record["sha256"]:
        raise ValueError("Development observation file changed or exceeds size limit")
    for forbidden in request.get("forbidden_sources", []):
        if path == inside(forbidden["path"], root) or record["sha256"] == forbidden["sha256"]:
            raise ValueError("Comparison observations cannot enter calibration")
    document = json.loads(path.read_text())
    if (document.get("compound_id") != request["compound_id"]
            or document.get("role") != record["role"]
            or document.get("matrix") != "plasma"
            or document.get("time_unit") != "min" or document.get("unit") != "umol/l"):
        raise ValueError("Development curve identity, role, matrix or units do not match")
    observed_rows(document)
    if any(row.get("unit", "umol/l") != "umol/l" or row.get("time_unit", "min") != "min"
           for row in document["observations"]):
        raise ValueError("Individual observation units must agree with document units")
    return document


def _annotate(output, training_source):
    path = output / "endpoints.json"
    if path.exists():
        document = json.loads(path.read_text())
        document.update(parameter_construction="regularized development curve calibration; no comparison curve fitting",
                        development_curve_used=True, development_source=training_source,
                        observations_used=True, endpoint_estimator_uses_observations=False,
                        independent_validation=False)
        save(path, document)


def _run_development(snapshot, request, output, root, *, audit, require_vss):
    """An unneeded AUMC gate must not suppress an independently valid CL."""
    result = _run(snapshot, request, output, root, audit=audit)
    if result.get("status") == "predicted" and result.get("clearance_l_h") is None and not require_vss:
        from .pk_anchor_moments import model_integral_pk
        curve = json.loads((output / "prediction_locked.json").read_text())
        moment = model_integral_pk(curve, request, require_vss=False)
        save(output / "clearance_only_integral.json", moment)
        if moment["status"] == "ok":
            result.update(clearance_l_h=moment["clearance_l_h"], vss_l=moment.get("vss_l"),
                pk_anchor_estimation_basis="converged model AUC; CL and optional Vss gates assessed separately")
    return result


def fit_development_mapping(*, reference_request, training_source, vss_anchor,
                            output_dir, project_root, max_evaluations=18,
                            recover_failed_trials=False):
    from .development_curve_optimizer import optimize_development_curve
    if not isinstance(recover_failed_trials, bool):
        raise ValueError("recover_failed_trials must be a boolean")
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    if output.exists():
        raise FileExistsError("Existing fit output is immutable")
    req = EvidenceBlueprintRequest.model_validate(reference_request).model_dump(mode="json", exclude_none=True)
    if req["dosing"]["route"] != "intravenous":
        raise ValueError("Current development calibration requires single-dose IV")
    observed = _training_document(training_source, req, root)
    blueprint = assemble_evidence_blueprint(req, root)
    if blueprint["status"] != "assembled":
        raise ValueError("Required drug evidence is incomplete")
    allocation = blueprint["manifest"]["clearance_allocation"]
    target_cl = sum(allocation[k] for k in ("hepatic_ml_min_kg", "renal_ml_min_kg")) * req["human"]["weight_kg"] * .06
    target_vss = _target_vss(vss_anchor, req, root)
    logp = next(e["value"] for e in req["evidence"] if e["parameter"] == "logp")
    output.mkdir(parents=True)
    save(output / "input.json", {"request": req, "training_source": training_source,
         "vss_anchor": vss_anchor, "target_clearance_l_h": target_cl, "target_vss_l": target_vss,
         "recover_failed_trials": recover_failed_trials})
    calls = 0

    def evaluate(lipophilicity, clearance_scale):
        nonlocal calls
        calls += 1
        folder = output / f"attempt_{calls:02d}"
        folder.mkdir()
        native = mapped_snapshot(blueprint, lipophilicity, clearance_scale)
        result = _run_development(native, req, folder, root, audit=False, require_vss=target_vss is not None)
        _annotate(folder, training_source)
        save(folder / "result.json", result)
        save(output / "progress.json", {"status": "running", "native_evaluations": calls,
             "maximum_evaluations": max_evaluations, "latest": result})
        if result.get("status") == "predicted":
            result = {**result, "curve": json.loads((folder / "prediction_locked.json").read_text())}
        return result

    fit = optimize_development_curve(evaluate, observations=observed,
        initial_lipophilicity=logp, logp_prior=logp, target_clearance_l_h=target_cl,
        target_vss_l=target_vss, max_evaluations=max_evaluations,
        recover_failed_trials=recover_failed_trials)
    _training_document(training_source, req, root)
    _target_vss(vss_anchor, req, root)
    fit.update(version=VERSION, compound_id=req["compound_id"], drug_signature=_drug_signature(req),
        construction_partition_method=req["partition_method"], reference_human=req["human"],
        reference_dosing=req["dosing"], evidence_ledger=req["evidence_ledger"],
        target_vss_anchor=vss_anchor, target_clearance_l_h=target_cl, target_vss_l=target_vss,
        training_source=training_source, development_curve_used=True, comparison_curve_used=False,
        used_complete_drug_model=False, independent_validation=False,
        eligible=fit.get("decision") is not None,
        interpretation="Regularized development calibration; candidate selection is in-sample",
        limitations=["Source CL/Vss may derive from the development study, and are not independent observations.",
            "Regularization widths and weights are engineering choices, not physiological confidence intervals.",
            "Effective membrane affinity is fitted; the measured logP record is unchanged.",
            "A feasible selected point is not necessarily numerically converged or clinically accurate.",
            "Generic physiology and digitized sampling times remain unchanged."])
    save(output / "mapping_locked.json", fit)
    save(output / "progress.json", {"status": fit["status"], "native_evaluations": calls,
         "eligible": fit["eligible"]})
    return fit


def predict_with_development_mapping(*, request, mapping_path, output_dir, project_root):
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    path = inside(mapping_path, root, exists=True)
    if output.exists():
        raise FileExistsError("Existing predictions are immutable")
    mapping_hash = sha256(path)
    mapping = json.loads(path.read_text())
    req = EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
    if (mapping.get("version") not in {"development-curve-pbpk-1.0", "development-curve-pbpk-1.1", VERSION} or not mapping.get("eligible")
            or not mapping.get("decision") or mapping["drug_signature"] != _drug_signature(req)):
        raise ValueError("Mapping is not eligible or belongs to different drug evidence")
    if req["dosing"]["route"] != "intravenous":
        raise ValueError("The current mapping supports only single-dose IV input")
    _training_document(mapping["training_source"], req, root)
    _target_vss(mapping["target_vss_anchor"], req, root)
    blueprint = assemble_evidence_blueprint(req, root)
    if blueprint["status"] != "assembled":
        raise ValueError("Required drug evidence is incomplete")
    point = mapping["decision"]
    native = mapped_snapshot(blueprint, point["lipophilicity"], point["clearance_scale"])
    output.mkdir(parents=True)
    save(output / "request.json", req)
    save(output / "construction.json", {"version": VERSION, "mapping_path": str(path.relative_to(root)),
        "mapping_version": mapping["version"], "failure_recovery": mapping.get("failure_recovery"),
        "mapping_sha256": mapping_hash, "training_source": mapping["training_source"],
        "evidence_manifest": blueprint["manifest"], "decision": point,
        "request_sha256": _digest(req), "snapshot_sha256": _digest(native)})
    result = _run_development(native, req, output, root, audit=True,
                              require_vss=mapping.get("target_vss_l") is not None)
    _annotate(output, mapping["training_source"])
    if sha256(path) != mapping_hash:
        raise ValueError("Frozen mapping changed during prediction")
    _training_document(mapping["training_source"], req, root)
    _target_vss(mapping["target_vss_anchor"], req, root)
    result.update(development_curve_used=True, comparison_curve_used=False)
    save(output / "result.json", result)
    return result
