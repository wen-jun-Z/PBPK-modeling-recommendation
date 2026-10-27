"""Map human PK evidence to a newly assembled model, without fitting test curves.

The measured logP and CL ledger remains immutable. Effective membrane affinity
and native plasma-clearance inputs are derived construction parameters, not new
measurements. One mapping is frozen on a declared reference human and reused.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from .evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint, _digest
from .extended_prediction import summarize_extended_prediction
from .source_workflow import inside, read_curve, save, sha256

VERSION = "evidence-pk-anchors-1.0"
HORIZON_H = 1344.


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive number")
    return float(value)


def _target_vss(anchor, request, root):
    if anchor is None:
        return None
    required_context = {"species": "human", "route": "intravenous", "matrix": "plasma", "definition": "steady_state_volume"}
    if (anchor.get("basis") != "measured" or anchor.get("compound_id") != request["compound_id"]
            or anchor.get("context", {}).get("study_role") != "development"
            or any(anchor.get("context", {}).get(k) != v for k, v in required_context.items())):
        raise ValueError("Vss anchor must be explicitly curated human IV plasma steady-state volume")
    source = anchor["source"]
    path = inside(source["path"], root, exists=True)
    forbidden = request.get("forbidden_sources", [])
    if (not source.get("location") or sha256(path) != source["sha256"]
            or any(source["sha256"] == f["sha256"] or path == inside(f["path"], root) for f in forbidden)):
        raise ValueError("Vss source changed or belongs to comparison data")
    value = _positive(anchor["value"], "Vss")
    if anchor["unit"] == "L":
        return value
    if anchor["unit"] == "L/kg":
        return value * request["human"]["weight_kg"]
    raise ValueError("Vss must be L or L/kg; terminal Vz or oral V/F is not accepted")


def _drug_signature(request):
    return _digest({k: request[k] for k in ("compound_id", "compound_name", "evidence", "evidence_ledger", "partition_method")})


def mapped_snapshot(blueprint, lipophilicity, clearance_scale):
    """Change two construction quantities only, preserving all evidence inputs."""
    scale = _positive(clearance_scale, "clearance_scale")
    if (not .25 <= scale <= 4. or isinstance(lipophilicity, bool)
            or not isinstance(lipophilicity, (int, float)) or not math.isfinite(lipophilicity)):
        raise ValueError("Construction mapping outside finite bounded range")
    native = deepcopy(blueprint["snapshot"])
    compound = native["Compounds"][0]
    logp = compound["Lipophilicity"][0]["Parameters"][0]["Value"]
    if abs(lipophilicity - logp) > 2.5 + 1e-12:
        raise ValueError("Effective membrane affinity exceeds declared logP search range")
    compound["Lipophilicity"][0]["Name"] = "PKAnchorEffectiveMembraneAffinity"
    compound["Lipophilicity"][0]["Parameters"][0]["Value"] = float(lipophilicity)
    # Keep the selected alternative in the simulation bound to the renamed
    # building block. Native readback below verifies the resulting values.
    def rename(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if value == "EvidenceLogP":
                    node[key] = "PKAnchorEffectiveMembraneAffinity"
                else:
                    rename(value)
        elif isinstance(node, list):
            for value in node:
                rename(value)
    rename(native["Simulations"])
    for process in compound["Processes"]:
        if process["InternalName"] not in {"LiverClearance", "KidneyClearance"}:
            raise ValueError("Only existing systemic clearance processes may be mapped")
        for parameter in process["Parameters"]:
            if parameter["Name"] == "Plasma clearance":
                parameter["Value"] *= scale
            elif parameter["Name"] == "Lipophilicity (experiment)":
                parameter["Value"] = float(lipophilicity)
    schema = native["Simulations"][0]["OutputSchema"]
    end = next(p["Value"] for p in schema[-1]["Parameters"] if p["Name"] == "End time")
    # Preserve the whole requested grid, add a dense early tail and a bounded
    # late tail. No clinical concentrations choose times or terminal windows.
    for boundary, resolution in ((max(48., end), 60.), (HORIZON_H, 4.)):
        if boundary > end:
            schema.append({"Parameters": [{"Name": "Start time", "Value": end, "Unit": "h"},
                {"Name": "End time", "Value": boundary, "Unit": "h"},
                {"Name": "Resolution", "Value": resolution, "Unit": "pts/h"}]})
            end = boundary
    return native


def _run(snapshot, request, output, root, *, audit):
    from .design_runtime import execute_snapshot
    from .evidence_native_audit import audit_evidence_execution
    from .source_adaptation_audit import audit_execution, derive_administration_expectations
    from .pk_anchor_moments import model_integral_pk
    execution = execute_snapshot(snapshot=snapshot, output_dir=output / "native", project_root=root,
        timeout_seconds=180, cpu_threads=2, max_output_bytes=64 * 1024**2)
    if not execution.get("engineering_accepted"):
        return {"status": "native_failed", "execution": execution}
    if audit:
        check = audit_execution(snapshot=snapshot, execution_manifest=execution, output_dir=output / "readback",
            project_root=root, **derive_administration_expectations(snapshot))
        save(output / "readback_result.json", check)
        if not check.get("accepted"):
            return {"status": "administration_readback_failed"}
    # Check physical capacity on every optimizer trial, not just its final
    # point: PK-Sim can otherwise fall back to an artificial 1e6 clearance.
    check = audit_evidence_execution(snapshot, execution["manifest_path"], output / "parameter_audit", root)
    save(output / "parameter_audit_result.json", check)
    if not check.get("accepted"):
        return {"status": "parameter_readback_failed", "reason": check.get("reasons", [])}
    manifest_path = Path(execution["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    path = snapshot["Simulations"][0]["OutputSelections"][0]
    curve = read_curve(manifest_path.parent, manifest, path)
    endpoints = summarize_extended_prediction(curve, request)
    endpoints.update(parameters_fitted=True, construction_parameters_fitted=True,
        endpoint_estimator_fitted_pbpk_parameters=False,
        parameter_construction="CL and optional Vss evidence anchors; no comparison curve fitting")
    save(output / "prediction_locked.json", curve)
    save(output / "endpoints.json", endpoints)
    values = endpoints["endpoints"]
    cl, vss = values["clearance"].get("value_l_per_h"), values["vss"].get("value_l")
    moment = model_integral_pk(curve, request)
    save(output / "model_integral_pk.json", moment)
    basis = "quality_gated_terminal_NCA"
    if cl is None or vss is None:
        if moment["status"] == "ok":
            cl, vss = moment["clearance_l_h"], moment["vss_l"]
            basis = "converged_model_integrals_only; terminal half-life not inferred"
    return {"status": "predicted", "clearance_l_h": cl,
        "vss_l": vss, "pk_anchor_estimation_basis": basis,
        "endpoint_statuses": {k: v.get("status", "completed") for k, v in values.items()},
        "prediction_sha256": sha256(output / "prediction_locked.json"), "native_audits_passed": audit,
        "mass_balance_checked": False}


def fit_pk_mapping(*, reference_request, vss_anchor, output_dir, project_root, max_evaluations=24):
    """Fit only CL/Vss evidence on one declared reference case; no curve input."""
    from .pk_anchor_optimizer import optimize_pk_anchors
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    if output.exists():
        raise FileExistsError("Existing fit output is immutable")
    request = EvidenceBlueprintRequest.model_validate(reference_request).model_dump(mode="json", exclude_none=True)
    if request["dosing"]["route"] != "intravenous":
        raise ValueError("Human CL/Vss anchors require single-dose IV")
    blueprint = assemble_evidence_blueprint(request, root)
    if blueprint["status"] != "assembled":
        raise ValueError("Core drug evidence is incomplete")
    allocation = blueprint["manifest"]["clearance_allocation"]
    target_cl = (allocation["hepatic_ml_min_kg"] + allocation["renal_ml_min_kg"]) * request["human"]["weight_kg"] * .06
    target_vss = _target_vss(vss_anchor, request, root)
    logp = next(e["value"] for e in request["evidence"] if e["parameter"] == "logp")
    output.mkdir(parents=True)
    save(output / "input.json", {"request": request, "vss_anchor": vss_anchor, "target_clearance_l_h": target_cl,
        "target_vss_l": target_vss, "parameter_sources": blueprint["manifest"]["sources"]})
    calls = 0
    def evaluate(lipophilicity, clearance_scale):
        nonlocal calls
        calls += 1
        folder = output / f"attempt_{calls:02d}"
        snapshot = mapped_snapshot(blueprint, lipophilicity, clearance_scale)
        folder.mkdir()
        result = _run(snapshot, request, folder, root, audit=False)
        save(folder / "result.json", result)
        save(output / "progress.json", {"status": "running", "native_evaluations": calls,
            "maximum_evaluations": max_evaluations, "latest": result})
        return result
    fit = optimize_pk_anchors(evaluate, initial_lipophilicity=logp, target_clearance_l_h=target_cl,
        target_vss_l=target_vss, max_evaluations=max_evaluations)
    fit.update(version=VERSION, compound_id=request["compound_id"], drug_signature=_drug_signature(request),
        construction_partition_method=request["partition_method"],
        reference_human=request["human"], reference_dosing=request["dosing"],
        evidence_ledger=request["evidence_ledger"], target_vss_anchor=vss_anchor,
        target_clearance_l_h=target_cl, target_vss_l=target_vss,
        reference_population_assumptions=request.get("assumptions", []),
        anchor_population_notes=vss_anchor.get("notes") if vss_anchor else None,
        observations_used=False, used_complete_drug_model=False, independent_validation=False,
        interpretation="PK evidence reconstruction, not independent clinical validation",
        limitations=["Effective membrane affinity is derived; measured logP remains unchanged.",
            "CL scale is a native-input translation, not a new measured systemic clearance.",
            "Reference human may contain declared population defaults.",
            "A total CL target does not validate hepatic/renal allocation or tissue-specific exposure.",
            "Numerical CL/Vss agreement does not prove curve, peak, terminal phase, or clinical decision accuracy."])
    save(output / "mapping_locked.json", fit)
    save(output / "progress.json", {"status": fit["status"], "native_evaluations": calls, "converged": fit["converged"]})
    return fit


def predict_with_pk_mapping(*, request, mapping_path, output_dir, project_root):
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    path = inside(mapping_path, root, exists=True)
    if output.exists():
        raise FileExistsError("Existing predictions are immutable")
    mapping_hash = sha256(path)
    mapping = json.loads(path.read_text())
    req = EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
    if not mapping.get("converged") or mapping.get("decision") is None or mapping["drug_signature"] != _drug_signature(req):
        raise ValueError("Mapping is unconverged or belongs to different drug evidence")
    if req["dosing"]["route"] != "intravenous":
        raise ValueError("This mapping has only been qualified numerically for IV input")
    # Recheck sources, including the optional Vss anchor, before every use.
    _target_vss(mapping["target_vss_anchor"], req, root)
    blueprint = assemble_evidence_blueprint(req, root)
    decision = mapping["decision"]
    native = mapped_snapshot(blueprint, decision["lipophilicity"], decision["clearance_scale"])
    output.mkdir(parents=True)
    save(output / "request.json", req)
    save(output / "construction.json", {"version": VERSION, "mapping_path": str(path.relative_to(root)),
        "mapping_sha256": mapping_hash, "evidence_manifest": blueprint["manifest"], "decision": decision,
        "request_sha256": _digest(req), "snapshot_sha256": _digest(native)})
    result = _run(native, req, output, root, audit=True)
    if sha256(path) != mapping_hash:
        raise ValueError("Locked mapping changed during prediction")
    save(output / "result.json", result)
    return result
