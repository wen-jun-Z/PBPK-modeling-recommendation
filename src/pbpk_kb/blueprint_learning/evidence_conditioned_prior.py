"""Training-only CL-conditioned distribution priors and half-life uncertainty.

The two residual slopes and shrinkage are fixed hypotheses, not a clinical
selection rule. The inherited first-stage consistency OOF is not fully nested.
No clinical outcomes are read here and no measured construction input is edited.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

from . import clearance_refinement as previous
from . import clinical_shared_prior as shared
from . import rich_parameter_prior as rich

VERSION = "evidence-conditioned-distribution-prior-1"
OUTPUT = "experiments/blueprint_design/evidence_conditioning_20261005/prior"
SHRINKAGE = .5
RIDGE_PER_ROW = .01
CODE_FILES = (*previous.CODE_FILES,
    "src/pbpk_kb/blueprint_learning/evidence_conditioned_prior.py",
    "scripts/train_evidence_conditioned_prior.py")


def residual_slope(cl_error, endpoint_error):
    """Fixed zero-intercept ridge slope on signed log10 residuals."""
    x, y = np.asarray(cl_error, dtype=float), np.asarray(endpoint_error, dtype=float)
    if x.ndim != 1 or not x.size or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Aligned finite nonempty training residuals required")
    return float(x @ y / (x @ x + len(x) * RIDGE_PER_ROW))


def _metrics(truth, predicted):
    error = np.abs(np.asarray(truth) - np.asarray(predicted))
    return {"n": len(error), "log10_mae": float(error.mean()),
        "gmfe": float(10. ** error.mean()),
        "fold_error_p90": float(np.quantile(10. ** error, .9)),
        "within_2_count": int(np.sum(error < math.log10(2.))), "threshold_comparison": "<"}


def crossfit_correction(truth_cl, predicted_cl, truth_z, predicted_z, folds):
    """Fit each slope without its scoring fold; freeze .5 shrink, report all."""
    yc, pc, yz, pz = [np.asarray(a, dtype=float) for a in (truth_cl, predicted_cl, truth_z, predicted_z)]
    f = np.asarray(folds)
    if (yc.ndim != 1 or not yc.size or any(a.shape != yc.shape or not np.isfinite(a).all() for a in (pc, yz, pz))
            or f.shape != yc.shape or not np.isfinite(yc).all() or len(set(f.tolist())) != 3):
        raise ValueError("Three aligned finite OOF folds required")
    corrected = np.empty(len(yc)); audit = []
    for fold in sorted(set(f.tolist())):
        fit, held = f != fold, f == fold
        beta = residual_slope(yc[fit] - pc[fit], yz[fit] - pz[fit])
        corrected[held] = pz[held] + SHRINKAGE * beta * (yc[held] - pc[held])
        audit.append({"fold": int(fold), "slope": beta, "train_count": int(fit.sum()),
            "held_count": int(held.sum()), "held_positions": np.flatnonzero(held).tolist()})
    return {"slope": residual_slope(yc-pc, yz-pz), "folds": audit,
        "original": _metrics(yz, pz), "conditioned": _metrics(yz, corrected),
        "conditioned_oof_log10": corrected.tolist(),
        "interpretation": "Training development diagnostic with a held-fold residual slope; inherited base OOF crossfold dependence remains"}


def train(root, output_dir=OUTPUT, jobs=4, preference="auto", progress=None):
    """Fit only twelve missing half-life OOF members; reuse all original priors."""
    root = Path(root).resolve(strict=True)
    output = shared._inside(output_dir, root)
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 4:
        raise ValueError("jobs must be an integer in [1,4]")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(root, output, jobs, preference, progress)


def _train(root, output, jobs, preference, progress):
    begin = time.monotonic()
    old, _, wheel, refs = previous._read_source(root)
    original = old["audit"]
    device = rich.choose_device(preference)
    code = {name: shared._sha(root / name) for name in CODE_FILES}
    config = {"version": VERSION, "source_refs": refs, "code_pins": code,
        "jobs": jobs, "device": device, "half_members": list(shared.METHODS["terminal_half_life"]),
        "seed": shared.SEED, "shrinkage": SHRINKAGE, "ridge_per_row": RIDGE_PER_ROW,
        "coefficient_rule": "sum(eCL*eZ)/(sum(eCL**2)+n*0.01); signed observed-minus-predicted log10 errors",
        "half_uncertainty": "80th percentile absolute natural-log half-life OOF residual",
        "center": "Original shared projected CL/Vss and original unprojected four-member half-life",
        "selection": "No model selection or clinical rejection; two fixed residual slopes reported regardless of direction"}
    fingerprint = shared._digest(config)
    state_path, result_path = output / "training_state.json", output / "predictions.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.pop("state_sha256", None) != shared._digest(state) or state["input_fingerprint"] != fingerprint:
            raise ValueError("Changed training checkpoint or input protocol")
    else:
        state = {"config": config, "input_fingerprint": fingerprint, "started_at": previous._now(), "fits": {}}
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (result.get("input_fingerprint") != fingerprint or result.get("result_sha256") !=
                shared._digest({k:v for k,v in result.items() if k != "result_sha256"})):
            raise ValueError("Changed frozen evidence-conditioning artifact")
        return result
    def emit(stage, **updates):
        state.update(stage=stage, updated_at=previous._now(), pid=os.getpid(), **updates)
        rich._atomic_json(state_path, {**state, "state_sha256": shared._digest(state)})
        if progress: progress({"stage": stage, **updates})
    emit("verifying_training_membership")
    cohort = shared._cohort([{"compound_id":cid, "smiles":r["input_smiles"]} for cid,r in original["targets"].items()])
    rows, labels, membership = shared._training_data(wheel, cohort)
    if rows != original["rows"] or membership["train_count"] != 1126:
        raise ValueError("Original training structures changed")
    for parameter, values in labels.items():
        if sorted(values) != original["parameter_training_indices"][parameter]:
            raise ValueError("Original training-label membership changed: " + parameter)
    train_rows = [r for r in rows if r["status"] == "train"]
    positions = {r["index"]:i for i,r in enumerate(train_rows)}
    rowmap = {r["index"]:r for r in train_rows}
    folds = original["folds"]
    seen = []
    for f in folds:
        fit, held = f["train_indices"], f["validation_indices"]
        if (set(fit) & set(held) or set(fit) | set(held) != set(positions)
                or {rowmap[i]["scaffold_group"] for i in fit} & {rowmap[i]["scaffold_group"] for i in held}):
            raise ValueError("Original scaffold training fold is invalid")
        seen.extend(held)
    if len(seen) != len(set(seen)) or set(seen) != set(positions):
        raise ValueError("OOF folds do not partition all 1126 training structures")
    projection = original["consistency_projection"]
    common = projection["common_oof_indices"]
    if projection["parameters"] != ["total_clearance", "vss", "mrt"]:
        raise ValueError("Unknown original consistency parameter order")
    center = np.asarray(projection["corrected_oof_log10"] if projection["enabled"] else projection["raw_oof_log10"], dtype=float)
    raw = {}
    for parameter in projection["parameters"]:
        raw[parameter] = {}
        for f in folds:
            record = original["consistency_oof"][parameter]["folds"][str(f["fold"])]
            if (record["train_indices"] != [i for i in f["train_indices"] if i in labels[parameter]] or
                    record["validation_indices"] != [i for i in f["validation_indices"] if i in labels[parameter]]):
                raise ValueError("Stored parameter OOF membership changed")
            for r in record["out_of_fold"]:
                if r["observed_log10"] != labels[parameter][r["row_index"]]:
                    raise ValueError("Stored parameter OOF training label changed")
                raw[parameter][r["row_index"]] = r["predicted_log10"]
    if set(common) != set.intersection(*(set(raw[p]) for p in projection["parameters"])):
        raise ValueError("Original shared projection common rows changed")
    replay = np.array([[raw[p][i] for p in projection["parameters"]] for i in common])
    fold_for = {i:f["fold"] for f in folds for i in f["validation_indices"]}
    if projection["enabled"]:
        for f in projection["folds"]:
            mask = np.array([fold_for[i] == f["fold"] for i in common])
            replay[mask] = rich.project_consistency(replay[mask], f["residual_variances"])
    if replay.shape != center.shape or not np.allclose(replay, center, rtol=0, atol=1e-12):
        raise ValueError("Stored projected CL/Vss OOF center does not reproduce")
    emit("building_features", planned_fits=12, completed_fits=len(state["fits"]))
    x, features = rich.rich_features([r["identity_smiles"] for r in train_rows])
    if features["descriptor_names"] != original["features"]["descriptor_names"]:
        raise ValueError("Frozen descriptor definition changed")
    half = {}
    parameter = "terminal_half_life"
    for f in folds:
        fit = [i for i in f["train_indices"] if i in labels[parameter]]
        held = [i for i in f["validation_indices"] if i in labels[parameter]]
        members = []
        for method in shared.METHODS[parameter]:
            key = f"{f['fold']}/{method}"
            metadata = {"fold":f["fold"], "method":method, "train_indices":fit, "held_indices":held}
            if key in state["fits"]:
                record = state["fits"][key]
                if any(record[k] != v for k,v in metadata.items()): raise ValueError("Checkpoint fold mismatch")
                predicted = np.asarray(record["predicted_log10"], dtype=float)
            else:
                emit("fitting_half_oof", current_fit=key, completed_fits=len(state["fits"]))
                started = time.monotonic()
                model = shared._fit(method, x[[positions[i] for i in fit]],
                                    [labels[parameter][i] for i in fit], jobs, device)
                predicted = rich.physical_log_predictions(model.predict(x[[positions[i] for i in held]]), parameter)
                state["fits"][key] = {**metadata, "predicted_log10":predicted.tolist(),
                    "elapsed_seconds":time.monotonic()-started}
                emit("half_oof_fit_complete", completed_fits=len(state["fits"]))
            if predicted.shape != (len(held),) or not np.isfinite(predicted).all():
                raise ValueError("Invalid half-life OOF prediction")
            members.append(predicted)
        half.update(zip(held, np.mean(members, axis=0).tolist()))
    if set(half) != set(labels[parameter]): raise ValueError("Incomplete half-life OOF coverage")
    cl_center = dict(zip(common, center[:,0])); vss_center = dict(zip(common, center[:,1]))
    coefficients, stats, residual_records = {}, {}, {}
    for endpoint, predicted in (("vss", vss_center), ("terminal_half_life", half)):
        indices = sorted(set(common) & set(predicted) & set(labels[endpoint]))
        yc = np.array([labels["total_clearance"][i] for i in indices]); pc = np.array([cl_center[i] for i in indices])
        yz = np.array([labels[endpoint][i] for i in indices]); pz = np.array([predicted[i] for i in indices])
        report = crossfit_correction(yc, pc, yz, pz, [fold_for[i] for i in indices])
        coefficients[endpoint] = report["slope"]
        stats[endpoint] = report
        residual_records[endpoint] = {"indices":indices, "fold_ids":[fold_for[i] for i in indices],
            "cl_observed_log10":yc.tolist(), "cl_center_log10":pc.tolist(),
            "endpoint_observed_log10":yz.tolist(), "endpoint_center_log10":pz.tolist(),
            "omitted_training_indices":sorted(set(positions)-set(indices)),
            "omission_reason":"Only rows with verified original projected CL center and available endpoint label enter this residual relation"}
    half_indices = sorted(half)
    half_truth = np.array([labels[parameter][i] for i in half_indices]); half_pred = np.array([half[i] for i in half_indices])
    q80 = float(np.quantile(np.abs(half_truth-half_pred)*math.log(10.), .8))
    previous._verify(root, refs); previous._verify(root, code)
    result = {"version":VERSION, "created_at":previous._now(), "input_fingerprint":fingerprint,
        "coefficients":coefficients, "shrinkage":SHRINKAGE, "ridge_per_row":RIDGE_PER_ROW,
        "half_log_uncertainty_q80":q80, "half_uncertainty_log_base":"natural",
        "half_uncertainty_interpretation":"Training OOF absolute log-error quantile; not a calibrated clinical confidence interval",
        "original_target_priors":deepcopy(old["parameters"]), "prior_units":deepcopy(old["units"]),
        "source_refs":refs, "code_pins":code, "trainingstats":stats,
        "audit":{"target_labels_used":False, "target_observation_files_opened":False,
            "training_structure_overlap":False, "training_generic_scaffold_overlap":False,
            "clinical_results_used_for_selection":False, "training_structures":len(train_rows),
            "cl_training_labels":len(labels["total_clearance"]), "projected_common_oof_rows":len(common),
            "half_training_labels":len(half), "target_count":len(cohort), "new_oof_fits":len(state["fits"]),
            "training_indices":sorted(positions), "residual_records":residual_records,
            "half_oof":{"indices":half_indices, "fold_ids":[fold_for[i] for i in half_indices],
                "observed_log10":half_truth.tolist(), "predicted_log10":half_pred.tolist(),
                "metrics":_metrics(half_truth, half_pred)},
            "base_projection_limitation":projection["selection_interpretation"],
            "second_stage_limitation":"Other folds' first-stage fits can include this held fold; residual crossfit is a development diagnostic, not nested independent evaluation",
            "selection":"No clinical or endpoint-specific acceptance filtering; report both fixed corrections"},
        "elapsed_seconds":time.monotonic()-begin}
    result["result_sha256"] = shared._digest(result)
    rich._atomic_json(result_path, result)
    emit("complete", result_path=str(result_path.relative_to(root)), result_sha256=shared._sha(result_path),
         elapsed_seconds=time.monotonic()-begin, coefficients=coefficients, half_log_uncertainty_q80=q80)
    return result


def _positive(value, name):
    if isinstance(value, bool): raise ValueError("Invalid " + name)
    try: value = float(value)
    except (TypeError, ValueError) as exc: raise ValueError("Invalid " + name) from exc
    if not math.isfinite(value) or value <= 0: raise ValueError("Expected positive finite " + name)
    return value


def cl_ml_min_kg(row, weight_kg):
    """Convert explicitly declared systemic CL units; never infer source weight."""
    value = _positive(row["value"], "CL"); weight = _positive(weight_kg, "weight")
    unit = str(row.get("unit", "")).lower().replace(" ", "")
    factors = {"ml/min/kg":1., "ml/min":1./weight, "l/h":1000./60./weight,
        "l/h/kg":1000./60., "l/min":1000./weight, "l/min/kg":1000.}
    if unit not in factors: raise ValueError("Unsupported systemic CL unit: " + unit)
    return value*factors[unit]


def condition_sidecars(request, sidecars, artifact, receipt):
    """Condition predicted-only sidecars; caller validates frozen source files.

    Legacy accepted construction rows need not have an explicit independence
    flag. They must remain bound to the runner's verified evidence policy, and
    explicit target roles, negative flags and forbidden sources are rejected.
    """
    result = deepcopy(sidecars)
    audit = {"version":VERSION, "applied":False, "changed_parameters":[], "changes":[],
        "cl_innovation_log10":None, "request_modified":False, "clinical_observations_used":False}
    if artifact.get("version") != VERSION or artifact.get("shrinkage") != SHRINKAGE:
        raise ValueError("Unsupported frozen conditioning artifact")
    rows = [r for r in request.get("evidence", []) if r.get("parameter") == "total_clearance"]
    if len(rows) != 1: raise ValueError("Exactly one systemic total CL record is required")
    row = rows[0]
    if row.get("basis") not in {"measured", "derived"}:
        audit["reason"] = "no_independent_systemic_CL"; return result, audit
    context = row.get("context") or {}; source = row.get("source") or {}
    if (row.get("compound_id") != request["compound_id"] or context.get("species") != "human"
            or context.get("matrix") != "plasma" or context.get("route") not in {"iv", "intravenous"}
            or context.get("independent_of_target") is False
            or context.get("study_role") in {"target", "evaluation", "test", "validation", "heldout"}):
        raise ValueError("Unqualified independent systemic CL context")
    for record in (source, receipt):
        if (not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"]
                or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) is None):
            raise ValueError("Bound source receipt required")
    forbidden = request.get("forbidden_sources", [])
    if any(source.get("path") == r.get("path") or source.get("sha256") == r.get("sha256") for r in forbidden):
        raise ValueError("Target observation source cannot condition the prior")
    cid = request["compound_id"]; center = artifact["original_target_priors"][cid]
    innovation = math.log10(cl_ml_min_kg(row, request["human"]["weight_kg"])) - math.log10(_positive(center["total_clearance"], "original CL"))
    audit.update(cl_innovation_log10=innovation, source_CL=deepcopy(source),
        reference_CL_ml_min_kg=center["total_clearance"], shrinkage=SHRINKAGE,
        independence_basis="Frozen caller-verified construction evidence, explicit target exclusions checked")
    for key, parameter in (("vss", "vss"), ("half", "terminal_half_life")):
        records = result.get(key, [])
        if not isinstance(records, list): raise ValueError("Expected sidecar list: " + key)
        for rec in records:
            if rec.get("basis") != "predicted": continue
            if rec.get("parameter") != parameter or rec.get("compound_id") != cid:
                raise ValueError("Predicted sidecar identity mismatch")
            # An observed/derived input of the same endpoint wins downstream;
            # preserve all lower-priority records as well to avoid fake changes.
            if any(r.get("basis") in {"measured", "derived"} for r in records): continue
            unit = str(rec.get("unit", "")).lower().replace(" ", "")
            weight = _positive(request["human"]["weight_kg"], "weight")
            factor = 1. if unit == ("l/kg" if parameter == "vss" else "h") else weight if parameter == "vss" and unit == "l" else None
            if factor is None: raise ValueError("Unsupported predicted sidecar unit")
            original = _positive(center[parameter], "original " + parameter) * factor
            if not math.isclose(float(rec["value"]), original, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Predicted sidecar differs from the frozen original center")
            beta = float(artifact["coefficients"][parameter])
            if not math.isfinite(beta): raise ValueError("Invalid frozen residual slope")
            shift = SHRINKAGE*beta*innovation
            if shift == 0.: continue
            updated = _positive(10. ** (math.log10(original) + shift), parameter)
            if updated == rec["value"]: continue
            before = deepcopy(rec)
            rec.update(value=updated, source={**receipt, "location":f"{cid}/{parameter}: fixed training-residual CL conditioning"},
                prediction_method={**rec.get("prediction_method", {}), "name":"evidence_conditioned_shared_prior", "version":VERSION},
                notes="Predicted prior conditioned on independent systemic CL; measured CL and all measured sidecars remain unchanged")
            audit["changed_parameters"].append(parameter)
            audit["changes"].append({"parameter":parameter, "before":before["value"], "after":updated,
                "unit":rec["unit"], "slope":beta, "original_source":before.get("source")})
    audit["applied"] = bool(audit["changes"])
    audit["reason"] = "predicted_sidecars_conditioned" if audit["applied"] else "no_predicted_prior_change"
    return result, audit
