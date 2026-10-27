"""CL-only refinement selected on the frozen training scaffold OOF split.

No clinical concentration, target parameter label, or PBPK score is an input.
The model-selection OOF scores are development diagnostics, not independent
post-selection performance. The retained consistency projection also inherits
its original non-nested, training-only weight-estimation limitation.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from . import clinical_shared_prior as shared
from . import rich_parameter_prior as rich

VERSION = "training-selected-clearance-refinement-1"
SOURCE_AUDIT = "experiments/blueprint_design/clinical_pbpk_comparison_20261004/parameter_model/predictions_audit.json"
OUTPUT = "experiments/blueprint_design/blueprint_cl_refinement_20261004/prior"
CANDIDATES = ("retained_projected_cl", "rf_cl", "retained_rf_log_mean",
              "retained_rf_auxiliary_log_median", "xgb_depth3_log_l1")
CODE_FILES = (
    "src/pbpk_kb/blueprint_learning/clearance_refinement.py",
    "scripts/train_clearance_refinement.py",
    "src/pbpk_kb/blueprint_learning/clinical_shared_prior.py",
    "src/pbpk_kb/blueprint_learning/rich_parameter_prior.py",
    "src/pbpk_kb/blueprint_learning/adaptive_parameter_prior.py",
    "src/pbpk_kb/blueprints/pk_parameter_learning.py",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic(path, value):
    rich._atomic_json(path, value)


def _verify(root, pins):
    for name, expected in pins.items():
        if shared._sha(shared._inside(name, root, exists=True)) != expected:
            raise ValueError("Pinned training dependency changed: " + name)


def candidate_predictions(retained, rf, auxiliary, l1):
    """All arguments are log10 CL in mL/min/kg, in identical row order."""
    arrays = [np.asarray(x, dtype=float) for x in (retained, rf, auxiliary, l1)]
    if (not arrays[0].size or arrays[0].ndim != 1
            or any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays)):
        raise ValueError("Candidate OOF arrays must be finite, nonempty and aligned")
    return dict(zip(CANDIDATES, (arrays[0], arrays[1],
        (arrays[0] + arrays[1]) / 2., np.median(arrays[:3], axis=0), arrays[3])))


def select_candidate(truth, candidates, fold_ids):
    """Predeclared CL-only selection; a tie retains the original predictor."""
    truth, folds = np.asarray(truth, dtype=float), np.asarray(fold_ids)
    if (set(candidates) != set(CANDIDATES) or truth.ndim != 1 or not truth.size
            or folds.shape != truth.shape or not np.isfinite(truth).all()):
        raise ValueError("Complete training labels, fold identities and five candidates required")
    scores = {}
    for name in CANDIDATES:
        predicted = np.asarray(candidates[name], dtype=float)
        if predicted.shape != truth.shape or not np.isfinite(predicted).all():
            raise ValueError("Candidates must cover exactly the same CL-labeled training rows")
        error = np.abs(predicted - truth)
        fe = 10. ** error
        scores[name] = {"n": len(error), "log10_mae": float(error.mean()),
            "gmfe": float(10. ** error.mean()), "fold_error_p90": float(np.quantile(fe, .9)),
            "fold_error_p95": float(np.quantile(fe, .95)), "fold_error_max": float(fe.max()),
            "threshold_comparison": "<", "within_counts": {str(t): int(np.sum(fe < t)) for t in (2, 3, 5)},
            "fold_log10_mae": {str(f): float(error[folds == f].mean()) for f in sorted(set(folds.tolist()))}}
    selected = min(CANDIDATES, key=lambda name: (scores[name]["log10_mae"], CANDIDATES.index(name)))
    if scores[selected]["log10_mae"] >= scores[CANDIDATES[0]]["log10_mae"] - 1e-12:
        selected = CANDIDATES[0]
    return {"selected": selected, "criterion": "minimum CL mean absolute log10 OOF error",
        "tie_tolerance": 1e-12, "candidate_order": list(CANDIDATES), "training_only": True,
        "independent_post_selection_evaluation": False, "scores": scores}


def make_l1_estimator(n_jobs, device):
    """Same depth-3 recipe and seed; only the scalar objective is changed."""
    model = rich.make_estimator("rich_xgb_depth3", shared.SEED, n_jobs, device)
    model.set_params(model__objective="reg:absoluteerror")
    return model


def _read_source(root):
    path = shared._inside(SOURCE_AUDIT, root, exists=True)
    data = json.loads(path.read_text())
    if data["result_sha256"] != shared._digest({k: v for k, v in data.items() if k != "result_sha256"}):
        raise ValueError("Original shared-prior audit checksum mismatch")
    original = data["audit"]
    model = shared._inside(data["model_path"], root, exists=True)
    source = shared._inside(original["frozen_input"]["source_path"], root, exists=True)
    refs = {str(path.relative_to(root)): shared._sha(path),
            str(model.relative_to(root)): data["model_sha256"],
            str(source.relative_to(root)): original["frozen_input"]["source_sha256"]}
    _verify(root, refs)
    _verify(root, original["frozen_input"]["code_sha256"])
    for flag in ("target_labels_used", "target_observation_files_opened",
                 "training_structure_overlap", "training_generic_scaffold_overlap"):
        if original[flag] is not False:
            raise ValueError("Original training isolation failed: " + flag)
    return data, model, source, refs


def fit_clearance_refinement(project_root, output_dir=OUTPUT, n_jobs=8, device="auto", progress=None):
    """Return frozen structure-only CL predictions; fit no native PBPK model."""
    root = Path(project_root).resolve(strict=True)
    output = shared._inside(output_dir, root)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or not 1 <= n_jobs <= 8:
        raise ValueError("n_jobs must be an integer in [1,8]")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This project CL trainer is already active") from error
        return _fit_locked(root, output, n_jobs, device, progress)


def _fit_locked(root, output, n_jobs, preference, progress):
    started = time.monotonic()
    old, model_path, wheel, refs = _read_source(root)
    original = old["audit"]
    device = rich.choose_device(preference)
    code = {name: shared._sha(shared._inside(name, root, exists=True)) for name in CODE_FILES}
    config = {"version": VERSION, "source_refs": refs, "code_pins": code,
        "n_jobs": n_jobs, "device": device, "candidates": list(CANDIDATES),
        "l1_recipe": {**rich.RECIPES["rich_xgb_depth3"], "objective": "reg:absoluteerror", "seed": shared.SEED},
        "rf_recipe": shared.RF_RECIPE, "expected_training_structures": 1126,
        "expected_cl_labels": 1124, "folds": 3, "cohort": original["targets"]}
    fingerprint = shared._digest(config)
    state_path = output / "training_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "input_fingerprint": fingerprint, "config": config, "started_at": _now(), "fits": {}}
    if state["input_fingerprint"] != fingerprint:
        raise ValueError("Training configuration/source changed; refusing mixed-version resume")
    result_path = output / "predictions.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("result_sha256") != shared._digest({k: v for k, v in result.items() if k != "result_sha256"}):
            raise ValueError("CL prediction result checksum mismatch")
        if result["input_fingerprint"] != fingerprint:
            raise ValueError("CL prediction input mismatch")
        if result.get("model_receipt"):
            _verify(root, {result["model_receipt"]["path"]: result["model_receipt"]["sha256"]})
        return result

    def emit(stage, **updates):
        state.update(stage=stage, updated_at=_now(), pid=os.getpid(), **updates)
        _atomic(state_path, state)
        if progress:
            progress({"stage": stage, **updates})

    emit("verifying_training_membership")
    cohort = shared._cohort([{"compound_id": cid, "smiles": r["input_smiles"]}
                             for cid, r in original["targets"].items()])
    rows, labels, membership = shared._training_data(wheel, cohort)
    train = [r for r in rows if r["status"] == "train"]
    if rows != original["rows"] or len(train) != 1126 or len(labels["total_clearance"]) != 1124:
        raise ValueError("Original training membership or CL label count changed")
    for p, values in labels.items():
        if sorted(values) != original["parameter_training_indices"][p]:
            raise ValueError("Original endpoint label membership changed: " + p)
    positions = {r["index"]: i for i, r in enumerate(train)}
    folds = original["folds"]
    check = list(GroupKFold(3).split(np.zeros(len(train)), groups=[r["scaffold_group"] for r in train]))
    for f, (fit_pos, held_pos) in zip(folds, check):
        if ([train[i]["index"] for i in fit_pos] != f["train_indices"]
                or [train[i]["index"] for i in held_pos] != f["validation_indices"]):
            raise ValueError("Frozen scaffold fold membership changed")
    emit("building_features")
    x_all, feature_audit = rich.rich_features([r["identity_smiles"] for r in train] +
                                             [r["identity_smiles"] for r in cohort.values()])
    if feature_audit["descriptor_names"] != original["features"]["descriptor_names"]:
        raise ValueError("Feature definitions differ from frozen shared prior")
    x, target_x = x_all[:len(train)], x_all[len(train):]
    oof = {}
    for p in rich.CONSISTENCY_PARAMETERS:
        oof[p] = {}
        for f in folds:
            record = original["consistency_oof"][p]["folds"][str(f["fold"])]
            expected_fit = [i for i in f["train_indices"] if i in labels[p]]
            expected_held = [i for i in f["validation_indices"] if i in labels[p]]
            if record["train_indices"] != expected_fit or record["validation_indices"] != expected_held:
                raise ValueError("Stored auxiliary OOF split differs from original split")
            for r in record["out_of_fold"]:
                if r["observed_log10"] != labels[p][r["row_index"]]:
                    raise ValueError("Stored training OOF label mismatch")
                oof[p][r["row_index"]] = r["predicted_log10"]

    def fit_predict(key, p, f, held, kind):
        fit = [i for i in f["train_indices"] if i in labels[p]]
        if set(fit) & set(held):
            raise ValueError("OOF training/validation overlap")
        metadata = {"parameter": p, "fold": f["fold"], "train_indices": fit,
                    "held_indices": held, "kind": kind}
        if key in state["fits"]:
            stored = state["fits"][key]
            if any(stored[k] != v for k, v in metadata.items()):
                raise ValueError("OOF fit checkpoint membership mismatch")
            predicted = np.asarray(stored["predicted_log10"], dtype=float)
            if predicted.shape != (len(held),) or not np.isfinite(predicted).all():
                raise ValueError("Invalid OOF checkpoint predictions")
            return predicted
        emit("fitting_oof", current_fit=key, completed_fits=len(state["fits"]))
        if kind == "rf_cl":
            estimator = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", RandomForestRegressor(**shared.RF_RECIPE, n_jobs=n_jobs))])
        elif kind == "xgb_depth3_log_l1":
            estimator = make_l1_estimator(n_jobs, device)
        else:
            estimator = rich.make_estimator(kind, shared.SEED, n_jobs, device)
        begin = time.monotonic()
        estimator.fit(x[[positions[i] for i in fit]], [labels[p][i] for i in fit])
        rich._verify_fitted_device(estimator, "rich_xgb_depth3" if kind == "xgb_depth3_log_l1" else kind, device)
        predicted = np.asarray(estimator.predict(x[[positions[i] for i in held]]), dtype=float)
        if predicted.shape != (len(held),) or not np.isfinite(predicted).all():
            raise ValueError("Invalid training OOF predictions")
        state["fits"][key] = {**metadata, "predicted_log10": predicted.tolist(),
                              "elapsed_seconds": time.monotonic() - begin}
        emit("oof_fit_complete", completed_fits=len(state["fits"]))
        return predicted

    for p in ("vss", "mrt"):
        for f in folds:
            held = [i for i in f["validation_indices"] if i in labels["total_clearance"] and i not in oof[p]]
            if not held:
                continue
            predictions = [fit_predict(f"aux/{p}/{f['fold']}/{method}", p, f, held, method)
                           for method in shared.METHODS[p]]
            oof[p].update(zip(held, np.mean(predictions, axis=0).tolist()))
    rf, l1, retained, aux, fold_by_index = {}, {}, {}, {}, {}
    projection = original["consistency_projection"]
    for f in folds:
        held = [i for i in f["validation_indices"] if i in labels["total_clearance"]]
        rf.update(zip(held, fit_predict(f"rf/{f['fold']}", "total_clearance", f, held, "rf_cl").tolist()))
        l1.update(zip(held, fit_predict(f"l1/{f['fold']}", "total_clearance", f, held, "xgb_depth3_log_l1").tolist()))
        raw = np.array([[oof[p][i] for p in rich.CONSISTENCY_PARAMETERS] for i in held])
        corrected = raw
        if projection["enabled"]:
            weights = next(r["residual_variances"] for r in projection["folds"] if r["fold"] == f["fold"])
            corrected = rich.project_consistency(raw, weights)
        retained.update(zip(held, corrected[:, 0].tolist()))
        aux.update(zip(held, (raw[:, 1] - raw[:, 2] - math.log10(.06)).tolist()))
        fold_by_index.update({i: f["fold"] for i in held})
    if projection["enabled"]:
        saved = np.asarray(projection["corrected_oof_log10"])[:, 0]
        replay = np.array([retained[i] for i in projection["common_oof_indices"]])
        if not np.allclose(saved, replay, rtol=0, atol=1e-12):
            raise ValueError("Retained projected OOF predictions do not reproduce original audit")
    indices = sorted(labels["total_clearance"])
    candidate_oof = candidate_predictions(*[[mapping[i] for i in indices] for mapping in (retained, rf, aux, l1)])
    selection = select_candidate([labels["total_clearance"][i] for i in indices], candidate_oof,
                                 [fold_by_index[i] for i in indices])
    if state.get("selection") and state["selection"] != selection:
        raise ValueError("Frozen training-only selection changed on resume")
    emit("selection_locked", selection=selection, selection_locked_at=state.get("selection_locked_at", _now()),
         target_predictions_started=False)
    _verify(root, refs); _verify(root, code)
    model_receipt = None
    chosen = selection["selected"]
    if chosen == "xgb_depth3_log_l1":
        emit("fitting_selected_final_model", target_predictions_started=True)
        estimator = make_l1_estimator(n_jobs, device)
        estimator.fit(x[[positions[i] for i in indices]], [labels["total_clearance"][i] for i in indices])
        rich._verify_fitted_device(estimator, "rich_xgb_depth3", device)
        target_log = np.asarray(estimator.predict(target_x), dtype=float)
        path = output / "model.joblib"
        rich._atomic_model(path, {"version": VERSION, "model": estimator, "unit": "mL/min/kg",
                                 "input_fingerprint": fingerprint, "selection": selection})
        model_receipt = {"path": str(path.relative_to(root)), "sha256": shared._sha(path)}
    elif chosen == "retained_projected_cl":
        target_log = np.array([math.log10(old["parameters"][cid]["total_clearance"]) for cid in cohort])
    elif chosen == "rf_cl":
        target_log = np.array([math.log10(old["rf_parameters"][cid]["total_clearance"]) for cid in cohort])
    else:
        first = np.array([math.log10(old["parameters"][cid]["total_clearance"]) for cid in cohort])
        second = np.array([math.log10(old["rf_parameters"][cid]["total_clearance"]) for cid in cohort])
        if chosen == "retained_rf_log_mean":
            target_log = (first + second) / 2.
        else:
            emit("predicting_selected_auxiliary", target_predictions_started=True)
            bundle = joblib.load(model_path)
            raw_target = {p: np.mean([bundle["models"][p][method].predict(target_x)
                          for method in shared.METHODS[p]], axis=0) for p in ("vss", "mrt")}
            third = raw_target["vss"] - raw_target["mrt"] - math.log10(.06)
            target_log = np.median([first, second, third], axis=0)
    values = 10. ** target_log
    if values.shape != (len(cohort),) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Invalid final target CL predictions")
    result = {"version": VERSION, "input_fingerprint": fingerprint, "unit": "mL/min/kg",
        "predictions": dict(zip(cohort, values.tolist())), "selected_using_training_only": True,
        "selection": selection, "oof_metrics": selection["scores"], "model_receipt": model_receipt,
        "source_refs": refs, "code_pins": code, "created_at": _now(), "elapsed_seconds": time.monotonic()-started,
        "audit": {"target_labels_used": False, "target_observation_files_opened": False,
            "training_structure_overlap": False, "training_generic_scaffold_overlap": False,
            "clinical_results_used_for_selection": False, "native_pbpk_runs": 0,
            "training_structures": len(train), "cl_training_labels": len(indices), "target_count": len(cohort),
            "target_identity_exclusion_count": membership["target_identity_exclusion_count"],
            "target_generic_scaffold_exclusion_count": membership["target_generic_scaffold_exclusion_count"],
            "training_indices": indices, "oof_fold_ids": [fold_by_index[i] for i in indices],
            "observed_training_log10": [labels["total_clearance"][i] for i in indices],
            "candidate_oof_log10": {k: v.tolist() for k,v in candidate_oof.items()},
            "selection_locked_at": state["selection_locked_at"], "descriptor_dimensions": feature_audit["dimensions"],
            "consistency_projection_inherited_limitation": projection.get("selection_interpretation"),
            "auxiliary_missing_label_predictions": {p: sum(i not in {r['row_index'] for f in original['consistency_oof'][p]['folds'].values() for r in f['out_of_fold']} for i in indices) for p in ('vss','mrt')}},
        "scope": "CL-only missing-parameter ablation; measured CL, fu, Vss, half-life and PBPK core unchanged",
        "consistency": "New CL need not satisfy the old prior projection; native PBPK/NCA derives study-specific MRT",
        "interpretation": "OOF scores select a development candidate; these are not independent post-selection or clinical performance"}
    _verify(root, refs); _verify(root, code)
    result["result_sha256"] = shared._digest(result)
    _atomic(result_path, result)
    emit("complete", result_path=str(result_path.relative_to(root)), result_sha256=shared._sha(result_path),
         selected=chosen, elapsed_seconds=time.monotonic()-started, target_predictions_started=True)
    return result
