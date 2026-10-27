"""Fixed TabPFNv2 CL context with a permanently held-out scaffold CQR set.

The other four parameters are copied from the frozen expanded prior. No target
clinical files are inputs. CQR coverage under scaffold shift is empirical, not
an IID or clinical guarantee. No checkpoint weights are fine-tuned.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

from . import clinical_shared_prior as shared
from . import expanded_selected_prior as expanded
from . import rich_parameter_prior as rich

VERSION = "foundation-clearance-prior-1"
METHOD_NAME = "tabpfn_v2_conformal_CL"
OUTPUT = "experiments/blueprint_design/foundation_prior_20261007/prior"
SOURCE_PRIOR = expanded.OUTPUT + "/predictions.json"
SOURCE_AUDIT = expanded.OUTPUT + "/shared/predictions_audit.json"
TARGETS = expanded.TARGETS
CHECKPOINT = "artifacts/pk_predictors/tabpfn_v2/tabpfn-v2-regressor.ckpt"
CHECKPOINT_SOURCE = "artifacts/pk_predictors/tabpfn_v2/source.json"
CHECKPOINT_REVISION = "4972a65a1b30806315c6f92499959ffbfc69a673"
CHECKPOINT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
CHECKPOINT_BYTES = 44390977
EXPECTED_TABPFN_VERSION = "2.1.3"
SEED = 20261007
COVERAGE = .90
ENSEMBLES = 4
MAX_BYTES = 10 * 1024 ** 3
PARAMETER = "total_clearance"
CODE_FILES = (
    "src/pbpk_kb/blueprint_learning/foundation_clearance_prior.py",
    "scripts/train_foundation_clearance_prior.py",
    "src/pbpk_kb/blueprint_learning/clinical_shared_prior.py",
    "src/pbpk_kb/blueprint_learning/expanded_selected_prior.py",
    "src/pbpk_kb/blueprint_learning/rich_parameter_prior.py",
    "src/pbpk_kb/blueprints/pk_parameter_learning.py",
)


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _receipt(root, path):
    path = shared._inside(path, root, exists=True)
    return {"path": str(path.relative_to(root)), "sha256": shared._sha(path)}


def _checked_json(root, path, *, digest=True):
    receipt = _receipt(root, path)
    value = json.loads((root / receipt["path"]).read_text())
    if digest and value.get("result_sha256") != shared._digest(
            {k: v for k, v in value.items() if k != "result_sha256"}):
        raise ValueError("Source result checksum mismatch: " + receipt["path"])
    return value, receipt


def _verify_pins(root, pins):
    for name, digest in pins.items():
        if _receipt(root, name)["sha256"] != digest:
            raise ValueError("Frozen source/code/checkpoint changed: " + name)


def checkpoint_sources(root, checkpoint):
    """Verify the pinned official v2 asset and its local provenance bundle."""
    metadata, source_receipt = _checked_json(root, CHECKPOINT_SOURCE, digest=False)
    if (metadata.get("repository") != "https://huggingface.co/Prior-Labs/TabPFN-v2-reg"
            or metadata.get("revision") != CHECKPOINT_REVISION
            or metadata.get("package") != "tabpfn==" + EXPECTED_TABPFN_VERSION):
        raise ValueError("Checkpoint source repository, revision or package differs from the fixed protocol")
    actual = _receipt(root, checkpoint)
    declared = metadata.get("checkpoint", {})
    if (declared.get("path") != actual["path"]
            or declared.get("sha256") != CHECKPOINT_SHA256
            or actual["sha256"] != CHECKPOINT_SHA256
            or declared.get("size_bytes") != CHECKPOINT_BYTES
            or (root / actual["path"]).stat().st_size != CHECKPOINT_BYTES):
        raise ValueError("Checkpoint path, SHA256 or size does not match the fixed official asset")
    files = metadata.get("source_files", {})
    if set(files) != {"LICENSE.txt", "config.json"}:
        raise ValueError("Checkpoint source bundle must contain LICENSE.txt and config.json")
    refs = {source_receipt["path"]: source_receipt["sha256"], actual["path"]: actual["sha256"]}
    source_directory = Path(CHECKPOINT_SOURCE).parent
    for name, digest in files.items():
        receipt = _receipt(root, source_directory / name)
        if receipt["sha256"] != digest:
            raise ValueError("Checkpoint source file checksum mismatch: " + name)
        refs[receipt["path"]] = receipt["sha256"]
    return actual, refs


def _check_budget(output):
    size = 0
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in the training output")
        if path.is_file():
            size += path.stat().st_size
    if size > MAX_BYTES:
        raise RuntimeError("Training output exceeds 10 GiB; no further work submitted")
    return size


def scaffold_split(rows, eligible_indices):
    """Select fold 4 of a fixed 5-fold split for calibration; 3-fold the rest."""
    rowmap = {r["index"]: r for r in rows}
    indices = sorted(eligible_indices)
    if len(indices) != len(set(indices)) or any(rowmap[i]["status"] != "train" for i in indices):
        raise ValueError("Training indices must be unique eligible rows")
    groups = [rowmap[i]["scaffold_group"] for i in indices]
    if len(set(groups)) < 5:
        raise ValueError("At least five eligible training scaffolds are required")
    outer = list(GroupKFold(5).split(indices, groups=groups))
    fit_pos, cal_pos = outer[4]
    fit = [indices[i] for i in fit_pos]
    calibration = [indices[i] for i in cal_pos]
    fit_groups = [rowmap[i]["scaffold_group"] for i in fit]
    folds = [{"fold": f, "train_indices": [fit[i] for i in a],
              "validation_indices": [fit[i] for i in b]}
             for f, (a, b) in enumerate(GroupKFold(3).split(fit, groups=fit_groups))]
    cal_groups = {rowmap[i]["scaffold_group"] for i in calibration}
    for fold in folds:
        a, b = set(fold["train_indices"]), set(fold["validation_indices"])
        if a & b or (a | b) != set(fit) or (a | b) & set(calibration):
            raise ValueError("Calibration/OOF membership is not disjoint")
        ga = {rowmap[i]["scaffold_group"] for i in a}
        gb = {rowmap[i]["scaffold_group"] for i in b}
        if ga & gb or (ga | gb) & cal_groups:
            raise ValueError("Calibration/OOF scaffold leakage")
    return {"method": "GroupKFold(5), calibration fold 4; GroupKFold(3) inside remaining context",
            "group_shuffle": False, "calibration_fold": 4, "eligible_indices": indices,
            "final_context_indices": fit, "calibration_indices": calibration, "folds": folds}


def cqr_calibration(lower, upper, truth, coverage=COVERAGE):
    """Finite-sample split CQR; never shrink raw intervals or interpolate rank."""
    lower, upper, truth = [np.asarray(v, dtype=float) for v in (lower, upper, truth)]
    if not 0 < coverage < 1 or truth.ndim != 1 or not truth.size:
        raise ValueError("CQR requires nonempty one-dimensional calibration labels")
    if any(v.shape != truth.shape or not np.isfinite(v).all() for v in (lower, upper, truth)):
        raise ValueError("CQR values must be aligned and finite")
    if np.any(lower > upper):
        raise ValueError("Quantile crossing is not silently repaired")
    scores = np.maximum.reduce([lower - truth, truth - upper, np.zeros_like(truth)])
    rank = math.ceil((len(scores) + 1) * coverage)
    if rank > len(scores):
        raise ValueError("Requested CQR order statistic is unattainable with this calibration size")
    qhat = float(np.sort(scores)[rank - 1])
    return {"qhat_log10": qhat, "rank_one_based": rank, "n": len(scores),
            "nominal_coverage": coverage, "scores_log10": scores.tolist(),
            "score": "max(q05-y, y-q95, 0)", "quantile_rule": "ceil((n+1)*0.90), no interpolation"}


def interval_metrics(truth, median, lower, upper):
    y, point, lo, hi = [np.asarray(a, dtype=float) for a in (truth, median, lower, upper)]
    if y.ndim != 1 or not y.size or any(a.shape != y.shape or not np.isfinite(a).all() for a in (y, point, lo, hi)):
        raise ValueError("Metrics require aligned finite nonempty arrays")
    if np.any(lo > point) or np.any(point > hi):
        raise ValueError("Median must lie within the reported interval")
    errors = np.abs(y - point)
    return {"n": len(y), "mean_absolute_log10_error": float(np.mean(errors)),
            "gmfe": float(10 ** np.mean(errors)), "median_fold_error": float(10 ** np.median(errors)),
            "p90_fold_error": float(10 ** np.quantile(errors, .9)),
            "within_twofold_n": int(np.sum(errors <= math.log10(2))),
            "coverage_n": int(np.sum((y >= lo) & (y <= hi))),
            "coverage": float(np.mean((y >= lo) & (y <= hi))),
            "mean_log10_width": float(np.mean(hi - lo)),
            "median_upper_lower_ratio": float(10 ** np.median(hi - lo)),
            "nominal_coverage": COVERAGE, "clinical_coverage_guaranteed": False}


def molecular_features(identities):
    features, audit = rich.rich_features(identities)
    x = features[:, 1024:]
    if features.shape[1] != 1024 + 167 + len(audit["descriptor_names"]) or not 1 <= x.shape[1] < 500:
        raise ValueError("Expected MACCS167 + installed RDKit 2D descriptors, fewer than 500 columns")
    return x, {**audit, "dimensions": x.shape[1], "morgan_bits": 0,
               "feature_order": "MACCS167 followed by RDKit descriptor_names",
               "selection": "rich_features[:,1024:]", "animal_features": False,
               "imputation": "fit-context-only median; all-missing columns retained"}


def _software():
    versions = {name: importlib.metadata.version(name)
                for name in ("tabpfn", "numpy", "scipy", "scikit-learn", "torch", "rdkit")}
    if versions["tabpfn"] != EXPECTED_TABPFN_VERSION:
        raise ValueError("This frozen protocol requires tabpfn==" + EXPECTED_TABPFN_VERSION)
    return versions


def _make_estimator(checkpoint, jobs):
    # Explicit existing checkpoint prevents the library's default model lookup
    # and download path. This module never reads a token or a user cache.
    import torch
    from tabpfn import TabPFNRegressor
    if not Path(checkpoint).is_file() or not torch.cuda.is_available():
        raise RuntimeError("An existing project checkpoint and CUDA are required")
    torch.set_num_threads(jobs)
    return TabPFNRegressor(n_estimators=ENSEMBLES, random_state=SEED, device="cuda",
                          model_path=str(checkpoint), n_jobs=jobs)


def _predict_distribution(model, x):
    value = model.predict(x, output_type="full", quantiles=[.05, .95])
    if not isinstance(value, dict) or len(value.get("quantiles", [])) != 2:
        raise ValueError("TabPFN must return its median and the two requested quantiles")
    lo, hi = [np.asarray(v, dtype=float) for v in value["quantiles"]]
    median = np.asarray(value["median"], dtype=float)
    if any(v.shape != (len(x),) or not np.isfinite(v).all() for v in (lo, median, hi)):
        raise ValueError("Invalid TabPFN predictive distribution")
    if np.any(lo > median) or np.any(median > hi):
        raise ValueError("TabPFN quantile crossing is not silently repaired")
    return {"q05": lo, "median": median, "q95": hi}


def fit_context_predict(x, y, indices, split, target_x, checkpoint, jobs, *, factory=None):
    """Fit one context; use a separate CQR set, then predict held rows/targets."""
    positions = {i: j for j, i in enumerate(indices)}
    fit = split["train_indices"]
    cal = split["calibration_indices"]
    held = split.get("validation_indices", [])
    if not fit or not cal or (set(fit) & set(cal)) or (set(fit) & set(held)) or (set(cal) & set(held)):
        raise ValueError("Fit, calibration and heldout rows must be disjoint")
    ix = lambda values: [positions[i] for i in values]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_x = imputer.fit_transform(x[ix(fit)])
    # Predict all unlabeled inputs together; cal/held outcomes are read only
    # after .fit and .predict have finished, never passed to an estimator.
    query = np.vstack([x[ix(cal)], x[ix(held)], target_x])
    model = (factory or _make_estimator)(checkpoint, jobs)
    try:
        model.fit(fit_x, np.asarray([y[i] for i in fit], dtype=float))
        predicted = _predict_distribution(model, imputer.transform(query))
    finally:
        del model
        gc.collect()
    ncal, nheld = len(cal), len(held)
    cp = {k: v[:ncal] for k, v in predicted.items()}
    hp = {k: v[ncal:ncal + nheld] for k, v in predicted.items()}
    tp = {k: v[ncal + nheld:] for k, v in predicted.items()}
    cqr = cqr_calibration(cp["q05"], cp["q95"], [y[i] for i in cal])
    record = {"train_indices": fit, "calibration_indices": cal, "validation_indices": held,
              "cqr": cqr, "imputer_statistics": imputer.statistics_.tolist(),
              "calibration_predictions": [{"row_index": i, "observed_log10": y[i],
                  **{k + "_log10": float(v[j]) for k, v in cp.items()}} for j, i in enumerate(cal)],
              "out_of_fold": [{"row_index": i, "observed_log10": y[i],
                  **{k + "_log10": float(v[j]) for k, v in hp.items()},
                  "lower_log10": float(hp["q05"][j] - cqr["qhat_log10"]),
                  "upper_log10": float(hp["q95"][j] + cqr["qhat_log10"])} for j, i in enumerate(held)]}
    if held:
        record["metrics"] = interval_metrics([y[i] for i in held], hp["median"],
            hp["q05"] - cqr["qhat_log10"], hp["q95"] + cqr["qhat_log10"])
    return record, tp


def _write_state(path, state):
    state["updated_at"] = _now()
    rich._atomic_json(path, {**state, "state_sha256": shared._digest(state)})


def _prepare(root, targets_path, source_prior, output, checkpoint, jobs):
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 8:
        raise ValueError("jobs must be an integer in [1,8]")
    source, source_receipt = _checked_json(root, source_prior)
    shared_result, audit_receipt = _checked_json(root, SOURCE_AUDIT)
    target_document, target_receipt = _checked_json(root, targets_path, digest=False)
    cohort = expanded.target_cohort(target_document)
    if source["targets"] != cohort or len(cohort) != 40:
        raise ValueError("Exactly the frozen forty excluded identities must be retained")
    for flag in ("target_labels_used", "target_observation_files_opened", "training_structure_overlap",
                 "training_generic_scaffold_overlap", "clinical_results_used_for_selection"):
        if source["audit"].get(flag) is not False:
            raise ValueError("Frozen source isolation audit failed: " + flag)
    wheel = _receipt(root, shared_result["audit"]["frozen_input"]["source_path"])
    if wheel["sha256"] != shared_result["audit"]["frozen_input"]["source_sha256"]:
        raise ValueError("Frozen human label source changed")
    rows, labels, data_audit = shared._training_data(root / wheel["path"], cohort)
    indices = sorted(labels[PARAMETER])
    if data_audit["train_count"] != 1047 or len(indices) != 1045:
        raise ValueError("Expected the frozen 1047-structure pool and 1045 usable CL labels")
    if rows != shared_result["audit"]["rows"] or indices != source["audit"]["human_parameter_training_indices"][PARAMETER]:
        raise ValueError("Current training membership differs from the frozen expanded prior")
    cp, checkpoint_refs = checkpoint_sources(root, checkpoint)
    refs = {r["path"]: r["sha256"] for r in (source_receipt, audit_receipt, target_receipt, wheel)}
    refs.update(checkpoint_refs)
    pins = {name: _receipt(root, name)["sha256"] for name in CODE_FILES}
    config = {"version": VERSION, "method_name": METHOD_NAME, "targets": cohort,
        "source_parameter_prior": source_receipt, "source_refs": refs, "code_pins": pins,
        "checkpoint": cp, "checkpoint_family": "TabPFN-v2-regressor",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_repository": "https://github.com/PriorLabs/TabPFN",
        "checkpoint_repository": "https://huggingface.co/Prior-Labs/TabPFN-v2-reg",
        "pretraining_domain": "Author pretrained synthetic tabular tasks; not a newly trained human PK model",
        "software": _software(), "split": scaffold_split(rows, indices),
        "n_estimators": ENSEMBLES, "random_state": SEED, "device": "cuda", "jobs": jobs,
        "coverage": COVERAGE, "quantiles": [.05, .95], "point_prediction": "median",
        "feature_slice": "rich_features[:,1024:]", "other_parameters": "Frozen expanded prior values, unchanged",
        "new_hyperparameter_selection": False, "clinical_results_used_for_selection": False,
        "planned_context_fits": 4, "gradient_training": False,
        "calibration_scope": "Never fitted in new CL estimator or imputer; other frozen parameter models predate this split",
        "max_output_bytes": MAX_BYTES}
    fingerprint = shared._digest(config)
    output.mkdir(parents=True, exist_ok=True)
    _check_budget(output)
    path = output / "training_state.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state.pop("state_sha256", None) != shared._digest(state) or state.get("input_fingerprint") != fingerprint:
            raise ValueError("Training state changed or configuration differs; refusing mixed resume")
    else:
        unrelated = [p.name for p in output.iterdir() if p.name not in {"training.lock", "training.log", "training.pid"}]
        if unrelated:
            raise ValueError("Refusing to reuse a directory with unrelated files: " + str(unrelated))
        state = {"version": VERSION, "stage": "prepared", "started_at": _now(), "config": config,
                 "input_fingerprint": fingerprint, "completed_context_fits": 0, "fold_results": {}}
        _write_state(path, state)
    return state, source, rows, labels[PARAMETER], data_audit


def prepare(root, targets_path=TARGETS, source_prior=SOURCE_PRIOR, output_dir=OUTPUT,
            checkpoint=CHECKPOINT, jobs=8):
    root = Path(root).resolve(strict=True)
    output = shared._inside(output_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(root, targets_path, source_prior, output, checkpoint, jobs)[0]


def _finite_power(log_value):
    try:
        result = float(10. ** float(log_value))
    except OverflowError as error:
        raise ValueError("CL distribution overflows its physical unit") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError("CL distribution must be finite and strictly positive")
    return result


def merge_predictions(source, target_prediction, qhat):
    parameters = deepcopy(source["parameters"])
    uncertainty = {}
    for j, cid in enumerate(source["targets"]):
        median = _finite_power(target_prediction["median"][j])
        lower = _finite_power(target_prediction["q05"][j] - qhat)
        upper = _finite_power(target_prediction["q95"][j] + qhat)
        if not lower <= median <= upper:
            raise ValueError("Invalid physical CL interval")
        parameters[cid][PARAMETER] = median
        uncertainty[cid] = {"lower": lower, "median": median, "upper": upper,
            "unit": "mL/min/kg", "nominal_coverage": COVERAGE, "calibration": "split_CQR",
            "clinical_coverage_guaranteed": False, "interval_scale": "log10 CL before exponentiation"}
    return parameters, uncertainty


def train(root, targets_path=TARGETS, source_prior=SOURCE_PRIOR, output_dir=OUTPUT,
          checkpoint=CHECKPOINT, jobs=8, *, progress=None, estimator_factory=None):
    root = Path(root).resolve(strict=True)
    output = shared._inside(output_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(root, targets_path, source_prior, output, checkpoint, jobs, progress, estimator_factory)


def _train(root, targets_path, source_prior, output, checkpoint, jobs, progress, factory):
    started = time.monotonic()
    state, source, rows, y, data_audit = _prepare(root, targets_path, source_prior, output, checkpoint, jobs)
    config = state["config"]
    def verify():
        _verify_pins(root, config["source_refs"])
        _verify_pins(root, config["code_pins"])
        _check_budget(output)
    verify()
    result_path = output / "predictions.json"
    if result_path.exists():
        result, _ = _checked_json(root, result_path)
        if result["input_fingerprint"] != state["input_fingerprint"]:
            raise ValueError("Committed predictions belong to another configuration")
        if _receipt(root, result["context_receipt"]["path"]) != result["context_receipt"]:
            raise ValueError("Frozen final context changed")
        return result
    def emit(stage, **updates):
        state.update(stage=stage, pid=os.getpid(), elapsed_seconds=time.monotonic() - started, **updates)
        _write_state(output / "training_state.json", state)
        if progress:
            progress({"stage": stage, "completed_context_fits": state["completed_context_fits"], **updates})
    try:
        indices = config["split"]["eligible_indices"]
        rowmap = {r["index"]: r for r in rows}
        target_ids = list(source["targets"])
        emit("features")
        x_all, feature_audit = molecular_features([rowmap[i]["identity_smiles"] for i in indices]
            + [source["targets"][cid]["identity_smiles"] for cid in target_ids])
        x, target_x = x_all[:len(indices)], x_all[len(indices):]
        calibration = config["split"]["calibration_indices"]
        cp = root / config["checkpoint"]["path"]
        for fold in config["split"]["folds"]:
            key = str(fold["fold"])
            if key in state["fold_results"]:
                continue
            verify()
            emit("fold_context", current_fold=fold["fold"])
            record, _ = fit_context_predict(x, y, indices, {**fold, "calibration_indices": calibration},
                np.empty((0, x.shape[1]), dtype=x.dtype), cp, jobs, factory=factory)
            state["fold_results"][key] = record
            state["completed_context_fits"] = len(state["fold_results"])
            verify()
            emit("fold_complete", current_fold=fold["fold"])
        verify()
        emit("final_context", current_fold=None)
        final_indices = config["split"]["final_context_indices"]
        final, prediction = fit_context_predict(x, y, indices,
            {"train_indices": final_indices, "calibration_indices": calibration}, target_x, cp, jobs, factory=factory)
        state["completed_context_fits"] = 4
        # Store only the reconstructible final context, not CV models or copies
        # of the author's checkpoint. Calibration labels are outside this file.
        positions = {i: j for j, i in enumerate(indices)}
        context_path = output / "context.npz"
        temporary = output / "context.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, raw_features=x[[positions[i] for i in final_indices]],
                labels_log10=np.array([y[i] for i in final_indices]), row_indices=np.array(final_indices),
                imputer_statistics=np.array(final["imputer_statistics"]))
        os.replace(temporary, context_path)
        parameters, uncertainty = merge_predictions(source, prediction, final["cqr"]["qhat_log10"])
        oof = [r for f in state["fold_results"].values() for r in f["out_of_fold"]]
        if sorted(r["row_index"] for r in oof) != sorted(final_indices):
            raise ValueError("OOF coverage differs from the final-context population")
        metrics = interval_metrics(*[[r[k] for r in oof]
            for k in ("observed_log10", "median_log10", "lower_log10", "upper_log10")])
        verify()
        result = {"version": VERSION, "algorithm_version": VERSION, "method_name": METHOD_NAME,
            "mode": "fixed_foundation_context_and_split_CQR", "created_at": _now(),
            "input_fingerprint": state["input_fingerprint"], "parameters": parameters,
            "units": deepcopy(source["units"]), "targets": deepcopy(source["targets"]),
            "uncertainty": uncertainty, "source_parameter_prior": config["source_parameter_prior"],
            "per_compound_audit": deepcopy(source["per_compound_audit"]),
            "source_refs": config["source_refs"], "code_pins": config["code_pins"],
            "checkpoint_receipt": config["checkpoint"], "context_receipt": _receipt(root, context_path),
            "audit": {"target_count": len(target_ids), "human_training_structures": data_audit["train_count"],
                "eligible_CL_labels": len(indices), "final_CL_context_count": len(final_indices),
                "CL_calibration_count": len(calibration), "human_parameter_training_indices": {
                    **deepcopy(source["audit"]["human_parameter_training_indices"]),
                    PARAMETER: final_indices}, "split": config["split"], "rows": rows,
                "feature_audit": feature_audit, "software": config["software"],
                "fold_results": state["fold_results"], "oof_metrics": metrics, "final_calibration": final,
                "context_fits": 4, "new_fit_count": 4, "gradient_training": False, "new_pretrained_weights": False,
                "clinical_files_accepted": False, "target_labels_used": False,
                "target_observation_files_opened": False, "training_structure_overlap": False,
                "training_generic_scaffold_overlap": False, "clinical_results_used_for_selection": False,
                "hyperparameters_tuned": False, "new_candidate_selection": False, "calibration_ever_in_CL_context": False,
                "calibration_used_to_select_configuration": False,
                "strict_scaffold_fit_calibration_evaluation_isolation": True,
                "calibration_scope": config["calibration_scope"],
                "other_parameter_training_indices": {p: ids for p, ids in source["audit"]["human_parameter_training_indices"].items()
                                                      if p != PARAMETER},
                "no_shared_CL_projection_after_foundation_prediction": True},
            "limitations": [
                "Measured scaffold-shift coverage is empirical; no IID, conditional or clinical 90% guarantee.",
                "The calibration set is reused across three fixed diagnostic folds, so fold intervals are dependent.",
                "CL alone uses the new permanently held-out calibration split; other parameters retain earlier full-pool weights.",
                "Prior method development viewed some compounds; exclusion does not establish a new blind clinical benchmark.",
                "No target curve or endpoint is used to fit, calibrate or select this candidate.",
                "There is no selection among checkpoints, features, hyperparameters or folds in this experiment."],
            "elapsed_seconds": time.monotonic() - started}
        result["result_sha256"] = shared._digest(result)
        rich._atomic_json(result_path, result)
        emit("complete", result_receipt=_receipt(root, result_path), current_fold=None)
        return result
    except Exception as error:
        emit("failed", failure_type=type(error).__name__, failure_reason=str(error))
        raise


def status(root, output_dir=OUTPUT):
    root = Path(root).resolve(strict=True)
    output = shared._inside(output_dir, root)
    path = output / "training_state.json"
    if not path.exists():
        return {"stage": "not_started"}
    state = json.loads(path.read_text())
    if state.pop("state_sha256", None) != shared._digest(state):
        raise ValueError("Training state checksum changed")
    return {k: state.get(k) for k in ("version", "stage", "pid", "completed_context_fits", "current_fold",
        "updated_at", "elapsed_seconds", "failure_reason", "result_receipt")}
