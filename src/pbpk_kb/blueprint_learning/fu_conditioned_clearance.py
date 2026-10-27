"""Training-only fu-conditioned relative updates to a frozen CL prior.

No absolute parameter in the retained prior is replaced. Nested scaffold fu
predictions simulate missing evidence; paired measured-fu rows evaluate the
incremental information. An offset is a statistical transform, not CLint.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from . import clinical_shared_prior as shared
from . import rich_parameter_prior as rich

VERSION = "fu-conditioned-clearance-prior-1"
METHOD_NAME = "frozen_CL_with_training_selected_fu_update"
OUTPUT = "experiments/blueprint_design/fu_conditioned_clearance_20261007/prior"
SOURCE_PRIOR = "experiments/blueprint_design/expanded_validation_20261007/prior/predictions.json"
SOURCE_SHARED = "experiments/blueprint_design/expanded_validation_20261007/prior/shared/predictions_audit.json"
TARGETS = "experiments/blueprint_design/expanded_validation_20261007/targets.json"
SEED = shared.SEED
BETAS = (0., .25, .5, 1.)
KINDS = ("conditioned", "offset")
DELTA_LIMIT = math.log10(2.)
MAX_BYTES = 10 * 1024 ** 3
FU_LOG10_TOLERANCE = 1e-5
CODE_FILES = (
    "src/pbpk_kb/blueprint_learning/fu_conditioned_clearance.py",
    "scripts/train_fu_conditioned_clearance.py",
    "src/pbpk_kb/blueprint_learning/clinical_shared_prior.py",
    "src/pbpk_kb/blueprint_learning/rich_parameter_prior.py",
    "src/pbpk_kb/blueprint_learning/adaptive_parameter_prior.py",
    "src/pbpk_kb/blueprints/pk_parameter_learning.py",
)


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _receipt(root, name):
    p = shared._inside(name, root, exists=True)
    return {"path": str(p.relative_to(root)), "sha256": shared._sha(p)}


def _verify(root, pins):
    for name, digest in pins.items():
        if _receipt(root, name)["sha256"] != digest:
            raise ValueError("Frozen source/code changed: " + name)


def _checked_json(root, name):
    receipt = _receipt(root, name)
    value = json.loads((root / receipt["path"]).read_text())
    if value.get("result_sha256") != shared._digest({k: v for k, v in value.items() if k != "result_sha256"}):
        raise ValueError("Result checksum mismatch: " + receipt["path"])
    return value, receipt


def _budget(output):
    total = 0
    for p in output.rglob("*"):
        if p.is_symlink():
            raise ValueError("Training outputs may not contain symlinks")
        if p.is_file():
            total += p.stat().st_size
    if total > MAX_BYTES:
        raise RuntimeError("Training output exceeds 10 GiB; further fits paused")
    return total


def scaffold_folds(rows, indices):
    """Three deterministic scaffold folds, including rows without CL/fu labels."""
    rowmap = {r["index"]: r for r in rows}
    indices = sorted(indices)
    if len(indices) != len(set(indices)) or any(rowmap[i]["status"] != "train" for i in indices):
        raise ValueError("Only unique eligible training indices are allowed")
    groups = [rowmap[i]["scaffold_group"] for i in indices]
    if len(set(groups)) < 3:
        raise ValueError("At least three training scaffolds are required")
    folds = [{"fold": f, "train_indices": [indices[j] for j in a],
              "validation_indices": [indices[j] for j in b]}
             for f, (a, b) in enumerate(GroupKFold(3).split(indices, groups=groups))]
    for f in folds:
        ga = {rowmap[i]["scaffold_group"] for i in f["train_indices"]}
        gb = {rowmap[i]["scaffold_group"] for i in f["validation_indices"]}
        if ga & gb or set(f["train_indices"]) & set(f["validation_indices"]):
            raise ValueError("Scaffold overlap")
    return folds


def build_views(x, indices, cl_labels, fu_labels, missing_fu, *, offset=False):
    """Each CL drug has weight one, split between missing/known-fu views."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] != len(indices) or len(set(indices)) != len(indices):
        raise ValueError("Features must align with unique training indices")
    features, labels, weights, records = [], [], [], []
    for j, i in enumerate(indices):
        if i not in cl_labels:
            continue
        if i not in missing_fu or not math.isfinite(missing_fu[i]) or missing_fu[i] > 0:
            raise ValueError("Every CL training row needs an out-of-fold log-fu prediction")
        available = i in fu_labels
        views = [(missing_fu[i], 0)] + ([(fu_labels[i], 1)] if available else [])
        for fu, flag in views:
            if not math.isfinite(fu) or fu > 0:
                raise ValueError("Log-fu must be finite and no greater than zero")
            features.append(np.r_[x[j], fu, float(flag)])
            labels.append(cl_labels[i] - (fu if offset else 0.))
            weights.append(.5 if available else 1.)
            records.append({"row_index": i, "measured_flag": flag, "log10_fu_input": fu})
    if not features:
        raise ValueError("At least one CL-labeled training row is required")
    return np.asarray(features), np.asarray(labels), np.asarray(weights), records


def bounded_delta(known, predicted, beta):
    if beta not in BETAS:
        raise ValueError("Unknown fixed shrinkage beta")
    a, b = np.asarray(known, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite aligned conditional predictions are required")
    return beta * np.clip(a - b, -DELTA_LIMIT, DELTA_LIMIT)


def metrics(truth, prediction):
    y, p = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or not y.size or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite nonempty aligned labels and predictions required")
    err = abs(p-y)
    score = {"n": len(y), "log10_mae": float(err.mean()), "gmfe": float(10. ** err.mean()),
             "p90_fold_error": float(np.quantile(10. ** err, .9)),
             "within_twofold_count": int(sum(err <= DELTA_LIMIT))}
    for name, mask in (("low_CL_lt1", y < 0.), ("high_CL_ge15", y >= math.log10(15.))):
        score[name] = {"n": int(mask.sum()), "log10_mae": float(err[mask].mean()) if mask.any() else None}
    return score


def select_candidate(truth, direct, deltas):
    """One fixed training OOF search; no clinical scores or target labels."""
    if set(deltas) != set(KINDS):
        raise ValueError("Both fixed conditional candidates are required")
    y, base = np.asarray(truth), np.asarray(direct)
    baseline = metrics(y, base)
    scores = {"none:0": baseline}
    selected, best = {"kind": "none", "beta": 0.}, baseline["log10_mae"]
    for kind in KINDS:
        d = np.asarray(deltas[kind], dtype=float)
        if d.shape != y.shape or not np.isfinite(d).all() or np.any(abs(d) > DELTA_LIMIT + 1e-12):
            raise ValueError("Candidate deltas must already be bounded and aligned")
        for beta in BETAS[1:]:
            name = f"{kind}:{beta:g}"
            value = metrics(y, base + beta*d)
            passes = value["log10_mae"] < baseline["log10_mae"] - 1e-12
            passes &= value["p90_fold_error"] <= baseline["p90_fold_error"] + 1e-12
            for region in ("low_CL_lt1", "high_CL_ge15"):
                passes &= (value[region]["n"] > 0 and
                           value[region]["log10_mae"] <= baseline[region]["log10_mae"] + 1e-12)
            value["passes_all_gates"] = bool(passes)
            scores[name] = value
            if passes and value["log10_mae"] < best - 1e-12:
                selected, best = {"kind": kind, "beta": beta}, value["log10_mae"]
    return {**selected, "scores": scores, "selected_using_training_only": True,
            "selection_scope": "Viewed scaffold OOF development selection, not independent post-selection validation",
            "gates": "Paired CL+fu overall logMAE improves; low CL<1, high CL>=15 logMAE and P90 FE do not increase"}


def _estimator(kind, jobs, device):
    model = rich.make_estimator("rich_xgb_depth3", SEED, jobs, device)
    if kind != "fu":
        model.set_params(model__objective="reg:absoluteerror")
    return model


def _predict_conditional(model, x, log_fu, kind):
    fu = np.asarray(log_fu, dtype=float)
    # BOTH queries use the measured-evidence flag. Equal fu gives zero update.
    query = np.column_stack([x, fu, np.ones(len(fu))])
    result = np.asarray(model.predict(query), dtype=float)
    if kind == "offset":
        result += fu
    if result.shape != fu.shape or not np.isfinite(result).all():
        raise ValueError("Invalid conditional log-CL prediction")
    return result


def _write_state(path, state):
    state["updated_at"] = _now()
    rich._atomic_json(path, {**state, "state_sha256": shared._digest(state)})


def fit_counts(state):
    attempts = state.get("fit_attempts", {})
    actual = sum(attempts.values())
    return {"actual_fit_attempts":actual, "unique_fit_tasks_started":len(attempts),
            "unique_fit_tasks_completed":len(state["fits"]), "retried_fit_attempts":actual-len(attempts)}


def verify_final_fu(source, target_ids, predicted_log10):
    predicted = np.asarray(predicted_log10,dtype=float)
    if predicted.shape != (len(target_ids),) or not np.isfinite(predicted).all():
        raise ValueError("Final fu predictions must match all target identities")
    comparison = {cid:{"new_final_fu":float(10.**predicted[j]),
        "retained_prior_fu":source["parameters"][cid]["fraction_unbound"],
        "log10_difference":float(predicted[j]-math.log10(source["parameters"][cid]["fraction_unbound"]))}
        for j,cid in enumerate(target_ids)}
    if any(abs(v["log10_difference"]) > FU_LOG10_TOLERANCE for v in comparison.values()):
        raise ValueError("Final fu baseline differs from retained fu beyond fixed log10 tolerance 1e-5")
    return comparison


def _prepare(root, targets_path, source_prior, output, jobs, device):
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 8:
        raise ValueError("jobs must be an integer in [1,8]")
    source, source_receipt = _checked_json(root, source_prior)
    tp = _receipt(root, targets_path)
    doc = json.loads((root / tp["path"]).read_text())
    if (set(doc) - {"targets", "source_refs", "version", "rule"} or
            any(set(r)-{"compound_id", "smiles", "drug", "role"} for r in doc["targets"])):
        raise ValueError("Target document may contain identities only")
    cohort = shared._cohort([{k:r[k] for k in ("compound_id", "smiles")} for r in doc["targets"]])
    if cohort != source["targets"] or len(cohort) != 40:
        raise ValueError("Exactly the frozen forty target identities must be excluded")
    for flag in ("target_labels_used", "target_observation_files_opened", "training_structure_overlap",
                 "training_generic_scaffold_overlap", "clinical_results_used_for_selection"):
        if source["audit"].get(flag) is not False:
            raise ValueError("Source training isolation failed: " + flag)
    wheel = _receipt(root, shared.WHEEL)
    if source["source_refs"].get(wheel["path"]) != wheel["sha256"]:
        raise ValueError("Current human PK labels differ from frozen source")
    rows, labels, data_audit = shared._training_data(root / wheel["path"], cohort)
    indices = [r["index"] for r in rows if r["status"] == "train"]
    if (len(indices), len(labels["total_clearance"]), len(labels["fraction_unbound"]),
            len(set(labels["total_clearance"]) & set(labels["fraction_unbound"]))) != (1047, 1045, 700, 698):
        raise ValueError("Frozen 1047/1045/700/698 training membership changed")
    if any(sorted(labels[p]) != source["audit"]["human_parameter_training_indices"][p] for p in labels):
        raise ValueError("Endpoint training membership changed")
    frozen_shared, shared_receipt = _checked_json(root, SOURCE_SHARED)
    if source["source_refs"].get(shared_receipt["path"]) != shared_receipt["sha256"]:
        raise ValueError("Original fu source audit changed")
    fu_config = frozen_shared["audit"]["frozen_input"]
    if (fu_config["methods"]["fraction_unbound"] != ["rich_xgb_depth3"] or
            fu_config["recipes"]["rich_xgb_depth3"] != rich.RECIPES["rich_xgb_depth3"] or
            fu_config["seed"] != SEED or
            frozen_shared["audit"]["parameter_training_indices"]["fraction_unbound"] != sorted(labels["fraction_unbound"]) or
            frozen_shared["audit"]["targets"] != cohort):
        raise ValueError("Retained fu recipe, seed or training membership differs from this candidate")
    for cid in cohort:
        if source["parameters"][cid]["fraction_unbound"] != frozen_shared["parameters"][cid]["fraction_unbound"]:
            raise ValueError("Source fu differs from original shared fu")
    refs = {r["path"]: r["sha256"] for r in (source_receipt, tp, wheel, shared_receipt)}
    config = {"version": VERSION, "source_parameter_prior": source_receipt,
        "source_refs": refs, "code_pins": {name:_receipt(root, name)["sha256"] for name in CODE_FILES},
        "targets": cohort, "eligible_indices": indices, "outer_folds": scaffold_folds(rows, indices),
        "recipe": rich.RECIPES["rich_xgb_depth3"], "seed": SEED, "jobs": jobs,
        "device": rich.choose_device(device), "betas": list(BETAS), "delta_log10_limit": DELTA_LIMIT,
        "inner_folds": 3, "CL_objective": "reg:absoluteerror", "fu_objective": "reg:squarederror",
        "view_weights": "Each drug weight 1: paired known/missing 0.5 each, unpaired missing 1",
        "update": "original_CL * 10**(beta * clip(g(x,knownfu,1)-g(x,priorfu,1),+-log10(2)))",
        "offset_interpretation": "Statistical logCL-logfu regression target; not intrinsic clearance",
        "features": "Full rich Morgan1024 + MACCS167 + RDKit2D; fu and flag appended to conditional models",
        "planned_unique_fit_tasks_max": 26, "retry_policy":"A failed/interrupted fit may be repeated; every real fit attempt is counted separately",
        "final_fu_log10_tolerance":FU_LOG10_TOLERANCE,
        "verified_original_fu_recipe":{"method":"rich_xgb_depth3","recipe":fu_config["recipes"]["rich_xgb_depth3"],
            "seed":fu_config["seed"],"training_count":len(labels["fraction_unbound"]),
            "source":shared_receipt}, "max_output_bytes": MAX_BYTES,
        "software": {"numpy": np.__version__, "sklearn": rich.sklearn.__version__, "xgboost": rich.xgb.__version__}}
    fp = shared._digest(config)
    output.mkdir(parents=True, exist_ok=True)
    _budget(output)
    state_path = output / "training_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.pop("state_sha256", None) != shared._digest(state) or state["input_fingerprint"] != fp:
            raise ValueError("State/configuration changed; refusing mixed resume")
    else:
        if any(p.name not in {"training.lock", "training.log", "training.pid"} for p in output.iterdir()):
            raise ValueError("Refusing unrelated output directory")
        state = {"version": VERSION, "stage": "prepared", "started_at": _now(),
                 "input_fingerprint": fp, "config": config, "fits": {}, "fit_attempts":{}, "fold_results": {}}
        _write_state(state_path, state)
    return state, source, rows, labels, data_audit


def prepare(root, targets_path=TARGETS, source_prior=SOURCE_PRIOR, output_dir=OUTPUT, jobs=8, device="cuda"):
    root = Path(root).resolve(strict=True)
    output = shared._inside(output_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(root, targets_path, source_prior, output, jobs, device)[0]


def train(root, targets_path=TARGETS, source_prior=SOURCE_PRIOR, output_dir=OUTPUT,
          jobs=8, device="cuda", *, progress=None, estimator_factory=None):
    root = Path(root).resolve(strict=True)
    output = shared._inside(output_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(root, targets_path, source_prior, output, jobs, device, progress, estimator_factory)


def _train(root, targets_path, source_prior, output, jobs, device, progress, factory):
    started = time.monotonic()
    state, source, rows, labels, data_audit = _prepare(root, targets_path, source_prior, output, jobs, device)
    cfg = state["config"]
    def verify():
        _verify(root, cfg["source_refs"]); _verify(root, cfg["code_pins"]); _budget(output)
    def emit(stage, **updates):
        state.update(stage=stage, pid=os.getpid(), elapsed_seconds=time.monotonic()-started, **fit_counts(state), **updates)
        _write_state(output / "training_state.json", state)
        if progress:
            progress({"stage": stage, "completed_fits": len(state["fits"]), **updates})
    verify()
    result_path = output / "predictions.json"
    if result_path.exists():
        result, _ = _checked_json(root, result_path)
        if result["input_fingerprint"] != state["input_fingerprint"]:
            raise ValueError("Completed artifact fingerprint differs")
        _verify(root, {result["model_receipt"]["path"]:result["model_receipt"]["sha256"]})
        return result
    try:
        rowmap = {r["index"]:r for r in rows}
        indices = cfg["eligible_indices"]
        positions = {i:j for j,i in enumerate(indices)}
        emit("features")
        x, fa = rich.rich_features([rowmap[i]["identity_smiles"] for i in indices])
        cl, fu = labels["total_clearance"], labels["fraction_unbound"]
        ix = lambda ids: x[[positions[i] for i in ids]]
        make = factory or _estimator

        def fit_predict(key, kind, fit, query, *, views=None, keep_model=False):
            """Checkpoint predictions and metadata only; no CV model files."""
            metadata = {"kind": kind, "train_indices": fit, "query_indices": query}
            if views is not None:
                metadata["views"] = views[3]
            if key in state["fits"] and not keep_model:
                saved = state["fits"][key]
                if saved["metadata"] != metadata:
                    raise ValueError("Fit resume membership changed")
                values = np.asarray(saved["prediction"], dtype=float)
                if values.shape != (len(query),) or not np.isfinite(values).all():
                    raise ValueError("Invalid resumed fit predictions")
                return values, None
            verify(); emit("fitting", current_fit=key)
            model = make(kind, jobs, cfg["device"])
            attempts = state.setdefault("fit_attempts",{})
            attempts[key] = attempts.get(key,0)+1
            # Persist BEFORE the estimator is called, so interrupted/failed fits
            # count as attempts and a subsequent replay is recorded as a retry.
            emit("fit_attempt_started", current_fit=key)
            if views is not None:
                vx, vy, weight, _ = views
                model.fit(vx, vy, model__sample_weight=weight)
                # Query values for conditional models are scored below with explicit fu.
                prediction = np.zeros(len(query))
            else:
                target = fu if kind == "fu" else cl
                model.fit(ix(fit), np.asarray([target[i] for i in fit]))
                prediction = np.asarray(model.predict(ix(query)), dtype=float) if query else np.empty(0)
                if kind == "fu":
                    prediction = rich.physical_log_predictions(prediction, "fraction_unbound")
            if factory is None:
                rich._verify_fitted_device(model, "rich_xgb_depth3", cfg["device"])
            if prediction.shape != (len(query),) or not np.isfinite(prediction).all():
                raise ValueError("Nonfinite fit predictions")
            state["fits"][key] = {"metadata": metadata, "prediction": prediction.tolist()}
            verify(); emit("fit_complete", current_fit=key)
            return prediction, model

        def nested_fu(scope, fit, held):
            missing = {}
            folds = scaffold_folds(rows, fit)
            for inner in folds:
                fi = [i for i in inner["train_indices"] if i in fu]
                qi = inner["validation_indices"]
                if not fi or set(fi) & set(qi) or set(fi+qi) & set(held):
                    raise ValueError("Inner fu isolation failed")
                p, _ = fit_predict(f"{scope}/fu_inner{inner['fold']}", "fu", fi, qi)
                missing.update(zip(qi, p.tolist()))
            if set(missing) != set(fit):
                raise ValueError("Missing-fu view coverage incomplete")
            p, model = fit_predict(scope+"/fu_full", "fu", [i for i in fit if i in fu], held,
                                   keep_model=(scope == "final"))
            return missing, dict(zip(held, p.tolist())), folds, model

        for fold in cfg["outer_folds"]:
            key = str(fold["fold"])
            if key in state["fold_results"]:
                continue
            fit, held = fold["train_indices"], fold["validation_indices"]
            scope = "outer"+key
            missing, predicted_fu, inner, _ = nested_fu(scope, fit, held)
            held_cl = [i for i in held if i in cl]
            paired = [i for i in held_cl if i in fu]
            direct, _ = fit_predict(scope+"/direct", "direct", [i for i in fit if i in cl], held_cl)
            direct_map = dict(zip(held_cl, direct.tolist()))
            deltas = {}
            for kind in KINDS:
                ck = scope+"/"+kind
                if ck in state["fits"] and "bounded_delta" in state["fits"][ck]:
                    saved = state["fits"][ck]
                    if saved["paired_indices"] != paired:
                        raise ValueError("Conditional resume membership changed")
                    deltas[kind] = saved["bounded_delta"]
                    continue
                views = build_views(ix(fit), fit, cl, fu, missing, offset=kind=="offset")
                # A conditional model's cached zero query is not a finished delta.
                _, model = fit_predict(ck, kind, [i for i in fit if i in cl], [], views=views, keep_model=True)
                known = _predict_conditional(model, ix(paired), [fu[i] for i in paired], kind)
                predicted = _predict_conditional(model, ix(paired), [predicted_fu[i] for i in paired], kind)
                delta = bounded_delta(known, predicted, 1.)
                deltas[kind] = delta.tolist()
                state["fits"][ck].update(paired_indices=paired, bounded_delta=delta.tolist(),
                    known_logCL=known.tolist(), predicted_fu_logCL=predicted.tolist())
                emit("conditional_delta_complete", current_fit=ck)
            # Held CL labels are read only after every held prediction is fixed.
            records = [{"row_index": i, "observed_log10_CL": cl[i], "direct_log10_CL":direct_map[i],
                "observed_log10_fu":fu[i], "predicted_log10_fu":predicted_fu[i],
                "bounded_delta":{k:deltas[k][j] for k in KINDS}} for j,i in enumerate(paired)]
            state["fold_results"][key] = {"outer":fold, "inner_fu_folds":inner,
                "paired_oof":records, "all_CL_oof":[{"row_index":i, "observed_log10_CL":cl[i],
                    "direct_log10_CL":direct_map[i]} for i in held_cl]}
            verify(); emit("outer_complete", current_fold=fold["fold"])
        paired = sorted([r for f in state["fold_results"].values() for r in f["paired_oof"]], key=lambda r:r["row_index"])
        if [r["row_index"] for r in paired] != sorted(set(cl)&set(fu)):
            raise ValueError("Paired OOF coverage incomplete")
        selection = select_candidate([r["observed_log10_CL"] for r in paired],
            [r["direct_log10_CL"] for r in paired], {k:[r["bounded_delta"][k] for r in paired] for k in KINDS})
        emit("selection_locked", selection=selection, selection_locked_at=_now())
        missing, _, inner, final_fu = nested_fu("final", indices, [])
        conditional = None
        if selection["kind"] != "none":
            kind = selection["kind"]
            views = build_views(x, indices, cl, fu, missing, offset=kind=="offset")
            _, conditional = fit_predict("final/"+kind, kind, sorted(cl), [], views=views, keep_model=True)
        target_ids = list(source["targets"])
        target_x, tfa = rich.rich_features([source["targets"][cid]["identity_smiles"] for cid in target_ids])
        if tfa["descriptor_names"] != fa["descriptor_names"]:
            raise ValueError("Training and inference feature order differs")
        final_fu_log = rich.physical_log_predictions(final_fu.predict(target_x), "fraction_unbound")
        fu_comparison = verify_final_fu(source,target_ids,final_fu_log)
        model_path = output / "model.joblib"
        bundle = {"version":VERSION, "input_fingerprint":state["input_fingerprint"],
            "selection":selection, "conditional_model":conditional, "fu_model":final_fu,
            "features":fa, "target_features":dict(zip(target_ids, target_x)),
            "target_prior_fu":{cid:source["parameters"][cid]["fraction_unbound"] for cid in target_ids}}
        rich._atomic_model(model_path, bundle)
        verify()
        result = {"version":VERSION, "algorithm_version":VERSION, "method_name":METHOD_NAME,
            "created_at":_now(), "input_fingerprint":state["input_fingerprint"],
            "parameters":deepcopy(source["parameters"]), "units":deepcopy(source["units"]),
            "targets":deepcopy(source["targets"]), "per_compound_audit":deepcopy(source["per_compound_audit"]),
            "source_parameter_prior":cfg["source_parameter_prior"], "source_refs":cfg["source_refs"],
            "code_pins":cfg["code_pins"], "model_receipt":_receipt(root, model_path), "selection":selection,
            "audit":{"target_count":len(target_ids), "human_training_structures":len(indices),
                "eligible_CL_labels":len(cl), "eligible_fu_labels":len(fu), "paired_labels":len(paired),
                "human_parameter_training_indices":deepcopy(source["audit"]["human_parameter_training_indices"]),
                "fold_results":state["fold_results"], "final_inner_fu_folds":inner,
                "training_rows":data_audit["rows"], "feature_audit":fa, "new_fit_count":len(state["fits"]),
                **fit_counts(state), "planned_unique_fit_tasks_max":26,
                "fit_budget_interpretation":"At most 26 distinct fitting tasks, with additional failed/interrupted retries counted separately",
                "selected_using_training_only":True, "clinical_results_used_for_selection":False,
                "target_labels_used":False, "target_observation_files_opened":False, "clinical_files_accepted":False,
                "training_structure_overlap":False, "training_generic_scaffold_overlap":False,
                "base_parameters_copied_exactly":True, "CL_updates_require_source_qualified_fu":True,
                "final_fu_comparison":fu_comparison, "final_fu_log10_tolerance":FU_LOG10_TOLERANCE,
                "final_fu_baseline_consistency_passed":True,"new_final_fu_replaces_retained_prior":False,
                "missing_fu_fits_use_inner_OOF":True, "conditional_queries_share_measured_flag":True},
            "limitations":["Candidate selection uses viewed training OOF, not independent post-selection validation.",
                "A measured fu association does not establish a causal clearance effect or intrinsic clearance.",
                "The OOF direct-CL baseline differs from the retained complementary CL baseline used at inference.",
                "Target clinical cohort is development data; current exclusion does not erase historical recipe selection.",
                "All original parameter values remain unchanged unless an eligible request applies the bounded relative CL factor."],
            "elapsed_seconds":time.monotonic()-started}
        result["result_sha256"] = shared._digest(result)
        rich._atomic_json(result_path, result)
        emit("complete", result_receipt=_receipt(root, result_path))
        return result
    except Exception as error:
        emit("failed", failure_type=type(error).__name__, failure_reason=str(error))
        raise


def predict_update(root, artifact, compound_id, observed_fu, prior_fu, *, evidence_status="measured"):
    """Return a bounded relative CL update; never fit or read clinical files.

    Provenance qualification (independent study/population) belongs to the caller.
    Only measured or independently derived fu is eligible. Caller must verify
    its source receipt; the string status alone is not evidence qualification.
    """
    root = Path(root).resolve(strict=True)
    value, _ = _checked_json(root, artifact) if not isinstance(artifact, dict) else (artifact, None)
    if (value.get("version") != VERSION or value.get("result_sha256") !=
            shared._digest({k:v for k,v in value.items() if k!="result_sha256"})):
        raise ValueError("Unsupported or changed fu update artifact")
    _verify(root, value["source_refs"]); _verify(root, value["code_pins"])
    if compound_id not in value["parameters"]:
        raise ValueError("Compound is outside the frozen target cohort")
    frozen_fu = value["parameters"][compound_id]["fraction_unbound"]
    if (not isinstance(prior_fu, (int,float)) or isinstance(prior_fu,bool) or
            not math.isclose(prior_fu, frozen_fu, rel_tol=1e-12, abs_tol=0.)):
        raise ValueError("Prior fu differs from the frozen parameter prior")
    audit = {"version":VERSION, "compound_id":compound_id, "prior_fu":prior_fu,
             "observed_fu":observed_fu, "evidence_status":evidence_status,
             "selection":{k:value["selection"][k] for k in ("kind","beta")}, "factor":1.,
             "candidate":value["selection"]["kind"], "beta":value["selection"]["beta"],
             "raw_delta_log10":0., "bounded_delta_log10":0., "applied_delta_log10":0.,
             "source_parameter_prior":value["source_parameter_prior"], "model_receipt":value["model_receipt"],
             "uses_clinical_observations":False, "reason":"no_measured_fu", "clinical_risk_guarantee":False}
    if observed_fu is None:
        return audit
    if evidence_status not in {"measured", "derived"}:
        raise ValueError("Only independently qualified measured/derived fu may update CL")
    if isinstance(observed_fu,bool) or not isinstance(observed_fu,(int,float)) or not math.isfinite(observed_fu) or not 0<observed_fu<=1:
        raise ValueError("Measured fu must be a finite fraction in (0,1]")
    selection = value["selection"]
    if selection["beta"] == 0. or observed_fu == prior_fu:
        audit["reason"] = "training_gate_retained_original" if selection["beta"] == 0. else "identical_fu"
        return audit
    receipt = value["model_receipt"]
    _verify(root, {receipt["path"]:receipt["sha256"]})
    bundle = joblib.load(root / receipt["path"])
    if bundle["input_fingerprint"] != value["input_fingerprint"] or bundle["selection"] != selection:
        raise ValueError("Conditional model differs from frozen artifact")
    x = np.asarray(bundle["target_features"][compound_id]).reshape(1,-1)
    kind, model = selection["kind"], bundle["conditional_model"]
    known = _predict_conditional(model, x, [math.log10(observed_fu)], kind)
    old = _predict_conditional(model, x, [math.log10(prior_fu)], kind)
    raw = float(known[0]-old[0])
    bounded = float(np.clip(raw,-DELTA_LIMIT,DELTA_LIMIT))
    delta = selection["beta"]*bounded
    audit.update(factor=float(10.**delta), reason="qualified_fu_relative_update",
        raw_delta_log10=raw, bounded_delta_log10=bounded, applied_delta_log10=delta,
        maximum_fold_update=float(2.**selection["beta"]))
    return audit


def status(root, output_dir=OUTPUT):
    root = Path(root).resolve(strict=True)
    output = shared._inside(output_dir, root)
    path = output / "training_state.json"
    if not path.exists():
        return {"version":VERSION, "stage":"not_prepared"}
    state = json.loads(path.read_text())
    if state.pop("state_sha256",None) != shared._digest(state):
        raise ValueError("Training state checksum changed")
    alive = False
    if isinstance(state.get("pid"),int):
        try:
            os.kill(state["pid"],0); alive = True
        except ProcessLookupError:
            pass
    return {k:state.get(k) for k in ("stage","pid","updated_at","elapsed_seconds","current_fit",
        "failure_reason","selection")} | {"completed_fits":len(state["fits"]),
        **fit_counts(state),"outer_folds_complete":len(state["fold_results"]), "pid_exists":alive, "output_bytes":_budget(output)}
