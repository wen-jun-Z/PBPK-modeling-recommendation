"""Train-only scaffold selection of richer five-endpoint human PK priors.

Endpoint labels are never imputed. Descriptor imputation/scaling is fitted
inside each training fold. Calibration labels diagnose frozen models only;
held-out compound labels and evaluation files are never accessed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import joblib
import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb

from . import adaptive_parameter_prior as base_prior

VERSION = "rich-scaffold-five-endpoint-prior-1"
SOURCE_RUN = "experiments/blueprint_design/purpose_hundred_20260930"
WHEEL = "artifacts/pk_predictors/pksmart/pksmart-3.0.1-py3-none-any.whl"
OUTPUT = "experiments/blueprint_design/accuracy_refinement_20261001/priors"
CODE_FILES = ["src/pbpk_kb/blueprint_learning/rich_parameter_prior.py",
              "scripts/train_rich_parameter_prior.py", *base_prior.CODE_FILES]
PARAMETERS = {**base_prior.PARAMETERS,
              "terminal_half_life": ("human_thalf", "h"), "mrt": ("human_mrt", "h")}
BASE_CANDIDATES = ("old_extra_trees", "rich_extra_trees", "rich_svr", "rich_xgb_depth3", "rich_xgb_depth5")
ENSEMBLE_MEMBERS = BASE_CANDIDATES[1:]
CANDIDATES = (*BASE_CANDIDATES, "rich_geometric_ensemble")
RECIPES = {
    "old_extra_trees": {"n_estimators": 400, "min_samples_leaf": 2, "max_features": "sqrt"},
    "rich_extra_trees": {"n_estimators": 300, "min_samples_leaf": 2, "max_features": .7},
    "rich_svr": {"kernel": "rbf", "C": 10., "epsilon": .1, "gamma": "scale"},
    "rich_xgb_depth3": {"n_estimators": 500, "max_depth": 3, "learning_rate": .03, "reg_lambda": 10.},
    "rich_xgb_depth5": {"n_estimators": 500, "max_depth": 5, "learning_rate": .03, "reg_lambda": 10.},
}
DESCRIPTORS = tuple(Descriptors._descList)
CONSISTENCY_PARAMETERS = ("total_clearance", "vss", "mrt")
CONSISTENCY_NORMAL = np.array([-1., 1., -1.])
CONSISTENCY_CONSTANT = math.log10(.06)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def _atomic_model(path, value):
    temp = path.with_name(path.name + ".tmp")
    joblib.dump(value, temp, compress=3)
    os.replace(temp, path)


def _save_audit(path, audit):
    payload = {k: v for k, v in audit.items() if k != "audit_digest"}
    payload["audit_digest"] = base_prior._digest(payload)
    _atomic_json(path, payload)


def _read_audit(path):
    data = json.loads(path.read_text())
    digest = data.pop("audit_digest", None)
    if digest != base_prior._digest(data):
        raise ValueError("Training audit changed")
    return data


def _status(output, state, **updates):
    path = output / "status.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    payload = {**old, "version": VERSION, "state": state, "updated_at": _now(), "pid": os.getpid(), **updates}
    payload.setdefault("started_at", _now())
    payload["events"] = [*payload.get("events", []), {"time": _now(), "state": state, **updates}][-150:]
    _atomic_json(path, payload)


def rich_features(identities):
    """Full installed RDKit 2D descriptor list + Morgan1024 + MACCS167."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=False)
    values = np.empty((len(identities), 1024 + 167 + len(DESCRIPTORS)), dtype=np.float32)
    missing = np.zeros(len(DESCRIPTORS), dtype=int)
    for index, identity in enumerate(identities):
        molecule = Chem.MolFromSmiles(identity)
        if molecule is None:
            raise ValueError("Invalid normalized molecule")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), values[index, :1024])
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(molecule), values[index, 1024:1191])
        for j, (_, function) in enumerate(DESCRIPTORS):
            try:
                value = float(function(molecule))
            except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
                value = math.nan
            if not math.isfinite(value) or abs(value) > np.finfo(np.float32).max:
                value = math.nan
                missing[j] += 1
            values[index, 1191 + j] = value
    return values, {"descriptor_names": [name for name, _ in DESCRIPTORS],
                    "descriptor_missing_counts": missing.tolist(), "dimensions": values.shape[1],
                    "morgan_bits": 1024, "maccs_bits": 167, "imputation": "fold_training_median_keep_empty_features"}


def choose_device(preference="auto"):
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda")
    if preference == "cpu":
        return "cpu"
    import torch
    available = bool(xgb.build_info().get("USE_CUDA")) and torch.cuda.is_available()
    if preference == "cuda" and not available:
        raise ValueError("Requested CUDA is unavailable for this project's XGBoost")
    return "cuda:0" if available else "cpu"


def make_estimator(method, seed, n_jobs, device):
    if method not in BASE_CANDIDATES:
        raise ValueError("Unknown base candidate")
    recipe = deepcopy(RECIPES[method])
    if method.endswith("extra_trees"):
        estimator = ExtraTreesRegressor(**recipe, random_state=seed, n_jobs=n_jobs)
    elif method == "rich_svr":
        estimator = SVR(**recipe)
    else:
        estimator = xgb.XGBRegressor(**recipe, objective="reg:squarederror", tree_method="hist",
                                    device=device, random_state=seed, n_jobs=n_jobs, verbosity=1)
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if method == "rich_svr":
        steps.append(("scale", StandardScaler()))
    return Pipeline([*steps, ("model", estimator)])


def physical_log_predictions(predictions, parameter):
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Nonfinite or malformed model prediction")
    return np.minimum(values, 0.) if parameter == "fraction_unbound" else values


def candidate_scores(oof, truth, parameter):
    """Six predeclared candidates; ensemble averages only OOF log predictions."""
    if set(oof) != set(BASE_CANDIDATES):
        raise ValueError("Every fixed base candidate must complete OOF predictions")
    truth = np.asarray(truth, dtype=float)
    if truth.ndim != 1 or not len(truth) or not np.isfinite(truth).all():
        raise ValueError("Finite nonempty training labels are required")
    predicted = {name: physical_log_predictions(value, parameter) for name, value in oof.items()}
    if any(value.shape != truth.shape for value in predicted.values()):
        raise ValueError("OOF predictions must match endpoint-labeled training membership")
    predicted["rich_geometric_ensemble"] = np.mean([predicted[name] for name in ENSEMBLE_MEMBERS], axis=0)
    scores = {name: float(np.mean(np.abs(value - truth))) for name, value in predicted.items()}
    selected = min(CANDIDATES, key=lambda name: (scores[name], CANDIDATES.index(name)))
    return selected, scores, predicted


def project_consistency(predicted_log10, residual_variances):
    """Variance-weighted projection onto Vss[L/kg] = .06 CL[mL/min/kg] MRT[h]."""
    predicted, variances = np.asarray(predicted_log10, dtype=float), np.asarray(residual_variances, dtype=float)
    if predicted.ndim != 2 or predicted.shape[1] != 3 or variances.shape != (3,):
        raise ValueError("Consistency projection requires CL,Vss,MRT and three variances")
    if not np.isfinite(predicted).all() or not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("Projection requires finite predictions and positive residual variances")
    error = predicted @ CONSISTENCY_NORMAL - CONSISTENCY_CONSTANT
    correction = variances * CONSISTENCY_NORMAL / np.sum(variances)
    return predicted - error[:, None] * correction[None, :]


def crossfit_consistency(audit):
    """Choose projection by training OOF error, with weights fitted outside each fold."""
    all_maps = {}
    for parameter in CONSISTENCY_PARAMETERS:
        selection = audit["selection"][parameter]
        predicted = dict(zip(selection["oof_indices"], selection["selected_oof_log10"]))
        truth = {r["row_index"]: r["observed_log10"] for fold in audit["cv_progress"][parameter]["old_extra_trees"].values() for r in fold["out_of_fold"]}
        if set(predicted) != set(truth):
            raise ValueError("OOF selection and training labels disagree")
        all_maps[parameter] = {i: (predicted[i], truth[i]) for i in predicted}
    common = sorted(set.intersection(*(set(v) for v in all_maps.values())))
    result = {"enabled": False, "parameters": CONSISTENCY_PARAMETERS,
              "equation": "Vss[L/kg] = 0.06 * CL[mL/min/kg] * MRT[h]",
              "weight_estimation": "diagonal signed OOF residual sample variances; floor 1e-8",
              "selection_rule": "strictly lower three-endpoint-equal OOF logMAE; no endpoint worsens over 1 percent",
              "common_oof_indices": common, "folds": [], "uses_calibration_or_target_labels": False,
              "clinical_covariance_model": False, "strict_nested_second_stage_validation": False,
              "selection_interpretation": "training-only heuristic: other folds' base fits may include the held fold; not an independent 1-percent regression guarantee"}
    if len(common) < 3:
        return {**result, "reason": "insufficient_common_training_labels"}
    predicted = np.array([[all_maps[p][i][0] for p in CONSISTENCY_PARAMETERS] for i in common])
    observed = np.array([[all_maps[p][i][1] for p in CONSISTENCY_PARAMETERS] for i in common])
    residuals = observed - predicted
    corrected = np.full_like(predicted, np.nan)
    for fold in audit["folds"]:
        held = [j for j, i in enumerate(common) if i in fold["validation_indices"]]
        fit = [j for j, i in enumerate(common) if i not in fold["validation_indices"]]
        if len(fit) < 2 or not held:
            return {**result, "reason": "insufficient_fold_specific_projection_support"}
        variances = np.maximum(np.var(residuals[fit], axis=0, ddof=1), 1e-8)
        corrected[held] = project_consistency(predicted[held], variances)
        result["folds"].append({"fold": fold["fold"], "weight_fit_indices": [common[j] for j in fit],
            "held_indices": [common[j] for j in held], "residual_variances": variances.tolist()})
    if not np.isfinite(corrected).all():
        raise ValueError("Projection OOF coverage is incomplete")
    raw_mae, corrected_mae = np.mean(np.abs(observed-predicted), axis=0), np.mean(np.abs(observed-corrected), axis=0)
    enabled = bool(corrected_mae.mean() < raw_mae.mean() - 1e-12 and np.all(corrected_mae <= raw_mae * 1.01 + 1e-12))
    result.update(enabled=enabled, reason="training_OOF_improvement_with_endpoint_guard" if enabled else "training_OOF_criterion_not_met",
        raw_log10_mae=dict(zip(CONSISTENCY_PARAMETERS, raw_mae.tolist())),
        corrected_log10_mae=dict(zip(CONSISTENCY_PARAMETERS, corrected_mae.tolist())),
        raw_endpoint_equal_log10_mae=float(raw_mae.mean()), corrected_endpoint_equal_log10_mae=float(corrected_mae.mean()),
        inference_residual_variances=np.maximum(np.var(residuals, axis=0, ddof=1), 1e-8).tolist(),
        corrected_oof_log10=corrected.tolist())
    return result


def _verify_fitted_device(estimator, method, requested):
    if method.startswith("rich_xgb") and requested.startswith("cuda"):
        actual = json.loads(estimator.named_steps["model"].get_booster().save_config())["learner"]["generic_param"]["device"]
        if not actual.startswith("cuda"):
            raise RuntimeError("CUDA requested but XGBoost silently fitted on " + actual)


def _range(log_nominal, width, parameter, n, level):
    nominal, lower, upper = 10. ** log_nominal, 10. ** (log_nominal - width), 10. ** (log_nominal + width)
    if parameter == "fraction_unbound":
        upper = min(upper, 1.)
    if not all(math.isfinite(v) and v > 0 for v in (nominal, lower, upper)):
        raise ValueError("Nonpositive or nonfinite empirical stress range")
    return {"central": nominal, "lower": lower, "upper": upper, "unit": PARAMETERS[parameter][1],
            "quantile": float(level) / 100, "log10_residual_quantile": width, "calibration_n": n,
            "clinical_confidence_interval": False, "quantile_method": "higher"}


def _predict(method, parameter, models, old_x, rich_x):
    if method == "rich_geometric_ensemble":
        return np.mean([_predict(name, parameter, models, old_x, rich_x) for name in ENSEMBLE_MEMBERS], axis=0)
    x = old_x if method == "old_extra_trees" else rich_x
    return physical_log_predictions(models[method].predict(x), parameter)


def fit_rich_prior(project_root, *, output_dir=OUTPUT, source_run=SOURCE_RUN, wheel=WHEEL,
                   n_jobs=8, device="auto"):
    """Resume fold/final-fit checkpoints and freeze all target predictions."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or not 1 <= n_jobs <= 8:
        raise ValueError("n_jobs must be an integer in [1, 8]")
    root = Path(project_root).resolve(strict=True)
    source_dir = base_prior._inside(source_run, root)
    source = base_prior._inside(wheel, root)
    output = base_prior._inside(output_dir, root, exists=False)
    manifest_path, baseline_path = source_dir / "cohort.json", source_dir / "priors/audit.json"
    baseline = json.loads(baseline_path.read_text())
    cohort = base_prior._cohort(json.loads(manifest_path.read_text()), baseline)
    if base_prior._sha(source) != baseline["source"]["source_sha256"]:
        raise ValueError("Frozen training dataset source changed")
    base_prior._verify(baseline["source_pins"], root)
    resolved_device = choose_device(device)
    pins = dict(baseline["source_pins"])
    for path in (source, manifest_path, baseline_path, *(base_prior._inside(name, root) for name in CODE_FILES)):
        pins[str(path.relative_to(root))] = base_prior._sha(path)
    frozen = {"version": VERSION, "source_pins": pins, "cohort": cohort, "n_jobs": n_jobs,
              "device": resolved_device, "seed": baseline["frozen_input"]["seed"], "parameters": PARAMETERS,
              "recipes": RECIPES, "ensemble_members": ENSEMBLE_MEMBERS, "ensemble_weights": "uniform_log_space",
              "folds": 3, "selection_metric": "compound_equal_mean_absolute_log10_OOF_error",
              "selection_tie_order": CANDIDATES, "descriptor_names": [name for name, _ in DESCRIPTORS],
              "software": {"numpy": np.__version__, "sklearn": sklearn.__version__, "rdkit": rdBase.rdkitVersion,
                           "xgboost": xgb.__version__},
              "fu_bound": "clip predicted log10 fu to at most 0 consistently in OOF,calibration,target"}
    frozen["consistency_projection"] = {"parameters": CONSISTENCY_PARAMETERS, "weighting": "OOF diagonal variances",
        "crossfit": "weights from other scaffold folds", "endpoint_max_relative_MAE_regression": .01,
        "requires_endpoint_equal_MAE_improvement": True}
    fingerprint = base_prior._digest(frozen)
    output.mkdir(parents=True, exist_ok=True)
    audit_path, model_path, prediction_path = output / "audit.json", output / "model.joblib", output / "predictions.json"
    known = {"status.json", "audit.json", "model.joblib", "predictions.json", "status.json.tmp", "audit.json.tmp", "model.joblib.tmp", "predictions.json.tmp"}
    if set(p.name for p in output.iterdir()) - known:
        raise ValueError("Output directory contains unrelated files; it will not be overwritten")
    if audit_path.exists():
        audit = _read_audit(audit_path)
        if audit.get("input_fingerprint") != fingerprint:
            raise ValueError("Existing run uses different frozen sources/code/configuration")
    else:
        if model_path.exists() or prediction_path.exists():
            raise ValueError("Prior artifacts without an audit are retained and rejected")
        audit = {"version": VERSION, "input_fingerprint": fingerprint, "frozen_input": frozen,
                 "source_pins": pins, "started_at": _now(), "cv_progress": {}, "selection": {},
                 "calibration": {}, "label_audit": {}, "target_observation_files_opened": False,
                 "calibration_labels_used_for_selection": False, "target_labels_used_for_selection": False}
        _save_audit(audit_path, audit)
    if prediction_path.exists():
        bundle = json.loads(prediction_path.read_text())
        if bundle["input_fingerprint"] != fingerprint or base_prior._digest(bundle["predictions"]) != audit.get("predictions_digest"):
            raise ValueError("Frozen target predictions changed")
        base_prior._verify({v["path"]: v["sha256"] for v in bundle["artifacts"].values()}, root)
        _status(output, "completed", reused=True, target_predictions_locked=True)
        return bundle
    _status(output, "running", stage="validate_dataset", device=resolved_device, target_predictions_locked=False)
    started = time.monotonic()
    dataset = base_prior.base.load_parameter_dataset(source)
    train, calibration = base_prior._frozen_membership(dataset, baseline, cohort)
    folds = base_prior._scaffold_folds(dataset, train)
    audit.update(train_indices=train, calibration_indices=calibration, train_count=len(train),
                 calibration_count=len(calibration), cohort_count=len(cohort), folds=folds,
                 split_structure_and_scaffold_exclusion_verified=True, source=dataset.provenance,
                 code_pins={name:pins[name] for name in CODE_FILES}, cohort_labels_used=False)
    included = sorted(train + calibration)
    identities = [dataset.rows[i]["identity_smiles"] for i in included] + [r["identity_smiles"] for r in cohort.values()]
    _status(output, "running", stage="features", compounds=len(identities))
    rich_x, feature_audit = rich_features(identities)
    old_x = base_prior.base._features(identities)
    positions = {i: j for j, i in enumerate(included)}
    target_positions = list(range(len(included), len(identities)))
    audit["features"] = feature_audit
    _save_audit(audit_path, audit)
    for parameter, (label, unit) in PARAMETERS.items():
        labeled = [i for i in train if dataset.rows[i]["labels_log10"][label] is not None]
        held = [i for i in calibration if dataset.rows[i]["labels_log10"][label] is not None]
        if not labeled or not held:
            raise ValueError("Endpoint lacks independent training or calibration labels: " + parameter)
        audit["label_audit"][parameter] = {"label": label, "unit": unit, "train_indices": labeled,
            "calibration_indices": held, "train_count": len(labeled), "calibration_count": len(held),
            "train_missing_indices": sorted(set(train) - set(labeled)), "calibration_missing_indices": sorted(set(calibration) - set(held))}
        audit["cv_progress"].setdefault(parameter, {})
        for method in BASE_CANDIDATES:
            audit["cv_progress"][parameter].setdefault(method, {})
            for fold in folds:
                key = str(fold["fold"])
                if key in audit["cv_progress"][parameter][method]:
                    continue
                fit = [i for i in fold["train_indices"] if i in labeled]
                validation = [i for i in fold["validation_indices"] if i in labeled]
                if not fit or not validation:
                    raise ValueError("A scaffold fold lacks endpoint labels")
                _status(output, "running", stage="cross_validation", parameter=parameter, method=method, fold=fold["fold"])
                x = old_x if method == "old_extra_trees" else rich_x
                estimator = make_estimator(method, frozen["seed"], n_jobs, resolved_device)
                before = time.monotonic()
                estimator.fit(x[[positions[i] for i in fit]], [dataset.rows[i]["labels_log10"][label] for i in fit])
                _verify_fitted_device(estimator, method, resolved_device)
                predicted = physical_log_predictions(estimator.predict(x[[positions[i] for i in validation]]), parameter)
                rows = [{"row_index": i, "fold": fold["fold"], "observed_log10": dataset.rows[i]["labels_log10"][label],
                         "predicted_log10": float(value)} for i, value in zip(validation, predicted)]
                audit["cv_progress"][parameter][method][key] = {"train_indices": fit, "validation_indices": validation,
                    "seconds": time.monotonic() - before, "out_of_fold": rows}
                base_prior._verify(pins, root)
                _save_audit(audit_path, audit)
        oof = {}
        for method in BASE_CANDIDATES:
            mapping = {r["row_index"]: r["predicted_log10"] for fold in audit["cv_progress"][parameter][method].values() for r in fold["out_of_fold"]}
            if set(mapping) != set(labeled):
                raise ValueError("Incomplete OOF coverage")
            oof[method] = [mapping[i] for i in labeled]
        selected, scores, all_oof = candidate_scores(oof, [dataset.rows[i]["labels_log10"][label] for i in labeled], parameter)
        audit["selection"][parameter] = {"method": selected, "candidate_log10_mae": scores,
            "oof_count": len(labeled), "selected_oof_log10": all_oof[selected].tolist(), "oof_indices": labeled,
            "ensemble_oof_log10": all_oof["rich_geometric_ensemble"].tolist(),
            "old_extra_trees_log10_mae": scores["old_extra_trees"],
            "selected_log10_mae": scores[selected], "uses_calibration_or_target_labels": False}
        _save_audit(audit_path, audit)
    # Endpoints are all selected before any calibration labels are scored.
    audit["consistency_projection"] = crossfit_consistency(audit)
    _save_audit(audit_path, audit)
    if model_path.exists():
        models_bundle = joblib.load(model_path)
        if models_bundle.get("input_fingerprint") != fingerprint:
            raise ValueError("Partial final-fit checkpoint has different frozen inputs")
    else:
        models_bundle = {"version": VERSION, "input_fingerprint": fingerprint, "models": {}, "selected_methods": {p: s["method"] for p, s in audit["selection"].items()}}
    predictions = {cid: {**identity, "methods": {}, "selected_models": {}, "target_labels_used": False,
                        "cohort_labels_used": False, "training_structure_overlap": False,
                        "training_scaffold_overlap": False, "calibration_structure_overlap": False,
                        "calibration_scaffold_overlap": False,
                        "source_kind": "predicted", "eligible_as_measured_evidence": False} for cid, identity in cohort.items()}
    logs, calibration_logs = {}, {}
    calibration_positions = {i:j for j,i in enumerate(calibration)}
    cx_old, cx_rich = old_x[[positions[i] for i in calibration]], rich_x[[positions[i] for i in calibration]]
    for parameter, (label, unit) in PARAMETERS.items():
        selected = audit["selection"][parameter]["method"]
        needed = tuple(dict.fromkeys(("old_extra_trees", *(ENSEMBLE_MEMBERS if selected == "rich_geometric_ensemble" else (selected,)))))
        models_bundle["models"].setdefault(parameter, {})
        labeled = audit["label_audit"][parameter]["train_indices"]
        held = audit["label_audit"][parameter]["calibration_indices"]
        for method in needed:
            if method in models_bundle["models"][parameter]:
                continue
            _status(output, "running", stage="final_fit", parameter=parameter, method=method)
            estimator = make_estimator(method, frozen["seed"], n_jobs, resolved_device)
            x = old_x if method == "old_extra_trees" else rich_x
            estimator.fit(x[[positions[i] for i in labeled]], [dataset.rows[i]["labels_log10"][label] for i in labeled])
            _verify_fitted_device(estimator, method, resolved_device)
            models_bundle["models"][parameter][method] = estimator
            _atomic_model(model_path, models_bundle)
        local_models = models_bundle["models"][parameter]
        for branch, method in (("baseline", "old_extra_trees"), ("selected", selected)):
            calibration_logs[(branch, parameter)] = _predict(method, parameter, local_models, cx_old, cx_rich)
            logs[(branch, parameter)] = _predict(method, parameter, local_models, old_x[target_positions], rich_x[target_positions])
        for cid in predictions:
            predictions[cid]["selected_models"][parameter] = selected
    projection = audit["consistency_projection"]
    if projection["enabled"]:
        for arrays in (logs, calibration_logs):
            matrix = np.column_stack([arrays[("selected",p)] for p in CONSISTENCY_PARAMETERS])
            corrected = project_consistency(matrix, projection["inference_residual_variances"])
            for j,p in enumerate(CONSISTENCY_PARAMETERS):
                arrays[("selected",p)] = corrected[:,j]
    for parameter, (label, unit) in PARAMETERS.items():
        held = audit["label_audit"][parameter]["calibration_indices"]
        truth = np.array([dataset.rows[i]["labels_log10"][label] for i in held])
        audit["calibration"][parameter] = {}
        for branch in ("baseline", "selected"):
            cal_logs = calibration_logs[(branch,parameter)][[calibration_positions[i] for i in held]]
            residuals = truth - cal_logs
            widths = {level: float(np.quantile(np.abs(residuals), float(level)/100, method="higher")) for level in ("80", "90")}
            audit["calibration"][parameter][branch] = {"indices": held, "n": len(held), "signed_log10_residuals": residuals.tolist(),
                "residual_quantiles": widths, "gmfe": float(10. ** np.mean(np.abs(residuals))),
                "mean_absolute_log10_error": float(np.mean(np.abs(residuals))), "within_twofold_n": int(np.sum(np.abs(residuals) <= math.log10(2))),
                "used_for_selection": False, "consistency_projection_applied": branch == "selected" and projection["enabled"] and parameter in CONSISTENCY_PARAMETERS}
    for j, (cid, record) in enumerate(predictions.items()):
        record["consistency_projection_applied"] = projection["enabled"]
        for branch in ("baseline", "selected"):
            nominal = {p: float(10. ** logs[(branch, p)][j]) for p in PARAMETERS}
            ranges = {p: {level: _range(float(logs[(branch,p)][j]), audit["calibration"][p][branch]["residual_quantiles"][level],
                p, audit["calibration"][p][branch]["n"], level) for level in ("80", "90")} for p in PARAMETERS}
            record["methods"][branch] = {"predict_nominal": nominal, "ranges": ranges}
    base_prior._verify(pins, root)
    audit.update(completed_at=_now(), elapsed_seconds=time.monotonic()-started,
        model=base_prior._ref(model_path, root), predictions_digest=base_prior._digest(predictions),
        target_predictions_locked_before_scoring=True, calibration_is_independent_test=False,
        limitations=["Development model selection uses fixed training-scaffold folds, not new independent clinical data.",
            "Calibration diagnostics do not choose models or ensemble weights.",
            "Human PK aggregate parameter prediction is not clinical curve validation.",
            "OOF projection weights exclude each row's fold residuals, but reuse base OOF models; this is not strict nested second-stage validation or an independent endpoint-error guarantee.",
            "Half-life and MRT predictions are drug-property priors, not automatically physically consistent PBPK outputs."])
    _save_audit(audit_path, audit)
    bundle = {"version": VERSION, "input_fingerprint": fingerprint, "predictions": predictions,
              "artifacts": {"model": audit["model"], "audit": base_prior._ref(audit_path, root)}}
    _atomic_json(prediction_path, bundle)
    _status(output, "completed", stage="all_predictions_locked", target_predictions_locked=True,
            target_count=len(predictions), selected_methods={p:s["method"] for p,s in audit["selection"].items()},
            elapsed_seconds=time.monotonic()-started)
    return bundle
