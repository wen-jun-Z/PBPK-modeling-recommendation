"""Train-only scaffold CV selects RF or ExtraTrees for CL, fu and Vss.

The old cohort and training/calibration membership are immutable inputs. Only
training-label out-of-fold errors select the endpoint model. Calibration labels
then set marginal 80/90% empirical residual stress widths, never clinical CIs.
The held-out cohort's evaluation files are not opened by this module.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import sklearn
from rdkit import rdBase
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from pbpk_kb.blueprints import pk_parameter_learning as base

VERSION = "adaptive-scaffold-parameter-prior-1"
PARAMETERS = {
    "total_clearance": ("human_CL_mL_min_kg", "ml/min/kg"),
    "fraction_unbound": ("human_fup", "fraction"),
    "vss": ("human_VDss_L_kg", "l/kg"),
}
CANDIDATES = {
    "rf": {"n_estimators": 200, "min_samples_leaf": 3, "max_features": "sqrt"},
    "extra_trees": {"n_estimators": 400, "min_samples_leaf": 2, "max_features": "sqrt"},
}
QUANTILES = {"80": .8, "90": .9}
FOLDS = 3
CODE_FILES = ["src/pbpk_kb/blueprint_learning/adaptive_parameter_prior.py",
              "src/pbpk_kb/blueprints/pk_parameter_learning.py"]


def _inside(path, root, *, exists=True):
    resolved = (root / path).resolve(strict=exists)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("All sources and outputs must remain inside this project")
    return resolved


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _ref(path, root):
    return {"path": str(path.relative_to(root)), "sha256": _sha(path)}


def _verify(pins, root):
    for name, expected in pins.items():
        if _sha(_inside(name, root)) != expected:
            raise ValueError("Frozen source/artifact changed: " + name)


def _save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _cohort(manifest, baseline):
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("A frozen, nonempty cohort manifest is required")
    rows = {}
    for case in cases:
        cid = case.get("compound_id")
        if not isinstance(cid, str) or not cid or cid in rows:
            raise ValueError("Cohort compound identifiers must be unique")
        normalized = base.normalize_structure(case.get("identity_smiles"))
        rows[cid] = {"input_smiles": case["identity_smiles"], **normalized}
    expected = baseline.get("external_exclusions", {})
    if set(rows) != set(expected) or len(rows) != baseline.get("cohort_count"):
        raise ValueError("Cohort membership differs from the frozen baseline")
    for cid, record in rows.items():
        if any(record[key] != expected[cid].get(key)
               for key in ("identity_smiles", "scaffold_group", "scaffold_smiles")):
            raise ValueError("Cohort identity/scaffold differs from the frozen baseline")
    if len({row["identity_smiles"] for row in rows.values()}) != len(rows):
        raise ValueError("Cohort contains duplicate normalized chemical identities")
    return dict(sorted(rows.items()))


def _frozen_membership(dataset, baseline, cohort):
    if dataset.provenance["csv_sha256"] != baseline.get("source", {}).get("csv_sha256"):
        raise ValueError("Dataset differs from the frozen baseline source")
    train, calibration = baseline.get("train_indices"), baseline.get("calibration_indices")
    for label, indices in (("train", train), ("calibration", calibration)):
        if (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(dataset.rows) for i in indices)
                or len(indices) != baseline.get(label + "_count")):
            raise ValueError("Invalid frozen " + label + " membership")
    if set(train) & set(calibration):
        raise ValueError("Training/calibration row leakage")
    audited = {r["index"]: r for r in baseline["rows"]}
    protected_ids = {r["identity_smiles"] for r in cohort.values()}
    protected_groups = {r["scaffold_group"] for r in cohort.values()}
    seen = set()
    for status, indices in (("train", train), ("calibration", calibration)):
        if {i for i, r in audited.items() if r["status"] == status} != set(indices):
            raise ValueError("Frozen row dispositions disagree with membership")
        for i in indices:
            row = dataset.rows[i]
            if row["structure_error"] or row["identity_smiles"] in seen:
                raise ValueError("Invalid or duplicate included chemical identity")
            seen.add(row["identity_smiles"])
            if any(row[key] != audited[i][key] for key in ("identity_smiles", "scaffold_group")):
                raise ValueError("Frozen source row identity/scaffold changed")
            if row["identity_smiles"] in protected_ids or row["scaffold_group"] in protected_groups:
                raise ValueError("External cohort identity/scaffold leakage")
    if {dataset.rows[i]["scaffold_group"] for i in train} & {dataset.rows[i]["scaffold_group"] for i in calibration}:
        raise ValueError("Training/calibration scaffold leakage")
    return list(train), list(calibration)


def _scaffold_folds(dataset, train):
    groups = [dataset.rows[i]["scaffold_group"] for i in train]
    if len(set(groups)) < FOLDS:
        raise ValueError("At least three independent training scaffolds are required")
    folds = []
    for number, (fit, validation) in enumerate(GroupKFold(n_splits=FOLDS).split(train, groups=groups)):
        fit_rows = [train[i] for i in fit]
        validation_rows = [train[i] for i in validation]
        if {groups[i] for i in fit} & {groups[i] for i in validation}:
            raise ValueError("Cross-validation scaffold leakage")
        folds.append({"fold": number, "train_indices": fit_rows, "validation_indices": validation_rows})
    if Counter(i for f in folds for i in f["validation_indices"]) != Counter(train):
        raise ValueError("Every training compound must have exactly one out-of-fold prediction")
    return folds


def _forest(method, seed, n_jobs):
    cls = RandomForestRegressor if method == "rf" else ExtraTreesRegressor
    return cls(**CANDIDATES[method], random_state=seed, n_jobs=n_jobs)


def _residuals(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if (truth.ndim != 1 or not len(truth) or truth.shape != prediction.shape
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError("Calibration needs nonempty matched finite log10 labels and predictions")
    errors = np.abs(truth - prediction)
    return errors, {key: float(np.quantile(errors, quantile, method="higher")) for key, quantile in QUANTILES.items()}


def _stress(log10_nominal, width, parameter, calibration_n, level):
    if (parameter not in PARAMETERS or level not in QUANTILES or calibration_n < 1
            or not math.isfinite(log10_nominal) or not math.isfinite(width) or width < 0):
        raise ValueError("Invalid finite stress range inputs")
    try:
        lower, nominal, upper_raw = [10. ** (log10_nominal + step) for step in (-width, 0., width)]
    except OverflowError as error:
        raise ValueError("Nonfinite physical stress range") from error
    if not all(math.isfinite(x) and x > 0 for x in (lower, nominal, upper_raw)):
        raise ValueError("Nonpositive or nonfinite physical stress range")
    if parameter == "fraction_unbound" and nominal > 1:
        raise ValueError("Fraction unbound nominal exceeds its physical bound")
    upper = min(upper_raw, 1.) if parameter == "fraction_unbound" else upper_raw
    return {"lower": lower, "central": nominal, "upper": upper,
            "unit": PARAMETERS[parameter][1], "quantile": QUANTILES[level],
            "log10_residual_quantile": width, "quantile_method": "higher", "calibration_n": calibration_n,
            "upper_clipped_to_one": upper != upper_raw, "upper_before_physical_bound": upper_raw,
            "degenerate": lower == upper, "limited_calibration_count": calibration_n < 20,
            "renal_clearance_used_for_clipping": False, "clinical_confidence_interval": False,
            "interpretation": "Marginal empirical engineering stress range, not a clinical confidence interval or joint parameter distribution."}


def fit_adaptive_prior(wheel, *, cohort_manifest: Path, source_baseline_audit: Path,
                       output_dir: Path, project_root: Path, n_jobs: int = 4):
    """Freeze RF and train-CV-selected priors in exactly three new artifacts.

    Baseline audit membership is reused verbatim, including endpoint missing
    labels. Input/source/code changes prevent reuse. Existing incomplete output
    is retained and rejected. No evaluation path in the manifest is opened.
    """
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or not 1 <= n_jobs <= 4:
        raise ValueError("n_jobs must be an integer from 1 to 4")
    root = Path(project_root).resolve(strict=True)
    wheel = _inside(wheel, root)
    manifest_path = _inside(cohort_manifest, root)
    audit_path = _inside(source_baseline_audit, root)
    output = _inside(output_dir, root, exists=False)
    baseline = json.loads(audit_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    cohort = _cohort(manifest, baseline)
    if baseline.get("source", {}).get("source_sha256") != _sha(wheel):
        raise ValueError("PK dataset source SHA256 differs from baseline")
    seed = baseline["frozen_input"]["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Frozen forest seed must be an unsigned 32-bit integer")
    if baseline["forest_parameters"] != {**CANDIDATES["rf"], "random_state": seed,
                                         "n_jobs": baseline["forest_parameters"]["n_jobs"]}:
        raise ValueError("Frozen RF baseline hyperparameters are not the declared comparator")
    pins = {str(p.relative_to(root)): _sha(p) for p in (wheel, manifest_path, audit_path)}
    pins.update({name: _sha(_inside(name, root)) for name in CODE_FILES})
    _verify(baseline["source_pins"], root)
    pins.update(baseline["source_pins"])
    baseline_model_ref = baseline.get("model")
    if not isinstance(baseline_model_ref, dict):
        raise ValueError("Frozen RF model artifact is required for prediction-equivalence verification")
    _verify({baseline_model_ref["path"]: baseline_model_ref["sha256"]}, root)
    pins[baseline_model_ref["path"]] = baseline_model_ref["sha256"]
    frozen = {"version": VERSION, "source_pins": pins, "cohort": cohort, "n_jobs": n_jobs,
              "seed": seed, "candidates": deepcopy(CANDIDATES), "folds": FOLDS,
              "cv_grouping": "fixed GroupKFold on training Murcko scaffolds; no calibration/cohort rows",
              "selection_metric": "compound_equal_mean_absolute_log10_out_of_fold_error",
              "tie_breaker": "rf", "quantiles": QUANTILES, "quantile_method": "higher",
              "parameters": PARAMETERS,
              "software": {"sklearn": sklearn.__version__, "numpy": np.__version__, "rdkit": rdBase.rdkitVersion}}
    fingerprint = _digest(frozen)
    if output.exists():
        bundle_path = output / "predictions.json"
        if not bundle_path.is_file():
            raise FileExistsError("Incomplete adaptive-prior output is retained; use a new directory")
        bundle = json.loads(bundle_path.read_text())
        if bundle.get("input_fingerprint") != fingerprint:
            raise ValueError("Existing adaptive prior has different frozen inputs or code")
        _verify(pins, root)
        if (set(bundle.get("artifacts", {})) != {"model", "audit"}
                or bundle["artifacts"]["model"]["path"] != str((output / "model.joblib").relative_to(root))
                or bundle["artifacts"]["audit"]["path"] != str((output / "audit.json").relative_to(root))):
            raise ValueError("Adaptive artifacts do not belong to this frozen output directory")
        _verify({v["path"]: v["sha256"] for v in bundle["artifacts"].values()}, root)
        audit = json.loads(_inside(bundle["artifacts"]["audit"]["path"], root).read_text())
        if (audit.get("input_fingerprint") != fingerprint or _digest(audit.get("frozen_input")) != fingerprint
                or audit.get("model") != bundle["artifacts"]["model"]):
            raise ValueError("Adaptive artifact audit does not bind these frozen inputs/models")
        if _digest(bundle["predictions"]) != audit["predictions_digest"]:
            raise ValueError("Frozen adaptive predictions changed")
        return {**bundle, "audit": audit, "artifacts": {**bundle["artifacts"], "predictions": _ref(bundle_path, root)}, "reused": True}
    dataset = base.load_parameter_dataset(wheel)
    train, calibration = _frozen_membership(dataset, baseline, cohort)
    folds = _scaffold_folds(dataset, train)
    included = sorted(train + calibration)
    features = base._features([dataset.rows[i]["identity_smiles"] for i in included])
    position = {i: at for at, i in enumerate(included)}
    external = base._features([v["identity_smiles"] for v in cohort.values()])
    models, external_logs = {"rf": {}, "selected": {}}, {"rf": {}, "selected": {}}
    selection, labels, calibration_audit, cv_audit = {}, {}, {}, {}
    old_models = joblib.load(_inside(baseline_model_ref["path"], root))
    if not all(PARAMETERS[p][0] in old_models for p in ("total_clearance", "fraction_unbound")):
        raise ValueError("Frozen RF artifact must retain both CL and fu comparators")
    equivalence = {}
    for parameter, (endpoint, unit) in PARAMETERS.items():
        labeled = [i for i in train if dataset.rows[i]["labels_log10"][endpoint] is not None]
        held = [i for i in calibration if dataset.rows[i]["labels_log10"][endpoint] is not None]
        if not labeled or not held:
            raise ValueError("Endpoint lacks training or calibration labels: " + endpoint)
        labels[parameter] = {"endpoint": endpoint, "unit": unit, "input_scale": base.ENDPOINTS[endpoint]["input_scale"],
            "model_target_scale": "log10", "train_indices": labeled, "calibration_indices": held,
            "train_count": len(labeled), "calibration_count": len(held),
            "train_missing_indices": sorted(set(train) - set(labeled)),
            "calibration_missing_indices": sorted(set(calibration) - set(held))}
        cv_audit[parameter] = {}
        for method in CANDIDATES:
            oof, fold_audit = {}, []
            for fold in folds:
                fit_rows = [i for i in fold["train_indices"] if i in labeled]
                validation_rows = [i for i in fold["validation_indices"] if i in labeled]
                if not fit_rows or not validation_rows:
                    raise ValueError("Each scaffold fold needs endpoint labels in fit and validation: " + endpoint)
                model = _forest(method, seed, n_jobs)
                model.fit(features[[position[i] for i in fit_rows]], [dataset.rows[i]["labels_log10"][endpoint] for i in fit_rows])
                predicted = model.predict(features[[position[i] for i in validation_rows]])
                if not np.isfinite(predicted).all():
                    raise ValueError("Nonfinite out-of-fold prediction")
                for i, value in zip(validation_rows, predicted):
                    truth = dataset.rows[i]["labels_log10"][endpoint]
                    oof[i] = {"row_index": i, "fold": fold["fold"], "observed_log10": truth,
                              "predicted_log10": float(value), "absolute_log10_error": abs(float(value) - truth)}
                fold_audit.append({"fold": fold["fold"], "train_indices": fit_rows, "validation_indices": validation_rows})
            if set(oof) != set(labeled):
                raise ValueError("Incomplete out-of-fold endpoint predictions")
            cv_audit[parameter][method] = {"n": len(oof), "mean_absolute_log10_error": float(np.mean([r["absolute_log10_error"] for r in oof.values()])),
                                          "folds": fold_audit, "out_of_fold": [oof[i] for i in sorted(oof)]}
        selected = min(CANDIDATES, key=lambda method: (cv_audit[parameter][method]["mean_absolute_log10_error"], method != "rf"))
        selection[parameter] = {"method": selected, "selection_uses_calibration_labels": False,
            "selected_cv_mae": cv_audit[parameter][selected]["mean_absolute_log10_error"],
            "rf_cv_mae": cv_audit[parameter]["rf"]["mean_absolute_log10_error"],
            "cv_mae_improvement_vs_rf": cv_audit[parameter]["rf"]["mean_absolute_log10_error"] - cv_audit[parameter][selected]["mean_absolute_log10_error"]}
        fitted = {}
        for method in dict.fromkeys(("rf", selected)):
            model = _forest(method, seed, n_jobs)
            model.fit(features[[position[i] for i in labeled]], [dataset.rows[i]["labels_log10"][endpoint] for i in labeled])
            fitted[method] = model
        models["rf"][parameter], models["selected"][parameter] = fitted["rf"], fitted[selected]
        calibration_audit[parameter] = {}
        truth = [dataset.rows[i]["labels_log10"][endpoint] for i in held]
        for branch in ("rf", "selected"):
            model = models[branch][parameter]
            predicted = model.predict(features[[position[i] for i in held]])
            errors, widths = _residuals(truth, predicted)
            calibration_audit[parameter][branch] = {"indices": held, "n": len(held), "residual_quantiles": widths,
                "absolute_log10_residuals": errors.tolist(), "mean_absolute_log10_error": float(errors.mean()),
                "used_for_selection": False, "independent_test": False}
            external_logs[branch][parameter] = model.predict(external)
        if endpoint in old_models:
            difference = float(np.max(np.abs(old_models[endpoint].predict(external) - external_logs["rf"][parameter])))
            if difference > 1e-12:
                raise ValueError("Refitted RF does not reproduce the frozen baseline: " + parameter)
            equivalence[parameter] = {"status": "verified", "max_absolute_log10_difference": difference, "tolerance": 1e-12}
        else:
            equivalence[parameter] = {"status": "new_endpoint_with_same_RF_recipe", "reason": "Endpoint was not retained in frozen CL/fu model artifact"}
    predictions = {}
    for at, (cid, identity) in enumerate(cohort.items()):
        branches = {}
        for branch in ("rf", "selected"):
            ranges = {p: {level: _stress(float(external_logs[branch][p][at]), calibration_audit[p][branch]["residual_quantiles"][level],
                                         p, calibration_audit[p][branch]["n"], level) for level in QUANTILES} for p in PARAMETERS}
            branches[branch] = {"predict_nominal": {p: ranges[p]["90"]["central"] for p in PARAMETERS}, "ranges": ranges}
        predictions[cid] = {**identity, "methods": branches, "selected_models": {p: selection[p]["method"] for p in PARAMETERS},
            "source_kind": "predicted", "eligible_as_measured_evidence": False,
            "cohort_labels_used": False, "training_structure_overlap": False, "training_scaffold_overlap": False,
            "calibration_structure_overlap": False, "calibration_scaffold_overlap": False}
    audit = {"version": VERSION, "input_fingerprint": fingerprint, "frozen_input": frozen,
        "source_pins": pins, "source": dataset.provenance, "external_exclusions": cohort,
        "cohort_count": len(cohort), "train_indices": train, "calibration_indices": calibration,
        "train_count": len(train), "calibration_count": len(calibration), "label_audit": labels,
        "folds": folds, "cross_validation": cv_audit, "selection": selection,
        "calibration": calibration_audit, "rf_baseline_equivalence": equivalence,
        "calibration_is_independent_test": False, "cohort_evaluation_files_opened": False,
        "selection_uses_cohort_labels": False, "selection_uses_calibration_labels": False,
        "feature_specification": base.FEATURE_SPEC, "models_trained_sequentially": True,
        "baseline_split_changed": False, "clinical_confidence_intervals": False,
        "predictions_digest": _digest(predictions),
        "limitations": ["Training CV model selection is not independent performance validation.",
                        "Calibration is reserved for marginal residual ranges and diagnostics, never model selection.",
                        "80/90 labels denote empirical residual quantiles; no clinical or conformal coverage guarantee.",
                        "The published source is loaded as one table, but excluded cohort labels are never used in fits, CV, ranges or model selection."]}
    _verify(pins, root)
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "model.joblib"
    joblib.dump(models, model_path, compress=3)
    audit["model"] = _ref(model_path, root)
    _save(output / "audit.json", audit)
    artifacts = {"model": audit["model"], "audit": _ref(output / "audit.json", root)}
    bundle = {"version": VERSION, "input_fingerprint": fingerprint, "predictions": predictions, "artifacts": artifacts}
    _save(output / "predictions.json", bundle)
    _verify(pins, root)
    return {**bundle, "audit": audit, "artifacts": {**artifacts, "predictions": _ref(output / "predictions.json", root)}, "reused": False}
