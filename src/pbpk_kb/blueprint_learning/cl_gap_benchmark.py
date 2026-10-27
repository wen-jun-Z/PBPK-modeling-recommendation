"""Frozen common-split CL gap benchmark; no clinical observations are inputs.

Run directly with the project app-env Python. The dedicated TabPFN overlay is
loaded before scientific packages. This module never downloads model weights.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
for _key, _sub in {"HF_HOME": "hf", "TABPFN_MODEL_CACHE_DIR": "tabpfn",
                   "TORCH_HOME": "torch", "TRITON_CACHE_DIR": "triton",
                   "CUDA_CACHE_PATH": "cuda"}.items():
    os.environ[_key] = str(ROOT / "runtime/cache/cl-gap-benchmark" / _sub)
for _key in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TABPFN_DISABLE_TELEMETRY"):
    os.environ[_key] = "1"
sys.path[:0] = [str(ROOT / "runtime/foundation-python"), str(ROOT / "src"), str(ROOT / "scripts")]

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from pbpk_kb.blueprints import pk_parameter_learning as learning
from pbpk_kb.blueprint_learning.tanimoto_prior import TanimotoKernelRidge

VERSION = "common-split-cl-gap-benchmark-1"
RUN = "experiments/blueprint_design/rf_gap_comparison_20261012"
SPLIT = "experiments/blueprint_design/sampling_expansion_20261009/inputs/priors/parameter_split.json"
WHEEL = "artifacts/pk_predictors/pksmart/pksmart-3.0.1-py3-none-any.whl"
ANIMAL = "experiments/blueprint_design/blueprint_cl_refinement_animal_20261004/prior/animal_source.csv"
CHECKPOINT = "artifacts/pk_predictors/tabpfn_v2/tabpfn-v2-regressor.ckpt"
CHECKPOINT_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
PARAMETER = "human_CL_mL_min_kg"
METHODS = ("rf", "tabpfn", "complementary")
SEED = 20260929
SPECIES = ("dog", "monkey", "rat")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def _sources(root):
    with zipfile.ZipFile(root / WHEEL) as archive:
        payload = archive.read(learning.CSV_MEMBER)
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


def _inputs(root):
    split = read(root / SPLIT)
    raw = _sources(root)
    rows = {r["index"]: r for r in split["rows"]}
    train = []
    labels = []
    for i in split["train_indices"]:
        value, reason = learning._label(raw[i][PARAMETER], PARAMETER)
        if reason is None:
            train.append(i)
            labels.append(value)
    test = list(split["test_indices"])
    # Targets contain metadata only. No development/target observation is read.
    meta = read(root / RUN / "clinical_inputs.json")
    cases = meta.get("cases", meta.get("targets", meta)) if isinstance(meta, dict) else meta
    if isinstance(cases, dict):
        cases = [dict(v, compound_id=k) for k, v in cases.items()]
    targets = []
    for case in cases:
        name = case.get("drug", case.get("name"))
        frozen = split["external_exclusions"].get(name)
        if frozen is None:
            matches = [v for v in split["external_exclusions"].values()
                       if v["input_smiles"] == case.get("smiles") or
                       v["identity_smiles"] == case.get("identity_smiles")]
            if len(matches) != 1:
                raise ValueError("Target cannot be mapped uniquely to frozen exclusions: " + str(case))
            frozen = matches[0]
        targets.append({"case_id": case["case_id"], "compound_id": case["compound_id"],
                        "drug": name, "identity_smiles": frozen["identity_smiles"],
                        "scaffold_group": frozen["scaffold_group"]})
    if (len(train), len(test), len(targets)) != (1006, 100, 30):
        raise ValueError("Expected 1006 CL labels, 100 test structures and 30 targets")
    if len({c["compound_id"] for c in targets}) != 30:
        raise ValueError("Target compounds must be unique")
    groups = {r["scaffold_group"] for r in (rows[i] for i in train)}
    protected = [rows[i] for i in test] + targets
    if groups & {r["scaffold_group"] for r in protected}:
        raise ValueError("Frozen training/test/clinical scaffold isolation failed")
    return split, rows, train, np.asarray(labels), test, targets


def prepare(root, jobs):
    output = root / RUN
    split, rows, train, y, test, targets = _inputs(root)
    if sha(root / WHEEL) != split["source"]["source_sha256"]:
        raise ValueError("Frozen source wheel changed")
    if sha(root / CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Pinned official TabPFN checkpoint changed")
    pins = {p: sha(root / p) for p in (SPLIT, WHEEL, ANIMAL, CHECKPOINT,
            RUN + "/clinical_inputs.json", str(Path(__file__).relative_to(root)))}
    protocol = {
        "version": VERSION, "seed": SEED, "jobs": jobs, "methods": list(METHODS),
        "source_pins": pins, "train_structures": split["train_count"],
        "train_cl_n": len(train), "test_structures": len(test), "clinical_targets": targets,
        "train_cl_indices": train, "test_indices": test,
        "train_cl_scaffolds": [rows[i]["scaffold_group"] for i in train],
        "train_log10_cl": y.tolist(), "feature_specification": split["feature_specification"],
        "rf": {"n_estimators": 200, "min_samples_leaf": 3, "max_features": "sqrt",
               "random_state": SEED, "n_jobs": jobs},
        "tabpfn": {"official_package": "tabpfn==2.1.3", "n_estimators": 4,
                   "checkpoint": CHECKPOINT, "device": "cuda", "point_summary": "predictive_median",
                   "feature_adapter": "training-only variance: highest-variance 494 Morgan bits plus all six descriptors; ascending original-index order; ties by original index",
                   "input_dimension": 500, "hyperparameter_search": False},
        "complementary": {
            "formula": "equal log10 mean of fixed four-model base, animal-augmented four-model base, Tanimoto KRR(alpha=0.01)",
            "base_members": {"extra_trees": {"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.7},
                             "svr": {"C": 10.0, "epsilon": 0.1, "kernel": "rbf", "gamma": "scale", "scaling": "training-only StandardScaler"},
                             "xgb_depth3": {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "reg_lambda": 10.0},
                             "xgb_depth5": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "reg_lambda": 10.0}},
            "adaptation": "fixed historical recipes on common 1030 features; no CL/Vss/MRT projection or auxiliary human PK labels; not the full historical 1408-feature pipeline",
            "krr_features": "same 1024 binary Morgan bits; six descriptors omitted for Tanimoto",
            "animal_extra_information": "dog, monkey and rat CL from frozen PKSmart animal source; no human columns",
            "animal_training": "RF300 leaf2 max_features1.0; global exclusion of all 100 parameter-test and 30 clinical target identities/scaffolds; three human scaffold folds additionally exclude held-fold identities/scaffolds before animal predictions used as human training features; final animal fit predicts external targets",
            "hyperparameter_search": False},
        "parameter_test_policy": "Previously viewed frozen parameter holdout; recipes fixed before rerun. Labels are interpreted only after all three predictions are locked; retrospective method-development comparison, not fresh blind validation.",
        "clinical_observation_files_opened": False, "label_imputation": False,
        "max_model_fits": {"rf": 1, "tabpfn": 1, "complementary": 21},
        "max_training_seconds": 5400,
    }
    path = output / "parameter_protocol.json"
    if path.exists():
        existing = read(path)
        expected = {k: v for k, v in existing.items() if k != "created_at"}
        if expected != protocol:
            raise ValueError("Frozen training protocol differs; refusing to mix runs")
        return existing
    save(path, {**protocol, "created_at": now()})
    save(output / "parameter_state.json", {"version": VERSION, "status": "prepared",
                                           "created_at": now(), "methods": {}})
    return read(path)


def _verify(root, protocol):
    for name, digest in protocol["source_pins"].items():
        if sha(root / name) != digest:
            raise ValueError("Pinned source changed: " + name)


def _animal_features(root, rows, train, test, targets, x, prediction_x, jobs):
    from prepare_sampling_expansion_priors import complete_external_normalizer
    normalizer = complete_external_normalizer(learning.normalize_structure)
    protected = [rows[i] for i in test] + targets
    pid = {r["identity_smiles"] for r in protected}
    psc = {r["scaffold_group"] for r in protected}
    allowed, excluded = [], []
    with (root / ANIMAL).open() as stream:
        for i, raw in enumerate(csv.DictReader(stream)):
            try:
                item = {"row_index": i, **normalizer(raw["smiles_r"])}
            except (ValueError, RuntimeError) as error:
                excluded.append({"row_index": i, "reason": "normalization: " + str(error)})
                continue
            if item["identity_smiles"] in pid or item["scaffold_group"] in psc:
                excluded.append({"row_index": i, "reason": "protected_identity_or_scaffold"})
                continue
            labels = {}
            for species in SPECIES:
                raw_value = raw[species + "_CL_mL_min_kg"]
                try:
                    value = float(raw_value)
                except (ValueError, TypeError):
                    continue
                if math.isfinite(value) and value > 0:
                    labels[species] = math.log10(value)
            allowed.append({**item, "labels": labels})
    if not allowed:
        raise ValueError("Animal exclusion left no training data")
    ax = learning._features([r["identity_smiles"] for r in allowed])
    groups = np.asarray([rows[i]["scaffold_group"] for i in train])
    train_aux = np.empty((len(train), 3))
    prediction_aux = np.empty((len(prediction_x), 3))
    animal_models, folds = {}, []
    for fold, (_, held) in enumerate(GroupKFold(3).split(x, groups=groups)):
        held_ids = {rows[train[j]]["identity_smiles"] for j in held}
        held_scaffolds = set(groups[held])
        eligible = [j for j, r in enumerate(allowed) if r["identity_smiles"] not in held_ids and r["scaffold_group"] not in held_scaffolds]
        info = {"fold": fold, "held_human_indices": [train[j] for j in held],
                "animal_indices": [allowed[j]["row_index"] for j in eligible], "species_n": {}}
        for s, species in enumerate(SPECIES):
            ix = [j for j in eligible if species in allowed[j]["labels"]]
            if len(ix) < 10:
                raise ValueError("Insufficient isolated animal labels")
            model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                          max_features=1.0, random_state=SEED, n_jobs=jobs)
            model.fit(ax[ix], [allowed[j]["labels"][species] for j in ix])
            train_aux[held, s] = model.predict(x[held])
            info["species_n"][species] = len(ix)
        folds.append(info)
    final_species = {}
    for s, species in enumerate(SPECIES):
        ix = [j for j, r in enumerate(allowed) if species in r["labels"]]
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                      max_features=1.0, random_state=SEED, n_jobs=jobs)
        model.fit(ax[ix], [allowed[j]["labels"][species] for j in ix])
        prediction_aux[:, s] = model.predict(prediction_x)
        animal_models[species] = model
        final_species[species] = len(ix)
    return train_aux, prediction_aux, animal_models, {
        "source": ANIMAL, "allowed_count": len(allowed), "excluded": excluded,
        "allowed_animal_indices": [r["row_index"] for r in allowed],
        "global_protected_overlap": False, "held_human_fold_overlap": False,
        "folds": folds, "final_species_n": final_species, "human_animal_source_columns_used": False}


def _ensemble(x, y, prediction_x, jobs):
    import xgboost as xgb
    estimators = {
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                        max_features=.7, random_state=SEED, n_jobs=jobs),
        "svr": make_pipeline(StandardScaler(), SVR(C=10., epsilon=.1, kernel="rbf", gamma="scale")),
        **{"xgb_depth" + str(depth): xgb.XGBRegressor(n_estimators=500, max_depth=depth,
            learning_rate=.03, reg_lambda=10., objective="reg:squarederror", tree_method="hist",
            device="cuda:0", random_state=SEED, n_jobs=jobs) for depth in (3, 5)}}
    predictions = []
    for name, model in estimators.items():
        print(json.dumps({"event": "fit_fixed_member", "name": name, "features": x.shape[1]}), flush=True)
        model.fit(x, y)
        predictions.append(model.predict(prediction_x))
    return estimators, np.mean(predictions, axis=0)


def train(root, method, jobs):
    protocol = prepare(root, jobs)
    _verify(root, protocol)
    output = root / RUN
    split, rows, ix, y, test, targets = _inputs(root)
    x = learning._features([rows[i]["identity_smiles"] for i in ix])
    prediction_x = learning._features([rows[i]["identity_smiles"] for i in test] + [c["identity_smiles"] for c in targets])
    if x.shape != (1006, 1030) or not np.isfinite(x).all() or not np.isfinite(prediction_x).all():
        raise ValueError("Common feature contract failed")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires CUDA for TabPFN and XGBoost")
    torch.set_num_threads(jobs)
    started = time.monotonic()
    selected = METHODS if method == "all" else (method,)
    for name in selected:
        prior_path = output / "priors" / (name + ".json")
        if prior_path.exists():
            prior = read(prior_path)
            if prior["protocol_sha256"] != sha(output / "parameter_protocol.json"):
                raise ValueError("Existing prediction has a different protocol")
            continue
        state = read(output / "parameter_state.json")
        state.update(status="training", current_method=name, updated_at=now())
        save(output / "parameter_state.json", state)
        stamp = time.monotonic()
        audit = {}
        if name == "rf":
            model = RandomForestRegressor(**protocol["rf"])
            model.fit(x, y)
            pred = model.predict(prediction_x)
        elif name == "tabpfn":
            from tabpfn import TabPFNRegressor
            import importlib.metadata
            if importlib.metadata.version("tabpfn") != "2.1.3":
                raise ValueError("Expected official tabpfn==2.1.3")
            order = np.lexsort((np.arange(1024), -np.var(x[:, :1024], axis=0)))
            columns = np.sort(np.r_[order[:494], np.arange(1024, 1030)]).astype(int)
            estimator = TabPFNRegressor(n_estimators=4, random_state=SEED, device="cuda",
                model_path=str(root / CHECKPOINT), n_jobs=jobs)
            estimator.fit(x[:, columns], y)
            full = estimator.predict(prediction_x[:, columns], output_type="full", quantiles=[.05, .95])
            pred = np.asarray(full["median"])
            model = {"estimator": estimator, "feature_columns": columns}
            audit = {"feature_columns": columns.tolist(), "point_summary": "predictive_median",
                     "quantile05_log10": np.asarray(full["quantiles"][0]).tolist(),
                     "quantile95_log10": np.asarray(full["quantiles"][1]).tolist()}
        else:
            base_models, base = _ensemble(x, y, prediction_x, jobs)
            train_aux, prediction_aux, animal_models, animal_audit = _animal_features(
                root, rows, ix, test, targets, x, prediction_x, jobs)
            augmented_models, augmented = _ensemble(np.column_stack((x, train_aux)), y,
                                                    np.column_stack((prediction_x, prediction_aux)), jobs)
            kernel = TanimotoKernelRidge(alpha=.01).fit(x[:, :1024], y)
            kernel_pred = kernel.predict(prediction_x[:, :1024])
            pred = (base + augmented + kernel_pred) / 3.
            model = {"base": base_models, "animal": animal_models,
                     "augmented": augmented_models, "kernel": kernel}
            audit = {"animal": animal_audit, "components_log10": {
                "fixed_base": base.tolist(), "animal_augmented": augmented.tolist(),
                "tanimoto_krr": kernel_pred.tolist()}}
        if pred.shape != (130,) or not np.isfinite(pred).all() or not np.isfinite(10.**pred).all():
            raise ValueError("Prediction coverage or finiteness failed")
        model_path = output / "models" / (name + ".joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path, compress=3)
        _verify(root, protocol)
        prior = {"version": VERSION, "method": name, "unit": "mL/min/kg",
                 "parameter": "total_clearance", "protocol_sha256": sha(output / "parameter_protocol.json"),
                 "prediction_locked_at": now(), "test_labels_used_for_fit_or_selection": False,
                 "clinical_observation_files_opened": False, "train_count": len(ix),
                 "predictions": {c["compound_id"]: float(10.**v) for c, v in zip(targets, pred[100:])},
                 "targets": targets,
                 "parameter_test_predictions": [{"row_index": i, "predicted_log10": float(v)} for i, v in zip(test, pred[:100])],
                 "model_receipt": {"path": str(model_path.relative_to(root)), "sha256": sha(model_path)},
                 "elapsed_seconds": time.monotonic() - stamp, "audit": audit}
        save(prior_path, prior)
        state = read(output / "parameter_state.json")
        state["methods"][name] = {"status": "predictions_locked", "path": str(prior_path.relative_to(root)),
                                  "sha256": sha(prior_path), "elapsed_seconds": prior["elapsed_seconds"]}
        state.update(updated_at=now(), current_method=None)
        save(output / "parameter_state.json", state)
        print(json.dumps({"event": "predictions_locked", "method": name, "seconds": prior["elapsed_seconds"]}), flush=True)
        if time.monotonic() - started > protocol["max_training_seconds"]:
            raise RuntimeError("Predeclared training time budget exhausted")
    if all((output / "priors" / (name + ".json")).exists() for name in METHODS):
        score(root)


def score(root):
    output = root / RUN
    protocol = read(output / "parameter_protocol.json")
    _verify(root, protocol)
    priors = {m: read(output / "priors" / (m + ".json")) for m in METHODS}
    lock = {"version": VERSION, "all_methods_locked_at": now(), "parameter_test_labels_scored": False,
            "priors": {m: {"path": str((output / "priors" / (m + ".json")).relative_to(root)),
                            "sha256": sha(output / "priors" / (m + ".json"))} for m in METHODS}}
    lock_path = output / "parameter_predictions_locked.json"
    if not lock_path.exists():
        save(lock_path, lock)
    # First interpretation of held-out labels occurs after the persisted lock.
    raw = _sources(root)
    indices = protocol["test_indices"]
    y = []
    for i in indices:
        label, error = learning._label(raw[i][PARAMETER], PARAMETER)
        if error is not None:
            raise ValueError("Frozen CL test cohort has a missing/invalid label")
        y.append(label)
    metrics = {}
    rows = []
    for name, prior in priors.items():
        records = prior["parameter_test_predictions"]
        if [r["row_index"] for r in records] != indices:
            raise ValueError("Common test membership changed")
        pred = [r["predicted_log10"] for r in records]
        metrics[name] = learning._metrics(np.asarray(y), np.asarray(pred))
        rows.extend({"row_index": i, "method": name, "observed_log10": observed,
                     "predicted_log10": predicted} for i, observed, predicted in zip(indices, y, pred))
    save(output / "parameter_metrics.json", {"version": VERSION, "scored_at": now(),
         "prediction_lock": {"path": str(lock_path.relative_to(root)), "sha256": sha(lock_path)},
         "metrics": metrics, "predictions": rows, "interpretation": protocol["parameter_test_policy"]})
    state = read(output / "parameter_state.json")
    state.update(status="complete", completed_at=now(), current_method=None)
    save(output / "parameter_state.json", state)
    print(json.dumps({"event": "complete", "metrics": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "status"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    if root != ROOT or not 1 <= args.jobs <= 4:
        raise ValueError("Fixed project root and jobs 1..4 required")
    if args.command == "status":
        print(json.dumps(read(root / RUN / "parameter_state.json"), indent=2))
    elif args.command == "prepare":
        p = prepare(root, args.jobs)
        print(json.dumps({"status": "prepared", "train": p["train_cl_n"], "test": len(p["test_indices"])}))
    else:
        train(root, args.method, args.jobs)


if __name__ == "__main__":
    main()
