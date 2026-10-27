"""One structure-only parameter prior shared by the clinical PBPK comparators.

The retained rich-model recipe is fixed before this cohort is supplied. Only
training-scaffold OOF residuals decide whether CL/Vss/MRT are projected onto
their consistency equation. Target observations are never accepted or opened.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import joblib
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from pbpk_kb.blueprints.pk_parameter_learning import CSV_MEMBER, _label
from . import rich_parameter_prior as rich

VERSION = "clinical-shared-rich-prior-1"
WHEEL = rich.WHEEL
EXPECTED_SOURCE_ROWS = 1283
SEED = 20260930
PARAMETERS = dict(rich.PARAMETERS)
UNITS = {"total_clearance": "mL/min/kg", "fraction_unbound": "fraction",
         "vss": "L/kg", "terminal_half_life": "h", "mrt": "h"}
METHODS = {p: ("rich_xgb_depth3",) if p == "fraction_unbound" else rich.ENSEMBLE_MEMBERS
           for p in PARAMETERS}
RF_RECIPE = {"n_estimators": 500, "random_state": 20261004, "min_samples_leaf": 2, "max_features": 1.0}
RF_PARAMETERS = ("total_clearance", "fraction_unbound")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inside(path, root, *, exists=False):
    path = (root / path).resolve(strict=exists)
    if path == root or not path.is_relative_to(root):
        raise ValueError("All sources and outputs must remain inside this project")
    return path


def normalize_structure(smiles):
    """Canonical stereochemical parent; generic Murcko scaffold for exclusion.

    No tautomer enumeration is performed: unlike the earlier connectivity
    normalizer, this preserves stereochemistry and avoids dropping rows when
    bounded tautomer enumeration cannot complete. Acyclic parents share a group.
    """
    if (not isinstance(smiles, str) or not smiles or len(smiles) > 4096
            or any(char.isspace() for char in smiles)):
        raise ValueError("Expected a bounded SMILES string without whitespace")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mol = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.SanitizeMol(mol)  # Fragment removal can invalidate RDKit's ring cache.
        identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        scaffold = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
        generic = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    if not identity:
        raise ValueError("Empty standardized parent")
    return {"identity_smiles": identity, "generic_scaffold_smiles": generic,
            "scaffold_group": "GENERIC_MURCKO:" + generic if generic else "ACYCLIC"}


def _cohort(targets):
    cohort = {}
    for target in targets:
        cid = target.get("compound_id")
        if not isinstance(cid, str) or not cid or cid in cohort:
            raise ValueError("Target compound_id values must be nonempty and unique")
        smiles = target.get("smiles")
        cohort[cid] = {"input_smiles": smiles, **normalize_structure(smiles)}
    if not cohort:
        raise ValueError("At least one target structure is required")
    return dict(sorted(cohort.items()))


def _training_data(source, cohort):
    with zipfile.ZipFile(source) as archive:
        payload = archive.read(CSV_MEMBER)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {"smiles_r", *(column for column, _ in PARAMETERS.values())}
    if not required <= set(reader.fieldnames or []):
        raise ValueError("Human parameter table lacks required columns")
    raw_rows = list(reader)
    if len(raw_rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"Expected {EXPECTED_SOURCE_ROWS} raw human parameter rows; got {len(raw_rows)}")
    protected_ids = {r["identity_smiles"] for r in cohort.values()}
    protected_groups = {r["scaffold_group"] for r in cohort.values()}
    rows, seen, duplicate_indices = [], {}, []
    labels = {p: {} for p in PARAMETERS}
    label_errors = {p: {} for p in PARAMETERS}
    for index, raw in enumerate(raw_rows):
        row = {"index": index, "csv_line": index + 2, "input_smiles": raw["smiles_r"]}
        try:
            row.update(normalize_structure(raw["smiles_r"]))
        except (ValueError, RuntimeError) as error:
            rows.append({**row, "status": "invalid_structure", "structure_error": str(error)})
            continue
        identity = row["identity_smiles"]
        duplicate = identity in seen
        if duplicate:
            duplicate_indices.append(index)
            row["representative_index"] = seen[identity]
        else:
            seen[identity] = index
        if identity in protected_ids:
            status = "excluded_target_identity"
        elif row["scaffold_group"] in protected_groups:
            status = "excluded_target_generic_scaffold"
        elif duplicate:
            status = "duplicate_parent"
        else:
            status = "train"
        rows.append({**row, "status": status})
        if status != "train":
            continue  # Excluded compounds' scalar labels are never interpreted.
        for parameter, (column, _) in PARAMETERS.items():
            value, error = _label(raw[column], column)
            if error:
                label_errors[parameter][str(index)] = error
            else:
                labels[parameter][index] = value
    counts = Counter(row["status"] for row in rows)
    audit = {"raw_row_count": len(rows), "expected_raw_row_count": EXPECTED_SOURCE_ROWS,
             "csv_member": CSV_MEMBER, "csv_sha256": hashlib.sha256(payload).hexdigest(),
             "row_disposition_counts": dict(counts), "normalization_failure_count": counts["invalid_structure"],
             "normalized_unique_parent_count": len(seen), "duplicate_parent_count": len(duplicate_indices),
             "duplicate_parent_indices": duplicate_indices, "train_count": counts["train"],
             "target_identity_exclusion_count": counts["excluded_target_identity"],
             "target_generic_scaffold_exclusion_count": counts["excluded_target_generic_scaffold"],
             "deduplication": "first raw CSV row per canonical stereochemical parent; independent of labels",
             "disposition_precedence": ["invalid_structure", "excluded_target_identity",
                                        "excluded_target_generic_scaffold", "duplicate_parent", "train"],
             "duplicate_count_includes_target_exclusions": True,
             "label_exclusions": label_errors, "rows": rows}
    return rows, labels, audit


def _fit(method, x, y, n_jobs, device):
    estimator = rich.make_estimator(method, SEED, n_jobs, device)
    estimator.fit(x, y)
    rich._verify_fitted_device(estimator, method, device)
    return estimator


def _projection(x, positions, labels, folds, n_jobs, device, emit):
    """Adapt fixed-ensemble OOF results to the existing consistency selector."""
    adapter = {"folds": folds, "selection": {}, "cv_progress": {}}
    oof_audit = {}
    for parameter in rich.CONSISTENCY_PARAMETERS:
        predictions, records = {}, {}
        for fold in folds:
            fit = [i for i in fold["train_indices"] if i in labels[parameter]]
            held = [i for i in fold["validation_indices"] if i in labels[parameter]]
            if not fit or not held:
                raise ValueError("A scaffold OOF fold lacks endpoint labels: " + parameter)
            member_predictions = []
            for method in METHODS[parameter]:
                emit(stage="consistency_oof", parameter=parameter, method=method, fold=fold["fold"])
                model = _fit(method, x[[positions[i] for i in fit]],
                             [labels[parameter][i] for i in fit], n_jobs, device)
                member_predictions.append(rich.physical_log_predictions(
                    model.predict(x[[positions[i] for i in held]]), parameter))
            predicted = np.mean(member_predictions, axis=0)
            predictions.update(zip(held, predicted.tolist()))
            records[str(fold["fold"])] = {"train_indices": fit, "validation_indices": held,
                "out_of_fold": [{"row_index": i, "observed_log10": labels[parameter][i],
                                 "predicted_log10": float(v)} for i, v in zip(held, predicted)]}
        if set(predictions) != set(labels[parameter]):
            raise ValueError("Incomplete training OOF coverage")
        indices = sorted(predictions)
        adapter["selection"][parameter] = {"oof_indices": indices,
            "selected_oof_log10": [predictions[i] for i in indices]}
        # crossfit_consistency reads this historical key only as a truth-label
        # carrier. These predictions came from METHODS, never old_extra_trees.
        adapter["cv_progress"][parameter] = {"old_extra_trees": records}
        oof_audit[parameter] = {"methods": list(METHODS[parameter]), "folds": records}
    projection = rich.crossfit_consistency(adapter)
    projection["parameters"] = list(projection["parameters"])
    return projection, oof_audit


def fit_shared_prior(targets, output_dir, project_root, n_jobs=8, device="auto", *, progress=None):
    """Return ``{parameters: {compound_id: {key: float}}, units, audit, ...}``.

    Write only model.joblib and predictions_audit.json inside output_dir. A
    completed unchanged run is reused; incomplete or changed artifacts fail
    closed. ``progress``, if given, receives one JSON-compatible event dict.
    """
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or not 1 <= n_jobs <= 8:
        raise ValueError("n_jobs must be an integer in [1, 8]")
    root = Path(project_root).resolve(strict=True)
    output, source = _inside(output_dir, root), _inside(WHEEL, root, exists=True)
    cohort = _cohort(targets)
    resolved_device = rich.choose_device(device)
    code_paths = [Path(__file__).resolve(), Path(rich.__file__).resolve()]
    frozen = {"version": VERSION, "targets": cohort, "source_sha256": _sha(source),
              "source_path": str(source.relative_to(root)), "seed": SEED, "n_jobs": n_jobs,
              "device": resolved_device, "methods": {p: list(v) for p, v in METHODS.items()},
              "rf_shared_predictor": {"recipe": RF_RECIPE, "parameters": list(RF_PARAMETERS),
                                      "consistency_projection": False},
              "recipes": {m: rich.RECIPES[m] for m in set(sum((list(v) for v in METHODS.values()), []))},
              "descriptor_names": [name for name, _ in rich.DESCRIPTORS],
              "software": {"rdkit": rdBase.rdkitVersion, "numpy": np.__version__,
                           "sklearn": rich.sklearn.__version__, "xgboost": rich.xgb.__version__},
              "code_sha256": {str(p.relative_to(root)): _sha(p) for p in code_paths}}
    fingerprint = _digest(frozen)
    model_path, audit_path = output / "model.joblib", output / "predictions_audit.json"
    if output.exists() and any(output.iterdir()):
        if {p.name for p in output.iterdir()} != {model_path.name, audit_path.name}:
            raise ValueError("Incomplete or unrelated prior outputs will not be overwritten")
        result = json.loads(audit_path.read_text())
        digest = result.pop("result_sha256", None)
        if (_digest(result) != digest or result.get("input_fingerprint") != fingerprint
                or result.get("model_sha256") != _sha(model_path)):
            raise ValueError("Existing prior sources, configuration, predictions or model changed")
        return {**result, "result_sha256": digest}
    started = time.monotonic()
    def emit(**event):
        if progress is not None:
            progress({"elapsed_seconds": time.monotonic() - started, **event})
    emit(stage="prepare_training")
    rows, labels, audit = _training_data(source, cohort)
    train = [r for r in rows if r["status"] == "train"]
    if any(len(labels[p]) < 2 for p in PARAMETERS):
        raise ValueError("Every parameter needs at least two eligible training labels")
    groups = [r["scaffold_group"] for r in train]
    if len(set(groups)) < 3:
        raise ValueError("At least three independent generic training scaffolds are required")
    positions = {r["index"]: j for j, r in enumerate(train)}
    folds = [{"fold": f, "train_indices": [train[i]["index"] for i in fit],
              "validation_indices": [train[i]["index"] for i in held]}
             for f, (fit, held) in enumerate(GroupKFold(3).split(train, groups=groups))]
    emit(stage="features", compounds=len(train) + len(cohort))
    x, feature_audit = rich.rich_features([r["identity_smiles"] for r in train] +
                                        [r["identity_smiles"] for r in cohort.values()])
    projection, oof = _projection(x, positions, labels, folds, n_jobs, resolved_device, emit)
    models, logs = {}, {}
    for parameter in PARAMETERS:
        indices = sorted(labels[parameter])
        models[parameter], predicted = {}, []
        for method in METHODS[parameter]:
            emit(stage="final_fit", parameter=parameter, method=method)
            model = _fit(method, x[[positions[i] for i in indices]],
                         [labels[parameter][i] for i in indices], n_jobs, resolved_device)
            models[parameter][method] = model
            predicted.append(rich.physical_log_predictions(model.predict(x[len(train):]), parameter))
        logs[parameter] = np.mean(predicted, axis=0)
    if projection["enabled"]:
        corrected = rich.project_consistency(np.column_stack([logs[p] for p in rich.CONSISTENCY_PARAMETERS]),
                                             projection["inference_residual_variances"])
        logs.update({p: corrected[:, j] for j, p in enumerate(rich.CONSISTENCY_PARAMETERS)})
    rf_models, rf_logs = {}, {}
    for parameter in RF_PARAMETERS:
        emit(stage="final_fit", parameter=parameter, method="rf_shared_predictor")
        indices = sorted(labels[parameter])
        model = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("model", RandomForestRegressor(**RF_RECIPE, n_jobs=n_jobs))])
        model.fit(x[[positions[i] for i in indices]], [labels[parameter][i] for i in indices])
        rf_models[parameter] = model
        rf_logs[parameter] = rich.physical_log_predictions(model.predict(x[len(train):]), parameter)
    with np.errstate(over="ignore", under="ignore"):
        parameters = {cid: {p: float(10. ** logs[p][j]) for p in PARAMETERS} for j, cid in enumerate(cohort)}
        rf_parameters = {cid: {p: float(10. ** rf_logs[p][j]) for p in RF_PARAMETERS} for j, cid in enumerate(cohort)}
    if any(not np.isfinite(v) or v <= 0 for collection in (parameters, rf_parameters)
           for values in collection.values() for v in values.values()):
        raise ValueError("Predictions must be finite and positive in physical units")
    audit.update(target_count=len(cohort), targets=cohort, frozen_input=frozen, features=feature_audit,
        parameter_training_indices={p: sorted(v) for p, v in labels.items()}, folds=folds, consistency_oof=oof,
        consistency_projection=projection, target_observation_files_opened=False, target_labels_used=False,
        training_structure_overlap=False, training_generic_scaffold_overlap=False,
        model_selection="fixed retained recipe; no model or hyperparameter selection for this cohort",
        rf_shared_predictor={"training_indices": {p: sorted(labels[p]) for p in RF_PARAMETERS},
                             "consistency_projection_applied": False,
                             "interpretation": "Fixed RandomForest comparator; not an official PKSmart reproduction"},
        ensemble_weighting="uniform mean of log10 predictions (geometric mean in physical units)",
        normalization="RDKit FragmentParent + Uncharger + canonical isomeric SMILES; no tautomer enumeration",
        differences_from_retained_prior=["Refit all eligible source rows after all target generic-scaffold exclusions.",
            "Retain stereochemical parent identities; exclude generic rather than atom-typed Murcko scaffolds.",
            "Fixed endpoint models; three-fold OOF is used only to re-estimate and select consistency projection.",
            "No calibration set, residual intervals or baseline-model fitting."],
        elapsed_seconds=time.monotonic() - started)
    if _sha(source) != frozen["source_sha256"] or any(_sha(p) != frozen["code_sha256"][str(p.relative_to(root))] for p in code_paths):
        raise ValueError("Prior source or code changed during training")
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": VERSION, "input_fingerprint": fingerprint, "models": models,
                 "rf_shared_predictor": rf_models,
                 "methods": frozen["methods"], "units": UNITS, "features": feature_audit,
                 "consistency_projection": projection, "normalization": audit["normalization"]}, model_path, compress=3)
    result = {"version": VERSION, "input_fingerprint": fingerprint, "parameters": parameters,
              "rf_parameters": rf_parameters, "units": UNITS,
              "audit": audit, "model_path": str(model_path), "audit_path": str(audit_path), "model_sha256": _sha(model_path)}
    result["result_sha256"] = _digest(result)
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    emit(stage="completed", targets=len(cohort))
    return result
