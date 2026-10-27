"""Small, structure-only PK learning benchmark with auditable scaffold isolation.

The PKSmart wheel supplies a published table, never trained weights/features.
Four columns already contain log10 labels; fup is converted to log10 here.
Missing/invalid labels are excluded endpoint by endpoint, never imputed.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import random
from typing import Mapping
import zipfile

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
import sklearn
from sklearn.ensemble import RandomForestRegressor


VERSION = "structure-only-pk-pilot-1.0"
SEED = 20260928
CSV_MEMBER = "pksmart/data/Train_data_log_transformed.csv"
ENDPOINTS = {
    "human_VDss_L_kg": {"unit": "L/kg", "input_scale": "log10"},
    "human_CL_mL_min_kg": {"unit": "mL/min/kg", "input_scale": "log10"},
    "human_thalf": {"unit": "h", "input_scale": "log10"},
    "human_mrt": {"unit": "h", "input_scale": "log10"},
    "human_fup": {"unit": "fraction", "input_scale": "linear"},
}
FEATURE_SPEC = {
    "fingerprint": "Morgan", "radius": 2, "bits": 1024,
    "use_chirality": False,
    "descriptors": ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"],
    "normalization": "RDKit FragmentParent; Uncharger; canonical tautomer; connectivity without stereochemistry/isotopes",
    "tautomer_limits": {"max_tautomers": 256, "max_transforms": 512, "incomplete_enumeration": "exclude"},
}


def normalize_structure(smiles: str) -> dict:
    """Return salt/tautomer-normalized connectivity and a conservative scaffold."""
    if not isinstance(smiles, str) or not smiles or len(smiles) > 4096 or any(c.isspace() for c in smiles):
        raise ValueError("Expected one bounded SMILES string without whitespace")
    # Suppress noisy candidate-kekulization logs only inside normalization;
    # invalid inputs and incomplete enumerations become explicit audit errors.
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(256)
        enumerator.SetMaxTransforms(512)
        candidates = enumerator.Enumerate(mol)
        if candidates.status != rdMolStandardize.TautomerEnumeratorStatus.Completed:
            raise ValueError("Incomplete bounded tautomer normalization: " + str(candidates.status))
        mol = enumerator.PickCanonical(candidates)
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    identity = Chem.MolToSmiles(mol, isomericSmiles=False)
    if not identity:
        raise ValueError("Empty standardized parent")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
    return {"identity_smiles": identity,
            "scaffold_group": "MURCKO:" + scaffold_smiles if scaffold_smiles else "ACYCLIC",
            "scaffold_smiles": scaffold_smiles}


def _label(value, endpoint):
    if value is None or str(value).strip().lower() in {"", "nan", "na", "null", "none"}:
        return None, "missing"
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None, "not_numeric"
    if not math.isfinite(number):
        return None, "not_finite"
    if endpoint == "human_fup":
        if not 0 < number <= 1:
            return None, "fraction_outside_(0,1]"
        return math.log10(number), None
    try:
        physical = 10 ** number
    except OverflowError:
        return None, "nonfinite_physical_value"
    if not math.isfinite(physical) or physical <= 0:
        return None, "nonpositive_or_nonfinite_physical_value"
    return number, None


@dataclass
class ParameterDataset:
    rows: list[dict]
    provenance: dict


def load_parameter_dataset(source: str | Path) -> ParameterDataset:
    """Read the table from a local CSV or wheel without importing its package."""
    source = Path(source)
    payload = source.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    if source.suffix == ".whl":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            payload = archive.read(CSV_MEMBER)
        member = CSV_MEMBER
    else:
        member = None
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    missing = {"smiles_r", *ENDPOINTS} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Dataset lacks columns: {sorted(missing)}")
    rows = []
    for index, raw in enumerate(reader):
        row = {"index": index, "csv_line": index + 2, "smiles_r": raw["smiles_r"],
               "labels_log10": {}, "label_exclusions": {}}
        try:
            row.update(normalize_structure(raw["smiles_r"]))
            row["structure_error"] = None
        except (ValueError, RuntimeError) as error:
            row.update(identity_smiles=None, scaffold_group=None, structure_error=str(error))
        for endpoint in ENDPOINTS:
            label, reason = _label(raw[endpoint], endpoint)
            row["labels_log10"][endpoint] = label
            if reason:
                row["label_exclusions"][endpoint] = reason
        rows.append(row)
    return ParameterDataset(rows, {
        "source_path": str(source.resolve()), "source_sha256": source_hash,
        "csv_member": member, "csv_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows), "index_definition": "zero-based data row, excluding CSV header",
        "endpoint_definitions": ENDPOINTS,
        "label_counts": {key: sum(row["labels_log10"][key] is not None for row in rows) for key in ENDPOINTS},
        "invalid_structure_indices": [row["index"] for row in rows if row["structure_error"]],
    })


def make_scaffold_split(dataset: ParameterDataset, *, exclude_smiles: Mapping[str, str] | None = None,
                        train_limit: int = 400, test_limit: int = 100, seed: int = SEED) -> dict:
    """Select scaffold-disjoint subsets, without inspecting endpoint labels.

    All acyclic molecules share one group. Duplicate connectivities retain the
    first CSV row, irrespective of label availability. Caps can leave rows unused.
    """
    if train_limit < 1 or test_limit < 1:
        raise ValueError("Both split limits must be positive")
    excluded = {name: {"input_smiles": smiles, **normalize_structure(smiles)}
                for name, smiles in (exclude_smiles or {}).items()}
    identities = {record["identity_smiles"] for record in excluded.values()}
    scaffolds = {record["scaffold_group"] for record in excluded.values()}
    groups, first_identity, dispositions = {}, {}, []
    for row in dataset.rows:
        identity, group = row["identity_smiles"], row["scaffold_group"]
        record = {"index": row["index"], "identity_smiles": identity, "scaffold_group": group}
        if row["structure_error"]:
            record.update(status="excluded", reason="invalid_or_unresolved_structure", structure_error=row["structure_error"])
        elif identity in identities:
            record.update(status="excluded", reason="external_test_identity")
        elif group in scaffolds:
            record.update(status="excluded", reason="external_test_scaffold")
        elif identity in first_identity:
            record.update(status="excluded", reason="duplicate_connectivity", representative_index=first_identity[identity])
        else:
            first_identity[identity] = row["index"]
            record.update(status="unused", reason="sample_limit")
            groups.setdefault(group, []).append(row["index"])
        dispositions.append(record)
    ordered_groups = sorted(groups)
    random.Random(seed).shuffle(ordered_groups)
    test_indices, test_groups = [], set()
    for group in ordered_groups:
        if len(test_indices) + len(groups[group]) <= test_limit:
            test_indices.extend(groups[group])
            test_groups.add(group)
    train_pool = [index for group in ordered_groups if group not in test_groups for index in groups[group]]
    random.Random(seed + 1).shuffle(train_pool)
    train_indices = sorted(train_pool[:train_limit])
    test_indices = sorted(test_indices)
    if not train_indices or not test_indices:
        raise ValueError("Insufficient independent scaffold groups for nonempty train and test sets")
    train_set, test_set = set(train_indices), set(test_indices)
    for record in dispositions:
        if record["index"] in train_set:
            record.update(status="train", reason=None)
        elif record["index"] in test_set:
            record.update(status="test", reason=None)
    train_scaffolds = {dataset.rows[i]["scaffold_group"] for i in train_indices}
    test_scaffolds = {dataset.rows[i]["scaffold_group"] for i in test_indices}
    assert not train_scaffolds & test_scaffolds
    return {
        "version": VERSION, "seed": seed, "train_limit": train_limit, "test_limit": test_limit,
        "train_indices": train_indices, "test_indices": test_indices,
        "train_count": len(train_indices), "test_count": len(test_indices),
        "train_scaffold_count": len(train_scaffolds), "test_scaffold_count": len(test_scaffolds),
        "scaffold_overlap": [], "external_exclusions": excluded,
        "deduplication": "first CSV row per standardized connectivity; no label-based selection",
        "acyclic_policy": "all acyclic parents in one conservative scaffold group",
        "selection_uses_labels": False, "rows": dispositions,
        "source": dataset.provenance,
    }


def _features(identities: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=False)
    features = np.empty((len(identities), 1030), dtype=np.float32)
    for index, identity in enumerate(identities):
        mol = Chem.MolFromSmiles(identity)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), features[index, :1024])
        features[index, 1024:] = [Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
                                 Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol)]
    return features


def _metrics(truth_log10, prediction_log10) -> dict:
    delta = np.asarray(prediction_log10) - np.asarray(truth_log10)
    if not len(delta):
        return {"n": 0, "gmfe": None, "within_2fold": None, "log10_rmse": None, "log10_mae": None,
                "geometric_bias": None, "linear_mae": None}
    return {"n": len(delta), "gmfe": float(10 ** np.mean(np.abs(delta))),
            "within_2fold": float(np.mean(np.abs(delta) <= math.log10(2))),
            "log10_rmse": float(np.sqrt(np.mean(delta ** 2))), "log10_mae": float(np.mean(np.abs(delta))),
            "geometric_bias": float(10 ** np.mean(delta)),
            "linear_mae": float(np.mean(np.abs(10 ** np.asarray(prediction_log10) - 10 ** np.asarray(truth_log10))))}


@dataclass
class PilotResult:
    audit: dict
    metrics: dict
    models: dict
    medians_log10: dict
    predictions: list[dict]

    def predict(self, smiles: str) -> dict:
        structure = normalize_structure(smiles)
        values = _features([structure["identity_smiles"]])
        training = [row for row in self.audit["rows"] if row["status"] == "train"]
        predictions = {}
        for endpoint, spec in ENDPOINTS.items():
            model = self.models.get(endpoint)
            predictions[endpoint] = {
                "random_forest": float(10 ** model.predict(values)[0]) if model is not None else None,
                "training_median": float(10 ** self.medians_log10[endpoint]) if model is not None else None,
                "unit": spec["unit"], "reason": None if model is not None else "no_training_labels",
            }
        return {**structure, "predictions": predictions, "source_kind": "predicted",
                "eligible_as_measured_evidence": False,
                "training_structure_overlap": any(row["identity_smiles"] == structure["identity_smiles"] for row in training),
                "training_scaffold_overlap": any(row["scaffold_group"] == structure["scaffold_group"] for row in training)}


def fit_parameter_pilot(dataset: ParameterDataset, split: dict, *, n_jobs: int = 4) -> PilotResult:
    """Fit the five fixed forests and the training-only log-median baseline."""
    if not 1 <= n_jobs <= 4:
        raise ValueError("n_jobs must be between 1 and 4")
    if split["source"]["csv_sha256"] != dataset.provenance["csv_sha256"]:
        raise ValueError("Split refers to a different dataset")
    train, test = split["train_indices"], split["test_indices"]
    if (not train or not test or len(set(train)) != len(train) or len(set(test)) != len(test)
            or any(not isinstance(i, int) or not 0 <= i < len(dataset.rows) for i in train + test)):
        raise ValueError("Split indices must be nonempty, unique, and in dataset bounds")
    if set(train) & set(test):
        raise ValueError("Train/test row leakage")
    if {dataset.rows[i]["scaffold_group"] for i in train} & {dataset.rows[i]["scaffold_group"] for i in test}:
        raise ValueError("Train/test scaffold leakage")
    reserved_scaffolds = {row["scaffold_group"] for row in split["external_exclusions"].values()}
    if any(dataset.rows[i]["scaffold_group"] in reserved_scaffolds for i in train + test):
        raise ValueError("External clinical-test scaffold leakage")
    selected = sorted(set(train + test))
    features = _features([dataset.rows[index]["identity_smiles"] for index in selected])
    position = {index: at for at, index in enumerate(selected)}
    models, medians, metrics, predictions, label_audit = {}, {}, {}, [], {}
    forest_parameters = {"n_estimators": 200, "min_samples_leaf": 3, "max_features": "sqrt",
                         "random_state": split["seed"], "n_jobs": n_jobs}
    for endpoint, spec in ENDPOINTS.items():
        labeled_train = [i for i in train if dataset.rows[i]["labels_log10"][endpoint] is not None]
        labeled_test = [i for i in test if dataset.rows[i]["labels_log10"][endpoint] is not None]
        label_audit[endpoint] = {"train_indices": labeled_train, "test_indices": labeled_test,
                                 "train_count": len(labeled_train), "test_count": len(labeled_test),
                                 "train_missing_indices": sorted(set(train) - set(labeled_train)),
                                 "test_missing_indices": sorted(set(test) - set(labeled_test))}
        if not labeled_train:
            metrics[endpoint] = {"unit": spec["unit"], "status": "no_training_labels", "random_forest": _metrics([], []),
                                 "training_median": _metrics([], [])}
            continue
        y_train = np.array([dataset.rows[i]["labels_log10"][endpoint] for i in labeled_train])
        model = RandomForestRegressor(**forest_parameters)
        model.fit(features[[position[i] for i in labeled_train]], y_train)
        median = float(np.median(y_train))
        models[endpoint], medians[endpoint] = model, median
        y_test = np.array([dataset.rows[i]["labels_log10"][endpoint] for i in labeled_test])
        predicted = model.predict(features[[position[i] for i in labeled_test]]) if labeled_test else np.array([])
        baseline = np.full(len(y_test), median)
        metrics[endpoint] = {"unit": spec["unit"], "status": "evaluated" if labeled_test else "no_test_labels",
                             "train_n": len(y_train), "random_forest": _metrics(y_test, predicted),
                             "training_median": _metrics(y_test, baseline)}
        for index, actual, estimate in zip(labeled_test, y_test, predicted):
            predictions.append({"row_index": index, "endpoint": endpoint,
                                "identity_smiles": dataset.rows[index]["identity_smiles"], "unit": spec["unit"],
                                "observed": float(10 ** actual), "random_forest": float(10 ** estimate),
                                "training_median": float(10 ** median), "fold_error": float(10 ** abs(estimate - actual)),
                                "baseline_fold_error": float(10 ** abs(median - actual))})
    audit = {**split, "feature_specification": FEATURE_SPEC, "forest_parameters": forest_parameters,
             "label_audit": label_audit, "hyperparameter_search": False, "pretrained_weights_used": False,
             "label_imputation": False, "software": {"rdkit": rdBase.rdkitVersion, "sklearn": sklearn.__version__,
                                                         "numpy": np.__version__},
             "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return PilotResult(audit, metrics, models, medians, predictions)


def run_parameter_pilot(source: str | Path, *, exclude_smiles: Mapping[str, str] | None = None,
                        train_limit: int = 400, test_limit: int = 100, seed: int = SEED, n_jobs: int = 4) -> PilotResult:
    dataset = load_parameter_dataset(source)
    split = make_scaffold_split(dataset, exclude_smiles=exclude_smiles, train_limit=train_limit,
                                test_limit=test_limit, seed=seed)
    return fit_parameter_pilot(dataset, split, n_jobs=n_jobs)
