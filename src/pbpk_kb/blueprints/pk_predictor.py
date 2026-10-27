"""Run the pinned, local PKSmart predictor as evidence, never as measurements.

The PBPK application keeps its existing environment. A small isolated Python
3.10 worker loads the authors' model package from runtime/pksmart-env. All
prediction inputs travel through stdin, and model/package checksums are checked
before unpickling official weights. No external prediction service is used.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


VERSION = "local-pksmart-evidence-1.0"
PACKAGE_VERSION = "3.0.1"
WHEEL_NAME = "pksmart-3.0.1-py3-none-any.whl"
WHEEL_SHA256 = "0226969196cade1954b8355b600132d32a709fc7f12ef71a5db102f52de84743"
WHEEL_BYTES = 42950001
ARTIFACT_DIRECTORY = "artifacts/pk_predictors/pksmart"
ENV_DIRECTORY = "runtime/pksmart-env"
COMPATIBILITY_NAME = "dimorphite_2.0.2"
COMPATIBILITY_WHEEL = "dimorphite_dl-2.0.2-py3-none-any.whl"
COMPATIBILITY_SHA256 = "44125dfe2a5e1a37239f61914428d7f51d8fa61ec701a1337e71c293514133a8"
COMPATIBILITY_DIRECTORY = "upstream_diagnostics/dimorphite_2_0_2"
ENDPOINTS = {
    "vdss": ("VDss_L_kg", "human_VDss_L_kg", "Vd", "L/kg", "vdss_iv"),
    "cl": ("CL_mL_min_kg", "human_CL_mL_min_kg", "CL", "mL/min/kg", "cl_systemic_iv"),
    "fu": ("Fraction_unbound_in_plasma_(fup)", "human_fup", "fup", "fraction", "fu_plasma"),
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path, root, *, must_exist=True):
    path = (root / path).resolve(strict=must_exist)
    if not path.is_relative_to(root) or path == root:
        raise ValueError("PK predictor resources must stay inside the project")
    return path


def _validate_request(smiles, compound_id):
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 4096:
        raise ValueError("Provide one nonempty SMILES string of at most 4096 characters")
    if any(char.isspace() for char in smiles):
        raise ValueError("SMILES must not include names, spaces, or line breaks")
    if not isinstance(compound_id, str) or not compound_id.strip() or len(compound_id) > 256:
        raise ValueError("Provide the identified compound_id; predictions cannot resolve an identity")


def _normalise_record(endpoint, raw, audit, provenance, compound_id):
    value_key, _, prefix, unit, definition = ENDPOINTS[endpoint]
    value = float(raw[value_key])
    if not math.isfinite(value) or value <= 0 or (endpoint == "fu" and value > 1):
        raise ValueError(f"PKSmart returned an invalid physical prediction for {endpoint}")
    if endpoint == "fu":
        fe_key = "Fraction_unbound_in_plasma_(fup)_folderror"
        lower_key = "Fraction_unbound_in_plasma_(fup)_lowerbound"
        upper_key = "Fraction_unbound_in_plasma_(fup)_upperbound"
    else:
        title = "Volume_of_distribution_(VDss)" if endpoint == "vdss" else "Clearance_(CL)"
        fe_key, lower_key, upper_key = title + "_folderror", title + "_lowerbound", title + "_upperbound"
    fold_error = float(raw[fe_key])
    valid_error = math.isfinite(fold_error) and fold_error >= 1
    uncertainty = {
        "kind": "similarity_conditioned_fold_error" if valid_error else "unknown",
        "label": "model-estimated error magnitude; not a confidence interval or calibrated prediction interval",
        "fold_error": fold_error if valid_error else None,
        "coverage_probability": None, "calibrated_prediction_interval": False,
        "valid_as_measurement_interval": False,
        "nominal_lower_bound": value / fold_error if valid_error else None,
        "nominal_upper_bound": min(1., value * fold_error) if valid_error and endpoint == "fu"
                               else value * fold_error if valid_error else None,
        "upstream_lower_bound": float(raw[lower_key]),
        "upstream_upper_bound": float(raw[upper_key]),
        "bound_review": ("Upstream 3.0.1 clips the fold-error multiplier before multiplying fu; "
                         "nominal upper bound here instead clips value*fold_error to 1, explicitly not a validated interval")
                         if endpoint == "fu" else "upstream multiplicative display range; no probability coverage claim",
    }
    similarity = audit["mean_top5_similarity_standardized"]
    return {
        "endpoint": endpoint, "value": value, "unit": unit, "definition": definition,
        "compound_id": compound_id, "source_kind": "predicted", "source": provenance,
        "context": {"species": "human", "population": "unknown_aggregated_published_human_dataset",
                    "matrix": "plasma", "route": "not_applicable" if endpoint == "fu" else "iv",
                    "dosing": "not_applicable" if endpoint == "fu" else "unknown_dataset_regimens",
                    "age_sex_disease_context_available": False},
        "training_overlap": audit["training_overlap"], "training_overlap_details": audit,
        "applicability": "in_domain" if similarity >= .25 else "out_of_domain",
        "applicability_details": {
            "criterion": "project chemical-similarity screen: mean top-5 Morgan radius2/2048 Tanimoto of standardized query to endpoint-labelled training structures >=0.25",
            "threshold": .25, "similarity": similarity,
            "threshold_source": "PKSmart 3.0.1 software warning threshold; not a validated clinical applicability threshold",
            "scope": "chemical similarity only; not patient, dose, disease, nonlinear-PK or calibration validity",
            "upstream_package_warning_threshold": .25,
            "upstream_and_project_domains_identical": False,
            "training_overlap_can_inflate_similarity": audit["training_overlap"],
        },
        "uncertainty": uncertainty,
        "eligible_as_measured_evidence": False,
        "independent_validation_evidence": False,
        "automatic_parameter_override_allowed": False,
    }


def _verified_package(root):
    """Verify actual installed Python/model bytes against the official wheel."""
    distribution = importlib.metadata.distribution("pksmart")
    if distribution.version != PACKAGE_VERSION:
        raise ValueError("PKSmart package version differs from the pinned release")
    package = Path(distribution.locate_file("pksmart")).resolve(strict=True)
    environment = _inside(ENV_DIRECTORY, root)
    if not package.is_relative_to(environment):
        raise ValueError("PKSmart must be loaded from its isolated project environment")
    wheel = _inside(f"{ARTIFACT_DIRECTORY}/{WHEEL_NAME}", root)
    if wheel.stat().st_size != WHEEL_BYTES or _sha(wheel) != WHEEL_SHA256:
        raise ValueError("Official PKSmart wheel checksum mismatch")
    receipts = []
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if not member.filename.startswith("pksmart/") or member.is_dir() or member.filename.endswith(".DS_Store"):
                continue
            relative = Path(member.filename).relative_to("pksmart")
            path = (package / relative).resolve(strict=True)
            if not path.is_relative_to(package):
                raise ValueError("Package member escapes its pinned package")
            expected = hashlib.sha256(archive.read(member)).hexdigest()
            if _sha(path) != expected:
                raise ValueError("Installed PKSmart package/model differs from official wheel: " + str(relative))
            receipts.append({"path": str(path.relative_to(root)), "sha256": expected, "bytes": member.file_size})
    versions = {name: importlib.metadata.version(name) for name in (
        "pksmart", "numpy", "pandas", "rdkit", "scikit-learn", "mordred", "dimorphite-dl", "scipy", "joblib")}
    lock_path = _inside(f"{ARTIFACT_DIRECTORY}/requirements.lock", root)
    for line in lock_path.read_text().splitlines():
        pinned = line.split(" --hash=", 1)[0]
        name, version = pinned.split("==", 1)
        if importlib.metadata.version(name) != version:
            raise ValueError("Predictor dependency version differs from installed lock: " + name)
    return package, receipts, versions


def _verified_compatibility(root, name):
    if name is None:
        return None
    if name != COMPATIBILITY_NAME:
        raise ValueError("Unknown explicitly requested preprocessing compatibility")
    base = _inside(ARTIFACT_DIRECTORY, root)
    wheel = _inside(base / COMPATIBILITY_WHEEL, root)
    if wheel.stat().st_size != 39888 or _sha(wheel) != COMPATIBILITY_SHA256:
        raise ValueError("Preprocessing compatibility wheel checksum mismatch")
    directory = _inside(base / COMPATIBILITY_DIRECTORY, root)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if not member.filename.startswith("dimorphite_dl/") or member.is_dir():
                continue
            installed = _inside(directory / member.filename, root)
            if not installed.is_relative_to(directory) or _sha(installed) != hashlib.sha256(archive.read(member)).hexdigest():
                raise ValueError("Compatibility code differs from official Dimorphite 2.0.2 wheel")
    return {"name": name, "directory": str(directory.relative_to(root)),
            "source_url": "https://pypi.org/project/dimorphite-dl/2.0.2/",
            "wheel_path": str(wheel.relative_to(root)), "wheel_sha256": COMPATIBILITY_SHA256,
            "reason": "Explicit compatibility branch: upstream 2.0.1 returns no protonation for rifampicin",
            "training_preprocessing_equivalence_verified": False,
            "original_environment_dependency_modified": False,
            "model_weights_modified": False, "prediction_is_exploratory": True}


def _training_audit(standardized, package, input_smiles=None):
    import pandas as pd
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    training = pd.read_csv(package / "data/Train_data_log_transformed.csv")
    query = Chem.MolFromSmiles(standardized)
    if query is None:
        raise ValueError("PKSmart standardized structure is invalid")
    query_key = Chem.MolToInchiKey(query)
    query_canonical = Chem.MolToSmiles(query, canonical=True, isomericSmiles=True)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(query, 2, nBits=2048)
    original = Chem.MolFromSmiles(input_smiles or standardized)
    heavy_atoms = sorted(atom.GetAtomicNum() for atom in original.GetAtoms() if atom.GetAtomicNum() > 1)
    def neutral_tautomer_key(molecule):
        neutral = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.Cleanup(molecule))
        canonical = rdMolStandardize.TautomerEnumerator().Canonicalize(neutral)
        return Chem.MolToInchiKey(canonical)
    original_neutral_key = neutral_tautomer_key(original)
    structures = []
    for index, row in training.iterrows():
        molecule = Chem.MolFromSmiles(row["smiles_r"])
        if molecule is None:
            raise ValueError("An endpoint training structure cannot be audited")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        key = Chem.MolToInchiKey(molecule)
        exact = canonical == query_canonical or key == query_key
        same_connectivity = key.split("-")[0] == query_key.split("-")[0]
        same_heavy_atoms = sorted(atom.GetAtomicNum() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1) == heavy_atoms
        same_neutral_tautomer = same_heavy_atoms and neutral_tautomer_key(molecule) == original_neutral_key
        similarity = float(DataStructs.TanimotoSimilarity(
            fingerprint, AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)))
        structures.append((int(index), exact, same_connectivity, similarity, same_neutral_tautomer))
    audits = {}
    for endpoint, (_, column, _, _, _) in ENDPOINTS.items():
        relevant = [row for row in structures if pd.notna(training.iloc[row[0]][column])]
        top = sorted((row[3] for row in relevant), reverse=True)[:5]
        if len(top) != 5:
            raise ValueError("Insufficient endpoint-specific training structures for applicability audit")
        exact = [row[0] for row in relevant if row[1]]
        connectivity = [row[0] for row in relevant if row[2]]
        neutral_tautomer = [row[0] for row in relevant if row[4]]
        audits[endpoint] = {
            "training_overlap": bool(exact or connectivity or neutral_tautomer),
            "matching_rule": "canonical standardized structure/InChIKey; connectivity or neutral-tautomer match conservatively flags overlap, never establishes independent validation",
            "exact_structure_row_indices": exact, "same_connectivity_row_indices": connectivity,
            "neutral_tautomer_row_indices": neutral_tautomer,
            "neutral_tautomer_identity_caution": "Neutralization and tautomer canonicalization can erase some double-bond stereochemistry; this match is a conservative overlap flag, not proof of preprocessing equivalence",
            "training_label_column": column, "endpoint_training_compound_count": len(relevant),
            "mean_top5_similarity_standardized": sum(top) / 5,
            "top5_similarity_standardized": top,
            "training_data_sha256": _sha(package / "data/Train_data_log_transformed.csv"),
            "overlap_excludes_use_as_independent_validation": True,
        }
    return audits, query_key


def _worker(payload):
    root = Path(payload["project_root"]).resolve(strict=True)
    smiles, compound_id = payload["smiles"], payload["compound_id"]
    _validate_request(smiles, compound_id)
    # Limit only this newly created worker, never inspect other processes.
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, cpus[:2])
    package, receipts, versions = _verified_package(root)
    compatibility = _verified_compatibility(root, payload.get("preprocessing_compatibility"))
    if compatibility is not None:
        sys.path.insert(0, str(root / compatibility["directory"]))
    import pksmart.main as upstream
    from mordred import Calculator

    # The official default starts a machine-wide descriptor process pool.
    # Change only its parallelism, leaving the exact descriptor/model functions.
    pandas_original = Calculator.pandas
    def bounded_pandas(self, mols, *args, **kwargs):
        kwargs.update(nproc=1, quiet=True)
        return pandas_original(self, mols, *args, **kwargs)
    Calculator.pandas = bounded_pandas
    captured = io.StringIO()
    upstream.logger.remove()
    started = time.monotonic()
    with redirect_stdout(captured), redirect_stderr(captured):
        standardized = upstream.standardize(smiles)
        if standardized == "Cannot_do":
            raise ValueError("Official PKSmart structure standardization failed")
        frame = upstream.predict_pk_params(smiles)
        if len(frame) != 1:
            raise ValueError("Expected exactly one official PKSmart prediction")
        raw = json.loads(frame.to_json(orient="records", double_precision=15))[0]
        audits, inchi_key = _training_audit(standardized, package, smiles)
    model_receipts = {Path(r["path"]).name: r for r in receipts}
    provenance = {
        "predictor": "PKSmart", "version": PACKAGE_VERSION, "adapter_version": VERSION,
        "source_url": "https://pypi.org/project/pksmart/3.0.1/",
        "official_repository": "https://github.com/srijitseal/PKSmart",
        "package_repository": "https://github.com/Manas02/pksmart-pip",
        "publication": "https://doi.org/10.1186/s13321-025-01066-5",
        "wheel_path": f"{ARTIFACT_DIRECTORY}/{WHEEL_NAME}", "wheel_sha256": WHEEL_SHA256,
        "package_license": "not_declared_in_PyPI_3.0.1_metadata_or_wheel",
        "separate_author_Zenodo_release_license": "CC-BY-4.0 (10611607; not asserted to license this different package)",
        "execution": "local_cpu_no_service_or_LLM", "dependency_versions": versions,
        "implementation_adjustment": "Mordred descriptor parallelism set to one worker; model and feature bytes unchanged",
        "population_and_individual_context": "not specified by structure-only predictor",
        "preprocessing_compatibility": compatibility,
    }
    records = []
    for endpoint, (_, column, _, _, _) in ENDPOINTS.items():
        source = {**provenance, "model": model_receipts[f"log_{column}_withanimaldata_artificial_model_FINAL.sav"],
                  "training_data": model_receipts["Train_data_log_transformed.csv"]}
        record = _normalise_record(endpoint, raw, audits[endpoint], source, compound_id)
        record.update(preprocessing_compatibility=payload.get("preprocessing_compatibility"),
                      exploratory=compatibility is not None, eligible_as_constraint=False)
        records.append(record)
    return {
        "status": "completed", "records": records, "raw_prediction": raw,
        "identity": {"compound_id": compound_id, "input_smiles": smiles,
                     "standardized_smiles": standardized, "standardized_inchi_key": inchi_key,
                     "input_identity_reference": payload.get("input_reference"),
                     "compound_id_structure_consistency_verified_by_predictor": False},
        "runtime": {"elapsed_seconds": time.monotonic() - started, "cpu_affinity_limit": 2,
                    "descriptor_workers": 1, "gpu_used": False, "offline_prediction": True},
        "package_artifacts": receipts,
        "claim_boundary": "Structure-based PK prediction only; not measured evidence, organ mechanism identification or independent validation",
    }


def predict_pk(smiles, compound_id, project_root, *, input_reference=None, timeout_seconds=180,
               preprocessing_compatibility=None):
    """Return {records, raw_prediction, identity, provenance} from local weights.

    Does not download, install, fit or modify any PBPK model. The caller must
    verify the input structure identity and choose the use of predicted evidence.
    """
    _validate_request(smiles, compound_id)
    if preprocessing_compatibility not in (None, COMPATIBILITY_NAME):
        raise ValueError("Unknown explicitly requested preprocessing compatibility")
    if type(timeout_seconds) not in (int, float) or not 1 <= timeout_seconds <= 600:
        raise ValueError("Prediction timeout must be between 1 and 600 seconds")
    root = Path(project_root).resolve(strict=True)
    python = root / ENV_DIRECTORY / "bin/python"
    if not python.is_file() or not python.parent.resolve().is_relative_to(root):
        raise ValueError("The isolated project PKSmart environment is not installed")
    worker = Path(__file__).resolve(strict=True)
    if not worker.is_relative_to(root):
        raise ValueError("Worker source must belong to the same project")
    if input_reference is not None:
        if not isinstance(input_reference, dict) or not all(input_reference.get(k) for k in ("path", "sha256", "locator")):
            raise ValueError("Input identity reference needs path, sha256 and locator")
        identity = _inside(input_reference["path"], root)
        if _sha(identity) != input_reference["sha256"]:
            raise ValueError("Input identity evidence checksum mismatch")
    payload = {"project_root": str(root), "smiles": smiles, "compound_id": compound_id,
               "input_reference": input_reference, "preprocessing_compatibility": preprocessing_compatibility}
    environment = {
        "PATH": str(root / ENV_DIRECTORY / "bin") + ":/usr/bin:/bin", "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(root / ENV_DIRECTORY / "tmp"),
        "XDG_CACHE_HOME": str(root / ENV_DIRECTORY / "cache"),
    }
    completed = subprocess.run([str(python), "-B", str(worker), "--worker"],
        input=json.dumps(payload, allow_nan=False), text=True, capture_output=True,
        cwd=root, env=environment, timeout=timeout_seconds, check=False)
    if completed.returncode:
        raise RuntimeError("Local PKSmart prediction failed: " + completed.stderr[-3000:])
    if len(completed.stdout) > 2 * 1024 * 1024:
        raise ValueError("Unexpectedly large local prediction response")
    result = json.loads(completed.stdout)
    if result.get("status") != "completed" or len(result.get("records", [])) != 3:
        raise ValueError("Local predictor did not return all three registered endpoints")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smiles")
    parser.add_argument("--compound-id")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preprocessing-compatibility", choices=[COMPATIBILITY_NAME])
    args = parser.parse_args(argv)
    if args.worker:
        request = sys.stdin.read(65537)
        if len(request) > 65536:
            raise ValueError("Worker request exceeds size limit")
        result = _worker(json.loads(request))
    else:
        result = predict_pk(args.smiles, args.compound_id, args.project_root,
                            preprocessing_compatibility=args.preprocessing_compatibility)
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output is not None:
        output = _inside(args.output, args.project_root.resolve(), must_exist=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
