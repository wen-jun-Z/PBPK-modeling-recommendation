"""Input-only empirical endpoint references for blueprint decisions.

The frozen parameter estimators are never refitted. Only the independent
calibration rows provide residuals; held-out endpoint labels are never used.
These local residual distributions are development aids, not calibrated
clinical probabilities or additional PBPK predictions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np

from . import adaptive_parameter_prior as prior

VERSION = "input-only-endpoint-reference-1"
NEIGHBORS = 20
MIN_CALIBRATION = 10
BODY_WEIGHT_KG = 70.0
SOURCE_RUN = "experiments/blueprint_design/purpose_refinement_20261001"
TARGET_RUN = "experiments/blueprint_design/purpose_hundred_20260930"
ENDPOINTS = {
    "clearance": ("human_CL_mL_min_kg", "l/h", 4.2),
    "vss": ("human_VDss_L_kg", "l", 70.0),
    "terminal_half_life": ("human_thalf", "h", 1.0),
    "mrt": ("human_mrt", "h", 1.0),
}
CODE = "src/pbpk_kb/blueprint_learning/endpoint_reference_prior.py"


def endpoint_proxies(clearance_ml_min_kg, vss_l_kg):
    """Return dimensional proxies, never PBPK/compartment model endpoints."""
    values = (float(clearance_ml_min_kg), float(vss_l_kg))
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Endpoint proxies require finite positive parameters")
    clearance, vss = values[0] * 4.2, values[1] * BODY_WEIGHT_KG
    mrt = vss / clearance
    result = {"clearance": clearance, "vss": vss,
              "mrt": mrt, "terminal_half_life": math.log(2) * mrt}
    if any(not math.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("Nonfinite endpoint proxy after dimensional conversion")
    return result


def observation_log(label_log10, endpoint):
    """PKSmart endpoint labels are base-10 logs, not physical observations."""
    if endpoint not in ENDPOINTS:
        raise ValueError("Endpoint lacks an independent residual reference")
    if label_log10 is None:
        return None
    value = float(label_log10)
    if not math.isfinite(value):
        raise ValueError("Nonfinite calibration label")
    return value * math.log(10) + math.log(ENDPOINTS[endpoint][2])


def tanimoto(bits, pool):
    """Morgan fingerprint Tanimoto; continuous descriptors are excluded."""
    bits, pool = np.asarray(bits), np.asarray(pool)
    if bits.shape != (1024,) or pool.ndim != 2 or pool.shape[1] != 1024:
        raise ValueError("Similarity requires 1024 Morgan bits only")
    if not np.isin(bits, [0, 1]).all() or not np.isin(pool, [0, 1]).all():
        raise ValueError("Similarity inputs must be binary fingerprints")
    intersection = np.sum(pool * bits, axis=1, dtype=float)
    union = np.sum(pool, axis=1, dtype=float) + float(np.sum(bits)) - intersection
    return np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)


def empirical_distribution(proxy_value, calibration, similarities, *, local):
    """Select neighbors by structure only, and shift their signed residuals."""
    if not math.isfinite(proxy_value) or proxy_value <= 0:
        raise ValueError("Reference distribution needs a positive endpoint proxy")
    if len(calibration) != len(similarities):
        raise ValueError("Calibration identities/similarities must align")
    for record, similarity in zip(calibration, similarities):
        if not math.isfinite(similarity) or not 0 <= similarity <= 1:
            raise ValueError("Tanimoto must lie in [0, 1]")
        if not math.isfinite(record["residual_log"]):
            raise ValueError("Residual must be finite")
    ordered = sorted(zip(calibration, similarities), key=lambda pair: (
        -float(pair[1]) if local else 0,
        hashlib.sha256(pair[0]["identity_smiles"].encode()).hexdigest()))
    chosen = ordered[:NEIGHBORS] if local else ordered
    n = len(chosen)
    if n < MIN_CALIBRATION:
        return {"status": "unassessed", "reason": "fewer_than_10_independent_calibration_compounds",
                "available_calibration_n": n, "n": 0,
                "reference_values_log": [], "weights": [], "calibration_indices": [],
                "calibration_ids": [], "similarities": []}
    log_proxy = math.log(proxy_value)
    return {"status": "empirical_development_reference", "n": n,
            "available_calibration_n": len(calibration),
            "reference_values_log": [log_proxy + row["residual_log"] for row, _ in chosen],
            "weights": [1.0 / n] * n,
            "calibration_indices": [row["index"] for row, _ in chosen],
            "calibration_ids": [row["calibration_id"] for row, _ in chosen],
            "similarities": [float(similarity) for _, similarity in chosen]}


def _read_reuse(output, pins, fingerprint):
    existing = json.loads(output.read_text())
    recorded = existing.pop("artifact_digest", None)
    if recorded != prior._digest(existing):
        raise ValueError("Frozen endpoint reference artifact changed")
    if existing.get("source_pins") != pins or existing.get("input_fingerprint") != fingerprint:
        raise ValueError("Existing endpoint references have different frozen sources or code")
    return {**existing, "artifact_digest": recorded}


def fit_reference_prior(project_root, output_path, *, source_run=SOURCE_RUN, target_run=TARGET_RUN):
    """Create one provenance-bound JSON, with no fitting or target label reads.

    ``source_run`` contains the frozen adaptive prior; ``target_run`` supplies
    its original cohort manifest and frozen train/calibration split. Existing
    output is verified and reused; changes require a separate output path.
    """
    root = Path(project_root).resolve(strict=True)
    source_dir = prior._inside(source_run, root)
    target_dir = prior._inside(target_run, root)
    output = prior._inside(output_path, root, exists=False)
    audit_path = source_dir / "priors/audit.json"
    frozen_audit = json.loads(audit_path.read_text())
    source = prior._inside(frozen_audit["source"]["source_path"], root)
    manifest_path = target_dir / "cohort.json"
    baseline_path = target_dir / "priors/audit.json"
    # This strictly read-only reuse also verifies estimator and prediction hashes.
    fitted = prior.fit_adaptive_prior(source, cohort_manifest=manifest_path,
        source_baseline_audit=baseline_path, output_dir=source_dir / "priors",
        project_root=root, n_jobs=frozen_audit["frozen_input"]["n_jobs"])
    if not fitted["reused"]:
        raise ValueError("Endpoint references may only use an existing frozen parameter prior")
    audit = fitted["audit"]
    pins = dict(audit["source_pins"])
    for ref in fitted["artifacts"].values():
        pins[ref["path"]] = ref["sha256"]
    pins[CODE] = prior._sha(prior._inside(CODE, root))
    specification = {"version": VERSION, "source_pins": pins,
        "neighbors": NEIGHBORS, "minimum_calibration": MIN_CALIBRATION,
        "weighting": "uniform", "tie_breaker": "SHA256(normalized_identity_smiles)",
        "body_weight_kg": BODY_WEIGHT_KG,
        "log_scale": "natural", "unsupported_endpoints": "unassessed"}
    fingerprint = prior._digest(specification)
    prior._verify(pins, root)
    if output.exists():
        return _read_reuse(output, pins, fingerprint)
    baseline = json.loads(baseline_path.read_text())
    cohort = prior._cohort(json.loads(manifest_path.read_text()), baseline)
    dataset = prior.base.load_parameter_dataset(source)
    train, calibration = prior._frozen_membership(dataset, baseline, cohort)
    if train != audit["train_indices"] or calibration != audit["calibration_indices"]:
        raise ValueError("Parameter-prior train/calibration membership changed")
    for parameter in ("total_clearance", "vss"):
        labeled_train = audit["label_audit"][parameter]["train_indices"]
        if set(labeled_train) - set(train) or set(labeled_train) & set(calibration):
            raise ValueError("Calibration rows participated in estimator training")
    model_path = prior._inside(fitted["artifacts"]["model"]["path"], root)
    models = joblib.load(model_path)["selected"]
    included = sorted(train + calibration)
    x = prior.base._features([dataset.rows[i]["identity_smiles"] for i in included])
    positions = {i: j for j, i in enumerate(included)}
    cx = x[[positions[i] for i in calibration]]
    target_ids = sorted(cohort)
    tx = prior.base._features([cohort[cid]["identity_smiles"] for cid in target_ids])
    cp = {p: models[p].predict(cx) for p in ("total_clearance", "vss")}
    tp = {p: models[p].predict(tx) for p in ("total_clearance", "vss")}
    if not all(np.isfinite(arr).all() for arr in (*cp.values(), *tp.values())):
        raise ValueError("Nonfinite frozen estimator prediction")
    residuals = {ep: [] for ep in ENDPOINTS}
    for j, i in enumerate(calibration):
        row = dataset.rows[i]
        proxies = endpoint_proxies(10. ** float(cp["total_clearance"][j]), 10. ** float(cp["vss"][j]))
        for ep, (label, unit, _) in ENDPOINTS.items():
            # This is the only observation-label access in this module.
            observed = observation_log(row["labels_log10"][label], ep)
            if observed is None:
                continue
            residuals[ep].append({"index": i, "calibration_id": f"pksmart_row_{i:06d}",
                "identity_smiles": row["identity_smiles"], "scaffold_group": row["scaffold_group"],
                "unit": unit, "proxy_value": proxies[ep], "observed_log": observed,
                "residual_log": observed - math.log(proxies[ep])})
    targets = {}
    train_bits = x[[positions[i] for i in train], :1024]
    calibration_positions = {i: j for j, i in enumerate(calibration)}
    for j, cid in enumerate(target_ids):
        frozen = fitted["predictions"][cid]["methods"]["selected"]["predict_nominal"]
        for parameter in ("total_clearance", "vss"):
            if not math.isclose(float(tp[parameter][j]), math.log10(frozen[parameter]), rel_tol=0, abs_tol=1e-12):
                raise ValueError("Frozen target parameter predictions no longer reproduce")
        proxies = endpoint_proxies(frozen["total_clearance"], frozen["vss"])
        cal_similarities = tanimoto(tx[j, :1024], cx[:, :1024])
        endpoint_refs = {}
        for ep in ENDPOINTS:
            similarities = [float(cal_similarities[calibration_positions[row["index"]]]) for row in residuals[ep]]
            local = empirical_distribution(proxies[ep], residuals[ep], similarities, local=True)
            global_ = empirical_distribution(proxies[ep], residuals[ep], similarities, local=False)
            endpoint_refs[ep] = {"status": local["status"], "unit": ENDPOINTS[ep][1],
                "proxy_value": proxies[ep], "proxy_log": math.log(proxies[ep]), "local": local, "global": global_}
        train_similarities = tanimoto(tx[j, :1024], train_bits)
        targets[cid] = {"identity_smiles": cohort[cid]["identity_smiles"],
            "scaffold_group": cohort[cid]["scaffold_group"],
            "max_training_similarity": float(max(train_similarities)),
            "max_calibration_similarity": float(max(cal_similarities)),
            "endpoints": endpoint_refs, "target_observations_used": False}
    artifact = {**specification, "input_fingerprint": fingerprint,
        "calibration_residuals": residuals, "targets": targets,
        "split_audit": {"train_indices": train, "calibration_indices": calibration,
            "train_count": len(train), "calibration_count": len(calibration), "target_count": len(cohort),
            "training_calibration_structures_disjoint": True, "training_calibration_scaffolds_disjoint": True,
            "training_target_structures_disjoint": True, "training_target_scaffolds_disjoint": True,
            "calibration_target_structures_disjoint": True, "calibration_target_scaffolds_disjoint": True,
            "calibration_labels_used_for_model_selection": False, "target_observation_files_opened": False},
        "proxy_definitions": {"clearance": "selected CL[mL/min/kg] * 70 kg * 60 min/h / 1000 mL/L",
            "vss": "selected Vss[L/kg] * 70 kg", "mrt": "proxy Vss / proxy CL",
            "terminal_half_life": "ln(2) * proxy MRT; residual indexing proxy only, not a terminal half-life identity"},
        "limitations": ["Local empirical residuals are not calibrated clinical probabilities or confidence intervals.",
            "The four endpoint distributions are marginal; they do not define a joint physiological population.",
            "No clinical concentration curves, dose matching, or other seven endpoint reference distributions are available here.",
            "PKSmart aggregate labels are normalized to an assumed 70 kg adult, not an individual patient's observations.",
            "Locality uses fixed 20 structural neighbors, minimum 10; neighbor count and weights never use target outcomes.",
            "Terminal-half-life proxy assumes an exponential residence timescale only to index residuals; PBPK scoring uses the actual solver endpoint.",
            "Source table is loaded by the existing loader, but only the frozen calibration rows' labels enter residuals."]}
    prior._verify(pins, root)
    artifact["artifact_digest"] = prior._digest(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior._save(output, artifact)
    return artifact
