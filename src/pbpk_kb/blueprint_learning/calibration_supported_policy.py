"""Select a residual-reference family within calibration scaffolds only.

Global residual references are the conservative default. Local references need
more than one standard error of scaffold-equal CRPS improvement, evaluated by
leaving each entire calibration scaffold out. No target outcomes are loaded.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import adaptive_parameter_prior as prior
from . import endpoint_reference_prior as reference

VERSION = "calibration-supported-reference-policy-1"
CODE = "src/pbpk_kb/blueprint_learning/calibration_supported_policy.py"
REFERENCE_PATH = "experiments/blueprint_design/decision_value_20261001/reference_prior.json"
BASELINE_AUDIT = "experiments/blueprint_design/purpose_hundred_20260930/priors/audit.json"
NUMERIC_TOLERANCE = 1e-12


def empirical_crps(samples, observation):
    """Proper CRPS of the empirical CDF, in natural-log residual units."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not math.isfinite(observation):
        raise ValueError("CRPS requires a finite observation and nonempty finite one-dimensional samples")
    return float(np.abs(values - observation).mean() - .5 * np.abs(values[:, None] - values[None, :]).mean())


def select_endpoint(rows, features_by_index):
    """LOSO predictions use no member of the held scaffold's residual labels."""
    if len({r["index"] for r in rows}) != len(rows) or len({r["identity_smiles"] for r in rows}) != len(rows):
        raise ValueError("Calibration references contain duplicate rows or structures")
    if any(not math.isfinite(r["residual_log"]) for r in rows):
        raise ValueError("Nonfinite calibration residual")
    groups = defaultdict(list)
    for row in rows:
        groups[row["scaffold_group"]].append(row)
    folds, differences, global_scores, local_scores = [], [], [], []
    for scaffold, held in sorted(groups.items()):
        pool = [r for r in rows if r["scaffold_group"] != scaffold]
        pool = sorted(pool, key=lambda r: hashlib.sha256(r["identity_smiles"].encode()).hexdigest())
        fold = {"held_scaffold": scaffold, "held_indices": sorted(r["index"] for r in held),
                "reference_indices": [r["index"] for r in pool], "reference_n": len(pool),
                "self_and_same_scaffold_excluded": True, "scores": []}
        if len(pool) < reference.MIN_CALIBRATION:
            fold.update(status="unassessed", reason="fewer_than_10_independent_reference_compounds")
            folds.append(fold)
            continue
        pool_bits = np.array([features_by_index[r["index"]][:1024] for r in pool])
        for target in sorted(held, key=lambda r: r["index"]):
            similarities = reference.tanimoto(features_by_index[target["index"]][:1024], pool_bits)
            # proxy=1 leaves the residual values unchanged. This reads only pool labels.
            local = reference.empirical_distribution(1.0, pool, similarities, local=True)
            global_ = reference.empirical_distribution(1.0, pool, similarities, local=False)
            local_score = empirical_crps(local["reference_values_log"], target["residual_log"])
            global_score = empirical_crps(global_["reference_values_log"], target["residual_log"])
            fold["scores"].append({"held_index": target["index"],
                "held_residual_log": target["residual_log"],
                "local_reference_indices": local["calibration_indices"],
                "local_reference_n": local["n"], "global_reference_n": global_["n"],
                "local_crps": local_score, "global_crps": global_score,
                "paired_improvement_global_minus_local": global_score - local_score})
        global_mean = float(np.mean([r["global_crps"] for r in fold["scores"]]))
        local_mean = float(np.mean([r["local_crps"] for r in fold["scores"]]))
        fold.update(status="scored", global_crps_mean=global_mean, local_crps_mean=local_mean,
                    paired_improvement_global_minus_local=global_mean - local_mean)
        differences.append(global_mean - local_mean)
        global_scores.append(global_mean)
        local_scores.append(local_mean)
        folds.append(fold)
    n = len(differences)
    difference = float(np.mean(differences)) if n else None
    se = float(np.std(differences, ddof=1) / math.sqrt(n)) if n > 1 else None
    supported = n == len(groups) and n >= 2
    local_supported = supported and difference > se + NUMERIC_TOLERANCE
    reason = ("local_CRPS_improvement_exceeds_one_scaffold_standard_error" if local_supported else
              "insufficient_leave_scaffold_out_support" if not supported else
              "local_CRPS_improvement_does_not_exceed_one_scaffold_standard_error")
    return {"family": "local" if local_supported else "global", "selection_reason": reason,
            "calibration_n": len(rows), "calibration_scaffold_n": len(groups), "paired_scaffold_n": n,
            "global_scaffold_equal_crps": float(np.mean(global_scores)) if n else None,
            "local_scaffold_equal_crps": float(np.mean(local_scores)) if n else None,
            "paired_scaffold_improvement_mean": difference, "paired_scaffold_improvement_se": se,
            "all_scaffolds_scored": n == len(groups), "folds": folds}


def _validate_rows(residuals, split, baseline):
    """Check source membership before any reference-family selection."""
    calibration = set(baseline["calibration_indices"])
    train = set(baseline["train_indices"])
    if calibration & train or calibration != set(split["calibration_indices"]) or train != set(split["train_indices"]):
        raise ValueError("Frozen calibration/training membership differs")
    indexed = {r["index"]: r for r in baseline["rows"]}
    training_scaffolds = {indexed[i]["scaffold_group"] for i in train}
    target_scaffolds = {r["scaffold_group"] for r in baseline["external_exclusions"].values()}
    for rows in residuals.values():
        for row in rows:
            i = row["index"]
            if i not in calibration or i in train:
                raise ValueError("Residual label lies outside the frozen calibration rows")
            expected = indexed[i]
            if expected["status"] != "calibration" or any(row[k] != expected[k] for k in ("identity_smiles", "scaffold_group")):
                raise ValueError("Residual calibration identity/scaffold changed")
            if row["scaffold_group"] in training_scaffolds | target_scaffolds:
                raise ValueError("Calibration scaffold overlaps training or target scaffolds")


def fit_calibration_policy(project_root, output_path, *, reference_path=REFERENCE_PATH, baseline_audit=BASELINE_AUDIT):
    """Write only one small policy artifact, preserving the parameter prior."""
    root = Path(project_root).resolve(strict=True)
    source_path = prior._inside(reference_path, root)
    baseline_path = prior._inside(baseline_audit, root)
    output = prior._inside(output_path, root, exists=False)
    source = json.loads(source_path.read_text())
    source_digest = source.pop("artifact_digest", None)
    if source_digest != prior._digest(source):
        raise ValueError("Frozen reference prior artifact changed")
    prior._verify(source["source_pins"], root)
    pins = dict(source["source_pins"])
    for path in (source_path, baseline_path, prior._inside(CODE, root)):
        pins[str(path.relative_to(root))] = prior._sha(path)
    specification = {"version": VERSION, "source_pins": pins,
        "neighbors": reference.NEIGHBORS, "minimum_reference_compounds": reference.MIN_CALIBRATION,
        "folds": "leave_one_entire_calibration_scaffold_out", "score": "empirical_CRPS_natural_log_residual",
        "scoring_weighting": "equal_scaffold_then_equal_compound_within_scaffold",
        "selection": "local only if mean(global CRPS minus local CRPS) > sample SD / sqrt(scaffolds) + 1e-12",
        "fallback": "global on insufficient support or insufficient improvement",
        "numeric_tolerance": NUMERIC_TOLERANCE, "clinical_probability_calibration": False}
    fingerprint = prior._digest(specification)
    if output.exists():
        existing = json.loads(output.read_text())
        digest = existing.pop("artifact_digest", None)
        if digest != prior._digest(existing) or existing.get("input_fingerprint") != fingerprint:
            raise ValueError("Frozen calibration policy changed or uses different sources/code")
        return {**existing, "artifact_digest": digest}
    baseline = json.loads(baseline_path.read_text())
    residuals = source["calibration_residuals"]
    if set(residuals) != set(reference.ENDPOINTS):
        raise ValueError("Only the four independently labeled endpoints are supported")
    _validate_rows(residuals, source["split_audit"], baseline)
    identities = {row["index"]: row["identity_smiles"] for rows in residuals.values() for row in rows}
    indices = sorted(identities)
    features = prior.base._features([identities[i] for i in indices])
    by_index = dict(zip(indices, features))
    endpoints = {ep: select_endpoint(rows, by_index) for ep, rows in residuals.items()}
    artifact = {**specification, "input_fingerprint": fingerprint, "endpoints": endpoints,
        "calibration_count": len(source["split_audit"]["calibration_indices"]),
        "calibration_reference_prior_digest": source_digest,
        "target_outcome_files_opened": False, "target_metrics_used_for_selection": False,
        "parameter_estimators_refitted": False,
        "limitations": ["Selection estimates support for a residual-reference family, not PBPK predictive accuracy.",
            "Calibration scaffolds provide development model selection; their scores are not an independent final test.",
            "Global is the default, not a claim of guaranteed clinical coverage.",
            "A same-development-cohort rerun after prior results has been observed cannot become a new blind test."]}
    prior._verify(pins, root)
    artifact["artifact_digest"] = prior._digest(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior._save(output, artifact)
    return artifact
