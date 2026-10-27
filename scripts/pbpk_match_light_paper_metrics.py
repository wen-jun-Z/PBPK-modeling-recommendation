"""Align completed PBPK-Match outputs with the existing paper's metric scopes.

This is reporting only: no training, recalibration, inference or PBPK refitting.
The old 102-request Y/N denominator is reused, while the distinct grouped
training/calibration protocol is retained explicitly in the exported metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_pbpk_match_light as light

OUTPUT = ROOT / "results/data/pbpk_match_light_paper_metrics.json"
CORE = ROOT / "experiments/model_recommendation/frontier_core_145_20260921"
METHODS = (light.DOCUMENT, light.OWN)


def _read(path):
    return json.loads(Path(path).read_text())


def _reference(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Paper metric inputs must remain in this project")
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def known_topk_coverage(scores, grades, k=10):
    """Mean known-grade fraction in full-catalog Top-k, including boundary ties."""
    scores, grades = np.asarray(scores), np.asarray(grades)
    if scores.ndim != 2 or grades.shape != scores.shape or not scores.size:
        raise ValueError("Require aligned nonempty query/candidate matrices")
    if not np.isfinite(scores).all() or not np.isin(grades, [-1, 0, 1, 2, 3]).all():
        raise ValueError("Invalid scores or grades")
    boundary = np.sort(scores, axis=1)[:, -min(k, scores.shape[1])]
    included = scores >= boundary[:, None]
    fractions = ((grades >= 0) & included).sum(1) / included.sum(1)
    return {"mean": float(fractions.mean()), "per_query": fractions.tolist(),
            "included_candidates": included.sum(1).tolist(), "k": k}


def binary_applicability(probability_y, labels, threshold=.5):
    """Evaluate unconditional P(Y); U and invalid labels never become negatives."""
    probability_y, labels = np.asarray(probability_y, float), np.asarray(labels)
    if probability_y.shape != labels.shape or probability_y.ndim != 2:
        raise ValueError("Require aligned query/candidate probability and label matrices")
    if (not np.isfinite(probability_y).all() or (probability_y < 0).any()
            or (probability_y > 1).any() or not np.isin(labels, [-1, 0, 1, 2]).all()):
        raise ValueError("Invalid probabilities or Y/N/U labels")
    known = (labels == 0) | (labels == 1)
    y, decision = labels[known] == 0, probability_y[known] >= threshold
    confusion = {"tp": int((y & decision).sum()), "fp": int((~y & decision).sum()),
                 "tn": int((~y & ~decision).sum()), "fn": int((y & ~decision).sum())}
    tp, fp, tn, fn = (confusion[key] for key in ("tp", "fp", "tn", "fn"))
    f1_y = 2 * tp / (2 * tp + fp + fn) if tp + fn else None
    f1_n = 2 * tn / (2 * tn + fp + fn) if tn + fp else None
    return {"threshold": threshold, "comparison": "P(Y) >= threshold", "class_order": ["Y", "N", "U"],
            "probability": "Unconditional saved outer-fold P(Y); no Y/(Y+N) renormalization",
            "confusion": confusion, "known_pairs": int(known.sum()), "positive_pairs": int(y.sum()),
            "unknown_pairs_excluded": int((labels == 2).sum()),
            "invalid_pairs_excluded": int((labels == -1).sum()),
            "positive_f1": f1_y, "positive_precision": tp / (tp + fp) if tp + fp else None,
            "positive_recall": tp / (tp + fn) if tp + fn else None,
            "macro_f1": (f1_y + f1_n) / 2 if f1_y is not None and f1_n is not None else None,
            "brier": float(np.mean((probability_y[known] - y) ** 2)) if known.any() else None}


def held_out_membership(data, indices, selections):
    """Assert that each displayed request is excluded from fitting and selection."""
    group = data["groups"]
    selection = {row["fold"]: row for row in selections}
    rows = []
    for index in indices:
        owners = [f for f, fold in enumerate(data["folds"]) if index in fold["outer_test"]]
        if len(owners) != 1:
            raise ValueError("Paper test request must have exactly one outer holdout")
        f = owners[0]
        fold = data["folds"][f]
        train_groups = {group[i] for i in fold["outer_train"]}
        selected = selection[f]
        selection_groups = {group[i] for i in selected["selection_indices"]}
        calibration_groups = set(selected["calibration"]["fit_groups"])
        if group[index] in train_groups | selection_groups | calibration_groups:
            raise ValueError("A displayed test group entered fitting, selection or calibration")
        if not selection_groups <= train_groups or not calibration_groups <= train_groups:
            raise ValueError("Inner selection or calibration escapes outer training groups")
        rows.append({"query_id": data["query_records"][index]["query_id"], "query_index": index,
                     "outer_fold": f, "group": group[index], "fitting_group_overlap": False,
                     "selection_group_overlap": False, "calibration_group_overlap": False})
    return rows


def scalar_rows(reports, error_records):
    """Keep every genuinely measured endpoint and every exact rank, including NA."""
    counts = Counter(row["endpoint"] for row in error_records)
    result = []
    for method in METHODS:
        lookup = {(row["endpoint"], row["rank"]): row
                  for row in reports[method]["execution"]["position_summary"]}
        for endpoint in sorted(counts):
            for rank in (1, 2, 3):
                if (endpoint, rank) not in lookup:
                    raise ValueError("Missing exact-rank summary for a measured endpoint")
                row = lookup[(endpoint, rank)]
                if row["coverage"] == 0 and row["gmfe"] is not None:
                    raise ValueError("Zero-coverage endpoint must not carry a numeric GMFE")
                result.append({**row, "source_scalar_comparisons": counts[endpoint]})
    return result, dict(sorted(counts.items()))


def build():
    from pbpk_kb.recommendation.pbpk_match_v2.data import load_dataset

    data = load_dataset()
    protocol, _, arrays, input_refs, _, selections = light.load_inputs(data)
    completed_path = light.OUT / "report/completion.json"
    completed = _read(completed_path)
    summary_path = light.OUT / "report/summary.json"
    summary_ref = _reference(summary_path)
    if not completed.get("completed") or summary_ref not in completed["outputs"]:
        raise ValueError("Unverified or incomplete source report")
    report = _read(summary_path)
    if report["protocol_sha256"] != _reference(light.OUT / "protocol.json")["sha256"]:
        raise ValueError("Report and held-out prediction protocol differ")
    selection_path = CORE / "selection_manifest.json"
    selection = _read(selection_path)
    core_indices = list(data["core_indices"])
    core_ids = [data["query_records"][i]["query_id"] for i in core_indices]
    if set(selection["selected_query_ids"]) != set(core_ids) or len(core_ids) != 145:
        raise ValueError("Paper core requests differ from the frozen 145-request selection")
    test_indices = [i for i in core_indices
                    if selection["compound_split"][data["query_records"][i]["compound_id"]] == "test"]
    test_ids = [data["query_records"][i]["query_id"] for i in test_indices]
    if len(test_ids) != 102 or len(set(test_ids)) != 102:
        raise ValueError("Paper Y/N scope must be exactly the original 102 test requests")
    audit = held_out_membership(data, test_indices, selections)
    app = binary_applicability(arrays["probabilities"][0, test_indices, :, 0],
                               np.asarray(data["applicability"])[test_indices])
    if app["known_pairs"] != 16410 or app["positive_pairs"] != 74:
        raise ValueError("Paper applicability denominator differs from 16,410 pairs / 74 Y")
    legacy_path = ROOT / "results/data/model_recommendation_quality.json"
    legacy = _read(legacy_path)
    for row in legacy["rows"]:
        if row.get("group") == "supervised_grouped_holdout":
            continue
        if row["test_known_pairs"] != 16410 or row["test_positive_pairs"] != 74:
            raise ValueError("Existing paper table has a different applicability denominator")

    rows, details = [], {}
    for method, score_key in ((light.DOCUMENT, "base_scores"), (light.OWN, "scores")):
        q = report["reports"][method]["quality"]
        coverage = known_topk_coverage(arrays[score_key][0, core_indices],
                                      np.asarray(data["grades"])[core_indices])
        preference = q["same_drug_different_family_preference"]
        row = {"method_id": method, "method": light.DISPLAY[method], "group": "supervised_grouped_holdout",
               "ranking_eligible_queries": q["judged_only"]["eligible_queries"],
               **{key: q["judged_only"]["metrics"][key] for key in ("ndcg@10", "recall@10", "mrr")},
               "same_drug_family_query_macro_accuracy": preference["query_macro_accuracy"],
               "same_drug_family_queries": preference["eligible_queries"], "same_drug_family_pairs": preference["pairs"],
               "top10_judged_fraction_all_queries": coverage["mean"],
               "test_known_pairs": app["known_pairs"], "test_positive_pairs": app["positive_pairs"],
               "test_positive_f1": app["positive_f1"], "test_positive_precision": app["positive_precision"],
               "test_positive_recall": app["positive_recall"], "test_applicability_macro_f1": app["macro_f1"],
               "test_brier": app["brier"], "test_ece": None}
        rows.append(row)
        details[method] = {"binary_applicability_102": app,
                           "three_class_applicability_145": q["applicability"],
                           "top10_known_coverage": {"query_ids": core_ids, **coverage},
                           "full_catalog_coverage": q["coverage"]}
    endpoints, counts = scalar_rows(report["reports"], data["error_records"])
    if sum(counts.values()) != 171 or any("tmax" in key.casefold() for key in counts):
        raise ValueError("Unexpected measured endpoint registry")
    sources = {entry["path"]: entry["sha256"] for entry in input_refs}
    for path in (Path(__file__), completed_path, summary_path, selection_path, legacy_path):
        item = _reference(path)
        sources[item["path"]] = item["sha256"]
    return {"schema": "pbpk-match-light-paper-compatible-metrics-v1", "state": "complete",
            "quality_rows": rows, "quality_details": details, "scalar_endpoint_rows": endpoints,
            "scalar_endpoint_counts": counts, "test_query_ids": test_ids,
            "held_out_membership": audit,
            "scope": {"core_requests": 145, "ranking_eligible_requests": 40, "paper_binary_test_requests": 102,
                      "paper_binary_known_pairs": 16410, "paper_binary_positive_pairs": 74,
                      "scalar_comparisons": 171, "scalar_endpoint_types": len(counts), "new_model_fits": 0,
                      "new_probability_calibration": False, "new_pbpk_execution": False},
            "protocol_notes": {
                "table_block": "Supervised grouped holdout; frozen external methods retain their original block.",
                "ranking": "Identical known-grade subsets and tie expectations; same 40 eligible requests.",
                "known_coverage": "Mean known-grade fraction in each full-catalog Top10 including boundary ties, all 145 requests.",
                "binary_f1": "Original 102 test requests and 16,410 known Y/N pairs (74 Y), unconditional P(Y)>=0.5; U/invalid excluded.",
                "protocol_difference": "Own saved probabilities come from five outer grouped folds with inner-OOF temperature calibration. External methods used 43 calibration and 102 test requests. No recalibration was performed to align the table denominator; training and calibration protocols remain different.",
                "three_class_f1": "The 145-request Y/N/U argmax F1 remains separate; it is not the paper-compatible binary Y-F1.",
                "endpoints": "Only seven endpoint definitions present in the 171 completed scalar comparisons; no Tmax reference is available. Exact ranks1/2/3, uniform ties, no replacement; zero coverage remains NA.",
                "curves": "Scalar GMFE is not concentration-curve GMFE; available curves are reported separately.",
                "training_scope": protocol["validation_scope"]},
            "source_files": sources,
            "reproduce": "runtime/app-env/bin/python -B scripts/pbpk_match_light_paper_metrics.py --check"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build()
    if args.check:
        if not OUTPUT.exists() or _read(OUTPUT) != value:
            raise ValueError("Published paper-compatible metrics differ from their frozen inputs")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(OUTPUT)
    print(json.dumps({"state": value["state"], "scope": value["scope"], "quality_rows": value["quality_rows"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
