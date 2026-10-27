"""Export the completed 145-request document-quality evaluation for the paper.

No inference, training, label replacement or ranking selection is performed.
Use --verify to recompute the frozen evaluation from scores without overwriting it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from model_recommendation_paper import METHODS, _fmt, _table, _write_table

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("experiments/model_recommendation/frontier_core_145_20260921")
ASSESSMENT = RUN / "assessment_repaired_v2"
METRICS = ("ndcg@10", "recall@10", "mrr")


def close_tree(actual, expected, path="report"):
    """Fail on missing fields, denominators, null changes or numeric disagreement."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Different keys at {path}")
        for key in expected:
            close_tree(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Different list at {path}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            close_tree(a, b, f"{path}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not math.isfinite(actual) or not math.isfinite(expected)
                or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError(f"Numeric mismatch at {path}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Value mismatch at {path}")


def positive_metrics(confusion):
    tp, fp, tn, fn = (confusion[k] for k in ("tp", "fp", "tn", "fn"))
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (tp, fp, tn, fn)):
        raise ValueError("Confusion counts must be nonnegative integers")
    return {
        "positive_f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
        "positive_recall": tp / (tp + fn) if tp + fn else None,
        "positive_precision": tp / (tp + fp) if tp + fp else None,
        "constant_zero_probability_brier": (tp + fn) / (tp + fp + tn + fn) if tp + fp + tn + fn else None,
    }


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_quality_results(root=ROOT, out=None, *, verify=False):
    root = Path(root).resolve()
    out = Path(out or root / "results").resolve()
    if not out.is_relative_to(root):
        raise ValueError("Output must remain inside the project")
    sources = {}

    def read(relative, *, binary=False):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Input must remain inside the project")
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return raw if binary else json.loads(raw)

    report = read(ASSESSMENT / "automatic_evaluation.json")
    agreement = read(ASSESSMENT / "judge_agreement.json")
    acceptance = report["completion_acceptance"]
    if report["state"] != "complete" or acceptance["state"] != "passed" or not all(acceptance["checks"].values()):
        raise ValueError("A complete accepted evaluation is required")
    if set(report["methods"]) != {method for method, _, _ in METHODS}:
        raise ValueError("Paper method list differs from the frozen experiment")
    if report["method_counts"]["pending"] or agreement["sampled_pairs"] != agreement["expected_sample_pairs"]:
        raise ValueError("Missing method matrices or secondary judgments")

    primary_raw = read(ASSESSMENT / "judgments.jsonl", binary=True)
    secondary_raw = read(ASSESSMENT / "secondary_judgments.jsonl", binary=True)
    if hashlib.sha256(primary_raw).hexdigest() != report["assessment"]["committed_judgments_sha256"]:
        raise ValueError("Judgments changed after evaluation")
    primary = [json.loads(line) for line in primary_raw.splitlines() if line]
    secondary = [json.loads(line) for line in secondary_raw.splitlines() if line]
    queries = [json.loads(line) for line in read(RUN / "inputs/queries.jsonl", binary=True).splitlines() if line]
    candidates = [json.loads(line) for line in read(RUN / "inputs/candidates.jsonl", binary=True).splitlines() if line]
    expected_pairs = {(q["query_id"], c["candidate_id"]) for q in queries for c in candidates}
    pair = lambda row: (row["query_id"], row["candidate_id"])
    if len(primary) != len(expected_pairs) or {pair(row) for row in primary} != expected_pairs:
        raise ValueError("Primary judgments must cover each frozen pair exactly once")
    from run_frontier_assessment import secondary_selected
    selected = {key for key in expected_pairs if secondary_selected(*key)}
    if len(secondary) != len(selected) or {pair(row) for row in secondary} != selected:
        raise ValueError("Secondary judgments must match the frozen identity-hash sample")
    if len(secondary) != agreement["sampled_pairs"]:
        raise ValueError("Agreement sample denominator differs from the judgments")

    for relative in [RUN / "protocol.json", RUN / "selection_manifest.json",
                     ASSESSMENT / "judge_protocol.json", ASSESSMENT / "evaluation_protocol.json",
                     ASSESSMENT / "evaluation_protocol_additions.json", ASSESSMENT / "evaluation_protocol_bounds.json",
                     ASSESSMENT / "evaluation_protocol_statistics.json"]:
        read(relative)
    for filename in ["build_model_quality_results.py", "model_recommendation_paper.py", "evaluate_frontier_automatic.py", "run_frontier_assessment.py"]:
        read(Path("scripts") / filename, binary=True)

    recomputation = {"requested": verify, "all_fields_match": None}
    if verify:
        from evaluate_frontier_automatic import evaluate_run
        reproduced = evaluate_run(root / RUN, root / ASSESSMENT, write_outputs=False, require_complete=True)
        close_tree({k: v for k, v in reproduced.items() if k != "created_at"},
                   {k: v for k, v in report.items() if k != "created_at"})
        recomputation["all_fields_match"] = True

    rows, details = [], {}
    for method, name, group in METHODS:
        item = report["methods"][method]
        progress = item["progress"]
        raw = read(RUN / progress["effective_score_file"], binary=True)
        if hashlib.sha256(raw).hexdigest() != progress["effective_scores_sha256"]:
            raise ValueError(f"Score file changed: {method}")
        original = item["variants"]["original"]
        overall = original["overall"]
        ranking = overall["judged_only"]
        app = original["calibrated_applicability"]
        positive = positive_metrics(app["confusion"])
        if sum(app["confusion"].values()) != app["test_pairs"] or app["confusion"]["tp"] + app["confusion"]["fn"] != app["positive_test_pairs"]:
            raise ValueError("Applicability denominators do not match the confusion matrix")
        decision = item["original_request_decisions"]
        preference = overall["different_registered_family_preference"]
        uncertainty = item["statistical_uncertainty"]
        row = {"method_id": method, "method": name, "group": group,
               "ranking_eligible_queries": ranking["eligible_queries"],
               **{key: ranking["metrics"][key] for key in METRICS},
               "same_drug_family_query_macro_accuracy": preference["query_macro_accuracy"],
               "same_drug_family_queries": preference["eligible_queries"], "same_drug_family_pairs": preference["pairs"],
               "top10_judged_fraction_all_queries": overall["topk_label_coverage"]["inclusive_judged@10"],
               "test_known_pairs": app["test_pairs"], "test_positive_pairs": app["positive_test_pairs"],
               "test_applicability_macro_f1": app["macro_f1"],
               **{"test_" + key: value for key, value in positive.items()},
               "test_brier": app["brier"], "test_ece": app["ece"],
               "request_recommendation_coverage": decision["recommendation_coverage"],
               "request_known_top1_error": decision["false_recommendation_rate_among_known_top1"],
               "request_wrong_rejection": decision["wrong_rejection_rate_when_applicable_exists"]}
        for metric in METRICS:
            for label, source in [("ci", "judged_only"), ("delta_vs_bm25_ci", "paired_judged_only_delta_vs_bm25")]:
                ci = uncertainty[source][metric]["ci95"]
                row[f"{metric}_{label}_low"], row[f"{metric}_{label}_high"] = ci
        rows.append(row)
        details[method] = {
            "by_task": {task: {key: value[key] for key in ("query_count", "judged_only", "same_drug_full_pool", "different_drug_full_pool")}
                        for task, value in original["by_group"]["task_type"].items()},
            "by_split": original["by_split"], "full_catalog_bounds": overall["full_catalog_bounds"],
            "same_family_versions": overall["same_family_version_preference"],
            "calibrator": item["calibrator"], "calibrated_applicability": app,
            "request_decisions": decision, "statistical_uncertainty": uncertainty,
            "perturbations": {variant: item["variants"][variant]["perturbation_subsets"]
                              for variant in ("reordered", "missing_evidence")},
        }

    primary_status = dict(Counter(row["status"] for row in primary))
    close_tree(primary_status, report["assessment"]["status_counts"])
    first_original = report["methods"][METHODS[0][0]]["variants"]["original"]
    eligible = [row for row in first_original["per_query"] if row["judged_only"] is not None]
    counts = Counter(str(row["grade"]) if row["grade"] is not None else "unknown" for row in primary)
    grade_counts = {key: counts[key] for key in ("0", "1", "2", "3", "unknown")}
    publication = {
        "schema_version": "pbpk-paper-model-quality-1.0", "state": "complete",
        "generated_from_saved_outputs_only": True, "evaluation_created_at": report["created_at"],
        "scope": {"queries": len(queries), "compounds": len({q["compound_id"] for q in queries}),
                  "candidates": len(candidates), "methods": len(rows), "primary_pairs": len(primary),
                  "secondary_pairs": len(secondary), "ranking_eligible_queries": len(eligible),
                  "ranking_eligible_compounds": len({q["compound_id"] for q in eligible}),
                  "ranking_eligible_tasks": dict(Counter(q["task_type"] for q in eligible)),
                  "primary_status_counts": primary_status, "grade_counts": grade_counts,
                  "secondary_status_counts": dict(Counter(row["status"] for row in secondary)),
                  "secondary_known_grades": sum(row["grade"] is not None for row in secondary)},
        "rows": rows, "details": details, "judge_agreement": agreement,
        "completion_acceptance": acceptance, "source_files": sources,
        "recomputation": recomputation,
        "interpretation": {
            "ranking": "Query-macro on explicitly known candidate subsets with at least one grade>=2; not full-catalog quality.",
            "coverage": "All 145 original queries; per-query known fraction in raw full-catalog Top10 including boundary ties, then query mean.",
            "applicability": "Binary automatic Y/N on fixed held-out calibration-test compounds; U excluded, not negative. This split does not establish unseen-drug generalization.",
            "family_preference": "Same-drug, different registered families, query-macro; registered families do not prove scientific independence.",
            "bootstrap": "Supplementary post-scoring analysis with fixed labels, 2000 whole-compound draws; no judge-error or multiple-comparison correction.",
            "self_trained_model": "Earlier PBPK-Match development snapshot excluded; its numbers are not comparable to this completed label pool.",
            "scientific_gold": False, "pk_prediction_accuracy_measured": False,
        },
        "reproduce": "runtime/app-env/bin/python scripts/build_model_quality_results.py --verify",
    }
    for directory in [out / "data", out / "tables"]:
        directory.mkdir(parents=True, exist_ok=True)
    _dump(out / "data/model_recommendation_quality.json", publication)
    with (out / "tables/model_recommendation_quality.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = _table("lrrrrrr") + [
        r"Method & NDCG@10 & Recall@10 & MRR & Pair acc. & Y-F1 & Known (\%) \\", r"\midrule"]
    for group, heading in [("full_catalog", "Full-catalog scoring"), ("top20", "BM25 top-20 reranking")]:
        if group == "top20":
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{@{}l@{}}{\textit{" + heading + r"}} \\")
        for row in (r for r in rows if r["group"] == group):
            values = [row["method"], *[_fmt(row[key]) for key in METRICS],
                      _fmt(row["same_drug_family_query_macro_accuracy"]), _fmt(row["test_positive_f1"]),
                      _fmt(100 * row["top10_judged_fraction_all_queries"], 1)]
            lines.append(" & ".join(values) + r" \\")
    _write_table(out / "tables/model_recommendation_quality.tex", lines)
    return publication


def update_paper_index(root, publication):
    path = Path(root) / "results/data/paper_results.json"
    if not path.exists():
        return
    index = json.loads(path.read_text())
    index["model_recommendation_quality"] = {
        "file": "results/data/model_recommendation_quality.json", "state": publication["state"],
        "scope": publication["scope"], "independent_scientific_validation": False}
    index["model_recommendation_quality_evaluation_complete"] = publication["state"] == "complete"
    for relative, digest in publication["source_files"].items():
        index["source_files"][relative] = {"sha256": digest, "bytes": (Path(root) / relative).stat().st_size}
    _dump(path, index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    publication = build_quality_results(verify=args.verify)
    update_paper_index(ROOT, publication)
    print(json.dumps({"state": publication["state"], "scope": publication["scope"],
                      "recomputation": publication["recomputation"]}, ensure_ascii=False))
