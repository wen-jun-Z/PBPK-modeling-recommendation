#!/usr/bin/env python3
"""Append full-list support for the five completed controls, without refitting.

The original control report and the original 15-method full-list report remain
unchanged.  This small entry point uses their exact-rank/tie metric definitions.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
from reviewer_full_list_evaluation import full_list_row, summarize, write_csv, digest, TASK_GROUP

OUT = ROOT / "experiments/model_recommendation/reviewer_revision_20261009/controls"
METHODS = ("structured_mlp", "text_mlp", "concat_mlp", "identity_condition_rule", "service_typed_rule")


def build():
    sources = {}

    def bind(path, expected=None):
        path = Path(path)
        path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Full-list controls must stay inside this project")
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError("Frozen control input changed: " + str(path))
        sources[str(path.relative_to(ROOT))] = actual
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    protocol = read(OUT / "protocol.json")
    manifest_ref = protocol["dataset_manifest"]
    manifest = read(manifest_ref["path"], manifest_ref["sha256"])
    if not manifest["completed"]:
        raise ValueError("Dataset incomplete")
    files = {Path(row["path"]).name: row for row in manifest["outputs"]}
    raw = read(files["dataset.json"]["path"], files["dataset.json"]["sha256"])
    with np.load(bind(files["arrays.npz"]["path"], files["arrays.npz"]["sha256"]), allow_pickle=False) as arrays:
        grades, apps = arrays["grades"].copy(), arrays["applicability"].copy()
    query_ids = [row["query_id"] for row in raw["query_records"]]
    candidate_ids = [row["candidate_id"] for row in raw["candidate_records"]]
    if query_ids != protocol["query_ids"] or candidate_ids != protocol["candidate_ids"]:
        raise ValueError("Frozen protocol axes changed")
    core = raw["core_indices"]
    queries = [raw["query_records"][i] for i in core]
    if (len(query_ids), len(core), len(candidate_ids)) != (813, 145, 280):
        raise ValueError("Unexpected frozen evaluation population")
    if grades.shape != (813, 280) or apps.shape != grades.shape:
        raise ValueError("Label axes changed")
    completion = read(OUT / "report/completion.json")
    report_ref = completion["summary"]
    original = read(report_ref["path"], report_ref["sha256"])
    quality = {row["method_id"]: row for row in original["quality_rows"]}
    aggregates, per_query, checks = [], [], []
    for method in METHODS:
        receipt = read(OUT / method / "predictions/seed17.json")
        if receipt["state"] != "complete" or receipt["method"] != method or receipt["protocol"]["sha256"] != sources[str((OUT / "protocol.json").relative_to(ROOT))]:
            raise ValueError("Incomplete or mismatched control prediction receipt")
        ref = receipt["predictions"]
        with np.load(bind(ref["path"], ref["sha256"]), allow_pickle=False) as arrays:
            if not np.array_equal(arrays["indices"], np.arange(813)) or arrays["scores"].shape != (3, 813, 280):
                raise ValueError("Incomplete full-candidate control output")
            scores = arrays["scores"][0, core].copy()
        rows = [{"method_id": method, "query_id": q["query_id"], "task_type": q["task_type"],
                 "purpose": TASK_GROUP[q["task_type"]], "compound": q["compound_id"],
                 **full_list_row(scores[i], grades[core[i]], apps[core[i]])} for i, q in enumerate(queries)]
        per_query.extend(rows)
        scopes = {}
        for scope in ("all", "direct_prediction", "mechanism_reference", "parameterization_template"):
            selected = rows if scope == "all" else [row for row in rows if row["purpose"] == scope]
            result = {"method_id": method, "method": quality[method]["method"], "scope": scope, **summarize(selected)}
            aggregates.append(result)
            scopes[scope] = result
        if [scopes[s]["requests"] for s in ("all", "direct_prediction", "mechanism_reference", "parameterization_template")] != [145, 85, 30, 30]:
            raise ValueError("Purpose partition differs from original report")
        if [scopes[s]["ranking_eligible_requests"] for s in ("all", "direct_prediction", "mechanism_reference", "parameterization_template")] != [40, 6, 7, 27]:
            raise ValueError("Known-positive ranking denominators differ")
        for metric in ("ndcg@10", "recall@10", "mrr"):
            if not np.isclose(scopes["all"][metric], quality[method][metric], rtol=0, atol=1e-12):
                raise ValueError("Previously published control metric not reproduced: " + metric)
        checks.append({"method_id": method, "all_280_candidates_retained": True,
                       "all_145_requests_retained": True, "purpose_partition_verified": True,
                       "original_judged_ranking_reproduced": True,
                       "four_top1_grade_states_partition_one": True})
    for source in (Path(__file__), ROOT / "scripts/reviewer_full_list_evaluation.py",
                   ROOT / "scripts/evaluate_frontier_automatic.py",
                   ROOT / "src/pbpk_kb/recommendation/pbpk_match_v2/evaluation.py"):
        bind(source)
    for path, expected in sources.items():
        if digest(ROOT / path) != expected:
            raise ValueError("Source changed during evaluation")
    result = {"schema": "pbpk-reviewer-controls-full-list-v1", "state": "complete",
        "methods": len(METHODS), "requests": 145, "candidates": 280,
        "expert_validation": False, "new_inference_or_training": False,
        "old_metrics_or_tables_modified": False,
        "metric_definitions": {
            "full_list": "All 145 requests retain original full-catalog ranks; g>=2 supported, g=1 limited, g=0 unsupported, g=-1 unknown.",
            "ties": "Uniform exact-score rank probabilities; no candidate-ID tie-breaking.",
            "topk": "Expected fraction over exactly ranks1..k; not boundary-inclusive coverage or full-catalog recall.",
            "judged_only": "Unknown candidates removed only here; same 40 known-positive requests, not real-list accuracy.",
            "applicability": "Top1 masses use saved Y/N/U labels; invalid labels retain unknown mass. They are distinct from grade states.",
            "scope": "Unvalidated automatic labels and internal grouped predictions; no expert or external correctness claim."},
        "rows": aggregates, "per_query": per_query, "checks": checks, "sources": sources,
        "producer": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(__file__)}}
    destination = OUT / "report"
    (destination / "full_list.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_csv(destination / "full_list.csv", aggregates)
    write_csv(destination / "full_list_per_query.csv", [{k: v for k, v in row.items() if k != "judged_only"} for row in per_query])
    print(json.dumps({"state": "complete", "methods": len(METHODS), "aggregates": len(aggregates), "per_query": len(per_query),
                      "all": [row for row in aggregates if row["scope"] == "all"]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    build()
