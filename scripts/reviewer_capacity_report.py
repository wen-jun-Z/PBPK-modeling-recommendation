#!/usr/bin/env python3
"""Evaluate a frozen capacity-matched experiment without retraining any model.

The primary sampling unit is an input-connected drug/study group. Three training
seeds first contribute their mean metric for each query, never three independent
copies of that query. Original labels, full-list positions and clinical outcome
qualification are retained. This reporter does not modify the manuscript.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np

from pbpk_kb.recommendation.pbpk_match_v2 import reporting as r
from pbpk_kb.recommendation.pbpk_match_v2.experiment import read, write, sha
from pbpk_match_light_paper_metrics import binary_applicability
from reviewer_controls_run import prepare
from reviewer_full_list_evaluation import TASK_GROUP, full_list_row, summarize
import summarize_pbpk_match_light as original

OUT = ROOT / "experiments/model_recommendation/reviewer_revision_20261009/capacity_control"
METHODS = ("pbpk_interaction", "concat_capacity")
SEEDS = (17, 23, 41)
BASELINE = "concat_capacity"
DISPLAY = {"pbpk_interaction": "PBPK-Match interaction", "concat_capacity": "Capacity-matched concatenation"}
METRICS = ("ndcg@10", "recall@10", "mrr")


def json_string_keys(value):
    """Normalize JSON object keys without changing seed fields or metric values.

    Seed-indexed outputs also contain the secondary ``rank_ensemble`` entry.
    JSON represents every object key as text; converting before sorted export
    avoids comparing integer seeds with that string. A collision is an error,
    never permission to overwrite one run with another.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise ValueError("Report object keys must be strings or integer seeds")
            target = str(key)
            if target in result:
                raise ValueError("Report key collision after JSON normalization: " + target)
            result[target] = json_string_keys(item)
        return result
    if isinstance(value, (list, tuple)):
        return [json_string_keys(item) for item in value]
    return value


def reference(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT) or "sealed" in path.relative_to(ROOT).parts:
        raise ValueError("Reporting remains in project and outside sealed data")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def validate_arrays(arrays, data):
    """Reject partial candidates, unexpected corrections and uncalibrated axes."""
    q, c, v = len(data["query_records"]), len(data["candidate_records"]), len(data["variant_names"])
    if not np.array_equal(arrays.get("indices"), np.arange(q)):
        raise ValueError("Prediction request axis is incomplete or reordered")
    scores, base = np.asarray(arrays.get("scores")), np.asarray(arrays.get("base_scores"))
    probabilities = np.asarray(arrays.get("probabilities"))
    if scores.shape != (v, q, c) or not np.isfinite(scores).all():
        raise ValueError("Every registered candidate and variant requires a finite score")
    if not np.array_equal(scores, base):
        raise ValueError("Architectural comparison must disable error correction")
    if probabilities.shape != (v, q, c, 3):
        raise ValueError("Probability candidate or variant axes differ")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or
            (probabilities > 1).any() or not np.allclose(probabilities.sum(-1), 1., atol=1e-6, rtol=0)):
        raise ValueError("Invalid normalized Y/N/U probabilities")
    return scores.copy(), probabilities.copy()


def average_query_quality(reports, seeds=SEEDS):
    """Average seed metrics *within* query; unknown grades never become zeros."""
    if set(reports) != set(seeds):
        raise ValueError("All prespecified seeds are required before aggregation")
    maps = []
    for seed in seeds:
        rows = reports[seed]["quality"]["per_query"]
        mapping = {row["query_id"]: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("Duplicate query in one seed")
        maps.append(mapping)
    if any(set(mapping) != set(maps[0]) for mapping in maps[1:]):
        raise ValueError("Seed query populations differ")
    result = []
    for qid in sorted(maps[0]):
        judged = [mapping[qid]["judged_only"] for mapping in maps]
        present = [item is not None for item in judged]
        if any(present) != all(present):
            raise ValueError("Seed-dependent judged-only eligibility")
        metrics = None if not any(present) else {
            metric: float(np.mean([item[metric] for item in judged])) for metric in METRICS}
        if metrics is not None and not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("Nonfinite judged metric")
        result.append({"query_id": qid, "judged_only": metrics, "training_seeds": list(seeds)})
    return {"per_query": result}


def paired_seed_average_quality(reports, groups, seeds=SEEDS):
    """One prespecified primary NDCG test; recall and MRR remain descriptive."""
    if set(reports) != set(METHODS):
        raise ValueError("Registered architecture family changed")
    averaged = {method: average_query_quality(reports[method], seeds) for method in METHODS}
    query_ids = {row["query_id"] for row in averaged[METHODS[0]]["per_query"]}
    if query_ids != {row["query_id"] for row in averaged[METHODS[1]]["per_query"]}:
        raise ValueError("Paired methods have different request populations")
    if not query_ids <= set(groups):
        raise ValueError("Missing input-connected query group")
    comparisons = original.stable_paired_quality(averaged, groups, baseline=BASELINE, draws=2000)
    for row in comparisons:
        method_rows = {x["query_id"]: x["judged_only"] for x in averaged[row["method"]]["per_query"]}
        baseline_rows = {x["query_id"]: x["judged_only"] for x in averaged[BASELINE]["per_query"]}
        if any((method_rows[qid] is None) != (baseline_rows[qid] is None) for qid in query_ids):
            raise ValueError("Methods disagree on known-positive request eligibility")
        eligible = [qid for qid in sorted(query_ids) if method_rows[qid] is not None]
        differences = [method_rows[qid][row["metric"]] - baseline_rows[qid][row["metric"]] for qid in eligible]
        row.update(eligible_queries=len(eligible), total_queries=len(query_ids), training_seeds=list(seeds),
                   query_macro_delta=float(np.mean(differences)) if differences else None,
                   sampling_unit="Input-connected drug/study group after seed averaging within query",
                   difference="PBPK interaction minus capacity-matched concatenation",
                   inference_role="primary" if row["metric"] == "ndcg@10" else "descriptive")
        if row["metric"] != "ndcg@10":
            for key in ("cluster_sign_flip_p", "holm_p", "holm_family"):
                row.pop(key, None)
            row["p_values_reported"] = False
        else:
            row["primary_family_size"] = 1
            row.pop("holm_p", None)
            row.pop("holm_family", None)
            row["multiplicity"] = "One prespecified primary contrast; cluster_sign_flip_p is unadjusted."
    return {"comparisons": comparisons, "per_query_seed_means": averaged,
            "policy": "Equal seed weight within request, then equal request weight within connected group; groups are resampled. Seeds are not independent study observations."}


def average_common_execution(reports, seeds=SEEDS):
    """Use only a scenario measurable for both architectures in *every* seed.

    Per-seed full coverage is separately retained in the parent report. Averaging
    a successful seed over a failed/unmeasured one would hide selection coverage.
    """
    if set(reports) != set(METHODS):
        raise ValueError("Execution architecture family changed")
    maps = {}
    for method in METHODS:
        if set(reports[method]) != set(seeds):
            raise ValueError("Execution aggregation requires all prespecified seeds")
        for seed in seeds:
            rows = reports[method][seed]["execution"]["position_rows"]
            mapping = {(row["scenario_id"], row["endpoint"], row["rank"], row["stratum"]): row for row in rows}
            if len(mapping) != len(rows):
                raise ValueError("Duplicate execution scenario/endpoint/rank")
            maps[method, seed] = mapping
    populations = [set(mapping) for mapping in maps.values()]
    if any(keys != populations[0] for keys in populations[1:]):
        raise ValueError("Execution scenario populations differ")
    averaged = {method: {"execution": {"position_rows": []}} for method in METHODS}
    exclusions = []
    for key in sorted(populations[0]):
        rows = [maps[method, seed][key] for method in METHODS for seed in seeds]
        identity = {(row["compound"], row["publication_cluster_id"]) for row in rows}
        if len(identity) != 1:
            raise ValueError("Scenario drug/study identity differs between methods or seeds")
        valid = all(math.isclose(row["coverage"], 1., abs_tol=1e-12, rel_tol=0) and
                    row["absolute_log_error"] is not None and math.isfinite(row["absolute_log_error"]) and
                    row["absolute_log_error"] >= 0 for row in rows)
        if not valid:
            exclusions.append({"scenario_id": key[0], "endpoint": key[1], "rank": key[2], "stratum": key[3],
                               "reason": "Not fully measurable for both methods in every prespecified seed"})
            continue
        for method in METHODS:
            source = [maps[method, seed][key] for seed in seeds]
            row = {**source[0], "training_seeds": list(seeds)}
            for field in ("absolute_log_error", "signed_log_error", "within_twofold", "within_1_5_fold", "regret"):
                values = [item.get(field) for item in source]
                row[field] = float(np.mean(values)) if all(value is not None for value in values) else None
            row.update(coverage=1., top1_evaluable_probability=1., top1_absolute_log_error=row["absolute_log_error"])
            averaged[method]["execution"]["position_rows"].append(row)
    common = original.common_coverage_comparisons(averaged, own="pbpk_interaction")
    paired = []
    for rank in (1, 2, 3):
        rows = [row for report in averaged.values() for row in report["execution"]["position_rows"] if row["rank"] == rank]
        if not rows:
            continue
        value = original.stable_paired_execution(rows, METHODS, baseline=BASELINE, rank=rank, draws=2000)
        for comparison in value["comparisons"]:
            row = {k: v for k, v in comparison.items() if k not in ("two_sided_p", "holm_adjusted_p", "holm_prespecified_family_size")}
            row.update(rank=rank, inference_role="descriptive", p_values_reported=False,
                       difference="PBPK interaction minus capacity-matched concatenation")
            paired.append(row)
    return {"common_positions": common, "paired_descriptive": paired, "excluded_positions": exclusions,
            "per_scenario_seed_means": averaged,
            "policy": "Identical scenario intersection across both methods and all three seeds; average absolute log errors within scenario before the unchanged compound/study estimator. Full per-seed coverage is separately reported."}


def load_predictions(data):
    protocol_path = OUT / "protocol.json"
    protocol = read(protocol_path)
    if tuple(protocol["methods"]) != METHODS or tuple(protocol["seeds"]) != SEEDS:
        raise ValueError("Capacity protocol method/seed family differs")
    for field, expected in (("query_ids", [x["query_id"] for x in data["query_records"]]),
                            ("candidate_ids", [x["candidate_id"] for x in data["candidate_records"]]),
                            ("groups", list(data["groups"])), ("folds", data["folds"])):
        if protocol[field] != expected:
            raise ValueError("Frozen capacity data binding changed: " + field)
    refs = [reference(protocol_path)]
    for path, digest in protocol.get("implementation_sha256", {}).items():
        item = reference(path)
        if item["sha256"] != digest:
            raise ValueError("Frozen capacity implementation changed: " + path)
        refs.append(item)
    predictions, receipts = {}, {}
    for method in METHODS:
        predictions[method], receipts[method] = {}, {}
        for seed in SEEDS:
            path = OUT / method / "predictions" / f"seed{seed}.npz"
            receipt_path = path.with_suffix(".json")
            receipt = read(receipt_path)
            if (receipt.get("state") != "complete" or receipt.get("method") != method or receipt.get("seed") != seed or
                    receipt.get("protocol") != reference(protocol_path) or receipt.get("predictions") != reference(path)):
                raise ValueError("Incomplete or mismatched capacity prediction receipt")
            refs.extend((reference(path), reference(receipt_path)))
            with np.load(path, allow_pickle=False) as arrays:
                predictions[method][seed] = validate_arrays(arrays, data)
            # Source receipts and their directly referenced weights/partitions
            # are checked, without trusting a completed flag by itself.
            bindings = receipt.get("fits", []) + receipt.get("sources", []) + receipt.get("reused_sources", [])
            if not bindings:
                raise ValueError("Missing training or reuse provenance")
            for item in bindings:
                if reference(item["path"]) != item:
                    raise ValueError("Capacity training/reuse source changed")
                refs.append(item)
                if Path(item["path"]).suffix == ".json":
                    source = read(ROOT / item["path"])
                    if isinstance(source, dict):
                        for value in source.values():
                            if isinstance(value, dict) and set(value) == {"path", "sha256"}:
                                if reference(value["path"]) != value:
                                    raise ValueError("Bound capacity fit artifact changed")
                                refs.append(value)
            receipts[method][seed] = receipt
    return predictions, refs, receipts


def full_list(data, scores, method, seed):
    rows = []
    for i in data["core_indices"]:
        query = data["query_records"][i]
        rows.append({"method": method, "seed": seed, "query_id": query["query_id"],
                     "task_type": query["task_type"], "purpose": TASK_GROUP[query["task_type"]],
                     "compound": query["compound_id"],
                     **full_list_row(scores[0, i], data["grades"][i], data["applicability"][i])})
    scopes = ("all", "direct_prediction", "mechanism_reference", "parameterization_template", *sorted(TASK_GROUP))
    summary = [{"method": method, "seed": seed, "scope": scope,
                **summarize(rows if scope == "all" else [row for row in rows if scope in (row["purpose"], row["task_type"])])}
               for scope in scopes]
    return {"summary": summary, "per_query": rows}


def rank_ensemble(predictions, seeds=SEEDS):
    """Secondary ensemble uses rank scales, never arbitrary raw score scales."""
    if set(predictions) != set(seeds):
        raise ValueError("Ensemble requires all prespecified seeds")
    shapes = {(scores.shape, probabilities.shape) for scores, probabilities in predictions.values()}
    if len(shapes) != 1:
        raise ValueError("Ensemble prediction axes differ")
    scores = np.mean([r.percentile_ranks(predictions[seed][0]) for seed in seeds], axis=0)
    probabilities = np.mean([predictions[seed][1].astype(np.float64) for seed in seeds], axis=0)
    return scores, probabilities


def extended_common(endpoints, seeds=SEEDS):
    """Descriptive 11-endpoint comparisons with identical all-seed coverage.

    Tmax remains an absolute-hours error, while fold-error endpoints are averaged
    in logarithmic space. Native and condition-matched outputs remain separate.
    """
    if not set(seeds) <= set(endpoints):
        raise ValueError("Missing seeds for extended common-scenario evaluation")
    definitions = {row["endpoint_key"]: row for row in endpoints[seeds[0]]["endpoint_definitions"]}
    maps = {}
    fields = ("scenario_id", "endpoint_key", "rank", "analysis_stratum", "prediction_kind")
    for seed in seeds:
        if {row["endpoint_key"]: row for row in endpoints[seed]["endpoint_definitions"]} != definitions:
            raise ValueError("Endpoint qualification differs between seeds")
        for method in METHODS:
            rows = [row for row in endpoints[seed]["position_rows"] if row["method_id"] == method]
            mapping = {tuple(row[k] for k in fields): row for row in rows}
            if len(mapping) != len(rows):
                raise ValueError("Duplicate extended endpoint row")
            maps[method, seed] = mapping
    populations = list(maps.values())
    if any(set(mapping) != set(populations[0]) for mapping in populations[1:]):
        raise ValueError("Extended endpoint scenario populations differ")
    totals, paired = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for key in sorted(populations[0]):
        sid, endpoint, rank, stratum, prediction_kind = key
        panel = (endpoint, rank, stratum, prediction_kind)
        totals[panel].append(sid)
        rows = [maps[method, seed][key] for method in METHODS for seed in seeds]
        if len({(row["compound"], row["publication_cluster_id"]) for row in rows}) != 1:
            raise ValueError("Extended scenario study identity differs")
        field = "absolute_error_h" if definitions[endpoint]["endpoint"] == "Tmax" else "absolute_log_error"
        if not all(math.isclose(row["coverage"], 1., abs_tol=1e-12, rel_tol=0) and
                   row.get(field) is not None and math.isfinite(row[field]) and row[field] >= 0 for row in rows):
            continue
        for method in METHODS:
            source = maps[method, seeds[0]][key]
            paired[panel][method].append({"scenario_id": sid, "compound": source["compound"],
                "publication_cluster_id": source["publication_cluster_id"],
                "error": float(np.mean([maps[method, seed][key][field] for seed in seeds]))})
    output = []
    for panel, population in sorted(totals.items()):
        endpoint, rank, stratum, kind = panel
        hours = definitions[endpoint]["endpoint"] == "Tmax"
        selected = paired[panel]
        row = {"endpoint_key": endpoint, "rank": rank, "analysis_stratum": stratum, "prediction_kind": kind,
               "seeds": list(seeds), "total_scenarios": len(population),
               "shared_scenarios": len(selected.get(METHODS[0], [])),
               "scenario_ids": [x["scenario_id"] for x in selected.get(METHODS[0], [])],
               "metric_kind": "MAE_h" if hours else "GMFE", "inference_role": "descriptive"}
        for method in METHODS:
            value = r.ev.hierarchical_mean(selected.get(method, []), "error")
            row[method] = value if hours or value is None else math.exp(value)
        output.append(row)
    return output


def extended_reports(predictions, data):
    import evaluate_component_model_integration as extended
    path = ROOT / "results/data/model_recommendation_extended_metrics.json"
    frozen = read(path)
    execution = data["execution_indices"]
    qids = [data["query_records"][i]["query_id"] for i in execution]
    cids = [row["candidate_id"] for row in data["candidate_records"]]
    saved = extended.METHODS
    try:
        extended.METHODS = [{"method_id": method, "method": DISPLAY[method]} for method in METHODS]
        result = {seed: extended.evaluate_extended(
            {method: predictions[method][seed][0][0, execution] for method in METHODS}, qids, cids, frozen)
            for seed in (*SEEDS, "rank_ensemble")}
    finally:
        extended.METHODS = saved
    return result, [reference(path), reference(extended.__file__)]


def report_text(summary):
    primary = next(row for row in summary["paired_seed_average_quality"]["comparisons"] if row["metric"] == "ndcg@10")
    lines = ["# 等容量交互结构对照", "", "固定相同文本、204维条件、标签、分组和训练规则；仅比较交互结构与等容量简单拼接。结果为已有开发资料上的内部验证。", "",
             "| 方法 | 种子 | 已判子集 NDCG@10 | Y-F1 | 敏感度 | 完整列表首位未知 |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary["quality_rows"]:
        lines.append(f"| {DISPLAY[row['method']]} | {row['seed']} | {row['ndcg@10']:.4f} | {row['positive_f1']:.4f} | {row['sensitivity']:.1%} | {row['top1_unknown']:.1%} |")
    delta, ci, p = primary["mean_group_delta"], primary["bootstrap_95ci"], primary.get("cluster_sign_flip_p")
    lines += ["", f"主比较先在每条请求内平均三个种子的NDCG，再按药物/研究关联组配对。交互减拼接的组平均差为 {delta:.4f}；95% CI " +
              (f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "不可估计") + (f"，p={p:.4f}。" if p is not None else "。"),
              f"该排序结果只涉及 {primary['eligible_queries']}/{primary['total_queries']} 条有已知正例的请求；未知候选仅在这个条件性指标中排除，完整列表统计仍保留。",
              "", "若区间跨零，不能据此证明方法等效或交互结构更优。多种子结果不等同于新增独立验证数据。",
              "", "完整输出：`report/summary.json`；逐种子文档指标 `report/quality.csv`；完整列表及分用途 `report/full_list.csv`；11类终点 `report/endpoint_positions.csv`。共同执行场景按两种方法、全部种子都可计分的交集比较；完整覆盖另行保留。两条曲线仍仅为同一研究的案例。"]
    return "\n".join(lines) + "\n"


def build():
    data = prepare()
    predictions, refs, receipts = load_predictions(data)
    selection_path = ROOT / "experiments/model_recommendation/frontier_core_145_20260921/selection_manifest.json"
    selection = read(selection_path)
    binary = [i for i in data["core_indices"] if selection["compound_split"][data["query_records"][i]["compound_id"]] == "test"]
    if len(binary) != 102:
        raise ValueError("Original binary applicability request denominator changed")
    refs += [reference(selection_path), reference(r.ev.PRIMARY), reference(__file__),
             reference(r.__file__), reference(original.__file__),
             reference(ROOT / "scripts/reviewer_full_list_evaluation.py"),
             reference(ROOT / "scripts/pbpk_match_light_paper_metrics.py")]
    contract = r.ev.clinical_contract()
    reports, full, rows = {}, {}, []
    for method in METHODS:
        reports[method], full[method] = {}, {}
        for seed in SEEDS:
            scores, probabilities = predictions[method][seed]
            report = r._evaluate(data, scores, probabilities[0], method, contract)
            report["execution"]["omission_summary"] = original.omission_summary(report["execution"]["omissions"])
            app = binary_applicability(probabilities[0, binary, :, 0], np.asarray(data["applicability"])[binary])
            if (app["known_pairs"], app["positive_pairs"]) != (16410, 74):
                raise ValueError("Original known Y/N evaluation denominator changed")
            report["quality"]["paper_binary_102"] = app
            reports[method][seed] = report
            full[method][seed] = full_list(data, scores, method, seed)
            all_row = next(row for row in full[method][seed]["summary"] if row["scope"] == "all")
            for metric in METRICS:
                if not np.isclose(all_row[metric], report["quality"]["judged_only"]["metrics"][metric], atol=1e-12, rtol=0):
                    raise ValueError("Full-list and original judged metrics differ")
            rows.append({"method": method, "seed": seed, **report["quality"]["judged_only"]["metrics"],
                         "eligible_queries": report["quality"]["judged_only"]["eligible_queries"],
                         "positive_f1": app["positive_f1"], "sensitivity": app["positive_recall"],
                         "positive_precision": app["positive_precision"], "brier": app["brier"],
                         "top1_supported": all_row["top1_supported"], "top1_unknown": all_row["top1_unknown"],
                         "top10_known_fraction": all_row["top10_known_fraction"]})
    groups = {row["query_id"]: data["groups"][i] for i, row in enumerate(data["query_records"])}
    ensemble_predictions = {method: rank_ensemble(predictions[method]) for method in METHODS}
    ensemble_reports, ensemble_full = {}, {}
    for method, (scores, probabilities) in ensemble_predictions.items():
        report = r._evaluate(data, scores, probabilities[0], method, contract)
        report["execution"]["omission_summary"] = original.omission_summary(report["execution"]["omissions"])
        report["quality"]["paper_binary_102"] = binary_applicability(
            probabilities[0, binary, :, 0], np.asarray(data["applicability"])[binary])
        ensemble_reports[method] = report
        ensemble_full[method] = full_list(data, scores, method, "rank_ensemble")
    endpoints, sources = extended_reports(
        {method: {**predictions[method], "rank_ensemble": ensemble_predictions[method]} for method in METHODS}, data)
    refs += sources
    summary = {"schema": "pbpk-capacity-matched-report-v1", "state": "complete", "methods": list(METHODS), "seeds": list(SEEDS),
               "independent_external_validation": False, "new_llm_inference": False, "new_pbpk_execution": False,
               "quality_rows": rows, "reports": reports, "full_list": full,
               "seed_variability": {method: r.seed_variability([reports[method][seed] for seed in SEEDS]) for method in METHODS},
               "paired_seed_average_quality": paired_seed_average_quality(reports, groups),
               "common_execution_by_seed": {seed: original.common_coverage_comparisons(
                   {method: reports[method][seed] for method in METHODS}, own="pbpk_interaction") for seed in SEEDS},
               "paired_seed_average_execution": average_common_execution(reports), "extended_endpoints": endpoints,
               "extended_common_seed_average": extended_common(endpoints),
               "extended_common_by_seed": {seed: extended_common(endpoints, seeds=(seed,)) for seed in (*SEEDS, "rank_ensemble")},
               "secondary_rank_ensemble": {"reports": ensemble_reports, "full_list": ensemble_full,
                   "common_execution": original.common_coverage_comparisons(ensemble_reports, own="pbpk_interaction"),
                   "inference_role": "secondary_descriptive", "construction": "Arithmetic mean of exact-tie percentile ranks across the three seeds; arithmetic mean of calibrated probabilities. No best-seed selection."},
               "prediction_receipts": receipts, "sources": refs,
               "limits": ["Labels remain the existing automatic judgments; this experiment does not validate unknown grades.",
                          "Known-positive judged-subset ranking is reported beside all-request original-position support, separately by use.",
                          "Only one primary test: seed-averaged paired grouped NDCG. Other contrasts are descriptive.",
                          "Three training seeds are algorithmic variability, not three independent data sets.",
                          "Clinical comparisons retain endpoint identity and intersection coverage; curve cases come from one study."]}
    # Bind immutable inputs once more before publishing; avoid mixed-run results.
    for item in refs:
        if reference(item["path"]) != item:
            raise ValueError("Input changed while producing capacity report")
    destination = OUT / "report"
    write(destination / "summary.json", json_string_keys(summary))
    tables = {"quality.csv": rows,
              "full_list.csv": [row for method in METHODS for seed in SEEDS for row in full[method][seed]["summary"]],
              "full_list_per_query.csv": [{k: v for k, v in row.items() if k != "judged_only"} for method in METHODS for seed in SEEDS for row in full[method][seed]["per_query"]],
              "common_execution.csv": [{"seed": seed, **row} for seed, values in summary["common_execution_by_seed"].items() for row in values],
              "endpoint_positions.csv": [{"seed": seed, **row} for seed, values in endpoints.items() for row in values["position_summary"]],
              "endpoint_common_seed_average.csv": summary["extended_common_seed_average"],
              "ensemble_full_list.csv": [row for method in METHODS for row in ensemble_full[method]["summary"]]}
    for filename, table in tables.items():
        (destination / filename).write_text(r._csv(table))
    (OUT / "REPORT.md").write_text(report_text(summary))
    write(destination / "completion.json", {"state": "complete", "protocol": reference(OUT / "protocol.json"),
                                           "summary": reference(destination / "summary.json"), "methods": list(METHODS),
                                           "seeds": list(SEEDS), "outputs": [reference(destination / name) for name in tables] + [reference(OUT / "REPORT.md")]})
    print(r._csv(rows))
    return summary


if __name__ == "__main__":
    build()
