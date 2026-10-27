#!/usr/bin/env python3
"""Publish all predeclared controls, using the unchanged paper estimands."""
from __future__ import annotations

from collections import defaultdict
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
from pbpk_kb.recommendation.pbpk_match_v2 import reporting as r
from pbpk_kb.recommendation.pbpk_match_v2.experiment import read, write, sha
from reviewer_controls_run import OUT, LIGHT, METHODS, RULES, prepare, reference
from pbpk_match_light_paper_metrics import binary_applicability, known_topk_coverage
import summarize_pbpk_match_light as original

DISPLAY = {"structured_mlp": "Structured MLP", "text_mlp": "Text MLP", "concat_mlp": "Simple concatenation MLP",
           "identity_condition_rule": "Identity/condition rule", "service_typed_rule": "Service typed-match rule",
           "pbpk_match_light_document": "PBPK-Match without error correction", "pbpk_match_light": "PBPK-Match"}
BASELINE = "pbpk_match_light_document"


def paper(summary):
    """Compact main-text inclusion and factual companion text, no winning claim."""
    directory = OUT / "paper"
    directory.mkdir(parents=True, exist_ok=True)
    order = ["identity_condition_rule", "service_typed_rule", "structured_mlp", "text_mlp", "concat_mlp", BASELINE]
    names = {**DISPLAY, "concat_mlp": "Text + conditions (concat)", BASELINE: "PBPK-Match (no error correction)"}
    rows = {row["method_id"]: row for row in summary["quality_rows"]}
    number = lambda x: "--" if x is None else f"{x:.3f}"
    lines = [r"\begingroup", r"\fontsize{8.3}{10}\selectfont", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Method & NDCG@10 & Recall@10 & Y-F1 & Sens. (\%) & Top-1 unknown (\%) \\", r"\midrule"]
    for method in order:
        row = rows[method]
        sensitivity = row["positive_recall"]
        unknown = summary["reports"][method]["quality"]["coverage"]["full_catalog_top1_unknown_rate"]
        lines.append(" & ".join([names[method], number(row["ndcg@10"]), number(row["recall@10"]), number(row["positive_f1"]),
                                   "--" if sensitivity is None else f"{100*sensitivity:.1f}", f"{100*unknown:.1f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    (directory / "table.tex").write_text("\n".join(lines) + "\n")
    clinical = {row["comparator"]: row for row in summary["common_execution"]
                if row["endpoint"] == "Cmax_reference_vs_group" and row["rank"] == 1}
    curve = {row["comparator"]: row for row in summary["common_execution"]
             if row["endpoint"] == "curve_gmfe" and row["rank"] == 1}
    lines = [r"\begingroup", r"\fontsize{8.3}{10}\selectfont", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Control & Cov. (\%) & Cmax $n$ & PBPK-Match & Control & Curve $n$ & GMFE \\", r"\midrule"]
    for method in order[:-1]:
        c, v = clinical[method], curve[method]
        full = next(row for row in summary["reports"][method]["execution"]["position_summary"]
                    if row["endpoint"] == "Cmax_reference_vs_group" and row["rank"] == 1)
        lines.append(" & ".join([names[method], f"{100*full['coverage']:.1f}", str(c["shared_scenarios"]), number(c["own_gmfe"]), number(c["comparator_gmfe"]),
                                   str(v["shared_scenarios"]), number(v["comparator_gmfe"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    (directory / "clinical_table.tex").write_text("\n".join(lines) + "\n")
    statements = [
        "使用相同自动标签、冻结文本向量和分组方案补充三个监督对照；简单拼接与PBPK-Match使用相同204维条件信息，区别为是否加入文本向量绝对差与逐元素乘积。每种方法完成5×(3+1)=20次拟合；内层确定训练轮次与温度，外层只作留出评价。主比较关闭误差修正，避免把执行误差监督计为文档融合结构的贡献。",
        "表注：NDCG与Recall沿用40条含已知正例请求的已判子集；敏感度为原102请求中74个已知Y对的检出比例，未知不作负例。首位未知率按全部145请求的完整列表及并列期望计算。两条规则无概率输出，Y-F1和敏感度不适用。PBPK-Match最终版的NDCG、Recall、Y-F1、敏感度与无误差修正版相同，故本表不重复列出。",
        "临床表注：每行仅在该控制与PBPK-Match无误差修正版共同可计分的场景比较首位GMFE；不同控制的交集可以不同，不能跨行宣称统一排名。曲线仅2场景、同一研究，是案例而非总体泛化证据。全部11终点、三个精确位置、完整覆盖和缺失仍在report/summary.json中保留。"]
    for method in order[:-1]:
        row = rows[method]
        test = next(t for t in summary["paired_quality"] if t["method"] == method and t["metric"] == "ndcg@10")
        delta, ci = test["mean_group_delta"], test["bootstrap_95ci"]
        c, v = clinical[method], curve[method]
        statements.append(f"{DISPLAY[method]}：NDCG={row['ndcg@10']:.4f}；相对PBPK-Match的分组平均NDCG差={delta:.4f}，95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]，Holm p={test['holm_p']:.4f}。共同{c['shared_scenarios']}个Cmax场景，PBPK-Match/控制GMFE={number(c['own_gmfe'])}/{number(c['comparator_gmfe'])}；共同{v['shared_scenarios']}个曲线场景的控制GMFE={number(v['comparator_gmfe'])}。")
        paired = next(t for t in summary["paired_execution"][0]["comparisons"] if t["method"] == method and t["endpoint"] == "Cmax_reference_vs_group")
        statements.append("首位Cmax配对绝对对数误差统计（控制减PBPK）：" + str({k: paired.get(k) for k in ("mean_paired_absolute_log_error_difference", "bootstrap_95ci", "holm_adjusted_p", "independent_compound_publication_blocks", "state", "reason")}))
    statements.append("这些新增对照仅检验现有开发资料和自动标签下的相对贡献；不解决专家标签可靠性，也不构成独立外部验证。若简单拼接与PBPK-Match差异不明确，不将新融合结构表述为已建立优势。")
    (directory / "analysis.txt").write_text("\n\n".join(statements) + "\n")


def load_predictions(data):
    protocol = read(OUT / "protocol.json")
    if protocol["folds"] != data["folds"]:
        raise ValueError("Control folds changed")
    for path, digest in protocol["implementation_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Control implementation changed after protocol freeze: " + path)
    q, c = len(data["query_records"]), len(data["candidate_records"])
    predictions, refs, integrity = {}, [reference(OUT / "protocol.json")], {}
    for method in (*METHODS, *RULES):
        path = OUT / method / "predictions/seed17.npz"
        receipt = read(path.with_suffix(".json"))
        if receipt["state"] != "complete" or receipt["predictions"] != reference(path):
            raise ValueError("Incomplete prediction receipt")
        refs.extend([reference(path), reference(path.with_suffix(".json"))])
        with np.load(path) as a:
            scores, probabilities = a["scores"].copy(), a["probabilities"].copy()
            if not np.array_equal(a["indices"], np.arange(q)):
                raise ValueError("Wrong request mapping")
        if scores.shape != (3, q, c) or not np.isfinite(scores).all() or probabilities.shape != (3, q, c, 3):
            raise ValueError("Incomplete full-catalog predictions")
        if method in METHODS:
            if not np.isfinite(probabilities).all():
                raise ValueError("Missing supervised probability")
            fold_audits = []
            if len(receipt["fits"]) != 20:
                raise ValueError("Missing fit receipt")
            for item in receipt["fits"]:
                if reference(ROOT / item["path"]) != item:
                    raise ValueError("Fit completion receipt changed")
                fit = read(ROOT / item["path"])
                for value in fit.values():
                    if isinstance(value, dict) and "path" in value and reference(ROOT / value["path"]) != value:
                        raise ValueError("Fit source changed")
            for f, fold in enumerate(data["folds"]):
                directory = OUT / method / f"runs/seed17/fold{f}"
                selected = read(directory / "selection.json")
                metadata = read(directory / "outer/metadata.json")
                groups = set(data["groups"][i] for i in fold["outer_test"])
                fitting = set(data["groups"][i] for i in metadata["train_indices"])
                selection = set(data["groups"][i] for i in selected["selection_indices"])
                calibration = set(selected["calibration"]["fit_groups"])
                if groups & (fitting | selection | calibration):
                    raise ValueError("Outer group leaked into fitting/selection/calibration")
                if metadata["error_train_indices"] or not metadata["fixed_epochs"]:
                    raise ValueError("Control used PK supervision or outer early stopping")
                fold_audits.append({"fold": f, "outer_groups": sorted(groups), "no_train_overlap": True,
                    "no_selection_overlap": True, "no_calibration_overlap": True,
                    "trainable_parameters": metadata["trainable_parameters"], "epochs": metadata["epochs_completed"],
                    "temperature": selected["calibration"]["temperature"], "integrity": read(directory / "outer/integrity.json")})
            integrity[method] = fold_audits
        elif not np.isnan(probabilities).all():
            raise ValueError("Rule probability must be explicitly unavailable")
        predictions[method] = (scores, probabilities[0])
    path = LIGHT / "predictions/seed17.npz"
    refs.append(reference(path))
    with np.load(path) as a:
        predictions[BASELINE] = (a["base_scores"].copy(), a["probabilities"][0].copy())
        predictions["pbpk_match_light"] = (a["scores"].copy(), a["probabilities"][0].copy())
    return predictions, refs, integrity


def main():
    data = prepare()
    predictions, refs, integrity = load_predictions(data)
    contract = r.ev.clinical_contract()
    reports = {method: r._evaluate(data, scores, probs, method, contract) for method, (scores, probs) in predictions.items()}
    selection = read(ROOT / "experiments/model_recommendation/frontier_core_145_20260921/selection_manifest.json")
    binary = [i for i in data["core_indices"] if selection["compound_split"][data["query_records"][i]["compound_id"]] == "test"]
    if len(binary) != 102:
        raise ValueError("Original paper binary evaluation scope changed")
    rows = []
    for method, report in reports.items():
        scores, probabilities = predictions[method]
        quality = report["quality"]
        report["execution"]["omission_summary"] = original.omission_summary(report["execution"]["omissions"])
        quality["paper_binary_102"] = binary_applicability(probabilities[binary, :, 0], np.asarray(data["applicability"])[binary]) if np.isfinite(probabilities[binary]).all() else None
        quality["known_top10_coverage"] = known_topk_coverage(scores[0, data["core_indices"]], np.asarray(data["grades"])[data["core_indices"]])
        app = quality["paper_binary_102"]
        preference = quality["same_drug_different_family_preference"]
        row = {"method_id": method, "method": DISPLAY[method], **quality["judged_only"]["metrics"],
            "eligible_queries": quality["judged_only"]["eligible_queries"],
            "pair_accuracy": preference["query_macro_accuracy"], "pair_queries": preference["eligible_queries"], "pair_count": preference["pairs"],
            "known_top10": quality["known_top10_coverage"]["mean"],
            "positive_f1": app["positive_f1"] if app else None, "positive_recall": app["positive_recall"] if app else None,
            "positive_precision": app["positive_precision"] if app else None, "brier": app["brier"] if app else None,
            "binary_known_pairs": app["known_pairs"] if app else None, "binary_positives": app["positive_pairs"] if app else None}
        rows.append(row)
        if app and (app["known_pairs"], app["positive_pairs"]) != (16410, 74):
            raise ValueError("Original paper binary denominators changed")
    published = read(ROOT / "results/data/pbpk_match_light_paper_metrics.json")
    for old in published["quality_rows"]:
        current = next(row for row in rows if row["method_id"] == old["method_id"])
        for key in ("ndcg@10", "recall@10", "mrr"):
            if not np.isclose(current[key], old[key], atol=1e-12, rtol=0):
                raise ValueError("Original PBPK metric failed exact reproduction: " + key)
    family = [BASELINE, *METHODS, *RULES]
    paired_reports = {m: reports[m] for m in family}
    groups = {row["query_id"]: data["groups"][i] for i, row in enumerate(data["query_records"])}
    paired_quality = original.stable_paired_quality({m: v["quality"] for m, v in paired_reports.items()}, groups, baseline=BASELINE)
    paired_execution = []
    for rank in (1, 2, 3):
        positions = [row for v in paired_reports.values() for row in v["execution"]["position_rows"] if row["rank"] == rank]
        paired_execution.append({"rank": rank, **original.stable_paired_execution(positions, family, baseline=BASELINE, rank=rank)})
    # All 11 endpoint families use already qualified, locked NCA candidate errors.
    import evaluate_component_model_integration as extended
    extended.METHODS = [{"method_id": m, "method": DISPLAY[m]} for m in predictions]
    execution_indices = data["execution_indices"]
    qids = [data["query_records"][i]["query_id"] for i in execution_indices]
    cids = [row["candidate_id"] for row in data["candidate_records"]]
    source = ROOT / "results/data/model_recommendation_extended_metrics.json"
    scores = {m: values[0][0, execution_indices] for m, values in predictions.items()}
    endpoint_report = extended.evaluate_extended(scores, qids, cids, read(source))
    refs += [reference(source), reference(Path(__file__)), reference(r.ev.PRIMARY), reference(ROOT / "scripts/evaluate_component_model_integration.py")]
    summary = {"schema": "reviewer-matched-controls-report-v1", "completed": True, "methods": list(predictions),
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "scikit-learn")},
        "same_information_comparison": "concat_mlp vs pbpk_match_light_document; same text embeddings, 204 condition features, labels, fold schedule and loss",
        "information_ablations": "structured_mlp excludes text; text_mlp excludes explicit structured inputs",
        "architecture": "Simple concatenation excludes explicit absolute-difference and elementwise-product text interactions; width/projection/heads/regularization retained",
        "external_baselines_rerun": False, "new_llm_labels": False, "independent_validation": False,
        "primary_architecture_comparator": BASELINE, "comparison_family": family,
        "quality_rows": rows, "reports": reports, "fold_audit": integrity, "paired_quality": paired_quality,
        "paired_execution": paired_execution, "common_execution": original.common_coverage_comparisons(reports, own=BASELINE),
        "extended_endpoints": endpoint_report, "sources": refs,
        "limits": ["Automatic labels remain unvalidated by experts; all estimates are development/internal validation.",
                   "Single seed 17; no claim of seed stability.", "No outcome-based hyperparameter search; same budget, not equal parameter count.",
                   "Original service rule is within-layer typed scoring, not the full online relevance/use-tier gate; missing rule probabilities are not fabricated."]}
    destination = OUT / "report"
    write(destination / "summary.json", summary)
    for filename, table in {"quality.csv": rows, "paired_quality.csv": paired_quality,
        "execution_positions.csv": [row for v in reports.values() for row in v["execution"]["position_summary"]],
        "common_execution.csv": summary["common_execution"],
        "endpoint_positions.csv": endpoint_report["position_summary"]}.items():
        (destination / filename).write_text(r._csv(table))
    write(destination / "completion.json", {"completed": True, "protocol": reference(OUT / "protocol.json"),
        "summary": reference(destination / "summary.json"), "methods": list(predictions), "fits_completed": 60,
        "new_llm_inference": False, "external_baselines_rerun": False})
    paper(summary)
    state = read(OUT / "status.json")
    if state.get("phase") == "complete":
        old_error = state.pop("error", None)
        if old_error:
            state["resolved_startup_error"] = old_error
        state["evaluation_completed"] = True
        state["evaluation_receipt"] = reference(destination / "completion.json")
        write(OUT / "status.json", state)
    print(r._csv(rows))


if __name__ == "__main__":
    main()
