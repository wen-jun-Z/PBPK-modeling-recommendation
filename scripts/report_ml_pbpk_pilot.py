#!/usr/bin/env python3
"""Report a completed ML/PBPK pilot without fitting or running any model.

Paper-reported endpoints, observed-curve comparisons, and native predictions
remain separate. The prediction used for a published Cmax/Tmax comparison is
sampled at available clinical times, preserving the dense native result too.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/blueprint_design/ml_hybrid_pilot_20260928"
METHOD_LABELS = {
    "median_direct": "Median direct",
    "rf_direct": "RF direct",
    "rf_vss_mapping": "RF + Vss mapping",
}
ENDPOINTS = {
    "plasma_concentration_time": ("Plasma curve", None, "mg/L vs h"),
    "auc": ("AUC (requested window)", "value_mg_h_per_l", "mg*h/L"),
    "cmax": ("Cmax (native grid)", "value_mg_per_l", "mg/L"),
    "tmax": ("Tmax (native grid)", "value_h", "h"),
    "c_last": ("Clast (requested end)", "value_mg_per_l", "mg/L"),
    "terminal_half_life": ("Terminal half-life", "value_h", "h"),
    "auc_inf": ("AUC infinity", "value_mg_h_per_l", "mg*h/L"),
    "clearance": ("CL", "value_l_per_h", "L/h"),
    "vz": ("Vz", "value_l", "L"),
    "mrt": ("MRT", "value_h", "h"),
    "vss": ("Vss", "value_l", "L"),
}
PARAMETER_LABELS = {
    "human_VDss_L_kg": "Vss",
    "human_CL_mL_min_kg": "CL",
    "human_thalf": "Half-life",
    "human_mrt": "MRT",
    "human_fup": "Unbound fraction",
}


def scoped(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    path.relative_to(ROOT)
    return path


def read(path):
    return json.loads(scoped(path).read_text())


def digest(path):
    return hashlib.sha256(scoped(path).read_bytes()).hexdigest()


def rel(path):
    return str(scoped(path).relative_to(ROOT))


def valid(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fold(predicted, observed):
    if valid(predicted) and valid(observed) and predicted > 0 and observed > 0:
        return max(predicted / observed, observed / predicted)
    return None


def fmt(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite report value")
        return f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    text = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    text.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in rows)
    return "\n".join(text)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty result table")
    with scoped(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def native_value(document, endpoint):
    item = document.get("endpoints", {}).get(endpoint, {})
    key = ENDPOINTS[endpoint][1]
    if item.get("status", "completed") != "completed":
        return None, item.get("reason", "native_endpoint_not_estimable")
    if key is None:
        return None, None if item.get("time_h") else "native_curve_unavailable"
    value = item.get(key)
    if not valid(value):
        return None, "native_endpoint_value_unavailable"
    return float(value), None


def provenance(case):
    source = case["clinical_source"]
    return {
        "reference_source_url": source.get("primary_url", source.get("url")),
        "reference_source_path": source.get("primary_local_path", source.get("path")),
        "reference_source_sha256": source.get("primary_sha256", source.get("sha256")),
        "reference_source_location": "Table 1; IV 1 mg" if case["drug"] == "midazolam" else "Table 1; IV control",
    }


def clinical_rows(cases, methods, summary, locks):
    rows, native_rows = [], []
    for case in cases:
        # Read the original request only after main() verifies both manifest
        # and run-protocol checksums; no unverified embedded copy supplies MW.
        request = read(case["request_path"])
        mw = next(float(e["value"]) for e in request["evidence"] if e["parameter"] == "molecular_weight")
        weight = float(request["human"]["weight_kg"])
        for method in methods:
            key = case["drug"] + "/" + method
            job = summary["results"][key]
            result = job["result"]
            metrics = job.get("metrics", {})
            document, endpoint_path = {}, None
            prediction_path = result.get("prediction_path")
            failure = result.get("reason") or result.get("gaps") or result.get("status")
            if isinstance(failure, (dict, list)):
                failure = json.dumps(failure, ensure_ascii=False)
            if result.get("status") == "predicted":
                lock = locks["predictions"].get(key)
                if not lock or scoped(prediction_path) != scoped(lock["path"]) or digest(prediction_path) != lock["sha256"]:
                    raise ValueError("Locked prediction mismatch: " + key)
                endpoint_path = scoped(result["endpoints_path"])
                endpoint_lock = lock.get("endpoints")
                if (not endpoint_lock or endpoint_path != scoped(endpoint_lock["path"])
                        or digest(endpoint_path) != endpoint_lock["sha256"]):
                    raise ValueError("Locked native endpoint mismatch: " + key)
                document = read(endpoint_path)
                failure = None
            for endpoint, (label, value_key, unit) in ENDPOINTS.items():
                value, reason = native_value(document, endpoint)
                item = document.get("endpoints", {}).get(endpoint, {})
                native_rows.append({
                    "drug": case["drug"], "case_id": case["case_id"], "study_id": case["study_id"],
                    "method": method, "endpoint": endpoint, "endpoint_label": label, "unit": unit,
                    "native_prediction": value,
                    "status": "completed" if not (failure or reason) else "unavailable",
                    "unavailable_reason": failure or reason,
                    "curve_points": len(item.get("time_h", [])) if endpoint == "plasma_concentration_time" else None,
                    "prediction_path": rel(prediction_path) if prediction_path else None,
                    "endpoint_path": rel(endpoint_path) if endpoint_path else None,
                    "definition": item.get("definition"),
                    "reference_value": None, "reference_status": "prediction_inventory_not_reference_table",
                })
            for reference in case.get("published_endpoint_references", []):
                endpoint = reference["endpoint"]
                native, reason = native_value(document, endpoint)
                unit = reference["unit"]
                native_unit = ENDPOINTS[endpoint][2]
                if unit == "mL/min/kg" and native_unit == "L/h":
                    if native is not None:
                        native = native / (weight * .06)
                    native_unit = unit
                if unit != native_unit:
                    raise ValueError(f"Unimplemented unit conversion: {endpoint}: {native_unit} -> {unit}")
                predicted, matched, basis = native, None, "native_model_endpoint"
                if endpoint in ("cmax", "tmax"):
                    metric = metrics.get("sampled_cmax" if endpoint == "cmax" else "sampled_tmax", {})
                    matched = metric.get("predicted_umol_l" if endpoint == "cmax" else "predicted_h")
                    if valid(matched):
                        matched = float(matched) * mw / 1000 if endpoint == "cmax" else float(matched)
                    else:
                        matched = None
                    predicted = matched
                    basis = "maximum_at_available_clinical_sample_times" if endpoint == "cmax" else "earliest_maximum_at_available_clinical_sample_times"
                    reason = "clinical_sampling_matched_prediction_unavailable" if predicted is None else None
                elif endpoint == "terminal_half_life":
                    matched = metrics.get("terminal_half_life", {}).get("predicted", {}).get("value_h")
                observed = float(reference["value"])
                reason = failure or reason
                status = "evaluated" if predicted is not None and not reason else "unavailable"
                if status != "evaluated":
                    predicted = None
                fold_error = None if endpoint == "tmax" else fold(predicted, observed)
                absolute = abs(predicted - observed) if endpoint == "tmax" and predicted is not None else None
                comparability = "representative_model_vs_published_cohort_summary"
                if endpoint in ("cmax", "tmax"):
                    comparability = "approximate_available_clinical_sampling_schedule"
                if endpoint == "terminal_half_life":
                    comparability = "limited_terminal_window_not_matched"
                    if status == "evaluated":
                        status = "descriptive_only"
                note = case.get("published_reference_caveat", "")
                if endpoint in ("cmax", "tmax"):
                    note += " Available digitized clinical sample times approximate the original individual schedules. Native dense-grid peak is supplementary, especially after IV bolus."
                if endpoint == "terminal_half_life":
                    note += " Native terminal window is model-only and can extend below clinical quantification; matched-window half-life is shown separately when estimable."
                rows.append({
                    "drug": case["drug"], "case_id": case["case_id"], "study_id": case["study_id"],
                    "method": method, "endpoint": endpoint, "unit": unit,
                    "predicted": predicted, "observed_reported": observed,
                    "fold_error": fold_error, "absolute_error_h": absolute,
                    "within_2fold": fold_error <= 2 if fold_error is not None and status == "evaluated" else None,
                    "native_prediction_supplement": native,
                    "native_fold_error_supplement": None if endpoint == "tmax" else fold(native, observed),
                    "sampling_matched_prediction_supplement": matched,
                    "sampling_matched_fold_error_supplement": None if endpoint == "tmax" else fold(matched, observed),
                    "prediction_basis": basis, "reference_basis": "clinical_paper_reported_summary",
                    "reference_statistic": reference.get("statistic"),
                    "reference_ci95": json.dumps(reference.get("ci95")) if reference.get("ci95") else None,
                    "reference_sd": reference.get("sd"), "status": status,
                    "comparability": comparability,
                    "unavailable_reason": reason, "notes": note.strip(),
                    "normalization_weight_kg": weight if unit == "mL/min/kg" else None,
                    "prediction_path": rel(prediction_path) if prediction_path else None,
                    "endpoint_path": rel(endpoint_path) if endpoint_path else None,
                    "endpoint_sha256": digest(endpoint_path) if endpoint_path else None,
                    **provenance(case),
                })
    return rows, native_rows


def markdown(cases, methods, summary, split, parameters, curve_rows, reported, native, run):
    lines = ["# ML + PBPK small pilot — 2026-09-28", "",
        f"本轮完成 {summary['attempted_clinical_predictions']} 个临床预测尝试，其中 {summary['successful_native_predictions']} 个通过原生执行。"
        f"参数训练集 {split['train_count']} 个化合物，测试集 {split['test_count']} 个，按分子骨架隔离。",
        "", f"总用时 {summary['elapsed_seconds'] / 60:.2f} 分钟，实际原生求解 "
        f"{sum(j['result'].get('native_evaluations', 0) for j in summary['results'].values())} 次。"
        "本轮只评价缺少目标药物临床PK参数时的构建分支；有可靠独立参数证据时，仍应优先使用证据。",
        "", "模型只从训练集学习 CL、Vss、fu、半衰期和 MRT；临床三药及其骨架从训练和参数测试集排除。"
        "PBPK 输入使用预测 CL/fu 与独立理化资料；映射方法进一步匹配预测 Vss，未使用临床曲线或临床终点拟合。",
        "", "Median direct = 训练集中位数直接输入；RF direct = 随机森林参数直接输入；RF + Vss mapping = 将随机森林的 CL/Vss 映射到 PBPK。",
        "", "**当前判断：尚不扩大临床实验。** 随机森林参数比中位数基线好，但咖啡因曲线与末段浓度仍明显失准；"
        "Vss映射也没有在共同药物上进一步降低整体曲线误差。下一步先改善CL、Vss、fu预测及其到PBPK的转换，"
        "保留本轮固定对照和失败记录。不能把参数映射收敛当作预测准确，也不能用这3个已查看病例反复调参后再称独立验证。",
        "", "## Parameter prediction on held-out scaffolds", ""]
    parameter_rows = []
    for endpoint, result in parameters.items():
        for method in ("training_median", "random_forest"):
            metric = result[method]
            parameter_rows.append([PARAMETER_LABELS.get(endpoint, endpoint), result.get("unit"), method,
                result.get("train_n"), metric.get("n"), metric.get("gmfe"), metric.get("log10_rmse"),
                f"{100 * metric['within_2fold']:.1f}%" if metric.get("within_2fold") is not None else None])
    lines += [table(["Target", "Unit", "Method", "Train n", "Test n", "GMFE", "log10 RMSE", "Within 2-fold"], parameter_rows),
        "", "## Clinical curve evaluation", "",
        "所有曲线均在原始临床采样时间比较。AUC 是共有采样窗口内的梯形积分；Cmax 是采样峰值；Clast 是末次采样值。", "",
        table(["Drug", "Method", "n", "Curve GMFE", "ln-RMSE", "AUC FE", "Sampled Cmax FE", "Clast FE", "Status"],
            [[r["drug"], METHOD_LABELS.get(r["method"], r["method"]), r.get("point_count"),
              number(r.get("gmfe")), number(r.get("ln_rmse")), number(r.get("auc_fold_error")),
              number(r.get("sampled_cmax_fold_error")), number(r.get("c_last_fold_error")), r.get("scoring_status")]
             for r in curve_rows]), ""]
    macro = []
    available_drugs = [{r["drug"] for r in curve_rows
        if r["method"] == method and number(r.get("gmfe")) is not None} for method in methods]
    common_drugs = set.intersection(*available_drugs) if available_drugs else set()
    for method in methods:
        selected = [r for r in curve_rows if r["method"] == method]
        good = [number(r.get("gmfe")) for r in selected if number(r.get("gmfe")) is not None]
        common = [number(r.get("gmfe")) for r in selected if r["drug"] in common_drugs]
        macro.append([METHOD_LABELS.get(method, method), f"{len(good)}/{len(selected)}",
            math.exp(math.fsum(math.log(v) for v in good) / len(good)) if good else None,
            len(common), math.exp(math.fsum(math.log(v) for v in common) / len(common)) if common else None])
    lines += [table(["Method", "Scored / attempted drugs", "GMFE (available drugs)", "Common drugs n", "GMFE (common drugs)"], macro),
        "", "GMFE按药物等权取几何平均。共同子集为：" + (", ".join(sorted(common_drugs)) or "无") + "。"
        "不同方法成功药物数不同时，不能用各自可用病例的平均值直接宣称最好；应同时看共同子集和全部尝试的覆盖率。",
        "", "## Clinical paper-reported endpoint references", "",
        "这里的 O 来自论文直接报告的临床 PK 结果，不是我们对平均曲线重算的 NCA。"
        "本批有10个已核对的参考值：Midazolam 6个，Alfentanil 4个；每种方法均保留全部参考比较。"
        "Caffeine 此表未收录论文标量参考，但其真实曲线已评分。",
        "", f"原文终点比较总计 {len(reported)} 项：{sum(r['status'] == 'evaluated' for r in reported)} 项可评价，"
        f"{sum(r['status'] == 'descriptive_only' for r in reported)} 项仅作描述，"
        f"{sum(r['status'] == 'unavailable' for r in reported)} 项预测不可用。",
        "", "P 为本次预测。Cmax/Tmax 的主比较使用可用临床时间点采样的预测值；CSV 另外保留原生密集网格结果。"
        "其他终点使用原生模型终点。FE = max(P/O, O/P)；Tmax只给绝对时间误差。", ""]
    for case in cases:
        subset = [r for r in reported if r["drug"] == case["drug"]]
        if not subset:
            continue
        lines += [f"### {case['drug'].title()} / {case['study_id']}", "",
            table(["Endpoint", "Unit", "Method", "Prediction P", "Clinical O", "FE", "Tmax AE (h)", "Status"],
                [[ENDPOINTS[r["endpoint"]][0].replace(" (native grid)", ""), r["unit"], METHOD_LABELS.get(r["method"], r["method"]),
                  r["predicted"], r["observed_reported"], r["fold_error"], r["absolute_error_h"], r["status"]] for r in subset]),
            "", "Reference: [clinical publication](" + subset[0]["reference_source_url"] + "). " +
            ("Geometric means; Tmax is a median." if case["drug"] == "midazolam" else "Arithmetic means of individual PK estimates."), ""]
    lines += ["原生终末半衰期来自模型很晚的低浓度尾部，原论文的终末拟合窗口无法匹配。"
        "这6项保留预测值、临床值和原始倍数误差，标记 descriptive_only，不计入临床验证成功数或两倍误差通过率。"
        "CSV 另外保留最后4个临床采样时间对应的半衰期预测；它也不代表已复现原论文个体NCA拟合窗口。", "",
        "## All 11 requested native prediction classes", "",
        "以下是模型输出清单，不代表每项都有临床参考。— 表示未得到通过质量检查的结果；空值未补成0。"
        "有限窗口和末次浓度按原请求定义，可能不同于临床首末采样窗口。曲线为完整数列，链接保留全部数值。", ""]
    for case in cases:
        rows = []
        for endpoint, (label, _, unit) in ENDPOINTS.items():
            values = []
            for method in methods:
                r = next(x for x in native if x["drug"] == case["drug"] and x["method"] == method and x["endpoint"] == endpoint)
                if r["status"] != "completed":
                    values.append("— (" + str(r["unavailable_reason"]) + ")")
                elif endpoint == "plasma_concentration_time":
                    values.append(f"[{r['curve_points']} points]({ROOT / r['prediction_path']})")
                else:
                    values.append(r["native_prediction"])
            rows.append([label, unit, *values])
        lines += [f"### {case['drug'].title()}", "", table(["Endpoint", "Unit", *[METHOD_LABELS.get(m, m) for m in methods]], rows), ""]
    failed = [r for r in summary["results"].values() if r["result"]["status"] != "predicted"]
    lines += ["## Limits and reproducibility", "",
        "半衰期与Vz还存在模型极晚低浓度尾部的问题；输出可计算不等于有生理或临床可信度。"
        "不通过放宽尾部质量门槛或隐藏失败来提高成绩。", "",
        "三种药物均是历史开发资料。本轮验证流程和暴露出的误差，不能证明临床决策能力或 SOTA；未调整测试标签、删除失败或根据临床结果选择参数。"
        "单个代表性人体与群体平均曲线存在人群差异；总清除率集中到肝脏是本轮约定，未验证器官归属。"
        "11类终点未完成逐病例全参考验证。", ""]
    if failed:
        lines += [table(["Drug", "Method", "Failure"], [[r["drug"],r["method"], r["result"].get("reason", r["result"]["status"])] for r in failed]), ""]
    lines += [f"数据来源和划分：[parameter_split.json]({run / 'parameter_split.json'})；"
        f"论文终点逐项评分：[reported_endpoint_comparison.csv]({run / 'reported_endpoint_comparison.csv'})；"
        f"全部预测终点：[native_endpoint_predictions.csv]({run / 'native_endpoint_predictions.csv'})。", "",
        "只重新生成报告，不启动实验：", "", "```bash",
        f"cd {ROOT}", f"runtime/app-env/bin/python -B scripts/report_ml_pbpk_pilot.py --run {run}", "```", ""]
    return "\n".join(lines)


def number(value):
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite metric")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=BASE / "run")
    parser.add_argument("--candidates", type=Path, default=BASE / "inputs/clinical_candidates.json")
    args = parser.parse_args()
    run = scoped(args.run)
    summary = read(run / "summary.json")
    if summary.get("status") not in ("completed", "completed_with_failures"):
        raise ValueError("Report requires a completed run, including retained failures")
    protocol = read(run / "protocol.json")
    expected_hash = protocol["pins"].get(rel(args.candidates))
    if not expected_hash or digest(args.candidates) != expected_hash:
        raise ValueError("Clinical manifest differs from the frozen run")
    cases = read(args.candidates)["cases"]
    for case in cases:
        original = case["request_source"]
        path = scoped(case["request_path"])
        expected = protocol["pins"].get(rel(path))
        if (scoped(original["path"]) != path or not expected
                or digest(path) != expected or digest(path) != original["sha256"]):
            raise ValueError("Original request differs from frozen sources: " + case["drug"])
        reference = provenance(case)
        reference_path = reference["reference_source_path"]
        reference_hash = reference["reference_source_sha256"]
        if not reference_path or not reference_hash or digest(reference_path) != reference_hash:
            raise ValueError("Published clinical source checksum mismatch: " + case["drug"])
        observed = case["observations_path"]
        if digest(observed) != protocol["pins"].get(rel(observed)):
            raise ValueError("Clinical observation file differs from frozen run: " + case["drug"])
    methods = protocol["methods"]
    if len(summary["results"]) != len(cases) * len(methods):
        raise ValueError("Incomplete attempt inventory")
    locks = read(run / "predictions_lock.json")
    if locks.get("all_attempts_completed_before_scoring") is not True:
        raise ValueError("Prediction lock lacks ordering guarantee")
    reported, native = clinical_rows(cases, methods, summary, locks)
    expected = sum(len(c.get("published_endpoint_references", [])) for c in cases) * len(methods)
    if len(reported) != expected or len(native) != len(cases) * len(methods) * 11:
        raise ValueError("Endpoint inventory incomplete")
    write_csv(run / "reported_endpoint_comparison.csv", reported)
    write_csv(run / "native_endpoint_predictions.csv", native)
    with (run / "clinical_comparison.csv").open() as handle:
        curves = list(csv.DictReader(handle))
    text = markdown(cases, methods, summary, read(run / "parameter_split.json"),
        read(run / "parameter_metrics.json"), curves, reported, native, run)
    (run.parent / "README.md").write_text(text)
    final = ROOT / "results/tables/ml_pbpk_pilot_20260928.md"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text(text)
    print(json.dumps({"status": "reported", "paper_reference_comparisons": len(reported),
        "paper_reference_comparisons_evaluated": sum(r["status"] == "evaluated" for r in reported),
        "paper_reference_comparisons_descriptive_only": sum(r["status"] == "descriptive_only" for r in reported),
        "native_endpoint_rows": len(native), "report": str(final)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
