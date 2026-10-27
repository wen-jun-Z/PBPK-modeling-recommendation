"""Merge verified held-out PBPK-Match results into the existing paper displays.

Frozen baseline JSONs and fitted weights are read-only. Repeated execution is
idempotent; this script does not train, infer baselines, or refit PBPK parameters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
DATA = ROOT / "results/data"
LIGHT = ROOT / "experiments/model_recommendation/pbpk_match_light"
OWN = ("pbpk_match_light_document", "pbpk_match_light")
BEGIN = "% BEGIN PBPK-MATCH HELD-OUT"
END = "% END PBPK-MATCH HELD-OUT"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def percent(value):
    return None if value is None else 100 * value


def validate_sources(record):
    for relative, expected in record["source_files"].items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT), relative
        assert digest(path) == expected, f"Stale publication source: {relative}"


def clean_blocks(text):
    while BEGIN in text:
        start = text.index(BEGIN)
        end = text.index(END, start) + len(END)
        if text[end:end + 1] == "\n":
            end += 1
        text = text[:start] + text[end:]
    return text


def append_block(name, lines, *, anchor=r"\bottomrule"):
    path = TABLES / f"{name}.tex"
    text = clean_blocks(path.read_text())
    assert text.count(anchor) == 1, (name, anchor)
    block = BEGIN + "\n" + "\n".join(lines) + "\n" + END + "\n"
    path.write_text(text.replace(anchor, block + anchor))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group(columns):
    return [r"\midrule", rf"\multicolumn{{{columns}}}{{@{{}}l@{{}}}}{{\textit{{Grouped held-out PBPK-Match}}}} \\"]


def row_tex(values):
    return " & ".join(map(str, values)) + r" \\"


def merged_results():
    paths = [DATA / "model_recommendation_quality.json", DATA / "model_recommendation_execution.json",
             DATA / "paper_results.json", DATA / "pbpk_match_light_paper_metrics.json",
             DATA / "pbpk_match_light_curves.json", LIGHT / "engineering/summary.json",
             LIGHT / "report/summary.json"]
    old_quality, old_execution, old_engineering, metrics, curves, engineering, report = map(read, paths)
    assert metrics["state"] == "complete", metrics["state"]
    validate_sources(metrics)
    validate_sources(engineering)
    assert len(set(metrics["test_query_ids"])) == 102
    for row in metrics["quality_rows"]:
        binary = metrics["quality_details"][row["method_id"]]["binary_applicability_102"]
        assert row["test_known_pairs"] == binary["known_pairs"] == 16410
        assert row["test_positive_pairs"] == binary["positive_pairs"] == 74
        assert row["test_positive_f1"] == binary["positive_f1"] and binary["threshold"] == .5
    quality = old_quality["rows"] + metrics["quality_rows"]
    cmax = list(old_execution["main_cmax_table"])
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_model_execution_results import independent_blocks
    common = []
    for method in OWN:
        source = report["reports"][method]["execution"]
        positions = {r["rank"]: r for r in source["position_summary"] if r["endpoint"] == "Cmax_reference_vs_group"}
        omission = next(r for r in source["omission_summary"] if r["endpoint"] == "Cmax_reference_vs_group" and r["k"] == 5)
        label = next(r["method"] for r in metrics["quality_rows"] if r["method_id"] == method)
        row = {"method_id": method, "method": label, "group": "trained", "cmax_requested_scenarios": 34,
               "conditional_within2fold_percent": percent(positions[1]["within_twofold"]),
               "top5_best_measured_miss_percent": percent(omission["best_measured_miss_probability"]),
               "top5_eligible_cases": omission["choice_comparison_scenarios"]}
        for rank, result in positions.items():
            assert result["scenarios"] == 34
            row[f"rank{rank}_conditional_gmfe"] = result["gmfe"]
            row[f"rank{rank}_coverage_percent"] = 100 * result["coverage"]
            row[f"rank{rank}_conditional_within2fold_percent"] = percent(result["within_twofold"])
        cmax.append(row)
        shared = next(r for r in source["common_positions"] if r["endpoint"] == "Cmax_reference_vs_group")
        members = [r for r in source["position_rows"] if r["scenario_id"] in shared["scenario_ids"]
                   and r["endpoint"] == "Cmax_reference_vs_group" and r["rank"] == 1]
        common.append({**shared, "method_id": method, "method": label,
                       "independent_blocks": len(independent_blocks(members))})
    old = old_engineering["model_recommendation"]
    engineering_rows = list(old["rows"])
    for method in OWN:
        stability = next(r for r in engineering["stability"] if r["method_id"] == method)
        runtime = next(r for r in engineering["runtime"] if r["method_id"] == method)
        engineering_rows.append({**stability, **runtime, "group": "trained"})
    generations = old["generation"] + [{**r, "total": 2316, "postprocessed_or_retried_valid": None}
                                       for r in engineering["generation"]]
    assert len(quality) == len(cmax) == len(engineering_rows) == 14
    assert len({r["method_id"] for r in quality}) == 14
    return {"schema": "pbpk-match-light-paper-v1", "quality_rows": quality, "cmax_rows": cmax,
            "own_common_positions": common,
            "engineering": {"rows": engineering_rows, "generation": generations},
            "scalar_endpoint_rows": metrics["scalar_endpoint_rows"],
            "curve_position_rows": curves["position_rows"],
            "source_files": {str(p.relative_to(ROOT)): digest(p) for p in paths},
            "protocol": "12 frozen external configurations plus 2 grouped held-out own configurations; seed 17",
            "baseline_values_unchanged": True, "training_or_baseline_inference_performed": False}


def tables(result):
    quality = result["quality_rows"][-2:]
    lines = group(7)
    for row in quality:
        lines.append(row_tex([row["method"]] + [fmt(row[k]) for k in
                         ("ndcg@10", "recall@10", "mrr", "same_drug_family_query_macro_accuracy", "test_positive_f1")]
                         + [fmt(100 * row["top10_judged_fraction_all_queries"], 1)]))
    append_block("model_recommendation_quality", lines)
    write_csv(TABLES / "model_recommendation_quality.csv", result["quality_rows"])
    lines = group(6)
    for row in result["cmax_rows"][-2:]:
        values = [row["method"]] + [f"{fmt(row[f'rank{k}_conditional_gmfe'])} ({fmt(row[f'rank{k}_coverage_percent'], 1)})"
                                    for k in (1, 2, 3)]
        values += [fmt(row["conditional_within2fold_percent"], 1), fmt(row["top5_best_measured_miss_percent"], 1)]
        lines.append(row_tex(values))
    append_block("model_recommendation_execution", lines)
    write_csv(TABLES / "model_recommendation_execution.csv", result["cmax_rows"])
    if (TABLES / "model_recommendation_execution_common_positions.tex").exists():
        lines = group(6)
        for r in result["own_common_positions"]:
            lines.append(row_tex([r["method"], r["scenarios"], r["independent_blocks"]] +
                                 [fmt(r[f"rank{k}_gmfe"]) for k in (1, 2, 3)]))
        append_block("model_recommendation_execution_common_positions", lines)
    rows = result["engineering"]["rows"]
    lines = group(8)
    for row in rows[-2:]:
        values = [row["method"], fmt(100 * row["tie_fraction"], 1)]
        for prefix in ("reorder", "missing"):
            values += [fmt(row[f"{prefix}_top1"]), fmt(row[f"{prefix}_top10"]), row[f"{prefix}_n"]]
        lines.append(row_tex(values))
    append_block("model_recommendation_stability", lines)
    write_csv(TABLES / "model_recommendation_stability.csv", rows)
    lines = group(4) + [row_tex([r["method"], fmt(r["p50_seconds"]), fmt(r["p95_seconds"]), fmt(r["peak_gpu_gib"], 2)]) for r in rows[-2:]]
    anchor = r"\midrule" + "\n" + r"\multicolumn{4}{@{}l@{}}{\textit{Recorded batch throughput (not single-query latency)}} \\"
    append_block("model_recommendation_runtime", lines, anchor=anchor)
    # Two insert locations share a marker; preserve the just-written first block.
    path = TABLES / "model_recommendation_runtime.tex"
    batch_lines = [row_tex([r["method"], r["batch_size"], r["batch_queries"], fmt(r["batch_queries_per_second"], 2)]) for r in rows[-2:]]
    text = path.read_text().replace(r"\bottomrule", BEGIN + "\n" + "\n".join(batch_lines) + "\n" + END + "\n" + r"\bottomrule")
    path.write_text(text)
    write_csv(TABLES / "model_recommendation_runtime.csv", rows)
    lines = group(5) + [row_tex([r["method"], 2316, r"\multicolumn{3}{c}{N/A (non-generative)}"]) for r in rows[-2:]]
    append_block("model_recommendation_generation", lines)
    write_csv(TABLES / "model_recommendation_generation.csv", result["engineering"]["generation"])
    endpoint_table(result["scalar_endpoint_rows"])


def endpoint_table(rows):
    labels = {"Cmax_ss_reference_vs_group": r"Steady-state $C_{\max}$", "Cmax_normalized_by_dose_per_weight_reference_vs_group": r"Normalized $C_{\max}$",
              "AUC_0_24h_reference_vs_group": r"AUC$_{0-24\,h}$", "AUC_0_72h_reference_vs_group": r"AUC$_{0-72\,h}$",
              "AUC_tau_12h_reference_vs_group": r"AUC$_{\tau=12\,h}$", "AUC_tau_24h_reference_vs_group": r"AUC$_{\tau=24\,h}$"}
    endpoints = sorted({r["endpoint"] for r in rows} - {"Cmax_reference_vs_group"})
    lines = [r"\begingroup", r"\fontsize{8.5}{10.5}\selectfont", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\toprule",
             row_tex(["Endpoint", "$n$", r"\shortstack{Rank 1 GMFE\\(Cov.\%)}", r"\shortstack{Rank 2 GMFE\\(Cov.\%)}",
                      r"\shortstack{Rank 3 GMFE\\(Cov.\%)}", r"\shortstack{Rank 1\\$\leq 2$-fold (\%)}"]), r"\midrule"]
    for endpoint in endpoints:
        by_method = [{r["rank"]: r for r in rows if r["endpoint"] == endpoint and r["method"] == method} for method in OWN]
        assert set(by_method[0]) == {1, 2, 3}
        for rank in (1, 2, 3):
            for key in ("gmfe", "coverage", "within_twofold", "scenarios"):
                assert by_method[0][rank][key] == by_method[1][rank][key], (endpoint, key)
        ranks = by_method[0]
        cells = [labels[endpoint], ranks[1]["scenarios"]]
        cells += [f"{fmt(ranks[k]['gmfe'])} ({fmt(100 * ranks[k]['coverage'], 1)})" for k in (1, 2, 3)]
        cells += [fmt(None if ranks[1]["within_twofold"] is None else 100 * ranks[1]["within_twofold"], 1)]
        lines.append(row_tex(cells))
    lines += [r"\bottomrule", r"\end{tabular*}", r"\endgroup", ""]
    (TABLES / "model_recommendation_scalar_endpoints.tex").write_text("\n".join(lines))
    write_csv(TABLES / "model_recommendation_scalar_endpoints.csv", rows)


def figures(result):
    from build_paper_results import plotting_environment, model_stability_matrix, model_operations_figure, figure_audit, export_outlined_pdf
    from PIL import Image
    plt = plotting_environment()
    destination = ROOT / "results/figures/experiments"
    audits = {}
    for name, function in (("model_recommendation_stability", model_stability_matrix), ("model_recommendation_operations", model_operations_figure)):
        fig = function(plt, result["engineering"])
        audits[name] = figure_audit(fig)
        png, svg = destination / f"{name}.png", destination / f"{name}.svg"
        fig.savefig(png, dpi=300, facecolor="white")
        fig.savefig(svg, metadata={"Date": None})
        audits[name].update(export_outlined_pdf(svg, png, fig.get_size_inches()))
        gray = ROOT / "runtime/cache/paper_figures" / f"{name}_grayscale.png"
        with Image.open(png) as source:
            source.convert("L").save(gray)
        plt.close(fig)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    result = merged_results()
    tables(result)
    from build_recommendation_endpoint_table import publish as publish_endpoint_table
    publish_endpoint_table(root=ROOT)
    endpoint_path = DATA / "model_recommendation_endpoint_accuracy.json"
    result["source_files"][str(endpoint_path.relative_to(ROOT))] = digest(endpoint_path)
    extended_path = DATA / "model_recommendation_extended_metrics.json"
    if extended_path.exists():
        from build_recommendation_endpoint_table import publish_extension
        result["eleven_endpoint_extension"] = publish_extension(root=ROOT)
        for name in ("model_recommendation_extended_references.json",
                     "model_recommendation_extended_predictions.json", "model_recommendation_extended_metrics.json",
                     "model_recommendation_endpoint_overview.json"):
            path = DATA / name
            result["source_files"][str(path.relative_to(ROOT))] = digest(path)
    if args.figures:
        result["figure_audit"] = figures(result)
    else:
        previous = DATA / "pbpk_match_light_publication.json"
        if previous.exists():
            result["figure_audit"] = read(previous).get("figure_audit", {})
    for relative, expected in result["source_files"].items():
        assert digest(ROOT / relative) == expected, relative
    (DATA / "pbpk_match_light_publication.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Merged 12 frozen configurations and 2 held-out own configurations; source hashes unchanged.")


if __name__ == "__main__":
    main()
