#!/usr/bin/env python3
"""Publish paired-input results from completed, source-bound reports.

No inference, fitting, candidate execution or changes to main.tex. The optional
main-table view adds the component variant and removes redundant/inapplicable
display rows; all original scientific artifacts remain unchanged. Identity
controls can only publish in runtime/cache and never to main paper tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from pbpk_kb.recommendation.pbpk_match_v2.evaluation import hierarchical_mean
SCHEMA = "pbpk-component-integration-supplement-v1"
REPORT_SCHEMA = "pbpk-component-model-integration-evaluation-v1"
CONTROL = "zero_effect_replay_control_not_experiment"
BEFORE = "pbpk_match_light"
AFTER = "pbpk_match_components"
C_MAX = "Cmax_reference_vs_group"
METHODS = [(BEFORE + "_document", "Original", "Document"),
           (BEFORE, "Original", "Final"),
           (AFTER + "_document", "+ Components", "Document"),
           (AFTER, "+ Components", "Final")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scope(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(ROOT), "Publication must remain in this project")
    return path


def sha(path):
    digest = hashlib.sha256()
    with scope(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path):
    path = scope(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def read(path):
    def nonfinite(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(scope(path).read_text(), parse_constant=nonfinite)


def unique(rows, key, description):
    values = {}
    for row in rows:
        identity = tuple(row[k] for k in key)
        require(identity not in values, "Duplicate " + description)
        values[identity] = row
    return values


def verify_ref(value):
    require(isinstance(value, dict) and {"path", "sha256"} <= value.keys(), "Missing artifact binding")
    path = scope(value["path"])
    require(path.is_file() and sha(path) == value["sha256"], "Changed artifact: " + str(path))


def validate_bridge(bridge):
    require(bridge.get("state") == "complete" and bridge.get("requests") == 186, "Bridge must cover all 186 requests")
    require(bridge.get("profile_eligibility_used_for_ranking") is False,
            "Bridge profile eligibility must not filter ranked candidates")
    require(bridge.get("invalid_component_results_use_unchanged_input") is True,
            "Failed component requests must retain their original input")
    statuses, cohorts = bridge.get("statuses", {}), bridge.get("by_cohort", {})
    require(statuses and all(type(v) is int and v >= 0 for v in statuses.values()) and sum(statuses.values()) == 186,
            "Invalid complete bridge status counts")
    combined = {}
    for name, count in (("core", 145), ("execution", 41)):
        values = cohorts.get(name, {})
        require(values and all(type(v) is int and v >= 0 for v in values.values()) and sum(values.values()) == count,
                "Bridge cohort coverage changed")
        for status, n in values.items():
            combined[status] = combined.get(status, 0) + n
    require(combined == statuses, "Bridge cohort and full counts disagree")
    return {"total": 186, "statuses": statuses, "by_cohort": cohorts,
            "augmented": statuses.get("augmented", 0), "unchanged": 186 - statuses.get("augmented", 0),
            "core_augmented": cohorts["core"].get("augmented", 0),
            "execution_augmented": cohorts["execution"].get("augmented", 0)}


def extract(report, bridge):
    require(report.get("schema") == REPORT_SCHEMA and report.get("state") == "complete", "Incomplete or incompatible paired report")
    require(report.get("scope", {}).get("document_requests") == 145
            and report["scope"].get("execution_requests") == 41
            and report["scope"].get("candidates") == 280, "Paired report scope differs")
    require(set(report["reports"]) == {m[0] for m in METHODS}, "Expected exactly original/augmented document/final scores")
    require(report.get("validation", {}).get("baseline_metrics_reproduced") is True, "Original baseline was not reproduced")
    bridge_counts = validate_bridge(bridge)
    doc, cmax = [], []
    for method, label, score in METHODS:
        q = report["reports"][method]["quality"]
        require(q["judged_only"]["eligible_queries"] == 40 and q["coverage"]["queries"] == 145,
                "Document evaluation denominator changed")
        app = q["paper_binary_102"]
        require(app["known_pairs"] == 16410 and app["positive_pairs"] == 74, "Binary F1 denominator changed")
        preference = q["same_drug_different_family_preference"]
        require(preference["pairs"] == 43 and preference["eligible_queries"] == 11, "Same-drug pair scope changed")
        doc.append({"method_id": method, "input": label, "score": score,
                    **{k: q["judged_only"]["metrics"][k] for k in ("ndcg@10", "recall@10", "mrr")},
                    "same_drug_accuracy": preference["query_macro_accuracy"], "positive_f1": app["positive_f1"],
                    "known_top10_fraction": q["known_top10_coverage"]["mean"]})
        rows = unique([r for r in report["reports"][method]["execution"]["position_summary"]
                       if r["endpoint"] == C_MAX], ("rank",), "Cmax rank")
        require(set(rows) == {(1,), (2,), (3,)}, "Cmax must retain all three exact ranks")
        for rank in (1, 2, 3):
            r = rows[rank,]
            require(r["scenarios"] == 34 and 0 <= r["coverage"] <= 1, "Ordinary Cmax denominator changed")
            require((r["coverage"] > 0 and r["gmfe"] is not None and r["gmfe"] >= 1)
                    or (r["coverage"] == 0 and r["gmfe"] is None), "Cmax accuracy/coverage contradiction")
            cmax.append({"method_id": method, "input": label, "score": score, "rank": rank,
                         "gmfe": r["gmfe"], "coverage": r["coverage"], "denominator": 34,
                         "covered_scenarios": r["covered_scenarios"]})
    common = []
    rows = unique([r for r in report["paired"]["common_positions"] if r["endpoint"] == C_MAX],
                  ("method", "comparator", "rank"), "paired Cmax comparison")
    for suffix, label in (("_document", "Document"), ("", "Final")):
        for rank in (1, 2, 3):
            require((AFTER + suffix, BEFORE + suffix, rank) in rows, "Missing common-position comparison")
            r = rows[AFTER + suffix, BEFORE + suffix, rank]
            require(r["shared_scenarios"] == len(set(r["scenario_ids"])), "Common scenario IDs and count differ")
            common.append({"score": label, "rank": rank, "scenarios": r["shared_scenarios"],
                           "scenario_ids": r["scenario_ids"], "original_gmfe": r["comparator_gmfe"],
                           "component_gmfe": r["own_gmfe"]})
    primary = unique(report["paired"]["primary_family"], ("comparison",), "primary comparison")
    require(set(primary) == {("document_ndcg@10",), ("ordinary_Cmax_rank1",)}, "Primary test family changed")
    primary_rows = []
    for key in ("document_ndcg@10", "ordinary_Cmax_rank1"):
        row = primary[key,]
        require(row["family_size"] == 2, "Primary Holm family must contain two tests")
        require(row["higher_is_better"] is (key == "document_ndcg@10"), "Primary improvement direction changed")
        require(row["p_holm"] is None or (0 <= row["p_raw"] <= row["p_holm"] <= 1), "Invalid corrected p-value")
        primary_rows.append(dict(row))
    curve = report["curves"]
    require(curve["independent_publications"] == 1 and curve["qualified_observations"] == 43, "Curve case scope changed")
    curve_rows = curve["position_rows"]
    require(len({r["scenario_id"] for r in curve_rows}) == 2 and len(curve_rows) == 24,
            "Curve table must retain two cases and three ranks for every method")
    return {"schema": SCHEMA, "state": "complete", "purpose": report.get("purpose"),
        "control_replay": report.get("purpose") == CONTROL, "bridge": bridge_counts,
        "document": doc, "ordinary_cmax": cmax, "common_cmax": common, "primary_comparisons": primary_rows,
        "final_omissions": final_omission_rows(report),
        "curve_cases": [{k: r[k] for k in ("method_id", "scenario_id", "position", "curve_gmfe", "curve_coverage")}
                        for r in curve_rows],
        "curve_display": curve_display_rows(curve),
        "interpretation": {"information_change": report["input_policy"]["information_change"],
            "frozen_weights_and_calibration": True, "internal_validation": True,
            "curves": "Two cases and 43 observations from one study; descriptive only.",
            "common_cmax": "Both inputs have full evaluable tie probability at the same exact position.",
            "full_endpoint_results": "All existing endpoint definitions remain in the associated report CSV/JSON.",
            "outcome_claim": "Report document and prediction differences separately; no combined winner score."}}


def final_omission_rows(report):
    """Audit final-score omissions on the same measured candidate pools.

    Dose branches are averaged within study and compound, not counted as
    independent experiments. Unmeasured candidates never become negatives.
    """
    output, signatures = [], {}
    for method, label in ((BEFORE, "PBPK-Match"), (AFTER, "+ Components")):
        execution = report["reports"][method]["execution"]
        all_rows = [r for r in execution["omissions"] if r["endpoint"] == C_MAX]
        unique(all_rows, ("scenario_id",), "ordinary Cmax omission scenario")
        rows = [r for r in all_rows if r["choice_comparison_possible"]]
        compounds = {r["compound"] for r in rows}
        studies = {(r["compound"], r["publication_cluster_id"]) for r in rows}
        require((len(all_rows), len(rows), len(compounds), len(studies)) == (34, 19, 17, 17),
                "Omission scenario/compound/study denominator changed")
        signatures[method] = {
            r["scenario_id"]: (r["compound"], r["publication_cluster_id"], r["measured_count"],
                               tuple(sorted(r["best_measured_candidate_ids"]))) for r in rows}
        summaries = unique([r for r in execution["omission_summary"] if r["endpoint"] == C_MAX],
                           ("k",), "ordinary Cmax omission summary")
        require(set(summaries) == {(3,), (5,)}, "Omission summary must retain Top-3 and Top-5")
        for k in (3, 5):
            summary = summaries[k,]
            require((summary["total_scenarios"], summary["choice_comparison_scenarios"],
                     summary["compounds"], summary["compound_study_pairs"]) == (34, 19, 17, 17),
                    "Omission summary denominator disagrees with individual scenarios")
            weighted = []
            for row in rows:
                require(row["measured_count"] >= 2 and row["best_measured_candidate_ids"],
                        "Omission comparison requires at least two measured candidates")
                rates = row["by_k"][str(k)]
                miss, hit = rates["best_measured_miss_probability"], rates["best_measured_hit_probability"]
                require(miss is not None and hit is not None and 0 <= miss <= 1 and 0 <= hit <= 1
                        and math.isclose(miss + hit, 1., abs_tol=1e-12), "Invalid omission probability")
                weighted.append({"compound": row["compound"], "publication_cluster_id": row["publication_cluster_id"],
                                 "miss": miss})
            miss = hierarchical_mean(weighted, "miss")
            require(math.isclose(summary["best_measured_miss_probability"], miss, abs_tol=1e-12, rel_tol=0)
                    and math.isclose(summary["best_measured_hit_probability"], 1 - miss, abs_tol=1e-12, rel_tol=0),
                    "Omission summary differs from hierarchical scenario aggregation")
            output.append({"method_id": method, "input": label, "k": k, "endpoint": C_MAX,
                           "scenarios": len(rows), "compounds": len(compounds), "compound_study_pairs": len(studies),
                           "miss_probability": miss, "scenario_ids": sorted(signatures[method]),
                           "aggregation": summary["aggregation"]})
    require(signatures[BEFORE] == signatures[AFTER], "Before/after omission comparison pools differ")
    return output


def curve_display_rows(curve):
    """Merge scores only when exact candidate identities and tie weights agree.

    Equal rounded GMFE, or two missing predictions, cannot establish equal
    rankings. Unmeasured members are included in the identity comparison.
    """
    positions = unique(curve["position_rows"], ("method_id", "scenario_id", "position"), "curve position")
    rankings = unique(curve.get("rankings", []), ("method_id", "scenario_id"), "curve ranking")
    output = []
    for sid in sorted({r["scenario_id"] for r in curve["position_rows"]}):
        compound = next(r.get("compound") for r in curve["position_rows"] if r["scenario_id"] == sid)
        require(isinstance(compound, str) and re.fullmatch(r"[A-Za-z-]+", compound), "Missing safe curve compound label")
        for base, label in ((BEFORE, "Original"), (AFTER, "+ Components")):
            def signature(method):
                item = rankings.get((method, sid))
                if item is None:
                    return None
                values = unique(item["positions"], ("position",), "curve ranking position")
                require(set(values) == {(1,), (2,), (3,)}, "Curve ranking must retain all three positions")
                return tuple((rank, tuple(sorted((m["candidate_id"], m["weight"])
                    for m in values[rank,]["members"]))) for rank in (1, 2, 3))
            document, final = signature(base + "_document"), signature(base)
            merged = document is not None and document == final
            methods = [(base, "Both", [base + "_document", base])] if merged else [
                (base + "_document", "Document", [base + "_document"]), (base, "Final", [base])]
            for method, score, ids in methods:
                values = [{k: positions[method, sid, rank][k] for k in ("position", "curve_gmfe", "curve_coverage")}
                          for rank in (1, 2, 3)]
                if merged:
                    require(values == [{k: positions[base + "_document", sid, rank][k]
                        for k in ("position", "curve_gmfe", "curve_coverage")} for rank in (1, 2, 3)],
                        "Identical curve ranking has inconsistent accuracy or coverage")
                output.append({"scenario_id": sid, "compound": compound, "input": label, "score": score,
                               "method_ids": ids, "positions": values})
    return output


def subset_metrics(report, changed_ids, cohort_by_id):
    """Descriptive only; membership is fixed by input availability, not errors."""
    changed_ids = set(changed_ids)
    require(changed_ids <= set(cohort_by_id), "Unknown request in changed-input mask")
    documents, execution = [], []
    for method, label, score in METHODS:
        quality = report["reports"][method]["quality"]
        per_query = quality["per_query"]
        known = dict(zip([q["query_id"] for q in per_query], quality["known_top10_coverage"]["per_query"], strict=True))
        selected = [q for q in per_query if q["query_id"] in changed_ids]
        eligible = [q["judged_only"] for q in selected if q["judged_only"] is not None]
        pairs = [q["same_drug_different_family_preference"]["accuracy"] for q in selected
                 if q["same_drug_different_family_preference"]["pairs"]]
        documents.append({"method_id": method, "input": label, "score": score,
            "changed_requests": len(selected), "ranking_eligible_requests": len(eligible),
            **{k: math.fsum(q[k] for q in eligible) / len(eligible) if eligible else None
               for k in ("ndcg@10", "recall@10", "mrr")},
            "same_drug_accuracy": math.fsum(pairs) / len(pairs) if pairs else None,
            "same_drug_eligible_requests": len(pairs),
            "known_top10_fraction": math.fsum(known[q["query_id"]] for q in selected) / len(selected) if selected else None})
        rows = report["reports"][method]["execution"]["position_rows"]
        for endpoint in sorted({r["endpoint"] for r in rows if r["endpoint"] == C_MAX}):
            for rank in (1, 2, 3):
                selected = [r for r in rows if r["scenario_id"] in changed_ids and r["endpoint"] == endpoint and r["rank"] == rank]
                error = hierarchical_mean(selected, "absolute_log_error")
                execution.append({"method_id": method, "endpoint": endpoint, "rank": rank,
                    "eligible_subset_scenarios": len(selected), "covered_scenarios": sum(r["coverage"] > 0 for r in selected),
                    "coverage": math.fsum(r["coverage"] for r in selected) / len(selected) if selected else None,
                    "gmfe": math.exp(error) if error is not None else None,
                    "within_twofold": hierarchical_mean(selected, "within_twofold")})
    return {"role": "descriptive only; no subset hypothesis tests or p-values",
            "membership": "Predictions NPZ changed flag, fixed before recommendation outcomes",
            "execution_scope": "Ordinary Cmax summary; full all-request endpoint results remain in the evaluation CSV/JSON",
            "query_ids": sorted(changed_ids), "total": len(changed_ids),
            "by_cohort": {c: sum(cohort_by_id[q] == c for q in changed_ids) for c in ("core", "execution")},
            "document": documents, "execution": execution}


def actual_input_coverage(report, bridge, run):
    candidates = [run / "predictions/manifest.json", run / "manifest.json"]
    bound = {r["path"]: r["sha256"] for r in report["sources"]}
    manifests = [p for p in candidates if p.is_file() and str(p.relative_to(ROOT)) in bound]
    require(len(manifests) == 1, "Need one report-bound prediction manifest for actual changed-input coverage")
    manifest_path = manifests[0]
    verify_ref({"path": str(manifest_path.relative_to(ROOT)), "sha256": bound[str(manifest_path.relative_to(ROOT))]})
    manifest = read(manifest_path)
    arrays = [r for r in report["sources"] if r["path"].endswith(".npz") and r["sha256"] == manifest["predictions_sha256"]]
    require(len(arrays) == 1, "Prediction NPZ is not uniquely bound to the report")
    verify_ref(arrays[0])
    verify_ref(bridge["inputs"])
    ledger = [json.loads(line) for line in scope(bridge["inputs"]["path"]).read_text().splitlines() if line.strip()]
    by_id = {r["query_id"]: r for r in ledger}
    ids = manifest["query_ids"]
    require(len(ledger) == len(by_id) == len(ids) == 186 and set(ids) == set(by_id), "Bridge ledger and ranked requests differ")
    with np.load(scope(arrays[0]["path"]), allow_pickle=False) as saved:
        indices = saved["indices"]
        changed = saved["changed"] if "changed" in saved.files else None
    if changed is None:
        require(report.get("purpose") == CONTROL, "Formal result requires its actual changed-input mask")
        changed = np.zeros(186, dtype=bool)
    require(changed.shape == (186,) and changed.dtype.kind == "b", "Changed-input mask must contain 186 booleans")
    require(indices.tolist() == [by_id[q]["global_index"] for q in ids], "Bridge global indices and NPZ rows differ")
    changed_ids = [q for q, flag in zip(ids, changed) if flag]
    require(all(by_id[q]["status"] == "augmented" for q in changed_ids), "Changed input lacks validated soft additions")
    if "coverage" in manifest:
        require(manifest["coverage"]["changed_requests"] == len(changed_ids)
                and manifest["coverage"]["unchanged_requests"] == 186 - len(changed_ids), "Manifest changed-input counts disagree")
    cohorts = {qid: row["cohort"] for qid, row in by_id.items()}
    result = subset_metrics(report, changed_ids, cohorts)
    result["fallback_requests"] = sum(row.get("audit", {}).get("fallback") is True for row in ledger)
    result["validated_no_soft_addition"] = sum(row["status"] == "validated_no_soft_addition" for row in ledger)
    result["unchanged_requests"] = 186 - len(changed_ids)
    result["sources"] = [ref(manifest_path), arrays[0], bridge["inputs"]]
    return result


def number(value, digits=3):
    if value is None:
        return "NA"
    require(isinstance(value, (int, float)) and math.isfinite(value), "Cannot print nonfinite metric")
    return f"{value:.{digits}f}"


def interval(row):
    point, ci = row["estimate"], row["bootstrap_95ci"]
    return number(point) + (" [" + ", ".join(number(x) for x in ci) + "]" if ci is not None else " [NA]")


def pvalue(value):
    return "$<0.001$" if value is not None and 0 <= value < .001 else number(value)


def latex(summary):
    control = summary["control_replay"]
    caption = ("Evaluator identity-replay control (not experimental evidence). " if control else "") + \
              "Component-input effects under frozen model weights."
    lines = [r"\begin{table}[!htbp]", r"\captionsetup{name=Table}", r"\caption{" + caption + "}",
        r"\label{tab:component_model_integration}", r"\centering", r"\begingroup",
        r"\fontsize{9}{11}\selectfont", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrr@{}}", r"\toprule",
        r"\multicolumn{4}{@{}l}{\textbf{A. Prespecified paired effects}} \\",
        r"Metric & Difference [95\% CI] & Groups & Holm $p$ \\", r"\midrule"]
    for row in summary["primary_comparisons"]:
        label = r"NDCG@10 ($\uparrow$)" if row["comparison"] == "document_ndcg@10" else r"Rank-1 $C_{\max}$ log error ($\downarrow$)"
        lines.append(" & ".join([label, interval(row), str(row["independent_blocks"]), pvalue(row["p_holm"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\vspace{5pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}", r"\toprule",
        r"\multicolumn{4}{@{}l}{\textbf{B. Best-measured candidate omissions ($C_{\max}$)}} \\",
        r"Metric & PBPK-Match & + Components & Change (pp) \\", r"\midrule"]
    omissions = unique(summary["final_omissions"], ("method_id", "k"), "display omission")
    for k in (3, 5):
        before, after = (omissions[method, k]["miss_probability"] for method in (BEFORE, AFTER))
        delta = 100 * (after - before)
        lines.append(" & ".join([f"Top-{k} miss (\\%)", number(100 * before, 2), number(100 * after, 2),
                                 f"{delta:+.2f}" if delta else "0.00"]) + r" \\")
    notes = ("Differences are augmented minus original input; positive NDCG and negative absolute-log-error changes favor augmentation. "
        "CI: grouped bootstrap; Holm correction: two tests. "
        "Miss rates use 19 scenarios with at least two measured candidates, weighted equally by compound after study averaging. "
        "pp: percentage points.")
    if control:
        notes = "IDENTITY CONTROL ONLY: inputs and scores are intentionally identical; these values are not experimental evidence. " + notes
    lines += [r"\bottomrule", r"\end{tabular*}", r"\vspace{4pt}",
              r"\begin{minipage}{\linewidth}\fontsize{8}{10}\selectfont", notes,
              r"\end{minipage}", r"\endgroup", r"\end{table}"]
    return "\n".join(lines) + "\n"


def publish(report_path, bridge_path, output=None):
    report_path = scope(report_path)
    if report_path.is_dir():
        report_path /= "summary.json"
    bridge_path = scope(bridge_path)
    report, bridge = read(report_path), read(bridge_path)
    summary = extract(report, bridge)
    run = report_path.parent.parent if report_path.parent.name == "report" else report_path.parent
    output = scope(output) if output is not None else run / "paper"
    require(output.is_relative_to(run), "Supplement must remain under this run")
    if summary["control_replay"]:
        require(output.is_relative_to(ROOT / "runtime/cache"), "Identity controls cannot publish to an experiment or paper directory")
    else:
        require(ref(bridge_path) in report.get("sources", []), "Report is not bound to this bridge manifest")
    for value in report.get("sources", []):
        verify_ref(value)
    for value in [bridge["inputs"], *bridge.get("sources", [])]:
        verify_ref(value)
    summary["actual_input_subset"] = actual_input_coverage(report, bridge, run)
    completion = report_path.parent / "completion.json"
    require(completion.is_file(), "Completed report receipt is required")
    receipt = read(completion)
    require(receipt.get("state") == "complete" and ref(report_path) in receipt.get("outputs", []), "Summary is not bound to completion")
    summary["sources"] = [ref(report_path), ref(bridge_path), ref(completion), ref(__file__)]
    summary["csv_directory"] = str(report_path.parent.relative_to(ROOT))
    table = latex(summary)
    if output.exists() and any(output.iterdir()):
        existing = output / "component_integration_summary.json"
        require(existing.is_file() and read(existing).get("schema") == SCHEMA, "Refusing unrelated publication directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "component_integration.tex").write_text(table)
    (output / "component_integration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return summary, output


def main_quality_rows(summary, baseline, own_metrics):
    """Same evaluation columns; the component intervention remains a separate row."""
    require(summary.get("state") == "complete" and not summary.get("control_replay"),
            "Only completed experiments may enter main paper tables")
    original = baseline["rows"]
    require(len(original) == 12 and len({r["method_id"] for r in original}) == 12,
            "Expected 12 frozen external configurations")
    own = unique(own_metrics["quality_rows"], ("method_id",), "original own quality")[(BEFORE,)]
    paired = unique(summary["document"], ("method_id",), "paired document quality")
    columns = {"ndcg@10": "ndcg@10", "recall@10": "recall@10", "mrr": "mrr",
               "same_drug_family_query_macro_accuracy": "same_drug_accuracy",
               "test_positive_f1": "positive_f1",
               "top10_judged_fraction_all_queries": "known_top10_fraction"}
    for target, source in columns.items():
        require(math.isclose(own[target], paired[BEFORE,][source], abs_tol=1e-12, rel_tol=0),
                "Paired evaluation does not reproduce original quality: " + target)
    augmented = {"method_id": AFTER, "method": "PBPK-Match + components",
                 "group": "frozen_heads_component_input",
                 "ranking_eligible_queries": 40, "same_drug_family_queries": 11,
                 "same_drug_family_pairs": 43, "test_known_pairs": 16410,
                 "test_positive_pairs": 74,
                 **{target: paired[AFTER,][source] for target, source in columns.items()}}
    rows = [*original, dict(own), augmented]
    require(len({r["method_id"] for r in rows}) == 14, "Duplicate main-table method")
    return rows


def publish_main_tables(summary, *, table_directory=None):
    """Publication-only overlay; never replace frozen metric/score JSON files."""
    import publish_pbpk_match_light as base

    require(not summary.get("control_replay"), "Controls cannot publish main tables")
    for value in summary["sources"]:
        verify_ref(value)
    directory = scope(table_directory) if table_directory is not None else ROOT / "results/tables"
    quality_path = ROOT / "results/data/model_recommendation_quality.json"
    own_path = ROOT / "results/data/pbpk_match_light_paper_metrics.json"
    operations_path = ROOT / "results/data/paper_results.json"
    baseline, own, operations = map(read, (quality_path, own_path, operations_path))
    base.validate_sources(own)
    rows = main_quality_rows(summary, baseline, own)
    quality_tex = directory / "model_recommendation_quality.tex"
    text = base.clean_blocks(quality_tex.read_text())
    # Check that overlaying our rows cannot silently alter the old twelve rows.
    def quality_line(row):
        return base.row_tex([row["method"]] + [base.fmt(row[k]) for k in
            ("ndcg@10", "recall@10", "mrr", "same_drug_family_query_macro_accuracy", "test_positive_f1")]
            + [base.fmt(100 * row["top10_judged_fraction_all_queries"], 1)])
    for row in rows[:12]:
        require(quality_line(row) in text, "Frozen external table value changed: " + row["method_id"])
    require(text.count(r"\bottomrule") == 1, "Unexpected quality table structure")
    block = "\n".join([base.BEGIN, r"\midrule",
        r"\multicolumn{7}{@{}l@{}}{\textit{Supervised PBPK-Match and component-input intervention}} \\",
        *(quality_line(r) for r in rows[12:]), base.END]) + "\n"
    quality_tex.write_text(text.replace(r"\bottomrule", block + r"\bottomrule"))
    base.write_csv(directory / "model_recommendation_quality.csv", rows)

    generation_tex = directory / "model_recommendation_generation.tex"
    generation_tex.write_text(base.clean_blocks(generation_tex.read_text()))
    # Numeric rankers have no text-generation outcomes. Keep their completion
    # records in the engineering JSON and manuscript, not fabricated zero counts.
    generation = operations["model_recommendation"]["generation"]
    require(len(generation) == 3, "Unexpected frozen generation methods")
    base.write_csv(directory / "model_recommendation_generation.csv", generation)
    return {"quality_rows": rows, "generation_rows": generation,
            "source_files": [ref(p) for p in (quality_path, own_path, operations_path)],
            "outputs": [ref(directory / f"model_recommendation_{name}.{suffix}")
                        for name in ("quality", "generation") for suffix in ("tex", "csv")],
            "omitted_display_rows": {
                "quality": "Document ablation retained in raw artifacts and endpoint/stability analyses",
                "generation": "Numeric rankers are outside the text-generation outcome denominator"},
            "baseline_values_unchanged": True,
            "training_or_inference_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Completed summary JSON or its report directory")
    parser.add_argument("--bridge-manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to this run's paper directory")
    parser.add_argument("--main-tables", action="store_true",
                        help="Also update main quality/generation displays from saved results")
    args = parser.parse_args()
    summary, output = publish(args.report, args.bridge_manifest, args.output)
    if args.main_tables:
        main_tables = publish_main_tables(summary)
        from publish_component_endpoint_table import publish as publish_endpoints
        main_tables["endpoints"] = publish_endpoints(ROOT)
        main_tables["outputs"] += [ref(ROOT / relative) for relative in (
            "results/data/component_model_recommendation_endpoint_overview.json",
            "results/tables/model_recommendation_eleven_endpoints.tex",
            "results/tables/model_recommendation_eleven_endpoints.csv",
            "results/tables/model_recommendation_eleven_endpoints_all_methods.tex")]
        main_tables["sources"] = [ref(output / "component_integration_summary.json"), ref(__file__)]
        (ROOT / "results/data/component_model_main_publication.json").write_text(
            json.dumps(main_tables, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": "complete", "control_replay": summary["control_replay"],
                      "output": str(output), "bridge": summary["bridge"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
