#!/usr/bin/env python3
"""Publish all measured scalar endpoints without changing saved statistics.

The 12 completed comparators and two own-model configurations retain their
original rank, conditional-error and all-request coverage definitions. This
exporter does not run inference, fit parameters or treat missing values as zero.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWN = ("pbpk_match_light_document", "pbpk_match_light")
OWN_NAMES = ("PBPK-Match (document)", "PBPK-Match")
ENDPOINTS = (
    ("Cmax_reference_vs_group", r"$C_{\max}$"),
    ("Cmax_ss_reference_vs_group", r"Steady-state $C_{\max}$"),
    ("Cmax_normalized_by_dose_per_weight_reference_vs_group", r"Dose/body-weight-normalized $C_{\max}$"),
    ("AUC_0_24h_reference_vs_group", r"AUC$_{0\text{--}24\,\mathrm{h}}$"),
    ("AUC_0_72h_reference_vs_group", r"AUC$_{0\text{--}72\,\mathrm{h}}$"),
    ("AUC_tau_12h_reference_vs_group", r"AUC$_{\tau=12\,\mathrm{h}}$"),
    ("AUC_tau_24h_reference_vs_group", r"AUC$_{\tau=24\,\mathrm{h}}$"),
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(left, right, context):
    if left is None or right is None:
        if left is not right:
            raise ValueError(f"Missing-value mismatch: {context}")
    elif not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Numeric mismatch: {context}: {left} != {right}")


def unique(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"Duplicated exact-rank endpoint summary: {key}")
        result[key] = row
    return result


def validate_row(row):
    if row["rank"] not in (1, 2, 3) or row["requested_scenarios"] < 1 or row["source_scalar_comparisons"] < 1:
        raise ValueError("Invalid endpoint position or source denominator")
    cov, within, gmfe = (row[k] for k in ("coverage_percent", "within_twofold_percent", "gmfe"))
    if not math.isfinite(cov) or not 0 <= cov <= 100:
        raise ValueError("Coverage must be an all-request percentage")
    if within is not None and (not math.isfinite(within) or not 0 <= within <= 100):
        raise ValueError("Conditional twofold accuracy must be a percentage or null")
    if gmfe is not None and (not math.isfinite(gmfe) or gmfe < 1 - 1e-12):
        raise ValueError("GMFE must be finite and at least one, or null")
    if cov == 0 and (gmfe is not None or within is not None):
        raise ValueError("A zero-coverage position cannot have observed accuracy")
    if cov > 0 and (gmfe is None or within is None):
        raise ValueError("Covered scalar positions require conditional error and accuracy")


def build(root=ROOT):
    root = Path(root)
    legacy_path = root / "results/data/model_recommendation_execution.json"
    own_path = root / "results/data/pbpk_match_light_paper_metrics.json"
    legacy, own = read(legacy_path), read(own_path)
    if legacy["state"] != "complete" or own["state"] != "complete":
        raise ValueError("Endpoint publication requires both completed source reports")
    count_rows = unique(legacy["main_endpoint_counts"], ("endpoint",))
    measured = {key[0]: row for key, row in count_rows.items() if row["valid_scalar_comparisons"] > 0}
    if set(measured) != {key for key, _ in ENDPOINTS}:
        raise ValueError("Measured endpoint registry changed; explicitly review its table labels")
    unavailable = [dict(row) for row in legacy["main_endpoint_counts"] if not row["valid_scalar_comparisons"]]
    if own["scalar_endpoint_counts"] != {key: row["valid_scalar_comparisons"] for key, row in measured.items()}:
        raise ValueError("Own and comparator scalar source counts differ")
    old_methods = [{key: row[key] for key in ("method_id", "method", "group")} for row in legacy["main_cmax_table"]]
    if len(old_methods) != 12 or len({r["method_id"] for r in old_methods}) != 12:
        raise ValueError("Expected all twelve original comparator configurations")
    methods = old_methods + [{"method_id": key, "method": name, "group": "supervised_grouped_holdout"}
                            for key, name in zip(OWN, OWN_NAMES)]
    # The second execution stratum is deliberately excluded, even when it has
    # the same endpoint/method/rank identifiers.
    old_lookup = unique([r for r in legacy["rank_positions"]["summaries"]
                         if r["stratum"] == "reference_diagnostic" and r["endpoint"] in measured],
                        ("method_id", "endpoint", "position"))
    own_lookup = unique(own["scalar_endpoint_rows"], ("method", "endpoint", "rank"))
    expected_old = {(m["method_id"], e, k) for m in old_methods for e in measured for k in (1, 2, 3)}
    expected_own = {(m, e, k) for m in OWN for e in measured for k in (1, 2, 3)}
    if set(old_lookup) != expected_old or set(own_lookup) != expected_own:
        raise ValueError("Missing or unexpected method/endpoint/rank; refusing partial table")
    rows = []
    for endpoint, _ in ENDPOINTS:
        count = measured[endpoint]
        for method in methods:
            method_id = method["method_id"]
            for rank in (1, 2, 3):
                key = (method_id, endpoint, rank)
                row = {**method, "endpoint": endpoint, "rank": rank,
                       "stratum": "reference_diagnostic", "requested_scenarios": count["requested_scenarios"],
                       "source_scalar_comparisons": count["valid_scalar_comparisons"]}
                if method_id in OWN:
                    original = own_lookup[key]
                    equal(original["scenarios"], count["requested_scenarios"], f"{key}:requested scenarios")
                    equal(original["source_scalar_comparisons"], count["valid_scalar_comparisons"], f"{key}:comparisons")
                    row.update(gmfe=original["gmfe"], within_twofold_percent=None if original["within_twofold"] is None else 100 * original["within_twofold"],
                               coverage_percent=100 * original["coverage"], covered_scenarios=original["covered_scenarios"],
                               source="own_scalar_endpoint_rows")
                else:
                    original = old_lookup[key]
                    equal(original["requested_scenarios"], count["requested_scenarios"], f"{key}:requested scenarios")
                    row.update(gmfe=original["conditional_gmfe"],
                               within_twofold_percent=None if original["conditional_within2fold"] is None else 100 * original["conditional_within2fold"],
                               coverage_percent=100 * original["evaluable_probability"],
                               covered_scenarios=original["scenarios_with_nonzero_coverage"], source="legacy_rank_positions")
                validate_row(row)
                rows.append(row)
    lookup = unique(rows, ("method_id", "endpoint", "rank"))
    # Existing Cmax table contains independently reaggregated rank-1/2/3
    # values. Cross-check every old value and all six own positions separately.
    cmax_checks = 0
    for original in legacy["main_cmax_table"]:
        for rank in (1, 2, 3):
            row = lookup[(original["method_id"], "Cmax_reference_vs_group", rank)]
            for new, old in (("gmfe", "conditional_gmfe"), ("within_twofold_percent", "conditional_within2fold_percent"), ("coverage_percent", "coverage_percent")):
                equal(row[new], original[f"rank{rank}_{old}"], f"Cmax table {original['method_id']} rank {rank}: {new}")
                cmax_checks += 1
    for method in OWN:
        for rank in (1, 2, 3):
            row = lookup[(method, "Cmax_reference_vs_group", rank)]
            original = own_lookup[(method, "Cmax_reference_vs_group", rank)]
            for new, old, scale in (("gmfe", "gmfe", 1), ("within_twofold_percent", "within_twofold", 100), ("coverage_percent", "coverage", 100)):
                expected = None if original[old] is None else scale * original[old]
                equal(row[new], expected, f"Own Cmax {method} rank {rank}: {new}")
                cmax_checks += 1
    return {"schema": "pbpk-recommendation-all-measured-scalar-endpoints-v1", "state": "complete",
            "methods": methods, "endpoint_panels": [{"endpoint": endpoint, "tex_label": label,
                "requested_scenarios": measured[endpoint]["requested_scenarios"],
                "source_scalar_comparisons": measured[endpoint]["valid_scalar_comparisons"]} for endpoint, label in ENDPOINTS],
            "rows": rows, "unmeasured_registered_endpoints": unavailable,
            "definitions": {"gmfe": "exp(compound-balanced, publication-balanced mean absolute log error); conditional on a measurable candidate at this exact rank",
                "within_twofold_percent": "100 times hierarchical mean conditional fraction within a factor of two among measurable tied candidates; not all-request accuracy",
                "coverage_percent": "100 times equal-request mean probability of selecting a measurable tied candidate at this exact rank; all requested scenarios included",
                "requested_scenarios": "Count of requests for this endpoint, including missing observations; not candidate count",
                "source_scalar_comparisons": "Available candidate/scenario scalar comparisons, shared across methods; not independent sample size",
                "ties": "Uniform probability within the exact-rank tie group; no backfilling from lower ranks",
                "curves": "Concentration-curve GMFE is reported separately; scalar endpoints and curve samples are never pooled",
                "unmeasured": "Registered endpoints with zero valid scalar comparisons retained in JSON but not rendered as all-NA panels; Tmax was not evaluated"},
            "weighting": legacy["weighting"],
            "checks": {"method_count": 14, "endpoint_count": 7, "rank_count": 3, "row_count": len(rows),
                       "legacy_rows": len(old_lookup), "own_rows": len(own_lookup), "cmax_value_checks": cmax_checks,
                       "reference_diagnostic_only": True, "missing_values_preserved": True, "inference_or_fitting_performed": False},
            "source_files": {str(p.relative_to(root)): digest(p) for p in (legacy_path, own_path)},
            "producer": {"path": "scripts/build_recommendation_endpoint_table.py", "sha256": digest(__file__)},
            "reproduce": "runtime/app-env/bin/python scripts/build_recommendation_endpoint_table.py"}


def fmt(value, decimals):
    return "NA" if value is None else f"{value:.{decimals}f}"


def latex(report):
    header = [r"\toprule", r"Method & Rank 1 & Rank 2 & Rank 3 \\",
              r" & GMFE / $\leq2$-fold / Cov. & GMFE / $\leq2$-fold / Cov. & GMFE / $\leq2$-fold / Cov. \\",
              r" & $\phantom{GMFE}$ / (\%) / (\%) & $\phantom{GMFE}$ / (\%) / (\%) & $\phantom{GMFE}$ / (\%) / (\%) \\", r"\midrule"]
    lines = [r"% Generated from saved completed results; scalar endpoints only.", r"\begingroup", r"\renewcommand{\tablename}{Table}",
             r"\fontsize{8.2}{10.2}\selectfont", r"\setlength{\tabcolsep}{2pt}",
             r"\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}",
             r"\setlength{\LTcapwidth}{\linewidth}",
             r"\begin{longtable}{@{}p{.29\linewidth}*{3}{>{\centering\arraybackslash}p{.22\linewidth}}@{}}",
             r"\caption{Endpoint-specific prediction accuracy and evaluable coverage at exact ranks 1--3. Lower GMFE is better. Within-twofold accuracy is conditional on measured predictions; coverage includes every requested scenario. $n$ counts requests and $m$ counts available scalar comparisons, not independent samples. Tied candidates retain their selection probabilities; missing positions are not backfilled. Fourteen registered endpoints without valid scalar comparisons remain in the data export rather than all-NA panels. Concentration curves are reported separately in Figure~\ref{fig:model_execution_curves}.}\label{tab:model_execution}\\",
             *header, r"\endfirsthead", r"\caption[]{Endpoint-specific prediction accuracy and coverage (continued).}\\", *header,
             r"\endhead", r"\midrule", r"\multicolumn{4}{r}{\textit{Continued on next page}}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    lookup = unique(report["rows"], ("method_id", "endpoint", "rank"))
    for panel in report["endpoint_panels"]:
        title = panel["tex_label"] + f"; $n={panel['requested_scenarios']}$, $m={panel['source_scalar_comparisons']}$"
        lines.append(r"\multicolumn{4}{@{}l@{}}{\textbf{" + title + r"}}\\*")
        for i, method in enumerate(report["methods"]):
            name = method["method"]
            if method["method_id"] in OWN:
                name = r"\textbf{" + name + "}"
            cells = [name]
            for rank in (1, 2, 3):
                row = lookup[(method["method_id"], panel["endpoint"], rank)]
                cells.append(" / ".join((fmt(row["gmfe"], 3), fmt(row["within_twofold_percent"], 1), fmt(row["coverage_percent"], 1))))
            # Keep each endpoint's fourteen methods on the same page. Breaks
            # between endpoint panels are allowed and repeat the column header.
            end = r" \\*" if i < len(report["methods"]) - 1 else r" \\[5pt]"
            lines.append(" & ".join(cells) + end)
    lines += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def publish(root=ROOT):
    root = Path(root)
    report = build(root)
    data = root / "results/data/model_recommendation_endpoint_accuracy.json"
    table = root / "results/tables/model_recommendation_endpoint_accuracy"
    data.parent.mkdir(parents=True, exist_ok=True)
    table.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    table.with_suffix(".tex").write_text(latex(report))
    with table.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader(); writer.writerows(report["rows"])
    return report


FAMILIES = (
    ("curve", r"Concentration curve $C(t)$"),
    ("finite_AUC", r"Finite-window AUC / AUC$_{\rm last}$"),
    ("Cmax", r"Peak concentration $C_{\max}$"),
    ("Tmax", r"Peak time $T_{\max}$"),
    ("C_last", r"Last concentration $C_{\rm last}$"),
    ("half_life", r"Terminal half-life $t_{1/2}$"),
    ("AUC_infinity", r"AUC$_\infty$"),
    ("CL", r"Clearance CL (oral: CL/F)"),
    ("Vz", r"Terminal volume Vz (oral: Vz/F)"),
    ("MRT", r"Mean residence time MRT"),
    ("Vss", r"Steady-state volume Vss"),
)


def tex_text(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(c, c) for c in str(value))


def extended_panel_title(panel):
    original = dict(ENDPOINTS).get(panel["quantity"])
    if original:
        return original
    names = {"concentration_time_curve": r"Concentration curve $C(t)$",
             "AUC_last": r"AUC$_{\rm last}$", "AUC_infinity": r"AUC$_\infty$",
             "AUC_window": r"Finite-window AUC", "C_last": r"$C_{\rm last}$",
             "Tmax": r"$T_{\max}$", "Cmax": r"$C_{\max}$",
             "apparent_terminal_half_life": r"Apparent terminal $t_{1/2}$",
             "terminal_half_life": r"Terminal $t_{1/2}$",
             "MRT_including_absorption": r"MRT (including absorption)"}
    title = names.get(panel["quantity"], tex_text(panel["quantity"]))
    notes = []
    if panel.get("route") in ("oral", "intravenous"):
        notes.append("oral" if panel["route"] == "oral" else "IV")
    if panel.get("steady_state"):
        notes.append("steady state")
    if panel.get("dose_normalization"):
        notes.append("dose/body-weight normalized")
    elif panel.get("body_weight_normalized"):
        notes.append("per kg")
    if panel.get("window_h") is not None and panel["family"] == "finite_AUC":
        a, b = panel["window_h"]
        notes.append(f"{a:g}--{b:g} h")
    return title + ("; " + ", ".join(notes) if notes else "")


def extension_data(report, methods):
    """Validate every registered method/rank before selecting display panels."""
    method_count = report.get('publication_method_count', 14)
    inventories = {row["family"]: row for row in report["family_inventory"]}
    if set(inventories) != {k for k, _ in FAMILIES}:
        raise ValueError("The endpoint extension must account for all eleven outputs")
    rows = report["publication_rows"]
    lookup = unique(rows, ("method_id", "endpoint_key", "source_stratum", "rank"))
    panels = unique(report["endpoint_panels"], ("endpoint_key", "source_stratum"))
    expected = {(m["method_id"], *key, rank) for key in panels for m in methods for rank in (1, 2, 3)}
    if set(lookup) != expected:
        raise ValueError(f"Incomplete endpoint extension: every panel requires {method_count} methods and 3 exact ranks")
    for row in rows:
        if not 0 <= row["coverage"] <= 1 or row["denominator"] < 0:
            raise ValueError("Invalid endpoint coverage or denominator")
        if row["denominator"] == 0 and row["coverage"] != 0:
            raise ValueError("An endpoint without an eligible reference cannot have coverage")
        error = row["mae_h"] if row["metric_kind"] == "absolute_hours" else row["gmfe"]
        if row["coverage"] == 0 and error is not None:
            raise ValueError("An unavailable endpoint cannot have a numeric error")
        if row["coverage"] > 0 and (error is None or not math.isfinite(error)):
            raise ValueError("A covered endpoint requires a finite error")
        if error is not None and error < (0 if row["metric_kind"] == "absolute_hours" else 1 - 1e-12):
            raise ValueError("Invalid absolute error or GMFE")
        if row["metric_kind"] == "absolute_hours" and row["gmfe"] is not None:
            raise ValueError("Tmax must not be represented by GMFE")
    if len(methods) != method_count or len({m['method_id'] for m in methods}) != method_count:
        raise ValueError(f"The overview must include all {method_count} registered methods")
    return inventories, lookup, panels


def extended_latex(report, methods, *, compact=False):
    """Retain the full detailed longtable exporter for reproducible supplementary data."""
    if compact:
        return overview_latex(report, methods)
    inventories, lookup, panels = extension_data(report, methods)

    head = [r"\toprule", r"Method & Rank 1 & Rank 2 & Rank 3 \\",
            r" & Error / 2F / Cov. & Error / 2F / Cov. & Error / 2F / Cov. \\",
            r" & $\phantom{Error}$ / (\%) / (\%) & $\phantom{Error}$ / (\%) / (\%) & $\phantom{Error}$ / (\%) / (\%) \\", r"\midrule"]
    caption = (r"Prediction accuracy and coverage for the eleven blueprint-aligned outputs. Error is GMFE except for Tmax (mean absolute error, h); 2F is the conditional percentage within twofold and is not defined for Tmax. Cov. is the selection probability of a measurable prediction at the exact rank, averaged over the $n$ scenarios of that endpoint definition. $m$ counts qualified candidate--scenario comparisons, not independent samples. Native and clinical-sampling NCA results remain separate, as do oral apparent quantities, steady-state outcomes and AUC windows. ")
    caption += r"All 14 methods and exact ranks 1--3 are retained; ties use uniform probabilities without backfilling."
    rendered_methods = methods
    label = "tab:model_execution_all_endpoints"
    lines = [r"% Eleven output families; all measured definitions retained.",
             r"\begingroup", r"\renewcommand{\tablename}{Table}",
             r"\fontsize{8.2}{10.2}\selectfont", r"\setlength{\tabcolsep}{2pt}",
             r"\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}",
             r"\setlength{\LTcapwidth}{\linewidth}",
             r"\begin{longtable}{@{}p{.29\linewidth}*{3}{>{\centering\arraybackslash}p{.22\linewidth}}@{}}",
             r"\caption{" + caption + r"}\label{" + label + r"}\\",
             *head, r"\endfirsthead", r"\caption[]{Eleven-output prediction accuracy and coverage (continued).}\\",
             *head, r"\endhead", r"\midrule", r"\multicolumn{4}{r}{\textit{Continued on next page}}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    strata = {"legacy:reference_individual": "original native comparison",
              "main:protocol_matched": "clinical-sampling NCA",
              "main:native": "native trajectory"}
    for number, (family, label) in enumerate(FAMILIES, 1):
        inventory = inventories[family]
        line = (f"{number}. " + label + f"; references: {inventory['reference_scenarios']}/41 scenes; "
                f"any qualified main prediction: {inventory['matched_scenarios']}/41 scenes")
        lines.append(r"\multicolumn{4}{@{}p{.98\linewidth}@{}}{\textbf{" + line + r"}}\\*")
        family_panels = [(key, panel) for key, panel in panels.items()
                         if panel["family"] == family and panel["source_comparisons"] > 0]
        if not family_panels:
            reason = {"C_last": "The sole reported reference has a time/concentration unit conflict; no qualified reference is available.",
                      "Vss": "One IV reference exists, but predictions fail the terminal-span or AUMC-extrapolation qualification."}.get(
                          family, "Missing or incompatible references and prediction qualifications are retained in the accompanying data.")
            lines.append(r"\multicolumn{4}{@{}p{.98\linewidth}@{}}{No qualified comparison. Accuracy is NA. " + reason + r"}\\[5pt]")
        for key, panel in family_panels:
            met = "MAE (h)" if panel["metric_kind"] == "absolute_hours" else "GMFE"
            definition = extended_panel_title(panel)
            stratum = tex_text(strata.get(panel["source_stratum"], panel["source_stratum"]))
            title = f"{definition}; {stratum}; {met}; $n={panel['denominator']}$, $m={panel['source_comparisons']}$"
            lines.append(r"\multicolumn{4}{@{}p{.98\linewidth}@{}}{\textit{" + title + r"}}\\*")
            for i, method in enumerate(rendered_methods):
                name = tex_text(method["method"])
                if method["method_id"] in OWN:
                    name = r"\textbf{" + name + "}"
                cells = [name]
                for rank in (1, 2, 3):
                    row = lookup[(method["method_id"], *key, rank)]
                    error = row["mae_h"] if row["metric_kind"] == "absolute_hours" else row["gmfe"]
                    accuracy = (None if row["within_twofold"] is None else 100 * row["within_twofold"])
                    acc = "--" if row["metric_kind"] == "absolute_hours" else fmt(accuracy, 1)
                    cells.append(" / ".join((fmt(error, 3), acc, fmt(100 * row["coverage"], 1))))
                lines.append(" & ".join(cells) + (r" \\*" if i < len(rendered_methods) - 1 else r" \\[5pt]"))
    lines.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines) + "\n"


SHORT_METHODS = {
    'bm25_text': 'BM25', 'qwen3_embedding_06b': 'Qwen Emb 0.6B',
    'f2llm_v2_06b': 'F2LLM-v2', 'qwen3_embedding_4b': 'Qwen Emb 4B',
    'qwen3_reranker_06b': 'Qwen Rerank', 'jina_reranker_v35': 'Jina',
    'g_reasoner_34m': 'G-Reasoner', 'gfm_rag_8m': 'GFM-RAG',
    'gfm_rag_8m_idf_continuous': 'GFM-RAG + IDF',
    'bm25_top20_reasonrank_7b_nf4': 'ReasonRank', 'bm25_top20_rearank_7b_nf4': 'REARANK',
    'bm25_top20_rank_r1_3b': 'Rank-R1',
    'pbpk_match_light_document': 'PBPK-Match w/o error', 'pbpk_match_light': 'PBPK-Match',
    'pbpk_match_components': 'PBPK-Match + components'}
SHORT_ENDPOINTS = {
    'curve': r'$C(t)$', 'finite_AUC': r'AUC$_{\rm last}$', 'Cmax': r'$C_{\max}$',
    'Tmax': r'$T_{\max}$ (h)', 'C_last': r'$C_{\rm last}$',
    'half_life': r'$t_{1/2}$', 'AUC_infinity': r'AUC$_\infty$',
    'CL': 'CL/F', 'Vz': 'Vz/F', 'MRT': r'MRT$^{a}$', 'Vss': 'Vss'}
OVERVIEW_COLUMN_ORDER = ('curve', 'Cmax', 'Tmax', 'finite_AUC', 'AUC_infinity',
                         'half_life', 'CL', 'Vz', 'MRT')


def panel_key(panel):
    return panel['endpoint_key'], panel['source_stratum']


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def overview_data(report, methods):
    """Select by reference count alone, with deterministic definition tie breaks."""
    inventories, lookup, panels = extension_data(report, methods)
    selected = []
    for family, _ in FAMILIES:
        available = [p for p in panels.values() if p['family'] == family]
        if available:
            panel = min(available, key=lambda p: (-p.get('eligible_reference_scenarios', p['denominator']),
                                                 p['source_stratum'], p['endpoint_key']))
            selected.append(dict(panel, displayed_column=panel['source_comparisons'] > 0))
        else:
            # Fixtures and future no-reference families still have an explicit
            # shared-NA entry, never a fabricated numeric zero-error column.
            selected.append({'family': family, 'endpoint_key': None, 'source_stratum': None,
                             'denominator': 0, 'eligible_reference_scenarios': 0,
                             'source_comparisons': 0, 'displayed_column': False})
    chosen_keys = {panel_key(p) for p in selected if p['endpoint_key'] is not None}
    order = {family: i for i, (family, _) in enumerate(FAMILIES)}
    remaining = sorted((p for p in panels.values() if p['source_comparisons'] > 0 and panel_key(p) not in chosen_keys),
                       key=lambda p: (order[p['family']], p['source_stratum'], p['endpoint_key']))
    layouts = []
    # Bind both displayed and undisplayed rows, preserving all 36 definitions.
    row_bindings = [{'method_id': r['method_id'], 'endpoint_key': r['endpoint_key'],
                     'source_stratum': r['source_stratum'], 'rank': r['rank'], 'sha256': canonical_hash(r)}
                    for r in report['publication_rows']]
    for panel in panels.values():
        key = panel_key(panel)
        for method in methods:
            ranks = []
            for rank in (1, 2, 3):
                row = lookup[(method['method_id'], *key, rank)]
                error = row['mae_h'] if row['metric_kind'] == 'absolute_hours' else row['gmfe']
                maximum = max(lookup[(m['method_id'], *key, rank)]['coverage'] for m in methods)
                ranks.append({'rank': rank, 'error': error,
                    'error_display': fmt(error, 3 if panel['family'] == 'Cmax' else 2),
                    'coverage': row['coverage'], 'coverage_percent': 100 * row['coverage'],
                    'coverage_display': fmt(100 * row['coverage'], 1),
                    'highest_coverage': maximum > 0 and math.isclose(row['coverage'], maximum, rel_tol=0, abs_tol=1e-12),
                    'source_row_sha256': canonical_hash(row)})
            layouts.append({'method_id': method['method_id'], 'endpoint_key': key[0], 'source_stratum': key[1], 'ranks': ranks})
    return {'schema': 'pbpk-recommendation-endpoint-overview-v1', 'state': 'complete',
        'selection_rule': 'For each of the eleven families select the definition with the most eligible reference scenarios; ties use source_stratum then endpoint_key lexicographic order. Errors and coverage are never used for definition selection.',
        'selected_definitions': selected, 'additional_measured_definitions': remaining,
        'unscorable_definitions': [p for p in panels.values() if p['source_comparisons'] == 0],
        'methods': [{**m, 'short_label': SHORT_METHODS.get(m['method_id'], m['method'])} for m in methods],
        'layout_data': layouts, 'source_row_bindings': row_bindings,
        'display_policy': {'ranks': [1, 2, 3], 'error_decimals': 2,
            'error_decimals_by_family': {'Cmax': 3}, 'coverage_percent_decimals': 1,
            'column_order': list(OVERVIEW_COLUMN_ORDER),
            'missing_error': 'NA', 'coverage_bold': False,
            'error_bold': False, 'method_rows': 2, 'all_method_count': len(methods),
            'shared_unavailable_families': [p['family'] for p in selected if not p['displayed_column']],
            'font_pt': 8.3, 'leading_pt': 9.4, 'coverage_font_pt': 7.4,
            'header_fill': 'pale sage', 'method_fill': None,
            'maximum_requested_width_pt': 648,
            'target_maximum_table_height_pt': 390, 'float': 'landscape table[p]',
            'physical_size_verified_by': 'Final PDF compilation and page-layout audit, not guessed from this data file.'},
        'checks': {'complete_rows_validated': len(report['publication_rows']), 'family_count': 11,
            'method_count': len(methods), 'ranks_per_cell': 3,
            'selected_source_rows': sum(p['endpoint_key'] is not None for p in selected) * len(methods) * 3,
            'displayed_source_rows': sum(p['displayed_column'] for p in selected) * len(methods) * 3,
            'errors_changed': False, 'source_metrics_modified': False},
        'family_inventory': inventories}


def prediction_symbol(panel):
    return r'$P^*$' if panel['source_stratum'] == 'main:protocol_matched' else r'$P$'


def matrix_latex(report, methods, panels, *, label, caption, notes, header_labels=None, column_groups=None):
    """Render a single landscape float; row heights do not scale with endpoints."""
    _, lookup, _ = extension_data(report, methods)
    if not panels:
        raise ValueError('Cannot render an empty endpoint matrix')
    header_labels = header_labels or [extended_panel_title(p) for p in panels]
    headers = [r'\shortstack[l]{Method\\Rank 1/2/3}']
    for title, panel in zip(header_labels, panels):
        headers.append(r'\shortstack{' + title + r'\\' + prediction_symbol(panel)
                       + f", $n={panel.get('eligible_reference_scenarios', panel['denominator'])}$" + '}')
    lines = [f'% Generated from completed {len(methods)}-configuration summaries; no model inference.',
        r'\begin{landscape}', r'\begin{table}[p]', r'\centering', r'\begingroup',
        r'\renewcommand{\tablename}{Table}', r'\captionsetup{font=small,skip=4pt}',
        r'\caption{' + caption + r'}\label{' + label + '}',
        r'\fontsize{8.3}{9.4}\selectfont', r'\setlength{\tabcolsep}{2pt}',
        r'\renewcommand{\arraystretch}{0}',
        r'\definecolor{PBPKoverviewhead}{RGB}{236,242,237}',
        r'\definecolor{PBPKoverviewink}{RGB}{64,83,69}',
        r'\definecolor{PBPKoverviewmuted}{RGB}{98,102,99}',
        r'\newcommand{\PBPKcoverage}[1]{\textcolor{PBPKoverviewmuted}{\fontsize{7.4}{8.5}\selectfont #1}}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l*{' + str(len(panels)) + r'}{c}@{}}',
        r'\toprule']
    if column_groups:
        if sum(n for n, _ in column_groups) != len(panels):
            raise ValueError('Endpoint group widths do not match the displayed columns')
        lines.append(r'\rowcolor{PBPKoverviewhead}')
        lines.append(r'\rule[-2.4pt]{0pt}{12pt} & ' + ' & '.join(
            r'\multicolumn{' + str(n) + r'}{c}{\textcolor{PBPKoverviewink}{\textbf{' + name + '}}}'
            for n, name in column_groups) + r' \\')
        first = 2
        rules = []
        for width, _ in column_groups:
            rules.append(r'\cmidrule(lr){' + str(first) + '-' + str(first + width - 1) + '}')
            first += width
        lines.append(''.join(rules))
    lines += [' & '.join(headers) + r' \\', r'\midrule']
    groups = {
        methods[0]['method_id']: 'Frozen full-library scorers',
        'bm25_top20_reasonrank_7b_nf4': 'BM25 Top-20 rerankers',
        OWN[0]: 'PBPK-Match configurations',
    }
    for method_index, method in enumerate(methods):
        mid = method['method_id']
        if mid in groups:
            if method_index:
                lines.append(r'\addlinespace[2pt]')
            lines.append(r'\multicolumn{' + str(len(panels) + 1)
                         + r'}{@{}l@{}}{\rule[-1.6pt]{0pt}{11pt}'
                         + r'\textcolor{PBPKoverviewink}{\fontsize{7.6}{8.8}\selectfont\itshape '
                         + groups[mid] + r'}} \\')
        name = tex_text(SHORT_METHODS.get(mid, method['method']))
        if mid == OWN[1]:
            name = r'\textbf{' + name + '}'
        error_cells = [r'\rule[-1.2pt]{0pt}{8.3pt}' + name]
        coverage_cells = [r'\rule[-1.1pt]{0pt}{6.9pt}\PBPKcoverage{Coverage (\%)}']
        for panel in panels:
            key = panel_key(panel)
            errors, coverage = [], []
            for rank in (1, 2, 3):
                row = lookup[(mid, *key, rank)]
                value = row['mae_h'] if row['metric_kind'] == 'absolute_hours' else row['gmfe']
                errors.append(fmt(value, 3 if panel['family'] == 'Cmax' else 2))
                token = fmt(100 * row['coverage'], 1)
                coverage.append(token)
            error_cells.append('/'.join(errors))
            coverage_cells.append(r'\PBPKcoverage{' + '/'.join(coverage) + '}')
        lines.append(' & '.join(error_cells) + r' \\')
        lines.append(' & '.join(coverage_cells) + r' \\[1.2pt]')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\par\vspace{3pt}\noindent',
              r'\begin{minipage}{\linewidth}\fontsize{7.4}{8.5}\selectfont', notes,
              r'\end{minipage}', r'\endgroup', r'\end{table}', r'\end{landscape}']
    rendered = '\n'.join(lines) + '\n'
    if label == 'tab:model_execution':
        # A graphicx-rotated normal float preserves the full landscape table
        # without forcing the preceding paragraph onto a near-empty page.
        rendered = rendered.replace(
            '\\begin{landscape}\n\\begin{table}[p]\n\\centering\n\\begingroup',
            '\\begin{table}[p]\n\\centering\n\\special{pdf: put @thispage << /Rotate 90 >>}\n\\rotatebox{90}{\\begin{minipage}{\\textheight}\n\\begingroup')
        rendered = rendered.replace(
            '\\endgroup\n\\end{table}\n\\end{landscape}',
            '\\endgroup\n\\end{minipage}}\n\\end{table}')
        rendered = rendered.replace('font=small,skip=4pt', 'font=small,skip=3pt')
        rendered = rendered.replace(r'\\[1.2pt]', r'\\[0.7pt]')
        rendered = rendered.replace(r'\par\vspace{3pt}\noindent', r'\par\vspace{1pt}\noindent')
        rendered = rendered.replace('font=small,skip=3pt', 'font=footnotesize,skip=2pt')
        rendered = rendered.replace(r'\addlinespace[2pt]', r'\addlinespace[1pt]')
        rendered = rendered.replace(r'\par\vspace{1pt}\noindent', r'\par\noindent')
        rendered = rendered.replace(r'\begin{minipage}{\linewidth}\fontsize{7.4}{8.5}', r'\begin{minipage}{\linewidth}\fontsize{7.4}{8}')
    return rendered


def overview_latex(report, methods):
    overview = overview_data(report, methods)
    panels = [p for p in overview['selected_definitions'] if p['displayed_column']]
    panels.sort(key=lambda p: OVERVIEW_COLUMN_ORDER.index(p['family']))
    notes = (r'Within each cell: ranks 1/2/3. Upper rows: GMFE (Tmax: MAE, h; lower is better); smaller grey rows: coverage (\%) of $n$ eligible reference scenarios. '
        r'Errors are conditional on scorable scenarios, which can differ across methods. '
        r'$P$: native reference-individual prediction; $P^*$: clinical-sampling NCA diagnostic. '
        r'One definition per family is selected solely by the largest eligible reference count (ties: stratum, then endpoint key); the other measured definitions are retained in the separate supplementary materials. '
        r'CL/F and Vz/F are oral apparent quantities; $^{a}$MRT includes absorption. Oral single-dose definitions are used for $P^*$ columns. '
        r'AUC$_{\rm last}$ uses predicted last-quantifiable time and does not establish individual observed $t_{\rm last}$ equivalence. '
        r'The curve column contains two cases from one study. Qwen abbreviates Qwen3; w/o error disables predicted-error correction. '
        r'Ties retain equal probabilities; unavailable ranks are not backfilled. ')
    if any(m['method_id'] == 'pbpk_match_components' for m in methods):
        notes += (r'+ components appends evidence-supported soft requirements to the same frozen PBPK-Match; '
                  r'9/41 execution requests receive component inputs, with original-input fallback for the rest. ')
    unavailable = [p for p in overview['selected_definitions'] if not p['displayed_column']]
    if unavailable:
        descriptions = []
        for panel in unavailable:
            family = panel['family']
            inventory = overview['family_inventory'][family]
            if family == 'C_last' and inventory['reference_scenarios']:
                text = r'$C_{\rm last}$: source time/concentration-unit conflict ($n=0$ eligible)'
            elif family == 'Vss' and inventory['reference_scenarios']:
                text = r'Vss: one IV reference; no prediction passes the terminal-span/AUMC gate'
            else:
                text = SHORT_ENDPOINTS[family] + ': no qualified comparison'
            descriptions.append(text)
        notes += r'\textbf{Shared NA outputs (all ' + str(len(methods)) + r' configurations, all three ranks):} ' + '; '.join(descriptions) + '. '
    notes += (f"The full CSV/JSON retain all {len(report['endpoint_panels'])} definitions, including "
              f"the {len(overview['unscorable_definitions'])} without qualified comparisons.")
    labels = []
    for panel in panels:
        label = SHORT_ENDPOINTS[panel['family']]
        if panel['family'] in ('CL', 'Vz', 'Vss'):
            label = tex_text(panel['quantity'])
        elif panel['family'] == 'finite_AUC' and panel['quantity'] != 'AUC_last':
            label = dict(ENDPOINTS).get(panel['quantity'], r'AUC$_{\rm window}$')
        elif panel['family'] == 'MRT' and panel['quantity'] != 'MRT_including_absorption':
            label = 'MRT'
        labels.append(label)
    return matrix_latex(report, methods, panels, label='tab:model_execution',
        caption=f'Prediction error and evaluable coverage for all {len(methods)} recommendation configurations.',
        notes=notes, header_labels=labels,
        column_groups=[(1, 'Curve'), (2, 'Peak and timing'), (2, 'Exposure'), (4, 'Disposition')]
                      if len(panels) == len(OVERVIEW_COLUMN_ORDER) else None)


def additional_matrices_latex(report, methods):
    overview = overview_data(report, methods)
    panels = overview['additional_measured_definitions']
    # At most eight endpoint triplets per landscape page, always all methods.
    chunks = [panels[i:i + 8] for i in range(0, len(panels), 8)]
    result = []
    for page, chunk in enumerate(chunks):
        labels = [f'D{i + page * 8 + 1}' for i in range(len(chunk))]
        definitions = [r'\textbf{' + tag + '}: ' + extended_panel_title(p) + '.' for tag, p in zip(labels, chunk)]
        notes = (r'Upper/smaller grey rows give exact rank-1/2/3 error and coverage (\%); error is GMFE except Tmax (MAE, h). '
            r'$P$: native prediction; $P^*$: clinical-sampling NCA. Errors are conditional on scorable scenarios; no cell is marked as a winner. '
            r'No rank backfilling. Different windows, routes, steady-state and normalization definitions remain separate. '
            r'Qwen denotes Qwen3; w/o error disables predicted-error correction. '
            + ' '.join(definitions))
        result.append(matrix_latex(report, methods, chunk,
            label='tab:model_execution_all_endpoints' if page == 0 else 'tab:model_execution_all_endpoints_extra',
            caption=f'Additional endpoint definitions for all {len(methods)} recommendation configurations ({page + 1}/{len(chunks)}).',
            notes=notes, header_labels=labels))
    return '\n'.join(result)


def publish_extension(root=ROOT):
    root = Path(root)
    path = root / "results/data/model_recommendation_extended_metrics.json"
    report = read(path)
    methods = build(root)["methods"]
    destination = root / "results/tables/model_recommendation_eleven_endpoints"
    destination.with_suffix(".tex").write_text(overview_latex(report, methods))
    destination.with_name(destination.name + "_all_methods").with_suffix(".tex").write_text(additional_matrices_latex(report, methods))
    overview = overview_data(report, methods)
    overview.update(source={'path': str(path.relative_to(root)), 'sha256': digest(path)},
                    producer={'path': 'scripts/build_recommendation_endpoint_table.py', 'sha256': digest(__file__)})
    (root / 'results/data/model_recommendation_endpoint_overview.json').write_text(
        json.dumps(overview, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    rows = report["publication_rows"]
    fields = sorted({key for row in rows for key in row})
    with destination.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"families": 11, "methods": 14, "panels": len(report["endpoint_panels"]),
            "metric_rows": len(rows), "source_sha256": digest(path)}


if __name__ == "__main__":
    report = publish()
    print(json.dumps({**report["checks"], "unmeasured_registered_endpoints": len(report["unmeasured_registered_endpoints"])}, indent=2))
    if (ROOT / "results/data/model_recommendation_extended_metrics.json").exists():
        print(json.dumps(publish_extension(), indent=2))
