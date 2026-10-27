#!/usr/bin/env python3
"""Render manuscript Table 10 from the frozen optimized Ours predictions.

No simulation, parameter fitting or clinical-reference estimation is performed.
P is read from each saved native endpoints.json; P* is the frozen prediction-only
clinical-protocol NCA. Only original-source scalar references are included.
Run with the project Python; outputs replace blueprint_clinical_metrics.tex/.csv
and blueprint_clinical_audit.json in results/tables. Input hashes, unit factors,
missing native values and all study limitations are retained in the CSV/audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import render_pbpk_method_comparison as comparison

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug/optimized_run/comparison_input.json"
DEST = ROOT / "results/tables"
TAIL_KEYS = {"terminal_half_life", "auc_inf", "clearance", "vz", "mrt", "vss"}
STUDIES = {
    "Treprostinil": "LRX-TRIUMPH BA.001",
    "Ertapenem": "Pletz 2004, day 1",
    "Ceftriaxone": "Pletz 2004, day 1",
    "Cefepime": "Garrelts 1999, 3-min infusion",
    "Dexmedetomidine": "Song 2018, control participant B",
    "Edaravone": "Shimizu 2021, J03",
    "Midazolam": "CA2560024, Example 2",
    "Moxifloxacin": "Stass 2005, Mono IV",
    "Nemonoxacin": "Cao 2014, 500 mg",
    "Remdesivir": "Humeniuk 2020, SAD 150 mg",
}
NATIVE_FIELDS = {
    "auc": ("value_mg_h_per_l", "mg*h/l"),
    "cmax": ("value_mg_per_l", "mg/l"),
    "tmax": ("value_h", "h"),
    "c_last": ("value_mg_per_l", "mg/l"),
    "terminal_half_life": ("value_h", "h"),
    "auc_inf": ("value_mg_h_per_l", "mg*h/l"),
    "clearance": ("value_l_per_h", "l/h"),
    "vz": ("value_l", "l"),
    "mrt": ("value_h", "h"),
    "vss": ("value_l", "l"),
}


def equal(left, right, label):
    if not np.allclose(left, right, rtol=1e-9, atol=1e-12, equal_nan=False):
        raise ValueError(f"Numerical verification failed: {label}")


def native_value(endpoint: dict, key: str, target_unit: str, weight: float):
    if endpoint.get("status") != "completed":
        return None, None, endpoint.get("reason", "Native endpoint unavailable")
    field, unit = NATIVE_FIELDS[key]
    if endpoint.get("units", {}).get(field) != unit:
        raise ValueError(f"Unexpected native unit for {key}: {endpoint.get('units')}")
    value = float(endpoint[field])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Nonfinite/negative native {key}")
    factors = {("mg*h/l", "ng.h/mL"): 1000., ("mg/l", "ng/mL"): 1000.,
               ("h", "h"): 1., ("l/h", "L/h"): 1.,
               ("l/h", "mL/min"): 1000/60, ("l/h", "L/h/kg"): 1/weight,
               ("l", "L"): 1., ("l", "L/kg"): 1/weight}
    if (unit, target_unit) not in factors:
        raise ValueError(f"Unsupported native-to-clinical unit conversion: {unit} -> {target_unit}")
    factor = factors[unit, target_unit]
    return value*factor, factor, ""


def scalar_native_audit(native: dict, curve: dict, mw: float, dose_mg: float):
    """Check saved scalar values against the locked curve; do not replace any."""
    t = np.asarray(curve["time_min"])/60
    c = np.asarray(curve["concentration"])*mw/1000
    if curve["time_unit"] != "min" or curve["unit"] != "umol/l":
        raise ValueError("Unexpected native trajectory units")
    start, end = (native["main_window"][k] for k in ("start_h", "end_h"))
    if start < t[0] or end > t[-1] or not np.all(np.diff(t) > 0):
        raise ValueError("Native endpoint window outside locked trajectory")
    grid = np.r_[start, t[(t > start) & (t < end)], end]
    values = np.interp(grid, t, c)
    ep = native["endpoints"]
    equal(ep["auc"]["value_mg_h_per_l"], np.trapezoid(values, grid), "native AUC")
    peak = int(np.argmax(values))
    equal(ep["cmax"]["value_mg_per_l"], values[peak], "native Cmax")
    equal(ep["tmax"]["value_h"], grid[peak], "native Tmax")
    equal(ep["c_last"]["value_mg_per_l"], values[-1], "native Clast")
    if ep["terminal_half_life"]["status"] == "completed":
        equal(ep["terminal_half_life"]["value_h"], math.log(2)/ep["terminal_half_life"]["lambda_z_per_h"], "native half-life")
    if ep["clearance"]["status"] == "completed":
        equal(ep["clearance"]["value_l_per_h"], dose_mg/ep["auc_inf"]["value_mg_h_per_l"], "native CL")
    if ep["vz"]["status"] == "completed":
        equal(ep["vz"]["value_l"], ep["clearance"]["value_l_per_h"]/ep["terminal_half_life"]["lambda_z_per_h"], "native Vz")
    if ep["vss"]["status"] == "completed":
        equal(ep["vss"]["value_l"], ep["clearance"]["value_l_per_h"]*ep["mrt"]["value_h"], "native Vss")


def number(value, dagger=False):
    text = comparison.number(value)
    return text + (r"\(^{\dagger}\)" if dagger and value is not None else "")


def row(cells, *, keep=True, stripe=False):
    return (r"\rowcolor{BPClinicalStripe}" if stripe else "") + " & ".join(cells) + (r" \\*" if keep else r" \\")


def note(text, *, keep=False):
    return (r"\multicolumn{7}{@{}p{250mm}@{}}{\fontsize{8.5}{10.5}\selectfont\color{BPClinicalGray}" + text + "}" + (r" \\*" if keep else r" \\"))


def band(text, *, dark=False):
    return (r"\rowcolor{" + ("BPClinicalInk" if dark else "BPClinicalFaint")
            + r"}\multicolumn{7}{@{}p{250mm}@{}}{\strut\hspace{1mm}"
            + (r"\color{white}\bfseries " if dark else "") + text + r"} \\*")


def table_tex(cases, rows):
    notes = [
        r"\textbf{Values.} P: saved native endpoints on the displayed request/model-tail windows. P$^*$: the same prediction with clinical sampling and cutoff rules; integration is linear for Pletz 2004, linear-up/log-down otherwise. Native terminal P is descriptive and not scored. O: original study-reported scalars, never reconstructed; missing-O rows are omitted.",
        r"\textbf{Errors.} FE=$\max(P^*/O,O/P^*)$ (ideal 1); $T_{\max}$ uses absolute hours (ideal 0). Curve metrics use observed sampling times and natural logs; $n$ counts positive observations. --- means unavailable, not zero. Values use four significant figures.",
        r"\textbf{Quality.} $\dagger$ flags failed endpoint-specific screens: span $\geq1$ half-life; span plus AUC extrapolation $\leq20\%$ for AUC$_\infty$/CL/$V_z$; additionally AUMC extrapolation $\leq20\%$ for MRT/$V_{\rm ss}$. Flagged errors are descriptive. Individual terminal fits are unavailable; NCA and adult-to-group comparisons are approximate. These viewed cases are development evaluations, not blinded validation.",
    ]
    tex = ["% Generated by scripts/render_current_clinical_table.py; optimized Ours only.",
           r"\begingroup\captionsetup[table]{name=Table}",
           r"\definecolor{BPClinicalInk}{HTML}{233A4B}\definecolor{BPClinicalFaint}{HTML}{EAF0F3}",
           r"\definecolor{BPClinicalStripe}{HTML}{F6F9FA}\definecolor{BPClinicalGray}{HTML}{56636E}",
           r"\setlength{\tabcolsep}{2.5pt}\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}",
           r"\setlength{\LTcapwidth}{250mm}\setlength{\LTpre}{2pt}\setlength{\LTpost}{2pt}",
           r"\renewcommand{\arraystretch}{1.08}\fontsize{10}{13}\selectfont",
           r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{39.8mm}>{\centering\arraybackslash}p{20mm}*{3}{>{\centering\arraybackslash}p{37.55mm}}>{\centering\arraybackslash}p{32mm}>{\centering\arraybackslash}p{35mm}@{}}",
           r"\caption{Optimized Ours: 80 scalar comparisons against original clinical references from 10 drugs; 104 clinical curve points.}\label{tab:blueprint_clinical_metrics}\\",
           r"\toprule"]
    header = row([r"\textbf{Endpoint}", r"\textbf{Unit}", r"\shortstack{Native prediction\\P}",
                  r"\shortstack{Matched prediction\\P$^*$}", r"\shortstack{Observation-based\\reference O}",
                  r"\shortstack{Fold error\\FE}", r"\shortstack{Absolute error\\h ($T_{\max}$ only)}"], keep=False, stripe=True)
    tex += [header, r"\midrule\endfirsthead", r"\multicolumn{7}{@{}l@{}}{\textcolor{BPClinicalGray}{\bfseries Table~\thetable\quad Optimized Ours: clinical endpoint comparisons (continued)}}\\[3pt]", r"\toprule", header, r"\midrule\endhead", r"\midrule"]
    tex += [note(x) for x in notes] + [r"\bottomrule\endfoot", r"\midrule"]
    tex += [note(x) for x in notes] + [r"\bottomrule\endlastfoot"]
    for index, case in enumerate(cases, 1):
        if index > 1:
            tex.append(r"\addlinespace[5pt]")
        metric = case["curve_metrics"]["ours"]
        title = comparison.escape(f"{index:02d}. {case['drug']} / {STUDIES[case['drug']]}")
        title += rf"\hfill GMFE {number(metric['gmfe'])}; ln-RMSE {number(metric['ln_rmse'])}; $n={metric['n']}$\hspace{{1mm}}"
        tex += [band(title, dark=True), band(comparison.escape(case["native_context"]))]
        records = [r for r in rows if r["case_id"] == case["case_id"]]
        for j, item in enumerate(records):
            label = comparison.TEX_ENDPOINTS[item["endpoint"]] if item["endpoint_label"] == comparison.ENDPOINTS[item["endpoint"]] else comparison.escape(item["endpoint_label"])
            unit = comparison.escape(item["display_unit"].replace(".h/", "·h/"))
            factor = item["display_value_factor"]
            native = item["predicted_value"]
            cells = [label, unit, number(None if native is None else native*factor),
                     number(item["evaluation_prediction"]*factor, item["matched_weak_tail"]),
                     number(item["observed_value"]*factor), number(item["fold_error"], item["matched_weak_tail"]),
                     number(item["absolute_error_h"])]
            tex.append(row(cells, stripe=j % 2 == 0))
        omitted = ", ".join(comparison.ENDPOINTS[k] for k in case["missing_clinical_endpoints"])
        case_notes = ["Source: "+case["citation"]+"."]
        if omitted:
            case_notes.append("Original scalar reference not reported/qualified (row omitted): "+omitted+".")
        source_notes = case.get("clinical_reference_quality_notes", [])
        if source_notes:
            case_notes.append("Reference limitations: "+" ".join(source_notes))
        for key in case["approximations"]:
            if key not in {"representative_adult_vs_group_summary", "standard_nca_from_reported_sampling"}:
                case_notes.append(comparison.APPROXIMATIONS[key])
        if case["native_missing"]:
            case_notes.append("Native P unavailable: "+", ".join(comparison.ENDPOINTS[k] for k in case["native_missing"])+"; the saved numerical-tail gate failed. P* is retained separately, without filling native P.")
        if case["endpoint_protocol"].get("mrt_infusion_correction") == "none":
            case_notes.append("P* MRT/Vss use the original study convention without infusion correction; native P retains its saved infusion-corrected definition.")
        for n, text in enumerate(case_notes):
            tex.append(note(comparison.escape(text), keep=n<len(case_notes)-1))
    return "\n".join(tex+[r"\end{longtable}\endgroup", ""])


def build(input_path=DEFAULT_INPUT):
    input_path = comparison.inside(input_path)
    data = comparison.read_json(input_path)
    cases, excluded = comparison.qualify_document(data)
    if excluded or len(cases) != 10:
        raise ValueError(f"Expected all ten frozen clinical cases, exclusions: {excluded}")
    raw = {c["case_id"]: c for c in data["cases"]}
    summary_path = input_path.parent/"summary.json"
    run = comparison.read_json(summary_path)
    configs = {c["case_id"]: c for c in run["protocol"]["cases"]}
    source_hashes = {}
    def track(path):
        path = comparison.inside(path)
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    track(input_path)
    track(summary_path)
    track(Path(__file__).resolve())
    rows = []
    for case in cases:
        cid = case["case_id"]
        if case["prediction_issues"].get("ours"):
            raise ValueError(f"Incomplete Ours prediction: {cid}")
        config = configs[cid]
        reported_path = comparison.source(raw[cid]["predictions"]["ours"])
        reported = comparison.read_json(track(reported_path))
        native_curve_path = comparison.source(reported["native_source"])
        curve = comparison.read_json(track(native_curve_path))
        native_path = native_curve_path.parent/"endpoints.json"
        native = comparison.read_json(track(native_path))
        physchem = comparison.read_json(track(ROOT/config["physchem"]))
        mw = physchem["identity"]["molecular_weight_g_mol"]
        policy = raw[cid]["nca_policy"]
        track(comparison.source(raw[cid]["clinical_reference"]["primary_source"]))
        scalar_native_audit(native, curve, mw, policy["dose_ng"]/1e6)
        if native.get("endpoint_estimator_uses_observations") is not False or reported.get("observed_values_used_for_endpoint_estimation") is not False:
            raise ValueError("Endpoint estimator must not use evaluation observations")
        equal(np.interp(reported["curve"]["time_h"], np.asarray(curve["time_min"])/60, np.asarray(curve["concentration"])*mw), reported["curve"]["concentration"], cid+" reported/native unit conversion")
        case["native_missing"] = []
        for item in case["rows"]:
            if item["clinical_reference"] is None:
                continue
            key = item["endpoint"]
            value, factor, reason = native_value(native["endpoints"][key], key, item["unit"], policy["weight_kg"])
            if value is None:
                case["native_missing"].append(key)
            pstar, observed = item["ours_prediction"], item["clinical_reference"]
            error = abs(pstar-observed) if key == "tmax" else max(pstar/observed, observed/pstar)
            equal(error, item["ours_error"], cid+key+" error")
            rows.append({"case_id": cid, "drug": case["drug"], "study_id": case["study_id"], "method": "ours_optimized",
                         "endpoint": key, "endpoint_label": item["endpoint_label"], "unit": item["unit"],
                         "predicted_value": value, "evaluation_prediction": pstar, "observed_value": observed,
                         "fold_error": None if key == "tmax" else error,
                         "absolute_error_h": error if key == "tmax" else None,
                         "matched_weak_tail": item["ours_weak_tail_estimate"], "native_blank_reason": reason,
                         "native_definition": native["endpoints"][key].get("definition", reason),
                         "matched_definition": item["prediction_definition"], "clinical_definition": item["definition"],
                         "clinical_statistic": item["statistic"], "clinical_matrix": item["matrix"],
                         "clinical_source": item["clinical_source"], "clinical_source_sha256": item["clinical_source_sha256"],
                         "clinical_location": item["clinical_location"], "citation": item["citation"],
                         "native_endpoint_source": str(native_path.relative_to(ROOT)), "reported_source": str(reported_path.relative_to(ROOT)),
                         "native_to_reference_unit_factor": factor, "display_unit": item["display_unit"], "display_value_factor": item["display_value_factor"],
                         "native_window_start_h": native["main_window"]["start_h"], "native_window_end_h": native["main_window"]["end_h"],
                         "native_tail_start_h": native["diagnostics"]["terminal_window_h"][0], "native_tail_end_h": native["diagnostics"]["terminal_window_h"][1],
                         "matched_last_quantified_h": reported["nca_diagnostics"]["tlast_h"],
                         "curve_gmfe": case["curve_metrics"]["ours"]["gmfe"], "curve_ln_rmse": case["curve_metrics"]["ours"]["ln_rmse"], "curve_n": case["curve_metrics"]["ours"]["n"]})
        request = config["target_request_metadata"]
        human, dosing = request["human"], request["dosing"]
        start, end = (native["main_window"][key] for key in ("start_h", "end_h"))
        tail = native["diagnostics"]["terminal_window_h"]
        context = (f"Dose {dosing['dose_mg']:g} mg; infusion {dosing['infusion_minutes']:g} min; "
                   f"{human['sex'][0]} {human['age_years']:g} y, {human['weight_kg']:g} kg, {human['height_cm']:g} cm. "
                   f"P window {start:g}-{end:g} h; native tail {tail[0]:.4g}-{tail[1]:.4g} h; P* last quantified {reported['nca_diagnostics']['tlast_h']:g} h.")
        if policy.get("auc_window_end_h") is not None:
            context += f" Scored AUC: 0-{policy['auc_window_end_h']:g} h."
        if policy.get("c_last_time_h") is not None:
            context += f" Scored concentration: {policy['c_last_time_h']:g} h."
        case["native_context"] = context
    if len(rows) != 80 or sum(c["curve_metrics"]["ours"]["n"] for c in cases) != 104:
        raise ValueError("Clinical coverage changed; review the manuscript table specification")
    text = table_tex(cases, rows)
    if re.search(r"[\u3400-\u9fff]", text):
        raise ValueError("English-only table contains CJK text")
    tex_path, csv_path = (DEST/("blueprint_clinical_metrics"+suffix) for suffix in (".tex", ".csv"))
    tex_path.write_text(text)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    audit = {"schema_version": "current_clinical_manuscript_table_v1", "method": "ours_optimized", "input": str(input_path.relative_to(ROOT)),
             "clinical_cases": len(cases), "endpoint_rows": len(rows), "reference_values": len(rows),
             "native_prediction_values": sum(r["predicted_value"] is not None for r in rows),
             "matched_prediction_values": len(rows), "fold_error_values": sum(r["fold_error"] is not None for r in rows),
             "absolute_error_values": sum(r["absolute_error_h"] is not None for r in rows),
             "weak_matched_error_values": sum(r["matched_weak_tail"] for r in rows), "curve_points": 104,
             "endpoint_counts": dict(Counter(r["endpoint"] for r in rows)),
             "case_endpoint_counts": {c["case_id"]: sum(r["case_id"] == c["case_id"] for r in rows) for c in cases},
             "native_missing": {c["case_id"]: c["native_missing"] for c in cases if c["native_missing"]},
             "study_limitations": {c["case_id"]: {"approximations": c["approximations"], "source_notes": c.get("clinical_reference_quality_notes", []), "missing_original_endpoints": c["missing_clinical_endpoints"]} for c in cases},
             "source_sha256": source_hashes, "output_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (tex_path, csv_path)},
             "native_terminal_interpretation": "Numerical long-trajectory diagnostics; saved gates and missing values preserved. Not clinical protocol estimates and not used for reference errors.",
             "checks": ["source/prediction hashes", "native curve-to-reported curve unit conversion", "native finite-window scalar reconstruction", "native CL/Vz/Vss algebra", "all matched FE/Tmax AE recalculated", "original clinical references only; no imputation"],
             "source_data_changed": False, "new_simulations": 0, "body_font_pt": 10, "note_font_pt": 8.5}
    (DEST/"blueprint_clinical_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({k:audit[k] for k in ("clinical_cases", "endpoint_rows", "native_prediction_values", "matched_prediction_values", "fold_error_values", "absolute_error_values", "weak_matched_error_values", "curve_points")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build(parser.parse_args().input)
