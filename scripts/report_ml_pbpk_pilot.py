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
        f"This batch attempted {summary['attempted_clinical_predictions']} clinical predictions; {summary['successful_native_predictions']} passed native execution. "
        f"Parameter training contains {split['train_count']} compounds and testing {split['test_count']}, separated by molecular scaffold.",
        "", f"Total elapsed time: {summary['elapsed_seconds'] / 60:.2f} minutes; actual native solves: "
        f"{sum(j['result'].get('native_evaluations', 0) for j in summary['results'].values())}. "
        'This evaluates only construction when target-drug clinical PK parameters are missing. Reliable independent parameter evidence should still take priority.',
        "", 'The model learns CL, Vss, fu, half-life, and MRT only from training data. The three clinical drugs and their scaffolds are excluded from training and parameter testing. '
        'PBPK inputs use predicted CL/fu and independent physicochemical data; mapping additionally matches predicted Vss without fitting clinical curves or endpoints.',
        "", 'Median direct uses training medians; RF direct uses random-forest parameters; RF + Vss mapping maps RF CL/Vss into PBPK.',
        "", '**Current decision: do not expand the clinical experiment yet.** RF parameter prediction beats the median baseline, but the Caffeine curve and terminal concentrations remain inaccurate. '
        'Vss mapping did not further reduce aggregate curve error on common drugs. First improve CL, Vss, and fu prediction and their conversion to PBPK, '
        'while retaining fixed controls and failures. Mapping convergence is not predictive accuracy; repeated tuning on these three inspected cases cannot be called independent validation.',
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
        'Curves are compared at original clinical sampling times. AUC is trapezoidal area over the common sampling window; Cmax is the sampled peak; Clast is the final sample. ', "",
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
        "", 'GMFE is the drug-equally-weighted geometric mean. Common subset: ' + (", ".join(sorted(common_drugs)) or 'None') + "."
        'When methods succeed on different drug counts, their available-case means cannot establish an overall winner; compare common subsets and coverage of all attempts.',
        "", "## Clinical paper-reported endpoint references", "",
        'O denotes PK endpoints reported directly in the publication, not our NCA recalculation of the mean curve. '
        'There are ten verified reference values: six for Midazolam and four for Alfentanil. Every method retains all reference comparisons. '
        'This table has no reported scalar references for Caffeine, but its real curve was scored.',
        "", f"Reported-endpoint comparisons: {len(reported)} total; {sum(r['status'] == 'evaluated' for r in reported)} evaluable, "
        f"{sum(r['status'] == 'descriptive_only' for r in reported)} descriptive only, "
        f"{sum(r['status'] == 'unavailable' for r in reported)} with unavailable predictions.",
        "", 'P denotes the current prediction. Primary Cmax/Tmax comparisons sample predictions at available clinical times; CSV also retains native dense-grid results. '
        'Other endpoints use native model outputs. FE = max(P/O, O/P); Tmax uses absolute time error only.', ""]
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
    lines += ['Native terminal half-life comes from a very late low-concentration model tail; the publication terminal fitting window cannot be matched. '
        'These six comparisons retain predictions, clinical values, and raw fold errors as descriptive_only, excluded from clinical-validation success counts and twofold pass rates. '
        'CSV also retains half-life prediction over the last four clinical times; this does not reproduce the original individual NCA fitting window.', "",
        "## All 11 requested native prediction classes", "",
        'This is a model-output inventory, not a claim of clinical references for every item. A dash indicates no result passing quality checks; missing values are not filled with zero.'
        'Finite-window and last-concentration outputs follow the original request, which may differ from the clinical first-to-last sampling window. Curves remain full numerical series. ', ""]
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
        'Half-life and Vz still depend on very late low-concentration tails; numerical computability is not physiological or clinical credibility. '
        'Scores are not improved by relaxing tail-quality gates or hiding failures.', "",
        'All three drugs are historical development cases. The workflow checks and observed errors establish neither clinical decision capability nor SOTA. Test labels were not changed, failures were not deleted, and parameters were not selected from clinical outcomes.'
        'A representative individual differs from a population mean curve. Concentrating total clearance in the liver is a batch convention, not verified organ attribution. '
        'Complete per-case clinical reference validation of all 11 endpoint types is not finished.', ""]
    if failed:
        lines += [table(["Drug", "Method", "Failure"], [[r["drug"],r["method"], r["result"].get("reason", r["result"]["status"])] for r in failed]), ""]
    lines += [f"Data and split: [parameter_split.json]({run / 'parameter_split.json'}); "
        f"Reported-endpoint comparisons: [reported_endpoint_comparison.csv]({run / 'reported_endpoint_comparison.csv'}); "
        f"All predicted endpoints: [native_endpoint_predictions.csv]({run / 'native_endpoint_predictions.csv'}).", "",
        'Regenerate the report only, without starting experiments:', "", "```bash",
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
