#!/usr/bin/env python3
"""Render Ours versus RF with at least eight original clinical endpoint classes.

Usage::

    runtime/app-env/bin/python -B scripts/render_pbpk_method_comparison.py \
      --input path/to/qualified_comparison.json

Optional paired-stage report::

    runtime/app-env/bin/python -B scripts/render_pbpk_method_comparison.py \
      --baseline-input path/to/frozen_baseline.json \
      --input path/to/optimized_comparison.json

Both inputs use the same schema and are qualified separately. The same cases,
clinical reference records, conditions, NCA policies and approximation declarations
must be retained. RF must reference the same hash-verified prediction or an exactly
equal prediction JSON; an identical explicit failure declaration is also retained.
Changed reference/evaluation protocols or controls abort before replacing a report.
The paired PDF starts with an Ours error comparison (up to ten drugs per overview
page), followed by adjacent baseline/optimized detail pages for each drug. Stage
statistics remain separate in summary.json; CSV adds stage and input provenance.
Missing or weak-tail errors receive no improvement/deterioration color. This is
an exploratory before/after comparison, not a new independent validation set.

Default output: results/tables/pbpk_ours_vs_rf/{comparison.csv,comparison.tex,
comparison.pdf,summary.json}. With no eligible cases, only summary.json is
written. No PNG is retained by default. --preview writes ONE temporary first-
page PNG for visual inspection; delete that file after inspection.

Input schema (old five-method ``blocks`` input is deliberately unsupported)::

    {
      "schema_version": "pbpk_ours_rf_clinical_v1",
      "title": "PBPK predictions against clinical references",
      "cases": [{
        "case_id": "unique_cohort_dose_id", "study_id": "StudyYear",
        "drug": "English drug name", "conditions": "Dose; route; cohort; window",
        "approximations": [],
        "clinical_reference": {
          "case_id": "unique_cohort_dose_id", "study_id": "StudyYear",
          "primary_source": {
            "path": "project/relative/original_article.pdf",
            "sha256": "64 hexadecimal characters", "citation": "DOI, PMID or original clinical report ID"
          },
          "verified_against_primary_source": true,
          "reviewer": "Name or extraction-review identifier",
          "curve": {
            "kind": "clinical_study_reported", "location": "Figure/table + cohort",
            "extraction": "reported_table", "matrix": "plasma",
            "statistic": "arithmetic_mean", "unit": "mg/L",
            "time_h": ["finite increasing hours"],
            "concentration": ["finite nonnegative observed concentrations"]
          },
          "endpoints": {
            "auc": {
              "value": "finite number", "unit": "mg*h/L",
              "definition": "AUC 0-24 h; linear-up/log-down NCA",
              "statistic": "arithmetic_mean", "matrix": "plasma",
              "kind": "clinical_study_reported", "location": "Table X, row Y, cohort Z"
            },
            "...": "at least seven of the ten scalar keys listed below"
          }
        },
        "predictions": {
          "ours": {"path": "project/relative/ours.json", "sha256": "..."},
          "rf": {"path": "project/relative/rf.json", "sha256": "..."}
        }
      }]
    }

Each hash-locked prediction JSON uses schema_version='pbpk_clinical_prediction_v1',
case_id, study_id, method ('ours' or 'rf'), curve, and endpoints. Prediction curve
has time_h, concentration, unit, matrix and statistic; endpoint entries have
value, unit, definition, statistic and matrix. Their identities, units, endpoint
definitions and aggregation statistics must match the clinical reference except
for the explicit, visible approximations described below.
Prediction endpoints must already use the declared clinical definition; this
renderer never refits a model or replaces a missing endpoint.

Optional case.approximations may contain representative_adult_vs_group_summary
and published_mean_last_sampling_time. standard_nca_from_reported_sampling is
also accepted when prediction NCA uses the frozen original sampling schedule
and a predeclared prediction cutoff instead of unavailable original individual
terminal-fit points. If the original LLOQ is unavailable and the reported LOD
is used as a cutoff proxy, also declare lod_used_as_prediction_cutoff; this
never treats a detection limit as a reported quantification limit.
The first flag permits prediction
statistic='representative_adult' versus a published group mean/median. The latter
permits a proxy definition for auc/c_last only: set the prediction endpoint's
definition to its actual proxy calculation and reference_definition to the
original clinical endpoint definition. Both methods must declare the same proxy
definition. The table and summary display these limitations; references remain
unchanged. These are explicitly approximate comparisons, not exact matches of
group NCA statistics or participant-specific last observation times.

The optional serum_as_plasma flag permits only predicted plasma versus reported
serum (never the reverse); the clinical matrix is preserved and the assumption
is displayed. uncertain_mrt_infusion_correction permits a different MRT
prediction definition only when reference_definition quotes the original
clinical definition and the prediction explicitly uses infusion-corrected MRT.
Both methods must still use the same prediction definition. The clinical paper's
unspecified infusion correction is disclosed rather than treated as an exact
endpoint match.

Exact scalar keys: auc, cmax, tmax, c_last, terminal_half_life, auc_inf, clearance,
vz, mrt, vss. AUC is the explicitly stated finite window, not AUC infinity.
At least seven scalars must be reported in the SAME original clinical study/cohort. The curve
is the eleventh endpoint class; it appears as GMFE/ln-RMSE/n in the section bar,
never as an empty scalar row. Curve extraction may be reported_table or
digitized_primary_figure; neither permits replacing unreported PK parameters
with NCA on a digitized mean curve. Individual observations and published NCA
summaries remain distinct through statistic and definition fields.

Provenance checking verifies project boundaries, file hashes, declarations and
case consistency. It is not automatic proof of a paper's contents: upstream
extraction review must actually verify each source location. An invented review
flag is not acceptable. Unqualified cases produce one concise entry each in
summary.json. Eligible clinical cases retain both method columns even when a
prediction is unavailable; failures and missing-value reasons are recorded in
summary.json and shown concisely below their table.
Missing clinical scalar keys may be omitted or expressed as
{"value": null, "reason": "Not reported", "unit": "L/kg"}. An omitted clinical
key means not reported; it never requests NCA reconstruction. Optional metadata
on missing values is retained, and a unit is recommended when both predictions
are unavailable. All ten scalar rows remain visible, with an em dash for absent
values and errors. Clinical curves are mandatory; fewer than seven valid,
original-source-reported scalars excludes a case. Unknown endpoint keys fail.
Prediction endpoints can likewise be omitted or use value=null plus an explicit
English reason. A hash-locked prediction document may instead declare
status='unavailable' or 'failed' and reason. A case.predictions method entry may
also use only {status: 'failed'/'unavailable', reason: '...'} without a path;
this declares absence and cannot supply numeric predictions. Missing, invalid or failed predictions
never remove an otherwise eligible clinical case. Untrusted source values are
withheld and their validation errors are recorded, not silently repaired.
Scientific errors are calculated from the submitted frozen values; predictions
and observations are never invented, repaired, or imputed.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pbpk_ours_rf_clinical_v1"
PREDICTION_SCHEMA = "pbpk_clinical_prediction_v1"
SUMMARY_SCHEMA = "pbpk_ours_rf_render_summary_v1"
DEFAULT_OUTPUT = ROOT / "results/tables/pbpk_ours_vs_rf"
ENDPOINTS = {
    "auc": "AUClast", "cmax": "Cmax", "tmax": "Tmax",
    "c_last": "Clast", "terminal_half_life": "Terminal half-life",
    "auc_inf": "AUC infinity", "clearance": "CL", "vz": "Vz",
    "mrt": "MRT", "vss": "Vss",
}
TEX_ENDPOINTS = {
    "auc": r"AUC$_{\mathrm{last}}$", "cmax": r"$C_{\max}$", "tmax": r"$T_{\max}$",
    "c_last": r"$C_{\mathrm{last}}$", "terminal_half_life": "Terminal half-life",
    "auc_inf": r"AUC$_\infty$", "clearance": "CL", "vz": r"$V_z$",
    "mrt": "MRT", "vss": r"$V_{\mathrm{ss}}$",
}
STATISTICS = {"individual", "arithmetic_mean", "geometric_mean", "median"}
APPROXIMATIONS = {
    "covariate_matched_adult_vs_individual": "The simulated adult matches the reported patient's age, sex, weight and height; remaining physiology is population-based. Individual measurements are retained, and patient-specific physiology or concomitant treatment is not fitted.",
    "assay_calibration_lower_bound_as_cutoff": "The cited assay's calibration-range lower bound is used as a declared prediction cutoff proxy; the original study did not report a numerical LLOQ. This is not a verified quantification limit.",
    "lod_used_as_prediction_cutoff": "NCA uses the reported detection limit as a predeclared prediction cutoff proxy because the original LLOQ is unavailable; it is not a reported quantification limit.",
    "serum_as_plasma": "The clinical source reports serum; predictions are plasma concentrations. Serum and plasma are assumed comparable for this evaluation. The original clinical matrix and values are unchanged.",
    "uncertain_mrt_infusion_correction": "The clinical report does not specify whether its reported MRT was corrected for infusion duration. Predicted MRT uses an infusion correction; this comparison is approximate and does not establish identical NCA definitions.",
    "representative_adult_vs_group_summary": "Both predictions represent one adult; the clinical reference is a group summary. Individual-NCA means and peaks of a group-mean curve are different quantities, so these comparisons are approximate.",
    "published_mean_last_sampling_time": "AUC(window) and Clast predictions use the reported mean last sampling time as a representative proxy; individual clinical last sampling times vary. Original reported reference values are unchanged.",
    "standard_nca_from_reported_sampling": "Original individual terminal-fit points are unavailable. Prediction endpoints use frozen standard NCA on the reported sampling schedule with a predeclared prediction cutoff; the original individual NCA is not exactly reproduced.",
}
MIN_REPORTED_SCALARS = 7
GENERATED = ("comparison.csv", "comparison.tex", "comparison.pdf", "preview.png")


class Ineligible(ValueError):
    """A case may not enter the main clinical comparison."""


def inside(path: str | Path) -> Path:
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise Ineligible("Source/output path is outside this project")
    return path


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise Ineligible("Expected a JSON object")
    return data


def text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Ineligible(f"Missing text: {field}")
    return value.strip()


def english(value: object, field: str) -> str:
    value = text(value, field)
    if re.search(r"[\u3400-\u9fff]", value):
        raise Ineligible(f"Use English display text: {field}")
    return value


def finite(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Ineligible(f"Missing/nonfinite numeric value: {field}")
    value = float(value)
    if value < 0 or (positive and value == 0):
        raise Ineligible(f"Invalid nonpositive value: {field}")
    return value


def source(reference: dict) -> Path:
    if not isinstance(reference, dict):
        raise Ineligible("Missing source provenance")
    path = inside(text(reference.get("path"), "source path"))
    expected = text(reference.get("sha256"), "source SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise Ineligible("Invalid SHA256 declaration")
    if not path.is_file():
        raise Ineligible(f"Source file missing: {path.relative_to(ROOT)}")
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise Ineligible(f"Source hash mismatch: {path.relative_to(ROOT)}")
    return path


def identity(record: dict, case: dict, label: str) -> None:
    if not isinstance(record, dict):
        raise Ineligible(f"Missing {label}")
    if any(record.get(k) != case[k] for k in ("case_id", "study_id")):
        raise Ineligible(f"{label} belongs to another case/study")


def checked_curve(curve: dict, label: str, *, clinical: bool = False) -> dict:
    if not isinstance(curve, dict):
        raise Ineligible(f"Missing {label} curve")
    t, c = curve.get("time_h"), curve.get("concentration")
    if not isinstance(t, list) or not isinstance(c, list) or len(t) != len(c) or len(t) < 2:
        raise Ineligible(f"{label} curve needs at least two paired points")
    t = [finite(v, f"{label} time") for v in t]
    c = [finite(v, f"{label} concentration") for v in c]
    if any(b <= a for a, b in zip(t, t[1:])):
        raise Ineligible(f"{label} curve times are not strictly increasing")
    if sum(v > 0 for v in c) < 2:
        raise Ineligible(f"{label} curve needs two positive measured/scored points")
    for field in ("unit", "matrix", "statistic"):
        text(curve.get(field), f"{label} curve {field}")
    allowed_statistics = STATISTICS if clinical else STATISTICS | {"representative_adult"}
    if curve["statistic"] not in allowed_statistics:
        raise Ineligible(f"Unknown {label} curve aggregation statistic")
    if clinical:
        if curve.get("kind") != "clinical_study_reported":
            raise Ineligible("Clinical curve is not declared as reported observations")
        if curve.get("extraction") not in {"reported_table", "digitized_primary_figure"}:
            raise Ineligible("Clinical curve lacks an accepted original-source extraction")
        text(curve.get("location"), "original curve location")
    return {**curve, "time_h": t, "concentration": c}


def missing_endpoint(reason: str, metadata: dict | None = None) -> dict:
    return {**(metadata or {}), "value": None, "reason": reason}


def checked_endpoints(endpoints: dict, label: str, *, clinical: bool = False) -> dict:
    if not isinstance(endpoints, dict):
        raise Ineligible(f"Missing {label} endpoint mapping")
    unknown = set(endpoints)-set(ENDPOINTS)
    if unknown:
        raise Ineligible(f"Unknown {label} endpoint keys: {','.join(sorted(unknown))}")
    result = {}
    for key in ENDPOINTS:
        if key not in endpoints:
            reason = "Not reported in the clinical source" if clinical else "Prediction endpoint not supplied"
            result[key] = missing_endpoint(reason)
            continue
        item = endpoints[key]
        if not isinstance(item, dict):
            raise Ineligible(f"Invalid {label} endpoint {key}; use a value-null record for a missing endpoint")
        if item.get("value") is None:
            if "value" not in item:
                raise Ineligible(f"Missing value field for {label}/{key}; use explicit null")
            reason = english(item.get("reason"), f"{label}/{key}/missing reason")
            for field in ("unit", "definition", "statistic", "matrix"):
                if field in item:
                    text(item[field], f"{label}/{key}/{field}")
            result[key] = missing_endpoint(reason, item)
            continue
        value = finite(item.get("value"), f"{label}/{key}", positive=key != "tmax")
        for field in ("unit", "definition", "statistic", "matrix"):
            text(item.get(field), f"{label}/{key}/{field}")
        allowed_statistics = STATISTICS if clinical else STATISTICS | {"representative_adult"}
        if item["statistic"] not in allowed_statistics:
            raise Ineligible(f"Unknown statistic for {label}/{key}")
        if clinical:
            if item.get("kind") != "clinical_study_reported":
                raise Ineligible(f"Clinical {key} is not reported by the original study")
            text(item.get("location"), f"original clinical {key} location")
        result[key] = {**item, "value": value}
    if clinical:
        count = sum(item["value"] is not None for item in result.values())
        if count < MIN_REPORTED_SCALARS:
            raise Ineligible(f"Clinical coverage is {count+1}/11 classes; a measured curve and at least {MIN_REPORTED_SCALARS} original-source-reported scalars are required")
    return result


def interpolate(t: list[float], c: list[float], x: float) -> float:
    index = bisect.bisect_left(t, x)
    if index < len(t) and t[index] == x:
        return c[index]
    if index == 0 or index == len(t):
        raise Ineligible("Prediction curve does not cover the clinical sampling window")
    return c[index-1] + (c[index]-c[index-1])*(x-t[index-1])/(t[index]-t[index-1])


def statistic_matches(predicted: str, observed: str, approximations: set[str]) -> bool:
    return predicted == observed or (
        "representative_adult_vs_group_summary" in approximations
        and predicted == "representative_adult"
        and observed in {"arithmetic_mean", "geometric_mean", "median"}
    ) or ("covariate_matched_adult_vs_individual" in approximations
          and predicted == "representative_adult" and observed == "individual")


def matrix_matches(predicted: str, observed: str, approximations: set[str]) -> bool:
    return predicted == observed or (
        "serum_as_plasma" in approximations and predicted == "plasma" and observed == "serum"
    )


def curve_metrics(prediction: dict, clinical: dict, approximations: set[str]) -> dict:
    if prediction["unit"] != clinical["unit"]:
        raise Ineligible("Prediction/clinical curve unit differs")
    if not matrix_matches(prediction["matrix"], clinical["matrix"], approximations):
        raise Ineligible("Prediction/clinical curve matrix differs without a declared approximation")
    if not statistic_matches(prediction["statistic"], clinical["statistic"], approximations):
        raise Ineligible("Prediction/clinical curve statistic differs without a declared approximation")
    differences = []
    for time, observed in zip(clinical["time_h"], clinical["concentration"]):
        predicted = interpolate(prediction["time_h"], prediction["concentration"], time)
        if observed == 0:
            continue  # Predose zero is preserved, but a logarithmic ratio is undefined.
        if predicted <= 0:
            raise Ineligible("Nonpositive prediction at a positive clinical concentration")
        differences.append(math.log(predicted)-math.log(observed))
    gmfe = math.exp(math.fsum(abs(x) for x in differences)/len(differences))
    return {"gmfe": gmfe, "ln_rmse": math.sqrt(math.fsum(x*x for x in differences)/len(differences)),
            "n": len(differences), "observed_points": len(clinical["time_h"]),
            "zero_observations_not_log_scored": len(clinical["time_h"])-len(differences)}


def qualify_case(case: dict) -> dict:
    for field in ("case_id", "study_id", "drug", "conditions"):
        english(case.get(field), field)
    declared_approximations = case.get("approximations", [])
    if not isinstance(declared_approximations, list) or any(not isinstance(x, str) or x not in APPROXIMATIONS for x in declared_approximations):
        raise Ineligible("Unknown comparison approximation declaration")
    approximations = set(declared_approximations)
    source_notes = case.get("clinical_reference_quality_notes", [])
    if not isinstance(source_notes, list):
        raise Ineligible("Clinical reference quality notes must be a list")
    source_notes = [english(note, "clinical reference quality note") for note in source_notes]
    clinical = case.get("clinical_reference")
    identity(clinical, case, "Clinical reference")
    primary = source(clinical.get("primary_source"))
    citation = text(clinical["primary_source"].get("citation"), "clinical primary citation")
    if clinical.get("verified_against_primary_source") is not True:
        raise Ineligible("Clinical reference has not been verified against the original paper")
    text(clinical.get("reviewer"), "clinical extraction reviewer")
    curve = checked_curve(clinical.get("curve"), "clinical", clinical=True)
    endpoints = checked_endpoints(clinical.get("endpoints"), "clinical", clinical=True)
    for item in (curve, *endpoints.values()):
        for field in ("case_id", "study_id"):
            if field in item and item[field] != case[field]:
                raise Ineligible(f"Clinical curve/endpoint has a conflicting {field}")
    declared = case.get("predictions")
    if not isinstance(declared, dict) or set(declared)-{"ours", "rf"}:
        raise Ineligible("Prediction mapping supports only ours and rf")
    predictions, metrics, curve_statistics, nca_quality, issues = {}, {}, {}, {}, {}
    empty_metrics = {"gmfe": None, "ln_rmse": None, "n": None,
                     "observed_points": len(curve["time_h"]),
                     "zero_observations_not_log_scored": sum(v == 0 for v in curve["concentration"])}
    for method in ("ours", "rf"):
        method_issues = {}
        metrics[method] = dict(empty_metrics)
        try:
            declaration = declared.get(method)
            if isinstance(declaration, dict) and declaration.get("status") in {"failed", "unavailable"}:
                raise Ineligible(english(declaration.get("reason"), method+" failure reason"))
            prediction = read_json(source(declaration))
            if prediction.get("schema_version") != PREDICTION_SCHEMA or prediction.get("method") != method:
                raise Ineligible(f"Unsupported/wrong prediction schema or method: {method}")
            identity(prediction, case, method)
            if prediction.get("status") in {"failed", "unavailable"}:
                raise Ineligible(english(prediction.get("reason"), method+" failure reason"))
            pe = checked_endpoints(prediction.get("endpoints"), method)
            for key in ENDPOINTS:
                item, reference = pe[key], endpoints[key]
                if item["value"] is None:
                    method_issues[key] = item["reason"]
                    continue
                if "representative_adult_vs_group_summary" in approximations and item["statistic"] != "representative_adult":
                    raise Ineligible(f"{method}/{key}: representative-adult declaration conflicts with prediction statistic")
                if reference.get("unit") is not None and item["unit"] != reference["unit"]:
                    raise Ineligible(f"{method}/{key}: clinical unit differs")
                if reference.get("matrix") is not None and not matrix_matches(item["matrix"], reference["matrix"], approximations):
                    raise Ineligible(f"{method}/{key}: clinical matrix differs without a declared approximation")
                if key == "tmax" and item["unit"] != "h":
                    raise Ineligible("Tmax must be normalized to hours")
                if reference["value"] is None:
                    continue
                if not statistic_matches(item["statistic"], reference["statistic"], approximations):
                    raise Ineligible(f"{method}/{key}: aggregation statistic differs without a declared approximation")
                if item["definition"] != reference["definition"]:
                    permitted_proxy = (key in {"auc", "c_last"}
                                       and "published_mean_last_sampling_time" in approximations
                                       and item.get("reference_definition") == reference["definition"])
                    permitted_mrt = (key == "mrt"
                                     and "uncertain_mrt_infusion_correction" in approximations
                                     and item["definition"] == "Infusion-corrected MRT from predicted NCA"
                                     and item.get("reference_definition") == reference["definition"])
                    if not (permitted_proxy or permitted_mrt):
                        raise Ineligible(f"{method}/{key}: clinical definition differs")
            predictions[method] = pe
            try:
                pc = checked_curve(prediction.get("curve"), method)
                if "representative_adult_vs_group_summary" in approximations and pc["statistic"] != "representative_adult":
                    raise Ineligible(f"{method}: representative-adult declaration conflicts with curve statistic")
                metrics[method] = curve_metrics(pc, curve, approximations)
                curve_statistics[method] = pc["statistic"]
            except (Ineligible, OverflowError) as exc:
                method_issues["curve"] = str(exc)
            diagnostics = prediction.get("nca_diagnostics", {})
            quality_keys = ("auc_extrapolated_fraction", "aumc_extrapolated_fraction", "terminal_span_in_half_lives")
            if isinstance(diagnostics, dict) and all(isinstance(diagnostics.get(k), (int, float)) and not isinstance(diagnostics[k], bool) and math.isfinite(diagnostics[k]) for k in quality_keys):
                quality = {k: diagnostics[k] for k in quality_keys}
                weak_slope = quality["terminal_span_in_half_lives"] < 1
                weak_auc = weak_slope or quality["auc_extrapolated_fraction"] > 0.2
                weak_moment = weak_auc or quality["aumc_extrapolated_fraction"] > 0.2
                quality["weak_endpoints"] = (
                    (["terminal_half_life"] if weak_slope else [])
                    + (["auc_inf", "clearance", "vz"] if weak_auc else [])
                    + (["mrt", "vss"] if weak_moment else []))
                quality["weak_tail"] = bool(quality["weak_endpoints"])
                if isinstance(diagnostics.get("tlast_h"), (int, float)) and math.isfinite(diagnostics["tlast_h"]):
                    quality["tlast_h"] = diagnostics["tlast_h"]
                nca_quality[method] = quality
        except (Ineligible, OSError, json.JSONDecodeError, OverflowError) as exc:
            reason = str(exc)
            predictions[method] = {key: missing_endpoint(reason) for key in ENDPOINTS}
            method_issues = {"method": reason}
        issues[method] = method_issues
    if len(curve_statistics) == 2 and curve_statistics["ours"] != curve_statistics["rf"]:
        for method in ("ours", "rf"):
            metrics[method] = dict(empty_metrics)
            issues[method]["curve"] = "Ours/RF curves use different aggregation statistics"
    rows = []
    policy = case.get("nca_policy", {})
    for key, label in ENDPOINTS.items():
        if key == "auc" and policy.get("auc_window_end_h") is not None:
            end = finite(policy["auc_window_end_h"], "AUC window end", positive=True)
            label = f"AUC (0-{end:g} h)"
        if key == "c_last" and policy.get("c_last_time_h") is not None:
            end = finite(policy["c_last_time_h"], "Concentration time", positive=True)
            label = f"C ({end:g} h)"
        reference = endpoints[key]
        available = [predictions[m][key] for m in ("ours", "rf") if predictions[m][key]["value"] is not None]
        if len(available) == 2:
            mismatch = [field for field in ("unit", "matrix", "definition", "statistic") if available[0][field] != available[1][field]]
            if mismatch:
                reason = "Ours/RF prediction " + "/".join(mismatch) + " differs"
                for method in ("ours", "rf"):
                    predictions[method][key] = missing_endpoint(reason, predictions[method][key])
                    issues[method][key] = reason
                available = []
        metadata = available[0] if available else {}
        row = {"endpoint": key, "endpoint_label": label,
               "unit": reference.get("unit", metadata.get("unit", "—")),
               "clinical_reference": reference["value"],
               "clinical_reference_available": reference["value"] is not None,
               "clinical_reference_missing_reason": reference.get("reason", ""),
               "error_metric": "absolute_error_h" if key == "tmax" else "fold_error",
               "definition": reference.get("definition", ""), "statistic": reference.get("statistic", ""),
               "prediction_definition": metadata.get("definition", ""),
               "prediction_statistic": metadata.get("statistic", ""),
               "prediction_matrix": metadata.get("matrix", ""),
               "matrix": reference.get("matrix", metadata.get("matrix", "")),
               "clinical_source": str(primary.relative_to(ROOT)),
               "clinical_source_sha256": clinical["primary_source"]["sha256"],
               "citation": citation, "clinical_location": reference.get("location", "")}
        if key == "tmax" and reference["value"] is not None and reference["unit"] != "h":
            raise Ineligible("Tmax must be normalized to hours in source-bound clinical and prediction records")
        for method in ("ours", "rf"):
            item = predictions[method][key]
            p, o = item["value"], reference["value"]
            error = None
            if p is not None and o is not None:
                error = abs(p-o) if key == "tmax" else max(p/o, o/p)
                if not math.isfinite(error):
                    raise Ineligible(f"{method}/{key}: error is not finite")
            row[method+"_prediction"], row[method+"_error"] = p, error
            row[method+"_missing_reason"] = item.get("reason", "") if p is None else ""
            row[method+"_weak_tail_estimate"] = bool(p is not None and
                key in nca_quality.get(method, {}).get("weak_endpoints", []))
        # Use readable mass units for high-concentration drugs in the PDF;
        # canonical source/prediction values and errors stay unchanged in CSV.
        cmax_reference = endpoints["cmax"].get("value")
        mass_display = (endpoints["cmax"].get("unit") == "ng/mL"
                        and cmax_reference is not None and cmax_reference >= 10000)
        scaled_units = {"ng/mL": "mg/L", "ng.h/mL": "mg.h/L"}
        row["display_unit"] = scaled_units[row["unit"]] if mass_display and row["unit"] in scaled_units else row["unit"]
        row["display_value_factor"] = 0.001 if row["display_unit"] != row["unit"] else 1.0
        rows.append(row)
    reported_count = sum(item["value"] is not None for item in endpoints.values())
    return {**{k: case[k] for k in ("case_id", "study_id", "drug", "conditions")},
            "rows": rows, "curve_metrics": metrics, "nca_quality": nca_quality,
            "clinical_reference_classes": reported_count+1,
            "missing_clinical_endpoints": {key: item["reason"] for key, item in endpoints.items() if item["value"] is None},
            "prediction_issues": {method: detail for method, detail in issues.items() if detail},
            "clinical_tlast_h": clinical.get("additional_reported_values", {}).get("tlast_h", {}).get("mean"),
            "approximations": sorted(approximations),
            "clinical_reference_quality_notes": source_notes,
            "endpoint_protocol": {key: policy[key] for key in
                ("auc_window_end_h", "c_last_time_h", "mrt_infusion_correction") if key in policy},
            "clinical_source": str(primary.relative_to(ROOT)), "citation": citation}


def qualify_document(data: dict) -> tuple[list[dict], list[dict]]:
    if data.get("schema_version") != SCHEMA:
        raise ValueError(f"Unsupported input schema. Required {SCHEMA}; old five-method/blocks inputs are rejected.")
    if not isinstance(data.get("cases"), list):
        raise ValueError("Input cases must be a list")
    english(data.get("title", "PBPK predictions against clinical references"), "title")
    qualified, excluded, identities = [], [], set()
    for index, case in enumerate(data["cases"]):
        case_id = case.get("case_id", f"case_{index+1}") if isinstance(case, dict) else f"case_{index+1}"
        try:
            if not isinstance(case, dict):
                raise Ineligible("Case must be a JSON object")
            english(case_id, "case_id")
            if case_id in identities:
                raise ValueError(f"Duplicate case_id: {case_id}")
            identities.add(case_id)
            qualified.append(qualify_case(case))
        except (Ineligible, OSError, json.JSONDecodeError, OverflowError) as exc:
            excluded.append({"case_id": case_id, "reason": str(exc)})
    return qualified, excluded


def stage_pairs(baseline: dict, optimized: dict, baseline_cases: list[dict],
                optimized_cases: list[dict]) -> list[dict]:
    """Pair frozen evaluations without treating two stages as extra drugs."""
    if any(not isinstance(case, dict) or not isinstance(case.get("case_id"), str)
           for document in (baseline, optimized) for case in document["cases"]):
        raise ValueError("Paired stage records require explicit case IDs, including excluded cases")
    raw = [{case["case_id"]: case for case in document["cases"]}
           for document in (baseline, optimized)]
    if raw[0].keys() != raw[1].keys():
        raise ValueError("Stage case IDs differ; paired reporting requires the same frozen cases")
    controls = {}
    for case_id, before in raw[0].items():
        after = raw[1][case_id]
        for field in ("case_id", "study_id", "drug", "conditions", "clinical_reference",
                      "clinical_reference_quality_notes", "target_request_metadata",
                      "evaluation_input_metadata", "nca_policy"):
            if before.get(field) != after.get(field):
                raise ValueError(f"{case_id}: stage {field} changed; evaluations cannot be paired")
        if not isinstance(before.get("nca_policy"), dict) or not before["nca_policy"]:
            raise ValueError(f"{case_id}: paired reporting requires an explicit frozen NCA policy")
        if sorted(before.get("approximations", [])) != sorted(after.get("approximations", [])):
            raise ValueError(f"{case_id}: stage approximation declarations changed")
        if any(not isinstance(case.get("predictions"), dict) for case in (before, after)):
            raise ValueError(f"{case_id}: both stages must declare prediction objects")
        rf = [case["predictions"].get("rf") for case in (before, after)]
        if any(not isinstance(item, dict) for item in rf):
            raise ValueError(f"{case_id}: both stages must declare the RF control")
        if any(item.get("status") in {"failed", "unavailable"} for item in rf):
            if rf[0] != rf[1]:
                raise ValueError(f"{case_id}: RF failure declarations differ between stages")
            controls[case_id] = "identical_explicit_failure"
        else:
            # Verify both hashes even when the paths are equal; paths alone are
            # not proof that the frozen RF control has been reused.
            paths = [source(item) for item in rf]
            if read_json(paths[0]) != read_json(paths[1]):
                raise ValueError(f"{case_id}: RF prediction changed between stages")
            controls[case_id] = ("same_prediction_hash" if rf[0]["sha256"] == rf[1]["sha256"]
                                 else "identical_prediction_content")
    qualified = [{case["case_id"]: case for case in cases}
                 for cases in (baseline_cases, optimized_cases)]
    if qualified[0].keys() != qualified[1].keys():
        raise ValueError("Clinical qualification differs between stages; do not silently drop unmatched cases")
    pairs = []
    for case_id, before in qualified[0].items():
        after = qualified[1][case_id]
        errors = {"curve": error_change(before["curve_metrics"]["ours"]["gmfe"],
                                         after["curve_metrics"]["ours"]["gmfe"])}
        for first, second in zip(before["rows"], after["rows"], strict=True):
            if first["endpoint"] != second["endpoint"]:
                raise ValueError(f"{case_id}: endpoint order changed")
            key = first["endpoint"]
            errors[key] = error_change(first["ours_error"], second["ours_error"],
                                      first["ours_weak_tail_estimate"], second["ours_weak_tail_estimate"])
        pairs.append({"case_id": case_id, "drug": before["drug"], "baseline": before,
                      "optimized": after, "rf_control": controls[case_id], "errors": errors})
    return pairs


def error_change(before: float | None, after: float | None,
                 before_weak: bool = False, after_weak: bool = False) -> dict:
    """Classify like-for-like errors, never impute missing/weak-tail improvements."""
    if before is None or after is None:
        status = "not_comparable"
    elif before_weak or after_weak:
        status = "descriptive_weak_tail"
    elif math.isclose(before, after, rel_tol=1e-6, abs_tol=1e-10):
        status = "unchanged"
    else:
        status = "improved" if after < before else "worsened"
    return {"baseline": before, "optimized": after, "status": status,
            "baseline_weak_tail": before_weak, "optimized_weak_tail": after_weak}


def escape(value: object) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                    "∞": r"$\infty$", "×": r"$\times$", "±": r"$\pm$",
                    "μ": r"$\mu$", "·": r"$\cdot$", "—": "---", "–": "--"}
    return "".join(replacements.get(c, c) for c in str(value))


def number(value: float | None) -> str:
    if value is None:
        return r"\textemdash{}"
    raw = f"{value:.4g}"
    if "e" in raw:
        base, exponent = raw.split("e")
        return rf"\({base}\!\times\!10^{{{int(exponent)}}}\)"
    return rf"\({raw}\)"


def common_notes() -> list[str]:
    return [r"\par\vspace{3mm}{\fontsize{8.5}{10}\selectfont\color{Muted}",
            r"RF+PBPK uses RF-predicted CL and unbound fraction with fixed PK-Sim Standard tissue distribution; RF-predicted Vss is not imposed. Ours additionally uses a separate clinical development curve. This comparison does not isolate a pure algorithm advantage.",
            r"\par Every case has a measured concentration curve and at least seven scalar references reported by the same clinical study/cohort. No missing clinical endpoint is reconstructed by NCA. The curve is the eleventh endpoint class, represented by GMFE, ln-RMSE and $n$.",
            r"\par Errors are fold error, $\max(P/O,O/P)$ (ideal 1), except $T_{\max}$: absolute error in hours (ideal 0). Curve metrics compare predictions interpolated at observed sampling times; $n$ counts positive clinical points. Observed zero concentrations are retained but excluded from logarithmic curve metrics.",
            r"\par \textemdash{}: reference or prediction unavailable; no corresponding error is calculated. Missing predictions retain their method and case. Excluded clinical cases, prediction failures and reasons are recorded in summary.json. Values use four significant figures; CSV retains full precision and explicit display-unit factors.}"]


def overview_tex(pairs: list[dict], page_index: int, total_pages: int) -> list[str]:
    """One compact page for at most ten paired drugs; never rank unlike errors."""
    labels = [r"\shortstack{Curve\\GMFE}", r"\shortstack{AUC\\FE}",
              r"\shortstack{$C_{\max}$\\FE}", r"\shortstack{$T_{\max}$\\AE (h)}",
              r"\shortstack{$C_{\mathrm{last}}$\\FE}", r"\shortstack{$t_{1/2}$\\FE}",
              r"\shortstack{AUC$_\infty$\\FE}", r"\shortstack{CL\\FE}",
              r"\shortstack{$V_z$\\FE}", r"\shortstack{MRT\\FE}",
              r"\shortstack{$V_{\mathrm{ss}}$\\FE}"]
    lines = [r"{\fontsize{13}{15}\selectfont\bfseries Ours: baseline $\rightarrow$ optimized\hfill "
             + f"{page_index+1}/{total_pages}" + r"\par}\vspace{2mm}",
             r"{\fontsize{9}{11}\selectfont Each cell: baseline (top), optimized (bottom). Lower error is better; endpoint dimensions are evaluated separately.\par}",
             r"\vspace{2mm}\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{31mm}*{11}{>{\centering\arraybackslash}X}@{}}",
             r"\rowcolor{Ink}\color{white}\bfseries Drug & "
             + " & ".join(r"\color{white}\bfseries "+label for label in labels) + r"\\\midrule"]
    for index, pair in enumerate(pairs):
        cells = [escape(pair["drug"])]
        for key in ("curve", *ENDPOINTS):
            item = pair["errors"][key]
            before, after = number(item["baseline"]), number(item["optimized"])
            if item["baseline_weak_tail"]:
                before += r"\(^{\dagger}\)"
            if item["optimized_weak_tail"]:
                after += r"\(^{\dagger}\)"
            shade = {"improved": "Improved", "worsened": "Worsened"}.get(item["status"])
            prefix = rf"\cellcolor{{{shade}}}" if shade else ""
            cells.append(prefix+r"\shortstack{"+before+r"\\"+after+"}")
        lines.append((r"\rowcolor{Stripe}" if index % 2 == 0 else "") + " & ".join(cells) + r"\\[1mm]")
    lines += [r"\bottomrule\end{tabularx}\par\vspace{2mm}",
              r"{\fontsize{8.5}{10}\selectfont\color{Muted}",
              r"\colorbox{Improved}{Lower error} \quad \colorbox{Worsened}{Higher error}. White/striped cells: unchanged, unavailable, or descriptive weak-tail comparisons. Color is redundant with the two numeric values and does not encode effect size or statistical significance.",
              r"\par $\dagger$: predicted terminal NCA quality screen not met in that stage; these errors are descriptive and never colored as improvement. Missing references/predictions remain \textemdash. AUC/Clast use the frozen case-specific windows shown on detail pages.",
              r"\par The two stages reuse identical clinical references, conditions, NCA policies and RF controls. Detail pages show only Ours versus RF+PBPK, with each drug's baseline and optimized pages adjacent. Reusing these drugs for optimization gives an exploratory before/after comparison, not an independent confirmatory test. No cross-endpoint composite score is calculated.}"]
    return lines + common_notes()


def make_tex(title: str, cases: list[dict], *, pairs: list[dict] | None = None) -> str:
    lines = [r"\documentclass[10pt]{article}",
             r"\usepackage[paperwidth=297mm,paperheight=210mm,margin=12mm]{geometry}",
             r"\usepackage{fontspec}", r"\setmainfont{texgyretermes-regular.otf}[BoldFont=texgyretermes-bold.otf]",
             r"\usepackage[table]{xcolor}\usepackage{booktabs,array,tabularx,amsmath}",
             r"\definecolor{Ink}{HTML}{233A4B}\definecolor{Faint}{HTML}{EAF0F3}\definecolor{Stripe}{HTML}{F6F9FA}\definecolor{Muted}{HTML}{52616C}",
             r"\definecolor{Improved}{HTML}{E2EEF5}\definecolor{Worsened}{HTML}{F8E8DD}",
             r"\pagestyle{empty}\setlength{\parindent}{0pt}\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.15}",
             r"\begin{document}\fontsize{10}{12}\selectfont"]
    overview_pages = math.ceil(len(pairs)/10) if pairs else 0
    details = ([(stage, pair[stage], index+1) for index, pair in enumerate(pairs)
                for stage in ("baseline", "optimized")] if pairs is not None
               else [(None, case, index+1) for index, case in enumerate(cases)])
    total_pages = overview_pages + len(details)
    for page_index in range(overview_pages):
        if page_index:
            lines.append(r"\newpage")
        lines += overview_tex(pairs[page_index*10:(page_index+1)*10], page_index, total_pages)
    for detail_index, (stage, case, drug_index) in enumerate(details):
        page_index = overview_pages + detail_index
        if page_index:
            lines.append(r"\newpage")
        page_title = title + (f" | {stage.capitalize()}" if stage else "")
        lines.append(r"{\fontsize{13}{15}\selectfont\bfseries " + escape(page_title) + r"\hfill " + f"{page_index+1}/{total_pages}" + r"\par}\vspace{3mm}")
        own, rf = case["curve_metrics"]["ours"], case["curve_metrics"]["rf"]
        metrics = rf"Curve GMFE: Ours {number(own['gmfe'])}; RF {number(rf['gmfe'])}"
        n = own["n"] if own["n"] is not None else rf["n"]
        if n is not None:
            metrics += rf"; $n={n}$"
        lines += [r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{39mm}>{\centering\arraybackslash}p{20mm}*{5}{>{\centering\arraybackslash}X}@{}}",
                  r"\rowcolor{Ink}\multicolumn{7}{@{}p{\linewidth}@{}}{\color{white}\bfseries\strut\hspace{1mm}" + escape(f"{drug_index:02d}. {case['drug']} / {case['study_id']}") + r"\hfill " + metrics + r"\hspace{1mm}}\\",
                  r"\rowcolor{Faint}\multicolumn{7}{@{}p{\linewidth}@{}}{\strut\hspace{1mm}" + escape(case["conditions"]) + r"}\\",
                  r"\rowcolor{Stripe}\textbf{Endpoint} & \textbf{Unit} & \shortstack{Clinical reference\\O} & \shortstack{Ours\\P} & \shortstack{RF+PBPK\\P} & \shortstack{Ours\\error} & \shortstack{RF+PBPK\\error}\\\midrule"]
        for index, row in enumerate(case["rows"]):
            display_unit = row["display_unit"].replace(".h/", "·h/")
            unit = escape(display_unit).replace("*", r"$\cdot$")
            def display_value(key):
                return None if row[key] is None else row[key]*row["display_value_factor"]
            endpoint_label = (TEX_ENDPOINTS[row["endpoint"]] if row["endpoint_label"] == ENDPOINTS[row["endpoint"]]
                              else escape(row["endpoint_label"]))
            cells = [endpoint_label, unit, number(display_value("clinical_reference")),
                     number(display_value("ours_prediction")), number(display_value("rf_prediction")),
                     number(row["ours_error"]), number(row["rf_error"])]
            for method, column in (("ours", 3), ("rf", 4)):
                if row[method+"_weak_tail_estimate"]:
                    cells[column] += r"\(^{\dagger}\)"
                    if row[method+"_error"] is not None:
                        cells[column+2] += r"\(^{\dagger}\)"
            lines.append((r"\rowcolor{Stripe}" if index % 2 == 0 else "") + " & ".join(cells) + r"\\")
        lines += [r"\bottomrule\end{tabularx}\par", r"{\fontsize{8.5}{10}\selectfont\color{Muted}Curve ln-RMSE: Ours " + number(own["ln_rmse"]) + "; RF " + number(rf["ln_rmse"]) + ". Source: " + escape(case["citation"]) + r".\par}"]
        coverage = f"Clinical reference coverage: {case['clinical_reference_classes']}/11 endpoint classes."
        if case["missing_clinical_endpoints"]:
            coverage += " Not reported: " + ", ".join(ENDPOINTS[key] for key in case["missing_clinical_endpoints"]) + "; no reference or error was imputed."
        lines.append(r"{\fontsize{8.5}{10}\selectfont\color{Muted}" + escape(coverage) + r"\par}")
        for note in case.get("clinical_reference_quality_notes", []):
            lines.append(r"{\fontsize{8.5}{10}\selectfont\color{Muted}" + escape("Reference limitation: " + note) + r"\par}")
        policy = case.get("endpoint_protocol", {})
        notes = []
        if "auc_window_end_h" in policy or "c_last_time_h" in policy:
            notes.append("Fixed-window AUC/concentration rows use their stated times and actual predicted values, including below-cutoff values; terminal NCA remains separate.")
        if policy.get("mrt_infusion_correction") == "none":
            notes.append("Reported-study convention: MRT=AUMC/AUC and Vss=CL*MRT without subtracting half the infusion duration; these are not bolus-equivalent moments.")
        if notes:
            lines.append(r"{\fontsize{8.5}{10}\selectfont\color{Muted}" + escape(" ".join(notes)) + r"\par}")
        for method, issue_map in case["prediction_issues"].items():
            label = "Ours" if method == "ours" else "RF"
            grouped = {}
            for key, reason in issue_map.items():
                grouped.setdefault(reason, []).append(ENDPOINTS.get(key, "curve" if key == "curve" else "all predictions"))
            details = "; ".join(", ".join(keys)+": "+reason for reason, keys in grouped.items())
            lines.append(r"{\fontsize{8.5}{10}\selectfont\color{Muted}" + escape(f"{label} unavailable -- {details}.") + r"\par}")
        for approximation in case["approximations"]:
            lines.append(r"{\fontsize{8.5}{10}\selectfont\color{Muted}" + escape(APPROXIMATIONS[approximation]) + r"\par}")
        for method, quality in case["nca_quality"].items():
            label = "Ours" if method == "ours" else "RF"
            note = (f"{label} predicted NCA: AUC extrapolation {100*quality['auc_extrapolated_fraction']:.1f}%; "
                    f"AUMC extrapolation {100*quality['aumc_extrapolated_fraction']:.1f}%; "
                    f"terminal-fit span {quality['terminal_span_in_half_lives']:.2f} half-lives.")
            if quality["weak_tail"]:
                affected = ", ".join(ENDPOINTS[k] for k in quality["weak_endpoints"])
                note += (f" Flagged endpoints: {affected}. The screen checks terminal span >=1 half-life, "
                         "AUC extrapolation <=20% for AUC/CL/volumes, and AUMC extrapolation <=20% additionally for MRT/Vss. Flagged errors are descriptive.")
            marker = r"$\dagger$ " if quality["weak_tail"] else ""
            lines.append(r"{\fontsize{8.5}{10}\selectfont\color{Muted}" + marker + escape(note) + r"\par}")
        if case["clinical_tlast_h"] is not None and all("tlast_h" in case["nca_quality"].get(m, {}) for m in ("ours", "rf")):
            note = (f"Clinical reported mean last time: {case['clinical_tlast_h']:.3g} h; last above-cutoff prediction time: "
                    f"Ours {case['nca_quality']['ours']['tlast_h']:.3g} h; RF {case['nca_quality']['rf']['tlast_h']:.3g} h. "
                    "Predicted AUClast and Clast follow each prediction's retained window; proximity to the cutoff can mask curve error in Clast.")
            lines.append(r"{\fontsize{8.5}{10}\selectfont\color{Muted}" + escape(note) + r"\par}")
        if stage is None:
            lines += common_notes()
        else:
            lines.append(r"\vspace{2mm}{\fontsize{8.5}{10}\selectfont\color{Muted}"
                         + escape(f"Stage: {stage}. Original references and RF control are unchanged across stages. "
                                  "Errors: fold error (ideal 1); Tmax absolute error in hours (ideal 0). "
                                  "Common definitions and the RF/Ours information difference are given on the overview page.")
                         + r"\par}")
    return "\n".join(lines+[r"\end{document}"]) + "\n"


def write_csv(cases: list[dict], path: Path, *, stages: dict | None = None) -> None:
    rows = []
    records = ([(stage, info, case) for stage, info in stages.items() for case in info["cases"]]
               if stages is not None else [(None, None, case) for case in cases])
    for stage, info, case in records:
        for row in case["rows"]:
            metrics = case["curve_metrics"]
            provenance = ({"stage": stage, "stage_input": info["input"],
                           "stage_input_sha256": info["input_sha256"]} if stage else {})
            rows.append({**provenance, "case_id": case["case_id"], "study_id": case["study_id"], "drug": case["drug"],
                         **row, "ours_curve_gmfe": metrics["ours"]["gmfe"], "rf_curve_gmfe": metrics["rf"]["gmfe"],
                         "ours_curve_ln_rmse": metrics["ours"]["ln_rmse"], "rf_curve_ln_rmse": metrics["rf"]["ln_rmse"],
                         "curve_scored_points": metrics["ours"]["n"],
                         "rf_curve_scored_points": metrics["rf"]["n"]})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def compile_pdf(tex: Path, expected_pages: int, *, preview: bool) -> dict:
    import pymupdf
    wrapper = ROOT / "runtime/latex/tectonic-project"
    completed = subprocess.run([str(wrapper), "--keep-logs", "--outdir", str(tex.parent), str(tex)],
                               cwd=ROOT, capture_output=True, text=True, timeout=180)
    if completed.returncode:
        raise RuntimeError("LaTeX compilation failed: " + (completed.stdout+completed.stderr)[-2500:])
    log = tex.with_suffix(".log")
    warnings = [line for line in log.read_text(errors="replace").splitlines() if re.search(r"Overfull|Missing character", line)]
    if warnings:
        raise RuntimeError("LaTeX layout failed: " + "; ".join(warnings))
    with pymupdf.open(tex.with_suffix(".pdf")) as document:
        if len(document) != expected_pages:
            raise RuntimeError("Unexpected pagination; table needs a layout adjustment")
        for page in document:
            for word in page.get_text("words"):
                if word[0] < 15 or word[1] < 12 or word[2] > page.rect.width-15 or word[3] > page.rect.height-12:
                    raise RuntimeError("Text clipping detected")
            if any(font[2] == "Type3" for font in page.get_fonts(full=True)):
                raise RuntimeError("Type3 font in PDF")
        if preview:
            document[0].get_pixmap(dpi=200, alpha=False).save(tex.parent/"preview.png")
    for suffix in (".log", ".aux", ".out"):
        tex.with_suffix(suffix).unlink(missing_ok=True)
    return {"pages": expected_pages, "overfull_boxes": 0, "missing_glyphs": 0, "text_clipping": False,
            "table_font_pt": 10, "note_font_pt": 8.5, "temporary_preview_created": preview}


def case_summary(data: dict, cases: list[dict], excluded: list[dict], input_path: Path) -> dict:
    """Single-stage counts only; never feed flattened baseline/optimized cases."""
    return {"input": str(input_path.relative_to(ROOT)),
               "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
               "total_cases": len(data["cases"]), "qualified_cases": len(cases), "excluded_cases": excluded,
               "case_ids": [c["case_id"] for c in cases], "scalar_rows": 10*len(cases),
               "comparison_limitations": {c["case_id"]: [APPROXIMATIONS[a] for a in c["approximations"]] for c in cases if c["approximations"]},
               "predicted_nca_quality": {c["case_id"]: c["nca_quality"] for c in cases if c["nca_quality"]},
               "clinical_reference_coverage": {c["case_id"]: {"available_classes": c["clinical_reference_classes"], "missing_scalars": c["missing_clinical_endpoints"]} for c in cases},
               "clinical_reference_quality_notes": {c["case_id"]: c["clinical_reference_quality_notes"] for c in cases if c["clinical_reference_quality_notes"]},
               "prediction_issues": {c["case_id"]: c["prediction_issues"] for c in cases if c["prediction_issues"]},
               "curve_counts": {
                   "clinical_points": sum(c["curve_metrics"]["ours"]["observed_points"] for c in cases),
                   "positive_clinical_points": sum(c["curve_metrics"]["ours"]["observed_points"]
                       - c["curve_metrics"]["ours"]["zero_observations_not_log_scored"] for c in cases),
                   "scored_points": {m: sum(c["curve_metrics"][m]["n"] or 0 for c in cases)
                                     for m in ("ours", "rf")},
               },
               "prediction_counts": {
                   "method_slots": 2*len(cases),
                   "complete": sum(not c["prediction_issues"].get(m) for c in cases for m in ("ours", "rf")),
                   "failed": sum("method" in c["prediction_issues"].get(m, {}) for c in cases for m in ("ours", "rf")),
                   "partial": sum(bool(c["prediction_issues"].get(m)) and "method" not in c["prediction_issues"][m] for c in cases for m in ("ours", "rf")),
                   "missing_scalar_predictions": sum(r[m+"_prediction"] is None for c in cases for r in c["rows"] for m in ("ours", "rf")),
                   "comparable_scalar_pairs": sum(r[m+"_error"] is not None for c in cases for r in c["rows"] for m in ("ours", "rf")),
               },
               "methods": ["ours", "rf"],
               "gate": "Same clinical case/study; observed curve plus at least seven original-source-reported scalars (8/11 classes); source review and hashes. All ten scalar rows and both methods remain visible. Missing references are not reconstructed; unavailable predictions are not imputed. Finite comparisons require matching units/matrices and matched or explicitly approximate definitions/statistics.",
               "source_content_review": "Verified declarations are supplied by upstream extraction review; renderer checks provenance consistency, not the semantic contents of papers."}


def render(input_path: Path, output: Path, *, preview: bool = False,
           baseline_input: Path | None = None) -> dict:
    input_path, output = inside(input_path), inside(output)
    data = read_json(input_path)
    cases, excluded = qualify_document(data)
    pairs, stages = None, None
    optimized_summary = case_summary(data, cases, excluded, input_path)
    if baseline_input is not None:
        baseline_input = inside(baseline_input)
        baseline_data = read_json(baseline_input)
        baseline_cases, baseline_excluded = qualify_document(baseline_data)
        # All pairing failures are checked before touching existing report files.
        pairs = stage_pairs(baseline_data, data, baseline_cases, cases)
        baseline_summary = case_summary(baseline_data, baseline_cases, baseline_excluded, baseline_input)
        stages = {"baseline": {**baseline_summary, "cases": baseline_cases},
                  "optimized": {**optimized_summary, "cases": cases}}
        summary = {"schema_version": SUMMARY_SCHEMA, "report_mode": "paired_stages",
                   "total_cases": len(baseline_data["cases"]), "qualified_cases": len(pairs),
                   "unique_drugs": len({pair["drug"] for pair in pairs}),
                   "stage_case_records": len(baseline_cases)+len(cases),
                   "scalar_rows": 10*(len(baseline_cases)+len(cases)),
                   "case_ids": [pair["case_id"] for pair in pairs],
                   "methods": ["ours", "rf"],
                   "stages": {"baseline": baseline_summary, "optimized": optimized_summary},
                   "paired_changes": {pair["case_id"]: {"drug": pair["drug"],
                                      "rf_control": pair["rf_control"], "ours_errors": pair["errors"]}
                                      for pair in pairs},
                   "change_counts_by_endpoint": {
                       key: {status: sum(pair["errors"][key]["status"] == status for pair in pairs)
                             for status in ("improved", "worsened", "unchanged", "not_comparable", "descriptive_weak_tail")}
                       for key in ("curve", *ENDPOINTS)},
                   "interpretation": "Paired exploratory before/after on the same clinical cases; stages are not independent samples. No cross-endpoint composite or pooled-stage performance metric. RF control, clinical references, conditions and NCA policy are unchanged."}
    else:
        summary = {"schema_version": SUMMARY_SCHEMA, **optimized_summary}
    summary.update(status="rendering" if cases else "no_qualified_cases", outputs=[])
    # Only replace files previously owned by this schema, not another report.
    previous = output/"summary.json"
    if any((output/name).exists() for name in GENERATED):
        if not previous.is_file() or read_json(previous).get("schema_version") != SUMMARY_SCHEMA:
            raise ValueError("Output contains another report; choose an empty project output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name in GENERATED:
        (output/name).unlink(missing_ok=True)
    try:
        if cases:
            write_csv(cases, output/"comparison.csv", stages=stages)
            tex = output/"comparison.tex"
            tex.write_text(make_tex(data.get("title", "PBPK predictions against clinical references"), cases, pairs=pairs))
            expected_pages = math.ceil(len(pairs)/10)+2*len(pairs) if pairs is not None else len(cases)
            summary["layout"] = compile_pdf(tex, expected_pages, preview=preview)
            summary["status"] = "rendered"
            summary["outputs"] = ["comparison.csv", "comparison.tex", "comparison.pdf"]
            if preview:
                summary["temporary_preview"] = "preview.png"
    except Exception as exc:
        for name in GENERATED:
            (output/name).unlink(missing_ok=True)
        for suffix in (".log", ".aux", ".out"):
            (output/("comparison"+suffix)).unlink(missing_ok=True)
        summary.update(status="render_failed", error=str(exc))
        raise
    finally:
        previous.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline-input", type=Path,
                        help="Frozen baseline with identical cases, references, NCA policies and RF controls; --input is the optimized stage")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", action="store_true", help="Write one temporary first-page PNG; delete after visual inspection")
    args = parser.parse_args()
    try:
        summary = render(args.input, args.output, preview=args.preview, baseline_input=args.baseline_input)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Comparison not produced: {exc}\n")
    print(json.dumps({k: summary[k] for k in ("status", "total_cases", "qualified_cases", "scalar_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
