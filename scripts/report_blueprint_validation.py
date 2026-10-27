#!/usr/bin/env python3
"""Score the locked blueprint validation experiment; never fit a model.

Input: run/protocol.json (version blueprint-validation-1, ten cases, source_pins)
and run/summary.json (status predictions_locked, all_attempts_locked_before_scoring,
units indexed case_id__full|five__method). Each terminal unit retains request,
prediction/mapping/fit SHA references when available, resources, and failure.
Sampling full/five refers ONLY to development observations; the original target
clinical sampling/NCA policy stays fixed. All outcomes are decoded after the
complete attempt set and source locks have been checked.

Output: only run/results.json, results.csv and comparison.pdf. All 100 attempts
and 11 endpoint classes remain in the long CSV, including failures. A curve has
no scalar O/P: its row contains GMFE, ln-RMSE and scored points. No clinical
reference is reconstructed from another endpoint or from a model prediction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_clinical_method_comparison import clinical_unit_factor, predicted_nca
from render_pbpk_method_comparison import checked_curve, checked_endpoints, curve_metrics
from summarize_additional_clinical_baselines import prediction_in_hours, weak_endpoints
from pbpk_kb.blueprints.clinical_prediction_nca import UNITS

VERSION = "blueprint-validation-report-1"
SEED, RESAMPLES = 20260929, 2000
METHODS = ("ours", "fixed_pbpk", "rf_only", "two_compartment_prior", "two_compartment")
SAMPLING = ("full", "five")
LABELS = {"ours": "Ours", "fixed_pbpk": "Fixed formula", "rf_only": "RF-only fit",
          "two_compartment_prior": "2-comp. + prior", "two_compartment": "2-comp. fit"}
CURVE = "concentration_time_curve"


class ValidationFault(ValueError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


def unit_factor(endpoint, unit, weight):
    try:
        return clinical_unit_factor(endpoint, unit, weight)
    except ValueError as exc:
        raise ValidationFault("unit_error",str(exc)) from exc


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_condition_matrix(protocol):
    """Condition identity is independent of scheduling order; duplicates are invalid."""
    for key, expected in (("methods", METHODS), ("sampling", SAMPLING)):
        declared = protocol.get(key)
        if (not isinstance(declared, list) or not all(isinstance(x,str) for x in declared)
                or len(declared) != len(expected) or len(set(declared)) != len(declared)
                or set(declared) != set(expected)):
            raise ValueError("Unknown or duplicated " + key + " experiment conditions")


def endpoint_error(prediction, observed, endpoint, unit):
    """Tmax uses absolute hours; all other positive scalars use symmetric FE."""
    if prediction is None or observed is None:
        return None, "absolute_error_h" if endpoint == "tmax" else "fold_error"
    if endpoint == "tmax":
        if unit not in {"h", "min"}:
            raise ValueError("Unsupported Tmax error unit: " + unit)
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (prediction, observed)):
            raise ValueError("Tmax must be finite and nonnegative")
        return abs(prediction-observed)/(60 if unit == "min" else 1), "absolute_error_h"
    if not positive(prediction) or not positive(observed):
        raise ValueError("A scalar fold error requires positive finite values")
    return max(prediction/observed, observed/prediction), "fold_error"


def clustered_geomean(values, clusters, *, resamples=RESAMPLES, seed=SEED):
    """Drug-equal estimate; resample whole study clusters, preserving paired drugs.

    `values` and `clusters` share case IDs. Missing/failed values are explicitly
    counted outside this function; CIs condition on available complete cases.
    This is descriptive cluster bootstrap, not an independent-test claim.
    """
    if set(values) != set(clusters) or any(not positive(v) for v in values.values()):
        raise ValueError("Bootstrap requires aligned finite positive case values")
    if not isinstance(resamples, int) or resamples < 2:
        raise ValueError("At least two bootstrap resamples are required")
    keys = sorted(values)
    names = sorted(set(clusters.values()))
    if not keys:
        return {"estimate": None, "ci95": None, "n_cases": 0, "n_clusters": 0}
    logs = np.asarray([math.log(values[k]) for k in keys])
    sums = np.asarray([sum(logs[i] for i,k in enumerate(keys) if clusters[k] == c) for c in names])
    counts = np.asarray([sum(clusters[k] == c for k in keys) for c in names])
    result = {"estimate": math.exp(float(logs.mean())), "n_cases": len(keys), "n_clusters": len(names)}
    if len(names) < 2:
        return {**result, "ci95": None, "ci_reason": "Fewer than two independent study clusters"}
    indices = np.random.default_rng(seed).integers(0, len(names), size=(resamples, len(names)))
    draws = np.exp(sums[indices].sum(axis=1)/counts[indices].sum(axis=1))
    return {**result, "ci95": [float(v) for v in np.quantile(draws, [.025, .975])],
            "resamples": resamples, "seed": seed}


def paired_ratio(numerator, denominator, clusters, *, expected_ids=None, resamples=RESAMPLES, seed=SEED):
    expected = sorted(set(expected_ids if expected_ids is not None else clusters))
    paired = {k: numerator[k]/denominator[k] for k in expected
              if positive(numerator.get(k)) and positive(denominator.get(k))}
    result = clustered_geomean(paired, {k: clusters[k] for k in paired}, resamples=resamples, seed=seed)
    result.update(attempted_pairs=len(expected), complete_pairs=len(paired),
                  incomplete_case_ids=[k for k in expected if k not in paired],
                  per_case_ratio=paired, ratio_below_one=sum(v < 1 for v in paired.values()),
                  ratio_above_one=sum(v > 1 for v in paired.values()),
                  ci_condition="Complete paired cases; failed attempts remain in the stated denominator")
    return result


class Store:
    def __init__(self, root):
        self.root, self.pins = Path(root).resolve(), {}

    def path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Source is outside the project")
        return path

    def checksum(self, name, expected=None):
        path = self.path(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = str(path.relative_to(self.root))
        if expected is not None and actual != expected:
            raise ValueError("Source lock mismatch: " + relative)
        if relative in self.pins and self.pins[relative] != actual:
            raise ValueError("Source changed during reporting: " + relative)
        self.pins[relative] = actual
        return path

    def read(self, name, expected=None):
        return json.loads(self.checksum(name, expected).read_text())

    def ref(self, reference):
        if not isinstance(reference, dict) or not isinstance(reference.get("sha256"), str):
            raise ValueError("A file reference requires its recorded SHA256")
        data = self.read(reference["path"], reference["sha256"])
        pointer = reference.get("json_pointer", "")
        if pointer and not pointer.startswith("/"):
            raise ValueError("Invalid source JSON pointer")
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            data = data[int(token)] if isinstance(data, list) else data[token]
        return data

    def verify_refs(self, value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.checksum(value["path"], value["sha256"])
            else:
                for item in value.values():
                    self.verify_refs(item)
        elif isinstance(value, list):
            for item in value:
                self.verify_refs(item)

    def unchanged(self):
        for name, expected in list(self.pins.items()):
            self.checksum(name, expected)


def _case_reference(config, store):
    document = store.ref(config["reference_source"])
    if "cases" in document:
        matches = [x for x in document["cases"] if x.get("case_id") == config["case_id"]]
        if len(matches) != 1:
            raise ValueError("Original reference must contain one matching case")
        clinical = matches[0]
    else:
        clinical = document
    if any(clinical.get(k) != config[k] for k in ("case_id", "study_id")):
        raise ValueError("Original reference case/study identity differs")
    raw = clinical["clinical_reference"]
    if any(raw.get(k) != config[k] for k in ("case_id", "study_id")):
        raise ValueError("Original clinical observations belong to a different study")
    store.verify_refs(raw)
    curve = checked_curve(raw["curve"], "clinical", clinical=True)
    endpoints = checked_endpoints(raw["endpoints"], "clinical", clinical=True)
    return clinical, curve, endpoints


def cutoff_from_policy(policy, approximations):
    if positive(policy.get("lloq_ng_ml")):
        return policy["lloq_ng_ml"], "reported_lloq"
    options = {
        "reported_lod": ("lod_ng_ml", "lod_used_as_prediction_cutoff", "reported_lod_proxy"),
        "cited_assay_calibration_lower_bound": ("calibration_lower_bound_ng_ml",
            "assay_calibration_lower_bound_as_cutoff", "cited_assay_calibration_lower_bound_proxy")}
    option = options.get(policy.get("prediction_cutoff_proxy"))
    if option is None or option[1] not in approximations or not positive(policy.get(option[0])):
        raise ValueError("A frozen NCA cutoff and its explicit source/proxy are required")
    return policy[option[0]], option[2]


def _score_prediction(config, clinical, reference_curve, endpoints, unit, store):
    native, request = store.ref(unit["prediction"]), store.ref(unit["request"])
    if native.get("unit") != "umol/l" or native.get("time_unit") != "min":
        raise ValidationFault("unit_error", "Native prediction must use min and umol/l")
    if len(native.get("time_min", [])) != len(native.get("concentration", [])) or len(native.get("time_min", [])) < 2:
        raise ValueError("Native curve arrays differ in length or are empty")
    policy = config["nca_policy"]
    expected = ((policy["weight_kg"], request["human"]["weight_kg"]),
                (policy["dose_ng"], request["dosing"]["dose_mg"]*1e6),
                (policy["infusion_h"], request["dosing"]["infusion_minutes"]/60))
    if not all(math.isclose(a,b,rel_tol=1e-12) for a,b in expected) or request["dosing"].get("start_h", 0) != 0:
        raise ValidationFault("parameter_constraint_error", "Frozen NCA dose/weight/infusion differs from the target request")
    physchem = store.read(config["physchem"])
    mw = physchem["identity"]["molecular_weight_g_mol"]
    if not positive(mw):
        raise ValidationFault("parameter_constraint_error", "Invalid molecular weight")
    times, values, merged = prediction_in_hours(native, mw)
    approximations = set(clinical.get("approximations", []))
    cutoff, basis = cutoff_from_policy(policy, approximations)
    result = predicted_nca(policy=policy, time_h=times, concentration_ng_ml=values,
        dose_ng=policy["dose_ng"], infusion_h=policy["infusion_h"],
        nominal_times_h=policy["sampling_time_h"], lloq_ng_ml=cutoff, weight_kg=policy["weight_kg"])
    result["diagnostics"].update(reported_lloq_ng_ml=policy.get("lloq_ng_ml"),
        quantification_threshold_basis=basis, coalesced_float_time_duplicates=merged)
    factor = unit_factor("curve", reference_curve["unit"], policy["weight_kg"])
    curve = checked_curve({"time_h": times, "concentration": [v*factor for v in values],
        "unit": reference_curve["unit"], "matrix": "plasma", "statistic": "representative_adult"}, "prediction")
    try:
        metrics = curve_metrics(curve, reference_curve, approximations)
    except (ValueError, OverflowError) as exc:
        metrics = {"gmfe": None, "reason": str(exc)}
    weak = set(weak_endpoints(result["diagnostics"]))
    predicted = {}
    for endpoint, default_unit in UNITS.items():
        original = endpoints[endpoint]
        unit_name = original.get("unit") or default_unit
        p = result["endpoints"][endpoint]
        value = p["value"]
        if value is not None:
            value *= unit_factor(endpoint, unit_name, policy["weight_kg"])
            if (not positive(value)) and not (endpoint == "tmax" and value == 0):
                raise ValidationFault("parameter_constraint_error", "Invalid converted prediction: " + endpoint)
        error, metric = endpoint_error(value, original["value"], endpoint, unit_name)
        predicted[endpoint] = {"value": value, "unit": unit_name, "error": error,
            "error_metric": metric, "weak_tail": value is not None and endpoint in weak,
            "reason": p.get("reason", "") if value is None else "",
            "reference": original["value"], "reference_reason": original.get("reason", "")}
    diagnostics = result["diagnostics"]
    cl = result["endpoints"]["clearance"]["value"]
    auc = result["endpoints"]["auc_inf"]["value"]
    cl_consistency = None if cl is None or auc is None else math.isclose(
        cl*policy["weight_kg"], policy["dose_ng"]/auc/1000, rel_tol=1e-10, abs_tol=1e-12)
    if cl_consistency is False:
        raise ValidationFault("clearance_consistency_error", "Prediction CL is inconsistent with dose/AUC infinity")
    return {"endpoints": predicted, "curve_metrics": metrics, "nca_status": result["status"],
        "nca_diagnostics": diagnostics, "weak_endpoints": sorted(weak),
        "checks": {"target_dosing_matches_frozen_policy": True, "units_valid": True,
                   "clearance_dose_auc_consistent": cl_consistency},
        "curve": curve}, native, mw


def _renal_scores(config, unit, native, mw, references, store):
    if "renal_prediction" not in unit:
        return []
    from pbpk_kb.blueprints.renal_extension import window_renal_endpoints
    renal = store.ref(unit["renal_prediction"])
    match = [r for r in references.get("cases", []) if r["case_id"] == config["case_id"]]
    if len(match) != 1 or not match[0].get("endpoints"):
        return [{"status": "not_scored", "reason": "No matching original renal endpoint reference"}]
    original = match[0]
    if original["study_id"] != config["study_id"]:
        raise ValueError("Renal source belongs to another clinical study")
    window = original["conditions"]["urine_collection_window_h"]
    if window[0] != 0:
        raise ValueError("The renal diagnostic requires a zero-start source collection window")
    calculated = window_renal_endpoints(renal, native, window[1], mw)
    records = []
    for endpoint, reference in original["endpoints"].items():
        if reference.get("kind") != "original_reported":
            raise ValueError("Renal reference is not an original reported clinical observation")
        if endpoint == "renal_clearance":
            value = calculated["renal_clearance_l_h"]
            if reference["unit"] == "mL/min":
                value *= 1000/60
            elif reference["unit"] != "L/h":
                raise ValueError("Unknown renal clearance unit")
        elif endpoint == "urinary_fraction" and reference["unit"] == "percent_of_administered_dose":
            value = calculated["fraction_percent"]
        else:
            raise ValueError("Unsupported original renal diagnostic")
        error, _ = endpoint_error(value, reference["value"], endpoint, reference["unit"])
        records.append({"endpoint": endpoint, "prediction": value, "reference": reference["value"],
            "unit": reference["unit"], "fold_error": error, "window_h": window,
            "source_location": reference["location"], "scope": "Exploratory renal diagnostic; not a tissue prediction"})
    return records


def _unit_row(record, endpoint, references):
    curve = record.get("curve_metrics", {})
    common = {"unit_id": record["unit_id"], "case_id": record["case_id"], "drug": record["drug"],
        "study_id": record["study_id"], "cluster_id": record["cluster_id"],
        "sampling": record["sampling"], "method": record["method"], "endpoint": endpoint,
        "attempt_status": record["attempt_status"], "scoring_status": record["scoring_status"],
        "failure": record.get("failure", ""), "prediction": None, "clinical_reference": None,
        "unit": "", "error": None, "error_metric": "", "weak_tail": False,
        "prediction_missing_reason": "", "reference_missing_reason": "", "reference_location": "",
        "curve_gmfe": None, "curve_ln_rmse": None, "curve_scored_points": None,"curve_observed_points": None,
        "wall_seconds": record.get("resources", {}).get("wall_seconds"),
        "actual_new_native_calls": record.get("resources", {}).get("actual_new_native_calls"),
        "cached_native_calls": record.get("resources", {}).get("cached_native_calls"),
        "analytic_residual_calls": record.get("resources", {}).get("analytic_residual_calls"),
        "cache_reused": record.get("resources", {}).get("cache_reused"),
        "prediction_path": record.get("prediction_source", {}).get("path", "")}
    if endpoint == CURVE:
        common.update(unit=references.get("curve", {}).get("unit", ""), error_metric="curve_gmfe",
            curve_gmfe=curve.get("gmfe"), curve_ln_rmse=curve.get("ln_rmse"), curve_scored_points=curve.get("n"),
            curve_observed_points=len(references.get("curve",{}).get("time_h",[])) or None,
            prediction_missing_reason=curve.get("reason", record.get("failure", "")),
            reference_location=references.get("curve", {}).get("location", ""))
        return common
    observed = references.get("endpoints", {}).get(endpoint, {})
    estimate = record.get("endpoints", {}).get(endpoint, {})
    common.update(unit=estimate.get("unit") or observed.get("unit") or UNITS[endpoint],
        clinical_reference=observed.get("value"), prediction=estimate.get("value"),
        error=estimate.get("error"), error_metric=estimate.get("error_metric", "absolute_error_h" if endpoint == "tmax" else "fold_error"),
        weak_tail=estimate.get("weak_tail", False),
        prediction_missing_reason=estimate.get("reason", record.get("failure", "")),
        reference_missing_reason=observed.get("reason", "Clinical reference unavailable" if not observed else ""),
        reference_location=observed.get("location", ""))
    return common


def summarize(records, cases):
    by_key = {(r["case_id"], r["sampling"], r["method"]): r for r in records}
    ids = [c["case_id"] for c in cases]
    clusters = {c["case_id"]: c["cluster_id"] for c in cases}

    def scores(sampling, method, subset=ids):
        return {cid: by_key[cid,sampling,method].get("curve_metrics", {}).get("gmfe") for cid in subset}

    def group(subset):
        output = {}
        for sampling in SAMPLING:
            output[sampling] = {}
            for method in METHODS:
                selected = [by_key[cid,sampling,method] for cid in subset]
                finite = {k:v for k,v in scores(sampling, method, subset).items() if positive(v)}
                curve = clustered_geomean(finite, {k: clusters[k] for k in finite})
                curve.update(attempted=len(subset), failed_or_unscorable=len(subset)-len(finite),
                    gmfe_within_twofold=sum(v <= 2 for v in finite.values()),
                    within_twofold_fraction_all_attempts=sum(v <= 2 for v in finite.values())/len(subset) if subset else None,
                    unavailable_case_ids=[k for k in subset if k not in finite])
                endpoints = {}
                for endpoint in UNITS:
                    entries = [r.get("endpoints", {}).get(endpoint, {}) for r in selected]
                    errors = [p["error"] for p in entries if p.get("error") is not None]
                    stat = {"attempted": len(subset), "reference_available": sum(p.get("reference") is not None for p in entries),
                        "predicted": sum(p.get("value") is not None for p in entries), "scored": len(errors),
                        "weak_scored": sum(p.get("error") is not None and p.get("weak_tail", False) for p in entries)}
                    if errors:
                        if endpoint == "tmax":
                            stat.update(mean_absolute_error_h=math.fsum(errors)/len(errors), median_absolute_error_h=float(np.median(errors)))
                        else:
                            stat.update(gmfe=math.exp(math.fsum(map(math.log, errors))/len(errors)), within_twofold=sum(v <= 2 for v in errors))
                    nonweak = [p["error"] for p in entries if p.get("error") is not None and not p.get("weak_tail",False)]
                    stat["nonweak_scored"] = len(nonweak)
                    stat["nonweak_metrics"] = ({"mean_absolute_error_h":math.fsum(nonweak)/len(nonweak)} if endpoint=="tmax" else
                        {"gmfe":math.exp(math.fsum(map(math.log,nonweak))/len(nonweak)),"within_twofold":sum(v <= 2 for v in nonweak)}) if nonweak else None
                    endpoints[endpoint] = stat
                output[sampling][method] = {"curve": curve, "endpoints": endpoints}
        return output

    aggregate = group(ids)
    ratios = {s: {m: paired_ratio(scores(s,"ours"), scores(s,m), clusters, expected_ids=ids)
        for m in METHODS if m != "ours"} for s in SAMPLING}
    degradation = {m: paired_ratio(scores("five",m), scores("full",m), clusters, expected_ids=ids) for m in METHODS}
    sensitivity = {}
    excluded_drugs = ("Nemonoxacin", "Dexmedetomidine", "Midazolam")
    for excluded in [(d,) for d in excluded_drugs]+[excluded_drugs]:
        subset = [c["case_id"] for c in cases if c["drug"] not in excluded]
        sensitivity["exclude_"+"_".join(d.lower() for d in excluded)] = {
            "excluded_drugs": list(excluded), "included_case_ids": subset, "metrics": group(subset),
            "paired_ours_over_comparator": {s:{m:paired_ratio(scores(s,"ours",subset),scores(s,m,subset),
                clusters,expected_ids=subset) for m in METHODS if m != "ours"} for s in SAMPLING}}
    transfer = {kind: {"case_ids": [c["case_id"] for c in cases if c["transfer_group"] == kind],
        "metrics": group([c["case_id"] for c in cases if c["transfer_group"] == kind])}
        for kind in ("primary", "confounded", "unchanged")}
    effective = [c["case_id"] for c in cases if c["effective_evidence_ablation"]]
    identity = [c["case_id"] for c in cases if not c["effective_evidence_ablation"]]
    ablation = {"effective_case_ids": effective, "identity_input_case_ids": identity,
        "interpretation": "Only the four independently changed evidence cases estimate an evidence-input effect; six identical-input cases are not additional ablations.",
        "ours_over_rf_only": {s: paired_ratio(scores(s,"ours",effective), scores(s,"rf_only",effective), clusters,
            expected_ids=effective) for s in SAMPLING}}
    return {"aggregate": aggregate, "paired_ours_over_comparator": ratios, "five_over_full": degradation,
            "prespecified_sensitivity": sensitivity, "transfer": transfer, "evidence_ablation": ablation}


def _resources(records):
    result = {}
    fields = ("wall_seconds", "actual_new_native_calls", "cached_native_calls", "analytic_residual_calls", "manual_interventions", "peak_memory_mb")
    for method in METHODS:
        entries = [r.get("resources", {}) for r in records if r["method"] == method]
        result[method] = {"attempted": len(entries), "cache_reused": sum(e.get("cache_reused") is True for e in entries)}
        for field in fields:
            known = [e[field] for e in entries if isinstance(e.get(field), (int, float)) and not isinstance(e[field], bool)
                     and math.isfinite(e[field]) and e[field] >= 0]
            result[method][field] = {"recorded_units": len(known), "missing_units": len(entries)-len(known),
                "sum": math.fsum(known) if known else None, "median": float(np.median(known)) if known else None,
                "maximum": max(known) if known else None}
    return {"by_method": result, "interpretation": "Only instrumented values are summarized. actual_new_native_calls records cumulative physical native calls for the batch, not calls added by the latest resume. Cached logical calls remain separate; no manual time savings are inferred. Wall seconds sum is not parallel experiment makespan. Peak-memory sum is not an aggregate peak."}


def render_comparison(document, destination):
    """A single vector PDF: case-level scores plus paired, clustered uncertainty."""
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runtime/cache/paper_figures"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    skill = Path("/home/xuyiqian/.codex/skills/scipilot-figure-skill/scripts")
    sys.path.insert(0, str(skill))
    from setup_style import setup_style
    from visual_qa import audit_layout
    setup_style(journal="nature", lang="en", constrained_layout=False)
    for font in (Path(matplotlib.get_data_path())/"fonts/ttf").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size": 7.5, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7.5, "pdf.fonttype": 42})
    cases, records = document["cases"], document["units"]
    by_key = {(r["case_id"],r["sampling"],r["method"]): r for r in records}
    arrays = [np.asarray([[by_key[c["case_id"],s,m].get("curve_metrics",{}).get("gmfe") or np.nan
                          for m in METHODS] for c in cases]) for s in SAMPLING]
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays])
    upper = max(2.0, float(finite.max())) if len(finite) else 2.0
    fig = plt.figure(figsize=(185/25.4, 235/25.4))
    axes = [fig.add_axes([x,y,.29,height]) for y,height in ((.53,.35),(.18,.23)) for x in (.155,.66)]
    cmap = plt.colormaps["YlGnBu"].copy(); cmap.set_bad("#ebedef")
    for ax, sampling, matrix, letter in zip(axes[:2], SAMPLING, arrays, ("A","B")):
        heat = ax.imshow(matrix, cmap=cmap, norm=LogNorm(vmin=1,vmax=upper), aspect="auto", interpolation="nearest")
        ax.set_title(letter+"  "+("Full development data" if sampling == "full" else "Five development points"),loc="left",pad=10)
        ax.set_xticks(range(len(METHODS)),[LABELS[m] for m in METHODS],rotation=48,ha="right")
        ax.set_yticks(range(len(cases)),[c["drug"] for c in cases])
        ax.tick_params(length=0)
        for i,j in np.ndindex(matrix.shape):
            val = matrix[i,j]
            color = "white" if np.isfinite(val) and math.log(val)/math.log(upper)>.58 else "#173e50"
            ax.text(j,i,format(val,".2f") if np.isfinite(val) else "—",ha="center",va="center",fontsize=7,color=color)
        for spine in ax.spines.values(): spine.set_visible(False)
    cb = fig.colorbar(heat, cax=fig.add_axes([.155,.94,.795,.019]),orientation="horizontal")
    cb.ax.xaxis.set_label_position("top");cb.ax.xaxis.tick_top()
    cb.set_label("Clinical curve GMFE (lower is better)",fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.ax.xaxis.set_major_locator(FixedLocator([v for v in (1,2,5,10,20,50,100) if v <= upper]))
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    cb.ax.xaxis.set_minor_formatter(NullFormatter())
    colors, markers = {"full":"#0072B2","five":"#D55E00"},{"full":"o","five":"s"}
    ratio = document["statistics"]["paired_ours_over_comparator"]
    controls = [m for m in METHODS if m != "ours"]
    ax = axes[2]
    for i,m in enumerate(controls):
        for s,dy in (("full",-.13),("five",.13)):
            estimate = ratio[s][m]
            if estimate["estimate"] is None:
                continue
            x,y = estimate["estimate"],i+dy
            ax.plot(x,y,markers[s],color=colors[s],markersize=4,label=s.title() if i==0 else None)
            if estimate["ci95"] is not None:
                ax.hlines(y,*estimate["ci95"],color=colors[s],linewidth=1.3)
    ax.set_yticks(range(len(controls)),[LABELS[m] for m in controls]);ax.invert_yaxis()
    ax.set_xscale("log");ax.axvline(1,color="#5e6c76",linestyle="--",linewidth=.8)
    ax.set_title("C  Paired Ours / comparator",loc="left",pad=10)
    ax.set_xlabel("GMFE ratio; <1 favors Ours")
    ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.27),ncol=2,fontsize=7)
    ax = axes[3]
    for i,m in enumerate(METHODS):
        estimate = document["statistics"]["five_over_full"][m]
        if estimate["estimate"] is None: continue
        ax.plot(estimate["estimate"],i,"o",color="#007f78",markersize=4)
        if estimate["ci95"] is not None: ax.hlines(i,*estimate["ci95"],color="#007f78",linewidth=1.3)
    ax.set_yticks(range(len(METHODS)),[LABELS[m] for m in METHODS]);ax.invert_yaxis()
    ax.set_xscale("log");ax.axvline(1,color="#5e6c76",linestyle="--",linewidth=.8)
    ax.set_title("D  Five / full sampling",loc="left",pad=10)
    ax.set_xlabel("GMFE ratio (>1: degradation)")
    for ax in axes[2:]:
        ax.grid(axis="x",alpha=.18);ax.spines[["top","right"]].set_visible(False)
        lower,upper_limit = ax.get_xlim()
        ax.xaxis.set_major_locator(FixedLocator(np.geomspace(lower,upper_limit,3)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
        ax.minorticks_off()
    fig.text(.035,.07,"Points: drug-equal paired ratios. Lines: 95% study-cluster bootstrap intervals (2,000 resamples).",fontsize=7)
    fig.text(.035,.048,"Pletz 2004 drugs share one cluster. Missing/failed runs remain in CSV and denominators; — unavailable.",fontsize=7)
    fig.text(.035,.026,"Exploratory, previously viewed cohort. Target sampling is unchanged; full/five changes development input only.",fontsize=7)
    fig.canvas.draw()
    issues = audit_layout(fig)
    if any(level == "FAIL" for level,_ in issues):
        plt.close(fig)
        raise ValueError("Figure layout validation failed: " + repr(issues))
    actual_font = min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text())
    if actual_font < 7:
        plt.close(fig)
        raise ValueError("Figure text is smaller than 7 pt")
    # Keep a machine-readable QA record; no per-drug image/report is emitted.
    qa = {"size_mm":[185,235],"minimum_font_pt":actual_font,"layout_issues":[str(x) for x in issues],
          "case_rows":len(cases),"matrix_cells":100,"bootstrap_resamples":RESAMPLES,
          "missing_scores":int(sum(np.isnan(a).sum() for a in arrays))}
    # Outline text glyphs to avoid ambiguous font embedding across PDF readers.
    import cairosvg
    plt.rcParams["svg.fonttype"] = "path"
    svg = destination.with_suffix(".svg")
    try:
        fig.savefig(svg,format="svg")
        cairosvg.svg2pdf(url=str(svg),write_to=str(destination))
    finally:
        svg.unlink(missing_ok=True)
    plt.close(fig)
    return qa


def report(run, project_root=ROOT):
    store = Store(project_root)
    run = store.path(run)
    if not run.is_dir():
        raise ValueError("The prepared validation run directory is required")
    protocol = store.read(run/"protocol.json")
    summary = store.read(run/"summary.json")
    if protocol.get("version") != "blueprint-validation-1":
        raise ValueError("Unknown validation protocol version")
    if summary.get("status") != "predictions_locked" or summary.get("all_attempts_locked_before_scoring") is not True:
        raise ValueError("All attempt outputs must be locked before outcomes may be decoded")
    validate_condition_matrix(protocol)
    if protocol.get("bootstrap",{}).get("seed") != SEED or protocol.get("bootstrap",{}).get("resamples") != RESAMPLES:
        raise ValueError("The bootstrap settings differ from the frozen report policy")
    if not isinstance(protocol.get("source_pins"),dict) or not protocol["source_pins"]:
        raise ValueError("The protocol must lock its source inputs and implementation")
    cases = protocol.get("cases", [])
    if len(cases) != 10 or len({c["case_id"] for c in cases}) != 10:
        raise ValueError("The protocol must retain all ten distinct cases")
    expected = {f'{c["case_id"]}__{s}__{m}':(c,s,m) for c in cases for s in SAMPLING for m in METHODS}
    units = summary.get("units", {})
    if set(units)-set(expected) or any(u.get("status") not in {"predicted","failed"} for u in units.values()):
        raise ValueError("Unknown or unfinished attempts; refusing to decode outcomes")
    errors = []
    for name, checksum in protocol.get("source_pins", {}).items():
        try: store.checksum(name, checksum)
        except (ValueError,OSError) as exc: errors.append({"scope":"protocol","path":name,"reason":str(exc)})
    global_errors = bool(errors)
    unit_errors = {}
    # Verify every unit before the first clinical file is decoded.
    for key,(config,sampling,method) in expected.items():
        unit = units.get(key)
        try:
            if unit is None: raise ValueError("Missing terminal attempt; retained as failed")
            if any(unit.get(k) != v for k,v in (("case_id",config["case_id"]),("sampling",sampling),("method",method))):
                raise ValueError("Unit identifier and metadata disagree")
            if "request" not in unit: raise ValueError("Missing locked target request")
            if unit["status"] == "predicted" and "prediction" not in unit: raise ValueError("Missing locked native prediction")
            if unit["status"] == "predicted":
                required = "fit" if method.startswith("two_compartment") else "mapping"
                if not unit.get(required): raise ValueError("Missing locked fitting provenance: " + required)
            store.verify_refs(unit)
        except (ValueError,OSError,KeyError) as exc:
            unit_errors[key] = str(exc);errors.append({"scope":"unit","unit_id":key,"reason":str(exc)})
    global_errors = bool(errors)
    # Invalid global locks invalidate scoring; keep every attempt, without reading O.
    references, originals = {}, {}
    if not global_errors:
        for config in cases:
            try:
                clinical, curve, endpoints = _case_reference(config,store)
                references[config["case_id"]] = {"curve":curve,"endpoints":endpoints}
                originals[config["case_id"]] = clinical
            except (ValueError,OSError,KeyError) as exc:
                errors.append({"scope":"clinical_reference","case_id":config["case_id"],"reason":str(exc)})
        if any(e["scope"]=="clinical_reference" for e in errors):
            global_errors = True
    renal_references = {}
    if not global_errors and protocol.get("renal_references"):
        try: renal_references = store.ref(protocol["renal_references"])
        except (ValueError,OSError,KeyError) as exc: errors.append({"scope":"renal_reference","reason":str(exc)})
    records, rows = [], []
    for key,(config,sampling,method) in expected.items():
        unit = units.get(key,{})
        cid = config["case_id"]
        record = {"unit_id":key,"case_id":cid,"drug":config["drug"],"study_id":config["study_id"],
            "cluster_id":config["cluster_id"],"sampling":sampling,"method":method,
            "attempt_status":unit.get("status","failed"),"scoring_status":"failed",
            "resources":deepcopy(unit.get("resources",{})),"elapsed_seconds":unit.get("elapsed_seconds"),
            "prediction_source":unit.get("prediction",{}),"request_source":unit.get("request",{}),
            "mapping_source":unit.get("mapping"),"fit_source":unit.get("fit"),"cached_from":unit.get("cached_from")}
        if global_errors:
            record["failure"] = "Invalid protocol/source or clinical-reference integrity; comparison scoring blocked"
        elif key in unit_errors:
            record["failure"] = unit_errors[key]
        elif cid not in originals:
            record["failure"] = "Original clinical reference is invalid or unavailable"
        elif unit.get("status") != "predicted":
            record["failure"] = json.dumps(unit.get("failure","Prediction failed"),ensure_ascii=False)
        else:
            try:
                scored,native,mw = _score_prediction(config, originals[cid], references[cid]["curve"],references[cid]["endpoints"],unit,store)
                record.update(scored)
                record.pop("curve",None)  # Dense curves stay in source artifacts, not duplicated.
                record["scoring_status"] = "scored" if scored["curve_metrics"].get("gmfe") is not None else "partial"
                if unit.get("renal_prediction"):
                    try: record["renal_diagnostics"] = _renal_scores(config,unit,native,mw,renal_references,store)
                    except (ValueError,OSError,KeyError,TypeError) as exc:
                        record["renal_diagnostics"] = [{"status":"failed","reason":str(exc)}]
                        errors.append({"scope":"renal_scoring","unit_id":key,"reason":str(exc)})
            except (ValueError,OSError,KeyError,TypeError,OverflowError) as exc:
                record["failure"] = str(exc);errors.append({"scope":"scoring","unit_id":key,"reason":str(exc),
                    "category":getattr(exc,"category","unclassified_scoring_error")})
        # Preserve reference availability even when execution/scoring failed.
        for endpoint, original in references.get(cid,{}).get("endpoints",{}).items():
            record.setdefault("endpoints",{}).setdefault(endpoint,{"value":None,"reference":original.get("value"),
                "unit":original.get("unit") or UNITS[endpoint],"error":None,"reason":record.get("failure","Unavailable")})
        records.append(record)
        rows.extend(_unit_row(record,e,references.get(cid,{})) for e in (CURVE,*UNITS))
    stats = summarize(records,cases)
    metadata = [{k:c[k] for k in ("case_id","drug","study_id","cluster_id","transfer_group","effective_evidence_ablation")} for c in cases]
    success = {"attempted":len(expected),"native_predicted":sum(r["attempt_status"]=="predicted" for r in records),
        "curve_scored":sum(positive(r.get("curve_metrics",{}).get("gmfe")) for r in records),
        "complete_ten_scalar_predictions":sum(all(r.get("endpoints",{}).get(k,{}).get("value") is not None for k in UNITS) for r in records),
        "failed_or_invalid_unit_ids":[r["unit_id"] for r in records if r["scoring_status"]=="failed"],
        "source_lock_error_count":sum(e["scope"] in {"unit","protocol"} for e in errors),
        "scoring_error_count":sum(e["scope"]=="scoring" for e in errors)}
    assessed = sum("checks" in r for r in records)
    fits = list(summary.get("fits",{}).values())
    tracked_trials = [f["failed_evaluations"] for f in fits if isinstance(f.get("failed_evaluations"),int)]
    intact_units = 0 if any(e["scope"]=="protocol" for e in errors) else len(expected)-len(unit_errors)
    success.update(logical_units=len(expected), planned_unique_prediction_units=protocol.get("engineering_checks",{}).get("unique_prediction_units"),
        unique_prediction_artifacts=len({r["prediction_source"]["path"] for r in records if r["prediction_source"]}),
        structured_unit_source_completeness={"attempted":len(expected),"complete_and_hash_verified":intact_units,
            "rate":intact_units/len(expected),
            "scope":"Required structured request, prediction and fit/mapping references plus source hashes; not automated literature curation"},
        structured_request_preparation_checks={"planned":4*len(cases),
            "passed":protocol.get("engineering_checks",{}).get("request_assembly_checks"),
            "basis":"Prepared protocol is persisted only after every structured request assembles successfully"},
        post_prediction_constraints={"complete_scoring_checks":assessed,
            "explicit_unit_errors":sum(e.get("category")=="unit_error" for e in errors),
            "explicit_parameter_constraint_errors":sum(e.get("category")=="parameter_constraint_error" for e in errors),
            "explicit_clearance_consistency_errors":sum(e.get("category")=="clearance_consistency_error" for e in errors),
            "unclassified_scoring_errors":sum(e.get("category")=="unclassified_scoring_error" for e in errors),
            "clearance_dose_auc_checks_passed":sum(r.get("checks",{}).get("clearance_dose_auc_consistent") is True for r in records)},
        development_failures={"tracked_fit_jobs":len(tracked_trials),"fit_jobs":len(fits),
            "failed_evaluations":sum(tracked_trials) if tracked_trials else None,
            "specifically_physical_rejections":None,
            "interpretation":"Failed development trials are not final construction failures. A separate physical-rejection classification was not instrumented."})
    document = {"schema_version":VERSION,"protocol_version":protocol["version"],
        "status":"integrity_failed" if global_errors else "reported" if not errors else "reported_with_errors",
        "all_attempts_locked_before_scoring":True,
        "clinical_outcomes_decoded":bool(originals),
        "cases":metadata,"methods":[{"id":m,"label":LABELS[m]} for m in METHODS],
        "sampling":list(SAMPLING),"units":records,"statistics":stats,"automation":success,
        "resources":{**_resources(records),"run_instrumentation":summary.get("resources",{})},"errors":errors,
        "data_profile":{"csv_rows":len(rows),"units":len(records),"endpoint_classes":11,
            "scalar_predictions":sum(r["prediction"] is not None for r in rows),
            "scalar_reference_cells":sum(r["clinical_reference"] is not None for r in rows),
            "unique_reported_scalar_references":sum(e["value"] is not None for ref in references.values() for e in ref["endpoints"].values()),
            "clinical_curve_points":sum(len(ref["curve"]["time_h"]) for ref in references.values()),
            "weak_predictions":sum(r["weak_tail"] for r in rows)},
        "statistical_policy":{"resamples":RESAMPLES,"seed":SEED,"unit":"study cluster; Pletz2004 is shared by its two drugs",
            "point_estimator":"Drug-equal geometric mean of curve GMFE / paired GMFE ratios",
            "missingness":"No imputation. Conditional-score intervals include complete pairs; all failures remain in attempted denominators.",
            "multiple_comparisons":"Descriptive prespecified intervals; no multiplicity-adjusted superiority test"},
        "interpretation":["Exploratory previously viewed ten-drug cohort; no newly blinded external validation claim.",
            "Full/five changes development observations, not target clinical sampling or reference values.",
            "All five methods fit the same development observations in each full/five condition. Fixed-formula PBPK fits two parameters with 48 calls; Ours and RF-only fit two parameters across two formula candidates with 24 calls each.",
            "Both two-compartment methods fit four parameters; the prior variant additionally uses the declared clearance prior. Analytic and native computation budgets are recorded and are not claimed equal.",
            "Clinical scalar references are original reports. Prediction-only NCA approximates group statistics and original terminal fitting.",
            "Weak-tail endpoints are retained and flagged, not classified as trustworthy because a fold error is small.",
            "Tmax is constrained by administration and sampling and is not a primary quality discriminator.",
            "Renal observations, when available, are exploratory diagnostics; no additional tissue endpoint is claimed."],
        "source_pins":store.pins}
    store.checksum(Path(__file__))
    store.unchanged()
    # Atomic per-file replacement; only the three final report artifacts persist.
    with tempfile.TemporaryDirectory(prefix=".validation_report_",dir=run) as temporary:
        temp = Path(temporary)
        if not global_errors:
            document["figure_qa"] = render_comparison(document,temp/"comparison.pdf")
        else:
            document["figure_qa"] = {"status":"not_published", "reason":"Source integrity failed; no scientific comparison figure generated"}
        with (temp/"results.csv").open("w",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        store.unchanged()
        (temp/"results.json").write_text(json.dumps(document,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
        for name in (("results.csv","results.json") if global_errors else ("results.csv","comparison.pdf","results.json")):
            (temp/name).replace(run/name)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    args = parser.parse_args()
    result = report(args.run)
    print(json.dumps({"status":result["status"],"automation":result["automation"],"data_profile":result["data_profile"]},indent=2))
    if result["status"] == "integrity_failed":
        raise SystemExit(2)
