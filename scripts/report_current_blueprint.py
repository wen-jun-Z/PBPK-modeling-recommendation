#!/usr/bin/env python3
"""Report the uniform evidence-priority run without editing historical results.

All prediction attempts and source locks are checked before clinical outcomes
are decoded. Only current Ours predictions are rescored; the four historical
comparators and every clinical observation retain their stored values.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path

from run_complete_reference_pilot import ROOT, read, now, ref, verify
from run_clinical_method_comparison import predicted_nca, clinical_unit_factor, reporting_curve_indices
from render_pbpk_method_comparison import checked_curve, curve_metrics, Ineligible
from summarize_additional_clinical_baselines import prediction_in_hours, weak_endpoints
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprints.clinical_prediction_nca import UNITS

VERSION = "evidence-priority-ten-drug-1"
OLD_RUN = ROOT / "experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug/optimized_run"
LEGACY = ROOT / "results/tables/pbpk_ours_vs_rf/additional_comparison.json"
DEFAULT_RUN = ROOT / "experiments/blueprint_design/evidence_priority_20260929/run"
MANIFEST = ROOT / "results/data/blueprint_current.json"
METHOD_IDS = ("ours", "rf", "svr", "xgboost", "two_compartment")


def _pinned(path, pins, location):
    item = ref(inside(path, ROOT, exists=True), location)
    prior = pins.get(item["path"])
    if prior is not None and prior != item["sha256"]:
        raise ValueError("Source checksum disagrees with the run: " + item["path"])
    pins[item["path"]] = item["sha256"]
    return item


def _save_report(path, value):
    if path.exists() and read(path) != value:
        raise ValueError("Refusing to overwrite different reported predictions: " + str(path))
    save(path, value)


def _cutoff(config, clinical, previous):
    policy = config["nca_policy"]
    cutoff, basis = policy.get("lloq_ng_ml"), "reported_lloq"
    if cutoff is None:
        options = {
            "reported_lod": ("lod_ng_ml", "lod_used_as_prediction_cutoff", "reported_lod_proxy"),
            "cited_assay_calibration_lower_bound": ("calibration_lower_bound_ng_ml",
                "assay_calibration_lower_bound_as_cutoff", "cited_assay_calibration_lower_bound_proxy"),
        }
        proxy = options.get(policy.get("prediction_cutoff_proxy"))
        if proxy is None or proxy[1] not in clinical.get("approximations", []):
            raise ValueError("Missing quantification threshold requires the original declared proxy")
        cutoff, basis = policy[proxy[0]], proxy[2]
    diag = previous["nca_diagnostics"]
    if not math.isclose(cutoff, diag["lloq_ng_ml"], rel_tol=1e-12, abs_tol=0):
        raise ValueError("The original clinical NCA cutoff changed")
    if diag.get("quantification_threshold_basis") != basis:
        raise ValueError("The original cutoff/proxy interpretation changed")
    return cutoff, basis


def _ours_report(config, clinical, native, lock, previous):
    policy, metadata = config["nca_policy"], config["target_request_metadata"]
    if not (math.isclose(policy["dose_ng"], metadata["dosing"]["dose_mg"]*1e6)
            and math.isclose(policy["infusion_h"], metadata["dosing"]["infusion_minutes"]/60)
            and math.isclose(policy["weight_kg"], metadata["human"]["weight_kg"])):
        raise ValueError("NCA policy differs from the declared target administration")
    if native["unit"] != "umol/l" or native["time_unit"] != "min":
        raise ValueError("Unexpected native prediction units")
    mw = read(ROOT / config["physchem"])["identity"]["molecular_weight_g_mol"]
    times, values, merged = prediction_in_hours(native, mw)
    cutoff, basis = _cutoff(config, clinical, previous)
    nca = predicted_nca(policy=policy, time_h=times, concentration_ng_ml=values,
        dose_ng=policy["dose_ng"], infusion_h=policy["infusion_h"],
        nominal_times_h=policy["sampling_time_h"], lloq_ng_ml=cutoff, weight_kg=policy["weight_kg"])
    nca["diagnostics"].update(reported_lloq_ng_ml=policy.get("lloq_ng_ml"),
        quantification_threshold_basis=basis, coalesced_float_time_duplicates=merged)
    endpoints = {}
    for key, default_unit in UNITS.items():
        observed = clinical["clinical_reference"]["endpoints"].get(key) or {}
        unit = observed.get("unit") or default_unit
        computed = nca["endpoints"][key]
        factor = clinical_unit_factor(key, unit, policy["weight_kg"])
        endpoints[key] = {**computed, "unit": unit, "matrix": "plasma", "statistic": "representative_adult",
            "value": None if computed["value"] is None else computed["value"]*factor,
            "definition": observed.get("definition") or "Standard NCA predicted " + key,
            "estimation": "Prediction-only NCA using the original frozen sampling schedule and quantification threshold"}
        if key == "mrt" and "uncertain_mrt_infusion_correction" in clinical.get("approximations", []):
            endpoints[key].update(definition="Infusion-corrected MRT from predicted NCA",
                reference_definition=observed.get("definition"))
    indices = reporting_curve_indices(times, policy, metadata)
    unit = clinical["clinical_reference"]["curve"]["unit"]
    factor = clinical_unit_factor("curve", unit, policy["weight_kg"])
    return {"schema_version": "pbpk_clinical_prediction_v1", "case_id": config["case_id"],
        "study_id": config["study_id"], "method": "ours", "method_version": VERSION,
        "native_source": lock, "status": nca["status"],
        "curve": {"time_h": [times[i] for i in indices], "concentration": [values[i]*factor for i in indices],
            "unit": unit, "matrix": "plasma", "statistic": "representative_adult"},
        "endpoints": endpoints, "nca_diagnostics": nca["diagnostics"],
        "observed_values_used_for_endpoint_estimation": False,
        "nca_code": ref(ROOT / "src/pbpk_kb/blueprints/clinical_prediction_nca.py", "Unchanged prediction-only NCA")}


def _score_ours(record, clinical, predicted, lock):
    record = deepcopy(record)
    reference = checked_curve(clinical["clinical_reference"]["curve"], "clinical", clinical=True)
    try:
        metrics = curve_metrics(predicted["curve"], reference, set(clinical.get("approximations", [])))
    except (Ineligible, OverflowError) as exc:
        metrics = {"gmfe": None, "reason": str(exc)}
    record["gmfe"]["ours"] = metrics["gmfe"]
    record["curve_metrics"]["ours"] = metrics
    record["prediction_sources"]["ours"] = lock
    diag = predicted["nca_diagnostics"]
    weak = weak_endpoints(diag)
    record["nca_quality"]["ours"] = {key: diag.get(key) for key in
        ("auc_extrapolated_fraction", "aumc_extrapolated_fraction", "terminal_span_in_half_lives", "failures")}
    record["nca_quality"]["ours"].update(weak_endpoints=weak,
        coalesced_float_time_duplicates=diag.get("coalesced_float_time_duplicates", []),
        quantification_threshold_basis=diag["quantification_threshold_basis"])
    for key, cell in record["endpoints"].items():
        endpoint = predicted["endpoints"][key]
        if endpoint["unit"] != cell["source_unit"]:
            raise ValueError("Prediction/reference display units changed: " + key)
        observed_source = clinical["clinical_reference"]["endpoints"].get(key)
        observed = observed_source.get("value") if observed_source else None
        if observed is not None:
            observed *= cell["display_factor"]
        if observed != cell["observed"]:
            raise ValueError("Historical observation differs from its primary curation: " + key)
        value = endpoint["value"]
        if value is not None:
            value *= cell["display_factor"]
            if not math.isfinite(value) or value < 0 or (key != "tmax" and value == 0):
                raise ValueError("Nonfinite or nonpositive endpoint: " + key)
        error = None
        if value is not None and observed is not None:
            if key == "tmax":
                error = abs(value-observed)
            elif observed > 0:
                error = max(value/observed, observed/value)
        cell["predictions"]["ours"] = {"value": value,
            "weak_tail": value is not None and key in weak,
            "reason": endpoint.get("reason", "Endpoint unavailable") if value is None else "",
            "error": error, "error_metric": "absolute_error_h" if key == "tmax" else "fold_error"}
    return record


def _totals(records):
    totals = {}
    for method in METHOD_IDS:
        scores = [c["gmfe"][method] for c in records if c["gmfe"][method] is not None]
        endpoints = {}
        for key in UNITS:
            entries = [c["endpoints"][key]["predictions"][method] for c in records]
            errors = [p["error"] for p in entries if p["error"] is not None]
            stats = {"n": len(errors), "weak_scored_n": sum(p["error"] is not None and p["weak_tail"] for p in entries)}
            if errors:
                stats.update({"mean_absolute_error_h": math.fsum(errors)/len(errors)} if key == "tmax" else
                    {"gmfe": math.exp(math.fsum(map(math.log, errors))/len(errors)),
                     "within_2fold": sum(e <= 2 for e in errors)})
            endpoints[key] = stats
        totals[method] = {"curve_drugs": len(scores),
            "drug_equal_curve_gmfe": math.exp(math.fsum(map(math.log, scores))/len(scores)) if scores else None,
            "curve_gmfe_within_2fold": sum(s <= 2 for s in scores), "endpoint_metrics": endpoints}
    return totals


def report(run, manifest=MANIFEST):
    run = inside(run, ROOT, exists=True)
    manifest = inside(manifest, ROOT)
    if run == OLD_RUN or not run.is_relative_to(ROOT / "experiments/blueprint_design/evidence_priority_20260929"):
        raise ValueError("Only the new evidence-priority run can be reported here")
    summary = read(run / "summary.json")
    if summary.get("status") == "reported" and manifest.exists():
        current = read(manifest)
        if current.get("method_version") != VERSION:
            raise ValueError("Existing manifest describes a different method version")
        expected = {"run_summary": run / "summary.json", "comparison": run / "comparison.json",
                    "comparison_input": run / "comparison_input.json"}
        for key, path in expected.items():
            record = current[key]
            if inside(record["path"], ROOT, exists=True) != path:
                raise ValueError("Existing current manifest points to a different run: " + key)
            verify({record["path"]: record["sha256"]})
        verify(current["source_pins"])
        verify(summary["protocol"]["pins"])
        for key in ("inputs", "mappings"):
            verify(summary["locks"][key])
        for methods in summary["locks"]["predictions"].values():
            verify({r["path"]: r["sha256"] for r in methods.values()})
        return current  # Do not invalidate figure/table locks by updating timestamps.
    locked = (summary.get("all_prediction_attempts_locked_before_scoring")
              or summary.get("all_attempts_locked_before_scoring"))
    if summary.get("status") not in {"predictions_locked", "reported"} or not locked:
        raise ValueError("All ten prediction attempts must be locked before target outcomes are decoded")
    protocol = summary["protocol"]
    configs = protocol["cases"]
    if len(configs) != 10 or len({c["case_id"] for c in configs}) != 10:
        raise ValueError("The uniform report requires exactly ten distinct clinical cases")
    pins = deepcopy(protocol["pins"])
    verify(pins)
    for key in ("inputs", "mappings"):
        verify(summary["locks"][key]); pins.update(summary["locks"][key])
    for config in configs:
        cid = config["case_id"]
        for method in ("ours", "rf"):
            result = summary["results"][cid][method]
            lock = summary["locks"]["predictions"].get(cid, {}).get(method)
            if result.get("status") != "predicted" or lock is None:
                raise ValueError("Cannot publish an uncompleted prediction as a complete ten-drug report: " + cid + "/" + method)
            verify({lock["path"]: lock["sha256"]}); pins[lock["path"]] = lock["sha256"]
    _pinned(Path(__file__), pins, "Current report implementation")
    _pinned(LEGACY, pins, "Frozen clinical observations and four comparator predictions")
    legacy = read(LEGACY)
    verify(legacy["source_pins"]); pins.update(legacy["source_pins"])
    original = {c["case_id"]: c for c in legacy["cases"]}
    if set(original) != {c["case_id"] for c in configs}:
        raise ValueError("Frozen comparison and current cohort differ")
    document = {"schema_version": "pbpk_ours_rf_clinical_v1", "method_version": VERSION,
        "title": "Uniform evidence-priority PBPK predictions versus clinical references", "cases": []}
    records, manifest_cases, scoring = [], [], {}
    for config in configs:
        cid = config["case_id"]
        reference = _pinned(ROOT / config["reference"], pins, "Original clinical outcomes; decoded only after prediction locks")
        candidates = read(ROOT / reference["path"])["cases"]
        index, case = next((i, deepcopy(c)) for i, c in enumerate(candidates) if c["case_id"] == cid)
        if case["study_id"] != config["study_id"] or case["clinical_reference"]["case_id"] != cid or case["clinical_reference"]["study_id"] != config["study_id"]:
            raise ValueError("Clinical study identity mismatch")
        old_report = OLD_RUN / "predictions" / cid / "ours_reported.json"
        _pinned(old_report, pins, "Unchanged original NCA quantification-threshold metadata")
        previous = read(old_report)
        lock = summary["locks"]["predictions"][cid]["ours"]
        expected_prediction = run / "predictions" / cid / "ours/prediction_locked.json"
        if inside(lock["path"], ROOT, exists=True) != expected_prediction:
            raise ValueError("Every current target prediction must have been executed in the new common run")
        request_ref = _pinned(run / "predictions" / cid / "ours/request.json", pins, "Final current target construction request")
        request = read(ROOT / request_ref["path"])
        if request["human"] != config["target_request_metadata"]["human"] or request["dosing"] != config["target_request_metadata"]["dosing"]:
            raise ValueError("Actual target request differs from the frozen clinical protocol")
        _pinned(ROOT / config["physchem"], pins, "Original independent chemical inputs")
        prediction = _ours_report(config, case, read(ROOT / lock["path"]), lock, previous)
        dest = run / "predictions" / cid / "ours_reported.json"
        _save_report(dest, prediction)
        case["predictions"] = {"ours": _pinned(dest, pins, "Current prediction-only NCA and clinical-window curve")}
        rf_src = OLD_RUN / "predictions" / cid / "rf_reported.json"
        _pinned(rf_src, pins, "Original frozen RF clinical prediction")
        rf_report = read(rf_src); rf_lock = summary["locks"]["predictions"][cid]["rf"]
        if any(rf_report["native_source"][key] != rf_lock[key] for key in ("path", "sha256")):
            raise ValueError("RF historical source was changed")
        rf_dest = run / "predictions" / cid / "rf_reported.json"
        _save_report(rf_dest, rf_report)
        case["predictions"]["rf"] = _pinned(rf_dest, pins, "Unchanged RF reported prediction")
        case["nca_policy"] = deepcopy(config["nca_policy"])
        case.setdefault("reference_screen", {})["predictions_not_started"] = False
        document["cases"].append(case)
        records.append(_score_ours(original[cid], case, prediction, lock))
        scoring[cid] = {"ours": {"status": prediction["status"], "diagnostics": prediction["nca_diagnostics"]},
            "rf": {"status": rf_report["status"], "diagnostics": rf_report["nca_diagnostics"]}}
        mapping_path = ROOT / summary["selection"][cid]["mapping_path"]
        mapping = read(mapping_path)
        if not mapping.get("eligible") or mapping.get("warm_start_source") is not None or mapping["max_evaluations"] != 24:
            raise ValueError("Selected mapping violates uniform fit policy")
        if mapping["construction_partition_method"] != summary["selection"][cid]["formula"]:
            raise ValueError("Selected partition method disagrees with its mapping")
        selected = mapping_path.parent / f"attempt_{mapping['decision']['evaluation']:02d}" / "prediction_locked.json"
        development_prediction = _pinned(selected, pins, "Selected development-only native prediction")
        outcome = read(selected.with_name("result.json"))
        if outcome.get("status") != "predicted" or outcome.get("prediction_sha256") != development_prediction["sha256"]:
            raise ValueError("Selected development prediction is not intact")
        manifest_cases.append({"case_id": cid, "drug": config["drug"], "config": deepcopy(config),
            "mapping": _pinned(mapping_path, pins, "Current selected development mapping"),
            "development_prediction": development_prediction, "target_prediction": deepcopy(lock),
            "development_source": deepcopy(config["development_source"]),
            "physchem": _pinned(ROOT / config["physchem"], pins, "Independent chemical identity and descriptors"),
            "target_reference": {**reference, "json_pointer": f"/cases/{index}/clinical_reference"},
            "target_request": request_ref})
    display = deepcopy(legacy)
    display.update(method_version=VERSION, cases=records, summary=_totals(records), source_pins=pins,
        original_values_preserved=True, original_observations_and_four_comparators_preserved=True,
        ours_recomputed_for_every_case=True, evaluation_outcomes_used_for_new_fit_selection=False,
        interpretation="Exploratory evaluation on previously viewed drugs. The same evidence-priority rule, two-parameter optimizer, two partition formulas and 24-call no-warm-start budget apply to all ten drugs. Ours receives independently qualified priors plus a same-drug development curve; molecular baselines and classical PK have different information sets. This is not blinded validation or an equal-information superiority test.")
    display["methods"][0]["information"] = "Qualified independent primary clinical/regulatory priors where available, frozen RF fallback otherwise, and a same-drug development clinical curve; two PBPK partition candidates, identical no-warm-start fitting rule"
    for before, after in zip(legacy["cases"], records):
        if before["case_id"] != after["case_id"]:
            raise ValueError("Original cohort ordering changed")
        for method in METHOD_IDS[1:]:
            if any(before[key][method] != after[key][method] for key in ("gmfe", "curve_metrics", "nca_quality", "prediction_sources")):
                raise ValueError("A frozen comparator changed")
        for key in UNITS:
            if before["endpoints"][key]["observed"] != after["endpoints"][key]["observed"]:
                raise ValueError("An original observation changed")
            for method in METHOD_IDS[1:]:
                if before["endpoints"][key]["predictions"][method] != after["endpoints"][key]["predictions"][method]:
                    raise ValueError("A frozen comparator endpoint changed")
    verify(pins)
    save(run / "comparison_input.json", document)
    save(run / "comparison.json", display)
    summary.update(status="reported", method_version=VERSION, clinical_scoring=scoring,
        all_prediction_attempts_locked_before_scoring=True,
        reported_at=now(), evaluation_outcomes_read_during_fitting=False,
        current_report={"comparison": ref(run / "comparison.json", "Five methods; all ten current Ours predictions"),
            "comparison_input": ref(run / "comparison_input.json", "Clinical table-compatible current report")})
    summary["protocol"]["source_pins"] = deepcopy(summary["protocol"]["pins"])
    save(run / "summary.json", summary)
    current = {"schema_version": "pbpk_current_blueprint_v1", "method_version": VERSION,
        "run_summary": ref(run / "summary.json", "Completed uniform ten-drug run"),
        "comparison": ref(run / "comparison.json", "Frozen comparators and current ten-drug method"),
        "comparison_input": ref(run / "comparison_input.json", "Current clinical reporting inputs"),
        "cases": manifest_cases, "source_pins": pins,
        "all_ten_final_predictions_newly_executed": True,
        "target_outcomes_used_for_fitting_or_formula_selection": False}
    save(manifest, current)
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    result = report(args.run, args.manifest)
    print("Reported", len(result["cases"]), "drugs:", args.manifest)


if __name__ == "__main__":
    main()
