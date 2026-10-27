#!/usr/bin/env python3
"""Recompute the existing twenty-drug clinical evaluation from locked trajectories.

No training, simulation, parameter update, target selection, or external-method
reproduction occurs here. Historical results remain unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_clinical_pbpk_comparison as runner
import report_blueprint_validation as legacy
from score_cross_drug_blueprint import score_curve_reference, _native_fields
from score_expanded_tail import _window_values
from render_current_clinical_table import native_value
from pbpk_kb.blueprints.clinical_prediction_nca import UNITS
from pbpk_kb.blueprints.source_workflow import save, sha256

VERSION = "existing-clinical-unified-evaluation-1"
SEED = 20261004
RESAMPLES = 2000
METHODS = runner.METHODS
LABELS = {"rf_standard": "In-house RF + PK-Sim Standard",
          "ml_standard": "In-house rich ML + PK-Sim Standard",
          "ml_five_partition": "In-house rich ML + five-formula PBPK ensemble",
          "ours": "Our learned-exchange PBPK blueprint"}
SENSITIVITY_EXCLUSIONS = {"Nemonoxacin", "Dexmedetomidine", "Midazolam"}
REFERENCE_FAMILIES = ("original_reported", "observation_derived")
EVALUATION_CODE = ("scripts/evaluate_existing_clinical_pbpk.py",
                   "scripts/run_clinical_method_comparison.py",
                   "scripts/summarize_additional_clinical_baselines.py")


def fold_statistics(errors, *, attempted, reference_available):
    """Strict thresholds, explicit scored/reference/request denominators."""
    values = list(errors)
    if (not 0 <= len(values) <= reference_available <= attempted
            or any(not legacy.positive(v) or v < 1 for v in values)):
        raise ValueError("Invalid fold-error values or denominators")
    scored = len(values)
    return {"attempted": attempted, "reference_available": reference_available, "scored": scored,
        "missing_reference": attempted - reference_available,
        "reference_available_but_unscored": reference_available - scored,
        "gmfe": math.exp(math.fsum(math.log(v) for v in values) / scored) if scored else None,
        "threshold_comparison": "<", "within_fold": {str(t): {
            "count": sum(v < t for v in values), "scored_denominator": scored,
            "reference_denominator": reference_available, "attempted_denominator": attempted,
            "fraction_of_scored": sum(v < t for v in values) / scored if scored else None,
            "fraction_of_attempted": sum(v < t for v in values) / attempted if attempted else None}
            for t in (2, 3, 5)}}


def reference_family(case):
    return "observation_derived" if case["reference_kind"] == "observed_curve" else "original_reported"


def endpoint_label(endpoint, family):
    if family == "observation_derived":
        return {"auc": "AUC over matched observed window", "cmax": "Sampled peak on observed time grid",
                "tmax": "Time of sampled peak", "c_last": "Last observed-time concentration"}.get(endpoint, endpoint)
    return {"auc": "AUC versus original reported clinical AUC",
            "cmax": "Cmax versus original reported clinical Cmax"}.get(endpoint, endpoint)


def validate_locked(run):
    """Validate every prediction and the original source contract before scoring."""
    state = runner.read(run / "state.json")
    protocol = runner.checked(state["protocol"])
    runner.verify(protocol, state.get("runtime_code_patches"))
    cases = protocol["cases"]
    if (not state.get("all_predictions_locked_before_scoring") or len(cases) != 20
            or len({c["compound_id"] for c in cases}) != 20
            or set(protocol["methods"]) != set(METHODS)
            or set(state["outcomes"]) != {c["case_id"] for c in cases}):
        raise ValueError("Need the complete originally locked twenty-by-four comparison")
    loaded = []
    for case in cases:
        receipt = state["outcomes"][case["case_id"]]
        bundle = runner.checked(receipt)
        if (bundle.get("protocol") != state["protocol"] or bundle.get("case_id") != case["case_id"]
                or bundle.get("compound_id") != case["compound_id"] or set(bundle["entries"]) != set(METHODS)
                or not bundle.get("locked_at")):
            raise ValueError("Prediction receipt identity or coverage changed")
        for method, entry in bundle["entries"].items():
            if (entry.get("method") != method or entry.get("request", {}).get("compound_id") != case["compound_id"]
                    or entry.get("status") not in {"predicted", "failed"}
                    or entry.get("target_observations_used") is not False):
                raise ValueError("Unfinished, mismatched or target-informed prediction")
            if entry["status"] == "predicted":
                _native_fields(entry)
        runner.collect_refs(bundle, {})
        loaded.append((case, receipt, bundle))
    prior = runner.checked(state["parameter_predictions"])
    audit = prior["audit"]
    if (set(prior["parameters"]) != {c["compound_id"] for c in cases}
            or audit.get("target_labels_used") is not False
            or audit.get("target_observation_files_opened") is not False
            or audit.get("training_structure_overlap") is not False
            or audit.get("training_generic_scaffold_overlap") is not False):
        raise ValueError("Parameter-training exclusion audit differs")
    model_path = Path(prior["model_path"])
    model_path = runner.inside(model_path, ROOT, exists=True)
    if sha256(model_path) != prior["model_sha256"]:
        raise ValueError("Cached parameter model changed")
    return state, protocol, loaded, audit


def clinical_truth(case, store):
    if case["reference_kind"] != "observed_curve":
        clinical, curve, endpoints = legacy._case_reference(case, store)
        return clinical, curve, endpoints
    clinical = store.ref(case["reference_source"])
    if (clinical.get("compound_id") != case["compound_id"] or clinical.get("study_id") != case["study_id"]
            or clinical.get("role") != "historical_cross_study_reference"
            or (clinical.get("matrix"), clinical.get("unit"), clinical.get("time_unit")) != ("plasma", "umol/l", "min")):
        raise ValueError("Curve-only clinical reference identity or units differ")
    store.verify_refs(clinical)
    times = [row["time_min"] / 60 for row in clinical["observations"]]
    observed = [row["concentration_umol_per_l"] * clinical["molecular_weight_g_mol"]
                for row in clinical["observations"]]
    window = _window_values(times, observed)
    endpoints = {name: {"value": window.get(name), "unit": unit,
        "reason": "" if name in window else "No independently reported clinical terminal reference"}
        for name, unit in UNITS.items()}
    return clinical, None, endpoints


def recompute(loaded, store):
    rows = []
    for case, receipt, bundle in loaded:
        clinical, curve, truth = clinical_truth(case, store)
        for method in METHODS:
            entry = bundle["entries"][method]
            row = {key: case[key] for key in ("case_id", "compound_id", "drug", "study_id", "cluster_id", "dataset")}
            row.update(method=method, reference_family=reference_family(case), attempt_status=entry["status"],
                scoring_status="failed", failure=entry.get("failure"), prediction_reused=True,
                prediction_receipt={**receipt, "json_pointer": "/entries/" + method},
                reference_source=case["reference_source"], nca_policy=case["nca_policy"],
                source_hash_verified=True, curve_metrics={}, endpoints={}, new_solver_calls=0)
            if entry["status"] == "predicted":
                try:
                    if case["reference_kind"] == "observed_curve":
                        row.update(score_curve_reference(case, entry, store))
                    else:
                        unit = {"prediction": {**receipt, "json_pointer": "/entries/" + method + "/curve"},
                                "request": {**receipt, "json_pointer": "/entries/" + method + "/request"}}
                        scored, _, _ = legacy._score_prediction(case, clinical, curve, truth, unit, store)
                        scored.pop("curve", None)
                        for endpoint, value in scored["endpoints"].items():
                            value["reference_kind"] = "original_reported" if value["reference"] is not None else "unavailable"
                            value["native_prediction"], _, value["native_prediction_reason"] = native_value(
                                entry["endpoints"]["endpoints"][endpoint], endpoint, value["unit"],
                                entry["request"]["human"]["weight_kg"])
                        row.update(scored, scoring_status="scored" if legacy.positive(scored["curve_metrics"].get("gmfe")) else "partial")
                except (ValueError, ArithmeticError, KeyError) as error:
                    if "Source lock mismatch" in str(error) or "Source changed during reporting" in str(error):
                        raise
                    row.update(scoring_status="scoring_failed", failure=type(error).__name__ + ": " + str(error))
            for endpoint, unit in UNITS.items():
                if endpoint not in row["endpoints"]:
                    actual = truth[endpoint]
                    row["endpoints"][endpoint] = {"value": None, "native_prediction": None,
                        "reference": actual["value"], "unit": actual.get("unit") or unit, "error": None,
                        "error_metric": "absolute_error_h" if endpoint == "tmax" else "fold_error",
                        "reference_kind": reference_family(case) if actual["value"] is not None else "unavailable",
                        "reason": row.get("failure") or "Prediction unavailable", "weak_tail": False,
                        "reference_reason": actual.get("reason", "")}
                row["endpoints"][endpoint]["endpoint_label"] = endpoint_label(endpoint, reference_family(case))
            rows.append(row)
    store.unchanged()
    return rows


def summarise(rows, cases):
    strata = {"all_20": cases, "original_10": [c for c in cases if c["cohort"] == "original"],
        "extension_10": [c for c in cases if c["cohort"] == "extension"],
        "source_sensitivity": [c for c in cases if c["drug"] not in SENSITIVITY_EXCLUSIONS]}
    summaries, pairs, common = [], {}, []
    for stratum, group in strata.items():
        ids = {c["case_id"] for c in group}
        clusters = {c["case_id"]: c["cluster_id"] for c in group}
        by_method = {m: [r for r in rows if r["case_id"] in ids and r["method"] == m] for m in METHODS}
        curve_values = {m: {r["case_id"]: r["curve_metrics"]["gmfe"] for r in rs
                           if legacy.positive(r["curve_metrics"].get("gmfe"))} for m, rs in by_method.items()}
        common_ids = set.intersection(*(set(v) for v in curve_values.values()))
        endpoint_common = {}
        for family in REFERENCE_FAMILIES:
            endpoint_common[family] = {}
            for endpoint in ("auc", "cmax"):
                endpoint_common[family][endpoint] = set.intersection(*({r["case_id"] for r in rs
                    if r["reference_family"] == family and legacy.positive(r["endpoints"][endpoint]["error"])}
                    for rs in by_method.values()))
        for method, selected in by_method.items():
            values = curve_values[method]
            record = {"dataset": stratum, "method": method,
                "attempted": len(group), "prediction_failed": sum(r["attempt_status"] == "failed" for r in selected),
                "prediction_successful": sum(r["attempt_status"] == "predicted" for r in selected),
                "scored_curves": len(values), "unscored_curves": len(group) - len(values),
                "curve": fold_statistics(values.values(), attempted=len(group), reference_available=len(group)),
                "curve_gmfe_ci": legacy.clustered_geomean(values, {k: clusters[k] for k in values}, resamples=RESAMPLES, seed=SEED),
                "endpoints": {}}
            for family in REFERENCE_FAMILIES:
                family_rows = [r for r in selected if r["reference_family"] == family]
                if not family_rows:
                    continue
                record["endpoints"][family] = {}
                for endpoint in UNITS:
                    items = [r["endpoints"][endpoint] for r in family_rows]
                    available = sum(v["reference"] is not None for v in items)
                    errors = [v["error"] for v in items if v["error"] is not None]
                    stats = ({"attempted": len(items), "reference_available": available, "scored": len(errors),
                              "mae_h": math.fsum(errors) / len(errors) if errors else None,
                              "within_fold": None, "gmfe": None, "error_metric": "absolute_error_h"}
                             if endpoint == "tmax" else fold_statistics(errors, attempted=len(items), reference_available=available))
                    stats.update(label=endpoint_label(endpoint, family), predicted=sum(v["value"] is not None for v in items),
                        weak_scored=sum(v["error"] is not None and bool(v.get("weak_tail")) for v in items))
                    record["endpoints"][family][endpoint] = stats
            summaries.append(record)
            common_values = {k: values[k] for k in common_ids}
            common.append({"dataset": stratum, "method": method, "attempted": len(group),
                "shared_scored_case_ids": sorted(common_ids),
                "curve": fold_statistics(common_values.values(), attempted=len(group), reference_available=len(group)),
                "curve_gmfe_ci": legacy.clustered_geomean(common_values, {k: clusters[k] for k in common_values},
                                                         resamples=RESAMPLES, seed=SEED),
                "endpoints": {family: {endpoint: {
                    "shared_scored_case_ids": sorted(shared_ids),
                    **fold_statistics([r["endpoints"][endpoint]["error"] for r in selected if r["case_id"] in shared_ids],
                        attempted=sum(r["reference_family"] == family for r in selected),
                        reference_available=sum(r["reference_family"] == family and
                            r["endpoints"][endpoint]["reference"] is not None for r in selected))}
                    for endpoint, shared_ids in endpoints.items()}
                    for family, endpoints in endpoint_common.items()}})
        pairs[stratum] = {"curve": {m: legacy.paired_ratio(curve_values["ours"], curve_values[m], clusters,
            expected_ids=ids, resamples=RESAMPLES, seed=SEED) for m in METHODS if m != "ours"}, "endpoints": {}}
        for family in REFERENCE_FAMILIES:
            family_ids = {r["case_id"] for r in by_method["ours"] if r["reference_family"] == family}
            if not family_ids:
                continue
            pairs[stratum]["endpoints"][family] = {}
            for endpoint in ("auc", "cmax"):
                values = {m: {r["case_id"]: r["endpoints"][endpoint]["error"] for r in rs
                    if r["case_id"] in family_ids and legacy.positive(r["endpoints"][endpoint]["error"])}
                    for m, rs in by_method.items()}
                pairs[stratum]["endpoints"][family][endpoint] = {m: legacy.paired_ratio(values["ours"], values[m],
                    clusters, expected_ids=family_ids, resamples=RESAMPLES, seed=SEED) for m in METHODS if m != "ours"}
    return summaries, common, pairs


def compare_historical(rows, historical):
    """Confirm the same metric definitions reproduce the historical locked scores."""
    old = {(r["case_id"], r["method"]): r for r in historical["units"]}
    checks = 0
    def equal(left, right):
        if left is None or right is None:
            return left is right
        return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
    for row in rows:
        previous = old[row["case_id"], row["method"]]
        if (row["attempt_status"], row["scoring_status"]) != (previous["attempt_status"], previous["scoring_status"]):
            raise ValueError("Historical prediction/scoring status differs")
        for key in ("gmfe", "ln_rmse", "n"):
            if not equal(row["curve_metrics"].get(key), previous.get("curve_metrics", {}).get(key)):
                raise ValueError("Historical curve metric differs: " + row["case_id"] + "/" + key)
            checks += 1
        for endpoint, previous_value in previous["endpoints"].items():
            for key in ("value", "reference", "error", "native_prediction"):
                if not equal(row["endpoints"][endpoint].get(key), previous_value.get(key)):
                    raise ValueError("Historical endpoint differs: " + row["case_id"] + "/" + endpoint + "/" + key)
                checks += 1
    return {"passed": True, "numeric_comparisons": checks, "rtol": 1e-10, "atol": 1e-12,
            "threshold_change": "New strict FE < 2/3/5; historical <= 2 counts are not overwritten"}


def write_csv(path, rows):
    fields = ["drug", "study_id", "dataset", "method", "method_label", "attempt_status", "scoring_status",
        "endpoint", "endpoint_label", "reference_family", "reference_kind", "unit", "prediction",
        "native_prediction", "observation", "error", "error_metric", "weak_tail", "curve_gmfe", "curve_ln_rmse",
        "curve_points", "prediction_reused", "prediction_sha256", "source_hash_verified", "reason"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            base = {key: row[key] for key in ("drug", "study_id", "dataset", "method", "attempt_status",
                                            "scoring_status", "reference_family", "prediction_reused", "source_hash_verified")}
            base.update(method_label=LABELS[row["method"]], prediction_sha256=row["prediction_receipt"]["sha256"])
            metric = row["curve_metrics"]
            writer.writerow({**base, "endpoint": "plasma_concentration_time", "endpoint_label": "Clinical concentration-time profile",
                "reference_kind": "clinical_observed_curve", "curve_gmfe": metric.get("gmfe"),
                "curve_ln_rmse": metric.get("ln_rmse"), "curve_points": metric.get("n"), "reason": row.get("failure")})
            for endpoint, value in row["endpoints"].items():
                writer.writerow({**base, "endpoint": endpoint, "endpoint_label": value["endpoint_label"],
                    "reference_kind": value["reference_kind"], "unit": value["unit"], "prediction": value["value"],
                    "native_prediction": value.get("native_prediction"), "observation": value["reference"],
                    "error": value["error"], "error_metric": value["error_metric"], "weak_tail": value.get("weak_tail", False),
                    "reason": value.get("reason") or value.get("reference_reason") or row.get("failure")})


def write_readme(path, result):
    lines = ["# Unified clinical evaluation: twenty existing drugs", "",
        "The four in-house pipelines are evaluated on the same existing clinical cases. This is not a reproduction of an external author's code.",
        "All 80 prediction attempts were reused after hash verification; clinical scores were recomputed from the original observations. No new model fitting or simulation was performed.", "",
        "| Method | Successful predictions | Available-case curve GMFE | Common-19 curve GMFE | FE < 2 | FE < 3 | FE < 5 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    common = {r["method"]: r for r in result["common_case_summaries"] if r["dataset"] == "all_20"}
    for row in result["summaries"]:
        if row["dataset"] == "all_20":
            c = row["curve"]
            ratios = [f"{c['within_fold'][str(t)]['count']}/{c['scored']}" for t in (2, 3, 5)]
            lines.append(f"| {LABELS[row['method']]} | {row['prediction_successful']}/{row['attempted']} | {c['gmfe']:.3f} | {common[row['method']]['curve']['gmfe']:.3f} | " + " | ".join(ratios) + " |")
    lines += ["", "Curve GMFE is the geometric mean of per-drug curve GMFEs, giving every drug equal weight; concentration points are not pooled. Thresholds are strictly FE < 2, < 3 and < 5. The historical results used <= 2 and remain unchanged.", "",
        "## Primary endpoint evaluation", "",
        "The common-19 column is the directly comparable curve result. Available-case columns and threshold counts retain each method's successes and must not be interpreted as identical case sets. Treprostinil failed before producing a trajectory: predicted Vss 41.793 L is below the immutable vascular-observer equivalent capacity 47.929 L (not anatomical blood volume), causing 17 structural rejections. The retained outer status is `no_usable_sampled_NCA_model`; this is not a numeric NCA failure. The failed attempt remains in all requested denominators.", "",
        "Originally reported clinical AUC/Cmax are evaluated separately from curve-derived AUC over the matched observation window and the sampled peak. A sampled peak is not a measured continuous-time Cmax. The table below uses endpoint-specific cases scoreable by all four methods.",
        "Reported AUC follows each case's frozen NCA policy and the source endpoint definition (AUC0-t or AUC-infinity where reported). Prediction-dependent LLOQ filtering can change the last quantifiable point; these comparisons are not all identical fixed-window integrations. Per-case source receipts, NCA policies and diagnostics are retained in the JSON. Curve-derived endpoints use the fixed observed time grid.", "",
        "| Reference family | Method | Common-case AUC GMFE (n) | Common-case Cmax or sampled peak GMFE (n) |",
        "|---|---|---:|---:|"]
    for row in common.values():
        for family, endpoints in row["endpoints"].items():
            a, c = endpoints["auc"], endpoints["cmax"]
            lines.append(f"| {family} | {LABELS[row['method']]} | {a['gmfe']:.3f} ({a['scored']}) | {c['gmfe']:.3f} ({c['scored']}) |")
    lines += ["", "Our/control FE ratios below use paired common cases and study-cluster 95% bootstrap intervals. A ratio below one favors our pipeline; it is not an absolute prediction-error estimate.", "",
        "| Outcome | Reference | Internal control | Paired n | Our/control FE ratio [95% CI] |",
        "|---|---|---|---:|---:|"]
    pairs = result["paired_ours_over_internal_controls"]["all_20"]
    def pair_line(outcome, family, method, pair):
        ci = pair["ci95"]
        return f"| {outcome} | {family} | {LABELS[method]} | {pair['n_cases']} | {pair['estimate']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |"
    for method, pair in pairs["curve"].items():
        lines.append(pair_line("Curve", "clinical observed", method, pair))
    for family, endpoints in pairs["endpoints"].items():
        for endpoint, methods in endpoints.items():
            for method, pair in methods.items():
                lines.append(pair_line(endpoint_label(endpoint, family), family, method, pair))
    lines += ["", "Our common-case curve estimate is lower than the three controls, but every curve ratio interval includes one. Originally reported AUC is worse than RF and rich ML Standard on the five common cases; reported Cmax is better in point estimate on nine common cases. These small, previously viewed cohorts do not establish general superiority or clinical reliability."]
    lines += ["", "The JSON includes all-20, original-10, extension-10 and source-sensitivity strata; complete-case paired study-cluster bootstrap uses 2,000 resamples and seed 20261004. Ertapenem and Ceftriaxone share a study cluster. Source sensitivity excludes the previously flagged Nemonoxacin, Dexmedetomidine and Midazolam cases; the main results retain them.",
        "Failed attempts remain in requested denominators. Threshold tables explicitly provide scored, reference-available and attempted denominators. All 11 output types are retained (one curve and ten scalar endpoints); missing clinical references stay empty. Weak terminal estimates are flagged.", "",
        "## Input and interpretation boundaries", "",
        "This retained comparison uses independently predicted CL/fu and fixed physicochemical inputs. Our method additionally uses independently predicted Vss/half-life. It does not yet constitute a new evaluation of an independent-evidence-first input policy.",
        "The cached parameter model excluded all twenty target identities and generic scaffolds from its 1,283-row source, retaining 1,126 compounds. Missing parameter labels were not imputed; training-only scaffold OOF residuals controlled consistency projection. No same-drug clinical development or target curves were fitted. These previously viewed drugs are not a new blinded test.",
        "RF + Standard versus rich ML + Standard compares parameter prediction; rich Standard versus five-formula PBPK compares formula combination; our method uses additional predicted parameter constraints. These controls do not isolate a pure architecture effect with identical information content.", "",
        "[Unified evaluation JSON](evaluation.json) · [All 880 curve/endpoint rows](evaluation.csv). Historical [results](results.json) and [table](results.csv) are unchanged; the paper is unchanged.", "",
        "```bash", "runtime/app-env/bin/python -B scripts/evaluate_existing_clinical_pbpk.py report",
        "runtime/app-env/bin/python -B scripts/evaluate_existing_clinical_pbpk.py status", "```", ""]
    path.write_text("\n".join(lines))


def report(run=runner.DEFAULT):
    run = runner.own(run)
    with (run / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, protocol, loaded, audit = validate_locked(run)
        original_state = deepcopy(state)
        scorer_sha = sha256(Path(__file__))
        evaluation_code_pins = {name: sha256(ROOT / name) for name in EVALUATION_CODE}
        store = legacy.Store(ROOT)
        rows = recompute(loaded, store)
        historical = runner.checked(state["results"])
        consistency = compare_historical(rows, historical)
        summaries, common, pairs = summarise(rows, protocol["cases"])
        if len(rows) != 80 or any(len(r["endpoints"]) != 10 for r in rows):
            raise ValueError("Expected eighty attempts with all ten scalar output slots")
        result = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": state["protocol"], "prediction_lock_at": state["predictions_locked_at"],
            "prediction_receipts": state["outcomes"], "historical_results": state["results"],
            "scorer_source": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": scorer_sha},
            "evaluation_code_pins": evaluation_code_pins,
            "verified_source_pins": {**protocol["source_pins"], **store.pins},
            "methods": LABELS, "prediction_reused": True, "new_training_calls": 0, "new_solver_calls": 0,
            "attempts": len(rows), "curve_aggregation": "geometric_mean_of_per_drug_GMFE_equal_weight",
            "threshold_comparison": "<", "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "unit": "study_cluster"},
            "source_sensitivity_exclusions": sorted(SENSITIVITY_EXCLUSIONS),
            "training_audit": {key: audit[key] for key in ("target_count", "raw_row_count", "train_count",
                "row_disposition_counts", "target_labels_used", "target_observation_files_opened",
                "training_structure_overlap", "training_generic_scaffold_overlap")},
            "historical_metric_check": consistency, "summaries": summaries, "common_case_summaries": common,
            "paired_ours_over_internal_controls": pairs, "units": rows,
            "limitations": ["Internal implementations; no external author code reproduction.",
                "Previously viewed development drugs, not a new blind clinical test.",
                "Original reports and matched-window curve-derived references are never pooled as one endpoint result.",
                "Reported AUC uses source-specific frozen NCA rules; LLOQ filtering can produce prediction-dependent last times rather than a common fixed window.",
                "Additional predicted Vss and half-life give our method different parameter inputs from CL/fu-only controls.",
                "No invented terminal labels or exclusion of failures; clinical references are unavailable for some outputs.",
                "This is a rescoring of retained predictions, not an independent-evidence-first rerun."], "paper_updated": False}
        store.unchanged()
        runner.verify(protocol, state.get("runtime_code_patches"))
        for name, expected in evaluation_code_pins.items():
            if sha256(ROOT / name) != expected:
                raise ValueError("Evaluation code changed while scoring: " + name)
        save(run / "evaluation.json", result)
        write_csv(run / "evaluation.csv", rows)
        write_readme(run / "README.md", result)
        state["result_readme"] = runner.ref(run / "README.md")
        state["unified_evaluation"] = {"result": runner.ref(run / "evaluation.json"),
            "table": runner.ref(run / "evaluation.csv"), "scorer_source": result["scorer_source"],
            "evaluation_code_pins": evaluation_code_pins,
            "source_pins": result["verified_source_pins"]}
        if {k: v for k, v in state.items() if k not in {"result_readme", "unified_evaluation"}} != {
                k: v for k, v in original_state.items() if k not in {"result_readme", "unified_evaluation"}}:
            raise ValueError("Unexpected historical state mutation")
        save(run / "state.json", state)
    return {"status": "complete", "attempts": len(rows), "reused_predictions": len(rows),
            "new_solver_calls": 0, "historical_metric_check": consistency,
            "curve_summary": [{"method": r["method"], "gmfe": r["curve"]["gmfe"],
                               "scored": r["scored_curves"], "attempted": r["attempted"]}
                              for r in summaries if r["dataset"] == "all_20"]}


def status(run=runner.DEFAULT):
    run = runner.own(run)
    state = runner.read(run / "state.json")
    receipt = state.get("unified_evaluation")
    if not receipt:
        return {"status": "not_evaluated", "original_prediction_status": state["status"]}
    result = runner.checked(receipt["result"])
    if sha256(ROOT / receipt["table"]["path"]) != receipt["table"]["sha256"]:
        raise ValueError("Evaluation CSV changed")
    return {"status": "complete", "attempts": result["attempts"], "new_solver_calls": result["new_solver_calls"],
            "new_training_calls": result["new_training_calls"], "historical_metric_check": result["historical_metric_check"],
            "result": receipt["result"], "table": receipt["table"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "status"))
    parser.add_argument("--run-dir", type=Path, default=runner.DEFAULT)
    args = parser.parse_args()
    print(json.dumps(report(args.run_dir) if args.command == "report" else status(args.run_dir),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
