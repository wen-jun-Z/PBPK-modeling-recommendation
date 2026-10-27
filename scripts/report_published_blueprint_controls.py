#!/usr/bin/env python3
"""Render one manuscript table from completed, locked external inference fits.

No fits, target scoring, source mutation, or incomplete-result publication occur
here. Every reference is verified within this project; printed audit information
can be incorporated into the existing publication manifest by the paper builder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

from published_blueprint_costs import load_baseline_costs


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/blueprint_design/published_inference_controls_20261011"
TABLE = ROOT / "results/tables/blueprint_published_inference_controls.tex"
METHODS = (
    ("pbpk", "Evidence-constrained PBPK (ours)"),
    ("bayesian_dram", "DRAM--PBPK"),
    ("pss_bayesian_dram", "PSS-SVD + DRAM"),
    ("snpe_c", "SNPE-C + PBPK"),
    ("two_compartment", "Two-compartment PK"),
)


def inside(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError("Reference outside the authorized project")
    return candidate


def sha(path: str | Path) -> str:
    return hashlib.sha256(inside(path).read_bytes()).hexdigest()


def read(path: str | Path):
    return json.loads(inside(path).read_text())


def reference(path: str | Path) -> dict:
    path = inside(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked(ref: dict):
    path = inside(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError(f"Changed source: {path.relative_to(ROOT)}")
    return read(path)


def checked_csv(ref: dict) -> list[dict]:
    path = inside(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("Changed endpoint/diagnostic CSV")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite summary value cannot be printed as a result")
    if abs(float(value)) >= 1e4 or (0 < abs(float(value)) < 10**(-digits)):
        return f"{float(value):.2e}"
    return f"{float(value):.{digits}f}"


def interval(stat: dict) -> str:
    estimate, ci = stat.get("estimate"), stat.get("ci95")
    if estimate is None:
        return "--"
    return number(estimate) + (f" [{number(ci[0])}, {number(ci[1])}]" if ci else "")


def endpoint_coverage(rows: list[dict], cases: dict) -> dict:
    """Classify endpoint availability independently of a job's 'predicted' flag.

    Reference values can be omitted by a failed method's row even though another
    method has the same clinical reference. Availability is therefore determined
    across methods within the same case/endpoint, never from the failed row alone.
    Curve rows intentionally lack scalar P/O; any score proves a usable reference
    exists, while a stored prediction with no curve score is a scoring failure.
    """
    def finite(value):
        return value not in (None, "") and math.isfinite(float(value))
    records = [r for r in rows if r["info"] == "time_five" and r["transfer"] == "linear_weight"]
    reference_exists = {}
    for row in records:
        key = row["case_id"], row["endpoint"]
        reference_exists[key] = reference_exists.get(key, False) or (
            finite(row.get("error")) if row["endpoint"] == "curve" else finite(row.get("reference")))
    output = {}
    for cohort in ("all_thirty", "added_twenty"):
        ids = {key for key, case in cases.items() if cohort == "all_thirty" or case["cohort"] == cohort}
        output[cohort] = {}
        for method, _ in METHODS:
            output[cohort][method] = {}
            method_rows = [row for row in records if row["case_id"] in ids and row["method"] == method]
            for endpoint in sorted({row["endpoint"] for row in method_rows}):
                selected = [row for row in method_rows if row["endpoint"] == endpoint]
                if len(selected) != len(ids) or len({row["case_id"] for row in selected}) != len(ids):
                    raise ValueError("Endpoint coverage requires one record per enrolled drug")
                counts = {"enrolled": len(ids), "scored": 0, "reference_missing": 0,
                          "prediction_missing": 0, "scoring_failed": 0,
                          "both_reference_and_prediction_missing": 0, "unscored_cases": []}
                for row in selected:
                    reference = reference_exists[row["case_id"], endpoint]
                    prediction = (row.get("status") in ("predicted", "frozen_predicted", "complete", "ok", "fitted")
                                  if endpoint == "curve" else finite(row.get("prediction")))
                    if finite(row.get("error")):
                        category = "scored"
                    elif not reference:
                        category = "reference_missing"
                        counts["both_reference_and_prediction_missing"] += int(not prediction)
                    elif not prediction:
                        category = "prediction_missing"
                    else:
                        category = "scoring_failed"
                    counts[category] += 1
                    if category != "scored":
                        counts["unscored_cases"].append({"case_id": row["case_id"], "classification": category,
                            "job_status": row.get("status"),
                            "detail": "Curve score unavailable; finite trajectory is not proof of scorable positive matched pairs."
                                      if endpoint == "curve" else "Missing endpoint estimate or clinical reference; see NCA diagnostics."})
                output[cohort][method][endpoint] = counts
    return output


def fit_cost(fit: dict) -> dict:
    """Return saved computation, including interruptions; missing is not zero."""
    inference = fit.get("inference", {})
    calls = inference.get("simulator_calls", fit.get("simulator_calls"))
    elapsed = inference.get("wall_seconds", inference.get("diagnostics", {}).get(
        "elapsed_seconds", fit.get("elapsed_seconds", fit.get("wall_seconds"))))
    if calls is not None and (isinstance(calls, bool) or int(calls) != calls or calls < 0):
        raise ValueError("Invalid saved simulator-call accounting")
    if elapsed is not None and (not math.isfinite(float(elapsed)) or float(elapsed) < 0):
        raise ValueError("Invalid saved inference-time accounting")
    reason = fit.get("reason", "")
    unattempted = fit.get("status") != "fitted" and reason.startswith("NotRun:")
    interrupted = fit.get("status") != "fitted" and reason.startswith("TimeoutError:")
    if unattempted and (calls != 0 or elapsed != 0):
        raise ValueError("An explicitly unattempted formula cannot consume saved inference work")
    if interrupted and (elapsed is None or elapsed <= 0 or not fit.get("time_accounting_scope")):
        raise ValueError("Timeout requires its nonzero interrupted wall time and measurement scope")
    return {"calls": None if calls is None else int(calls),
            "elapsed_seconds": None if elapsed is None else float(elapsed),
            "interrupted": interrupted, "unattempted": unattempted,
            "scope": fit.get("time_accounting_scope", "Saved inference-function wall time")}


def verify_compatibility_amendment(directory: Path, final_state: dict) -> dict | None:
    """Verify a namespace-only execution amendment without changing fit sources."""
    path = directory / "compatibility_amendment.json"
    if not path.exists():
        return None
    amendment = read(path)
    if (amendment.get("kind") != "numpy2_pymcmcstat_exception_namespace_compatibility"
            or amendment["original_protocol"] != final_state["protocol"]
            or amendment.get("pinned_source_changed") is not False
            or amendment.get("algorithms_or_budgets_changed") is not False
            or amendment.get("predictions_reused_without_change") is not True
            or amendment.get("target_scoring_started") is not False):
        raise ValueError("Unsupported or scientifically altered compatibility amendment")
    protocol = checked(amendment["original_protocol"])
    for name in ("source_file", "launcher_script"):
        ref = amendment[name]
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Changed compatibility source/launcher")
    known_launcher_hashes = {amendment["launcher_script"]["sha256"]}
    for name in ("single_recovery_launcher_source", "waiting_launcher_source"):
        original_launcher = amendment.get(name)
        if original_launcher:
            if hashlib.sha256(original_launcher["code"].encode()).hexdigest() != original_launcher["sha256"]:
                raise ValueError("Recovery/waiting launcher text differs from its preserved hash")
            known_launcher_hashes.add(original_launcher["sha256"])
    pause = checked(amendment["state_at_pause"])
    if pause["protocol"] != final_state["protocol"] or set(pause["completed"]) != set(amendment["completed_at_pause"]):
        raise ValueError("Compatibility pause snapshot differs from declared preserved jobs")
    for job, ref in pause["completed"].items():
        if final_state["completed"].get(job) != ref:
            raise ValueError("An already completed fit was replaced on compatibility resume")
        checked(ref)
    for ref in amendment.get("already_locked_unregistered", []):
        document = checked(ref)
        if final_state["completed"].get(document["job"]) != ref:
            raise ValueError("An unregistered locked checkpoint changed on resume")
    validation = amendment["independent_validation"]
    if (validation.get("positive_definite_singular_and_indefinite_match_modern_exception_reference") is not True
            or validation.get("cholesky_and_exception_object_identity_preserved") is not True
            or validation.get("same_seed_gaussian_dram_posterior_bitwise_equal") is not True
            or validation.get("simulator_calls_before") != validation.get("simulator_calls_after")):
        raise ValueError("Compatibility equivalence checks are missing")
    recovered = directory / "compatibility_recovered_job.json"
    recovery_ref = None
    if recovered.exists():
        recovery = read(recovered)
        if (recovery["protocol"] != final_state["protocol"] or recovery["target_scoring_started"]
                or final_state["completed"].get(recovery["job"]) != recovery["result"]):
            raise ValueError("Recovered job identity changed or used target scoring")
        checked(recovery["result"])
        recovery_ref = reference(recovered)
    resume = amendment.get("resume_execution")
    execution_counts = None
    if resume:
        if resume["original_protocol"] != final_state["protocol"] or resume["launcher_script"] != amendment["launcher_script"]:
            raise ValueError("Resume execution changed its protocol or launcher identity")
        pending = set(resume["pending_keys"])
        cached = resume["cached_job_refs"]
        if (not pending.issubset(final_state["completed"])
                or set(resume["new_inference_job_keys"]) != pending-set(cached)):
            raise ValueError("Resume execution lists inconsistent pending and new jobs")
        for job, ref in cached.items():
            if final_state["completed"].get(job) != ref:
                raise ValueError("A cached job changed during compatibility resume")
            checked(ref)
        for job, refs in resume["partial_formula_refs"].items():
            final_job = checked(final_state["completed"][job])
            for ref in refs:
                if ref not in final_job["fits"]:
                    raise ValueError("An already locked formula changed during compatibility resume")
                checked(ref)
        execution_counts = {"pending_jobs": len(pending), "cached_jobs": len(cached),
                            "new_inference_jobs": len(resume["new_inference_job_keys"]),
                            "preserved_partial_formula_fits": sum(map(len, resume["partial_formula_refs"].values()))}
    parallel = amendment.get("parallel_recovery")
    parallel_refs = []
    if parallel:
        if (parallel.get("algorithm_or_seed_changes") is not False
                or parallel["maximum_active_neural_jobs"] > parallel["original_limit"]
                or parallel["launcher_script"]["sha256"] not in known_launcher_hashes
                or {r["job"] for r in parallel["execution_records"]} != set(parallel["job_keys"])):
            raise ValueError("Parallel recovery exceeds original limits or changes the frozen experiment")
        for record in parallel["execution_records"]:
            completion = read(record["completion_record"])
            recovered_timeout = "deadline_script" in completion
            expected_pid = completion.get("original_worker_pid") if recovered_timeout else completion["pid"]
            if (completion["job"] != record["job"] or completion["protocol"] != final_state["protocol"]
                    or completion["target_scoring_started"] is not False or expected_pid != record["pid"]
                    or completion["launcher_script"]["sha256"] not in known_launcher_hashes
                    or final_state["completed"].get(record["job"]) != completion["result"]):
                raise ValueError("Parallel recovery completion changed identity or prediction locks")
            checked(completion["result"])
            parallel_refs.append(reference(record["completion_record"]))
    deadline_records = []
    for name in ("deadline_enforcement", "parallel_deadline_enforcement"):
        enforcement = amendment.get(name)
        if not enforcement:
            continue
        script = enforcement["script"]
        if (sha(script["path"]) != script["sha256"] or enforcement["target_observations_used"] is not False
                or enforcement["original_budget_seconds"] != protocol["maximum_job_wall_seconds"]):
            raise ValueError("Deadline enforcement changed its source, budget or target isolation")
        records = ([enforcement["recovery"]] if enforcement.get("recovery") else [])
        records += list(enforcement.get("recoveries", {}).values())
        for ref in records:
            recovery = checked(ref)
            if (recovery["protocol"] != final_state["protocol"] or recovery["target_scoring_started"] is not False
                    or final_state["completed"].get(recovery["job"]) != recovery["result"]
                    or recovery.get("deadline_script", recovery.get("script")) != script):
                raise ValueError("Timeout completion changed its source or locked result")
            job_result = checked(recovery["result"])
            preserved = [recovery[k] for k in ("first_formula_preserved", "second_formula") if k in recovery]
            preserved += recovery.get("interrupted_formula_refs", [])
            for fitref in preserved:
                fit = checked(fitref)
                if fitref not in job_result["fits"]:
                    raise ValueError("Timeout recovery replaced its preserved formula output")
                if fit.get("job_deadline_utc_unix"):
                    if (fit["locked_at"] < fit["job_deadline_utc_unix"]
                            or not 0 <= fit["job_deadline_utc_unix"]-fit["job_start_utc_unix"]-enforcement["original_budget_seconds"] <= 1):
                        raise ValueError("Formula was interrupted before its original wall budget")
                    fit_cost(fit)
            deadline_records.append(ref)
    adjudication = amendment.get("timeout_adjudication_amendment")
    if deadline_records and (not adjudication or adjudication.get("before_any_target_scoring") is not True
                             or adjudication.get("failure_adjudication_changed") is not True):
        raise ValueError("Computational timeout finalization lacks its pre-scoring policy amendment")
    if adjudication and (adjudication["original_policy"] != protocol["wall_limit_policy"]
                         or adjudication["original_job_budget_seconds"] != protocol["maximum_job_wall_seconds"]
                         or adjudication["budget_or_inference_algorithm_changed"] is not False
                         or adjudication["target_results_exist"] is not False
                         or adjudication["recorded_at"] > final_state["predictions_locked_at"]):
        raise ValueError("Timeout adjudication changed the computation budget or lacks pre-scoring chronology")
    return {"amendment": reference(path), "preserved_completed_jobs": len(pause["completed"]),
            "preserved_unregistered_jobs": len(amendment.get("already_locked_unregistered", [])),
            "recovered_job": recovery_ref, "scientific_source_files_changed": False,
            "resume_execution_counts": execution_counts,
            "parallel_recovery_records": parallel_refs,
            "computational_timeout_recovery_records": deadline_records,
            "timeout_adjudication_amendment": adjudication,
            "execution_change": amendment["shim"],
            "phase_caution": "A result locked after amendment creation may still have finished in the original executor; timestamps alone do not identify its execution version."}


def recompute_paired_interval(values: dict[str, float], cases: dict, seed: int,
                              resamples: int = 2000) -> dict:
    """Independent literal cluster resampling; never import the scorer/bootstrap."""
    keys = sorted(values)
    clusters = sorted({cases[key]["cluster_id"] for key in keys})
    if not keys:
        return {"estimate": None, "ci95": None, "n_cases": 0, "n_clusters": 0}
    log_errors = {key: math.log(values[key]) for key in keys}
    estimate = math.exp(statistics.mean(log_errors.values()))
    ci = None
    if len(clusters) > 1:
        groups = [[log_errors[key] for key in keys if cases[key]["cluster_id"] == cluster]
                  for cluster in clusters]
        rng = np.random.default_rng(seed)
        means = []
        for _ in range(resamples):
            sampled = [value for index in rng.integers(0, len(clusters), size=len(clusters))
                       for value in groups[index]]
            means.append(math.exp(statistics.mean(sampled)))
        ci = np.quantile(means, [0.025, 0.975]).tolist()
    return {"estimate": estimate, "ci95": ci, "n_cases": len(keys), "n_clusters": len(clusters)}


def validate_and_load(directory: Path = EXPERIMENT) -> dict:
    directory = inside(directory)
    state = read(directory / "state.json")
    if state.get("status") != "complete" or not state.get("results"):
        raise ValueError("All predictions must be locked and scored before table publication")
    protocol, results = checked(state["protocol"]), checked(state["results"])
    jobs, cases = protocol["jobs"], protocol["cases"]
    if (len(cases) != 30 or len(jobs) != 90
            or set(state["completed"]) != set(jobs)
            or results["status"] != "complete" or results["protocol"] != state["protocol"]):
        raise ValueError("Expected the complete thirty-drug, three-algorithm batch")
    if not results["predictions_locked_at"] <= results["scored_at"]:
        raise ValueError("Scoring must follow all prediction locks")
    for path, expected in protocol["source_pins"].items():
        if sha(path) != expected:
            raise ValueError(f"Changed protocol source: {path}")
    rows = checked_csv(results["endpoints"])
    checked_csv(results["diagnostics"])
    if set(results["methods"]) != {m for m, _ in METHODS}:
        raise ValueError("Unexpected comparison algorithms")
    coverage = endpoint_coverage(rows, cases)
    audit = {"input_protocol": state["protocol"], "input_results": state["results"],
             "input_endpoints": results["endpoints"], "input_diagnostics": results["diagnostics"],
             "source_pins_verified": len(protocol["source_pins"]),
             "prediction_locks_verified": 0, "formula_fits_verified": 0,
             "posterior_medians_verified": 0, "convergence_flags_verified": 0,
             "paired_intervals_recomputed": 0, "gmfe_estimates_recomputed": 0,
             "max_recomputation_difference": 0.0, "endpoint_coverage": coverage,
             "coverage_interpretation": "Finite trajectories can have unscorable curves; compare paired common-case ratios, not unmatched conditional GMFEs."}
    audit["compatibility"] = verify_compatibility_amendment(directory, state)
    diagnostic_groups = {m: {"predictions": 0, "selected_converged": 0,
                            "selected_rhat_assessed": 0, "total_calls": [],
                            "call_lower_bounds": [], "fit_time_lower_bounds": [],
                            "unrecorded_call_jobs": 0, "wall_seconds": [],
                            "formula_fit_seconds": [], "unrecorded_fit_time_jobs": 0,
                            "failed_formula_fits": 0, "timed_out_formula_fits": 0,
                            "unattempted_formula_fits": 0, "interrupted_wall_seconds": [],
                            "other_failed_formula_fits": 0, "partial_formula_predictions": 0,
                            "unavailable_jobs": [], "computational_timeout_jobs": [],
                            "selected_dimension_counts": {}}
                         for m in protocol["methods"]}
    for key, ref in state["completed"].items():
        document = checked(ref)
        if (document["job"] != key or document["protocol"] != state["protocol"]
                or document["target_observations_used"]
                or document["prediction_locked_at"] > results["predictions_locked_at"]):
            raise ValueError("Unverified prediction-before-observation lock")
        audit["prediction_locks_verified"] += 1
        group = diagnostic_groups[document["method"]]
        group["wall_seconds"].append(float(document["wall_seconds"]))
        call_count, complete_call_count, elapsed, complete_elapsed = 0, True, 0.0, True
        fitted_formulae, has_timeout = 0, False
        for fitref in document["fits"]:
            fit = checked(fitref)
            if fit["protocol"] != state["protocol"] or fit["target_observations_used"]:
                raise ValueError("Unverified inference identity or target isolation")
            audit["formula_fits_verified"] += 1
            cost = fit_cost(fit)
            if cost["calls"] is None:
                complete_call_count = False
            else:
                call_count += cost["calls"]
            if cost["elapsed_seconds"] is None:
                complete_elapsed = False
            else:
                elapsed += cost["elapsed_seconds"]
            group["timed_out_formula_fits"] += int(cost["interrupted"])
            group["unattempted_formula_fits"] += int(cost["unattempted"])
            if cost["interrupted"]:
                group["interrupted_wall_seconds"].append(cost["elapsed_seconds"])
                has_timeout = True
            if fit["status"] == "fitted":
                fitted_formulae += 1
                calls = cost["calls"]
                if calls is None or int(calls) < 1:
                    raise ValueError("A fitted external method lacks actual simulator-call accounting")
                inference = fit["inference"]
                if fit["formula"] == document.get("formula"):
                    dimension = str(len(inference["selected_indices"]))
                    group["selected_dimension_counts"][dimension] = group["selected_dimension_counts"].get(dimension, 0)+1
                if fit.get("posterior") and sha(fit["posterior"]["path"]) != fit["posterior"]["sha256"]:
                    raise ValueError("Changed locked posterior samples")
                with np.load(inside(fit["posterior"]["path"]), allow_pickle=False) as archive:
                    samples = archive["log_parameters"]
                    if (samples.ndim != 2 or samples.shape[1] != 4 or not np.isfinite(samples).all()
                            or not np.allclose(np.median(samples, axis=0), fit["parameters"], rtol=1e-12, atol=1e-12)):
                        raise ValueError("Locked point parameters differ from posterior medians")
                    declared = jobs[key]["adapter"]["formulas"][fit["formula"]]
                    if (np.any(samples < np.asarray(declared["lower"])-1e-12)
                            or np.any(samples > np.asarray(declared["upper"])+1e-12)):
                        raise ValueError("Posterior sample outside frozen legal bounds")
                audit["posterior_medians_verified"] += 1
            else:
                group["failed_formula_fits"] += 1
                group["other_failed_formula_fits"] += int(not cost["interrupted"] and not cost["unattempted"])
        if has_timeout:
            group["computational_timeout_jobs"].append(key)
        group["call_lower_bounds"].append(call_count)
        group["fit_time_lower_bounds"].append(elapsed)
        if complete_call_count:
            group["total_calls"].append(call_count)
        else:
            group["unrecorded_call_jobs"] += 1
        if complete_elapsed:
            group["formula_fit_seconds"].append(elapsed)
        else:
            group["unrecorded_fit_time_jobs"] += 1
        if document["status"] == "predicted":
            group["partial_formula_predictions"] += int(fitted_formulae < 2)
            curve = checked(document["prediction"])
            concentration = curve["concentration"]
            if not concentration or any(not math.isfinite(float(v)) or float(v) < 0 for v in concentration):
                raise ValueError("Published predictor has a nonfinite/negative target trajectory")
            group["predictions"] += 1
            selected_diagnostics = document["inference_diagnostics"]
            if document["method"] != "snpe_c":
                if selected_diagnostics.get("rank_normalized_split_rhat"):
                    group["selected_rhat_assessed"] += 1
                    rhat = selected_diagnostics["rank_normalized_split_rhat"]
                    ess = selected_diagnostics["bulk_ess"] + selected_diagnostics["tail_ess"]
                    quality = (all(v is not None and math.isfinite(v) for v in rhat+ess)
                               and max(rhat) <= 1.05 and min(ess) >= 100)
                    if quality != selected_diagnostics["converged"]:
                        raise ValueError("Convergence flag disagrees with its explicit R-hat/ESS thresholds")
                    audit["convergence_flags_verified"] += 1
                group["selected_converged"] += int(selected_diagnostics.get("converged") is True)
            elif selected_diagnostics.get("convergence_assessed") is not False:
                raise ValueError("SNPE must not be presented as MCMC-converged")
        else:
            group["unavailable_jobs"].append({"job": key, "classification":
                "computational_timeout" if has_timeout else "numerical_or_inference_failure"})
    for cohort in ("all_thirty", "added_twenty"):
        wanted = {k for k, c in cases.items() if cohort == "all_thirty" or c["cohort"] == cohort}
        for method, _ in METHODS:
            for endpoint in ("cmax", "curve"):
                subset = [r for r in rows if r["case_id"] in wanted and r["method"] == method
                          and r["endpoint"] == endpoint and r["info"] == "time_five"
                          and r["transfer"] == "linear_weight" and r["error"] not in ("", None)]
                if len({r["case_id"] for r in subset}) != len(subset):
                    raise ValueError("Duplicate drug-level metric in published comparison")
                errors = [float(r["error"]) for r in subset]
                stat = results["summary"][cohort]["time_five"]["methods"][method][endpoint]
                if len(errors) != stat["n_cases"]:
                    raise ValueError("Metric denominator mismatch")
                if coverage[cohort][method][endpoint]["scored"] != stat["n_cases"]:
                    raise ValueError("Metric coverage differs from endpoint classification")
                actual = math.exp(sum(math.log(v) for v in errors)/len(errors)) if errors else None
                expected = stat["estimate"]
                if actual is None:
                    if expected is not None:
                        raise ValueError("Metric must be absent without scored cases")
                elif not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("GMFE differs from independently recomputed endpoint CSV")
                else:
                    audit["max_recomputation_difference"] = max(
                        audit["max_recomputation_difference"], abs(actual-expected))
                audit["gmfe_estimates_recomputed"] += 1
        for method, _ in METHODS[1:]:
            for endpoint in ("cmax", "curve"):
                def values(model):
                    return {row["case_id"]: float(row["error"]) for row in rows
                            if row["case_id"] in wanted and row["method"] == model
                            and row["endpoint"] == endpoint and row["info"] == "time_five"
                            and row["transfer"] == "linear_weight" and row["error"] not in ("", None)}
                numerator, denominator = values("pbpk"), values(method)
                ratios = {key: math.exp(math.log(numerator[key])-math.log(denominator[key]))
                          for key in numerator.keys() & denominator.keys()}
                actual = recompute_paired_interval(ratios, cases, seed=protocol["bootstrap"]["seed"],
                                                   resamples=protocol["bootstrap"]["resamples"])
                expected = results["summary"][cohort]["time_five"]["comparisons"]["pbpk/"+method][endpoint]
                for key in ("estimate", "ci95", "n_cases", "n_clusters"):
                    if actual[key] is None:
                        if expected[key] is not None:
                            raise ValueError("Paired interval missingness differs from recomputation")
                    elif not np.allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12):
                        raise ValueError("Paired cluster-bootstrap result differs from independent resampling")
                improved, worse = sum(v < 1-1e-10 for v in ratios.values()), sum(v > 1+1e-10 for v in ratios.values())
                if (improved != expected["improved"] or worse != expected["worse"]
                        or len(ratios)-improved-worse != expected["tied"]):
                    raise ValueError("Paired improvement/regression counts changed")
                audit["paired_intervals_recomputed"] += 1
    baseline_costs = load_baseline_costs(protocol, checked=checked)
    audit["baseline_costs"] = baseline_costs
    return {"state": state, "protocol": protocol, "results": results,
            "baseline_costs": baseline_costs,
            "diagnostics": diagnostic_groups, "coverage": coverage, "audit": audit}


def render(bundle: dict) -> str:
    result, protocol, diagnostics = bundle["results"], bundle["protocol"], bundle["diagnostics"]
    summary, baseline_costs = result["summary"], bundle["baseline_costs"]
    labels = dict(METHODS)
    labels["pbpk"] = "PBPK blueprint (ours)"
    lines = [r"% Generated from locked predictions and verified saved computation records.",
        r"\begingroup", r"\definecolor{BPIHead}{HTML}{EDF3F5}",
        r"\definecolor{BPIAccent}{HTML}{215168}",
        r"\begin{table}[!htbp]\centering",
        r"\captionsetup{name=Table,font=footnotesize}",
        r"\caption{Predictive accuracy and recorded computation cost on the same five development observations.}"
        r"\label{tab:blueprint_reviewer_comparators}",
        r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.18}\fontsize{8.5}{10.8}\selectfont",
        r"\begin{tabularx}{\linewidth}{@{}Xrrrrrr@{}}\toprule",
        r"\rowcolor{BPIHead}\multicolumn{7}{@{}l}{\color{BPIAccent}\textbf{A\quad Predictive accuracy (drug-equal GMFE)}}\\",
        r"& \multicolumn{3}{c}{All 30 drugs} & \multicolumn{3}{c}{Added 20 drugs}\\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Method & $n$ & Sampled $C_{\max}$ & Curve & $n$ & Sampled $C_{\max}$ & Curve\\\midrule"]
    for method, _ in METHODS:
        cells = []
        for cohort, enrolled in (("all_thirty", 30), ("added_twenty", 20)):
            stats = summary[cohort]["time_five"]["methods"][method]
            if stats["cmax"]["n_cases"] != stats["curve"]["n_cases"]:
                raise ValueError("Shared coverage column requires identical endpoint coverage")
            cells += [f"{stats['curve']['n_cases']}/{enrolled}",
                      number(stats["cmax"]["estimate"]), number(stats["curve"]["estimate"])]
        if method == "pbpk":
            lines.append(r"\rowcolor{BPIHead}")
        lines.append(labels[method] + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabularx}\par\vspace{7pt}",
        r"\begin{tabularx}{\linewidth}{@{}Xrcc@{}}\toprule",
        r"\rowcolor{BPIHead}\multicolumn{4}{@{}l}{\color{BPIAccent}\textbf{B\quad Paired ours/control GMFE ratio [95\% CI]}}\\",
        r"Control & $n$ & Sampled $C_{\max}$ & Curve\\\midrule"]
    for method, _ in METHODS[1:]:
        values = summary["all_thirty"]["time_five"]["comparisons"]["pbpk/"+method]
        if values["cmax"]["n_cases"] != values["curve"]["n_cases"]:
            raise ValueError("Shared paired coverage column requires identical endpoint coverage")
        lines.append(labels[method] + f" & {values['curve']['n_cases']}/30 & "
                     + interval(values["cmax"]) + " & " + interval(values["curve"]) + r"\\")
    lines += [r"\bottomrule\end{tabularx}\par\vspace{7pt}",
        r"\begin{tabularx}{\linewidth}{@{}Xrrl@{}}\toprule",
        r"\rowcolor{BPIHead}\multicolumn{4}{@{}l}{\color{BPIAccent}\textbf{C\quad Recorded computation cost (median per drug)}}\\",
        r"Method & Model calls & Time (s) & Timing scope\\\midrule"]
    for method, _ in METHODS:
        if method in baseline_costs:
            cost = baseline_costs[method]
            call_median, fit_time = cost["median_calls"], cost["median_seconds"]
            missing_calls, missing_time = False, False
        else:
            d = diagnostics[method]
            call_median = statistics.median(d["call_lower_bounds"])
            fit_time = statistics.median(d["fit_time_lower_bounds"])
            missing_calls, missing_time = d["unrecorded_call_jobs"], d["unrecorded_fit_time_jobs"]
        calls = f"{call_median:,.0f}" if call_median == int(call_median) else f"{call_median:,.1f}"
        if missing_calls:
            calls = r"$\geq$" + calls
        timing = number(fit_time, 3 if method == "two_compartment" else 1)
        if missing_time:
            timing = r"$\geq$" + timing
        scope = r"Task$^{\dagger}$" if method == "two_compartment" else "Parameter fit"
        if method == "pbpk":
            lines.append(r"\rowcolor{BPIHead}")
        lines.append(labels[method] + f" & {calls} & {timing} & {scope}" + r"\\")
    lines += [r"\bottomrule\end{tabularx}",
        r"\par\vspace{4pt}\begin{minipage}{\linewidth}\fontsize{7.5}{9.8}\selectfont",
        r"Lower GMFE is better; $n$ = scored/enrolled drugs. Ratios $<1$ favor ours; CIs use "
        + f"{protocol['bootstrap']['resamples']:,}" + r" paired study-cluster bootstrap draws, without multiplicity adjustment. "
        r"Costs use all 30 drugs and both PBPK formulae. $^{\dagger}$Task time also includes prediction and I/O. "
        r"$\geq$ marks unrecorded interrupted calls. Timing scopes and compute differ; these are not prediction latencies or controlled speed comparisons. "
        r"Details: Appendix~\ref{app:blueprint_comparator_settings}.",
        r"\end{minipage}\end{table}\endgroup"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--check-only", action="store_true", help="verify complete inputs without writing the table")
    args = parser.parse_args()
    bundle = validate_and_load(args.experiment)
    content = render(bundle)
    if not args.check_only:
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        TABLE.write_text(content)
        bundle["audit"]["table"] = reference(TABLE)
    bundle["audit"]["renderer"] = reference(Path(__file__))
    bundle["audit"]["cost_extractor"] = reference(ROOT / "scripts/published_blueprint_costs.py")
    bundle["audit"]["check_only"] = args.check_only
    bundle["audit"]["diagnostics"] = bundle["diagnostics"]
    # Full per-case detail remains reproducible via validate_and_load and the
    # source CSVs. Keep terminal status compact rather than repeating every
    # absent clinical reference and every raw per-job duration.
    audit = bundle["audit"]
    for methods in audit["endpoint_coverage"].values():
        for endpoints in methods.values():
            for counts in endpoints.values():
                counts.pop("unscored_cases", None)
    for method, values in list(audit["diagnostics"].items()):
        audit["diagnostics"][method] = {k: v for k, v in values.items()
            if not isinstance(v, list) or k in ("unavailable_jobs", "computational_timeout_jobs")}
        compact = audit["diagnostics"][method]
        compact["median_calls_lower_bound"] = statistics.median(values["call_lower_bounds"])
        compact["median_cumulative_fit_seconds_lower_bound"] = statistics.median(values["fit_time_lower_bounds"])
        compact["recorded_interrupted_wall_seconds"] = sum(values["interrupted_wall_seconds"])
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
