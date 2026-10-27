"""Deterministic, failure-preserving aggregation of preregistered PBPK jobs.

This module reads no files, runs no solver, selects no model, and imports no
GPU framework. ``planned_jobs`` fixes case/compound/tier/method/seed denominators.
``case_results[case_id]`` may contain decoded ``results``, ``external_results``,
``completion``, plus ``queue_status`` and ``queue_error``. Missing results remain
pending; an explicitly failed queue is failed, never silently pending.

Job states: pending, running, awaiting_evaluation, blocked, failed,
missing_result, missing_evaluation, invalid_report, evaluation_failed, scored.
scored_with_fallback is also terminal. Submission status is retained separately:
a blocked submission with a valid common fallback is not an executed method.
Historical development/reconstruction and main_holdout are always separate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math


TIERS = frozenset({"main_holdout", "bottom_up_external", "reconstruction", "development"})
TERMINAL_STATES = frozenset({"blocked", "failed", "missing_result", "missing_evaluation",
                             "invalid_report", "evaluation_failed", "scored", "scored_with_fallback"})
_SCORED_STATES = {"scored", "scored_with_fallback"}
_FINISHED_QUEUES = {"completed", "completed_with_blocked_methods", "completed_with_unsuccessful_methods", "evaluated"}
_FAILED_QUEUES = {"failed", "evaluation_failed", "interrupted"}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _number(value, *, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _identity(row):
    return row.get("method"), row.get("seed")


def _plan(planned_jobs):
    if not isinstance(planned_jobs, list):
        raise ValueError("planned_jobs must be a list")
    seen, cases, plan = set(), {}, []
    for source in planned_jobs:
        row = deepcopy(source)
        for field in ("case_id", "compound_id", "tier", "method"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"A planned job requires {field}")
        if type(row.get("seed")) is not int or row["tier"] not in TIERS:
            raise ValueError("Use an integer seed and an explicit registered tier")
        if row.get("eligibility_status", "ready") not in {"ready", "blocked"}:
            raise ValueError("eligibility_status must be ready or blocked")
        for field in ("expected_scenario_ids", "expected_study_group_ids"):
            if field in row:
                values = row[field]
                if (not isinstance(values, list) or any(not isinstance(x, str) or not x.strip() for x in values)
                        or len(values) != len(set(values))):
                    raise ValueError(f"{field} must contain unique nonempty IDs")
                row[field] = sorted(values)
        key = (row["case_id"], row["method"], row["seed"])
        if key in seen:
            raise ValueError("Duplicate preregistered job")
        seen.add(key)
        case_identity = (row["compound_id"], row["tier"])
        if cases.setdefault(row["case_id"], case_identity) != case_identity:
            raise ValueError("A case cannot change compound or evaluation tier")
        row.setdefault("eligibility_status", "ready")
        plan.append(row)
    return sorted(plan, key=lambda r: (r["tier"], r["method"], r["compound_id"], r["seed"], r["case_id"]))


def _report_rows(report, job):
    if _identity(report) != _identity(job):
        raise ValueError("Report method/seed differs from the registered job")
    if report.get("prediction_integrity_verified") is not True or report.get("candidate_selection_performed") is not False:
        raise ValueError("Final report lacks prediction integrity or permits model selection")
    fingerprint = report.get("predictions_sha256")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in fingerprint)):
        raise ValueError("Final report lacks the prediction lock")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Final report has no registered curve rows")
    retained, scenarios = [], set()
    for row in rows:
        if row.get("compound_id") != job["compound_id"] or row.get("tier") != job["tier"]:
            raise ValueError("Report compound/tier does not match the registered case")
        for field in ("scenario_id", "study_group_id", "curve_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("Every curve requires scenario, study and physical curve IDs")
        if row["scenario_id"] in scenarios:
            raise ValueError("Repeated scenario result")
        scenarios.add(row["scenario_id"])
        if job["tier"] == "main_holdout":
            # Import lazily: ordinary progress aggregation needs no torch import.
            from .heldout_evaluation import _main_provenance
            if row.get("main_holdout_provenance_verified") is not True:
                raise ValueError("A main_holdout label is not a validated study qualification")
            _main_provenance(row)
        if row.get("status") in {"scored", "scored_with_fallback"}:
            if not _number(row.get("log_rmse")) or not _number(row.get("gmfe"), positive=True):
                raise ValueError("Scored curve has missing/nonfinite metrics")
        elif row.get("log_rmse") is not None or row.get("gmfe") is not None:
            raise ValueError("Unscorable curve must not carry numeric success metrics")
        if row.get("duplicate_of") is None:
            retained.append(deepcopy(row))
    if not retained:
        raise ValueError("A report cannot contain only duplicate aliases")
    for field, observed in (("expected_scenario_ids", scenarios),
                            ("expected_study_group_ids", {r["study_group_id"] for r in rows})):
        if field in job and set(job[field]) != observed:
            raise ValueError("Final report does not match the preregistered " + field)
    return retained


def _job_result(job, bundle):
    item = {**job, "status": "pending", "reason": job.get("reason"), "method_status": None,
            "submission_status": "pending",
            "executed": False, "evaluation_present": False, "selected_candidate_id": None,
            "curve_count": 0, "scored_curve_count": 0, "native_scored_curve_count": 0}
    if job["eligibility_status"] == "blocked":
        item.update(status="blocked", submission_status="blocked")
        return item, []
    if not isinstance(bundle, dict):
        raise ValueError("Each case_results entry must be an object")
    queue = bundle.get("queue_status") or (bundle.get("completion") or {}).get("status")
    methods = (bundle.get("results") or {}).get("methods", [])
    matching = [r for r in methods if _identity(r) == _identity(job)]
    if len(matching) > 1:
        item.update(status="invalid_report", reason="Duplicate method result; no best-run selection permitted")
        return item, []
    method = matching[0] if matching else None
    if method:
        status = method.get("status")
        calls = (method.get("budget") or {}).get("totals", {}).get("logical_calls", 0)
        item.update(method_status=status, submission_status=status, selected_candidate_id=method.get("selected_candidate_id"),
                    executed=status != "blocked" and bool(calls or method.get("evaluated_candidates")))
        if status == "blocked":
            item.update(status="blocked", reason=method.get("blocked_reason") or "method_blocked")
    else:
        if queue in _FAILED_QUEUES or queue == "blocked":
            status = "blocked" if queue == "blocked" else "failed"
            item.update(status=status, submission_status=status, reason=bundle.get("queue_error") or "case_queue_" + queue)
        elif queue in _FINISHED_QUEUES or queue == "predictions_locked":
            item.update(status="missing_result", reason="Finished case omitted a preregistered method")
        elif queue == "running":
            item["status"] = "running"
        return item, []
    external = bundle.get("external_results") or {}
    matching = [r for r in external.get("evaluations", []) if _identity(r) == _identity(job)]
    if len(matching) > 1:
        item.update(status="invalid_report", reason="Duplicate external report; no seed/run cherry-picking permitted")
        return item, []
    if not matching:
        if method.get("status") == "blocked":
            return item, []
        if method.get("status") in {"failed", "error"} or queue in _FAILED_QUEUES:
            item.update(status="failed", reason=bundle.get("queue_error") or method.get("error") or "method_failed")
        elif queue in _FINISHED_QUEUES:
            item.update(status="missing_evaluation", reason="Finished case has no external evaluation")
        else:
            item["status"] = "awaiting_evaluation"
        return item, []
    item["evaluation_present"] = True
    try:
        if external.get("compound_id", job["compound_id"]) != job["compound_id"]:
            raise ValueError("External result compound differs from registered compound")
        entry = matching[0]
        if entry.get("tier", job["tier"]) != job["tier"]:
            raise ValueError("External result tier differs from registered tier")
        report = entry["report"]
        if report.get("candidate_id") != item["selected_candidate_id"]:
            raise ValueError("External prediction differs from the method's locked selected candidate")
        rows = _report_rows(report, job)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        item.update(status="invalid_report", reason=str(exc))
        return item, []
    scored = [r for r in rows if r.get("log_rmse") is not None]
    item.update(curve_count=len(rows), scored_curve_count=len(scored),
                native_scored_curve_count=sum(not r.get("fallback_used", False) for r in scored),
                status=("scored_with_fallback" if any(r.get("fallback_used", False) for r in scored) else "scored")
                       if len(scored) == len(rows) else "evaluation_failed")
    if item["status"] == "evaluation_failed":
        item["reason"] = "; ".join(sorted({r.get("error", r["status"]) for r in rows if r.get("log_rmse") is None}))
    return item, rows


def _curve_key(row):
    return row["study_group_id"], row["curve_id"]


def _curve_signature(row):
    # Canonical physical IDs are supplied by the data audit, never by a ranking.
    return _canonical({k: row.get(k) for k in ("observation_sha256", "status", "log_rmse", "gmfe", "fallback_used", "endpoints")})


def _cross_case_checks(jobs, curves):
    """A later case cannot promote a development study or cherry-pick its alias."""
    development = set()
    memberships = defaultdict(list)
    for job in jobs:
        key = (job["case_id"], job["method"], job["seed"])
        for row in curves[key]:
            development.update(row.get("development_study_group_ids", []))
            if row["tier"] in {"development", "reconstruction"} or row.get("study_role") in {"calibration", "selection"}:
                development.add(row["study_group_id"])
            membership = (job["tier"], job["method"], job["seed"], job["compound_id"], *_curve_key(row))
            memberships[membership].append((job, row))
    for members in memberships.values():
        if len({_curve_signature(row) for _, row in members}) > 1:
            for job, _ in members:
                job.update(status="invalid_report", reason="Conflicting results for one physical curve across cases")
    for job in jobs:
        key = (job["case_id"], job["method"], job["seed"])
        if job["tier"] == "main_holdout" and any(r["study_group_id"] in development for r in curves[key]):
            job.update(status="invalid_report", reason="A final study also appears in a development/reconstruction case",
                       scored_curve_count=0, native_scored_curve_count=0)
            for row in curves[key]:
                row.update(status="qualification_failed", log_rmse=None, gmfe=None)


def _seed_result(compound, seed, jobs, curves):
    unique, conflict, duplicates = {}, set(), 0
    for row in curves:
        key = _curve_key(row)
        if key in unique:
            duplicates += 1
            if _curve_signature(unique[key]) != _curve_signature(row):
                conflict.add(key)
        else:
            unique[key] = row
    available = [r for key, r in sorted(unique.items()) if key not in conflict and r.get("log_rmse") is not None]
    def metric(name, *, geometric=False, native_only=False):
        studies = defaultdict(list)
        for row in available:
            if native_only and row.get("fallback_used", False):
                continue
            value = row[name]
            studies[row["study_group_id"]].append(math.log(value) if geometric else value)
        result = _mean([_mean(v) for _, v in sorted(studies.items())])
        return math.exp(result) if geometric and result is not None else result
    complete = bool(unique) and not conflict and all(j["status"] in _SCORED_STATES for j in jobs) and len(available) == len(unique)
    result = {"seed": seed, "planned_jobs": len(jobs), "complete": complete,
              "unique_curve_count": len(unique), "duplicate_curve_count": duplicates, "conflicting_curve_count": len(conflict),
              "scored_curve_count": len(available), "native_scored_curve_count": sum(not r.get("fallback_used", False) for r in available),
              "study_group_ids": sorted({k[0] for k in unique}),
              "mean_curve_log_rmse": metric("log_rmse") if complete else None,
              "gmfe": metric("gmfe", geometric=True) if complete else None,
              "available_curve_log_rmse": metric("log_rmse"), "available_curve_gmfe": metric("gmfe", geometric=True),
              "native_success_only_log_rmse": metric("log_rmse", native_only=True)}
    return result, set(unique)


def _method_summary(jobs, curves_by_job):
    grouped = defaultdict(list)
    for job in jobs:
        grouped[job["compound_id"]].append(job)
    drugs, physical_curves, global_studies, memberships = [], set(), set(), set()
    for compound, drug_jobs in sorted(grouped.items()):
        seeds = []
        for seed in sorted({j["seed"] for j in drug_jobs}):
            selected = [j for j in drug_jobs if j["seed"] == seed]
            curves = [r for j in selected for r in curves_by_job[(j["case_id"], j["method"], seed)]]
            result, physical = _seed_result(compound, seed, selected, curves)
            seeds.append(result)
            for study, curve in physical:
                global_studies.add(study); memberships.add((compound, study)); physical_curves.add((compound, study, curve))
        complete = all(s["complete"] for s in seeds)
        def across_seeds(field, geometric=False):
            values = [s[field] for s in seeds if s[field] is not None]
            value = _mean([math.log(x) for x in values] if geometric else values)
            return math.exp(value) if geometric and value is not None else value
        drugs.append({"compound_id": compound, "planned_seed_count": len(seeds), "complete": complete,
                      "mean_curve_log_rmse": across_seeds("mean_curve_log_rmse") if complete else None,
                      "gmfe": across_seeds("gmfe", True) if complete else None,
                      "available_curve_log_rmse": across_seeds("available_curve_log_rmse"),
                      "available_curve_gmfe": across_seeds("available_curve_gmfe", True),
                      "native_success_only_log_rmse": across_seeds("native_success_only_log_rmse"), "seeds": seeds})
    complete = bool(drugs) and all(d["complete"] for d in drugs)
    def across_drugs(field, geometric=False):
        values = [d[field] for d in drugs if d[field] is not None]
        value = _mean([math.log(x) for x in values] if geometric else values)
        return math.exp(value) if geometric and value is not None else value
    scored_jobs = sum(j["status"] in _SCORED_STATES for j in jobs)
    native_jobs = sum(j["status"] in _SCORED_STATES and j["native_scored_curve_count"] == j["curve_count"] for j in jobs)
    return {"planned_jobs": len(jobs), "status_counts": dict(sorted(Counter(j["status"] for j in jobs).items())),
            "submission_status_counts": dict(sorted(Counter(j["submission_status"] for j in jobs).items())),
            "executed_jobs": sum(j["executed"] for j in jobs), "scored_jobs": scored_jobs,
            "job_coverage": scored_jobs / len(jobs), "policy_job_coverage": scored_jobs / len(jobs), "native_job_coverage": native_jobs / len(jobs),
            "planned_compound_count": len(drugs), "complete_compound_count": sum(d["complete"] for d in drugs),
            "available_compound_count": sum(d["available_curve_log_rmse"] is not None for d in drugs),
            "unique_global_study_groups": len(global_studies), "drug_study_memberships": len(memberships),
            "unique_observed_curves": len(physical_curves), "analysis_complete": complete,
            "mean_curve_log_rmse": across_drugs("mean_curve_log_rmse") if complete else None,
            "gmfe": across_drugs("gmfe", True) if complete else None,
            "complete_compound_subset_log_rmse": across_drugs("mean_curve_log_rmse"),
            "complete_compound_subset_gmfe": across_drugs("gmfe", True),
            "available_curve_log_rmse": across_drugs("available_curve_log_rmse"),
            "available_curve_gmfe": across_drugs("available_curve_gmfe", True),
            "native_success_only_log_rmse": across_drugs("native_success_only_log_rmse"), "by_compound": drugs}


def summarize_cohort_results(*, planned_jobs: list[dict], case_results: dict[str, dict]) -> dict:
    """Aggregate a frozen plan without dropping unavailable drugs or bad seeds.

    Curves are equally weighted within study, studies within each drug/seed,
    preregistered seeds within drug, then drugs. Seeds/duplicate case aliases do
    not increase the independent drug/study count. Full means are null whenever
    any planned job/curve is missing or invalid; explicitly conditional summaries
    and coverage remain available. No score, observation or time is imputed.
    """
    plan = _plan(planned_jobs)
    if not isinstance(case_results, dict):
        raise ValueError("case_results must map case IDs to decoded result bundles")
    jobs, curves = [], {}
    for job in plan:
        result, rows = _job_result(job, case_results.get(job["case_id"], {}))
        jobs.append(result)
        curves[(job["case_id"], job["method"], job["seed"])] = rows
    _cross_case_checks(jobs, curves)
    tiers = {}
    for tier in sorted({j["tier"] for j in jobs}):
        tier_jobs = [j for j in jobs if j["tier"] == tier]
        tiers[tier] = {"methods": {method: _method_summary([j for j in tier_jobs if j["method"] == method], curves)
                                  for method in sorted({j["method"] for j in tier_jobs})}}
    return {"schema_version": 1, "planned_jobs_sha256": hashlib.sha256(_canonical(plan).encode()).hexdigest(),
            "planned_jobs": len(jobs), "planned_cases": len({j["case_id"] for j in jobs}),
            "planned_compounds": len({j["compound_id"] for j in jobs}),
            "status_counts": dict(sorted(Counter(j["status"] for j in jobs).items())),
            "submission_status_counts": dict(sorted(Counter(j["submission_status"] for j in jobs).items())),
            "terminal_jobs": sum(j["status"] in TERMINAL_STATES for j in jobs),
            "all_jobs_terminal": bool(jobs) and all(j["status"] in TERMINAL_STATES for j in jobs),
            "executed_jobs": sum(j["executed"] for j in jobs),
            "aggregation": "curve_RMSE_then_equal_studies_then_equal_registered_seeds_within_drug_then_equal_drugs",
            "estimand": "locked_submission_with_common_prelocked_fallback_when_available; native_success_reported_separately",
            "seeds_are_independent_drugs": False, "candidate_selection_performed": False,
            "scientific_acceptance_inferred": False, "jobs": jobs, "tiers": tiers}
