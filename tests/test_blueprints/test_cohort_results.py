"""Small synthetic aggregation fixtures; no model, solver, network or GPU."""
from copy import deepcopy
import math

import pytest

from pbpk_kb.blueprints.cohort_results import summarize_cohort_results
from pbpk_kb.blueprints.heldout_evaluation import evaluate_locked_predictions, lock_predictions


def job(case="a", drug="drug_a", method="M6", seed=17, tier="development", **extra):
    return dict(case_id=case, compound_id=drug, method=method, seed=seed, tier=tier, **extra)


def row(name="curve", study="study", drug="drug_a", error=1.0, tier="development", **extra):
    return dict(scenario_id=name, curve_id=name, study_group_id=study, compound_id=drug, tier=tier,
                study_role="external", status="scored", duplicate_of=None, fallback_used=False,
                log_rmse=error, gmfe=math.exp(error), endpoints={}, **extra)


def bundle(jobs, rows=None, *, queue="completed", status="completed"):
    methods, evaluations = [], []
    for j in jobs:
        candidate = None if status == "blocked" else "locked"
        methods.append(dict(method=j["method"], seed=j["seed"], status=status,
                            selected_candidate_id=candidate, evaluated_candidates=0 if status == "blocked" else 1,
                            blocked_reason="QWEN_PROPOSER_UNAVAILABLE" if status == "blocked" else None))
        if rows is not None:
            evaluations.append(dict(method=j["method"], seed=j["seed"], tier=j["tier"], report=dict(
                method=j["method"], seed=j["seed"], candidate_id=candidate, rows=deepcopy(rows),
                prediction_integrity_verified=True, candidate_selection_performed=False, predictions_sha256="a" * 64)))
    return dict(results=dict(methods=methods), external_results=dict(evaluations=evaluations), queue_status=queue)


def method_summary(result, method="M6", tier="development"):
    return result["tiers"][tier]["methods"][method]


def test_blocked_missing_and_failed_drugs_never_leave_denominator():
    plan = [job(), job("b", "drug_b", eligibility_status="blocked", reason="no_data"),
            job("c", "drug_c"), job("d", "drug_d")]
    out = summarize_cohort_results(planned_jobs=plan, case_results={
        "a": bundle([plan[0]], [row(error=2)]), "c": {"queue_status": "failed", "queue_error": "disk_guard"}})
    summary = method_summary(out)
    assert out["status_counts"] == {"blocked": 1, "failed": 1, "pending": 1, "scored": 1}
    assert out["executed_jobs"] == 1
    assert summary["planned_compound_count"] == 4
    assert summary["mean_curve_log_rmse"] is None
    assert summary["complete_compound_subset_log_rmse"] == 2
    assert summary["job_coverage"] == .25
    assert summary["native_job_coverage"] == .25


def test_m2_three_blocked_seeds_are_three_missing_jobs_not_successes():
    plan = [job(method="M2", seed=seed) for seed in (17, 43, 101)]
    out = summarize_cohort_results(planned_jobs=plan, case_results={"a": bundle(plan, status="blocked")})
    summary = method_summary(out, "M2")
    assert out["all_jobs_terminal"] is True
    assert out["executed_jobs"] == 0
    assert summary["planned_jobs"] == 3
    assert summary["planned_compound_count"] == 1
    assert summary["mean_curve_log_rmse"] is None
    assert summary["job_coverage"] == 0


@pytest.mark.parametrize("queue,expected", [(None, "pending"), ("running", "running"),
    ("failed", "failed"), ("interrupted", "failed"), ("evaluation_failed", "failed"),
    ("completed", "missing_result"), ("evaluated", "missing_result"), ("blocked", "blocked")])
def test_missing_result_has_explicit_queue_state(queue, expected):
    out = summarize_cohort_results(planned_jobs=[job()], case_results={"a": {"queue_status": queue}})
    assert out["jobs"][0]["status"] == expected


@pytest.mark.parametrize("queue,expected", [("running", "awaiting_evaluation"),
    ("predictions_locked", "awaiting_evaluation"), ("evaluated", "missing_evaluation"),
    ("completed", "missing_evaluation"), ("failed", "failed")])
def test_missing_external_report_does_not_mean_zero_error(queue, expected):
    out = summarize_cohort_results(planned_jobs=[job()], case_results={"a": bundle([job()], queue=queue)})
    assert out["jobs"][0]["status"] == expected
    assert method_summary(out)["mean_curve_log_rmse"] is None


def test_curve_study_seed_drug_weighting_and_gmfe_are_hierarchical():
    plan = [job(seed=17), job(seed=43), job("b", "drug_b", seed=17)]
    first = bundle([plan[0]], [row("one", "s1", error=0), row("two", "s1", error=0), row("three", "s2", error=4)])
    second = bundle([plan[1]], [row("one", "s1", error=2), row("two", "s1", error=2), row("three", "s2", error=6)])
    first["results"]["methods"] += second["results"]["methods"]
    first["external_results"]["evaluations"] += second["external_results"]["evaluations"]
    out = summarize_cohort_results(planned_jobs=plan, case_results={"a": first,
        "b": bundle([plan[2]], [row(drug="drug_b", study="s3", error=9)])})
    summary = method_summary(out)
    # drug_a: mean(seed17 mean(0,4), seed43 mean(2,6))=3; drug_b=9.
    assert summary["mean_curve_log_rmse"] == pytest.approx(6)
    assert summary["gmfe"] == pytest.approx(math.exp(6))
    assert summary["planned_compound_count"] == 2
    assert summary["unique_global_study_groups"] == 3
    assert summary["unique_observed_curves"] == 4  # seed replication adds no curves/studies


def test_one_missing_seed_makes_full_drug_estimate_incomplete():
    plan = [job(seed=s) for s in (17, 43, 101)]
    partial = bundle([plan[0]], [row(error=1)], queue="running")
    out = summarize_cohort_results(planned_jobs=plan, case_results={"a": partial})
    summary = method_summary(out)
    assert summary["mean_curve_log_rmse"] is None
    assert summary["complete_compound_subset_log_rmse"] is None
    assert summary["available_curve_log_rmse"] == 1
    assert summary["job_coverage"] == pytest.approx(1 / 3)
    assert summary["by_compound"][0]["planned_seed_count"] == 3


def test_repeated_case_does_not_double_weight_same_physical_curve():
    plan = [job(), job("alias")]
    out = summarize_cohort_results(planned_jobs=plan, case_results={
        "a": bundle([plan[0]], [row("one", "s1", error=0), row("two", "s1", error=4)]),
        "alias": bundle([plan[1]], [row("one", "s1", error=0)])})
    summary = method_summary(out)
    assert summary["mean_curve_log_rmse"] == 2
    assert summary["unique_global_study_groups"] == 1
    assert summary["unique_observed_curves"] == 2
    assert summary["by_compound"][0]["seeds"][0]["duplicate_curve_count"] == 1


def test_conflicting_duplicate_curve_is_not_averaged_or_best_selected():
    plan = [job(), job("alias")]
    out = summarize_cohort_results(planned_jobs=plan, case_results={
        "a": bundle([plan[0]], [row(error=0)]), "alias": bundle([plan[1]], [row(error=2)])})
    summary = method_summary(out)
    assert summary["mean_curve_log_rmse"] is None
    assert summary["available_curve_log_rmse"] is None
    assert summary["by_compound"][0]["seeds"][0]["conflicting_curve_count"] == 1
    assert summary["job_coverage"] == 0
    assert out["status_counts"] == {"invalid_report": 2}


def test_shared_study_does_not_become_two_independent_studies():
    plan = [job(), job("b", "drug_b")]
    out = summarize_cohort_results(planned_jobs=plan, case_results={"a": bundle([plan[0]], [row()]),
        "b": bundle([plan[1]], [row(drug="drug_b")])})
    summary = method_summary(out)
    assert summary["unique_global_study_groups"] == 1
    assert summary["drug_study_memberships"] == 2


def test_reconstruction_never_enters_main_holdout_summary():
    plan = [job(tier="reconstruction"), job("b", "drug_b", tier="main_holdout", eligibility_status="blocked")]
    out = summarize_cohort_results(planned_jobs=plan, case_results={
        "a": bundle([plan[0]], [row(tier="reconstruction", error=1)])})
    assert method_summary(out, tier="reconstruction")["mean_curve_log_rmse"] == 1
    assert method_summary(out, tier="main_holdout")["mean_curve_log_rmse"] is None


def test_main_label_without_verified_provenance_is_invalid_not_strict_success():
    j = job(tier="main_holdout")
    out = summarize_cohort_results(planned_jobs=[j], case_results={"a": bundle([j], [row(tier="main_holdout")])})
    assert out["jobs"][0]["status"] == "invalid_report"
    assert method_summary(out, tier="main_holdout")["mean_curve_log_rmse"] is None


def test_actual_toy_heldout_report_carries_recheckable_main_qualification():
    j = job(tier="main_holdout")
    scenario = dict(scenario_id="s", curve_id="s", compound_id="drug_a", study_group_id="external",
        tier="main_holdout", role="test", development_study_group_ids=["fit", "selection"],
        development_cohort_ids=["fit_cohort", "selection_cohort"],
        study_reference=dict(study_id="external", role="test", source_id="source", source_checksum="a" * 64,
                             cohort_ids=["external_cohort"]),
        observations=[dict(time_min=i, concentration_umol_per_l=1.0) for i in range(1, 6)])
    curves = {"s": dict(time_min=[0, 5], concentration=[math.e, math.e], unit="umol/l", time_unit="min")}
    locked = dict(method="M6", seed=17, candidate_id="locked", predictions_sha256=lock_predictions(curves))
    report = evaluate_locked_predictions([scenario], curves, locked, device="cpu")
    b = bundle([j], [row()]); b["external_results"]["evaluations"][0]["report"] = report
    out = summarize_cohort_results(planned_jobs=[j], case_results={"a": b})
    assert method_summary(out, tier="main_holdout")["mean_curve_log_rmse"] == pytest.approx(1)
    assert report["rows"][0]["scoring_device_used"] == "cpu"
    # A separate case cannot reuse an already-development study as a main test.
    other = job("historical", "drug_a", tier="reconstruction")
    contaminated = summarize_cohort_results(planned_jobs=[j, other], case_results={
        "a": b, "historical": bundle([other], [row(study="external", tier="reconstruction")])})
    assert method_summary(contaminated, tier="main_holdout")["mean_curve_log_rmse"] is None
    assert next(r for r in contaminated["jobs"] if r["case_id"] == "a")["status"] == "invalid_report"
    report["rows"][0]["study_reference"]["cohort_ids"] = ["fit_cohort"]
    invalid = summarize_cohort_results(planned_jobs=[j], case_results={"a": b})
    assert invalid["jobs"][0]["status"] == "invalid_report"


def test_zero_prediction_is_not_repaired_or_deleted():
    j = job()
    scenario = dict(scenario_id="s", compound_id="drug_a", study_group_id="s", tier="development",
                    observations=[dict(time_min=0, concentration_umol_per_l=.24), dict(time_min=1, concentration_umol_per_l=.1)])
    curves = {"s": dict(time_min=[0, 1], concentration=[0, .1], unit="umol/l", time_unit="min")}
    locked = dict(method="M6", seed=17, candidate_id="locked", predictions_sha256=lock_predictions(curves))
    report = evaluate_locked_predictions([scenario], curves, locked, device="cpu")
    b = bundle([j], [row()]); b["external_results"]["evaluations"][0]["report"] = report
    out = summarize_cohort_results(planned_jobs=[j], case_results={"a": b})
    assert out["jobs"][0]["status"] == "evaluation_failed"
    assert out["jobs"][0]["curve_count"] == 1
    assert out["jobs"][0]["scored_curve_count"] == 0
    assert method_summary(out)["mean_curve_log_rmse"] is None
    assert report["rows"][0]["scoring_device_used"] is None


def test_order_independence_and_input_immutability():
    plan = [job(), job("b", "drug_b")]
    bundles = {"a": bundle([plan[0]], [row()]), "b": bundle([plan[1]], [row(drug="drug_b")])}
    before = deepcopy((plan, bundles))
    first = summarize_cohort_results(planned_jobs=plan, case_results=bundles)
    second = summarize_cohort_results(planned_jobs=list(reversed(plan)), case_results=dict(reversed(list(bundles.items()))))
    assert first == second
    assert (plan, bundles) == before


@pytest.mark.parametrize("mutation", ["seed", "tier", "compound", "candidate", "lock", "nan", "duplicate_report"])
def test_invalid_result_retains_job_and_rejects_numeric_score(mutation):
    j = job(); b = bundle([j], [row()]); report = b["external_results"]["evaluations"][0]["report"]
    if mutation == "seed": report["seed"] = 43
    elif mutation == "tier": report["rows"][0]["tier"] = "main_holdout"
    elif mutation == "compound": report["rows"][0]["compound_id"] = "other"
    elif mutation == "candidate": report["candidate_id"] = "cherry_picked"
    elif mutation == "lock": report["prediction_integrity_verified"] = False
    elif mutation == "nan": report["rows"][0]["log_rmse"] = float("nan")
    else: b["external_results"]["evaluations"] *= 2
    out = summarize_cohort_results(planned_jobs=[j], case_results={"a": b})
    assert out["planned_jobs"] == 1
    assert out["jobs"][0]["status"] == "invalid_report"
    assert method_summary(out)["mean_curve_log_rmse"] is None


def test_duplicate_plan_is_an_error_not_two_independent_jobs():
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_cohort_results(planned_jobs=[job(), job()], case_results={})


def test_fallback_has_coverage_but_not_native_success():
    j = job(); r = row(); r.update(fallback_used=True, status="scored_with_fallback")
    out = summarize_cohort_results(planned_jobs=[j], case_results={"a": bundle([j], [r])})
    summary = method_summary(out)
    assert summary["mean_curve_log_rmse"] == 1
    assert summary["job_coverage"] == 1
    assert summary["native_job_coverage"] == 0
    assert summary["native_success_only_log_rmse"] is None
    assert out["jobs"][0]["status"] == "scored_with_fallback"


def test_blocked_m2_can_have_prelocked_policy_fallback_without_native_success():
    plan = [job(method="M2", seed=seed) for seed in (17, 43, 101)]
    r = row(); r.update(fallback_used=True, status="scored_with_fallback")
    out = summarize_cohort_results(planned_jobs=plan, case_results={"a": bundle(plan, [r], status="blocked", queue="evaluated")})
    summary = method_summary(out, "M2")
    assert out["status_counts"] == {"scored_with_fallback": 3}
    assert out["submission_status_counts"] == {"blocked": 3}
    assert out["executed_jobs"] == 0
    assert summary["mean_curve_log_rmse"] == 1
    assert summary["policy_job_coverage"] == 1
    assert summary["native_job_coverage"] == 0
    assert summary["native_success_only_log_rmse"] is None
    assert all(j["reason"] == "QWEN_PROPOSER_UNAVAILABLE" for j in out["jobs"])


@pytest.mark.parametrize("field,ids", [("expected_scenario_ids", ["curve", "missing"]),
                                     ("expected_study_group_ids", ["study", "missing"])])
def test_preregistered_missing_study_or_curve_invalidates_truncated_report(field, ids):
    j = job(**{field: ids})
    out = summarize_cohort_results(planned_jobs=[j], case_results={"a": bundle([j], [row()])})
    assert out["jobs"][0]["status"] == "invalid_report"
    assert method_summary(out)["mean_curve_log_rmse"] is None
