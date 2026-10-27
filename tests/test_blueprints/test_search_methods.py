"""Strategy/callback tests with invented finite states, never a PBPK solver."""
from copy import deepcopy
import json

import pytest

from pbpk_kb.blueprints.design_diagnostics import BudgetExceeded
from pbpk_kb.blueprints.search_methods import run_method_search


def states():
    return [
        {"candidate_id": f"{structure}_{suffix}", "structure_id": structure,
         "parameter_ids": ["CL", "Pint"], "free_parameter_ids": free,
         "features": [float(structure == "b"), float("CL" in free), float("Pint" in free)],
         "requirement_score": 1.0 if structure == "b" else 0.0,
         "evidence_score": 1.0 if structure == "a" else 0.0,
         "structure_edit_count": int(structure == "b")}
        for structure in ("a", "b")
        for suffix, free in (("none", []), ("CL", ["CL"]), ("Pint", ["Pint"]), ("full", ["CL", "Pint"]))
    ]


def evaluator(losses=None, suggestions=None, calls=2):
    observed = []

    def evaluate(candidate, *, method_config):
        observed.append((deepcopy(candidate), deepcopy(method_config)))
        loss = (losses or {}).get(candidate["candidate_id"], 0.5)
        suggestion = (suggestions or {}).get(candidate["candidate_id"], candidate["free_parameter_ids"])
        return {"status": "completed", "selection_log_rmse": loss,
                "calibration_log_rmse": loss / 2, "solver_calls": calls,
                "identifiability": {"rank": len(suggestion), "selected_parameter_ids": suggestion},
                "effective_free_parameter_ids": candidate["free_parameter_ids"]}

    return evaluate, observed


def run(method, **kwargs):
    evaluate, _ = evaluator()
    return run_method_search(method=method, candidates=states(), evaluate=evaluate,
                             base_candidate_id="a_full", rule_candidate_id="b_full", **kwargs)


@pytest.mark.parametrize("method,candidate_id", [("M0", "a_full"), ("M1", "b_full")])
def test_fixed_methods_only_evaluate_registered_anchor(method, candidate_id):
    result = run(method)
    assert result["evaluated_candidate_count"] == 1
    assert result["selected_candidate"]["candidate_id"] == candidate_id
    assert result["solver_calls"] == 2


@pytest.mark.parametrize("method,reason", [("M0", "BASE_CANDIDATE_REQUIRED"), ("M1", "RULE_CANDIDATE_REQUIRED")])
def test_fixed_method_requires_explicit_anchor(method, reason):
    evaluate, observed = evaluator()
    result = run_method_search(method=method, candidates=states(), evaluate=evaluate)
    assert result["blocked_reason"] == reason
    assert not observed


def test_missing_qwen_does_not_silently_run_rules():
    result = run("M2")
    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "QWEN_PROPOSER_UNAVAILABLE"
    assert result["selected_candidate"] is None
    assert result["evaluations"] == []


def test_qwen_called_once_and_can_only_choose_registered_state():
    seen = []

    def propose(candidates, *, seed):
        seen.append((len(candidates), seed))
        return {"candidate_id": "b_CL"}

    result = run("M2", qwen_proposer=propose)
    assert seen == [(8, 17)]
    assert result["selected_candidate"]["candidate_id"] == "b_CL"
    assert result["evaluated_candidate_count"] == 1
    invalid = run("M2", qwen_proposer=lambda candidates, seed: "unregistered")
    assert invalid["blocked_reason"] == "QWEN_PROPOSAL_NOT_REGISTERED"
    assert not invalid["evaluations"]


def test_qwen_failure_is_explicit_and_error_text_is_not_copied():
    def fail(candidates, *, seed):
        raise RuntimeError("PRIVATE_SERVICE_DETAIL")

    result = run("M2", qwen_proposer=fail)
    assert result["blocked_reason"] == "QWEN_PROPOSER_FAILED"
    assert "PRIVATE_SERVICE_DETAIL" not in json.dumps(result)


def test_seeded_random_order_is_permutation_invariant():
    first = run("M3", max_candidates=4)
    evaluate, _ = evaluator()
    second = run_method_search(method="M3", candidates=list(reversed(states())), evaluate=evaluate,
                               base_candidate_id="a_full", rule_candidate_id="b_full", max_candidates=4)
    assert first == second
    assert first["evaluated_candidate_count"] == 4
    assert first["history"][:2] == second["history"][:2]
    assert first["history"][2:] != run("M3", seed=43, max_candidates=4)["history"][2:]


def test_random_forest_uses_real_observed_feedback_and_is_reproducible():
    pytest.importorskip("sklearn")
    first = run("M4", max_candidates=4)
    assert first == run("M4", max_candidates=4)
    assert any(row["reason"] == "random_forest_lower_confidence_bound" for row in first["history"])
    assert [row["candidate_id"] for row in first["history"][:2]] == ["a_full", "b_full"]


def test_m5_locks_structure_then_only_searches_its_subsets():
    evaluate, observed = evaluator({"a_full": .10, "b_full": .50, "a_CL": .08, "b_CL": .001})
    result = run_method_search(method="M5", candidates=states(), evaluate=evaluate,
                               base_candidate_id="a_full", rule_candidate_id="b_full")
    assert result["locked_structure_id"] == "a"
    assert len(observed) == 5
    assert all(c["free_parameter_ids"] == ["CL", "Pint"] for c, cfg in observed if cfg["phase"] == "structure")
    assert all(not cfg["identifiability_prescreen"] and not cfg["identifiability_feedback"] and cfg["fixed_parameterization"]
               for _, cfg in observed if cfg["phase"] == "structure")
    assert all(cfg["identifiability_prescreen"] and cfg["identifiability_feedback"] and not cfg["fixed_parameterization"]
               for _, cfg in observed if cfg["phase"] == "parameter_subset")
    assert all(c["structure_id"] == "a" for c, cfg in observed if cfg["phase"] == "parameter_subset")
    assert "b_CL" not in [c["candidate_id"] for c, _ in observed]
    assert result["selected_candidate"]["candidate_id"] == "a_CL"


def test_m5_rejects_pool_without_common_full_parameterization():
    evaluate, observed = evaluator()
    result = run_method_search(method="M5", candidates=[c for c in states() if c["candidate_id"] != "b_full"],
                               evaluate=evaluate)
    assert result["blocked_reason"] == "M5_COMMON_PARAMETERIZATION_MISSING"
    assert not observed


def test_m5_evaluator_must_not_change_structure_stage_parameterization():
    evaluate, _ = evaluator()

    def alter(candidate, *, method_config):
        result = evaluate(candidate, method_config=method_config)
        result["effective_free_parameter_ids"] = ["CL"]
        return result

    with pytest.raises(ValueError, match="must remain fixed"):
        run_method_search(method="M5", candidates=states(), evaluate=alter)


@pytest.mark.parametrize("scope,resumes", [("phase", True), ("global", False), (None, False)])
def test_m5_phase_batch_overflow_only_resumes_with_explicit_phase_scope(scope, resumes):
    observed = []

    def evaluate(candidate, *, method_config):
        observed.append(deepcopy(method_config))
        if len(observed) == 2:
            # Four units remain in the stage allocation, but the next batch
            # cannot fit. The reserved phase-two budget is still available.
            result = {"status": "budget_exhausted", "solver_calls": 0}
            if scope is not None:
                result["budget_scope"] = scope
            return result
        return {"status": "completed", "solver_calls": 6 if len(observed) == 1 else 1,
                "selection_log_rmse": .1, "calibration_log_rmse": .1}

    result = run_method_search(method="M5", candidates=states(), evaluate=evaluate,
                               base_candidate_id="a_full", rule_candidate_id="b_full", max_solver_calls=20)
    assert observed[0]["remaining_solver_calls"] == 10
    assert observed[1]["remaining_solver_calls"] == 4
    assert any(config["phase"] == "parameter_subset" for config in observed) is resumes
    if resumes:
        assert observed[2]["remaining_solver_calls"] == 14
        assert result["locked_structure_id"] == "a"
        assert result["status"] == "completed"
    else:
        assert result["status"] == "budget_exhausted"
        assert len(observed) == 2


def test_joint_method_uses_dsel_and_identifiability_to_revisit_structure():
    losses = {"a_full": .10, "b_full": .5, "a_CL": .08, "b_CL": .001}
    suggestions = {"a_full": ["CL"], "b_full": ["CL"]}
    evaluate, _ = evaluator(losses, suggestions)
    result = run_method_search(method="M6", candidates=states(), evaluate=evaluate,
                               base_candidate_id="a_full", rule_candidate_id="b_full", max_candidates=4)
    assert [row["candidate_id"] for row in result["history"]] == ["a_full", "b_full", "a_CL", "b_CL"]
    assert result["selected_candidate"]["candidate_id"] == "b_CL"
    assert result["history"][2]["reason"] == "identifiability_subset_feedback"
    changed_evaluate, _ = evaluator({"a_full": .8, "b_full": .01}, suggestions)
    changed = run_method_search(method="M6", candidates=states(), evaluate=changed_evaluate,
                                base_candidate_id="a_full", rule_candidate_id="b_full", max_candidates=3)
    assert changed["history"][2]["candidate_id"] == "b_CL"


@pytest.mark.parametrize("method,flag", [("A1", "requirement_guidance"), ("A2", "evidence_guidance")])
def test_guidance_ablations_have_different_order_and_explicit_evaluator_flags(method, flag):
    pool = [c for c in states() if c["free_parameter_ids"] == ["CL", "Pint"]]
    # Competing, prespecified guidance: requirement favors b, evidence favors a.
    if method == "A2":
        for candidate in pool:
            candidate["requirement_score"] = 0
    evaluate, observed = evaluator()
    ablated = run_method_search(method=method, candidates=pool, evaluate=evaluate, max_candidates=1, seed=43)
    normal_evaluate, _ = evaluator()
    normal = run_method_search(method="M6", candidates=pool, evaluate=normal_evaluate, max_candidates=1, seed=43)
    assert all(config[flag] is False for _, config in observed)
    assert ablated["history"][0]["candidate_id"] != normal["history"][0]["candidate_id"]


def test_a3_does_not_use_diagnostic_subset_feedback():
    evaluate, observed = evaluator({"a_full": .1, "b_full": .2}, {"a_full": [], "b_full": []})
    result = run_method_search(method="A3", candidates=states(), evaluate=evaluate,
                               base_candidate_id="a_full", rule_candidate_id="b_full", max_candidates=3)
    assert all(not config["identifiability_prescreen"] and not config["identifiability_feedback"] for _, config in observed)
    assert all(row["reason"] != "identifiability_subset_feedback" for row in result["history"])


@pytest.mark.parametrize("method", ["M0", "M1", "M2", "M3", "M4", "E0"])
def test_baselines_retain_registered_parameterization_and_do_not_adapt_to_diagnostics(method):
    evaluate, observed = evaluator(suggestions={"a_full": [], "b_full": []})
    result = run_method_search(method=method, candidates=states(), evaluate=evaluate,
        base_candidate_id="a_full", rule_candidate_id="b_full",
        qwen_proposer=lambda candidates, seed: "a_full")
    assert result["status"] == "completed"
    assert observed
    assert all(not config["identifiability_prescreen"] and not config["identifiability_feedback"]
               and config["fixed_parameterization"] for _, config in observed)
    # Diagnostics are still returned and preserved, but cannot rewrite a baseline.
    assert all("identifiability" in row["result"] for row in result["evaluations"])


@pytest.mark.parametrize("method", ["M6", "A1", "A2"])
def test_joint_methods_enable_diagnostic_parameter_adaptation(method):
    evaluate, observed = evaluator()
    run_method_search(method=method, candidates=states(), evaluate=evaluate,
                      base_candidate_id="a_full", max_candidates=2)
    assert all(config["identifiability_prescreen"] and config["identifiability_feedback"]
               and not config["fixed_parameterization"] for _, config in observed)


@pytest.mark.parametrize("count,expected", [(0, 2), (1, 4), (2, 2)])
def test_exact_count_filters_candidates_before_any_evaluation(count, expected):
    evaluate, observed = evaluator()
    result = run_method_search(method="E0", candidates=states(), evaluate=evaluate,
                               exact_free_parameter_count=count)
    assert result["status"] == "completed"
    assert result["input_candidate_count"] == 8
    assert result["registered_candidate_count"] == expected
    assert result["evaluated_candidate_count"] == expected
    assert result["exact_free_parameter_count"] == count
    assert result["enumeration_complete"] is True
    assert all(len(candidate["free_parameter_ids"]) == count for candidate, _ in observed)


@pytest.mark.parametrize("anchor,reason", [("base_candidate_id", "BASE_CANDIDATE_OUTSIDE_LEGAL_SPACE"),
                                         ("rule_candidate_id", "RULE_CANDIDATE_OUTSIDE_LEGAL_SPACE")])
def test_filtered_anchor_blocks_instead_of_being_replaced(anchor, reason):
    evaluate, observed = evaluator()
    result = run_method_search(method="M6", candidates=states(), evaluate=evaluate,
        exact_free_parameter_count=1, **{anchor: "a_full"})
    assert result["status"] == "blocked"
    assert result["blocked_reason"] == reason
    assert not observed
    assert result["selected_candidate"] is None


def test_empty_count_space_is_explicit_and_does_not_call_qwen():
    def unavailable_proposer(candidates, *, seed):
        pytest.fail("Qwen must not be called for an empty candidate space")
    evaluate, observed = evaluator()
    result = run_method_search(method="M2", candidates=states(), evaluate=evaluate,
        exact_free_parameter_count=3, qwen_proposer=unavailable_proposer)
    assert result["blocked_reason"] == "EMPTY_EXACT_FREE_PARAMETER_SPACE"
    assert not observed


def test_qwen_cannot_select_a_state_removed_by_count_filter():
    received = []
    def propose(candidates, *, seed):
        received.extend(candidates)
        return "a_full"
    evaluate, observed = evaluator()
    result = run_method_search(method="M2", candidates=states(), evaluate=evaluate,
                               exact_free_parameter_count=1, qwen_proposer=propose)
    assert len(received) == 4
    assert all(len(c["free_parameter_ids"]) == 1 for c in received)
    assert result["blocked_reason"] == "QWEN_PROPOSAL_NOT_REGISTERED"
    assert not observed


def test_count_matched_m5_compares_same_registered_subset_before_locking_structure():
    evaluate, observed = evaluator({"a_CL": .2, "b_CL": .4, "a_Pint": .1, "b_Pint": .001})
    result = run_method_search(method="M5", candidates=states(), evaluate=evaluate,
        exact_free_parameter_count=1, base_candidate_id="a_CL", rule_candidate_id="b_CL")
    assert result["status"] == "completed"
    assert result["common_structure_parameter_ids"] == ["CL"]
    assert result["locked_structure_id"] == "a"
    assert [c["candidate_id"] for c, _ in observed] == ["a_CL", "b_CL", "a_Pint"]
    assert all(config["fixed_parameterization"] and not config["identifiability_prescreen"]
               and config["exact_free_parameter_count"] == 1 for _, config in observed)
    assert [config["identifiability_feedback"] for _, config in observed] == [False, False, True]
    assert result["selected_candidate"]["candidate_id"] == "a_Pint"


def test_count_matched_m5_requires_a_common_subset_without_fallback_to_different_subsets():
    evaluate, observed = evaluator()
    pool = [c for c in states() if c["candidate_id"] in {"a_CL", "b_Pint"}]
    result = run_method_search(method="M5", candidates=pool, evaluate=evaluate,
        exact_free_parameter_count=1, base_candidate_id="a_CL", rule_candidate_id="b_Pint")
    assert result["blocked_reason"] == "M5_COMMON_MATCHED_PARAMETERIZATION_MISSING"
    assert not observed


def test_count_matched_parameterization_cannot_silently_shrink_after_diagnosis():
    def reduce(candidate, *, method_config):
        assert method_config["identifiability_prescreen"] is False
        assert method_config["fixed_parameterization"] is True
        return {"status": "completed", "solver_calls": 1, "calibration_log_rmse": .2,
                "selection_log_rmse": .3, "effective_free_parameter_ids": []}
    with pytest.raises(ValueError, match="must remain fixed"):
        run_method_search(method="M6", candidates=states(), evaluate=reduce,
                          exact_free_parameter_count=1, base_candidate_id="a_CL")


@pytest.mark.parametrize("budget", [500, 2000, 5000])
def test_independent_budget_runs_preserve_limit_and_last_valid_prediction(budget):
    observed = []
    def bounded(candidate, *, method_config):
        allowance = method_config["remaining_solver_calls"]
        observed.append(allowance)
        if allowance < 300:
            return {"status": "budget_exhausted", "solver_calls": allowance}
        return {"status": "completed", "solver_calls": 300,
                "calibration_log_rmse": .2, "selection_log_rmse": .3}
    result = run_method_search(method="M6", candidates=states(), evaluate=bounded,
                               max_solver_calls=budget, base_candidate_id="a_full")
    assert observed[0] == budget
    assert result["solver_calls"] <= budget
    assert result["selected_candidate"] is not None
    assert result["status"] == ("budget_exhausted" if budget < 2400 else "completed")


def test_common_selector_tolerance_parameter_count_then_structure_edits():
    evaluate, _ = evaluator({"a_full": .10, "a_CL": .115, "a_none": .2,
                            "b_none": .3, "b_CL": .101, "b_Pint": .3, "b_full": .3, "a_Pint": .3})
    result = run_method_search(method="E0", candidates=states(), evaluate=evaluate)
    # a_CL is within .02 and ties b_CL in free count, but has fewer edits.
    assert result["selected_candidate"]["candidate_id"] == "a_CL"
    assert result["enumeration_complete"] is True
    assert result["evaluated_candidate_count"] == 8


def test_enumeration_cannot_claim_complete_when_candidate_budget_is_insufficient():
    result = run("E0", max_candidates=7)
    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "ENUMERATION_EXCEEDS_CANDIDATE_BUDGET"
    assert not result["enumeration_complete"]
    assert not result["evaluations"]


def test_budget_exhaustion_keeps_previous_valid_prediction_and_counts_failed_cost():
    calls = []

    def evaluate(candidate, *, method_config):
        calls.append(method_config["remaining_solver_calls"])
        if len(calls) == 2:
            return {"status": "budget_exhausted", "solver_calls": 3}
        return {"status": "completed", "solver_calls": 7, "selection_log_rmse": .3, "calibration_log_rmse": .2}

    result = run_method_search(method="M6", candidates=states(), evaluate=evaluate,
                               base_candidate_id="a_full", rule_candidate_id="b_full", max_solver_calls=10)
    assert calls == [10, 3]
    assert result["status"] == "budget_exhausted"
    assert result["solver_calls"] == 10
    assert result["selected_candidate"]["candidate_id"] == "a_full"
    assert len(result["evaluations"]) == 2


def test_budget_contract_violation_stops_without_hiding_overspend():
    evaluate, observed = evaluator(calls=6)
    result = run_method_search(method="M6", candidates=states(), evaluate=evaluate, max_solver_calls=5)
    assert len(observed) == 1
    assert result["status"] == "failed"
    assert result["stop_reason"] == "evaluator_budget_contract_violation"
    assert result["solver_calls"] == 6


def test_exception_without_increment_stops_and_marks_unknown_cost():
    def evaluate(candidate, *, method_config):
        raise BudgetExceeded("search", requested=1, remaining=0)

    result = run_method_search(method="M6", candidates=states(), evaluate=evaluate)
    assert result["stop_reason"] == "external_budget_exhausted"
    assert not result["solver_cost_complete"]
    assert len(result["history"]) == 1


def test_failed_evaluations_never_win_even_with_low_numbers():
    def evaluate(candidate, *, method_config):
        return {"status": "failed", "solver_calls": 1, "selection_log_rmse": 0.0, "calibration_log_rmse": 0.0}

    result = run_method_search(method="E0", candidates=states(), evaluate=evaluate)
    assert result["selected_candidate"] is None
    assert result["status"] == "failed"
    assert result["solver_calls"] == 8


def test_inputs_and_results_cannot_share_mutable_state_or_external_scores():
    pool = states()
    original = deepcopy(pool)
    evaluate, _ = evaluator()
    run_method_search(method="M6", candidates=pool, evaluate=evaluate)
    assert pool == original
    pool[0]["selection_log_rmse"] = .01
    with pytest.raises(ValueError, match="previously evaluated"):
        run_method_search(method="M6", candidates=pool, evaluate=evaluate)


def test_runs_have_independent_scores_and_no_implicit_cross_method_cache():
    evaluate, observed = evaluator({"a_full": .1, "b_full": .2})
    run_method_search(method="M6", candidates=states(), evaluate=evaluate, max_candidates=2)
    first = len(observed)
    run_method_search(method="M3", candidates=states(), evaluate=evaluate, max_candidates=2)
    assert len(observed) == first + 2


@pytest.mark.parametrize("field,value", [("max_candidates", 25), ("max_candidates", True),
                                       ("max_solver_calls", 0), ("max_solver_calls", 5001),
                                       ("phase1_fraction", 1.0), ("phase1_fraction", float("nan")),
                                       ("seed", -1), ("selection_tolerance", float("inf")),
                                       ("exact_free_parameter_count", True), ("exact_free_parameter_count", -1),
                                       ("exact_free_parameter_count", 7), ("exact_free_parameter_count", 1.5)])
def test_bad_settings_rejected_before_evaluation(field, value):
    evaluate, observed = evaluator()
    with pytest.raises(ValueError):
        run_method_search(method="M6", candidates=states(), evaluate=evaluate, **{field: value})
    assert not observed


@pytest.mark.parametrize("field,value", [("features", [float("nan")]), ("requirement_score", float("inf")),
                                       ("free_parameter_ids", ["missing"]), ("parameter_ids", ["CL", "CL"]),
                                       ("structure_edit_count", -1)])
def test_bad_candidate_rejected_before_evaluation(field, value):
    pool = states()
    pool[0][field] = value
    evaluate, observed = evaluator()
    with pytest.raises(ValueError):
        run_method_search(method="M6", candidates=pool, evaluate=evaluate)
    assert not observed


def test_nonfinite_successful_metric_is_not_silently_ranked():
    def evaluate(candidate, *, method_config):
        return {"status": "completed", "solver_calls": 1, "selection_log_rmse": float("nan"), "calibration_log_rmse": .1}

    with pytest.raises(ValueError, match="finite"):
        run_method_search(method="M0", candidates=states(), evaluate=evaluate, base_candidate_id="a_full")


@pytest.mark.parametrize("method", ["M0", "M1", "M3", "M4", "M5", "M6", "A1", "A2", "A3", "E0"])
def test_all_method_outputs_are_finite_serializable_and_do_not_claim_validation(method):
    result = run(method)
    json.dumps(result, allow_nan=False)
    assert result["scientific_validation_performed"] is False
