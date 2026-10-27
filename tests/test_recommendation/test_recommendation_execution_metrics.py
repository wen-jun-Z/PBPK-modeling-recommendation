"""Independent arithmetic, complete-pool and missing-outcome contracts."""
import itertools
import json
import math

import pytest

from scripts.recommendation_execution_metrics import (
    curve_metrics, endpoint_error, evaluate_selection, summarize_errors,
)


def outcome(cid, status="validated", predicted=2., observed=1., **extra):
    row = {"candidate_id": cid, "scientific_status": status,
           "execution_status": "succeeded" if status == "validated" else "not_run",
           "endpoints": {"auc": {"predicted": predicted, "observed": observed}} if status == "validated" else {},
           "reason": "specified scenario exclusion" if status in {"incompatible", "reference_only"} else None}
    row.update(extra)
    return row


def test_signed_bias_and_gmfe_do_not_cancel_opposite_errors():
    a = endpoint_error(4., 1., endpoint="auc")
    b = endpoint_error(.25, 1., endpoint="auc")
    report = summarize_errors([a, b, endpoint_error(None, 1, endpoint="auc")])
    assert a["ratio"] == 4
    assert b["fold_error"] == 4
    assert report["gmfe"] == 4
    assert report["afe"] == 1
    assert report["valid_count"] == 2 and report["unknown_count"] == 1


def test_different_endpoint_errors_are_not_combined_into_one_gmfe():
    with pytest.raises(ValueError, match="separately"):
        summarize_errors([endpoint_error(2,1,endpoint="auc"),endpoint_error(1,2,endpoint="cmax")])


@pytest.mark.parametrize("p,o,bql,reason", [(0, 1, False, "nonpositive_value"), (-1, 2, False, "nonpositive_value"),
    (1, 0, False, "nonpositive_value"), (None, 1, False, "missing_value"),
    (float("nan"), 1, False, "nonfinite_or_nonnumeric_value"), (1, float("inf"), False, "nonfinite_or_nonnumeric_value"),
    (False, 1, False, "nonfinite_or_nonnumeric_value"), (1, 1, True, "below_quantification")])
def test_invalid_endpoint_never_becomes_zero_error(p, o, bql, reason):
    r = endpoint_error(p, o, endpoint="auc", bql=bql)
    assert r["state"] == "unknown" and r["reason"] == reason
    assert r["absolute_log_error"] is None
    json.dumps(r, allow_nan=False)


def test_extreme_finite_ratios_keep_log_error_without_infinity():
    r = endpoint_error(1e300, 1e-300, endpoint="auc")
    assert r["state"] == "valid" and math.isfinite(r["absolute_log_error"])
    assert r["fold_error"] is None and r["ratio_or_fold_overflow"]
    json.dumps(r, allow_nan=False)


def test_curve_window_and_sampled_peak_are_not_full_auc_or_continuous_peak():
    r = curve_metrics([2, 3, 5], [1, 3, 1], [2, 3, 5], [2, 6, 2])
    assert r["state"] == "computed"
    assert r["window"]["start"] == 2
    assert r["observed_window_auc"]["observed"] == 6
    assert r["observed_window_auc"]["predicted"] == 12
    assert r["sampled_peak"]["observed"] == 3
    assert r["continuous_cmax"] is None
    assert r["curve"]["gmfe"] == pytest.approx(2)


@pytest.mark.parametrize("ot,ov,pt,pv,reason", [
    ([0,1],[1,2],[0,2],[1,2],"time_grid_mismatch_no_implicit_interpolation_or_extrapolation"),
    ([0,1],[1,2],[0,.5,1],[1,2,3],"length_mismatch_or_fewer_than_two_points"),
    ([1,1],[1,2],[1,1],[1,2],"times_not_strictly_increasing"),
    ([1,0],[1,2],[1,0],[1,2],"times_not_strictly_increasing"),
    ([-1,1],[1,2],[-1,1],[1,2],"invalid_time"),
    ([0,1],[0,2],[0,1],[1,2],"invalid_concentration_points"),
    ([0,1],[1,2],[0,1],[float('nan'),2],"invalid_concentration_points"),
])
def test_curve_rejects_unaligned_nonpositive_or_unusable_inputs(ot, ov, pt, pv, reason):
    r = curve_metrics(ot, ov, pt, pv)
    assert r["reason"] == reason and r["curve"] is None
    json.dumps(r, allow_nan=False)


def test_curve_bql_not_silently_dropped():
    r = curve_metrics([0,1],[1,2],[0,1],[1,2],below_quantification=[False, True])
    assert r["invalid_points"] == [{"index": 1, "reason": "below_quantification"}]
    with pytest.raises(ValueError):
        curve_metrics([0,1],[1,2],[0,1],[1,2],below_quantification=[False])


def test_uniform_tied_hit_matches_all_permutations_not_candidate_id():
    ids = ["z", "a", "m", "x"]
    rows = [outcome("z"), outcome("a"), outcome("m", "incompatible"), outcome("x", "incompatible")]
    r = evaluate_selection(ids, rows, dict.fromkeys(ids, 1), ks=(1,3,5))
    for k in (1,3,5):
        orders = list(itertools.permutations(ids))
        expected_hit = sum(any(c in {"z","a"} for c in order[:k]) for order in orders) / len(orders)
        expected_recall = sum(len(set(order[:k]) & {"z","a"}) / 2 for order in orders) / len(orders)
        assert r["by_k"][str(k)]["known_hit"] == pytest.approx(expected_hit)
        assert r["by_k"][str(k)]["exact_recall"] == pytest.approx(expected_recall)
    assert r["by_k"]["5"]["ranking_slots"] == 4


@pytest.mark.parametrize("scores", [[3,2,2,1], [1,1,1,1], [4,3,2,1]])
def test_recall_bounds_equal_brute_force_all_unknown_label_completions(scores):
    ids = list("abcd")
    rows = [outcome("a"), outcome("b", "unknown"), outcome("c", "unknown"), outcome("d", "incompatible")]
    ranking = dict(zip(ids, scores))
    partial = evaluate_selection(ids, rows, ranking, ks=(1,2,3))
    for k in (1,2,3):
        possible = []
        for assignment in itertools.product([False, True], repeat=2):
            full = [outcome("a"), outcome("d", "incompatible")]
            full += [outcome(cid, "validated" if positive else "incompatible") for cid, positive in zip("bc", assignment)]
            possible.append(evaluate_selection(ids, full, ranking, ks=(k,))["by_k"][str(k)]["exact_recall"])
        bounds = partial["by_k"][str(k)]["recall_bounds"]
        assert bounds["lower"] == pytest.approx(min(possible))
        assert bounds["upper"] == pytest.approx(max(possible))
        assert partial["by_k"][str(k)]["exact_recall"] is None


def test_missing_top_result_is_unknown_and_never_reranked_away():
    r = evaluate_selection(["missing","good"], [outcome("good")], {"missing":2,"good":1}, endpoint_names=["auc"])
    assert r["unknown_candidate_count"] == 1
    assert r["by_k"]["1"]["evaluable_coverage"] == 0
    assert r["by_k"]["1"]["known_hit"] == 0
    assert r["by_k"]["1"]["hit_bounds"] == {"lower":0,"upper":1}
    assert r["endpoints"]["auc"]["exact_full_pool_regret"] is None
    assert r["endpoints"]["auc"]["full_pool_regret_bounds"]["upper"] is None


def test_no_known_positive_recall_bounds_are_conditional_not_a_zero_score():
    r = evaluate_selection(list("abc"), [outcome("b","incompatible")], {"a":2,"b":1,"c":0})
    bounds = r["by_k"]["1"]["recall_bounds"]
    assert bounds["zero_positive_completion_possible"] is True
    assert bounds["lower"] == 0 and bounds["upper"] == 1
    assert r["by_k"]["1"]["exact_recall"] is None
    assert r["by_k"]["1"]["known_positive_recall"] is None


def test_all_failed_or_not_run_is_not_a_complete_negative_pool():
    rows = [outcome("a", "unknown", execution_status="failed"), outcome("b", "unknown")]
    r = evaluate_selection(list("ab"), rows, {"a":0,"b":0}, abstained=True, endpoint_names=["auc"])
    assert not r["scientific_pool_complete"]
    assert r["decision"]["no_applicable_established"] is False
    assert r["decision"]["exact_false_rejection"] is None
    assert r["decision"]["false_rejection_event_bounds"] == {"lower":0,"upper":1}
    assert r["by_k"]["1"]["known_miss"] is None
    assert r["by_k"]["1"]["evaluable_coverage"] == 0
    json.dumps(r, allow_nan=False)


def test_reference_only_is_not_a_direct_positive_and_adaptation_stays_separate():
    rows = [outcome("ref", "reference_only"), outcome("yes")]
    r = evaluate_selection(["ref","yes"], rows, {"ref":2,"yes":1})
    assert r["by_k"]["1"]["exact_recall"] == 0
    assert r["by_k"]["1"]["known_hit"] == 0
    with pytest.raises(ValueError, match="must not be mixed"):
        evaluate_selection(["yes"], [outcome("yes", track="reference_adaptation")], {"yes":1})
    r = evaluate_selection(["ref"], [outcome("ref","reference_only",track="reference_adaptation")], {"ref":1},track="reference_adaptation")
    assert not r["scientific_pool_complete"] and r["unknown_candidate_count"] == 1


def test_complete_rejection_and_known_positive_denominators():
    yes = evaluate_selection(["a"], [outcome("a")], {"a":1},abstained=True)
    no = evaluate_selection(["a"], [outcome("a","incompatible")], {"a":1},abstained=True)
    assert yes["decision"]["exact_false_rejection"] == 1
    assert yes["decision"]["exact_false_rejection_denominator"] == 1
    assert no["decision"]["exact_false_rejection"] is None
    assert no["decision"]["exact_correct_refusal"] == 1
    assert no["by_k"]["1"]["exact_recall"] is None
    assert yes["endpoints"]["auc"]["exact_full_pool_regret"] is None


def test_endpoint_regret_is_uniform_top1_error_minus_best_same_pool():
    r = evaluate_selection(list("abc"), [outcome("a",predicted=4),outcome("b",predicted=1),outcome("c",predicted=2)],
                           {"a":2,"b":2,"c":1})
    e = r["endpoints"]["auc"]
    assert e["exact_full_pool_regret"] == pytest.approx(math.log(2))
    assert e["selected_known_conditional_absolute_log_error"] == pytest.approx(math.log(2))
    assert e["best_observed_absolute_log_error"] == 0


def test_incomplete_oracle_only_gets_explicit_partial_regret_and_bounds():
    r = evaluate_selection(list("abc"), [outcome("a",predicted=4),outcome("b",predicted=2)], {"a":3,"b":2,"c":1})
    e = r["endpoints"]["auc"]
    assert e["known_pool_conditional_regret"] == pytest.approx(math.log(2))
    assert e["exact_full_pool_regret"] is None
    assert e["full_pool_regret_bounds"]["lower"] == pytest.approx(math.log(2))
    assert e["full_pool_regret_bounds"]["upper"] == pytest.approx(math.log(4))


def test_endpoint_missing_is_separate_from_other_endpoint_success():
    r = evaluate_selection(list("ab"), [outcome("a"),outcome("b")], {"a":2,"b":1},endpoint_names=["auc","cmax"])
    assert r["endpoints"]["auc"]["exact_full_pool_regret"] == 0
    assert r["endpoints"]["cmax"]["exact_full_pool_regret"] is None
    assert r["endpoints"]["cmax"]["valid_candidate_count"] == 0
    assert r["by_k"]["1"]["evaluable_coverage"] == 0
    assert r["by_k"]["1"]["evaluable_any_endpoint_coverage"] == 1


@pytest.mark.parametrize("ids,rows,ranking", [
    (["a","a"],[],{"a":1}), (["a"],[],{}), (["a"],[],{"a":float("nan")}),
    (["a"],[outcome("b")],{"a":1}), (["a"],[outcome("a"),outcome("a")],{"a":1}),
    (["a"],[outcome("a",execution_status="failed")],{"a":1}),
    (["a"],[outcome("a","incompatible",execution_status="failed")],{"a":1}),
    (["a"],[outcome("a","validated",endpoints={})],{"a":1}),
    (["a"],[outcome("a","incompatible",reason=None)],{"a":1}),
])
def test_bad_contract_rejected_instead_of_silently_changing_denominator(ids, rows, ranking):
    with pytest.raises(ValueError):
        evaluate_selection(ids, rows, ranking)


def test_candidate_rename_and_outcome_order_do_not_change_tied_metric():
    a = evaluate_selection(["x","y"],[outcome("x"),outcome("y","incompatible")],{"x":1,"y":1})
    b = evaluate_selection(["a","z"],[outcome("z","incompatible"),outcome("a")],{"z":1,"a":1})
    assert a["by_k"]["1"]["known_hit"] == b["by_k"]["1"]["known_hit"] == .5
    assert a["by_k"]["1"]["evaluable_coverage"] == b["by_k"]["1"]["evaluable_coverage"] == .5
