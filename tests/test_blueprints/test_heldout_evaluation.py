"""Toy locked curves only; no scientific data, solver, service or GPU."""
from copy import deepcopy
import math

import pytest

from pbpk_kb.blueprints.heldout_evaluation import evaluate_locked_predictions, lock_predictions


def scenario(name="s1", *, drug="toy", study="study1", tier="development", n=2):
    return {"scenario_id": name, "compound_id": drug, "study_group_id": study,
            "curve_id": name, "tier": tier, "role": "calibration",
            "observations": [{"time_min": float(i + 1), "concentration_umol_per_l": 1.0} for i in range(n)]}


def curve(value=1.0, *, start=1.0, end=2.0):
    return {"time_min": [start, end], "concentration": [value, value], "time_unit": "min", "unit": "umol/l"}


def selected(curves, fallbacks=None, **extra):
    return {"method": "toy_method", "seed": 17, "candidate_id": "frozen_candidate", "status": "completed",
            "predictions_sha256": lock_predictions(curves, fallbacks), **extra}


def main_scenario(name="s1", study="external"):
    result = scenario(name, study=study, tier="main_holdout", n=5)
    result.update(role="test", development_study_group_ids=["fit", "selection"],
                  development_cohort_ids=["fit_cohort", "selection_cohort"],
                  study_reference={"study_id": study, "role": "test", "source_id": "original_source",
                                   "source_checksum": "a" * 64, "cohort_ids": ["external_cohort"]})
    return result


def test_primary_is_mean_of_curve_rmse_not_root_of_mean_mse():
    scenarios = [scenario("a"), scenario("b")]
    curves = {"a": curve(1), "b": curve(math.exp(2))}
    result = evaluate_locked_predictions(scenarios, curves, selected(curves))
    summary = result["tiers"]["development"]
    assert summary["mean_curve_log_rmse"] == pytest.approx(1.0)
    assert summary["mean_curve_log_rmse"] != pytest.approx(math.sqrt(2))
    assert summary["gmfe"] == pytest.approx(math.e)


def test_studies_then_drugs_are_equally_weighted_regardless_curve_counts():
    scenarios = [scenario("a", study="shared"), scenario("b", study="shared"),
                 scenario("c", study="second"), scenario("d", drug="other", study="third")]
    curves = {"a": curve(), "b": curve(), "c": curve(math.exp(4)), "d": curve(math.exp(6))}
    result = evaluate_locked_predictions(scenarios, curves, selected(curves))["tiers"]["development"]
    assert result["mean_curve_log_rmse"] == pytest.approx(4)  # mean(mean(0,4),6)
    assert result["compound_count"] == 2
    assert result["unique_global_study_groups"] == 3


def test_shared_study_across_drugs_is_counted_as_one_global_study():
    scenarios = [scenario("a", drug="first"), scenario("b", drug="second")]
    curves = {"a": curve(), "b": curve()}
    result = evaluate_locked_predictions(scenarios, curves, selected(curves))["tiers"]["development"]
    assert result["unique_global_study_groups"] == 1
    assert result["drug_study_memberships"] == 2


def test_duplicate_curve_alias_is_not_another_scientific_sample():
    first, alias = scenario("a"), scenario("alias")
    alias["curve_id"] = "a"
    curves = {"a": curve(math.e), "alias": curve(math.e)}
    result = evaluate_locked_predictions([first, alias], curves, selected(curves))
    summary = result["tiers"]["development"]
    assert summary["scenario_count"] == 2
    assert summary["duplicate_scenario_count"] == 1
    assert summary["unique_curve_count"] == 1
    assert result["rows"][1]["duplicate_of"] == "a"


@pytest.mark.parametrize("change", ["prediction", "observation"])
def test_conflicting_repeated_curve_is_rejected_instead_of_picking_best(change):
    first, alias = scenario("a"), scenario("alias")
    alias["curve_id"] = "a"
    curves = {"a": curve(), "alias": curve()}
    if change == "prediction":
        curves["alias"] = curve(math.e)
    else:
        alias["observations"][0]["concentration_umol_per_l"] = 2.0
    with pytest.raises(ValueError, match="Conflicting observations"):
        evaluate_locked_predictions([first, alias], curves, selected(curves))


def test_same_scenario_cannot_be_submitted_twice():
    curves = {"s1": curve()}
    with pytest.raises(ValueError, match="Duplicate scenario_id"):
        evaluate_locked_predictions([scenario(), scenario()], curves, selected(curves))


@pytest.mark.parametrize("tamper", ["primary", "fallback"])
def test_prediction_lock_includes_primary_and_fallback(tamper):
    curves, fallback = {"s1": curve()}, {"s1": curve()}
    locked = selected(curves, fallback)
    target = curves if tamper == "primary" else fallback
    target["s1"]["concentration"][0] = 2.0
    with pytest.raises(ValueError, match="changed after"):
        evaluate_locked_predictions([scenario()], curves, locked, fallback_curves=fallback)


def test_evaluator_does_not_claim_that_hash_proves_pretest_chronology():
    curves = {"s1": curve()}
    result = evaluate_locked_predictions([scenario()], curves, selected(curves))
    assert result["prediction_integrity_verified"] is True
    assert result["pretest_lock_timing_verified"] is False
    assert result["candidate_selection_performed"] is False


def test_candidate_lists_cannot_enter_final_evaluation():
    curves = {"s1": curve()}
    with pytest.raises(ValueError, match="cannot select"):
        evaluate_locked_predictions([scenario()], curves, selected(curves, candidates=["one", "two"]))


def test_bad_final_error_never_triggers_fallback():
    curves, fallback = {"s1": curve(math.exp(8))}, {"s1": curve()}
    result = evaluate_locked_predictions([scenario()], curves, selected(curves, fallback), fallback_curves=fallback)
    row = result["rows"][0]
    assert row["log_rmse"] == pytest.approx(8)
    assert row["fallback_used"] is False


@pytest.mark.parametrize("bad", [None, curve(0), curve(-1), curve(end=1.5), {"unit": "wrong"}])
def test_only_engineering_failures_trigger_prelocked_fallback(bad):
    curves, fallback = {"s1": bad}, {"s1": curve(math.e)}
    result = evaluate_locked_predictions([scenario()], curves, selected(curves, fallback), fallback_curves=fallback)
    row = result["rows"][0]
    assert row["fallback_used"] is True
    assert row["log_rmse"] == pytest.approx(1)
    assert row["native_prediction_error"] is not None
    assert result["tiers"]["development"]["native_success_count"] == 0


def test_budget_exhausted_with_valid_prediction_keeps_it():
    curves, fallback = {"s1": curve(math.e)}, {"s1": curve()}
    result = evaluate_locked_predictions([scenario()], curves, selected(curves, fallback, status="budget_exhausted"),
                                         fallback_curves=fallback)
    assert result["rows"][0]["log_rmse"] == pytest.approx(1)
    assert result["rows"][0]["fallback_used"] is False


def test_explicit_engineering_failure_cannot_use_residual_primary_curve():
    curves, fallback = {"s1": curve()}, {"s1": curve(math.e)}
    result = evaluate_locked_predictions([scenario()], curves, selected(curves, fallback, engineering_valid=False),
                                         fallback_curves=fallback)
    assert result["rows"][0]["log_rmse"] == pytest.approx(1)
    assert result["rows"][0]["fallback_used"] is True


def test_missing_predictions_retain_denominator_and_do_not_become_zero_error():
    curves = {"a": curve(math.e)}
    result = evaluate_locked_predictions([scenario("a"), scenario("missing")], curves, selected(curves))["tiers"]["development"]
    assert result["unique_curve_count"] == 2
    assert result["scored_curve_count"] == 1
    assert result["unscored_curve_count"] == 1
    assert result["mean_curve_log_rmse"] is None
    assert result["analysis_complete"] is False
    assert result["available_curve_log_rmse"] == pytest.approx(1)


@pytest.mark.parametrize("value", [0.0, -1.0, None])
def test_unknown_or_unquantifiable_observations_are_not_replaced_by_epsilon(value):
    data = scenario()
    data["observations"][0]["concentration_umol_per_l"] = value
    curves = {"s1": curve()}
    result = evaluate_locked_predictions([data], curves, selected(curves))
    assert result["rows"][0]["status"] == "observations_unscorable"
    assert result["tiers"]["development"]["mean_curve_log_rmse"] is None


def test_endpoints_default_to_unknown_instead_of_assuming_adequate_sampling():
    curves = {"s1": curve()}
    result = evaluate_locked_predictions([scenario()], curves, selected(curves))
    assert result["rows"][0]["endpoints"] == {"observed_window_auc": None, "sampled_cmax": None}


def test_auc_uses_actual_observation_window_and_cmax_only_sampled_points():
    data = scenario()
    data.update(auc_evaluable=True, cmax_evaluable=True,
                observations=[{"time_min": 2.0, "concentration_umol_per_l": 2.0},
                              {"time_min": 4.0, "concentration_umol_per_l": 4.0}])
    curves = {"s1": {"time_min": [0., 2., 3., 4., 8.], "concentration": [10., 4., 100., 8., 1.],
                     "time_unit": "min", "unit": "umol/l"}}
    result = evaluate_locked_predictions([data], curves, selected(curves))
    endpoints = result["rows"][0]["endpoints"]
    assert endpoints["observed_window_auc"]["observed"] == 6
    assert endpoints["observed_window_auc"]["predicted"] == 12
    assert endpoints["observed_window_auc"]["start_min"] == 2
    assert endpoints["sampled_cmax"]["predicted"] == 8  # Never claim the continuous peak of 100.
    assert endpoints["sampled_cmax"]["continuous_peak_estimated"] is False


def test_endpoint_failure_denominator_is_preserved():
    data = scenario()
    data.update(auc_evaluable=True, cmax_evaluable=True)
    curves = {"s1": None}
    result = evaluate_locked_predictions([data], curves, selected(curves))["tiers"]["development"]["endpoints"]
    assert result["observed_window_auc"]["requested_curve_count"] == 1
    assert result["observed_window_auc"]["evaluable_curve_count"] == 0
    assert result["observed_window_auc"]["mean_absolute_log_error"] is None


def test_all_tiers_are_separate_and_no_combined_best_score_is_returned():
    data = [scenario("dev"), scenario("reconstruct", tier="reconstruction"),
            scenario("external", tier="bottom_up_external"), main_scenario("main")]
    data[2]["role"] = "external"
    data[3]["study_group_id"] = "heldout_study"
    data[3]["study_reference"]["study_id"] = "heldout_study"
    curves = {s["scenario_id"]: curve(math.exp(index), end=5.) for index, s in enumerate(data)}
    result = evaluate_locked_predictions(data, curves, selected(curves))
    assert set(result["tiers"]) == {"development", "reconstruction", "bottom_up_external", "main_holdout"}
    assert "mean_curve_log_rmse" not in result
    assert result["tiers"]["main_holdout"]["mean_curve_log_rmse"] == pytest.approx(3)


@pytest.mark.parametrize("tamper", ["role", "reference_role", "reference_id", "checksum", "study_overlap", "cohort_overlap", "missing_reference", "missing_development"])
def test_main_holdout_rejects_inconsistent_roles_and_provenance(tamper):
    data = main_scenario()
    if tamper == "role":
        data["role"] = "calibration"
    elif tamper == "reference_role":
        data["study_reference"]["role"] = "selection"
    elif tamper == "reference_id":
        data["study_reference"]["study_id"] = "wrong"
    elif tamper == "checksum":
        data["study_reference"]["source_checksum"] = "unverified"
    elif tamper == "study_overlap":
        data["development_study_group_ids"].append("external")
    elif tamper == "cohort_overlap":
        data["development_cohort_ids"].append("external_cohort")
    elif tamper == "missing_reference":
        del data["study_reference"]
    else:
        del data["development_study_group_ids"]
    curves = {"s1": curve(end=5.)}
    with pytest.raises(ValueError):
        evaluate_locked_predictions([data], curves, selected(curves))


def test_main_holdout_can_use_existing_source_audit_study_reference():
    data = main_scenario()
    data["source_audit"] = {"study_reference": data.pop("study_reference")}
    curves = {"s1": curve(end=5.)}
    result = evaluate_locked_predictions([data], curves, selected(curves))
    assert result["tiers"]["main_holdout"]["scored_curve_count"] == 1


@pytest.mark.parametrize("place", ["point", "provenance", "development_roles"])
def test_main_tier_cannot_hide_development_roles_in_point_provenance(place):
    data = main_scenario()
    if place == "point":
        data["observations"][0]["split"] = "calibration"
    elif place == "provenance":
        data["observations"][0]["provenance"] = [{"study_role": "selection"}]
    else:
        data["development_study_group_ids"] = ["single_study"]
    curves = {"s1": curve(end=5.)}
    with pytest.raises(ValueError):
        evaluate_locked_predictions([data], curves, selected(curves))


def test_single_study_development_cannot_be_relabelled_external():
    data = scenario(tier="bottom_up_external")
    curves = {"s1": curve()}
    with pytest.raises(ValueError, match="development study"):
        evaluate_locked_predictions([data], curves, selected(curves))


def test_main_study_cannot_also_appear_in_reconstruction_tier():
    final, historic = main_scenario(), scenario("old", study="external", tier="reconstruction")
    curves = {"s1": curve(end=5.), "old": curve()}
    with pytest.raises(ValueError, match="also appears"):
        evaluate_locked_predictions([final, historic], curves, selected(curves))


def test_scoring_cannot_mutate_locked_input():
    data, curves = [scenario()], {"s1": curve()}
    before_data, before_curves = deepcopy(data), deepcopy(curves)
    evaluate_locked_predictions(data, curves, selected(curves))
    assert data == before_data
    assert curves == before_curves
