"""Independent statistical/isolation checks, not training-label assertions."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import predictive_risk as risk


def row(case, cluster, *, sampling="full", endpoint="terminal_half_life", error=1.5, features=None):
    return {"case_id": case, "cluster_id": cluster, "sampling": sampling,
            "endpoint": endpoint, "matched_fold_error": error, "dense_warning": False,
            "features": features or dict(zip(risk.FEATURES, (.2, .5, .25)))}


@pytest.fixture
def records():
    return [row(case, cluster, sampling=sampling, endpoint=endpoint, error=error,
                features=dict(zip(risk.FEATURES, (rmse, gap, .1))))
            for case, cluster, error, rmse, gap in [
                ("a", "one", 3., .8, 1.), ("b", "two", 1.2, .1, .1),
                ("c", "shared", 2.5, .3, .5), ("d", "shared", 1.5, .2, .4)]
            for sampling in ("full", "five") for endpoint in ("terminal_half_life", "clearance")]


def test_study_weights_share_total_equally_across_drugs_sampling_and_endpoints(records):
    weights = risk.cluster_weights(records)
    for cluster in ("one", "two", "shared"):
        assert sum(w for w, r in zip(weights, records) if r["cluster_id"] == cluster) == pytest.approx(1.)
    assert sum(w for w, r in zip(weights, records) if r["case_id"] == "c") == pytest.approx(.5)
    assert sum(w for w, r in zip(weights, records) if r["case_id"] == "a") == pytest.approx(1.)


def test_heldout_labels_cannot_change_their_own_risk_predictions(records):
    changed = deepcopy(records)
    for r in changed:
        if r["cluster_id"] == "shared":
            r["matched_fold_error"] = 5. if r["matched_fold_error"] <= 2 else 1.1
    first, second = risk.evaluate_records(records), risk.evaluate_records(changed)
    for a, b in zip(first["records"], second["records"]):
        if a["cluster_id"] == "shared":
            assert a["held_out_predictions"] == b["held_out_predictions"]
    fold = next(f for f in first["folds"] if f["held_out_cluster"] == "shared")
    assert fold["training_clusters"] == ["one", "two"]
    assert fold["held_out_records"] == 8


def test_scaling_and_missing_value_imputation_fit_training_fold_only(records):
    first = risk.evaluate_records(records)
    changed = deepcopy(records)
    for r in changed:
        if r["cluster_id"] == "shared":
            r["features"][risk.FEATURES[0]] = 1e8
            r["features"][risk.FEATURES[1]] = None
    second = risk.evaluate_records(changed)
    a = next(f for f in first["folds"] if f["held_out_cluster"] == "shared")
    b = next(f for f in second["folds"] if f["held_out_cluster"] == "shared")
    assert a["models"] == b["models"]
    assert a["models"]["risk_model"]["training_weighted_means"][0] == pytest.approx(.45)


@pytest.mark.parametrize("point_count", [2, 3, 4])
def test_single_class_training_fallback_is_smoothed_and_not_fitted_to_test_labels(point_count):
    data = [row(str(i), str(i), sampling=f"{point_count}_points", error=1.5) for i in range(3)]
    result = risk.evaluate_records(data)
    assert all(f["fallback"] == "too_few_clusters_or_single_class" for f in result["folds"])
    for r in result["records"]:
        assert r["held_out_predictions"]["risk_model"] == pytest.approx(.5 / 3.)
    assert result["metrics"]["risk_model"]["auroc"] is None


def test_two_cluster_data_uses_declared_low_sample_fallback():
    result = risk.evaluate_records([row("a", "a", error=3), row("b", "b", error=1)])
    assert all(f["fallback"] for f in result["folds"])
    probabilities = [r["held_out_predictions"]["risk_model"] for r in result["records"]]
    assert probabilities == [.25, .75]


@pytest.mark.parametrize("error", [None, float("nan"), float("inf"), .5, True])
def test_missing_or_invalid_reference_error_is_not_a_negative_label(error):
    with pytest.raises(ValueError):
        risk.evaluate_records([row("a", "a", error=error), row("b", "b")])


def test_duplicate_aliases_and_split_drug_identity_are_rejected(records):
    with pytest.raises(ValueError, match="Duplicate"):
        risk.evaluate_records(records + [deepcopy(records[0])])
    broken = deepcopy(records)
    broken[0]["cluster_id"] = "new"
    with pytest.raises(ValueError, match="one drug case"):
        risk.evaluate_records(broken)


def test_fixed_bins_thresholds_and_abstention_denominators_are_explicit():
    data = [row(str(i), str(i), error=error) for i, error in enumerate((3., 1., 4., 2.))]
    out = risk._metrics(data, [.8, .4, .2, .1])
    assert out["brier"] == pytest.approx((.04 + .16 + .64 + .01) / 4)
    main = next(r for r in out["thresholds"] if r["primary"])
    assert main["threshold"] == .35
    assert [main[k] for k in ("tp", "fp", "fn", "tn")] == [1, 1, 1, 1]
    assert main["coverage"] == main["accepted_large_error_rate"] == .5
    always = risk._metrics(data, np.ones(4))
    assert always["thresholds"][0]["coverage"] == 0
    assert always["thresholds"][0]["accepted_large_error_rate"] is None
    assert always["reliability_bins"][-1]["records"] == 4
    assert risk.CONFIG["diagnostic_thresholds"] == [.35, .5, .8]


def test_order_and_unrelated_provenance_are_not_risk_features(records):
    first = risk.evaluate_records(records)
    altered = deepcopy(records)
    for r in altered:
        r["source_path"] = "fabricated-file-name"
    second = risk.evaluate_records(altered)
    assert [r["held_out_predictions"] for r in first["records"]] == [r["held_out_predictions"] for r in second["records"]]


@pytest.mark.parametrize("n", [2, 3, 4])
def test_development_support_respects_actual_sparse_points_without_fabrication(n):
    training = {"observations": [{"time_min": i * 60, "concentration_umol_per_l": 1.} for i in range(1, n + 1)]}
    prediction = {"time_min": [0, 60, 120, 180, 240, 300], "concentration": [1.] * 6}
    endpoints = {"endpoints": {"terminal_half_life": {"value_h": 2.}}}
    mapping = {"decision": {"parameters": {"clearance_scale": 1.}, "data_only_ln_rmse": .2}}
    request = {"dosing": {"infusion_minutes": 0}}
    features, audit = risk.development_features(training=training, prediction=prediction, endpoints=endpoints, mapping=mapping, request=request)
    assert audit["development_points"] == n
    assert features["auc_inf"][risk.FEATURES[1]] == pytest.approx((5 - n) / 5)
    assert features["mrt"][risk.FEATURES[1]] == pytest.approx((25 - n * n) / 25)
    assert features["auc_inf"][risk.FEATURES[2]] == pytest.approx(.5)
    if n == 2:
        assert features["terminal_half_life"][risk.FEATURES[1]] is None
    else:
        assert features["terminal_half_life"][risk.FEATURES[1]] == pytest.approx(np.log(2.))


def test_original_baseline_collection_uses_twenty_predictions_and_keeps_missing_references_separate():
    root = Path(__file__).resolve().parents[2]
    collected = risk.collect_current_records(root)
    assert len(collected["records"]) == 106
    assert len(collected["excluded"]) == 14
    assert len({r["cluster_id"] for r in collected["records"]}) == 9
    assert len({(r["case_id"], r["sampling"]) for r in collected["records"]}) == 20
    assert all(set(r["features"]) == set(risk.FEATURES) for r in collected["records"])


@pytest.fixture
def nested_records():
    return [row(str(i), "shared" if i in (2, 6) else str(i), sampling=sampling,
                endpoint=endpoint, error=3. if i in (2, 4, 5, 6) else 1.,
                features=dict(zip(risk.FEATURES, (.05 + .12 * i + .01 * (sampling == "five"), .5, .2))))
            for i in range(7) for sampling in ("full", "five")
            for endpoint in ("clearance", "terminal_half_life")]


def test_nested_target_labels_cannot_affect_own_threshold_or_predictions(nested_records):
    original = risk.evaluate_nested_rmse_policy(nested_records)
    changed = deepcopy(nested_records)
    for r in changed:
        if r["cluster_id"] == "shared":
            r["matched_fold_error"] = 1.1
    altered = risk.evaluate_nested_rmse_policy(changed)
    a = next(f for f in original["folds"] if f["held_out_cluster"] == "shared")
    b = next(f for f in altered["folds"] if f["held_out_cluster"] == "shared")
    for key in ("threshold", "fallback", "outer_fit", "inner_folds", "inner_oof", "inner_selection_metrics"):
        assert a[key] == b[key]
    assert [r for r in original["records"] if r["cluster_id"] == "shared"] == [
        r for r in altered["records"] if r["cluster_id"] == "shared"]


def test_nested_full_five_all_drugs_and_endpoints_share_both_fold_boundaries(nested_records):
    result = risk.evaluate_nested_rmse_policy(nested_records)
    lookup = {tuple(risk._record_id(r)): r["cluster_id"] for r in nested_records}
    for outer in result["folds"]:
        held = outer["held_out_cluster"]
        train_ids = {tuple(x) for x in outer["training_ids"]}
        valid_ids = {tuple(x) for x in outer["held_out_ids"]}
        assert train_ids.isdisjoint(valid_ids)
        assert {lookup[x] for x in valid_ids} == {held}
        assert held not in {lookup[x] for x in train_ids}
        seen = []
        for inner in outer["inner_folds"]:
            iti = {tuple(x) for x in inner["training_ids"]}
            ivi = {tuple(x) for x in inner["held_out_ids"]}
            assert iti.isdisjoint(ivi)
            assert iti | ivi == train_ids
            assert set(inner["training_clusters"]).isdisjoint(inner["held_out_clusters"])
            assert held not in inner["training_clusters"] + inner["held_out_clusters"]
            seen.extend(ivi)
        assert len(seen) == len(set(seen)) == outer["inner_oof_records"]
        assert set(seen) == train_ids


def test_nested_extreme_outer_features_do_not_change_threshold_or_inner_preprocessing(nested_records):
    original = risk.evaluate_nested_rmse_policy(nested_records)
    changed = deepcopy(nested_records)
    for r in changed:
        if r["cluster_id"] == "shared":
            r["features"][risk.FEATURES[0]] = 1e6
    altered = risk.evaluate_nested_rmse_policy(changed)
    a = next(f for f in original["folds"] if f["held_out_cluster"] == "shared")
    b = next(f for f in altered["folds"] if f["held_out_cluster"] == "shared")
    for key in ("threshold", "fallback", "outer_fit", "inner_folds", "inner_oof"):
        assert a[key] == b[key]


def test_nested_threshold_is_highest_with_cluster_weighted_recall_not_record_recall():
    records = [row("many", "one", sampling=str(i), error=3.) for i in range(8)]
    records += [row("one", "two", error=3.), row("good", "three", error=1.)]
    probabilities = [.9] * 8 + [.2, .8]
    threshold, fallback = risk._select_nested_threshold(records, probabilities)
    # Row recall .8 at .9 would pass incorrectly; cluster recall is only .5.
    assert threshold == .2
    assert fallback is None
    assert risk.NESTED_CONFIG["inner_recall_target"] == .8


def test_nested_constant_or_single_class_fallback_warns_everyone_and_has_no_accepted_risk():
    for errors in ((1., 1., 1., 1.), (1., 3., 1., 3.)):
        records = [row(str(i), str(i), error=error) for i, error in enumerate(errors)]
        result = risk.evaluate_nested_rmse_policy(records)
        assert result["fallback_folds"] == 4
        assert all(r["warning"] and r["threshold"] is None for r in result["records"])
        assert result["metrics"]["coverage"] == 0
        assert result["metrics"]["accepted_large_error_rate"] is None
        assert all(f["held_out_metrics"]["coverage"] == 0 for f in result["folds"])


def test_nested_two_cluster_empty_inner_fit_is_explicit_and_conservative():
    result = risk.evaluate_nested_rmse_policy([row("a", "a", error=3.), row("b", "b", error=1.)])
    assert result["metrics"]["coverage"] == 0
    assert all(f["inner_folds"][0]["fallback"] == "empty_inner_training_set" for f in result["folds"])
