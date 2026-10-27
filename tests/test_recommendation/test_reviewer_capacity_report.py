"""Seed pooling must not invent sample size, label coverage or execution data."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import reviewer_capacity_report as report


def test_complete_report_json_accepts_seed_and_ensemble_keys_without_changing_values(tmp_path):
    metrics = {17: {"seed": 17, "ndcg@10": .9094818276371732},
               23: {"seed": 23, "ndcg@10": .91},
               41: {"seed": 41, "ndcg@10": .92},
               "rank_ensemble": {"seed": "rank_ensemble", "ndcg@10": .93}}
    summary = {"seeds": [17, 23, 41], "extended_endpoints": metrics,
               "extended_common_by_seed": metrics,
               "reports": {"pbpk_interaction": {17: {"quality": {"ndcg@10": .9094818276371732}}}}}
    original = deepcopy(summary)
    output = tmp_path / "summary.json"
    # Exercise the same sort_keys=True publication writer that failed only
    # when the actual complete report included the secondary ensemble.
    report.write(output, report.json_string_keys(summary))
    loaded = json.loads(output.read_text())
    assert loaded["seeds"] == [17, 23, 41]
    assert set(loaded["extended_endpoints"]) == {"17", "23", "41", "rank_ensemble"}
    for key, value in metrics.items():
        assert loaded["extended_endpoints"][str(key)] == value
        assert loaded["extended_common_by_seed"][str(key)] == value
    assert loaded["reports"]["pbpk_interaction"]["17"]["quality"]["ndcg@10"] == .9094818276371732
    assert summary == original
    with pytest.raises(ValueError, match="collision"):
        report.json_string_keys({"nested": {17: "one", "17": "another"}})


def quality(values):
    return {"quality": {"per_query": [
        {"query_id": key, "judged_only": None if value is None else
         {"ndcg@10": value, "recall@10": value, "mrr": value}}
        for key, value in values.items()]}}


def test_seed_average_keeps_one_request_and_unknown_is_not_zero():
    reports = {seed: quality({"q": value, "unknown": None})
               for seed, value in zip(report.SEEDS, (.2, .5, .8))}
    averaged = report.average_query_quality(reports)
    assert len(averaged["per_query"]) == 2
    by_id = {row["query_id"]: row for row in averaged["per_query"]}
    assert by_id["q"]["judged_only"]["ndcg@10"] == pytest.approx(.5)
    assert by_id["unknown"]["judged_only"] is None
    reports[41]["quality"]["per_query"][1]["judged_only"] = dict.fromkeys(report.METRICS, 0.)
    with pytest.raises(ValueError, match="eligibility"):
        report.average_query_quality(reports)


@pytest.mark.parametrize("fault", ("missing_seed", "duplicate_query", "changed_query"))
def test_seed_average_rejects_incomplete_axes(fault):
    reports = {seed: quality({"q": .5}) for seed in report.SEEDS}
    if fault == "missing_seed":
        reports.pop(41)
    elif fault == "duplicate_query":
        reports[41]["quality"]["per_query"] *= 2
    else:
        reports[41]["quality"]["per_query"][0]["query_id"] = "wrong"
    with pytest.raises(ValueError):
        report.average_query_quality(reports)


def test_primary_groups_not_seeds_are_sampling_units_and_no_secondary_p_values():
    groups = {"a": "drug1", "b": "drug1", "c": "drug2", "d": "drug3", "u": "unknown_group"}
    reports = {method: {} for method in report.METHODS}
    for seed in report.SEEDS:
        reports["concat_capacity"][seed] = quality({"a": .5, "b": .5, "c": .5, "d": .5, "u": None})
        reports["pbpk_interaction"][seed] = quality({"a": .8, "b": .6, "c": .4, "d": .5, "u": None})
    result = report.paired_seed_average_quality(reports, groups)
    primary = next(row for row in result["comparisons"] if row["metric"] == "ndcg@10")
    assert primary["eligible_queries"] == 4
    assert primary["total_queries"] == 5
    assert primary["independent_groups"] == 3
    assert primary["mean_group_delta"] == pytest.approx((.2 - .1 + 0) / 3)
    assert primary["query_macro_delta"] == pytest.approx((.3 + .1 - .1) / 4)
    for row in result["comparisons"]:
        if row["metric"] != "ndcg@10":
            assert "holm_p" not in row and "cluster_sign_flip_p" not in row
    # Duplicating identical training seeds cannot shrink the sampling interval.
    one_seed = {method: {17: reports[method][17]} for method in report.METHODS}
    single = report.paired_seed_average_quality(one_seed, groups, seeds=(17,))
    primary_single = next(row for row in single["comparisons"] if row["metric"] == "ndcg@10")
    assert primary_single["bootstrap_95ci"] == pytest.approx(primary["bootstrap_95ci"], abs=1e-14)
    assert primary_single["cluster_sign_flip_p"] == primary["cluster_sign_flip_p"]


def test_input_validation_requires_complete_candidate_axes_and_no_pk_correction():
    data = {"query_records": [{}, {}], "candidate_records": [{}, {}, {}], "variant_names": ["original", "reordered"]}
    values = {"indices": np.arange(2), "scores": np.ones((2, 2, 3)), "base_scores": np.ones((2, 2, 3)),
              "probabilities": np.full((2, 2, 3, 3), 1 / 3)}
    report.validate_arrays(values, data)
    changed = deepcopy(values)
    changed["base_scores"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="error correction"):
        report.validate_arrays(changed, data)
    changed = deepcopy(values)
    changed["scores"] = changed["scores"][:, :, :2]
    with pytest.raises(ValueError, match="Every registered candidate"):
        report.validate_arrays(changed, data)
    changed = deepcopy(values)
    changed["probabilities"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="probabilities"):
        report.validate_arrays(changed, data)


def execution_row(method, sid, error, coverage=1.):
    return {"method": method, "scenario_id": sid, "compound": sid, "publication_cluster_id": "study_" + sid,
            "endpoint": "Cmax", "stratum": "reference_diagnostic", "rank": 1,
            "coverage": coverage, "absolute_log_error": error,
            "top1_evaluable_probability": coverage, "top1_absolute_log_error": error,
            "signed_log_error": error, "within_twofold": 1., "within_1_5_fold": 1., "regret": 0.}


def test_execution_common_intersection_excludes_any_unmeasured_seed_and_retains_identity():
    reports = {method: {seed: {"execution": {"position_rows": [
        execution_row(method, "complete", value), execution_row(method, "missing", value)]}}
        for seed, value in zip(report.SEEDS, (.1, .2, .3))} for method in report.METHODS}
    reports["concat_capacity"][41]["execution"]["position_rows"][1].update(
        coverage=0., absolute_log_error=None, top1_evaluable_probability=0., top1_absolute_log_error=None)
    result = report.average_common_execution(reports)
    common = next(row for row in result["common_positions"] if row["rank"] == 1)
    assert common["shared_scenarios"] == 1
    assert common["scenario_ids"] == ["complete"]
    assert common["own_gmfe"] == pytest.approx(np.exp(.2))
    assert result["excluded_positions"][0]["scenario_id"] == "missing"
    rows = result["per_scenario_seed_means"]["pbpk_interaction"]["execution"]["position_rows"]
    assert len(rows) == 1 and rows[0]["absolute_log_error"] == pytest.approx(.2)
    assert all("two_sided_p" not in row for row in result["paired_descriptive"])
    reports["concat_capacity"][23]["execution"]["position_rows"][0]["compound"] = "other_drug"
    with pytest.raises(ValueError, match="identity"):
        report.average_common_execution(reports)


def test_full_list_retains_unknown_top1_and_classifies_purpose():
    data = {"core_indices": [0], "query_records": [{"query_id": "q", "compound_id": "drug",
            "task_type": "plasma_concentration_time_prediction"}],
            "grades": np.array([[-1, 3, 0]]), "applicability": np.array([[-1, 0, 1]])}
    value = report.full_list(data, np.array([[[3., 2., 1.]]]), "pbpk_interaction", 17)
    rows = {row["scope"]: row for row in value["summary"]}
    assert rows["all"]["top1_unknown"] == 1.
    assert rows["all"]["top1_supported"] == 0.
    assert rows["all"]["ndcg@10"] == 1.
    assert rows["direct_prediction"]["requests"] == 1
    assert rows["parameterization_template"]["requests"] == 0


def test_secondary_ensemble_uses_ranks_not_incompatible_score_scales():
    predictions = {17: (np.array([[[1000., 0., 0.]]]), np.full((1, 1, 3, 3), 1 / 3)),
                   23: (np.array([[[0., 1., .1]]]), np.full((1, 1, 3, 3), 1 / 3)),
                   41: (np.array([[[0., 1., .1]]]), np.full((1, 1, 3, 3), 1 / 3))}
    scores, probabilities = report.rank_ensemble(predictions)
    assert scores.argmax() == 1  # Raw-score averaging would incorrectly choose 0.
    np.testing.assert_allclose(probabilities.sum(-1), 1.)
    predictions[17] = (predictions[17][0] * 100, predictions[17][1])
    np.testing.assert_array_equal(report.rank_ensemble(predictions)[0], scores)


def test_extended_common_retains_tmax_hours_native_strata_and_zero_coverage():
    endpoints = {}
    definitions = [{"endpoint_key": "Tmax", "endpoint": "Tmax"}, {"endpoint_key": "Cmax", "endpoint": "Cmax"}]
    for seed, error in zip(report.SEEDS, (1., 2., 3.)):
        rows = []
        for method in report.METHODS:
            for endpoint in ("Tmax", "Cmax"):
                for kind in ("native", "matched"):
                    rows.append({"method_id": method, "scenario_id": "q", "compound": "drug", "publication_cluster_id": "study",
                                 "endpoint_key": endpoint, "rank": 1, "analysis_stratum": "main", "prediction_kind": kind,
                                 "coverage": 1. if kind == "native" else 0.,
                                 "absolute_error_h": error if endpoint == "Tmax" else None,
                                 "absolute_log_error": .1 * error if endpoint == "Cmax" else None})
        endpoints[seed] = {"endpoint_definitions": definitions, "position_rows": rows}
    rows = {(row["endpoint_key"], row["prediction_kind"]): row for row in report.extended_common(endpoints)}
    assert rows["Tmax", "native"]["metric_kind"] == "MAE_h"
    assert rows["Tmax", "native"]["pbpk_interaction"] == 2.
    assert rows["Cmax", "native"]["pbpk_interaction"] == pytest.approx(np.exp(.2))
    assert rows["Tmax", "matched"]["shared_scenarios"] == 0
    assert rows["Tmax", "matched"]["pbpk_interaction"] is None
