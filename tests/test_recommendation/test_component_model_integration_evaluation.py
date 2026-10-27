"""Paired input interventions must preserve identities, unknowns and estimands."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("component_integration_eval", ROOT / "scripts/evaluate_component_model_integration.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def matrices():
    data = {"core_indices": [0, 2], "execution_indices": [1],
            "query_records": [{"query_id": f"q{i}"} for i in range(3)],
            "candidate_records": [{"candidate_id": f"c{i}"} for i in range(3)]}
    arrays = {"indices": np.array([2, 0, 1]), "scores": np.arange(9).reshape(3, 3).astype(float),
              "base_scores": np.arange(9).reshape(3, 3).astype(float),
              "probabilities": np.full((3, 3, 3), 1 / 3)}
    manifest = {"query_ids": ["q2", "q0", "q1"], "candidate_ids": ["c0", "c1", "c2"]}
    return data, arrays, manifest


def test_permuted_rows_require_matching_global_identity():
    data, arrays, manifest = matrices()
    assert m.validate_arrays(arrays, manifest, data, expected_queries=3, expected_candidates=3) == {2: 0, 0: 1, 1: 2}
    manifest["query_ids"][0] = "q1"
    with pytest.raises(ValueError, match="query IDs"):
        m.validate_arrays(arrays, manifest, data, expected_queries=3, expected_candidates=3)


@pytest.mark.parametrize("defect", ["duplicate", "candidate", "nonfinite", "probability"])
def test_axis_and_probability_failures_cannot_be_filled(defect):
    data, arrays, manifest = matrices()
    if defect == "duplicate":
        arrays["indices"][0] = 0
    elif defect == "candidate":
        manifest["candidate_ids"].reverse()
    elif defect == "nonfinite":
        arrays["scores"][0, 0] = np.nan
    else:
        arrays["probabilities"][0, 0, 0] = 1.
    with pytest.raises(ValueError):
        m.validate_arrays(arrays, manifest, data, expected_queries=3, expected_candidates=3)


@pytest.mark.parametrize("flag", list(m.REQUIRED_FLAGS))
def test_changed_task_or_retrained_weights_reject_existing_label_comparison(flag):
    manifest = {"state": "complete", **m.REQUIRED_FLAGS}
    m.validate_semantics(manifest)
    manifest[flag] = not m.REQUIRED_FLAGS[flag]
    with pytest.raises(ValueError, match=flag):
        m.validate_semantics(manifest)


def test_top1_unavailable_stays_unavailable_despite_good_rank3():
    contract = {"q": {"compound": "drug", "publication_cluster_id": "study", "endpoints": ["Cmax"],
        "same_parent_ids": ["a", "b", "c"], "errors": {"c": {"Cmax": {"absolute": math.log(1.2), "signed": math.log(1.2)}}}}}
    result = m.ev.clinical_evaluation([[3., 2., 1.]], ["q"], ["a", "b", "c"], method=m.AFTER, contract=contract)
    rows = result["position_rows"]
    assert [r["coverage"] for r in rows] == [0, 0, 1]
    assert rows[0]["absolute_log_error"] is None
    assert rows[2]["absolute_log_error"] == pytest.approx(math.log(1.2))


def test_unknown_probability_is_not_a_negative_for_binary_f1():
    value = m.binary_applicability(np.array([[.9, .1, .99, .99]]), np.array([[0, 1, 2, -1]]))
    assert value["known_pairs"] == 2 and value["positive_f1"] == 1
    assert value["unknown_pairs_excluded"] == 1 and value["invalid_pairs_excluded"] == 1


def test_primary_holm_counts_unavailable_second_test_and_keeps_sign():
    quality = {"mean_group_delta": .1, "bootstrap_95ci": [.01, .2], "independent_groups": 20, "cluster_sign_flip_p": .03}
    execution = {"mean_paired_absolute_log_error_difference": -.1, "two_sided_p": None,
                 "independent_compound_publication_blocks": 2}
    result = m.joint_primary_holm(quality, execution)
    assert result[0]["p_holm"] == .06 and result[0]["higher_is_better"] is True
    assert result[1]["p_holm"] is None and result[1]["higher_is_better"] is False


def tmax_fixture():
    definition = {"endpoint_key": "Tmax|oral", "endpoint": "Tmax", "quantity": "Tmax", "family": "Tmax"}
    identity = {"scenario_id": "q", "endpoint_key": "Tmax|oral", "analysis_stratum": "main", "prediction_kind": "protocol_matched",
                "compound": "drug", "publication_cluster_id": "study", "family": "Tmax"}
    error = {**identity, "candidate_id": "a", "endpoint": "Tmax", "scorable": True, "reason": None,
             "absolute_log_error": None, "signed_log_error": None, "absolute_error_h": 2., "within_twofold": None}
    frozen = {"query_ids": ["q"], "candidate_ids": ["a", "b", "c"], "endpoint_definitions": [definition],
              "position_rows": [{**identity, "reference_eligible": True}], "candidate_errors": [error],
              "endpoint_panels": [{"endpoint_key": "Tmax|oral", "source_stratum": "main:protocol_matched"}], "family_inventory": []}
    return definition, frozen


def test_extended_ties_preserve_conditional_hours_and_fractional_coverage():
    definition, frozen = tmax_fixture()
    result = m.evaluate_extended({m.AFTER: np.array([[3., 3., 1.]])}, ["q"], ["a", "b", "c"], frozen)
    rows = result["position_summary"]
    assert rows[0]["coverage_eligible_reference"] == .5
    assert rows[0]["mae_h"] == 2 and rows[0]["gmfe"] is None
    assert rows[2]["mae_h"] is None and rows[2]["coverage_total41"] == 0


def test_tmax_omission_does_not_report_fold_thresholds():
    _, frozen = tmax_fixture()
    rows = m.extended_omissions({m.AFTER: np.array([[3., 2., 1.]])}, ["q"], ["a", "b", "c"], frozen,
                              {"q": {"same_parent_ids": ["a", "b", "c"]}})
    assert rows[0]["best_measured_absolute_error_h"] == 2
    assert "best_measured_absolute_log_error" not in rows[0]
    assert "thresholds" not in rows[0]["by_k"]["3"]
    assert rows[0]["by_k"]["3"]["best_measured_miss_probability"] is None


def test_paired_tmax_keeps_hour_units_and_withholds_single_study_interval():
    definition, frozen = tmax_fixture()
    scores = {method["method_id"]: np.array([[3., 2., 1.]]) for method in m.METHODS}
    result = m.evaluate_extended(scores, ["q"], ["a", "b", "c"], frozen)
    paired = m.paired_extended(result["position_rows"], [definition])
    row = next(r for r in paired["comparisons"] if r["method"] == m.AFTER and r["rank"] == 1)
    assert row["metric_kind"] == "absolute_hours" and row["mean_paired_error_difference"] == 0
    assert row["bootstrap_95ci"] is None and row["independent_compound_publication_blocks"] == 1
    assert "two_sided_p" not in row and "mean_paired_absolute_log_error_difference" not in row


def test_curve_exact_tie_does_not_replace_missing_candidate():
    case = {"scenario_id": "q", "compound": "drug", "publication_cluster_id": "study",
            "observation": {"time_h": [1, 2]}, "curves": [{"candidate_id": "a", "label": "A"}]}
    candidates = {c: {"family_name": c, "family_id": c, "compounds": ["drug"]} for c in ("a", "b", "c")}
    positions = m.curves.positions([3, 3, 1], ["a", "b", "c"], candidates, case)
    measured = {"a": {"mean_abs_log_error": math.log(2), "mean_squared_log_error": math.log(2) ** 2, "within_twofold_fraction": 1.}}
    result = m.curves.score_position(case, {"method_id": m.AFTER, "method": "after"}, positions[0], measured)
    assert result["curve_coverage"] == .5 and result["curve_gmfe"] == 2
    result = m.curves.score_position(case, {"method_id": m.AFTER, "method": "after"}, positions[2], measured)
    assert result["curve_coverage"] == 0 and result["curve_gmfe"] is None


def test_numpy_counts_serialize_as_numbers_and_nonfinite_output_is_rejected():
    result = json.loads(json.dumps({"count": np.int64(3), "flag": np.bool_(True), "error": np.float32(1.5)},
                                  default=m.json_default, allow_nan=False))
    assert result == {"count": 3, "flag": True, "error": 1.5}
    with pytest.raises(ValueError):
        json.dumps({"error": np.float32(np.nan)}, default=m.json_default, allow_nan=False)


PUBSPEC = importlib.util.spec_from_file_location("component_integration_publish", ROOT / "scripts/publish_component_model_integration.py")
pub = importlib.util.module_from_spec(PUBSPEC)
PUBSPEC.loader.exec_module(pub)


def publication_fixture():
    reports = {}
    for method, _, _ in pub.METHODS:
        reports[method] = {"quality": {"judged_only": {"eligible_queries": 40,
            "metrics": {"ndcg@10": .8, "recall@10": .9, "mrr": .7}}, "coverage": {"queries": 145},
            "paper_binary_102": {"known_pairs": 16410, "positive_pairs": 74, "positive_f1": .5},
            "same_drug_different_family_preference": {"pairs": 43, "eligible_queries": 11, "query_macro_accuracy": .8},
            "known_top10_coverage": {"mean": .3}},
            "execution": {"position_summary": [{"endpoint": pub.C_MAX, "rank": rank, "scenarios": 34,
                 "coverage": .5, "gmfe": 1.5, "covered_scenarios": 17} for rank in (1, 2, 3)]}}
        reports[method]["execution"]["omissions"] = [
            {"endpoint": pub.C_MAX, "scenario_id": f"scenario-{i}", "compound": f"drug-{i % 17}",
             "publication_cluster_id": f"study-{i % 17}", "choice_comparison_possible": i < 19,
             "measured_count": 2, "best_measured_candidate_ids": ["candidate-best"],
             "by_k": {str(k): {"best_measured_miss_probability": .25,
                               "best_measured_hit_probability": .75} for k in (3, 5)}} for i in range(34)]
        reports[method]["execution"]["omission_summary"] = [
            {"endpoint": pub.C_MAX, "k": k, "total_scenarios": 34, "choice_comparison_scenarios": 19,
             "compounds": 17, "compound_study_pairs": 17, "best_measured_miss_probability": .25,
             "best_measured_hit_probability": .75, "aggregation": "Within study and then compound."}
            for k in (3, 5)]
    common = [{"method": pub.AFTER + suffix, "comparator": pub.BEFORE + suffix, "rank": rank,
               "endpoint": pub.C_MAX, "shared_scenarios": 2, "scenario_ids": ["s1", "s2"],
               "own_gmfe": 1.5, "comparator_gmfe": 1.5} for suffix in ("", "_document") for rank in (1, 2, 3)]
    primary = [{"comparison": key, "family_size": 2, "higher_is_better": key == "document_ndcg@10",
                "estimate": 0., "bootstrap_95ci": [0., 0.], "independent_blocks": 20, "p_raw": 1., "p_holm": 1.}
               for key in ("document_ndcg@10", "ordinary_Cmax_rank1")]
    curve_rows = [{"method_id": method, "scenario_id": scenario, "position": rank,
                   "compound": "atorvastatin" if scenario == "s1" else "rosuvastatin",
                   "curve_gmfe": 1.5, "curve_coverage": 1.} for method, _, _ in pub.METHODS
                  for scenario in ("s1", "s2") for rank in (1, 2, 3)]
    curve_rankings = [{"method_id": method, "scenario_id": scenario,
        "positions": [{"position": rank, "members": [{"candidate_id": f"candidate-{rank}", "weight": 1.}]}
                      for rank in (1, 2, 3)]} for method, _, _ in pub.METHODS for scenario in ("s1", "s2")]
    report = {"schema": pub.REPORT_SCHEMA, "state": "complete", "purpose": pub.CONTROL,
              "scope": {"document_requests": 145, "execution_requests": 41, "candidates": 280},
              "validation": {"baseline_metrics_reproduced": True}, "reports": reports,
              "paired": {"primary_family": primary, "common_positions": common},
              "curves": {"independent_publications": 1, "qualified_observations": 43,
                         "position_rows": curve_rows, "rankings": curve_rankings},
              "input_policy": {"information_change": "identity control only"}}
    bridge = {"state": "complete", "requests": 186, "statuses": {"augmented": 2, "blocked": 184},
              "by_cohort": {"core": {"augmented": 1, "blocked": 144}, "execution": {"augmented": 1, "blocked": 40}},
              "profile_eligibility_used_for_ranking": False, "invalid_component_results_use_unchanged_input": True}
    return report, bridge


def test_publication_retains_full_fallback_denominator_and_control_warning():
    report, bridge = publication_fixture()
    result = pub.extract(report, bridge)
    table = pub.latex(result)
    assert result["bridge"]["augmented"] == 2 and result["bridge"]["unchanged"] == 184
    assert "2/186" in table and "IDENTITY CONTROL ONLY" in table
    assert len(result["document"]) == 4 and len(result["ordinary_cmax"]) == 12
    assert len(result["common_cmax"]) == 6
    assert len(result["curve_display"]) == 4
    assert "A. Prespecified paired effects" in table and "B. Best-measured candidate omissions" in table
    assert "Document" not in table and "Input & Score" not in table
    assert "Recall@10" not in table and "Concentration-curve GMFE" not in table
    assert len(result["final_omissions"]) == 4
    assert pub.number(None) == "NA" and pub.number(0.) == "0.000"


def test_publication_rejects_missing_rank_and_different_test_family():
    report, bridge = publication_fixture()
    report["reports"][pub.AFTER]["execution"]["position_summary"].pop()
    with pytest.raises(ValueError, match="all three"):
        pub.extract(report, bridge)
    report, bridge = publication_fixture()
    report["paired"]["primary_family"][0]["family_size"] = 1
    with pytest.raises(ValueError, match="two tests"):
        pub.extract(report, bridge)


def test_publication_checks_cohort_status_totals():
    _, bridge = publication_fixture()
    bridge["by_cohort"]["execution"]["blocked"] = 39
    with pytest.raises(ValueError, match="cohort"):
        pub.validate_bridge(bridge)


def test_changed_subset_is_fixed_by_input_mask_even_when_performance_is_worse():
    report, _ = publication_fixture()
    for value in report["reports"].values():
        value["quality"]["per_query"] = [
            {"query_id": q, "judged_only": {"ndcg@10": metric, "recall@10": metric, "mrr": metric},
             "same_drug_different_family_preference": {"pairs": 1, "accuracy": metric}}
            for q, metric in [("changed", .1), ("unchanged", .9)]]
        value["quality"]["known_top10_coverage"]["per_query"] = [.2, .8]
        value["execution"]["position_rows"] = []
    result = pub.subset_metrics(report, {"changed"}, {"changed": "core", "unchanged": "core"})
    assert result["total"] == 1 and result["document"][0]["ndcg@10"] == .1
    assert result["document"][0]["known_top10_fraction"] == .2
    assert "no subset hypothesis" in result["role"]
    empty = pub.subset_metrics(report, set(), {"changed": "core", "unchanged": "core"})
    assert empty["document"][0]["ndcg@10"] is None


def test_curve_display_does_not_merge_equal_errors_with_different_candidate_ids():
    report, _ = publication_fixture()
    report["curves"]["rankings"][0]["positions"][0]["members"][0]["candidate_id"] = "different"
    rows = pub.curve_display_rows(report["curves"])
    assert len(rows) == 5
    subset = [r for r in rows if r["scenario_id"] == "s1" and r["input"] == "Original"]
    assert {r["score"] for r in subset} == {"Document", "Final"}


def test_curve_display_does_not_merge_equal_missing_errors_with_different_members():
    report, _ = publication_fixture()
    for row in report["curves"]["position_rows"]:
        if row["scenario_id"] == "s1" and row["position"] == 3:
            row.update(curve_gmfe=None, curve_coverage=0.)
    report["curves"]["rankings"][0]["positions"][2]["members"][0]["candidate_id"] = "unmeasured-other"
    assert len(pub.curve_display_rows(report["curves"])) == 5


def test_curve_display_requires_equal_tie_weights_and_keeps_unbound_rankings_separate():
    report, _ = publication_fixture()
    report["curves"]["rankings"][0]["positions"][0]["members"][0]["weight"] = .5
    assert len(pub.curve_display_rows(report["curves"])) == 5
    del report["curves"]["rankings"]
    assert len(pub.curve_display_rows(report["curves"])) == 8


def test_curve_display_rejects_inconsistent_error_for_identical_candidates():
    report, _ = publication_fixture()
    report["curves"]["position_rows"][0]["curve_gmfe"] = 1.6
    with pytest.raises(ValueError, match="inconsistent accuracy"):
        pub.curve_display_rows(report["curves"])


def test_main_quality_uses_same_denominators_and_keeps_unfavorable_component_results():
    run = ROOT / "experiments/model_recommendation/component_integration_20261007"
    summary = pub.extract(pub.read(run / "report/summary.json"), pub.read(run / "inputs/bridge_manifest.json"))
    baseline = pub.read(ROOT / "results/data/model_recommendation_quality.json")
    own = pub.read(ROOT / "results/data/pbpk_match_light_paper_metrics.json")
    rows = pub.main_quality_rows(summary, baseline, own)
    assert rows[:12] == baseline["rows"]
    assert [r["method_id"] for r in rows[-2:]] == [pub.BEFORE, pub.AFTER]
    assert rows[-1]["ranking_eligible_queries"] == rows[-2]["ranking_eligible_queries"] == 40
    assert rows[-1]["test_known_pairs"] == rows[-2]["test_known_pairs"] == 16410
    assert rows[-1]["test_positive_f1"] == rows[-2]["test_positive_f1"] == .52
    assert rows[-1]["ndcg@10"] < rows[-2]["ndcg@10"]
    assert rows[-1]["mrr"] < rows[-2]["mrr"]
    assert rows[-1]["recall@10"] > rows[-2]["recall@10"]
    summary["document"][1]["ndcg@10"] += .01
    with pytest.raises(ValueError, match="reproduce original quality"):
        pub.main_quality_rows(summary, baseline, own)


def test_identity_control_cannot_enter_main_quality_table():
    report, bridge = publication_fixture()
    with pytest.raises(ValueError, match="Only completed experiments"):
        pub.main_quality_rows(pub.extract(report, bridge), {}, {})


def test_paired_omissions_reject_changed_candidate_pool_and_denominator():
    report, _ = publication_fixture()
    report["reports"][pub.AFTER]["execution"]["omissions"][0]["best_measured_candidate_ids"] = ["other"]
    with pytest.raises(ValueError, match="comparison pools differ"):
        pub.final_omission_rows(report)
    report, _ = publication_fixture()
    report["reports"][pub.AFTER]["execution"]["omission_summary"][0]["choice_comparison_scenarios"] = 18
    with pytest.raises(ValueError, match="denominator disagrees"):
        pub.final_omission_rows(report)


def test_paired_omissions_recalculate_hierarchical_rates_and_reject_stale_summary():
    report, _ = publication_fixture()
    execution = report["reports"][pub.AFTER]["execution"]
    execution["omissions"][0]["by_k"]["3"].update(
        best_measured_miss_probability=.75, best_measured_hit_probability=.25)
    with pytest.raises(ValueError, match="hierarchical scenario aggregation"):
        pub.final_omission_rows(report)
    expected = .25 + .5 / 2 / 17  # One of two dose branches in one of 17 compounds.
    execution["omission_summary"][0].update(
        best_measured_miss_probability=expected, best_measured_hit_probability=1-expected)
    rows = pub.final_omission_rows(report)
    result = next(r for r in rows if r["method_id"] == pub.AFTER and r["k"] == 3)
    assert result["miss_probability"] == pytest.approx(expected)
    assert result["miss_probability"] != pytest.approx(.25 + .5 / 19)


def test_compact_paired_table_retains_real_unfavorable_effect_and_omission_scope():
    run = ROOT / "experiments/model_recommendation/component_integration_20261007"
    summary = pub.extract(pub.read(run / "report/summary.json"), pub.read(run / "inputs/bridge_manifest.json"))
    table = pub.latex(summary)
    assert "-0.005 [-0.020, 0.004]" in table and "0.151 [-0.004, 0.462]" in table
    assert table.count("0.973") == 2
    assert "26.47 & 20.59 & -5.88" in table and "5.88 & 5.88 & 0.00" in table
    assert "19 scenarios" in table and "17 compounds and 17 compound--study groups" in table
    assert "Document" not in table and "Score" not in table
    assert len(summary["document"]) == 4 and len(summary["ordinary_cmax"]) == 12
