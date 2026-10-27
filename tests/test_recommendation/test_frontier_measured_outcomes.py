from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from evaluate_frontier_measured_outcomes import (curve_metrics, recompute, source_role, study_tokens,
                                                compare_snapshot_versions)


def example():
    return {"outcome_record_id": "p:1", "context": {"time_values": [1, 2, 4],
        "observed_values": [1, 3, 1], "observed_unit": "µmol/l", "time_unit": "h"},
        "positive_pairs": {"observed_mol_per_litre": [1e-6, 3e-6, 1e-6],
                           "predicted_mol_per_litre": [2e-6, 6e-6, 2e-6]},
        "observation_count": 3, "positive_pair_count": 3,
        "metrics": {**curve_metrics([1, 3, 1], [2, 6, 2]),
                    "auc_observed_window_ratio": 2, "sampled_cmax_ratio": 2}}


def test_endpoints_are_recomputed_on_observed_window():
    result = recompute(example())
    assert result["endpoints"]["start_h"] == 1
    assert result["endpoints"]["observed_auc"] == pytest.approx(6e-6)
    assert result["endpoints"]["auc_ratio"] == 2
    assert result["curve_metrics"]["gmfe"] == 2


def test_days_are_converted_before_auc_integration():
    row = example()
    row["context"]["time_unit"] = "day(s)"
    result = recompute(row)
    assert result["endpoints"]["start_h"] == 24
    assert result["endpoints"]["observed_auc"] == pytest.approx(144e-6)
    assert result["endpoints"]["auc_ratio"] == 2


def test_source_ratio_cannot_override_actual_predictions():
    row = example()
    row["metrics"]["auc_observed_window_ratio"] = 1
    with pytest.raises(ValueError, match="endpoint differs"):
        recompute(row)


def test_unaligned_positive_subset_cannot_produce_auc():
    row = example()
    row["observation_count"] = 4
    assert recompute(row)["endpoints"]["auc_ratio"] is None


def test_single_point_is_not_a_peak_series():
    row = example()
    row["context"]["time_values"] = [1]
    row["context"]["observed_values"] = [1]
    row["positive_pairs"] = {"observed_mol_per_litre": [1e-6], "predicted_mol_per_litre": [2e-6]}
    row["observation_count"] = row["positive_pair_count"] = 1
    assert recompute(row)["endpoints"]["sampled_cmax_ratio"] is None


def test_same_paper_building_overlap_is_not_independent_validation():
    protocol = {"source": {"repository_url": "repo"}, "simulations": [
        {"role": "model_building", "observed_dataset_names": ["build"]},
        {"role": "model_verification", "observed_dataset_names": ["test"]}]}
    snapshot = {"ObservedData": [{"Name": name, "ExtendedProperties": [
        {"Name": "Reference", "Value": "https://pubmed.ncbi.nlm.nih.gov/12345/"}]} for name in ["build", "test"]]}
    row = {"dataset_name": "test", "source_location": "ObservedData/1", "context": {}}
    result = source_role(row, protocol, snapshot, "snapshot.json")
    assert result["same_study_in_model_building"]
    assert not result["same_dataset_in_model_building"]
    assert result["independence_status"] == "known_model_building_overlap"
    assert result["independence_verified"] is False


def test_missing_reference_remains_unknown():
    assert study_tokens({}, "repo") == []


def test_retrospective_comparison_requires_distinct_snapshot_same_repository():
    originals = [{"outcome_record_id": name, "protocol_key": name} for name in ["a", "b"]]
    protocols = {name: ({"source_snapshot_sha256": name}, {"source": {"repository_url": "repo"}}, {}, None)
                 for name in ["a", "b"]}
    rows = [{"observation_fingerprint": "same", "source_outcome_ids": [name], "measurement_id": name,
             "dataset_name": "study-dose", "exact_candidate_ids": [name] if name == "a" else [],
             "curve_metrics": {"gmfe": 1 if name == "a" else 2, "log_rmse": 0 if name == "a" else 1},
             "endpoints": {"auc_absolute_log_error": None, "sampled_cmax_absolute_log_error": None}}
            for name in ["a", "b"]]
    result = compare_snapshot_versions(rows, originals, protocols)
    assert result["pair_count"] == 1
    assert result["lower_error_counts"]["gmfe"] == {"left": 1}
    assert not result["pairs"][0]["current_recommender_evaluation_eligible"]
    assert not result["ranking_labels_created"]
    protocols["b"][0]["source_snapshot_sha256"] = "a"
    assert compare_snapshot_versions(rows, originals, protocols)["pair_count"] == 0
    protocols["b"][0]["source_snapshot_sha256"] = "b"
    protocols["b"][1]["source"]["repository_url"] = "another repository"
    assert compare_snapshot_versions(rows, originals, protocols)["pair_count"] == 0


@pytest.mark.parametrize("observed,predicted", [([], []), ([1], [0]), ([1], [float("nan")]), ([1], [1, 2])])
def test_bad_pairs_rejected(observed, predicted):
    with pytest.raises(ValueError):
        curve_metrics(observed, predicted)
