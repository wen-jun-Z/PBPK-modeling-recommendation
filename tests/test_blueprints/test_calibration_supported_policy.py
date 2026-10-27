import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pbpk_kb.blueprint_learning import calibration_supported_policy as policy

ROOT = Path(__file__).resolve().parents[2]


def sample_rows(n=60, grouped=True):
    rows, features = [], {}
    for i in range(n):
        group = i // 3 if grouped else i
        cluster = group % 2
        rows.append({"index": i, "identity_smiles": f"id_{i}", "calibration_id": f"cal_{i}",
                     "scaffold_group": f"scaffold_{group}", "residual_log": -2 if cluster == 0 else 2})
        bits = np.zeros(1024)
        bits[cluster] = 1
        features[i] = bits
    return rows, features


def test_empirical_crps_known_scores_and_translation_invariance():
    assert policy.empirical_crps([0], 2) == 2
    assert policy.empirical_crps([-1, 1], 0) == .5
    assert policy.empirical_crps([-1, 1], 1) == .5
    assert policy.empirical_crps([-1, 1], 3) == 2.5
    assert policy.empirical_crps([99, 101], 100) == .5
    assert policy.empirical_crps([1, 1, 1], 1) == 0


@pytest.mark.parametrize("values,truth", [([], 1), ([float('nan')], 1), ([0], float('inf')), ([[1, 2]], 1)])
def test_invalid_scores_rejected(values, truth):
    with pytest.raises(ValueError, match="CRPS requires"):
        policy.empirical_crps(values, truth)


def test_loso_excludes_entire_scaffold_and_supports_useful_locality():
    rows, features = sample_rows()
    result = policy.select_endpoint(rows, features)
    assert result["family"] == "local"
    assert result["local_scaffold_equal_crps"] == 0
    assert result["paired_scaffold_improvement_mean"] > result["paired_scaffold_improvement_se"]
    assert result["paired_scaffold_n"] == 20
    for fold in result["folds"]:
        held = set(fold["held_indices"])
        assert len(held) == 3
        assert not held & set(fold["reference_indices"])
        for scored in fold["scores"]:
            assert not held & set(scored["local_reference_indices"])
            assert scored["local_reference_n"] == 20


def test_held_truth_and_same_scaffold_truth_never_enter_reference(monkeypatch):
    rows, features = sample_rows()
    rows[0]["residual_log"] = 1000
    rows[1]["residual_log"] = 777
    score = policy.empirical_crps
    seen = []
    def guard(samples, truth):
        if truth == 1000:
            assert 1000 not in samples and 777 not in samples
            seen.append(len(samples))
        return score(samples, truth)
    monkeypatch.setattr(policy, "empirical_crps", guard)
    policy.select_endpoint(rows, features)
    assert sorted(seen) == [20, 57]


def test_no_score_improvement_defaults_global():
    rows, features = sample_rows(12, grouped=False)
    result = policy.select_endpoint(rows, features)
    assert result["all_scaffolds_scored"]
    assert result["family"] == "global"
    assert abs(result["paired_scaffold_improvement_mean"]) <= 1e-12


@pytest.mark.parametrize("n", [0, 9, 12])
def test_insufficient_fold_reference_support_defaults_global(n):
    rows, features = sample_rows(n, grouped=True)
    result = policy.select_endpoint(rows, features)
    assert result["family"] == "global"
    assert result["selection_reason"] == "insufficient_leave_scaffold_out_support"


def test_equal_scaffold_not_equal_row_aggregation():
    rows, features = sample_rows()
    # A single large cluster must not count as multiple independent scaffolds.
    for row in rows[:12]:
        row["scaffold_group"] = "large_scaffold"
    result = policy.select_endpoint(rows, features)
    deltas = [f["paired_improvement_global_minus_local"] for f in result["folds"]]
    assert result["paired_scaffold_improvement_mean"] == pytest.approx(np.mean(deltas))
    assert result["paired_scaffold_improvement_se"] == pytest.approx(np.std(deltas, ddof=1)/np.sqrt(len(deltas)))


def test_duplicate_calibration_identity_is_rejected():
    rows, features = sample_rows()
    rows[1]["identity_smiles"] = rows[0]["identity_smiles"]
    with pytest.raises(ValueError, match="duplicate"):
        policy.select_endpoint(rows, features)


def test_real_sources_are_bound_no_target_metrics_are_read_and_tampering_is_rejected(tmp_path, monkeypatch):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use --basetemp inside project runtime")
    source = ROOT / policy.REFERENCE_PATH
    if not source.exists():
        pytest.skip("Frozen calibration reference artifact unavailable")
    actual_read = Path.read_text
    def guard(path, *args, **kwargs):
        if "evaluation" in path.parts or path.name in {"results.json", "results.csv", "decisions.json"}:
            raise AssertionError("Target decisions/outcomes opened")
        return actual_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guard)
    output = tmp_path / "calibration_policy.json"
    artifact = policy.fit_calibration_policy(ROOT, output)
    assert not artifact["target_outcome_files_opened"]
    assert not artifact["target_metrics_used_for_selection"]
    assert not artifact["parameter_estimators_refitted"]
    assert artifact["calibration_count"] == 100
    assert set(artifact["endpoints"]) == {"clearance", "vss", "terminal_half_life", "mrt"}
    assert policy.fit_calibration_policy(ROOT, output) == artifact
    altered = copy.deepcopy(artifact)
    altered["endpoints"]["clearance"]["family"] = "tampered"
    output.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="policy changed"):
        policy.fit_calibration_policy(ROOT, output)
    bad_source = json.loads(source.read_text())
    bad_source["calibration_residuals"]["clearance"][0]["residual_log"] += 1
    changed = tmp_path / "altered_reference.json"
    changed.write_text(json.dumps(bad_source))
    with pytest.raises(ValueError, match="reference prior artifact changed"):
        policy.fit_calibration_policy(ROOT, tmp_path / "unused.json", reference_path=changed)
