"""CPU-only reporting checks: no scientific labels or real checkpoints loaded."""
import hashlib
import json

import numpy as np
import pytest

from scripts import summarize_pbpk_match as summary


def cube(values):
    return np.tile(np.asarray(values, dtype=float), (3, 2, 1))


def test_percentile_ensemble_is_invariant_to_per_seed_strictly_increasing_transforms():
    first = cube([-3, -1, -1, 0, 5])
    second = cube([0, 9, 2, 2, 1])
    third = cube([1, 0, -4, 7, 2])
    actual = summary.percentile_ensemble([first, second, third])
    transformed = summary.percentile_ensemble([first ** 3, np.exp(second / 3), 123 * third + 4])
    np.testing.assert_array_equal(actual, transformed)


def test_exact_within_seed_ties_are_preserved():
    actual = summary.percentile_ensemble([cube([2, 2, 1]), cube([5, 5, 0]), cube([9, 9, -1])])
    np.testing.assert_array_equal(actual[..., 0], actual[..., 1])
    assert np.all(actual[..., 0] == 2.5 / 3)


def test_equal_rank_sums_remain_exactly_tied_despite_different_seed_orders():
    actual = summary.percentile_ensemble([cube([1, 2, 3]), cube([2, 3, 1]), cube([3, 1, 2])])
    # All three candidates have rank sum 6, rather than almost-equal float sums.
    np.testing.assert_array_equal(actual, np.full((3, 2, 3), 2 / 3))


@pytest.mark.parametrize("arrays", [[], [np.zeros((2, 2, 3))], [cube([1, 2]), cube([1, 2, 3])], [cube([1, float('nan')])]])
def test_percentile_ensemble_rejects_partial_or_nonfinite_matrices(arrays):
    with pytest.raises(ValueError):
        summary.percentile_ensemble(arrays)


def test_seed_completion_does_not_call_three_of_four_seeds_complete():
    result = summary.seed_completion([17, 29, 43, 59], [17, 29, 43])
    assert result["state"] == "partial"
    assert result["missing_seeds"] == [59]
    assert summary.seed_completion([17], [])["state"] == "not_started"
    assert summary.seed_completion([17, 29], [29, 17])["state"] == "complete"


@pytest.mark.parametrize("expected,completed", [([17, 17], [17]), ([17], [17, 17]), ([17], [29]), ([True], []), ([], [])])
def test_duplicate_or_undeclared_seeds_cannot_manufacture_replicates(expected, completed):
    with pytest.raises(ValueError):
        summary.seed_completion(expected, completed)


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    output = tmp_path / "pbpk_match"
    run = output / "runs" / "demo"
    run.mkdir(parents=True)
    snapshot = output / "development_snapshot.json"
    snapshot.write_text('{"fixture":"no real labels"}')
    data = {
        "grades": np.array([[3, 0, -1], [3, 1, 0]]),
        "applicability": np.array([[0, 1, 2], [0, 2, 1]]),
        "query_records": [{"query_id": f"q{i}", "compound_id": f"drug{i}", "compounds": [f"drug{i}"],
                           "variants": {"missing_evidence": {"text_changed": True}}} for i in range(2)],
        "candidate_records": [{"candidate_id": f"c{i}", "family_id": f"f{i}", "compounds": []} for i in range(3)],
    }
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    monkeypatch.setattr(summary, "scoped", lambda path: summary.Path(path).resolve())
    monkeypatch.setattr(summary, "sha", lambda path: hashlib.sha256(summary.Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(summary, "build_data", lambda *args, **kwargs: data)
    def write_json(path, values):
        summary.Path(path).write_text(json.dumps(values, allow_nan=False))
    monkeypatch.setattr(summary, "write_json", write_json)
    monkeypatch.setattr(summary, "baseline_reports", lambda *args, **kwargs: {
        "jina_reranker_v35": summary.evaluate(data, np.tile([2., 1., 0.], (2, 1)))})
    def save_run(expected, completed, *, missing_prediction_seed=None):
        (run / "protocol.json").write_text(json.dumps({"snapshot_sha256": summary.sha(snapshot),
                                                       "methods": ["mlp"], "seeds": expected}))
        for seed in completed:
            folder = run / f"mlp_seed{seed}"
            folder.mkdir(exist_ok=True)
            (folder / "summary.json").write_text(json.dumps({"fold_count": 3, "method": "mlp", "seed": seed}))
            if seed != missing_prediction_seed:
                np.savez(folder / "out_of_fold_predictions.npz", scores=cube([2, 1, 0]),
                         probabilities=np.tile([.5, .3, .2], (2, 3, 1)))
        return output
    return save_run, data, run


def test_partial_run_keeps_completed_prototypes_but_does_not_publish_partial_ensemble(fake_run):
    save, _, _ = fake_run
    output = save([17, 29, 43, 59], [17, 29, 43])
    result = summary.summarize(output)
    assert result["completion_state"] == "partial"
    assert len(result["completed"]) == 3
    assert result["incomplete_configurations"] == ["demo/mlp_seed59"]
    assert result["seed_summaries"]["demo/mlp"]["state"] == "partial"
    assert result["ensembles"] == {}
    assert result["scientific_gold"] is False


def test_summary_without_predictions_is_incomplete_not_complete(fake_run):
    save, _, _ = fake_run
    result = summary.summarize(save([17], [17], missing_prediction_seed=17))
    assert result["completion_state"] == "partial"
    assert result["completed"] == {}
    assert result["seed_summaries"]["demo/mlp"]["state"] == "not_started"


def test_no_positive_labels_keep_null_metrics_without_crash_or_perfect_scores(fake_run):
    save, data, _ = fake_run
    data["grades"][:] = np.array([[0, 1, -1], [0, 1, 0]])
    result = summary.summarize(save([17, 29, 43], [17, 29, 43]))
    assert result["completion_state"] == "complete"
    metric = result["seed_summaries"]["demo/mlp"]["metrics"]["ndcg@10"]
    assert metric["mean"] is None and metric["seed_sd"] is None
    assert metric["defined_seed_count"] == 0
    assert result["ensembles"]["demo/mlp"]["judged_only"]["metrics"]["mrr"] is None


def test_complete_ensemble_has_verified_identical_label_pools_and_seed_sd_not_ci(fake_run):
    save, _, _ = fake_run
    result = summary.summarize(save([17, 29, 43], [17, 29, 43]))
    assert result["completion_state"] == "complete"
    group = result["seed_summaries"]["demo/mlp"]
    assert group["metrics"]["mrr"]["seed_sd"] == 0
    assert "not a confidence interval" in group["interpretation"]
    ensemble = result["ensembles"]["demo/mlp"]
    paired = ensemble["paired_external_comparisons"]["jina_reranker_v35"]
    assert paired["label_pools_verified"] is True
    assert paired["eligible_queries"] == 2
    assert paired["eligible_compounds"] == 2


def test_summary_checks_all_variant_predictions_not_only_original(fake_run):
    save, _, run = fake_run
    output = save([17], [17])
    scores = cube([2, 1, 0])
    scores[2, 0, 0] = np.nan
    np.savez(run / "mlp_seed17" / "out_of_fold_predictions.npz", scores=scores,
             probabilities=np.tile([.5, .3, .2], (2, 3, 1)))
    with pytest.raises(ValueError, match="three finite"):
        summary.summarize(output)


@pytest.mark.parametrize('complete_baseline', [True, False])
def test_supervised_comparison_needs_both_complete_matched_seed_ensembles(fake_run, complete_baseline):
    save, _, run = fake_run
    seeds = [17, 29, 43]
    output = save(seeds, seeds if complete_baseline else seeds[:2])
    protocol = json.loads((run / 'protocol.json').read_text())
    protocol['methods'].append('hybrid')
    (run / 'protocol.json').write_text(json.dumps(protocol))
    for seed in seeds:
        folder = run / f'hybrid_seed{seed}'
        folder.mkdir()
        (folder / 'summary.json').write_text(json.dumps({'fold_count': 3, 'method': 'hybrid', 'seed': seed}))
        np.savez(folder / 'out_of_fold_predictions.npz', scores=cube([0, 1, 2]),
                 probabilities=np.tile([.5, .3, .2], (2, 3, 1)))
    result = summary.summarize(output)
    comparisons = result['paired_supervised_comparisons']
    if not complete_baseline:
        assert comparisons == {}
        assert 'demo/mlp' not in result['ensembles']
    else:
        comparison = comparisons['demo/hybrid']
        assert comparison['baseline'] == 'demo/mlp'
        assert comparison['raw_ranking']['label_pools_verified'] is True
        assert comparison['raw_ranking']['metrics']['ndcg@10']['point_difference'] < 0
