"""CPU-only tests for nested selection, support gates and immutable job reuse.

The trainer/predictor are mocked: these tests never initialize CUDA, load model
weights or consume an external clinical observation.
"""
import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from pbpk_kb.recommendation.pbpk_match_v2 import evaluation as ev
from pbpk_kb.recommendation.pbpk_match_v2 import experiment as ex


def prediction(rows=3, candidates=3):
    base = np.tile(np.arange(candidates, dtype=np.float32), (rows, 1))
    return {"scores": base.copy(), "base_scores": base, "selected_predicted_error": np.ones_like(base),
            "error_supported": np.ones(rows, bool),
            "probabilities": np.full((rows, candidates, 3), 1 / 3)}


def test_correction_only_affects_explicitly_supported_requests():
    p = prediction()
    p["error_supported"] = np.array([True, False, True])
    corrected = ex.correction(p, .3)
    np.testing.assert_allclose(corrected[0], p["base_scores"][0] - .3)
    np.testing.assert_array_equal(corrected[1], p["base_scores"][1])
    np.testing.assert_array_equal(ex.correction(p, 0), p["base_scores"])
    assert not np.shares_memory(corrected, p["base_scores"])


@pytest.mark.parametrize("bad", ["missing_support", "numeric_support", "wrong_support_shape", "negative_weight", "nan_weight", "nonfinite_base", "wrong_risk_shape", "negative_risk"])
def test_correction_rejects_unproven_support_or_invalid_arrays(bad):
    p, weight = prediction(), .3
    if bad == "missing_support": p.pop("error_supported")
    if bad == "numeric_support": p["error_supported"] = np.array([1, 2, 0])
    if bad == "wrong_support_shape": p["error_supported"] = np.array([True])
    if bad == "negative_weight": weight = -.1
    if bad == "nan_weight": weight = float("nan")
    if bad == "nonfinite_base": p["base_scores"][0, 0] = float("inf")
    if bad == "wrong_risk_shape": p["selected_predicted_error"] = np.ones((3, 2))
    if bad == "negative_risk": p["selected_predicted_error"][0, 0] = -1
    with pytest.raises(ValueError):
        ex.correction(p, weight)


def selection_fixture(monkeypatch, outcomes, ndcgs=None):
    """Each score row is -alpha, making controlled OOF decisions transparent."""
    count = len(next(iter(outcomes.values())))
    data = {"query_records": [{"query_id": f"q{i}"} for i in range(count + 2)],
            "candidate_records": [{"candidate_id": f"c{i}"} for i in range(3)],
            "core_indices": [0], "execution_indices": list(range(1, count + 1))}
    captured = []
    def quality(data, scores, indices):
        assert indices == [0]
        alpha = round(-float(scores[0, 0]), 1)
        return {"judged_only": {"metrics": {"ndcg@10": (ndcgs or {}).get(alpha, 1.)}}}
    def clinical(scores, query_ids, candidate_ids, **kwargs):
        assert query_ids == [f"q{i}" for i in range(1, count + 1)]
        captured.append(query_ids)
        alpha = round(-float(scores[0, 0]), 1)
        return {"position_rows": [{"scenario_id": sid, "compound": sid, "publication_cluster_id": sid,
                                   "endpoint": "Cmax_reference_vs_group", "rank": 1,
                                   "coverage": coverage, "absolute_log_error": error}
                                  for sid, (coverage, error) in zip(query_ids, outcomes[alpha])]}
    monkeypatch.setattr(ev, "quality_evaluation", quality)
    monkeypatch.setattr(ev, "clinical_evaluation", clinical)
    pred = prediction(count + 1)
    pred["scores"][:] = 0
    pred["base_scores"][:] = 0
    return data, pred, captured


def test_alpha_rejects_doc_decline_or_coverage_loss_and_never_uses_outer_rows(monkeypatch):
    outcomes = {0.: [(1, 1.), (1, 1.)], .1: [(1, 0.), (0, None)],
                .3: [(1, 0.), (1, 0.)], 1.: [(1, 2.), (1, 2.)]}
    data, pred, captured = selection_fixture(monkeypatch, outcomes, {.3: .97})
    result = ex.select_error_weight(data, [0, 1, 2], pred)
    assert result["weight"] == 0
    trials = {r["weight"]: r for r in result["trials"]}
    assert not trials[.1]["feasible"]  # Better conditional error cannot conceal lost coverage.
    assert not trials[.3]["feasible"]  # A .03 document NDCG loss exceeds the .02 bound.
    assert all("q3" not in rows for rows in captured)


def test_alpha_comparison_uses_one_common_case_set_across_feasible_weights(monkeypatch):
    # All nonzero feasible weights preserve 2/3 coverage, but alpha .1 swaps
    # case q2 for q3. Comparing its q1-only -.5 to alpha .3's q1/q2 mean -.4
    # spuriously chooses .1. On the shared q1, .3 has the lower error (.2).
    outcomes = {0.: [(1, 1.), (1, 1.), (0, None)],
                .1: [(1, .5), (0, None), (1, 10.)],
                .3: [(1, .2), (1, 1.), (0, None)],
                1.: [(1, 2.), (1, 2.), (0, None)]}
    data, pred, _ = selection_fixture(monkeypatch, outcomes)
    result = ex.select_error_weight(data, [0, 1, 2, 3], pred)
    assert result["weight"] == .3
    feasible = [r for r in result["trials"] if r["feasible"]]
    assert len({r["common_cases"] for r in feasible}) == 1


def test_alpha_without_inner_clinical_pool_stays_zero():
    data = {"query_records": [{"query_id": "q"}], "candidate_records": [{"candidate_id": "c"}],
            "core_indices": [], "execution_indices": []}
    assert ex.select_error_weight(data, [0], prediction(1, 1))["weight"] == 0


def test_quality_unknown_and_invalid_labels_never_become_negative_examples():
    data = {"query_records": [{"query_id": "q", "compounds": ["drug"]}],
            "candidate_records": [{"candidate_id": f"c{i}", "compounds": ["drug"], "family_id": f"f{i}"} for i in range(4)],
            "core_indices": [0], "grades": np.array([[3, 0, -1, -1]]), "applicability": np.array([[0, 1, 2, -1]])}
    report = ev.quality_evaluation(data, [[1, 2, 4, 3]], np.full((1, 4, 3), 1 / 3))
    assert report["coverage"]["known_grade_pairs"] == 2
    assert report["coverage"]["unknown_grade_pairs"] == 2
    assert report["coverage"]["full_catalog_top1_unknown_rate"] == 1
    assert report["strict_full_pool"]["eligible_queries"] == 0
    assert report["applicability"]["class_support"] == {"Y": 1, "N": 1, "U": 1}
    assert report["applicability"]["binary_known_pairs"] == 2
    assert report["applicability"]["invalid_or_unprocessed_pairs"] == 1


@pytest.mark.parametrize("bad", ["nan", "not_normalized", "negative", "wrong_shape", "bad_label", "wrong_groups"])
def test_temperature_rejects_invalid_inputs(bad):
    probabilities = np.full((1, 3, 3), 1 / 3)
    labels, groups = np.array([[0, 1, 2]]), ["g"]
    if bad == "nan": probabilities[0, 0, 0] = np.nan
    if bad == "not_normalized": probabilities *= 2
    if bad == "negative": probabilities[0, 0] = [-1, 1, 1]
    if bad == "wrong_shape": probabilities = probabilities[:, :2]
    if bad == "bad_label": labels[0, 0] = 3
    if bad == "wrong_groups": groups = []
    with pytest.raises(ValueError):
        ev.fit_oof_temperature(probabilities, labels, groups)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_calibration_requires_finite_positive_temperature(temperature):
    with pytest.raises(ValueError):
        ev.calibrate(np.full((1, 1, 3), 1 / 3), temperature)


@pytest.fixture
def orchestration(monkeypatch, tmp_path):
    data = {"query_records": [{"query_id": f"q{i}"} for i in range(6)],
            "candidate_records": [{"candidate_id": f"c{i}"} for i in range(3)],
            "document_texts": [f"candidate-{i}" for i in range(3)],
            "no_numeric_document_texts": [f"candidate-{i}" for i in range(3)],
            "pair_features": np.zeros((6, 3, 2), np.float32),
            "pair_features_by_variant": np.zeros((3, 6, 3, 2), np.float32),
            "core_indices": list(range(6)), "execution_indices": [], "groups": [f"g{i}" for i in range(6)],
            "variant_names": ["original", "reordered", "missing_evidence"],
            "grades": np.tile([3, 0, -1], (6, 1)), "applicability": np.tile([0, 1, 2], (6, 1)), "folds": []}
    for test in ([3, 4, 5], [0, 1, 2]):
        train = sorted(set(range(6)) - set(test))
        data["folds"].append({"outer_test": test, "outer_train": train,
                              "inner": [{"train": sorted(set(train) - {v}), "validation": [v]} for v in train]})
    train_calls, predict_calls, selection_calls, calibration_groups = [], [], [], []
    fake = types.ModuleType("pbpk_kb.recommendation.pbpk_match_v2.training")
    def train(data, indices, validation, output, **kwargs):
        train_calls.append((list(indices), list(validation), Path(output), kwargs))
        assert not set(indices) & set(validation)
        assert not {data["groups"][i] for i in indices} & {data["groups"][i] for i in validation}
        checkpoint = Path(output) / "checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "heads.pt").write_bytes(b"mock CPU checkpoint")
        metadata = {"checkpoint": str(checkpoint), "best_epoch": 2, "epochs_trained": 2,
                    "method": kwargs["method"], "seed": kwargs["seed"], "train_indices": list(indices),
                    "validation_indices": list(validation)}
        (checkpoint / "metadata.json").write_text(json.dumps(metadata))
        return metadata
    def predict(data, indices, checkpoint, **kwargs):
        predict_calls.append((list(indices), Path(checkpoint), kwargs))
        result = prediction(len(indices), 3)
        # Candidate-identity scores, so the real audit can inverse-permute them.
        result["base_scores"][:] = [int(c["candidate_id"][1:]) for c in data["candidate_records"]]
        result["scores"][:] = result["base_scores"]
        return result
    fake.train_fold, fake.predict_fold = train, predict
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(ex, "OUT", tmp_path / "experiment")
    monkeypatch.setattr(ex, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(ex, "METHODS", ("full",))
    monkeypatch.setattr(ex, "SEEDS", (17,))
    monkeypatch.setattr(ex, "status", lambda *args, **kwargs: None)
    def select(data, indices, prediction):
        selection_calls.append(list(indices))
        return {"weight": 0., "trials": [], "reason": "mock inner selection"}
    def calibration(probabilities, labels, groups):
        calibration_groups.append(list(groups))
        return {"temperature": 1., "state": "fitted"}
    monkeypatch.setattr(ex, "select_error_weight", select)
    monkeypatch.setattr(ev, "fit_oof_temperature", calibration)
    # Real protocols also bind these files; harmless placeholders make the
    # mocked integration exercise the actual job persistence code.
    ex.OUT.mkdir()
    (ex.OUT / "manifest.json").write_text('{"test_fixture":true}')
    (ex.OUT / "protocol.json").write_text('{"test_fixture":true}')
    return data, train_calls, predict_calls, selection_calls, calibration_groups


def test_outer_labels_never_choose_epochs_alpha_or_calibration(orchestration):
    data, train, predict, selection, calibration = orchestration
    ex.train_experiment(data, device="cpu", batch_size=4)
    assert len(train) == 8  # 2 outer folds × (3 inner fits + 1 outer refit).
    for fold, used, groups in zip(data["folds"], selection, calibration):
        assert set(used) == set(fold["outer_train"])
        assert set(groups) == {data["groups"][i] for i in fold["outer_train"]}
        assert not set(used) & set(fold["outer_test"])
    for indices, validation, path, kwargs in train:
        fold = data["folds"][int(path.parent.name.removeprefix("fold"))]
        assert set(indices) <= set(fold["outer_train"])
        assert set(validation) <= set(fold["outer_train"])
        if path.name == "outer":
            assert validation == [] and kwargs["epochs"] == 2
    with np.load(ex.OUT / "predictions/full_seed17.npz", allow_pickle=False) as saved:
        assert saved["scores"].shape == (3, 6, 3) and np.isfinite(saved["scores"]).all()
    calls = len(train)
    ex.train_experiment(data, device="cpu", batch_size=4)
    assert len(train) == calls  # Unchanged, bound complete checkpoints may resume.


def test_resume_rejects_finite_but_changed_outer_prediction(orchestration):
    data, *_ = orchestration
    ex.train_experiment(data, device="cpu", batch_size=4)
    path = ex.OUT / "runs/full/seed17/fold0/outer/predictions.npz"
    with np.load(path, allow_pickle=False) as saved:
        values = {k: saved[k] for k in saved.files}
    values["scores_0"][0, 0] += 100  # Remains finite; needs content binding, not merely a NaN check.
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError):
        ex.train_experiment(data, device="cpu", batch_size=4)


def test_resume_rejects_outer_row_substituted_into_inner_oof(orchestration, monkeypatch):
    data, *_ = orchestration
    def stop_before_outer(*args, **kwargs):
        raise RuntimeError("intentional stop after inner checkpoints")
    monkeypatch.setattr(ex, "select_error_weight", stop_before_outer)
    with pytest.raises(RuntimeError, match="intentional stop"):
        ex.train_experiment(data, device="cpu", batch_size=4)
    path = ex.OUT / "runs/full/seed17/fold0/inner0/validation.npz"
    with np.load(path, allow_pickle=False) as saved:
        values = {k: saved[k] for k in saved.files}
    values["indices"][0] = data["folds"][0]["outer_test"][0]
    np.savez_compressed(path, **values)
    monkeypatch.setattr(ex, "select_error_weight", lambda *a: {"weight": 0, "trials": []})
    with pytest.raises(ValueError):
        ex.train_experiment(data, device="cpu", batch_size=4)


def test_compare_inference_distinguishes_score_identity_rank_identity_and_tie_splitting():
    scores = np.array([[3., 3., 1.], [2., 2., 2.]])
    same = ex.compare_inference(scores, scores.copy())
    assert same["exact_equal"] and same["top1_tie_set_agreement"] == 1
    assert same["mean_spearman"] == 1 and same["max_absolute_score_delta"] == 0
    shifted = ex.compare_inference(scores, scores + 10)
    assert not shifted["exact_equal"] and shifted["top1_tie_set_agreement"] == 1
    assert shifted["mean_spearman"] == 1 and shifted["max_absolute_score_delta"] == 10
    split = scores.copy()
    split[0, 1] -= 1e-5
    different = ex.compare_inference(scores, split)
    assert different["top1_tie_set_agreement"] == .5
    assert different["mean_spearman"] < 1


@pytest.mark.parametrize("right", [np.ones((1, 2)), np.array([[np.nan, 1, 2]]), np.array([[1, np.inf, 2]]), np.ones(3)])
def test_compare_inference_rejects_invalid_inputs(right):
    with pytest.raises(ValueError):
        ex.compare_inference(np.ones((1, 3)), right)


def test_inference_integrity_is_uncached_and_restores_candidate_identity(monkeypatch):
    candidate_count = 4
    features = np.tile(np.arange(candidate_count, dtype=np.float32)[None, :, None], (6, 1, 2))
    data = {"candidate_records": [{"candidate_id": f"c{i}"} for i in range(candidate_count)],
            "document_texts": [f"doc{i}" for i in range(candidate_count)],
            "no_numeric_document_texts": [f"plain{i}" for i in range(candidate_count)],
            "pair_features": features, "pair_features_by_variant": np.stack([features] * 3)}
    # There are no outcome fields at all in this fixture.
    calls = []
    fake = types.ModuleType("pbpk_kb.recommendation.pbpk_match_v2.training")
    def predict(d, indices, checkpoint, **kwargs):
        ids = [int(c["candidate_id"][1:]) for c in d["candidate_records"]]
        assert d["document_texts"] == [f"doc{i}" for i in ids]
        assert d["no_numeric_document_texts"] == [f"plain{i}" for i in ids]
        np.testing.assert_array_equal(d["pair_features"][0, :, 0], ids)
        np.testing.assert_array_equal(d["pair_features_by_variant"][0, 0, :, 0], ids)
        assert kwargs["use_cache"] is False
        calls.append((list(indices), kwargs["batch_size"], ids))
        result = prediction(len(indices), candidate_count)
        result["base_scores"][:] = ids
        result["scores"][:] = result["base_scores"]
        return result
    fake.predict_fold = predict
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    audit = ex.inference_integrity(data, [5, 2, 4], "/unused-mock-checkpoint", device="cpu", batch_size=32, weight=.3)
    assert audit["indices"] == [2, 4]
    assert [c[1] for c in calls] == [32, 32, 16, 32]
    assert all(c[0] == [2, 4] for c in calls)
    for key in ("repeat", "candidate_order", "batch_size"):
        assert audit[key]["exact_equal"] and audit[key]["top1_tie_set_agreement"] == 1
    assert calls[-1][2] != list(range(candidate_count))
