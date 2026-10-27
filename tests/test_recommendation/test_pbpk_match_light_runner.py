"""Orchestration boundaries, using tiny substitute heads rather than GPU fits."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_pbpk_match_light as runner
import pbpk_match_light_training as training
from pbpk_kb.recommendation.pbpk_match_v2 import evaluation


def configured(tmp_path, monkeypatch, *, fail_final=False):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUT", tmp_path / "light")
    monkeypatch.setattr(runner, "DATA", tmp_path / "data")
    monkeypatch.setattr(runner, "RUNTIME", tmp_path / "runtime")
    for path in (runner.OUT / "protocol.json", runner.OUT / "cache/manifest.json", runner.DATA / "manifest.json"):
        runner.write(path, {"test": "synthetic orchestrator inputs"})
    query_count, candidate_count = 10, 3
    folds = []
    for f in range(5):
        test = [2 * f, 2 * f + 1]
        train = sorted(set(range(query_count)) - set(test))
        inner = []
        for validation in np.array_split(train, 3):
            validation = validation.tolist()
            inner.append({"train": sorted(set(train) - set(validation)), "validation": validation})
        folds.append({"outer_train": train, "outer_test": test, "inner": inner})
    data = {"query_records": [{"query_id": f"q{i}"} for i in range(query_count)],
            "candidate_records": [{"candidate_id": f"c{i}"} for i in range(candidate_count)],
            "groups": list(range(query_count)), "folds": folds,
            "core_indices": list(range(0, query_count, 2)), "execution_indices": list(range(1, query_count, 2)),
            "applicability": np.zeros((query_count, candidate_count), np.int8),
            "variant_names": ["original", "reordered", "missing_evidence"],
            "light_candidate_embeddings": np.ones((candidate_count, 2), np.float32),
            "pair_features_by_variant": np.ones((3, query_count, candidate_count, 2), np.float32)}
    fits, selections, events = [], [], []

    def fake_fit(d, train, validation, output, **kwargs):
        output = Path(output)
        fold = folds[int(output.parent.name.removeprefix("fold"))]
        assert not set(train) & set(validation)
        assert not set(train + validation) & set(fold["outer_test"])
        assert kwargs["seed"] == 17
        assert kwargs["pk_epochs"] == 80
        fits.append((output.parent.name, output.name, list(train), list(validation)))
        meta = {"checkpoint": str(output), "train_indices": list(train), "validation_indices": list(validation),
                "best_epoch": 2, "epochs_completed": 2, "seed": 17}
        runner.write(output / "metadata.json", meta)
        (output / "heads.pt").write_bytes(b"synthetic-small-head-weights")
        return meta

    def fake_predict(d, indices, checkpoint, *, variant="original", **kwargs):
        path = Path(checkpoint)
        if fail_final and path.parent.name == "fold4" and path.name == "outer" and variant == "missing_evidence":
            raise RuntimeError("Synthetic final-fold interruption")
        c = np.asarray([int(row["candidate_id"][1:]) for row in d["candidate_records"]])
        v = d["variant_names"].index(variant)
        base = (np.asarray(indices)[:, None] * 10 + c[None, :] + 100 * v).astype(np.float32)
        return {"scores": base, "base_scores": base, "probabilities": np.ones((*base.shape, 3)) / 3,
                "selected_predicted_error": np.ones_like(base), "error_supported": np.ones(len(indices), bool),
                "predicted_errors": {"Cmax": np.ones_like(base)}}

    def fake_selection(d, indices, prediction):
        matching = [f for f in folds if set(indices) == set(f["outer_train"])]
        assert len(matching) == 1
        assert not set(indices) & set(matching[0]["outer_test"])
        assert prediction["scores"].shape == (len(indices), candidate_count)
        selections.append(indices)
        return {"weight": .1, "reason": "synthetic test only"}

    monkeypatch.setattr(training, "train_fold", fake_fit)
    monkeypatch.setattr(training, "predict_fold", fake_predict)
    monkeypatch.setattr(runner, "select_error_weight", fake_selection)
    monkeypatch.setattr(evaluation, "fit_oof_temperature", lambda *args: {"temperature": 1., "state": "synthetic-test"})
    monkeypatch.setattr(runner, "status", lambda phase, **kwargs: events.append((phase, kwargs)))
    return data, fits, selections, events


def test_all_twenty_fits_cover_exact_outer_membership_and_resume_without_refitting(tmp_path, monkeypatch):
    data, fits, selections, events = configured(tmp_path, monkeypatch)
    runner.train(data, "cpu")
    assert len(fits) == 20
    assert len(selections) == 5
    with np.load(runner.OUT / "predictions/seed17.npz", allow_pickle=False) as artifact:
        assert artifact["scores"].shape == (3, 10, 3)
        assert artifact["probabilities"].shape == (3, 10, 3, 3)
        np.testing.assert_array_equal(artifact["indices"], np.arange(10))
        for v in range(3):
            expected = np.arange(10)[:, None] * 10 + np.arange(3)[None, :] + 100 * v
            np.testing.assert_allclose(artifact["base_scores"][v], expected)
            np.testing.assert_allclose(artifact["scores"][v], expected - .1)
    receipt = runner.read(runner.OUT / "predictions/seed17.json")
    assert len(receipt["fit_receipts"]) == 20
    assert set(receipt["outer_receipts"]) == {str(i) for i in range(5)}
    assert events[-1] == ("training_complete", {"completed_jobs": 20, "total_jobs": 20})
    runner.train(data, "cpu")
    assert len(fits) == 20  # Every completed checkpoint/prediction was verified and reused.


def test_interruption_cannot_emit_final_complete_receipt(tmp_path, monkeypatch):
    data, fits, _, events = configured(tmp_path, monkeypatch, fail_final=True)
    with pytest.raises(RuntimeError, match="final-fold interruption"):
        runner.train(data, "cpu")
    assert len(fits) == 20  # Fit completion alone is not experiment completion.
    assert not (runner.OUT / "predictions/seed17.json").exists()
    assert not (runner.OUT / "predictions/seed17.npz").exists()
    assert not (runner.OUT / "runs/seed17/fold4/outer/complete.json").exists()
    assert not any(phase == "training_complete" for phase, _ in events)


def test_changed_prediction_receipt_prevents_silent_resume(tmp_path, monkeypatch):
    data, fits, _, _ = configured(tmp_path, monkeypatch)
    runner.train(data, "cpu")
    target = runner.OUT / "runs/seed17/fold0/inner0/validation.npz"
    target.write_bytes(target.read_bytes() + b"unexpected mutation")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.train(data, "cpu")
    assert len(fits) == 20
