"""Reject altered nested artifacts and invalid holdout identity before reuse."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import reviewer_capacity_run as runner


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUT", tmp_path / "capacity")
    monkeypatch.setattr(runner, "LIGHT", tmp_path / "original")

    def reference(path):
        path = Path(path).resolve()
        assert path.is_relative_to(tmp_path)
        return {"path": str(path.relative_to(tmp_path)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    monkeypatch.setattr(runner, "reference", reference)
    _write(runner.OUT / "protocol.json", {"frozen": "test"})
    return runner


def test_completed_prediction_checks_nested_weights(sandbox):
    out = sandbox.OUT
    weight = out / "fit/heads.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"original learned weights")
    complete = out / "fit/complete.json"
    _write(complete, {"state": "complete", "weights": sandbox.reference(weight)})
    prediction = out / "predictions.npz"
    np.savez(prediction, scores=np.zeros((3, 2, 2)))
    receipt = out / "predictions.json"
    _write(receipt, {"state": "complete", "protocol": sandbox.reference(out / "protocol.json"),
                     "predictions": sandbox.reference(prediction), "fits": [sandbox.reference(complete)]})
    assert sandbox.receipt_valid(receipt)
    weight.write_bytes(b"changed but same complete.json")
    with pytest.raises(ValueError):
        sandbox.receipt_valid(receipt)


@pytest.mark.parametrize("indices", ([0, 0], [1]))
def test_fold_publication_rejects_duplicate_or_wrong_holdout_identity(sandbox, indices):
    method, seed = "concat_capacity", 17
    for fold in range(5):
        folder = sandbox.OUT / method / f"runs/seed{seed}/fold{fold}"
        for kind in ("inner0", "inner1", "inner2", "outer"):
            _write(folder / kind / "complete.json", {"state": "complete"})
        chosen = indices if fold == 0 else ([0] if fold == 1 and indices == [1] else [fold])
        shape = (3, len(chosen), 2)
        np.savez(folder / "outer/predictions.npz", indices=np.asarray(chosen), scores=np.zeros(shape),
                 probabilities=np.full((*shape, 3), 1 / 3))
    data = {"query_records": [{}] * 5, "candidate_records": [{}] * 2,
            "variant_names": ["original", "reordered", "missing_evidence"],
            "folds": [{"outer_test": [i]} for i in range(5)]}
    with pytest.raises(ValueError):
        sandbox.publish_predictions(data, method, seed)


def original_fixture(sandbox):
    data = {"query_records": [{}] * 5, "candidate_records": [{}] * 2,
            "variant_names": ["original", "reordered", "missing_evidence"],
            "groups": list(range(5)), "folds": [{"outer_test": [i]} for i in range(5)]}
    _write(sandbox.LIGHT / "protocol.json", {"groups": data["groups"], "folds": data["folds"]})
    fit_refs, outer_refs = [], {}
    for f in range(5):
        for kind in ("inner0", "inner1", "inner2", "outer"):
            folder = sandbox.LIGHT / f"runs/seed17/fold{f}/{kind}"
            _write(folder / "metadata.json", {"seed": 17})
            (folder / "heads.pt").write_bytes(b"unchanged original weights")
            _write(folder / "complete.json", {"state": "complete", "artifacts_sha256": {
                name: sandbox.reference(folder / name)["sha256"] for name in ("metadata.json", "heads.pt")}})
            fit_refs.append(sandbox.reference(folder / "complete.json"))
            if kind == "outer":
                outer_refs[str(f)] = fit_refs[-1]
    source = sandbox.LIGHT / "predictions/seed17.npz"
    source.parent.mkdir(parents=True)
    base = np.arange(30, dtype=np.float32).reshape(3, 5, 2)
    np.savez(source, indices=np.arange(5), scores=base - 100, base_scores=base,
             probabilities=np.full((3, 5, 2, 3), 1 / 3))
    _write(source.with_suffix(".json"), {"state": "complete", "score_sha256": sandbox.reference(source)["sha256"],
        "protocol_sha256": sandbox.reference(sandbox.LIGHT / "protocol.json")["sha256"],
        "fit_receipts": fit_refs, "outer_receipts": outer_refs})
    return data, source, base


def test_seed17_reuses_only_document_scores_without_modifying_original(sandbox):
    data, source, base = original_fixture(sandbox)
    before = sandbox.reference(source)
    sandbox.reuse_seed17(data)
    with np.load(sandbox.OUT / "pbpk_interaction/predictions/seed17.npz") as arrays:
        np.testing.assert_array_equal(arrays["scores"], base)
    assert sandbox.reference(source) == before


def test_seed17_reuse_checks_original_receipt(sandbox):
    data, source, _ = original_fixture(sandbox)
    receipt = json.loads(source.with_suffix(".json").read_text())
    receipt["score_sha256"] = "0" * 64
    _write(source.with_suffix(".json"), receipt)
    with pytest.raises(ValueError):
        sandbox.reuse_seed17(data)


def test_seed17_reuse_checks_original_nested_weight(sandbox):
    data, _, _ = original_fixture(sandbox)
    (sandbox.LIGHT / "runs/seed17/fold0/inner0/heads.pt").write_bytes(b"modified original weights")
    with pytest.raises(ValueError):
        sandbox.reuse_seed17(data)

