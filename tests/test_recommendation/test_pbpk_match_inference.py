"""CPU checkpoint reconstruction and outcome-free inference contract."""
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts import predict_pbpk_match as inference
from pbpk_kb.recommendation.pbpk_match.model import PBPKMatch


@pytest.fixture
def frozen_demo(tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    root = tmp_path / "pbpk_match"
    fold = root / "runs" / "demo" / "full_seed17" / "fold0"
    fold.mkdir(parents=True)
    snapshot = {"development_query_ids": ["q0", "q1"], "test_query_ids_excluded": ["locked_q"],
                "labels": [{"query_id": "q0", "grade": 3, "documented_applicability": True}]}
    snapshot_path = root / "development_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))
    snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    rng = np.random.default_rng(17)
    data = {
        "q_embeddings": rng.normal(size=(3, 2, 2, 8)).astype(np.float32),
        "q_types": np.zeros((3, 2, 2), np.int64), "q_mask": np.ones((3, 2, 2), bool),
        "c_embeddings": rng.normal(size=(280, 2, 8)).astype(np.float32),
        "c_types": np.zeros((280, 2), np.int64), "c_mask": np.ones((280, 2), bool),
        "pair_features": rng.normal(size=(3, 2, 280, 5)).astype(np.float32),
        "task_ids": np.array([0, 1]), "num_types": 3, "num_tasks": 2,
        "variant_names": ["original", "reordered", "missing_evidence"],
        "grades": np.zeros((2, 280)), "applicability": np.zeros((2, 280)), "snapshot": snapshot,
        "query_records": [{"query_id": f"q{i}", "compound_id": f"drug{i}", "compounds": [f"drug{i}"],
                           "task_type": "plasma_concentration_time_prediction", "features": {"routes": ["oral"], "endpoints": ["auc"]},
                           "provenance": {"secret_judge_label": 3}, "legacy_split": "train"} for i in range(2)],
        "candidate_records": [{"candidate_id": f"c{i:03}", "family_id": f"f{i}", "family_name": f"Model {i}",
                               "compounds": ["drug0"], "features": {"routes": ["oral"]},
                               "provenance": {"outcome": "hidden"}} for i in range(280)],
    }
    config = {"hidden_dim": 8, "dropout": 0.0}
    model = PBPKMatch(8, 5, 3, 2, **config).eval()
    payload = {"state_dict": model.state_dict(), "model_kind": "full", "config": config,
               "embedding_dim": 8, "pair_feature_dim": 5, "num_types": 3, "num_tasks": 2,
               "snapshot_sha256": snapshot_hash}
    checkpoint = fold / "model.pt"
    torch.save(payload, checkpoint)
    protocol = {"methods": ["full"], "seeds": [17], "config": config, "snapshot_sha256": snapshot_hash,
                "source_sha256": {name: hashlib.sha256((inference.ROOT / name).read_bytes()).hexdigest() for name in inference.INFERENCE_SOURCE_FILES}}
    (fold.parent.parent / "protocol.json").write_text(json.dumps(protocol))
    report = {"method": "full", "seed": 17, "partition": {"outer": [0], "calibration": [1], "refit": []},
              "calibrator": {"state": "unavailable"}}
    (fold / "report.json").write_text(json.dumps(report))
    # Checkpoint fixtures are beneath --basetemp in the project during our run.
    # Test isolation permits pytest's arbitrary temp root, never production paths.
    original_scoped = inference.scoped
    def allow_fixture(path):
        path = inference.Path(path).resolve()
        return path if path.is_relative_to(tmp_path.resolve()) else original_scoped(path)
    monkeypatch.setattr(inference, "scoped", allow_fixture)
    monkeypatch.setattr(inference, "sha", lambda path: hashlib.sha256(allow_fixture(path).read_bytes()).hexdigest())
    monkeypatch.setattr(inference, "_load_cached_data", lambda output_root: data)
    yield checkpoint, data, payload
    torch.set_num_threads(old_threads)


def test_real_checkpoint_cpu_predicts_all_candidates_without_targets(frozen_demo):
    checkpoint, data, _ = frozen_demo
    result = inference.predict_checkpoint(checkpoint, "q0", 10)
    assert len(result["ranking"]) == 280
    assert result["candidate_count"] == 280
    assert result["device"] == "cpu"
    assert result["query_role_in_checkpoint"] == "outer"
    assert result["status"] == "uncertain"
    assert result["pk_simulation_executed"] is False
    assert all(row["status"] == "uncertain" for row in result["ranking"])
    assert all(sum(row["probabilities"].values()) == pytest.approx(1) for row in result["ranking"])
    assert all(any(gap["field"] == "endpoints" for gap in row["field_gaps"]["fields"]) for row in result["ranking"])
    before = [(row["candidate_id"], row["score"]) for row in result["ranking"]]
    data["grades"][:] = 999
    data["applicability"][:] = -999
    data["snapshot"]["labels"] = [{"grade": -999}]
    changed = inference.predict_checkpoint(checkpoint, "q0", 10)
    assert before == [(row["candidate_id"], row["score"]) for row in changed["ranking"]]
    json.dumps(result, allow_nan=False)


def test_boundary_ties_retained_and_predictor_sees_no_label_keys(frozen_demo, monkeypatch):
    checkpoint, _, _ = frozen_demo
    def fake_predict(model, clean, indices, device):
        assert device == "cpu"
        assert not {"grades", "applicability", "snapshot", "labels"} & set(clean)
        assert all("provenance" not in row and "legacy_split" not in row for row in clean["query_records"])
        assert all("provenance" not in row for row in clean["candidate_records"])
        scores = np.zeros((1, 280))
        scores[0, :3] = 1
        return scores, np.tile([.99, .005, .005], (1, 280, 1))
    monkeypatch.setattr(inference, "predict_neural", fake_predict)
    result = inference.predict_checkpoint(checkpoint, "q0", 1)
    assert result["top_k_with_boundary_ties_count"] == 3
    assert all(row["rank_min"] == 1 and row["rank_max"] == 3 for row in result["top_k_with_boundary_ties"])
    assert all(row["status"] == "uncertain" for row in result["top_k_with_boundary_ties"])


def test_locked_query_rejected_before_predictor(frozen_demo, monkeypatch):
    checkpoint, _, _ = frozen_demo
    monkeypatch.setattr(inference, "predict_neural", lambda *a, **k: pytest.fail("must not predict locked test query"))
    with pytest.raises(ValueError, match="Only frozen development"):
        inference.predict_checkpoint(checkpoint, "locked_q")


@pytest.mark.parametrize("field,value", [("snapshot_sha256", "bad"), ("embedding_dim", 999), ("model_kind", "tree")])
def test_checkpoint_bindings_and_dimensions_checked(frozen_demo, field, value):
    checkpoint, _, payload = frozen_demo
    payload[field] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        inference.predict_checkpoint(checkpoint, "q0")


def test_protocol_source_mismatch_rejected(frozen_demo):
    checkpoint, _, _ = frozen_demo
    protocol_path = checkpoint.parent.parent.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["source_sha256"][inference.INFERENCE_SOURCE_FILES[0]] = "bad"
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Frozen inference dependency"):
        inference.predict_checkpoint(checkpoint, "q0")


def test_training_module_update_is_disclosed_but_architecture_still_restores(frozen_demo):
    checkpoint, _, _ = frozen_demo
    protocol_path = checkpoint.parent.parent.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["source_sha256"][inference.INFERENCE_SOURCE_FILES[2]] = "historical-training-implementation"
    protocol_path.write_text(json.dumps(protocol))
    result = inference.predict_checkpoint(checkpoint, "q0")
    assert result["source_verification"]["predictor_module_changed"] is True
    assert result["source_verification"]["warning"]
    assert len(result["ranking"]) == 280


def test_cli_writes_full_ranking_and_refuses_existing_file(frozen_demo, capsys):
    checkpoint, _, _ = frozen_demo
    destination = checkpoint.parent / "demo.json"
    inference.main(["--checkpoint", str(checkpoint), "--query-id", "q0", "--output", str(destination)])
    stored = json.loads(destination.read_text())
    printed = json.loads(capsys.readouterr().out)
    assert len(stored["ranking"]) == 280
    assert len(printed["preview"]) <= 10
    with pytest.raises(FileExistsError):
        inference.main(["--checkpoint", str(checkpoint), "--query-id", "q0", "--output", str(destination)])


def test_missing_cache_never_starts_embedding(tmp_path, monkeypatch):
    original_loader = inference._load_cached_data
    monkeypatch.setattr(inference, "scoped", lambda path: inference.Path(path))
    monkeypatch.setattr(inference, "build_data", lambda **kwargs: pytest.fail("must not build missing cache"))
    with pytest.raises(FileNotFoundError, match="inference will not build"):
        original_loader(tmp_path)
