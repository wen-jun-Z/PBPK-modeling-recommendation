"""New-request inference uses CPU mocks; no GPU or external model is loaded."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pbpk_kb.recommendation.pbpk_match import serving
from pbpk_kb.recommendation.pbpk_match.data import FIELDS, TASKS, TYPES, node_specs
from pbpk_kb.recommendation.pbpk_match.model import PBPKMatch
from scripts import predict_pbpk_match as existing
from scripts import recommend_pbpk_match as cli


NEW_REQUEST = {"task_type": "plasma_concentration_time_prediction", "compounds": ["new_molecule"],
               "features": {"species": ["human"], "routes": ["oral"], "endpoints": ["auc"]}}


@pytest.fixture
def new_request_checkpoint(tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    root = tmp_path / "pbpk_match"
    fold = root / "runs/demo/full_seed17/fold0"
    fold.mkdir(parents=True)
    cache = root / "cache"
    cache.mkdir()
    snapshot = {"development_query_ids": ["q0", "q1"], "test_query_ids_excluded": ["locked_q"],
                "labels": [{"grade": 3, "documented_applicability": True}]}
    snapshot_path = root / "development_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))
    snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    np.savez_compressed(cache / "frozen_embeddings.npz", embeddings=np.zeros((2, 8), dtype=np.float16))
    manifest = {"repo": "Qwen/Qwen3-Embedding-0.6B", "revision": "1" * 40, "dimension": 8,
                "array_sha256": hashlib.sha256((cache / "frozen_embeddings.npz").read_bytes()).hexdigest()}
    (cache / "embedding_manifest.json").write_text(json.dumps(manifest))
    rng = np.random.default_rng(17)
    data = {
        "q_embeddings": rng.normal(size=(3, 2, 2, 8)).astype(np.float32),
        "q_types": np.zeros((3, 2, 2), np.int64), "q_mask": np.ones((3, 2, 2), bool),
        "c_embeddings": rng.normal(size=(280, 3, 8)).astype(np.float32),
        "c_types": np.tile([0, TYPES.index("compounds"), TYPES.index("routes")], (280, 1)),
        "c_mask": np.ones((280, 3), bool), "pair_features": rng.normal(size=(3, 2, 280, 61)).astype(np.float32),
        "task_ids": np.asarray([0, 1]), "num_types": len(TYPES), "num_tasks": len(TASKS),
        "variant_names": ["original", "reordered", "missing_evidence"],
        "grades": np.full((2, 280), 999), "applicability": np.full((2, 280), 999),
        "snapshot": snapshot, "embedding_manifest": manifest,
        "query_records": [{"query_id": f"q{i}", "compound_id": f"drug{i}", "compounds": [f"drug{i}"],
                           "task_type": TASKS[0], "features": {}, "provenance": {"grade": 3}} for i in range(2)],
        "candidate_records": [{"candidate_id": f"c{i:03}", "family_id": f"f{i}", "family_name": f"model{i}",
                               "compounds": ["old_molecule"], "features": {"routes": ["intravenous"]},
                               "provenance": {"grade": 3}, "outcome": 1} for i in range(280)],
    }
    config = {"hidden_dim": 8, "dropout": 0.}
    model = PBPKMatch(8, 61, len(TYPES), len(TASKS), **config)
    payload = {"state_dict": model.state_dict(), "model_kind": "full", "snapshot_sha256": snapshot_hash,
               "embedding_dim": 8, "pair_feature_dim": 61, "num_types": len(TYPES), "num_tasks": len(TASKS), "config": config}
    checkpoint = fold / "model.pt"
    torch.save(payload, checkpoint)
    protocol = {"methods": ["full"], "seeds": [17], "snapshot_sha256": snapshot_hash, "config": config,
                "source_sha256": {name: hashlib.sha256((existing.ROOT / name).read_bytes()).hexdigest() for name in existing.INFERENCE_SOURCE_FILES}}
    (fold.parent.parent / "protocol.json").write_text(json.dumps(protocol))
    report = {"method": "full", "seed": 17, "partition": {"calibration": [1], "refit": [0], "outer": []},
              "calibrator": {"state": "unavailable"}}
    (fold / "report.json").write_text(json.dumps(report))
    original_scoped = serving.scoped
    def local(path):
        resolved = serving.Path(path).resolve()
        return resolved if resolved.is_relative_to(tmp_path.resolve()) else original_scoped(path)
    def digest(path):
        return hashlib.sha256(local(path).read_bytes()).hexdigest()
    for module in (serving, existing, cli):
        monkeypatch.setattr(module, "scoped", local)
        monkeypatch.setattr(module, "sha", digest)
    monkeypatch.setattr(existing, "_load_cached_data", lambda output_root: data)
    monkeypatch.setattr(serving, "_verify_encoder_registry", lambda *a, **kw: {**manifest, "path": str(root), "registry_sha256": "mock"})
    def encoder(texts, frozen_manifest, **kwargs):
        assert frozen_manifest == manifest
        rows = [np.asarray(list(hashlib.sha256(text.encode()).digest())[:8], dtype=np.float32) + 1 for text in texts]
        vectors = np.stack([row / np.linalg.norm(row) for row in rows])
        return vectors, {"repo": manifest["repo"], "revision": manifest["revision"], "encoding": serving.ENCODING, "test_cpu_mock": True}
    yield checkpoint, data, encoder, payload
    torch.set_num_threads(old_threads)


def test_new_request_outside_frozen_ids_returns_all_280_without_target_access(new_request_checkpoint):
    checkpoint, data, encoder, _ = new_request_checkpoint
    first = serving.recommend_request(checkpoint, NEW_REQUEST, device="cpu", encoder=encoder)
    assert first["candidate_count"] == len(first["ranking"]) == 280
    assert first["query_id"].startswith("structured-request:")
    assert first["request"]["compounds"] == ["new_molecule"]
    assert first["query_compounds_not_exactly_named_in_catalog"] == ["new_molecule"]
    assert first["scientific_gold"] is False and first["pk_simulation_executed"] is False
    assert first["version"] == "identity-policy-v1"
    assert first["abstain"] is True and first["status"] == "needs_catalog_review"
    assert first["top_k_with_boundary_ties"] == []
    assert first["policy_eligible_candidate_count"] == 0
    assert len(first["raw_top_k"]) >= 10
    before = [r["score"] for r in first["ranking"]]
    data["grades"][:] = -900
    data["applicability"][:] = -800
    data["snapshot"]["labels"] = [{"grade": -999}]
    second = serving.recommend_request(checkpoint, NEW_REQUEST, device="cpu", encoder=encoder)
    assert before == [r["score"] for r in second["ranking"]]
    assert first["probability_calibration"] == "uncalibrated_development_probabilities"
    json.dumps(first, allow_nan=False)


def test_new_request_feature_contract_and_boundary_ties(new_request_checkpoint, monkeypatch):
    checkpoint, raw_data, encoder, _ = new_request_checkpoint
    for candidate in raw_data["candidate_records"][:4]:
        candidate["compounds"] = ["new_molecule"]
    def predictor(model, clean, indices, **kwargs):
        assert indices == [0] and clean["variant_names"] == ["original"]
        assert not {"grades", "applicability", "labels", "snapshot"} & set(clean)
        assert all("provenance" not in row and "outcome" not in row for row in clean["candidate_records"])
        assert clean["pair_features"].shape == (1, 1, 280, 61)
        # Explicit oral and IV strings are listed disjoint, not source-proven conflict.
        pos = FIELDS.index("routes") * 5
        np.testing.assert_array_equal(clean["pair_features"][0, 0, 0, pos:pos + 5], [1, 1, 0, 0, 1])
        scores = np.zeros((1, 280)); scores[0, :3] = 1
        return scores, np.tile([.7, .2, .1], (1, 280, 1))
    monkeypatch.setattr(serving, "predict_neural", predictor)
    result = serving.recommend_request(checkpoint, NEW_REQUEST, 1, device="cpu", encoder=encoder)
    assert result["top_k_with_boundary_ties_count"] == 3
    assert all(r["rank_min"] == 1 and r["rank_max"] == 3 for r in result["top_k_with_boundary_ties"])
    assert all(r["policy_eligible"] and r["policy_rank"] == 1 and r["raw_rank"] == 1 for r in result["top_k_with_boundary_ties"])
    assert all(r["status"] == "needs_review" and r["scientific_applicability_verified"] is False for r in result["top_k_with_boundary_ties"])
    assert all(r["status"] == "uncertain" for r in result["ranking"])


def test_direct_request_excludes_wrong_drug_but_preserves_raw_scores_and_negative_app_signal(new_request_checkpoint, monkeypatch):
    checkpoint, data, encoder, _ = new_request_checkpoint
    data["candidate_records"][0]["compounds"] = ["isoniazid"]
    data["candidate_records"][1]["compounds"] = ["metformin"]
    def predictor(*args, **kwargs):
        scores = np.zeros((1, 280)); scores[0, 0] = 100.; scores[0, 1] = .5
        return scores, np.tile([.02, .93, .05], (1, 280, 1))
    monkeypatch.setattr(serving, "predict_neural", predictor)
    request = {**NEW_REQUEST, "compounds": ["metformin"]}
    result = serving.recommend_request(checkpoint, request, top_k=1, device="cpu", encoder=encoder)
    assert result["raw_top_k"][0]["candidate_id"] == "c000"
    assert result["ranking"][0]["score"] == 100.
    assert result["ranking"][0]["rank_min"] == 1
    assert result["ranking"][0]["policy_eligible"] is False
    selected = result["top_k_with_boundary_ties"][0]
    assert selected["candidate_id"] == "c001"
    assert selected["score"] == .5 and selected["raw_rank"] == 2 and selected["policy_rank"] == 1
    assert selected["recommendation_status"] == "retrieval_candidate_needs_review"
    assert selected["model_applicability_y_leading"] is False
    assert selected["scientific_applicability_verified"] is False
    assert result["abstain"] is False


def test_template_request_keeps_cross_drug_candidates(new_request_checkpoint, monkeypatch):
    checkpoint, _, encoder, _ = new_request_checkpoint
    def predictor(*args, **kwargs):
        scores = np.zeros((1, 280)); scores[0, 0] = 1.
        return scores, np.tile([.1, .2, .7], (1, 280, 1))
    monkeypatch.setattr(serving, "predict_neural", predictor)
    request = {**NEW_REQUEST, "task_type": "mechanistic_template_recommendation"}
    result = serving.recommend_request(checkpoint, request, top_k=1, device="cpu", encoder=encoder)
    assert result["policy_eligible_candidate_count"] == 280
    assert result["top_k_with_boundary_ties"][0]["candidate_id"] == "c000"
    assert result["abstain"] is False


@pytest.mark.parametrize("extra", [{"grade": 3}, {"query_id": "locked_q"}, {"provenance": {}}, {"text": "raw prompt"}])
def test_outcome_and_unstructured_request_fields_rejected(extra):
    with pytest.raises(ValueError, match="only task_type"):
        serving.validate_request({**NEW_REQUEST, **extra})


def test_nested_outcomes_bad_shapes_and_conflicting_compound_fields_rejected():
    for request in ({**NEW_REQUEST, "features": {"grade": ["3"]}}, {**NEW_REQUEST, "compounds": "drug"},
                    {**NEW_REQUEST, "features": {"routes": "oral"}}, {**NEW_REQUEST, "task_type": "unseen_task"},
                    {**NEW_REQUEST, "features": {"compounds": ["different_drug"]}}):
        with pytest.raises(ValueError):
            serving.validate_request(request)


def test_fitted_fold_calibration_applied_and_overlap_rejected(new_request_checkpoint):
    checkpoint, _, encoder, _ = new_request_checkpoint
    path = checkpoint.parent / "report.json"
    report = json.loads(path.read_text())
    report["calibrator"] = {"state": "fitted", "temperature": 2., "fit_compound_ids": ["drug1"],
                            "model_fit_compound_ids": ["drug0"], "excluded_compound_ids": []}
    path.write_text(json.dumps(report))
    result = serving.recommend_request(checkpoint, NEW_REQUEST, device="cpu", encoder=encoder)
    assert result["probability_calibration"] == "checkpoint_fold_independent_calibration_applied"
    report["calibrator"]["model_fit_compound_ids"] = ["drug1"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not held out"):
        serving.load_checkpoint(checkpoint)


def test_checkpoint_and_embedding_bindings_are_enforced(new_request_checkpoint):
    checkpoint, _, _, payload = new_request_checkpoint
    payload["embedding_dim"] = 9
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="dimensions"):
        serving.load_checkpoint(checkpoint)


def test_cache_manifest_mismatch_rejected_before_encoding(new_request_checkpoint):
    checkpoint, _, _, _ = new_request_checkpoint
    root = checkpoint.parents[4]
    path = root / "cache/embedding_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["revision"] = "2" * 40
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="embedding manifest/cache"):
        serving.load_checkpoint(checkpoint)


def test_directory_gap_explains_unlisted_mechanisms_without_claiming_conflict():
    request = {**NEW_REQUEST, "features": {"enzymes": ["CYP3A4"]}}
    candidate = {"candidate_id": "c", "compounds": ["new_molecule"], "features": {}}
    result = serving._catalog_field_gaps(request, candidate)
    enzyme = [row for row in result["fields"] if row["field"] == "enzymes"][0]
    assert enzyme["state"] == "not_recorded_in_catalog"
    assert "incompatibility" in result["interpretation"]


def test_new_query_encoder_revision_mismatch_rejected(new_request_checkpoint):
    checkpoint, _, encoder, _ = new_request_checkpoint
    def wrong_encoder(*args, **kwargs):
        vectors, metadata = encoder(*args, **kwargs)
        return vectors, {**metadata, "revision": "2" * 40}
    with pytest.raises(ValueError, match="encoder differs"):
        serving.recommend_request(checkpoint, NEW_REQUEST, device="cpu", encoder=wrong_encoder)


def test_cli_writes_full_result_and_preserves_existing_file(new_request_checkpoint, monkeypatch, capsys):
    checkpoint, _, encoder, _ = new_request_checkpoint
    request = checkpoint.parent / "request.json"
    request.write_text(json.dumps(NEW_REQUEST))
    destination = checkpoint.parent / "new_result.json"
    monkeypatch.setattr(cli, "recommend_request", lambda *args, **kwargs: serving.recommend_request(*args, **{**kwargs, "device": "cpu", "encoder": encoder}))
    cli.main(["--checkpoint", str(checkpoint), "--request-json", str(request), "--output", str(destination)])
    result = json.loads(destination.read_text())
    assert len(result["ranking"]) == 280 and result["request_file_sha256"]
    preview = json.loads(capsys.readouterr().out)
    assert preview["candidate_count"] == 280
    assert preview["policy_eligible_candidate_count"] == 0
    assert preview["abstain"] is True and preview["version"] == "identity-policy-v1"
    with pytest.raises(FileExistsError):
        cli.main(["--checkpoint", str(checkpoint), "--request-json", str(request), "--output", str(destination)])


def test_gpu_guard_does_not_allocate_when_less_than_five_gib_free(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a: pytest.fail("must not allocate"))
    monkeypatch.setattr(serving, "_verify_encoder_registry", lambda *a, **kw: {})
    with pytest.raises(RuntimeError, match="Less than 5 GiB"):
        serving.encode_new_query(["node"], {}, device="cuda:0")


def test_encoder_pooling_and_options_without_gpu(monkeypatch):
    import transformers

    calls = {}
    for name, function in {"is_available": lambda: True, "mem_get_info": lambda device: (8 * 1024**3, 24 * 1024**3),
                           "set_per_process_memory_fraction": lambda fraction, device: calls.update(cap=fraction),
                           "reset_peak_memory_stats": lambda device: None, "empty_cache": lambda: None,
                           "max_memory_allocated": lambda device: 1024}.items():
        monkeypatch.setattr(torch.cuda, name, function)
    monkeypatch.setattr(serving, "_verify_encoder_registry", lambda *a, **kw: {
        "repo": "Qwen/Qwen3-Embedding-0.6B", "revision": "1" * 40, "path": "mock", "registry_sha256": "mock"})
    class Batch(dict):
        def to(self, device):
            assert device == "cuda:0"
            return self
    class Tokenizer:
        padding_side = "left"
        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            if not kwargs.get("return_tensors"):
                return {"input_ids": [[1, 2] for _ in texts]}
            return Batch(input_ids=torch.ones((len(texts), 2), dtype=torch.long), attention_mask=torch.ones((len(texts), 2), dtype=torch.long))
    class Model:
        def to(self, device):
            assert device == "cuda:0"
            return self
        def eval(self):
            return self
        def __call__(self, input_ids, attention_mask, use_cache):
            assert use_cache is False
            values = torch.arange(1., 9.).expand(len(input_ids), 2, 8).clone()
            values[:, 0] = 999  # First token must never be pooled.
            return SimpleNamespace(last_hidden_state=values)
    def tokenizer_load(path, **kwargs):
        assert kwargs == dict(padding_side="left", local_files_only=True, token=False, trust_remote_code=False)
        return Tokenizer()
    def model_load(path, **kwargs):
        assert kwargs["local_files_only"] is True and kwargs["token"] is False and kwargs["trust_remote_code"] is False
        assert kwargs["torch_dtype"] == torch.bfloat16
        return Model()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_load)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", model_load)
    values, metadata = serving.encode_new_query(["first", "second"], {}, device="cuda:0")
    expected = np.arange(1., 9., dtype=np.float32)
    expected = (expected / np.linalg.norm(expected)).astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(values, np.stack([expected, expected]))
    assert metadata["truncated_inputs"] == 0 and metadata["gpu_memory_cap_gib"] == 5
    assert calls["cap"] == pytest.approx(5 / 24)
