"""Leakage, restart, condition-sensitivity and full-catalogue inference checks."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import pbpk_match_light_training as light


def dataset():
    rng = np.random.default_rng(81)
    q, c, f, d = 7, 5, 4, 8
    return {"query_records": [{"query_id": f"q{i}", "compound_id": f"drug{i}", "compounds": [f"drug{i}"],
                               "features": {"endpoints": ["Cmax"]}} for i in range(q)],
            "candidate_records": [{"candidate_id": f"c{i}", "family_id": f"family{i // 2}", "compounds": ["drug0"]} for i in range(c)],
            "light_query_embeddings": rng.normal(size=(3, q, d)).astype(np.float32),
            "light_candidate_embeddings": rng.normal(size=(c, d)).astype(np.float32),
            "light_embedding_manifest": {"revision": "fixed-test-encoder", "cache_sha256": "synthetic"},
            "pair_features_by_variant": rng.normal(size=(3, q, c, f)).astype(np.float32),
            "pair_feature_names": [f"field{i}" for i in range(f)],
            "variant_names": ["original", "reordered", "missing_evidence"],
            "grades": np.tile([3, 2, 1, 0, -1], (q, 1)),
            "applicability": np.tile([0, 0, 2, 1, -1], (q, 1)),
            "groups": np.arange(q), "error_records": []}


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def fit(data, tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(light, "ROOT", tmp_path)
    return light.train_fold(data, [0, 1, 2], [], tmp_path / "fit", epochs=2, pk_epochs=2, **kwargs)


def test_inference_needs_no_outcomes_and_preserves_full_catalogue(tmp_path, monkeypatch):
    data = dataset()
    data["grades"][6] = 99; data["applicability"][6] = 99
    metadata = fit(data, tmp_path, monkeypatch)
    assert metadata["trainable_text_encoder"] is False
    assert metadata["best_epoch"] == 2
    del data["grades"], data["applicability"], data["error_records"]
    result = light.predict_fold(data, [6], metadata["checkpoint"], return_representations=True)
    assert result["scores"].shape == (1, 5)
    assert result["representations"].shape == (1, 5, 132)
    np.testing.assert_allclose(result["probabilities"].sum(-1), 1., atol=1e-6)
    assert not result["error_supported"].any()


def test_grouped_leakage_rejected_before_fitting(tmp_path, monkeypatch):
    data = dataset(); data["groups"][3] = data["groups"][0]
    monkeypatch.setattr(light, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="grouped"):
        light.train_fold(data, [0, 1, 2], [3], tmp_path / "fit")
    with pytest.raises(ValueError, match="belong"):
        light.train_fold(data, [0, 1, 2], [], tmp_path / "fit", error_train_indices=[4])


def test_exact_duplicate_ties_candidate_permutation_and_batch_size(tmp_path, monkeypatch):
    data = dataset()
    data["light_candidate_embeddings"][4] = data["light_candidate_embeddings"][3]
    data["pair_features_by_variant"][:, :, 4] = data["pair_features_by_variant"][:, :, 3]
    metadata = fit(data, tmp_path, monkeypatch)
    small = light.predict_fold(data, [5], metadata["checkpoint"], batch_size=1)
    large = light.predict_fold(data, [5], metadata["checkpoint"], batch_size=512)
    np.testing.assert_array_equal(small["scores"], large["scores"])
    assert small["scores"][0, 3] == small["scores"][0, 4]
    order = np.array([4, 1, 3, 0, 2])
    altered = copy.deepcopy(data)
    altered["candidate_records"] = [data["candidate_records"][i] for i in order]
    altered["light_candidate_embeddings"] = data["light_candidate_embeddings"][order]
    altered["pair_features_by_variant"] = data["pair_features_by_variant"][:, :, order]
    permuted = light.predict_fold(altered, [5], metadata["checkpoint"])
    np.testing.assert_array_equal(small["scores"], permuted["scores"][:, np.argsort(order)])


def test_error_gating_uses_train_studies_and_request_endpoint_only(tmp_path, monkeypatch):
    data = dataset()
    for qi in [0, 1, 2]:
        for ci in range(5):
            data["error_records"].append({"query_index": qi, "candidate_index": ci,
                                          "endpoint": "Cmax_reference_vs_group", "study_id": f"study{qi}",
                                          "absolute_log_error": .1 + ci / 3})
    data["error_records"].append({"query_index": 6, "candidate_index": 0, "endpoint": "Cmax_reference_vs_group",
                                  "study_id": "heldout", "absolute_log_error": "MUST NOT READ"})
    data["execution_requested_endpoints"] = {5: ["Cmax_reference_vs_group"], 6: ["AUC_0_24h_reference_vs_group"]}
    metadata = fit(data, tmp_path, monkeypatch)
    assert metadata["error_metadata"]["training_studies"] == 3
    assert metadata["error_metadata"]["enabled_endpoints"] == ["Cmax_reference_vs_group"]
    del data["error_records"], data["grades"], data["applicability"]
    result = light.predict_fold(data, [5, 6], metadata["checkpoint"], error_weight=.3)
    assert result["error_supported"].tolist() == [True, False]
    assert np.all(result["scores"][0] < result["base_scores"][0])
    np.testing.assert_array_equal(result["scores"][1], result["base_scores"][1])


def test_training_statistics_ignore_outer_holdout_and_conditions_change_score(tmp_path, monkeypatch):
    data = dataset()
    data["pair_features_by_variant"][0, :3, :, 0] = 0.
    data["pair_features_by_variant"][0, 6, :, 0] = 1e6
    metadata = fit(data, tmp_path, monkeypatch)
    assert metadata["feature_mean"][0] == 0.
    assert metadata["feature_std"][0] == 1.
    model = light.LightMatch(8, 4, dropout=0).eval()
    q, c = torch.ones(8), torch.ones(5, 8)
    f = torch.zeros(5, 4)
    with torch.inference_mode():
        original = model(q, c, f)["scores"]
        f[:, 0] = 1.
        changed = model(q, c, f)["scores"]
    assert not torch.equal(original, changed)


def test_unknown_labels_are_not_negative_or_observed_execution_failure():
    torch.manual_seed(2)
    model = light.LightMatch(8, 4, dropout=0)
    features = torch.randn(5, 4, requires_grad=True)
    result = model(torch.ones(8), torch.randn(5, 8), features)
    loss, _ = light.document_loss(result, torch.tensor([3, 0, -1, -1, -1]),
                                  torch.tensor([0, 1, -1, 2, -1]), torch.ones(5))
    loss.backward()
    assert torch.equal(features.grad[2], torch.zeros(4))
    assert features.grad[3].abs().sum() > 0  # semantic U is supervised
    assert torch.equal(features.grad[4], torch.zeros(4))


def test_resume_rejects_changed_labels_and_recovers_identical_weights(tmp_path, monkeypatch):
    data = dataset()
    monkeypatch.setattr(light, "ROOT", tmp_path)
    def interrupt(event):
        if event["stage"] == "epoch_complete":
            raise RuntimeError("test interruption after durable epoch")
    with pytest.raises(RuntimeError, match="test interruption"):
        light.train_fold(data, [0, 1, 2], [], tmp_path / "resume", epochs=2, pk_epochs=2, callback=interrupt)
    poisoned = copy.deepcopy(data); poisoned["grades"][0, 0] = 2
    with pytest.raises(ValueError, match="changed"):
        light.train_fold(poisoned, [0, 1, 2], [], tmp_path / "resume", epochs=2, pk_epochs=2)
    resumed = light.train_fold(data, [0, 1, 2], [], tmp_path / "resume", epochs=2, pk_epochs=2)
    uninterrupted = light.train_fold(data, [0, 1, 2], [], tmp_path / "fresh", epochs=2, pk_epochs=2)
    a = light.predict_fold(data, [6], resumed["checkpoint"])
    b = light.predict_fold(data, [6], uninterrupted["checkpoint"])
    np.testing.assert_array_equal(a["scores"], b["scores"])


def test_all_unknown_validation_uses_explicit_fallback(tmp_path, monkeypatch):
    data = dataset(); data["grades"][3] = -1; data["applicability"][3] = -1
    monkeypatch.setattr(light, "ROOT", tmp_path)
    metadata = light.train_fold(data, [0, 1, 2], [3], tmp_path / "fit", epochs=2, pk_epochs=2)
    assert metadata["history"][-1]["selection_metric"] == "no_validation_annotations_fixed_epoch_fallback"
    assert metadata["best_epoch"] == 2


def test_completed_checkpoint_reuses_weights_and_rejects_changed_configuration(tmp_path, monkeypatch):
    data = dataset()
    metadata = fit(data, tmp_path, monkeypatch)
    before = (Path(metadata["checkpoint"]) / "heads.pt").read_bytes()
    result = fit(data, tmp_path, monkeypatch)
    assert result["reused_checkpoint"] is True
    assert (Path(metadata["checkpoint"]) / "heads.pt").read_bytes() == before
    with pytest.raises(ValueError, match="configuration changed"):
        fit(data, tmp_path, monkeypatch, learning_rate=.01)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_frozen_int8_labels_work_for_training_and_no_positive_validation(tmp_path, monkeypatch, device):
    data = dataset()
    data["grades"] = data["grades"].astype(np.int8)
    data["applicability"] = data["applicability"].astype(np.int8)
    data["grades"][3] = 0
    monkeypatch.setattr(light, "ROOT", tmp_path)
    result = light.train_fold(data, [0, 1, 2], [3], tmp_path / "int8", epochs=1, pk_epochs=2, device=device)
    assert result["best_epoch"] == 1
    assert result["history"][0]["selection_metric"] == "negative_masked_validation_loss_no_positive_query"
