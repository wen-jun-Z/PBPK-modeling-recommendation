"""Identical actual inputs stay tied despite chunk-dependent numeric noise."""

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pbpk_kb.recommendation.pbpk_match.model import PBPKMatch, PairMLP
from pbpk_kb.recommendation.pbpk_match.training import predict_neural


def fixture_data():
    rng = np.random.default_rng(81)
    c = rng.normal(size=(4, 2, 8)).astype(np.float32)
    c[2] = c[0]
    features = rng.normal(size=(2, 2, 4, 3)).astype(np.float32)
    features[:, :, 2] = features[:, :, 0]
    return {
        "q_embeddings": rng.normal(size=(2, 2, 3, 8)).astype(np.float32),
        "q_types": np.tile([0, 1, 2], (2, 2, 1)),
        "q_mask": np.ones((2, 2, 3), dtype=bool),
        "c_embeddings": c,
        "c_types": np.tile([0, 1], (4, 1)),
        "c_mask": np.ones((4, 2), dtype=bool),
        "pair_features": features,
        "task_ids": np.asarray([0, 1]),
        "variant_names": ["original", "reordered"],
        # Matching metadata/families must never establish tensor equivalence.
        "candidate_records": [{"candidate_id": str(i), "family_id": "one_family"} for i in range(4)],
        "candidate_packet_groups": ["one_metadata_hash"] * 4,
    }


class NumericNoiseModel(torch.nn.Module):
    """Instrumented forward with deliberately chunk-position-sensitive noise."""

    def __init__(self):
        super().__init__()
        self.candidates_evaluated = 0

    def forward(self, q, qt, qm, c, ct, cm, features, task):
        self.candidates_evaluated += len(c)
        score = c.sum((1, 2)) + ct.sum(1) + cm.sum(1) + features.sum(1)
        score = score + torch.arange(len(c), device=c.device) * 1e-5
        return {"rank_score": score, "app_logits": torch.stack((score, -score, torch.zeros_like(score)), -1)}


def test_only_actual_identical_inputs_share_one_prediction():
    data, model = fixture_data(), NumericNoiseModel()
    scores, probabilities = predict_neural(model, data, [0, 1], chunk_size=2)
    assert model.candidates_evaluated == 6  # Three unique inputs per query.
    np.testing.assert_array_equal(scores[:, 0], scores[:, 2])
    np.testing.assert_array_equal(probabilities[:, 0], probabilities[:, 2])
    assert np.any(scores[:, 0] != scores[:, 1])
    assert model.training  # The caller's mode is restored.


@pytest.mark.parametrize("changed", ["c_embeddings", "c_types", "c_mask", "pair_features"])
def test_each_actual_input_is_checked_before_reuse(changed):
    data, model = fixture_data(), NumericNoiseModel()
    if changed == "c_mask":
        data[changed][2, 0] = False
    elif changed == "c_types":
        data[changed][2, 0] = 2
    elif changed == "c_embeddings":
        data[changed][2, 0, 0] += 1
    else:
        data[changed][0, 0, 2, 0] += 1
    scores, _ = predict_neural(model, data, [0], chunk_size=2)
    assert model.candidates_evaluated == 4
    assert scores[0, 0] != scores[0, 2]


def test_pair_equivalence_is_specific_to_query_and_variant():
    data, model = fixture_data(), NumericNoiseModel()
    data["pair_features"][0, 1, 2, 0] += 1
    data["pair_features"][1, 0, 2, 0] += 2
    original, _ = predict_neural(model, data, [0, 1])
    assert model.candidates_evaluated == 7
    assert original[0, 0] == original[0, 2]
    assert original[1, 0] != original[1, 2]
    reordered, _ = predict_neural(model, data, [0], variant="reordered")
    assert model.candidates_evaluated == 11
    assert reordered[0, 0] != reordered[0, 2]


def test_fingerprints_are_not_used_as_proof_of_equality(monkeypatch):
    class ConstantHash:
        def digest(self):
            return b"deliberate-collision"

    monkeypatch.setattr(hashlib, "sha256", lambda *_: ConstantHash())
    data, model = fixture_data(), NumericNoiseModel()
    data["pair_features"][0, 1, 2, 0] += 1
    scores, _ = predict_neural(model, data, [0, 1])
    assert model.candidates_evaluated == 7
    assert scores[0, 0] == scores[0, 2]
    assert scores[1, 0] != scores[1, 2]


def test_repeated_call_rechecks_mutated_inputs():
    data, model = fixture_data(), NumericNoiseModel()
    predict_neural(model, data, [0])
    data["c_embeddings"][2, 0, 0] += 1
    scores, _ = predict_neural(model, data, [0])
    assert model.candidates_evaluated == 7
    assert scores[0, 0] != scores[0, 2]


@pytest.mark.parametrize("model_type", [PBPKMatch, PairMLP])
def test_real_network_predictions_are_exactly_tied_across_chunks(model_type):
    torch.manual_seed(23)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = fixture_data()
        model = model_type(8, 3, 3, 2, hidden_dim=16, dropout=.1).eval()
        first, first_p = predict_neural(model, data, [0, 1], chunk_size=1)
        second, second_p = predict_neural(model, data, [0, 1], chunk_size=3)
        for scores, probabilities in ((first, first_p), (second, second_p)):
            np.testing.assert_array_equal(scores[:, 0], scores[:, 2])
            np.testing.assert_array_equal(probabilities[:, 0], probabilities[:, 2])
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(first_p, second_p, rtol=1e-5, atol=1e-6)
    finally:
        torch.set_num_threads(old_threads)
