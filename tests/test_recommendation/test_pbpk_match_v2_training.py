"""Scientifically material checks: isolation, unknown masks, gating and true ties."""
from pathlib import Path

import numpy as np
import pytest
import torch

from pbpk_kb.recommendation.pbpk_match_v2.model import ErrorHead, MatchHeads, document_loss, last_token, yes_no_margin
from pbpk_kb.recommendation.pbpk_match_v2 import training as t


def dataset():
    rng = np.random.default_rng(4)
    return {"query_records": [{"query_id": f"q{i}", "compound_id": f"drug{i}", "compounds": [f"drug{i}"], "endpoint": "Cmax"} for i in range(6)],
            "candidate_records": [{"candidate_id": f"c{i}", "family_id": f"f{i // 2}", "compounds": ["drug0"]} for i in range(4)],
            "query_texts": [[f"Drug {i} oral 40 mg" for i in range(6)]] * 3,
            "no_numeric_query_texts": [[f"Drug {i} oral [dose missing]" for i in range(6)]] * 3,
            "document_texts": ["candidate one", "candidate two", "candidate three", "candidate four"],
            "no_numeric_document_texts": ["candidate one omitted", "candidate two omitted", "candidate three omitted", "candidate four omitted"],
            "pair_features": rng.normal(size=(6, 4, 3)).astype(np.float32),
            "pair_features_by_variant": rng.normal(size=(3, 6, 4, 3)).astype(np.float32),
            "numeric_feature_indices": [1], "variant_names": ["original", "sentence_order", "missing_mechanism"],
            "grades": np.tile([3, 1, 0, -1], (6, 1)), "applicability": np.tile([0, 2, 1, -1], (6, 1)),
            "groups": np.arange(6), "error_records": []}


def test_last_token_handles_padding_and_empty_sequence():
    hidden = torch.arange(24).reshape(2, 4, 3)
    assert torch.equal(last_token(hidden, torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])), hidden[[0, 1], [3, 1]])
    with pytest.raises(ValueError):
        last_token(hidden, torch.zeros(2, 4))


def test_pretrained_margin_avoids_bfloat16_subtractive_rounding():
    hidden = torch.tensor([[100., 100.], [100., 101.]], dtype=torch.bfloat16)
    weights = torch.tensor([[1., 1.], [1.0078125, 1.]], dtype=torch.bfloat16)
    margin = yes_no_margin(hidden, weights)
    reference = torch.nn.functional.linear(hidden.float(), weights.float())
    torch.testing.assert_close(margin, reference[:, 1] - reference[:, 0])
    assert margin.dtype == torch.float32
    rounded = torch.nn.functional.linear(hidden, weights)
    assert not torch.equal(margin, (rounded[:, 1] - rounded[:, 0]).float())


def test_unknown_labels_have_exactly_zero_gradient():
    torch.manual_seed(3)
    heads = MatchHeads(5, 2, dropout=0)
    representation = torch.randn(4, 5, requires_grad=True)
    output = heads(representation, torch.randn(4, 2), torch.zeros(4))
    loss, _ = document_loss(output, torch.tensor([3, 0, -1, -1]), torch.tensor([0, 1, -1, 2]), torch.ones(4))
    loss.backward()
    assert torch.equal(representation.grad[2], torch.zeros(5))
    assert representation.grad[3].abs().sum() > 0  # U is a real class, unlike unannotated -1.
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in heads.parameters())


def test_all_unknown_batch_is_differentiable_zero():
    heads = MatchHeads(5, 2, dropout=0)
    output = heads(torch.randn(3, 5), torch.randn(3, 2), torch.zeros(3))
    loss, _ = document_loss(output, torch.full((3,), -1), torch.full((3,), -1), torch.ones(3))
    loss.backward()
    assert loss.item() == 0


def test_ordinal_probabilities_are_monotonic_and_scores_unrounded():
    heads = MatchHeads(5, 2, dropout=0)
    output = heads(torch.zeros(3, 5), torch.zeros(3, 2), torch.tensor([.01, .01001, .02]))
    assert torch.all(output["ordinal_logits"][:, :-1] > output["ordinal_logits"][:, 1:])
    assert output["scores"][0] != output["scores"][1]


def test_numeric_ablation_removes_text_and_features_and_uses_perturbed_features():
    data = dataset()
    assert "40" not in t._texts(data, 0, "no_numeric")[0]
    out = t._features(data, 0, 2, "no_numeric")
    assert np.all(out[:, 1] == 0)
    np.testing.assert_array_equal(out[:, 0], data["pair_features_by_variant"][2, 0, :, 0])


def test_error_head_gates_by_independent_studies_not_rows_and_never_reads_held_out_targets():
    data = dataset()
    for q, study in [(0, "s1"), (1, "s1"), (2, "s2"), (3, "s3")]:
        data["error_records"].extend({"query_index": q, "candidate_index": i, "endpoint": "Cmax",
                                      "study_id": study, "absolute_log_error": .2 + i / 10} for i in range(4))
    # Poisoned held-out target would throw if inspected, even before fitting.
    data["error_records"].append({"query_index": 5, "candidate_index": 0, "endpoint": "Cmax",
                                  "study_id": "test", "absolute_log_error": "DO NOT READ"})
    representations = {q: np.ones((4, 8), np.float32) * q for q in range(4)}
    head, metadata = t.train_error_head(data, [0, 1, 2], representations, epochs=2)
    assert head is None and metadata["enabled_endpoints"] == []
    head, metadata = t.train_error_head(data, [0, 1, 2, 3], representations, epochs=2)
    assert metadata["enabled_endpoints"] == ["Cmax"]
    assert metadata["training_rows"] == 16 and metadata["training_studies"] == 3
    assert torch.all(head(torch.ones(4, 8), torch.zeros(4, dtype=torch.long)) >= 0)


def test_error_head_does_not_backpropagate_to_document_representation():
    representation = torch.randn(4, 8, requires_grad=True)
    model = ErrorHead(8, 2)
    model(representation, torch.tensor([0, 1, 0, 1])).sum().backward()
    assert representation.grad is None
    assert model.endpoint.weight.grad is not None


def test_tie_expected_ndcg_is_invariant_to_catalogue_order():
    labels, scores = np.array([3, 0, 2, 0]), np.array([1., 1., 1., 0.])
    value = t._ndcg(labels, scores, k=2)
    permutation = np.array([2, 3, 0, 1])
    assert t._ndcg(labels[permutation], scores[permutation], k=2) == pytest.approx(value)


def test_partition_rejects_group_overlap_before_loading_backbone(tmp_path):
    data = dataset(); data["groups"][2] = data["groups"][0]
    with pytest.raises(ValueError, match="grouped"):
        t.train_fold(data, [0, 1], [2], tmp_path / "checkpoint")


class FakeEncoder(torch.nn.Module):
    """Fast input-only test encoder; no downloaded model and no fake experiment results."""
    def __init__(self):
        super().__init__()
        self.hidden_size = 5
        self.trainable_text = False
        self.truncations = 0
        self.tokenized_pairs = 0

    def tokenize(self, queries, documents, max_length):
        values = [[len(q) % 17, sum(map(ord, d)) % 17, 1] for q, d in zip(queries, documents)]
        return {"input_ids": torch.tensor(values), "attention_mask": torch.ones(len(values), 3, dtype=torch.long)}

    def forward(self, tokens):
        x = tokens["input_ids"].float()
        return torch.cat((x, x[:, :2] / 3), -1), x[:, 1] / 10


def test_fixed_refit_only_trains_requested_labels_and_predict_needs_no_labels(monkeypatch, tmp_path):
    data = dataset()
    # Invalid values outside development would fail the training validator if inspected.
    data["grades"][5] = 99; data["applicability"][5] = 99
    monkeypatch.setattr(t, "_encoder", lambda *a, **kw: FakeEncoder())
    # Cache safety is separately tested; tmp_path is pytest owned rather than project runtime.
    monkeypatch.setattr(t, "ROOT", Path(tmp_path))
    output = tmp_path / "checkpoint"
    result = t.train_fold(data, [0, 1, 2], [], output, method="mlp", epochs=2,
                          batch_size=4, cache_directory=tmp_path / "cache")
    assert result["epochs_completed"] == 2 and result["best_epoch"] == 2
    assert result["fixed_epochs"]
    del data["grades"], data["applicability"], data["error_records"]
    events = []
    predictions = t.predict_fold(data, [5], output, batch_size=4, return_representations=True, callback=events.append)
    assert predictions["scores"].shape == (1, 4)
    np.testing.assert_allclose(predictions["probabilities"].sum(-1), 1, atol=1e-6)
    assert not predictions["error_supported"].any()
    progress = [event for event in events if event["stage"] == "document_predict"]
    assert len(progress) == 1 and progress[0]["completed_queries"] == progress[0]["total_queries"] == 1
    assert progress[0]["query_index"] == 5 and progress[0]["elapsed_seconds"] >= 0


def test_identical_token_inputs_share_exact_inference_scores(monkeypatch, tmp_path):
    data = dataset()
    data["document_texts"][1] = data["document_texts"][0]
    meta = {"method": "full", "max_length": 768, "batch_size": 2}
    hidden, margin = t.encode_query(data, 0, 0, FakeEncoder(), meta)
    np.testing.assert_array_equal(hidden[0], hidden[1])
    assert margin[0] == margin[1]


def test_full_query_gradient_cache_matches_direct_graph_with_dropout():
    torch.manual_seed(21)
    direct = MatchHeads(5, 2, dropout=.3)
    import copy
    cached = copy.deepcopy(direct)
    text, features, margin = torch.randn(7, 5), torch.randn(7, 2), torch.randn(7)
    grades, apps = torch.tensor([3, 1, 0, 0, -1, 2, 0]), torch.tensor([0, 2, 1, 1, -1, 0, 1])
    weights, same = torch.ones(7), torch.tensor([1, 1, 0, 0, 0, 1, 0]).bool()
    torch.manual_seed(45)
    outputs = [direct(text[i:i+3], features[i:i+3], margin[i:i+3]) for i in range(0, 7, 3)]
    out = {k: torch.cat([o[k] for o in outputs]) for k in ("scores", "app_logits", "ordinal_logits")}
    loss, _ = document_loss(out, grades, apps, weights, same)
    loss.backward()
    torch.manual_seed(45)
    cached_loss, _ = t.full_query_backward(lambda chosen: cached(text[chosen], features[chosen], margin[chosen]),
                                          np.arange(7), grades, apps, weights, same, batch_size=3, device="cpu")
    assert cached_loss == pytest.approx(float(loss.detach()), abs=1e-6)
    for p, q in zip(direct.parameters(), cached.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)


def test_no_positive_validation_uses_masked_loss_without_looking_at_unknowns():
    data = dataset(); data["grades"][0] = [0, 0, -1, -1]
    prediction = {"scores": np.zeros((1, 4), np.float32), "logits": np.zeros((1, 4, 3), np.float32),
                  "ordinal_logits": np.zeros((1, 4, 3), np.float32)}
    value, reason = t._validation_objective(data, [0], prediction, np.ones(4), np.ones(3), "cpu")
    assert value < 0 and reason == "negative_masked_validation_loss_no_positive_query"


def test_endpoint_selection_uses_requested_design_even_when_unsupported():
    assert t._query_endpoint({}, ["AUCtau", "Cmax"]) == "Cmax"
    assert t._query_endpoint({}, ["Cmax_ss", "AUCtau"]) == "Cmax_ss"
    assert t._query_endpoint({"features": {"endpoints": ["cmax", "auc"]}}) == "Cmax_reference_vs_group"
    assert t._query_endpoint({}, ["AUC_infinity", "Cmax_reference_vs_group"]) == "Cmax_reference_vs_group"


def test_train_only_class_weights_respond_to_rare_positive_not_outer_labels():
    data = dataset(); data["applicability"][1] = [1, 1, 1, -1]
    qweights, cweights = t._weights(data, [0, 1])
    before = t._app_class_weights(data, [0, 1], qweights, cweights)
    data["applicability"][2:] = 0
    after = t._app_class_weights(data, [0, 1], qweights, cweights)
    assert before[0] > before[1]
    np.testing.assert_array_equal(before, after)


def test_epoch_resume_reproduces_uninterrupted_training(monkeypatch, tmp_path):
    data = dataset()
    monkeypatch.setattr(t, "_encoder", lambda *a, **kw: FakeEncoder())
    monkeypatch.setattr(t, "ROOT", Path(tmp_path))
    common = dict(method="mlp", epochs=2, batch_size=3, cache_directory=tmp_path / "cache", seed=17)
    complete = tmp_path / "complete"
    t.train_fold(data, [0, 1, 2], [], complete, **common)
    def interrupt(event):
        if event["stage"] == "document_epoch_complete" and event["epoch"] == 1:
            raise RuntimeError("simulated interruption")
    resumed = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="simulated"):
        t.train_fold(data, [0, 1, 2], [], resumed, callback=interrupt, **common)
    assert (resumed / "training_state.pt").exists()
    metadata = t.train_fold(data, [0, 1, 2], [], resumed, **common)
    assert metadata["epochs_completed"] == 2
    assert not (resumed / "training_state.pt").exists()
    a = torch.load(complete / "heads.pt", weights_only=True)
    b = torch.load(resumed / "heads.pt", weights_only=True)
    for name in a["heads"]:
        torch.testing.assert_close(a["heads"][name], b["heads"][name], atol=0, rtol=0)
