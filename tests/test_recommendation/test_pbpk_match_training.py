from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pbpk_kb.recommendation.pbpk_match.training import (
    _candidate_weights,
    _checkpoint_choice,
    _class_weights,
    _ordinal_class_weights,
    _query_loss,
    _sample_candidates,
    _validation_ordinal_loss,
    predict_neural,
    predict_tree,
    train_neural,
    train_tree,
)


def fixture_data():
    rng = np.random.default_rng(17)
    q = rng.normal(size=(3, 4, 3, 8)).astype(np.float32)
    q[1] = q[0]
    pair = rng.normal(size=(3, 4, 5, 6)).astype(np.float32)
    pair[1] = pair[0]
    return {
        "q_embeddings": q, "q_types": np.tile([0, 1, 2], (3, 4, 1)),
        "q_mask": np.ones((3, 4, 3), dtype=bool),
        "c_embeddings": rng.normal(size=(5, 3, 8)).astype(np.float32),
        "c_types": np.tile([0, 1, 2], (5, 1)), "c_mask": np.ones((5, 3), dtype=bool),
        "pair_features": pair, "task_ids": np.asarray([0, 1, 0, 1]),
        "grades": np.asarray([[3, 2, 0, -1, 1], [0, 1, 3, 2, -1], [3, 0, -1, 2, 1], [999] * 5]),
        "applicability": np.asarray([[0, 2, 1, -1, 2], [1, 2, 0, 0, -1], [0, 1, 2, 0, 2], [999] * 5]),
        "query_records": [{"query_id": f"q{i}", "compound_id": f"drug{i}", "compounds": [f"drug{i}"]} for i in range(4)],
        "candidate_records": [{"candidate_id": f"c{i}", "family_id": f"f{i}", "compounds": ["drug0", "drug1"] if i < 3 else ["other"]} for i in range(5)],
        "variant_names": ["original", "reordered", "missing_evidence"],
    }


def test_unknown_grade_and_unprocessed_app_have_zero_loss_gradient():
    scores = torch.tensor([2., -1., 99.], requires_grad=True)
    logits = torch.zeros((3, 3), requires_grad=True)
    loss, _ = _query_loss(scores, logits, torch.tensor([3, 0, -1]), torch.tensor([0, 1, -1]),
                         torch.ones(3), torch.ones(3), torch.ones(3, dtype=torch.bool), {})
    loss.backward()
    assert scores.grad[2].item() == 0
    assert torch.count_nonzero(logits.grad[2]).item() == 0
    assert scores.grad[:2].abs().sum() > 0


def test_all_negative_queries_receive_ordinal_training():
    scores = torch.zeros(3, requires_grad=True)
    logits = torch.zeros((3, 3), requires_grad=True)
    loss, pieces = _query_loss(scores, logits, torch.zeros(3, dtype=torch.long), torch.full((3,), -1),
                              torch.ones(3), torch.ones(3), torch.zeros(3, dtype=torch.bool), {})
    loss.backward()
    assert pieces["rank_loss"] == 0
    assert pieces["ordinal_loss"] > 0
    assert torch.all(scores.grad > 0)


def test_class_weights_ignore_other_partitions_and_sampling_keeps_hard_negatives():
    data = fixture_data()
    before = _class_weights(data, np.asarray([0, 1]))
    data["applicability"][2:] = -200
    np.testing.assert_array_equal(before, _class_weights(data, np.asarray([0, 1])))
    selected = _sample_candidates(data, 0, 1, np.random.default_rng(3))
    assert {0, 1, 2}.issubset(selected)


def test_family_packet_weights_do_not_count_versions_as_independent():
    data = fixture_data()
    data["candidate_records"][1]["family_id"] = "f0"
    data["candidate_packet_groups"] = ["p0", "p0", "p2", "p3", "p4"]
    weights = _candidate_weights(data, 0)
    assert weights[0] == weights[1]
    assert weights[0] + weights[1] == pytest.approx(weights[2])


@pytest.mark.parametrize("kind", ["full", "mlp", "no_type", "no_null", "hybrid"])
def test_train_predict_cpu_and_reject_overlap(kind):
    torch.set_num_threads(1)
    data = fixture_data()
    config = {"epochs": 1, "hidden_dim": 16, "dropout": 0., "chunk_size": 3, "consistency_weight": .05}
    model, history = train_neural(data, [0, 1], [2], model_kind=kind, config=config)
    # Malformed held-out labels are intentional: prediction must never read them.
    scores, probabilities = predict_neural(model, data, [3], chunk_size=2)
    assert scores.shape == (1, 5)
    assert probabilities.shape == (1, 5, 3)
    np.testing.assert_allclose(probabilities.sum(-1), 1., atol=1e-6)
    assert history["best_epoch"] == 1
    assert "known_pool" in history["selection_metric"]
    reordered, _ = predict_neural(model, data, [3], variant="reordered", chunk_size=2)
    np.testing.assert_allclose(scores, reordered, atol=1e-6)
    with pytest.raises(ValueError, match="disjoint"):
        train_neural(data, [0, 1], [1, 2], config=config)


def test_test_labels_do_not_change_fitted_parameters():
    torch.set_num_threads(1)
    data = fixture_data()
    config = {"epochs": 1, "hidden_dim": 16, "dropout": 0., "consistency_weight": 0.}
    first, _ = train_neural(data, [0, 1], [2], model_kind="mlp", seed=29, config=config)
    altered = deepcopy(data)
    altered["grades"][3] = -700
    altered["applicability"][3] = -800
    second, _ = train_neural(altered, [0, 1], [2], model_kind="mlp", seed=29, config=config)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0., atol=0.)


def test_tree_uses_development_only_and_returns_full_app_classes():
    pytest.importorskip("xgboost")
    data = fixture_data()
    model, history = train_tree(data, [0, 1], [2], config={"n_estimators": 4, "tree_patience": 2, "n_jobs": 1})
    scores, probabilities = predict_tree(model, data, [3])
    assert scores.shape == (1, 5)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(probabilities.sum(-1), 1.)
    assert history["train_query_indices"] == [0, 1]


def test_fixed_refit_requires_explicit_epoch_budget_and_uses_last_checkpoint():
    torch.set_num_threads(1)
    data = fixture_data()
    with pytest.raises(ValueError, match="nonempty"):
        train_neural(data, [0, 1], [], model_kind="mlp", config={"epochs": 1})
    model, history = train_neural(data, [0, 1], [], model_kind="mlp", config={"fixed_epochs": 2, "hidden_dim": 16})
    assert history["best_epoch"] == 2
    assert history["best_validation_value"] is None
    assert history["validation_query_indices"] == []
    assert history["init_args"]["hidden_dim"] == 16
    assert history["selection_metric"] == "fixed_epochs_last_checkpoint_no_validation"
    scores, _ = predict_neural(model, data, [3])
    assert np.isfinite(scores).all()


def test_tree_fixed_refit_and_no_positive_validation_fallback():
    pytest.importorskip("xgboost")
    data = fixture_data()
    model, history = train_tree(data, [0, 1], [], config={"fixed_n_estimators": 3, "n_jobs": 1})
    assert history["best_iteration"] == 2
    assert history["evaluation_history"] == {}
    assert history["selection_metric"] == "fixed_n_estimators"
    assert predict_tree(model, data, [3])[0].shape == (1, 5)
    data["grades"][2] = [0, 0, -1, 1, 1]
    _, fallback = train_tree(data, [0, 1], [2], config={"n_estimators": 3, "n_jobs": 1})
    assert fallback["selection_metric"] == "rmse_no_positive_validation"


def test_ordinal_balance_uses_only_known_training_grades_and_handles_absent_g2():
    data = fixture_data()
    data["grades"][0] = [0, 0, 0, 3, -1]
    weights = _ordinal_class_weights(data, np.asarray([0]), "sqrt_inverse")
    frequency = np.asarray([.75, 0., 0., .25])
    assert np.isfinite(weights).all()
    assert weights[1] == weights[2] == 0
    assert weights[3] > weights[0]
    assert np.dot(frequency, weights) == pytest.approx(1.)
    data["grades"][1:] = -900
    np.testing.assert_array_equal(weights, _ordinal_class_weights(data, np.asarray([0]), "sqrt_inverse"))
    data["grades"][0, 4] = -1
    np.testing.assert_array_equal(weights, _ordinal_class_weights(data, np.asarray([0]), "sqrt_inverse"))
    np.testing.assert_array_equal(_ordinal_class_weights(data, np.asarray([0]), "none"), np.ones(4))


def test_ordinal_balance_caps_rare_grade_before_expected_mass_normalization():
    data = {"query_records": [{"query_id": "q", "compound_id": "drug"}],
            "candidate_records": [{"candidate_id": f"c{i}"} for i in range(201)],
            "grades": np.asarray([[0] * 200 + [3]])}
    weights = _ordinal_class_weights(data, np.asarray([0]), "sqrt_inverse")
    assert weights.max() <= 10
    assert weights[3] / weights[0] == pytest.approx(10 * np.sqrt(200 / 201))
    assert np.dot(np.asarray([200 / 201, 0., 0., 1 / 201]), weights) == pytest.approx(1.)


def test_ordinal_balance_matches_query_normalized_loss_not_annotation_count():
    data = fixture_data()
    data["grades"][0] = [0, 0, 0, 0, -1]
    data["grades"][1] = [3, -1, -1, -1, -1]
    weights = _ordinal_class_weights(data, np.asarray([0, 1]), "sqrt_inverse")
    np.testing.assert_allclose(weights, [1., 0., 0., 1.])


def test_balanced_ordinal_loss_masks_unknown_and_remains_finite_for_all_negative():
    data = fixture_data()
    data["grades"][0] = [0, 0, 0, 0, -1]
    weights = torch.as_tensor(_ordinal_class_weights(data, np.asarray([0]), "sqrt_inverse"))
    torch.testing.assert_close(weights, torch.tensor([1., 0., 0., 0.]))
    scores = torch.zeros(5, requires_grad=True)
    logits = torch.zeros((5, 3), requires_grad=True)
    loss, pieces = _query_loss(scores, logits, torch.as_tensor(data["grades"][0]), torch.full((5,), -1),
                              torch.ones(5), torch.ones(3), torch.zeros(5, dtype=torch.bool), {}, ordinal_weights=weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert pieces["rank_loss"] == 0
    assert torch.all(scores.grad[:4] > 0)
    assert scores.grad[4] == 0
    data["grades"][0] = -1
    np.testing.assert_array_equal(_ordinal_class_weights(data, np.asarray([0]), "sqrt_inverse"), np.zeros(4))


@pytest.mark.parametrize("kind", ["full", "mlp"])
def test_full_and_mlp_share_training_fold_ordinal_balance(kind):
    torch.set_num_threads(1)
    data = fixture_data()
    config = {"fixed_epochs": 1, "hidden_dim": 16, "dropout": 0., "consistency_weight": 0., "ordinal_balance": "sqrt_inverse"}
    _, history = train_neural(data, [0, 1], [], model_kind=kind, config=config)
    expected = _ordinal_class_weights(data, np.asarray([0, 1]), "sqrt_inverse")
    np.testing.assert_allclose(history["ordinal_weights_train_only"], expected)
    assert history["config"]["ordinal_balance"] == "sqrt_inverse"
    assert "visible-metadata" in history["candidate_weighting"]


def test_checkpoint_selector_never_trades_primary_quality_for_loss():
    assert _checkpoint_choice(.81, .8, 10., .1, allow_secondary=True) == (True, "primary_improvement")
    assert _checkpoint_choice(.79, .8, .01, 10., allow_secondary=True)[0] is False
    assert _checkpoint_choice(.8, .8, .1, .2, allow_secondary=True) == (True, "ordinal_loss_breaks_primary_tie")
    assert _checkpoint_choice(.8, .8, .1, .2, allow_secondary=False)[0] is False
    assert _checkpoint_choice(.8 - 1e-9, .8, .1, .2, allow_secondary=True)[0] is False
    assert _checkpoint_choice(.8 - 5e-13, .8, .1, .2, allow_secondary=True)[0] is True


def test_validation_ordinal_uses_only_supplied_rows_and_ignores_unknown_scores():
    data = fixture_data()
    scores = np.asarray([[2., -2., 1000., .5, -.5]])
    first = _validation_ordinal_loss(data, np.asarray([2]), scores)
    data["grades"][[0, 1, 3]] = 900
    scores[0, 2] = np.nan
    assert _validation_ordinal_loss(data, np.asarray([2]), scores) == first
    data["grades"][2] = -1
    assert _validation_ordinal_loss(data, np.asarray([2]), scores) is None


@pytest.mark.parametrize("tie_break,expected_epoch", [("none", 1), ("ordinal_loss", 2)])
def test_primary_plateau_selects_later_better_ordinal_checkpoint_only_when_enabled(monkeypatch, tie_break, expected_epoch):
    import pbpk_kb.recommendation.pbpk_match.training as training

    torch.set_num_threads(1)
    data = fixture_data()
    validation_scores = iter([
        np.asarray([[1., -1., 999., .5, 0.]], dtype=np.float32),
        np.asarray([[2., -2., -999., .75, -.5]], dtype=np.float32),
    ])
    def fake_validation(model, supplied, indices, **kwargs):
        assert supplied is data and indices == [2]
        return next(validation_scores), np.full((1, 5, 3), 1 / 3)
    monkeypatch.setattr(training, "predict_neural", fake_validation)
    _, history = training.train_neural(data, [0, 1], [2], model_kind="mlp", config={
        "epochs": 2, "hidden_dim": 16, "dropout": 0., "consistency_weight": 0., "validation_tie_break": tie_break})
    assert history["best_epoch"] == expected_epoch
    assert history["epochs"][0]["validation_value"] == history["epochs"][1]["validation_value"]
    if tie_break == "ordinal_loss":
        assert history["epochs"][1]["validation_secondary_ordinal_loss"] < history["epochs"][0]["validation_secondary_ordinal_loss"]
        assert history["epochs"][1]["checkpoint_selection_reason"] == "ordinal_loss_breaks_primary_tie"
        assert history["best_validation_secondary_ordinal_loss"] == history["epochs"][1]["validation_secondary_ordinal_loss"]


def test_no_positive_fallback_does_not_apply_ordinal_as_secondary_twice():
    torch.set_num_threads(1)
    data = fixture_data()
    data["grades"][2] = [0, 0, -1, 1, 1]
    _, history = train_neural(data, [0, 1], [2], model_kind="mlp", config={
        "epochs": 1, "hidden_dim": 16, "validation_tie_break": "ordinal_loss"})
    assert history["selection_metric"] == "negative_known_ordinal_loss_no_positive_validation"
    assert history["best_validation_secondary_ordinal_loss"] is None
    assert history["epochs"][0]["validation_secondary_ordinal_loss"] is None
