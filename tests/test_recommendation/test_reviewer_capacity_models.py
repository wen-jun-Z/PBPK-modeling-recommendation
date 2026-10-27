"""Capacity, leakage and restart checks for the fixed structural comparison."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from reviewer_capacity_models import CapacityConcat, METHODS, parameter_counts, trainer
from test_reviewer_controls import synthetic


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_capacity_is_matched_with_active_units_and_fixed_projection():
    assert parameter_counts() == {"pbpk_interaction": 224136, "concat_capacity": 224128}
    model = CapacityConcat(1024, 204)
    assert model.projection[0].out_features == 128
    assert model.fusion[0].in_features == 2 * 128 + 204
    assert model.fusion[0].out_features == 198
    assert model.rank.in_features == model.applicability.in_features == model.ordinal_location.in_features == 198
    with pytest.raises(ValueError, match="128-dimensional"):
        CapacityConcat(1024, 204, width=129)


@pytest.mark.parametrize("method", METHODS)
def test_all_registered_parameters_participate_in_predictions(method):
    torch.manual_seed(17)
    model = trainer(method).LightMatch(1024, 204, dropout=0)
    query = torch.randn(1024, requires_grad=True)
    candidates = torch.randn(7, 1024, requires_grad=True)
    features = torch.randn(7, 204, requires_grad=True)
    result = model(query, candidates, features)
    loss = sum(result[key].square().mean() for key in ("scores", "app_logits", "ordinal_logits"))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad, name
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) == parameter.numel(), name
    assert query.grad is None and candidates.grad is None  # frozen embedding inputs
    assert features.grad.abs().sum() > 0


def test_concat_fusion_receives_only_projected_text_and_conditions():
    torch.manual_seed(23)
    model = CapacityConcat(8, 4, dropout=0).eval()
    query, candidates, features = torch.randn(8), torch.randn(5, 8), torch.randn(5, 4)
    captured = []
    handle = model.fusion.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].detach().clone()))
    try:
        result = model(query, candidates, features)
    finally:
        handle.remove()
    q = model.projection(F.normalize(query, dim=-1)).expand(5, -1)
    c = model.projection(F.normalize(candidates, dim=-1))
    torch.testing.assert_close(captured[0], torch.cat((q, c, features), -1), rtol=0, atol=0)
    assert result["representation"].shape == (5, 198 + 4)


@pytest.mark.parametrize("method", METHODS)
def test_training_normalization_and_inference_do_not_read_holdout_outcomes(tmp_path, method):
    module, data = trainer(method), synthetic()
    module.ROOT = tmp_path
    data["pair_features_by_variant"][0, :3, :, 0] = 0
    data["pair_features_by_variant"][0, 6, :, 0] = 1e6
    data["grades"][6] = data["applicability"][6] = 99
    metadata = module.train_fold(data, [0, 1, 2], [], tmp_path / method,
                                 epochs=2, pk_epochs=1, error_train_indices=[])
    assert metadata["feature_mean"][0] == 0
    assert metadata["feature_std"][0] == 1
    assert metadata["error_train_indices"] == []
    assert metadata["error_metadata"]["enabled_endpoints"] == []
    assert metadata["trainable_text_encoder"] is False
    for key in ("grades", "applicability", "error_records"):
        del data[key]
    output = module.predict_fold(data, [6], tmp_path / method, return_representations=True)
    assert output["scores"].shape == (1, 5)
    assert output["representations"].shape == (1, 5, (198 if method == "concat_capacity" else 128) + 4)
    assert not output["error_supported"].any()
    np.testing.assert_allclose(output["probabilities"].sum(-1), 1, atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_unknown_labels_are_masked_but_semantic_insufficiency_is_supervised(method):
    torch.manual_seed(41)
    module = trainer(method)
    model = module.LightMatch(8, 4, dropout=0)
    features = torch.randn(5, 4, requires_grad=True)
    result = model(torch.randn(8), torch.randn(5, 8), features)
    loss, _ = module.document_loss(result, torch.tensor([3, 0, -1, -1, -1]),
                                   torch.tensor([0, 1, -1, 2, -1]), torch.ones(5))
    loss.backward()
    torch.testing.assert_close(features.grad[2], torch.zeros(4), rtol=0, atol=0)
    torch.testing.assert_close(features.grad[4], torch.zeros(4), rtol=0, atol=0)
    assert features.grad[3].abs().sum() > 0


@pytest.mark.parametrize("method", METHODS)
def test_group_overlap_and_error_supervision_outside_train_are_rejected(tmp_path, method):
    module, data = trainer(method), synthetic()
    module.ROOT = tmp_path
    data["groups"][3] = data["groups"][0]
    with pytest.raises(ValueError, match="grouped"):
        module.train_fold(data, [0, 1, 2], [3], tmp_path / "overlap")
    with pytest.raises(ValueError, match="belong"):
        module.train_fold(data, [0, 1, 2], [], tmp_path / "error", error_train_indices=[6])


def test_capacity_resume_preserves_weights_and_rejects_changed_seed(tmp_path):
    module, data = trainer("concat_capacity"), synthetic()
    module.ROOT = tmp_path

    def interrupt(event):
        if event["stage"] == "epoch_complete":
            raise RuntimeError("bounded interruption after durable epoch")

    kwargs = {"epochs": 2, "pk_epochs": 1, "error_train_indices": []}
    with pytest.raises(RuntimeError, match="bounded interruption"):
        module.train_fold(data, [0, 1, 2], [], tmp_path / "resume", callback=interrupt, **kwargs)
    with pytest.raises(ValueError, match="changed"):
        module.train_fold(data, [0, 1, 2], [], tmp_path / "resume", seed=23, **kwargs)
    resumed = module.train_fold(data, [0, 1, 2], [], tmp_path / "resume", **kwargs)
    fresh = module.train_fold(data, [0, 1, 2], [], tmp_path / "fresh", **kwargs)
    a = module.predict_fold(data, [6], resumed["checkpoint"])
    b = module.predict_fold(data, [6], fresh["checkpoint"])
    np.testing.assert_array_equal(a["scores"], b["scores"])
    np.testing.assert_array_equal(a["probabilities"], b["probabilities"])
    completed = module.train_fold(data, [0, 1, 2], [], tmp_path / "resume", **kwargs)
    assert completed["reused_checkpoint"] is True
    altered = copy.deepcopy(data)
    altered["grades"][0, 0] = 2
    with pytest.raises(ValueError, match="changed"):
        module.train_fold(altered, [0, 1, 2], [], tmp_path / "resume", **kwargs)
