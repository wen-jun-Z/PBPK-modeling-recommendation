"""Structural correctness checks, not claims about scientific ranking quality."""

import pytest
import torch

from pbpk_kb.recommendation.pbpk_match.model import APPLICABILITY_CLASSES, PBPKMatch, PairMLP


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.manual_seed(23)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(batch=2, nq=3, nc=4, dim=16):
    return {
        "q_embeddings": torch.randn(batch, nq, dim),
        "q_types": torch.arange(nq).repeat(batch, 1) % 3,
        "q_mask": torch.ones(batch, nq, dtype=torch.bool),
        "c_embeddings": torch.randn(batch, nc, dim),
        "c_types": torch.arange(nc).repeat(batch, 1) % 3,
        "c_mask": torch.ones(batch, nc, dtype=torch.bool),
        "pair_features": torch.randn(batch, 5),
        "task_ids": torch.arange(batch) % 2,
    }


def make_model(cls=PBPKMatch, **kwargs):
    return cls(16, 5, 3, 2, hidden_dim=24, dropout=0, **kwargs).eval()


def test_no_cross_type_fallback_and_explicit_null():
    data = inputs()
    data["q_types"].fill_(2)
    data["c_types"].fill_(0)
    model = make_model()
    result = model(**data)
    alignment = result["matching"]
    assert not alignment["allowed_real_matches"].any()
    assert not alignment["has_same_type"].any()
    assert torch.equal(alignment["null_probability"], torch.ones(2, 3))
    assert torch.equal(alignment["attention"][..., :-1], torch.zeros(2, 3, 4))
    data["c_embeddings"] = torch.randn_like(data["c_embeddings"]) * 100
    changed = model(**data)
    torch.testing.assert_close(result["rank_score"], changed["rank_score"])
    torch.testing.assert_close(result["app_logits"], changed["app_logits"])


@pytest.mark.parametrize("cls", [PBPKMatch, PairMLP])
def test_node_permutation_invariant(cls):
    data = inputs()
    model = make_model(cls)
    original = model(**data)
    q_order = torch.tensor([2, 0, 1])
    c_order = torch.tensor([3, 1, 0, 2])
    permuted = dict(data)
    for key in ("q_embeddings", "q_types", "q_mask"):
        permuted[key] = data[key][:, q_order]
    for key in ("c_embeddings", "c_types", "c_mask"):
        permuted[key] = data[key][:, c_order]
    actual = model(**permuted)
    torch.testing.assert_close(original["rank_score"], actual["rank_score"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(original["app_logits"], actual["app_logits"], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("cls", [PBPKMatch, PairMLP])
@pytest.mark.parametrize("nq,nc", [(0, 0), (0, 4), (3, 0), (3, 4)])
def test_empty_or_masked_nodes_are_finite(cls, nq, nc):
    data = inputs(nq=nq, nc=nc)
    data["q_mask"][0].fill_(False)
    data["c_mask"][0].fill_(False)
    data["q_embeddings"][0].fill_(float("nan"))
    data["c_embeddings"][0].fill_(float("nan"))
    data["q_types"][0].fill_(-500)
    data["c_types"][0].fill_(500)
    result = make_model(cls)(**data)
    assert result["rank_score"].shape == (2,)
    assert result["app_logits"].shape == (2, 3)
    assert torch.isfinite(result["rank_score"]).all()
    assert torch.isfinite(result["app_logits"]).all()
    if cls is PBPKMatch:
        assert torch.equal(result["matching"]["requirement_weights"][0], torch.zeros(nq))


def test_padding_cannot_change_outputs():
    data = inputs()
    model = make_model()
    original = model(**data)
    padded = dict(data)
    for prefix in ("q", "c"):
        padded[prefix + "_embeddings"] = torch.cat((data[prefix + "_embeddings"], torch.full((2, 5, 16), float("nan"))), 1)
        padded[prefix + "_types"] = torch.cat((data[prefix + "_types"], torch.full((2, 5), -7)), 1)
        padded[prefix + "_mask"] = torch.cat((data[prefix + "_mask"], torch.zeros(2, 5, dtype=torch.bool)), 1)
    actual = model(**padded)
    torch.testing.assert_close(original["rank_score"], actual["rank_score"])
    torch.testing.assert_close(original["app_logits"], actual["app_logits"])


@pytest.mark.parametrize("ablation", ["full", "no_type", "no_null"])
def test_null_ablation_semantics(ablation):
    data = inputs()
    data["q_types"].fill_(1)
    data["c_types"].fill_(0)
    model = make_model(ablation=ablation)
    alignment = model(**data)["matching"]
    if ablation == "no_type":
        assert alignment["allowed_real_matches"].all()
        assert (alignment["null_probability"] < 1).all()
    else:
        assert not alignment["allowed_real_matches"].any()
        assert (alignment["null_probability"] == 1).all()
    if ablation == "no_null":
        data["c_types"].fill_(1)
        assert (model(**data)["matching"]["null_probability"] == 0).all()
        assert not model.null_node.requires_grad


@pytest.mark.parametrize("cls", [PBPKMatch, PairMLP])
def test_finite_gradients_and_frozen_encoder_inputs(cls):
    data = inputs()
    data["q_embeddings"].requires_grad_(True)
    data["c_embeddings"].requires_grad_(True)
    model = make_model(cls).train()
    result = model(**data)
    loss = result["rank_score"].square().mean() + torch.nn.functional.cross_entropy(result["app_logits"], torch.tensor([0, 2]))
    loss.backward()
    assert data["q_embeddings"].grad is None
    assert data["c_embeddings"].grad is None
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.projection[0].weight.grad.abs().sum() > 0


def test_normalized_requirement_gate_and_parameter_budget():
    result = make_model()(**inputs())
    torch.testing.assert_close(result["matching"]["requirement_weights"].sum(1), torch.ones(2))
    production = PBPKMatch(1024, 32, 16, 8)
    assert production.parameter_count < 1_000_000
    assert APPLICABILITY_CLASSES == ("Y", "N", "U")


def test_invalid_real_types_fail_explicitly():
    data = inputs()
    data["q_types"][0, 0] = -1
    with pytest.raises(ValueError, match="real query"):
        make_model()(**data)


def test_state_dict_roundtrip():
    data = inputs()
    first, second = make_model(), make_model()
    second.load_state_dict(first.state_dict())
    torch.testing.assert_close(first(**data)["rank_score"], second(**data)["rank_score"])
