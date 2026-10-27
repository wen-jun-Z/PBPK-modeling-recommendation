"""Architectural contracts for the bounded residual, not quality guarantees."""

import io

import pytest
import torch

from pbpk_kb.recommendation.pbpk_match.hybrid import PBPKMatchHybrid
from pbpk_kb.recommendation.pbpk_match.model import PBPKMatch, PairMLP


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.manual_seed(37)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(nq=3, nc=4):
    return dict(
        q_embeddings=torch.randn(2, nq, 16),
        q_types=torch.arange(nq).repeat(2, 1) % 3,
        q_mask=torch.ones(2, nq, dtype=torch.bool),
        c_embeddings=torch.randn(2, nc, 16),
        c_types=torch.arange(nc).repeat(2, 1) % 3,
        c_mask=torch.ones(2, nc, dtype=torch.bool),
        pair_features=torch.randn(2, 5),
        task_ids=torch.tensor([0, 1]),
    )


def make_model(**kwargs):
    return PBPKMatchHybrid(16, 5, 3, 2, hidden_dim=24, dropout=0, **kwargs).eval()


def test_shared_parameters_are_deduplicated_and_not_frozen():
    model = make_model()
    for name in ("projection", "type_embedding", "task_embedding"):
        assert getattr(model.typed_branch, name) is getattr(model.global_branch, name)
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})
    shared = sum(p.numel() for name in ("projection", "type_embedding", "task_embedding")
                 for p in getattr(model.global_branch, name).parameters())
    expected = model.global_branch.parameter_count + model.typed_branch.parameter_count - shared + 2
    assert model.parameter_count == expected
    assert all(p.requires_grad for p in model.global_branch.parameters())
    assert PBPKMatchHybrid(1024, 61, 14, 5).parameter_count < 1_000_000


@pytest.mark.parametrize("gate_value", [-20., 0., 20.])
def test_correction_is_bounded_even_for_extreme_typed_logits(gate_value):
    model, data = make_model(residual_scale=2.), inputs()
    with torch.no_grad():
        model.rank_gate_logit.fill_(gate_value)
        model.app_gate_logit.fill_(gate_value)
        model.typed_branch.rank_head.weight.zero_()
        model.typed_branch.rank_head.bias.fill_(1000.)
        model.typed_branch.app_head.weight.zero_()
        model.typed_branch.app_head.bias.copy_(torch.tensor([-1000., 0., 1000.]))
    result = model(**data)
    baseline = model.global_branch(**data)
    assert torch.all(result["matching"]["rank_correction"].abs() <= 2.)
    assert torch.all(result["matching"]["app_correction"].abs() <= 2.)
    torch.testing.assert_close(result["rank_score"], baseline["rank_score"] + result["matching"]["rank_correction"])
    torch.testing.assert_close(result["app_logits"], baseline["app_logits"] + result["matching"]["app_correction"])


def test_initial_gate_is_point_one_and_disabling_is_exact_mlp():
    model, data = make_model(), inputs()
    result = model(**data)
    assert result["matching"]["rank_gate"].item() == pytest.approx(.1)
    assert result["matching"]["app_gate"].item() == pytest.approx(.1)
    assert result["matching"]["rank_correction"].abs().max() <= .1
    disabled = make_model(residual_enabled=False)
    mlp = PairMLP(16, 5, 3, 2, hidden_dim=24, dropout=0).eval()
    mlp.load_state_dict(disabled.global_branch.state_dict())
    actual, expected = disabled(**data), mlp(**data)
    torch.testing.assert_close(actual["rank_score"], expected["rank_score"], rtol=0, atol=0)
    torch.testing.assert_close(actual["app_logits"], expected["app_logits"], rtol=0, atol=0)
    assert disabled.typed_branch is None
    assert disabled.parameter_count == mlp.parameter_count
    assert not actual["matching"]["residual_enabled"]


@pytest.mark.parametrize("nq,nc", [(0, 0), (0, 4), (3, 0), (3, 4)])
def test_empty_and_masked_inputs_are_finite(nq, nc):
    data = inputs(nq, nc)
    for prefix in ("q", "c"):
        data[prefix + "_mask"][0].fill_(False)
        data[prefix + "_embeddings"][0].fill_(float("nan"))
        data[prefix + "_types"][0].fill_(-100)
    result = make_model()(**data)
    assert result["rank_score"].shape == (2,)
    assert result["app_logits"].shape == (2, 3)
    assert torch.isfinite(result["rank_score"]).all()
    assert torch.isfinite(result["app_logits"]).all()


def test_padding_and_node_permutation_cannot_change_outputs():
    data, model = inputs(), make_model()
    expected = model(**data)
    padded = dict(data)
    for prefix, nodes in (("q", 3), ("c", 4)):
        padded[prefix + "_embeddings"] = torch.cat((data[prefix + "_embeddings"], torch.full((2, 2, 16), float("nan"))), 1)
        padded[prefix + "_types"] = torch.cat((data[prefix + "_types"], torch.full((2, 2), -10)), 1)
        padded[prefix + "_mask"] = torch.cat((data[prefix + "_mask"], torch.zeros(2, 2, dtype=torch.bool)), 1)
        permutation = torch.arange(nodes + 2).flip(0)
        for suffix in ("_embeddings", "_types", "_mask"):
            padded[prefix + suffix] = padded[prefix + suffix][:, permutation]
    actual = model(**padded)
    torch.testing.assert_close(actual["rank_score"], expected["rank_score"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual["app_logits"], expected["app_logits"], rtol=1e-5, atol=1e-6)


def test_typed_branch_stays_type_constrained_and_global_branch_can_see_other_fields():
    data, model = inputs(), make_model()
    data["q_types"].fill_(2)
    data["c_types"].fill_(0)
    original = model(**data)
    assert not original["matching"]["allowed_real_matches"].any()
    assert torch.all(original["matching"]["null_probability"] == 1)
    data["c_embeddings"] = torch.randn_like(data["c_embeddings"])
    changed = model(**data)
    torch.testing.assert_close(original["matching"]["rank_correction"], changed["matching"]["rank_correction"])
    assert not torch.equal(original["matching"]["global_rank_score"], changed["matching"]["global_rank_score"])


def test_both_branches_and_gates_train_but_external_embeddings_stay_frozen():
    model, data = make_model().train(), inputs()
    data["q_embeddings"].requires_grad_(True)
    data["c_embeddings"].requires_grad_(True)
    before = model.global_branch.rank_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    result = model(**data)
    loss = result["rank_score"].square().mean() + torch.nn.functional.cross_entropy(result["app_logits"], torch.tensor([0, 2]))
    loss.backward()
    for parameter in (model.global_branch.rank_head.weight, model.typed_branch.rank_head.weight,
                      model.rank_gate_logit, model.app_gate_logit, model.global_branch.projection[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert data["q_embeddings"].grad is None
    assert data["c_embeddings"].grad is None
    optimizer.step()
    assert not torch.equal(before, model.global_branch.rank_head.weight)


def test_identical_rows_and_checkpoint_restore():
    model, data = make_model(), inputs()
    for value in data.values():
        value[1] = value[0]
    expected = model(**data)
    torch.testing.assert_close(expected["rank_score"][0], expected["rank_score"][1], rtol=0, atol=0)
    torch.testing.assert_close(expected["app_logits"][0], expected["app_logits"][1], rtol=0, atol=0)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = make_model()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert restored.typed_branch.projection is restored.global_branch.projection
    actual = restored(**data)
    torch.testing.assert_close(actual["rank_score"], expected["rank_score"], rtol=0, atol=0)
    torch.testing.assert_close(actual["app_logits"], expected["app_logits"], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"initial_gate": 0}, {"initial_gate": 1}, {"residual_scale": 0}, {"residual_scale": float("nan")}])
def test_invalid_residual_settings_fail(kwargs):
    with pytest.raises(ValueError):
        make_model(**kwargs)
