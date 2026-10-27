from __future__ import annotations

import pytest

from pbpk_kb.recommendation.soft_graph_prompt import (
    _aggregate_set_metrics,
    _alignment_loss,
    _decode_and_validate_handles,
    _f_score,
    _masked_soft_tokens,
    _target_only_causal_loss,
    _target_loss_weights,
    build_projector,
)


def test_canonical_output_id_is_normalized_to_unique_allowed_handle() -> None:
    row = {
        "output_handle_map": {
            "component_ids": {"C01": "metabolism"},
            "evidence_ids": {"E01": "evidence-1"},
            "public_model_ids": {},
            "gap_codes": {"G01": "missing_component:metabolism"},
        }
    }
    selection, errors, normalizations = _decode_and_validate_handles(
        (
            '{"component_ids":["C01"],"evidence_ids":["E01"],'
            '"public_model_ids":[],"gap_codes":'
            '["missing_component:metabolism"]}'
        ),
        row,
    )

    assert errors == []
    assert selection is not None
    assert selection["gap_codes"] == ["G01"]
    assert normalizations == [
        {
            "field": "gap_codes",
            "from": "missing_component:metabolism",
            "to": "G01",
            "method": "unique_allowed_canonical_id_to_handle",
        }
    ]


def test_unknown_canonical_output_id_remains_invalid() -> None:
    row = {
        "output_handle_map": {
            "component_ids": {},
            "evidence_ids": {},
            "public_model_ids": {},
            "gap_codes": {"G01": "missing_component:metabolism"},
        }
    }
    selection, errors, normalizations = _decode_and_validate_handles(
        (
            '{"component_ids":[],"evidence_ids":[],"public_model_ids":[],'
            '"gap_codes":["missing_component:distribution"]}'
        ),
        row,
    )

    assert selection is not None
    assert errors == [
        "unsupported_gap_codes:missing_component:distribution"
    ]
    assert normalizations == []


def test_set_metrics_use_micro_counts_and_keep_empty_exact_matches() -> None:
    rows = [
        {"metrics": {"gap_codes": _f_score(set(), set())}},
        {"metrics": {"gap_codes": _f_score({"G01"}, {"G01"})}},
        {"metrics": {"gap_codes": _f_score({"G02"}, {"G01"})}},
    ]

    metrics = _aggregate_set_metrics(rows, "gap_codes")

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["exact_set_accuracy"] == pytest.approx(2 / 3)
    assert metrics["empty_target_specificity"] == 1.0


def test_soft_graph_projector_emits_global_and_local_tokens() -> None:
    torch = pytest.importorskip("torch")
    projector = build_projector(
        input_dimension=8,
        hidden_dimension=16,
        output_dimension=32,
    )
    vectors = torch.randn(2, 4, 8)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    projected, attention = projector(vectors, mask)

    assert projected.shape == (2, 5, 32)
    assert attention.shape == (2, 4)
    assert torch.allclose(attention.sum(dim=1), torch.ones(2))
    assert attention[0, 2:].sum() == 0


def test_alignment_loss_backpropagates_to_projector() -> None:
    torch = pytest.importorskip("torch")
    projector = build_projector(
        input_dimension=8,
        hidden_dimension=16,
        output_dimension=32,
    )
    vectors = torch.randn(2, 4, 8)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    target = torch.randn(2, 4, 32)
    projected, _ = projector(vectors, mask)
    loss = _alignment_loss(projected, target, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in projector.parameters()
    )


def test_masked_soft_tokens_use_model_dtype_and_zero_padding() -> None:
    torch = pytest.importorskip("torch")
    projector = build_projector(
        input_dimension=8,
        hidden_dimension=16,
        output_dimension=32,
    )
    vectors = torch.randn(4, 8)
    mask = torch.tensor([True, True, False, False])

    tokens = _masked_soft_tokens(
        projector,
        vectors,
        mask,
        dtype=torch.bfloat16,
    )

    assert tokens.shape == (5, 32)
    assert tokens.dtype == torch.bfloat16
    assert torch.count_nonzero(tokens[3:]) == 0


def test_target_only_loss_selects_exact_autoregressive_positions() -> None:
    torch = pytest.importorskip("torch")

    class DummyModel:
        class Config:
            vocab_size = 7

        config = Config()

        def __call__(
            self,
            *,
            inputs_embeds,
            attention_mask,
            use_cache,
            logits_to_keep,
        ):
            del attention_mask, use_cache
            return type(
                "Output",
                (),
                {"logits": inputs_embeds[:, logits_to_keep, :7]},
            )()

    inputs = torch.randn(1, 8, 7)
    target_ids = torch.tensor([[1, 3, 5]])
    loss, output = _target_only_causal_loss(
        model=DummyModel(),
        inputs=inputs,
        attention_mask=torch.ones(1, 8, dtype=torch.long),
        target_ids=target_ids,
    )
    expected_logits = inputs[:, 4:7, :]
    expected = torch.nn.functional.cross_entropy(
        expected_logits.reshape(-1, 7),
        target_ids.reshape(-1),
    )

    assert output.logits.shape == (1, 3, 7)
    assert torch.allclose(loss, expected)


def test_gap_target_tokens_receive_locked_weight() -> None:
    torch = pytest.importorskip("torch")
    target = (
        '{"component_ids":[],"evidence_ids":[],"gap_codes":["G01"],'
        '"public_model_ids":[]}'
    )

    class CharacterTokenizer:
        def __call__(self, value, **kwargs):
            assert value == target
            assert kwargs["return_offsets_mapping"] is True
            return {
                "offset_mapping": [
                    (index, index + 1) for index in range(len(value))
                ]
            }

    target_ids = torch.zeros((1, len(target) + 1), dtype=torch.long)
    weights = _target_loss_weights(
        tokenizer=CharacterTokenizer(),
        target_json=target,
        target_ids=target_ids,
        device=torch.device("cpu"),
    )[0]
    start = target.index('"gap_codes"')
    end = target.index("]", start) + 1

    assert torch.all(weights[start:end] == 4.0)
    assert torch.all(weights[:start] == 1.0)
    assert torch.all(weights[end:] == 1.0)
