from __future__ import annotations

import pytest

from pbpk_kb.recommendation.model_matching.graphs import TypedGraph
from pbpk_kb.recommendation.model_matching.learning_to_rank import (
    candidate_ids_for_dataset_split,
)
from pbpk_kb.recommendation.model_matching.training import audited_candidate_ids


def _graph(model_id: str, split: str | None) -> TypedGraph:
    return TypedGraph(
        graph_id=model_id,
        graph_kind="executable_model",
        nodes=(),
        edges=(),
        metadata={"candidate_split": split},
    )


def test_candidate_pool_uses_audited_split_assignment() -> None:
    graphs = {
        "train-model": _graph("train-model", "train"),
        "validation-model": _graph("validation-model", "validation"),
        "test-model": _graph("test-model", "test"),
    }

    assert audited_candidate_ids(graphs, allowed_splits={"train"}) == [
        "train-model"
    ]
    assert audited_candidate_ids(
        graphs, allowed_splits={"train", "validation"}
    ) == ["train-model", "validation-model"]


def test_candidate_pool_rejects_missing_audited_split() -> None:
    with pytest.raises(RuntimeError, match="invalid audited candidate splits"):
        audited_candidate_ids(
            {"unknown": _graph("unknown", None)}, allowed_splits={"train"}
        )


def test_lambdamart_uses_audited_candidate_splits_without_family_strings() -> None:
    graphs = {
        "train-composite": _graph("train-composite", "train"),
        "validation-single": _graph("validation-single", "validation"),
        "test-composite": _graph("test-composite", "test"),
    }

    assert candidate_ids_for_dataset_split(graphs, "train") == [
        "train-composite"
    ]
    assert candidate_ids_for_dataset_split(graphs, "validation") == [
        "train-composite",
        "validation-single",
    ]
    assert candidate_ids_for_dataset_split(graphs, "test") == sorted(graphs)


def test_lambdamart_rejects_unknown_dataset_split() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset split"):
        candidate_ids_for_dataset_split({}, "holdout")
