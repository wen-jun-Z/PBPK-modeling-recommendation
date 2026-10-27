from __future__ import annotations

import pytest

from pbpk_kb.recommendation.gnn_index import _selected_run, _subgraph_vector


def test_selected_run_uses_validation_not_test() -> None:
    comparison = {
        "selected_architecture": "hgt",
        "architectures": {
            "hgt": {
                "runs": [
                    {
                        "seed": 17,
                        "validation": {"overall": {"composite": 0.9}},
                        "test": {"overall": {"composite": 0.1}},
                    },
                    {
                        "seed": 29,
                        "validation": {"overall": {"composite": 0.8}},
                        "test": {"overall": {"composite": 1.0}},
                    },
                ]
            }
        },
    }
    assert _selected_run(comparison)["seed"] == 17


def test_subgraph_vector_uses_center_conditioned_attention() -> None:
    torch = pytest.importorskip("torch")
    embeddings = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ]
    )
    vector, attention = _subgraph_vector(
        node_embeddings=embeddings,
        member_indices=torch.tensor([0, 1]),
        center_index=0,
    )
    assert float(attention[0]) > float(attention[1])
    assert torch.allclose(torch.linalg.vector_norm(vector), torch.tensor(1.0))
    simple_mean = torch.nn.functional.normalize(
        embeddings.mean(dim=0), dim=0
    )
    assert not torch.allclose(vector, simple_mean)
