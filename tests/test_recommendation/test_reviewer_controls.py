"""Control isolation, same-loss unknown masking, and outcome-free inference."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from reviewer_controls_models import ControlMatch, METHODS, trainer, identity_condition_rule, visible_graph
from pbpk_kb.recommendation.pbpk_match_v2.data import feature_names, pair_feature_row


@pytest.fixture(autouse=True)
def thread_limit():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def synthetic():
    rng = np.random.default_rng(81)
    return {"query_records": [{"query_id": f"q{i}", "compound_id": f"drug{i}", "compounds": [f"drug{i}"]} for i in range(7)],
            "candidate_records": [{"candidate_id": f"c{i}", "family_id": f"family{i // 2}", "compounds": ["drug0"]} for i in range(5)],
            "light_query_embeddings": rng.normal(size=(3, 7, 8)).astype(np.float32),
            "light_candidate_embeddings": rng.normal(size=(5, 8)).astype(np.float32),
            "light_embedding_manifest": {"revision": "synthetic"},
            "pair_features_by_variant": rng.normal(size=(3, 7, 5, 4)).astype(np.float32),
            "pair_feature_names": [f"f{i}" for i in range(4)], "variant_names": ["original", "reordered", "missing_evidence"],
            "grades": np.tile([3, 2, 1, 0, -1], (7, 1)), "applicability": np.tile([0, 0, 2, 1, -1], (7, 1)),
            "groups": np.arange(7), "error_records": []}


def test_restricted_inputs_cannot_affect_excluded_controls():
    q, c, f = torch.randn(8), torch.randn(5, 8), torch.randn(5, 4)
    model = ControlMatch(8, 4, dropout=0, mode="structured_mlp").eval()
    torch.testing.assert_close(model(q, c, f)["scores"], model(q + 20, -c, f)["scores"], rtol=0, atol=0)
    model = ControlMatch(8, 4, dropout=0, mode="text_mlp").eval()
    torch.testing.assert_close(model(q, c, f)["scores"], model(q, c, f + 100)["scores"], rtol=0, atol=0)
    model = ControlMatch(8, 4, dropout=0, mode="concat_mlp").eval()
    assert not torch.equal(model(q, c, f)["scores"], model(q, c, f + 100)["scores"])


@pytest.mark.parametrize("mode", METHODS)
def test_same_trainer_masks_unknown_and_never_uses_outer_labels(tmp_path, mode):
    module, data = trainer(mode), synthetic()
    module.ROOT = tmp_path
    data["grades"][6] = 99
    data["applicability"][6] = 99
    metadata = module.train_fold(data, [0, 1, 2], [], tmp_path / mode, epochs=2, pk_epochs=1, error_train_indices=[])
    assert metadata["error_train_indices"] == []
    assert metadata["trainable_text_encoder"] is False
    for key in ("grades", "applicability", "error_records"):
        data.pop(key)
    output = module.predict_fold(data, [6], tmp_path / mode)
    assert output["scores"].shape == (1, 5)
    assert not output["error_supported"].any()
    np.testing.assert_allclose(output["probabilities"].sum(-1), 1, atol=1e-6)


def test_group_overlap_and_outer_error_supervision_are_rejected(tmp_path):
    module, data = trainer("concat_mlp"), synthetic()
    module.ROOT = tmp_path
    data["groups"][3] = data["groups"][0]
    with pytest.raises(ValueError, match="grouped"):
        module.train_fold(data, [0, 1, 2], [3], tmp_path / "bad")
    with pytest.raises(ValueError, match="belong"):
        module.train_fold(data, [0, 1, 2], [], tmp_path / "bad", error_train_indices=[6])


def test_rule_uses_no_label_or_outcome_keys():
    q = {"query_id": "q", "text": "drug a oral human", "features": {"compounds": ["drug a"], "routes": ["oral"], "species": ["human"]}}
    a = {"candidate_id": "a", "text": "drug a oral human", "features": {"compounds": ["drug a"], "routes": ["oral"], "species": ["human"]}}
    b = {"candidate_id": "b", "text": "drug b intravenous human", "features": {"compounds": ["drug b"], "routes": ["intravenous"], "species": ["human"]}}
    scores = identity_condition_rule(np.array([pair_feature_row(q, a), pair_feature_row(q, b)]), feature_names())
    assert scores[0] > scores[1]
    before = visible_graph(a, candidate=True)
    a["grade"] = 0; a["observed"] = 999
    assert before == visible_graph(a, candidate=True)
