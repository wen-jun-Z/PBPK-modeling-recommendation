"""Input invariants for the frozen text cache, without loading a language model."""
import importlib.util
from pathlib import Path

import numpy as np


_PATH = Path(__file__).resolve().parents[2] / "scripts/pbpk_match_light_inputs.py"
_SPEC = importlib.util.spec_from_file_location("light_inputs_test", _PATH)
light = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(light)


def _data():
    features = np.zeros((3, 2, 2, 204), np.float32)
    features[:, 0, :, 60] = np.log1p(10)
    features[:, 1, :, 60] = np.log1p(20)
    return {"query_texts": [["dose 10 mg", "dose 20 mg"], ["10 mg dose", "20 mg dose"],
                           ["dose 10 mg", "dose 20 mg"]],
            "document_texts": ["same candidate", "same candidate"],
            "condition_features": features,
            "grades": np.ones((2, 2)), "error_records": [{"observed": 1234}]}


def test_unique_text_cache_preserves_condition_and_variant_distinctions():
    data = _data()
    texts = light.input_texts(data)
    assert len(texts) == 5
    vectors = np.arange(15, dtype=np.float16).reshape(5, 3)
    result = light.assemble(data, texts, vectors)
    assert result["raw_query_embeddings"].shape == (3, 2, 3)
    assert not np.array_equal(result["raw_query_embeddings"][0, 0], result["raw_query_embeddings"][0, 1])
    assert not np.array_equal(result["raw_query_embeddings"][0], result["raw_query_embeddings"][1])
    np.testing.assert_array_equal(result["raw_query_embeddings"][0], result["raw_query_embeddings"][2])
    assert data["condition_features"][0, 0, 0, 60] != data["condition_features"][0, 1, 0, 60]
    assert "pair_features" not in result  # existing frozen arrays are not duplicated


def test_duplicate_candidates_keep_exact_ties_and_targets_are_excluded():
    data = _data()
    texts = light.input_texts(data)
    vectors = np.arange(15, dtype=np.float16).reshape(5, 3)
    first = light.assemble(data, texts, vectors)
    data["grades"] *= 42
    data["error_records"] = [{"observed": -99}]
    second = light.assemble(data, texts, vectors)
    assert "grades" not in first and "error_records" not in first
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(first["raw_candidate_embeddings"][0], first["raw_candidate_embeddings"][1])


def test_existing_typed_node_cache_is_reused_only_on_exact_text_match():
    # Real historical files are verified read-only and no encoder is loaded.
    lookup, manifest = light.old_text_vectors()
    assert len(lookup) == manifest["text_count"] == 689
    assert all(v.shape == (1024,) and v.dtype == np.float16 for v in lookup.values())
    visible = light.visible_inputs()
    assert not {"grades", "applicability", "error_records", "folds", "groups"} & set(visible)
    assert len(light.input_texts(visible)) == 2166
