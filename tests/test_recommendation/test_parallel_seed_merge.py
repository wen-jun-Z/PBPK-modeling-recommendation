from __future__ import annotations

import json
from pathlib import Path

import pytest

from pbpk_kb.recommendation.model_matching.merge_parallel_seeds import (
    merge_parallel_seed_runs,
)


def _seed_run(root: Path, seed: int) -> Path:
    source = root / f"seed_{seed}"
    variant = source / "gmn"
    variant.mkdir(parents=True)
    result = {"seed": seed, "test": {"ndcg@10": seed / 100.0}}
    manifest = {
        "trainer_version": "trainer-test",
        "benchmark_directory": "/benchmark",
        "benchmark_level": "execution_derived",
        "architectures": ["gmn"],
        "variants": [{"name": "gmn", "architecture": "gmn"}],
        "seeds": [seed],
        "device": "cuda",
        "epochs": 40,
        "candidates_per_query": 16,
        "learning_rate": 2e-4,
        "node_embedding_manifest": {"node_embeddings_sha256": "a" * 64},
        "feature_dimensions": 1292,
        "run_input_contract": {"benchmark_file_sha256": {"labels": "b" * 64}},
        "decision_weighting_policy": "train-only",
        "validation_selection_policy": "validation-only",
        "perturbation_evaluation_available": True,
        "results": {"gmn": [result]},
        "elapsed_seconds": float(seed),
        "resumed": False,
    }
    (source / "training_manifest.json").write_text(json.dumps(manifest))
    for suffix in (
        ".pt",
        "_history.json",
        "_metrics.json",
        "_predictions.jsonl",
        "_perturbation_predictions.jsonl",
        "_perturbation_metrics.json",
    ):
        (variant / f"seed_{seed}{suffix}").write_text(f"{seed}:{suffix}")
    return source


def test_parallel_seed_merge_accepts_gmn_and_builds_training_manifest(
    tmp_path: Path,
) -> None:
    sources = [_seed_run(tmp_path / "sources", seed) for seed in (17, 29)]
    target = tmp_path / "merged"

    result = merge_parallel_seed_runs(
        source_directories=sources,
        target_directory=target,
        expected_seeds=[17, 29],
    )

    assert result["merged_seed_count"] == 2
    manifest = json.loads((target / "training_manifest.json").read_text())
    assert manifest["architectures"] == ["gmn"]
    assert manifest["seeds"] == [17, 29]
    assert [row["seed"] for row in manifest["results"]["gmn"]] == [17, 29]
    assert (target / "gmn" / "seed_17.pt").is_file()


def test_parallel_seed_merge_rejects_incomplete_seed_inventory(
    tmp_path: Path,
) -> None:
    source = _seed_run(tmp_path / "sources", 17)

    with pytest.raises(ValueError, match="inventory mismatch"):
        merge_parallel_seed_runs(
            source_directories=[source],
            target_directory=tmp_path / "merged",
            expected_seeds=[17, 29],
        )


def test_parallel_grid_merge_requires_every_variant_seed_pair(tmp_path: Path) -> None:
    sources = [_seed_run(tmp_path / "sources", seed) for seed in (17, 29)]

    with pytest.raises(ValueError, match="inventory mismatch"):
        merge_parallel_seed_runs(
            source_directories=sources,
            target_directory=tmp_path / "merged",
            expected_seeds=[17, 29],
            expected_variants=["gmn", "gmn_comparison_fixture"],
        )
