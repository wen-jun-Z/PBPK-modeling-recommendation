from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbpk_kb.recommendation.model_matching.benchmark import _atomic_json


MERGE_VERSION = "pbpk-model-matcher-parallel-grid-merge-2.0.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _load_manifest(source: Path) -> tuple[Path, dict[str, Any]]:
    path = source / "training_manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _base_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    architectures = list(manifest.get("architectures") or [])
    if not architectures:
        raise ValueError("Architectures are required for parallel merge")
    if manifest.get("variants") and all(row.get("architecture") == "gmn" for row in manifest["variants"]):
        architectures = ["gmn"]
    return {
        "trainer_version": manifest.get("trainer_version"),
        "benchmark_directory": manifest.get("benchmark_directory"),
        "benchmark_level": manifest.get("benchmark_level"),
        "architectures": architectures,
        "device": manifest.get("device"),
        "epochs": manifest.get("epochs"),
        "candidates_per_query": manifest.get("candidates_per_query"),
        "learning_rate": manifest.get("learning_rate"),
        "node_embedding_sha256": (
            manifest.get("node_embedding_manifest") or {}
        ).get("node_embeddings_sha256"),
        "feature_dimensions": manifest.get("feature_dimensions"),
        "run_input_contract": manifest.get("run_input_contract"),
        "decision_weighting_policy": manifest.get("decision_weighting_policy"),
        "validation_selection_policy": manifest.get(
            "validation_selection_policy"
        ),
        "perturbation_evaluation_available": manifest.get(
            "perturbation_evaluation_available"
        ),
    }


def merge_parallel_seed_runs(
    *,
    source_directories: list[Path],
    target_directory: Path,
    expected_seeds: list[int] | None = None,
    expected_variants: list[str] | None = None,
) -> dict[str, Any]:
    if not source_directories:
        raise ValueError("At least one parallel training directory is required")
    target_directory.mkdir(parents=True, exist_ok=True)
    manifest_directory = target_directory / "parallel_training_manifests"
    manifest_directory.mkdir(exist_ok=True)
    reference_contract: dict[str, Any] | None = None
    reference_manifest: dict[str, Any] | None = None
    variant_definitions: dict[str, dict[str, Any]] = {}
    variant_order: list[str] = []
    merged: list[dict[str, Any]] = []
    merged_results: dict[str, list[dict[str, Any]]] = {}
    source_manifests: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, int]] = set()
    seen_seeds: set[int] = set()

    for source in source_directories:
        manifest_path, manifest = _load_manifest(source)
        source_manifests.append(manifest)
        seeds = list(manifest.get("seeds") or [])
        variants = list(manifest.get("variants") or [])
        if len(seeds) != 1:
            raise ValueError(f"Expected one seed in {manifest_path}")
        if not variants:
            raise ValueError(f"Expected at least one variant in {manifest_path}")
        seed = int(seeds[0])
        seen_seeds.add(seed)

        contract = _base_contract(manifest)
        if reference_contract is None:
            reference_contract = contract
            reference_manifest = manifest
        elif contract != reference_contract:
            raise ValueError(
                f"Parallel training contract mismatch in {manifest_path}: {contract}"
            )

        copied_files = []
        for variant in variants:
            name = str(variant["name"])
            existing_definition = variant_definitions.get(name)
            if existing_definition is not None and existing_definition != variant:
                raise ValueError(f"Variant definition drift for {name}")
            if existing_definition is None:
                variant_definitions[name] = variant
                variant_order.append(name)
            pair = (name, seed)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate parallel variant/seed pair: {pair}")
            seen_pairs.add(pair)

            result_rows = list((manifest.get("results") or {}).get(name) or [])
            if len(result_rows) != 1 or int(result_rows[0].get("seed", -1)) != seed:
                raise ValueError(
                    f"Expected one result for {name} seed {seed} in {manifest_path}"
                )
            merged_results.setdefault(name, []).append(result_rows[0])
            source_variant = source / name
            target_variant = target_directory / name
            target_variant.mkdir(exist_ok=True)
            suffixes = [".pt", "_history.json", "_metrics.json", "_predictions.jsonl"]
            if manifest.get("perturbation_evaluation_available"):
                suffixes.extend(
                    (
                        "_perturbation_predictions.jsonl",
                        "_perturbation_metrics.json",
                    )
                )
            for suffix in suffixes:
                source_path = source_variant / f"seed_{seed}{suffix}"
                target_path = target_variant / source_path.name
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                if target_path.exists():
                    if _sha256(target_path) != _sha256(source_path):
                        raise ValueError(f"Conflicting training artifact: {target_path}")
                else:
                    _link_or_copy(source_path, target_path)
                copied_files.append(
                    {"path": str(target_path), "sha256": _sha256(target_path)}
                )

        manifest_sha256 = _sha256(manifest_path)
        copied_manifest = manifest_directory / (
            f"seed_{seed}_{manifest_sha256[:12]}_training_manifest.json"
        )
        if not copied_manifest.exists():
            shutil.copy2(manifest_path, copied_manifest)
        merged.append(
            {
                "seed": seed,
                "variants": [str(variant["name"]) for variant in variants],
                "source_directory": str(source.resolve()),
                "source_manifest_sha256": manifest_sha256,
                "artifact_count": len(copied_files),
                "artifacts": copied_files,
            }
        )

    actual_seeds = sorted(seen_seeds)
    required_seeds = sorted(set(expected_seeds or actual_seeds))
    actual_variants = set(variant_definitions)
    required_variant_order = list(expected_variants or variant_order)
    if len(required_variant_order) != len(set(required_variant_order)):
        raise ValueError("Expected variant inventory contains duplicates")
    required_variants = set(required_variant_order)
    if actual_seeds != required_seeds or actual_variants != required_variants:
        raise ValueError(
            "Parallel inventory mismatch: "
            f"expected_seeds={required_seeds}, actual_seeds={actual_seeds}, "
            f"expected_variants={sorted(required_variants)}, "
            f"actual_variants={sorted(actual_variants)}"
        )
    required_pairs = {
        (variant, seed)
        for variant in required_variants
        for seed in required_seeds
    }
    if seen_pairs != required_pairs:
        missing = sorted(required_pairs - seen_pairs)
        extra = sorted(seen_pairs - required_pairs)
        raise ValueError(
            f"Parallel variant/seed grid mismatch: missing={missing}, extra={extra}"
        )

    assert reference_manifest is not None
    merged_training_manifest = {
        **reference_manifest,
        "architectures": reference_contract["architectures"],
        "variants": [variant_definitions[name] for name in required_variant_order],
        "requested_variant_names": required_variant_order,
        "seeds": actual_seeds,
        "results": {
            name: sorted(merged_results[name], key=lambda row: int(row["seed"]))
            for name in required_variant_order
        },
        "elapsed_seconds": sum(
            float(manifest.get("elapsed_seconds") or 0.0)
            for manifest in source_manifests
        ),
        "parallel_grid_merge": True,
        "parallel_grid_merge_version": MERGE_VERSION,
        "parallel_source_count": len(source_directories),
        "created_at": datetime.now(UTC).isoformat(),
        "resumed": any(bool(manifest.get("resumed")) for manifest in source_manifests),
    }
    _atomic_json(target_directory / "training_manifest.json", merged_training_manifest)
    result = {
        "merge_version": MERGE_VERSION,
        "contract": reference_contract,
        "merged_source_count": len(merged),
        "merged_seed_count": len(actual_seeds),
        "merged_variant_count": len(required_variants),
        "merged_pair_count": len(seen_pairs),
        "expected_seeds": required_seeds,
        "expected_variants": required_variant_order,
        "merged": sorted(
            merged,
            key=lambda row: (int(row["seed"]), tuple(row["variants"])),
        ),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(target_directory / "parallel_seed_merge_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checksum-merge isolated model-matcher training runs."
    )
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-seed", action="append", type=int)
    parser.add_argument("--expected-variant", action="append")
    args = parser.parse_args()
    result = merge_parallel_seed_runs(
        source_directories=args.source,
        target_directory=args.target,
        expected_seeds=args.expected_seed,
        expected_variants=args.expected_variant,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
