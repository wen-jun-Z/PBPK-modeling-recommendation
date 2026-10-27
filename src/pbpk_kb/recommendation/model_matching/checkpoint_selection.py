from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbpk_kb.recommendation.model_matching.benchmark import _atomic_json


SELECTION_VERSION = "pbpk-model-matcher-validation-selection-1.1.0"


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["validation_joint_score"]),
        float(row["validation_ndcg@10"]),
        float(row["validation_raw_decision_macro_f1"]),
        -int(row["epoch"]),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_checkpoint(
    *, variant_directory: Path, output_directory: Path
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for history_path in sorted(variant_directory.glob("seed_*_history.json")):
        seed = int(history_path.stem.removeprefix("seed_").removesuffix("_history"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history:
            raise ValueError(f"Empty training history: {history_path}")
        best = max(history, key=_selection_key)
        checkpoint = variant_directory / f"seed_{seed}.pt"
        metrics = variant_directory / f"seed_{seed}_metrics.json"
        if not checkpoint.is_file() or not metrics.is_file():
            raise FileNotFoundError(
                f"Incomplete checkpoint artifacts for seed {seed}: {variant_directory}"
            )
        candidates.append(
            {
                "seed": seed,
                "best_validation_joint_score": float(
                    best["validation_joint_score"]
                ),
                "best_validation_ndcg@10": float(best["validation_ndcg@10"]),
                "best_validation_raw_decision_macro_f1": float(
                    best["validation_raw_decision_macro_f1"]
                ),
                "selected_epoch": int(best["epoch"]),
                "history_path": str(history_path.resolve()),
                "history_sha256": _sha256(history_path),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "metrics_path": str(metrics.resolve()),
                "metrics_sha256": _sha256(metrics),
            }
        )
    if not candidates:
        raise FileNotFoundError(f"No completed seed histories in {variant_directory}")
    selected = max(
        candidates,
        key=lambda row: (
            row["best_validation_joint_score"],
            row["best_validation_ndcg@10"],
            row["best_validation_raw_decision_macro_f1"],
            -row["seed"],
        ),
    )
    result = {
        "selection_version": SELECTION_VERSION,
        "selection_policy": (
            "highest_harmonic_mean_validation_ndcg10_and_raw_decision_macro_f1_"
            "then_ndcg_then_macro_f1_then_lowest_seed"
        ),
        "test_metrics_used_for_selection": False,
        "variant_directory": str(variant_directory.resolve()),
        "candidate_count": len(candidates),
        "candidates": sorted(candidates, key=lambda row: row["seed"]),
        "selected": selected,
        "created_at": datetime.now(UTC).isoformat(),
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_directory / "selection_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a matcher checkpoint using validation metrics only."
    )
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_checkpoint(
        variant_directory=args.variant,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
