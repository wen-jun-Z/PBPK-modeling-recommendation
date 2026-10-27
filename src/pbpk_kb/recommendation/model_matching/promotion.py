from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROMOTION_VERSION = "pbpk-model-matcher-promotion-1.5.0"

PERTURBATION_PROMOTION_THRESHOLDS = {
    "top1_stability_rate": 0.95,
    "mean_topk_jaccard": 0.90,
    "mean_kendall_tau": 0.90,
    "reuse_layer_stability_rate": 0.99,
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def select_strongest_baseline(benchmark: dict[str, Any], statistics: dict[str, Any] | None):
    """Include every audited learned/auxiliary baseline, not just rule methods."""
    comparison = dict(benchmark.get("test_comparison") or {})
    if statistics is not None:
        if statistics.get("evaluation_scope") != "isolated_test_split":
            raise ValueError("Baseline statistics must use the isolated test split")
        for name, detail in statistics.get("methods", {}).items():
            if detail.get("role") in {"proposed_method", "ablation"}:
                continue
            value = detail.get("interval", {}).get("mean")
            if value is None:
                raise ValueError(f"Baseline primary metric missing: {name}")
            comparison[name] = {"ndcg@10": value, **detail.get("safety", {})}
    if not comparison:
        raise ValueError("No isolated-test baseline comparison is available")
    return max(comparison.items(), key=lambda item: float(item[1]["ndcg@10"]))


def evaluate_promotion(
    *,
    checkpoint: Path,
    seed_metrics: Path,
    benchmark_manifest: Path,
    feature_store: Path,
    perturbation_metrics: Path | None,
    output_directory: Path,
    method: str,
    suite_statistics_path: Path | None = None,
    checkpoint_metadata_path: Path | None = None,
    execution_audit_path: Path | None = None,
) -> dict[str, Any]:
    metrics = _read(seed_metrics).get("aggregate") or {}
    benchmark = _read(benchmark_manifest)
    feature_manifest = _read(feature_store / "manifest.json")
    perturbation = (
        _read(perturbation_metrics) if perturbation_metrics is not None else None
    )
    if checkpoint_metadata_path is None:
        import torch
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    else:
        audited = _read(checkpoint_metadata_path)
        if audited.get("checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError("Audited checkpoint metadata checksum mismatch")
        checkpoint_payload = audited["metadata"]
        archive_audit = checkpoint_payload.get("archive_audit") or {}
        if archive_audit.get("finite_weights_checked") is not True or int(archive_audit.get("tensor_count") or 0) <= 0:
            raise ValueError("Checkpoint metadata lacks a completed finite-weight archive audit")
    variant = checkpoint_payload.get("variant") or {}
    suite_statistics = _read(suite_statistics_path) if suite_statistics_path else None
    strongest_baseline_name, strongest_baseline_metrics = select_strongest_baseline(benchmark, suite_statistics)
    strongest_baseline = strongest_baseline_metrics
    blockers = []
    if suite_statistics is None:
        blockers.append("FULL_LEARNED_BASELINE_COMPARISON_MISSING")
    for field in ("ndcg@10", "false_reuse_rate", "false_adaptation_rate", "hard_constraint_violation_rate", "unsafe_top1_rate"):
        import math
        if metrics.get(field) is None or not math.isfinite(float(metrics[field])):
            blockers.append("REQUIRED_TEST_METRIC_MISSING_OR_NONFINITE:" + field)
    if metrics.get("reuse_adapt_availability_recall") is None or metrics.get("reuse_adapt_availability_recall") == 0:
        blockers.append("USEFUL_REUSE_ADAPT_RECALL_MISSING_OR_ZERO")
    if execution_audit_path is None:
        blockers.append("REAL_SIMULATION_SCOPE_AUDIT_MISSING")
    else:
        execution = _read(execution_audit_path)
        if execution.get("promotion_eligible") is not True:
            blockers.extend(execution.get("promotion_blockers") or ["REAL_SIMULATION_GATES_NOT_PASSED"])
    feature_file = feature_store / "node_embeddings.jsonl"
    if not feature_file.is_file() or _sha256(feature_file) != feature_manifest.get("node_embeddings_sha256"):
        blockers.append("FEATURE_STORE_CHECKSUM_NOT_VERIFIED")
    contract = (checkpoint_payload.get("resume_contract") or {}).get("run_inputs") or {}
    if contract.get("node_embeddings_sha256") != feature_manifest.get("node_embeddings_sha256"):
        blockers.append("CHECKPOINT_FEATURE_CONTRACT_NOT_VERIFIED")
    frozen_hashes = contract.get("benchmark_file_sha256") or {}
    if not frozen_hashes:
        blockers.append("CHECKPOINT_BENCHMARK_CONTRACT_MISSING")
    for filename, expected in frozen_hashes.items():
        path = benchmark_manifest.parent / filename
        if filename == "experiment_manifest.json" and benchmark.get("source_frozen_benchmark_manifest_sha256"):
            actual = benchmark["source_frozen_benchmark_manifest_sha256"]
        else:
            actual = _sha256(path) if path.is_file() else None
        if actual != expected:
            blockers.append("CHECKPOINT_BENCHMARK_CHECKSUM_MISMATCH:" + filename)
    if float(variant.get("uncertainty_weight", 0.1)) <= 0:
        blockers.append("UNCERTAINTY_HEAD_DISABLED")
    if float(metrics.get("ndcg@10") or 0.0) < float(
        strongest_baseline.get("ndcg@10") or 0.0
    ):
        blockers.append("DOES_NOT_OUTPERFORM_STRONGEST_BASELINE_NDCG10")
    if float(metrics.get("hard_constraint_violation_rate") or 0.0) > 0.0:
        blockers.append("HARD_CONSTRAINT_VIOLATION_NONZERO")
    if float(metrics.get("false_reuse_rate") or 0.0) > float(
        strongest_baseline.get("false_reuse_rate") or 0.0
    ):
        blockers.append("FALSE_REUSE_WORSE_THAN_STRONGEST_BASELINE")
    if float(metrics.get("false_adaptation_rate") or 0.0) > float(
        strongest_baseline.get("false_adaptation_rate") or 0.0
    ):
        blockers.append("FALSE_ADAPTATION_WORSE_THAN_STRONGEST_BASELINE")
    if float(metrics.get("unsafe_top1_rate") or 0.0) > float(
        strongest_baseline.get("unsafe_top1_rate") or 0.0
    ):
        blockers.append("UNSAFE_TOP1_WORSE_THAN_STRONGEST_BASELINE")
    benchmark_level = str(benchmark.get("benchmark_level") or "unknown")
    if benchmark_level != "execution_derived":
        blockers.append("BENCHMARK_IS_NOT_EXECUTION_DERIVED")
    if not bool(benchmark.get("scientific_claim_eligible")):
        blockers.append("BENCHMARK_NOT_SCIENTIFICALLY_ELIGIBLE")
    perturbed_query_count = int(benchmark.get("perturbed_query_count") or 0)
    if perturbed_query_count:
        if perturbation is None:
            blockers.append("PERTURBATION_METRICS_MISSING")
        else:
            for metric, threshold in PERTURBATION_PROMOTION_THRESHOLDS.items():
                value = perturbation.get(metric)
                if value is None or float(value) < threshold:
                    blockers.append(
                        f"PERTURBATION_{metric.upper()}_BELOW_THRESHOLD"
                    )
    architecture = str(variant.get("architecture") or "")
    required_variant_flags = (
        "use_hard_gate",
        "use_relation_typing",
        "use_pbpk_hierarchy",
        "use_validation_features",
        "use_provenance_features",
        "use_molecular_representation",
        "use_simulation_verification",
    )
    architecture_flags = {
        "cg_hhgm": ("use_cross_graph_attention",),
    }
    if architecture not in architecture_flags or any(
        not bool(variant.get(field, True)) for field in required_variant_flags
        + architecture_flags.get(architecture, ())
    ):
        blockers.append("CHECKPOINT_IS_NOT_FULL_PROPOSED_VARIANT")
    adaptation_supervision = bool(
        benchmark.get("adaptation_supervision_available")
    )
    if adaptation_supervision:
        if float(variant.get("adaptation_weight") or 0.0) <= 0:
            blockers.append("ADAPTATION_HEAD_DISABLED")
        if metrics.get("adaptation_cost_mae") is None:
            blockers.append("ADAPTATION_COST_METRIC_MISSING")
        if metrics.get("adaptation_operation_micro_f1") is None:
            blockers.append("ADAPTATION_OPERATION_METRIC_MISSING")
        baseline_cost = strongest_baseline.get("adaptation_cost_mae")
        if baseline_cost is not None and float(
            metrics.get("adaptation_cost_mae") or float("inf")
        ) > float(baseline_cost):
            blockers.append("ADAPTATION_COST_WORSE_THAN_STRONGEST_BASELINE")
    serving_scope = (
        "validated"
        if not blockers and benchmark_level == "execution_derived"
        and bool(benchmark.get("scientific_claim_eligible"))
        else "research_only"
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "promotion_version": PROMOTION_VERSION,
        "status": "accepted" if not blockers else "rejected",
        "method": method,
        "serving_scope": serving_scope,
        "benchmark_level": benchmark_level,
        "scientific_claim_eligible": bool(benchmark.get("scientific_claim_eligible")),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "feature_store_path": str(feature_store.resolve()),
        "feature_store_sha256": feature_manifest["node_embeddings_sha256"],
        "adaptation_supervision_available": adaptation_supervision,
        "adaptation_operation_types": checkpoint_payload.get(
            "adaptation_operation_types"
        ),
        "checkpoint_variant": variant,
        "seed_metrics_path": str(seed_metrics.resolve()),
        "test_metrics": metrics,
        "perturbation_metrics": perturbation,
        "perturbation_thresholds": PERTURBATION_PROMOTION_THRESHOLDS,
        "strongest_baseline": {
            "method": strongest_baseline_name,
            "metrics": strongest_baseline,
        },
        "baseline_comparison_scope": (
            "all_learned_deterministic_and_auxiliary_test_baselines" if suite_statistics else "deterministic_test_baselines_only_ineligible"
        ),
        "suite_statistics_path": str(suite_statistics_path) if suite_statistics_path else None,
        "suite_statistics_sha256": _sha256(suite_statistics_path) if suite_statistics_path else None,
        "execution_audit_sha256": _sha256(execution_audit_path) if execution_audit_path else None,
        "execution_audit_path": str(execution_audit_path) if execution_audit_path else None,
        "blockers": sorted(set(blockers)),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "serving_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply non-overridable acceptance gates to a matcher checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-metrics", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--perturbation-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="CG-HHGM")
    parser.add_argument("--suite-statistics", type=Path)
    parser.add_argument("--checkpoint-metadata", type=Path)
    parser.add_argument("--execution-audit", type=Path)
    args = parser.parse_args()
    result = evaluate_promotion(
        checkpoint=args.checkpoint,
        seed_metrics=args.seed_metrics,
        benchmark_manifest=args.benchmark_manifest,
        feature_store=args.feature_store,
        perturbation_metrics=args.perturbation_metrics,
        output_directory=args.output,
        method=args.method,
        suite_statistics_path=args.suite_statistics,
        checkpoint_metadata_path=args.checkpoint_metadata,
        execution_audit_path=args.execution_audit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
