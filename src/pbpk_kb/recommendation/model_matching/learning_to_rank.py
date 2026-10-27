from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from pbpk_kb.recommendation.model_matching.benchmark import _atomic_json, _atomic_jsonl
from pbpk_kb.recommendation.model_matching.metrics import (
    METADATA_RELEVANCE_SEMANTICS,
    evaluate_metadata_rankings,
    evaluate_rankings,
)
from pbpk_kb.recommendation.model_matching.scoring import (
    assess_candidate_use,
    cross_graph_attention_score,
    domain_ged_score,
    score_candidate,
)
from pbpk_kb.recommendation.model_matching.training import (
    _load_benchmark,
    audited_candidate_ids,
    load_benchmark_metadata,
)


LTR_VERSION = "pbpk-lambdamart-baseline-1.3.0"

METADATA_FEATURE_NAMES = (
    "identity_coverage", "context_coverage", "mechanism_coverage", "structure_coverage",
    "cross_graph_attention", "domain_graph_edit_similarity",
    "query_node_count", "model_node_count", "query_edge_count", "model_edge_count",
    "semantic_match_fraction", "semantic_conflict_fraction",
)


def _features(query_graph: Any, model_graph: Any, gate_graph: Any) -> list[float]:
    match = score_candidate(query_graph, gate_graph)
    assessment = assess_candidate_use(query_graph, gate_graph, match=match)
    ged, _, _ = domain_ged_score(query_graph, model_graph)
    hierarchy = match.hierarchy_scores
    return [
        float(assessment.direct_reuse_gate_passed),
        *(float(hierarchy.get(level, 0.0)) for level in (
            "identity", "context", "mechanism", "structure", "validation"
        )),
        float(cross_graph_attention_score(query_graph, model_graph)),
        float(ged),
        float(match.adaptation_cost),
        float(match.uncertainty),
        float(len(query_graph.nodes)),
        float(len(model_graph.nodes)),
    ]


def _metadata_features(query_graph: Any, model_graph: Any) -> list[float]:
    """Observable graph compatibility only; no label or scientific-use target."""
    match = score_candidate(query_graph, model_graph)
    ged, _, _ = domain_ged_score(query_graph, model_graph)
    matched = len(match.matched_semantic_keys)
    missing = len(match.missing_semantic_keys)
    conflicting = len(match.conflicting_semantic_keys)
    return [
        *(float(match.hierarchy_scores.get(level, 0.0)) for level in (
            "identity", "context", "mechanism", "structure"
        )),
        float(cross_graph_attention_score(query_graph, model_graph)),
        float(ged),
        float(len(query_graph.nodes)),
        float(len(model_graph.nodes)),
        float(len(query_graph.edges)),
        float(len(model_graph.edges)),
        matched / max(1, matched + missing),
        conflicting / max(1, matched + missing + conflicting),
    ]


def candidate_ids_for_dataset_split(
    model_graphs: Mapping[str, Any], split: str
) -> list[str]:
    allowed_splits = {
        "train": {"train"},
        "validation": {"train", "validation"},
        "test": {"train", "validation", "test"},
    }
    if split not in allowed_splits:
        raise ValueError(f"Unsupported dataset split: {split}")
    return audited_candidate_ids(
        model_graphs,
        allowed_splits=allowed_splits[split],
    )


def run_lambdamart(
    *,
    benchmark_directory: Path,
    output_directory: Path,
    seed: int,
    device: str = "cpu",
) -> dict[str, Any]:
    from xgboost import XGBRanker

    if device != "cpu" and device != "cuda" and not (
        device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
    ):
        raise ValueError("LambdaMART device must be cpu, cuda, or cuda:<ordinal>.")
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite LambdaMART results: {output_directory}")
    queries, query_graphs, model_graphs, model_gate_graphs, labels = _load_benchmark(
        benchmark_directory
    )
    benchmark_metadata = load_benchmark_metadata(benchmark_directory)
    metadata_mode = benchmark_metadata.get("supervision_mode") == "metadata_ranking"
    if metadata_mode and benchmark_metadata.get("relevance_label_semantics") != METADATA_RELEVANCE_SEMANTICS:
        raise ValueError("Metadata LambdaMART requires explicit compatibility label semantics.")
    query_by_split = {
        split: [query for query in queries if query["split"] == split]
        for split in ("train", "validation", "test")
    }
    candidate_ids_by_split = {
        split: candidate_ids_for_dataset_split(model_graphs, split)
        for split in ("train", "validation", "test")
    }

    def dataset(split: str) -> tuple[list[list[float]], list[int | None], list[int]]:
        rows: list[list[float]] = []
        targets: list[int | None] = []
        qids: list[int] = []
        for group_id, query in enumerate(
            sorted(query_by_split[split], key=lambda value: value["query_id"])
        ):
            query_id = str(query["query_id"])
            for model_id in candidate_ids_by_split[split]:
                if metadata_mode and split != "test" and (query_id, model_id) not in labels:
                    # An unjudged pair is absent supervision, never grade zero.
                    continue
                rows.append(
                    _metadata_features(query_graphs[query_id], model_graphs[model_id])
                    if metadata_mode else _features(
                        query_graphs[query_id],
                        model_graphs[model_id],
                        model_gate_graphs[model_id],
                    )
                )
                targets.append(labels.get((query_id, model_id)) if metadata_mode else labels[(query_id, model_id)])
                qids.append(group_id)
        if not rows:
            raise RuntimeError(
                f"LambdaMART {split} dataset is empty after audited candidate "
                "split filtering."
            )
        return rows, targets, qids

    train_x, train_y, train_qid = dataset("train")
    validation_x, validation_y, validation_qid = dataset("validation")
    test_x, test_y, test_qid = dataset("test")
    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        tree_method="hist",
        device=device,
        n_jobs=4,
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
    )
    model.fit(
        train_x,
        train_y,
        qid=train_qid,
        eval_set=[(validation_x, validation_y)],
        eval_qid=[validation_qid],
        verbose=False,
    )
    actual_device = str(json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"])
    if device.startswith("cuda") and not actual_device.startswith("cuda"):
        raise RuntimeError(f"Requested CUDA LambdaMART but XGBoost used {actual_device}; refusing silent CPU fallback.")
    scores = model.predict(test_x)
    candidate_ids = candidate_ids_by_split["test"]
    predictions = []
    cursor = 0
    for query in sorted(query_by_split["test"], key=lambda value: value["query_id"]):
        query_id = str(query["query_id"])
        for candidate_id in candidate_ids:
            raw_score = float(scores[cursor])
            score = raw_score
            cursor += 1
            if metadata_mode:
                predictions.append({
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "score": score,
                    "raw_score": raw_score,
                    "metadata_relevance": labels.get((query_id, candidate_id)),
                    "compatibility_label_known": (query_id, candidate_id) in labels,
                    "relevance_label_semantics": METADATA_RELEVANCE_SEMANTICS,
                    "scientific_use_label_known": False,
                    "use_label": None,
                })
                continue
            deterministic = score_candidate(
                query_graphs[query_id], model_gate_graphs[candidate_id]
            )
            assessment = assess_candidate_use(
                query_graphs[query_id],
                model_gate_graphs[candidate_id],
                match=deterministic,
            )
            reuse_decision = bool(
                assessment.direct_reuse_gate_passed and score >= 0.5
            )
            predictions.append(
                {
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "score": score,
                    "raw_score": raw_score,
                    "relevance_grade": labels[(query_id, candidate_id)],
                    "hard_gate_passed": (
                        assessment.direct_reuse_gate_passed
                    ),
                    "structural_gate_passed": (
                        assessment.structural_gate_passed
                    ),
                    "maximum_use_tier": assessment.maximum_use_tier,
                    "predicted_use_tier": (
                        "direct_reuse"
                        if reuse_decision
                        else "reject"
                        if assessment.maximum_use_tier == "reject"
                        else "mechanistic_reference"
                    ),
                    "reuse_decision": reuse_decision,
                    "adaptation_decision": False,
                }
            )
    output_directory.mkdir(parents=True, exist_ok=False)
    prediction_checksum = _atomic_jsonl(
        output_directory / "predictions_lambdamart.jsonl", predictions
    )
    metrics = evaluate_metadata_rankings(predictions) if metadata_mode else evaluate_rankings(predictions)
    metrics_checksum = _atomic_json(output_directory / "metrics_lambdamart.json", metrics)
    model.save_model(output_directory / "lambdamart.ubj")
    manifest = {
        "method": "LambdaMART/XGBoost Ranker",
        "version": LTR_VERSION,
        **benchmark_metadata,
        "seed": seed,
        "device": device,
        "actual_device": actual_device,
        "feature_count": 12,
        "qid_policy": "zero-based stable integer per lexicographically sorted query_id",
        "candidate_pool_policy": (
            "audited_candidate_split_train_then_train_plus_validation_then_all"
        ),
        "candidate_counts": {
            split: len(values)
            for split, values in candidate_ids_by_split.items()
        },
        "train_pair_count": len(train_x),
        "validation_pair_count": len(validation_x),
        "test_pair_count": len(test_x),
        "test_metrics": metrics["aggregate"],
        **({
            "metrics_scope": METADATA_RELEVANCE_SEMANTICS,
            "scientific_claim_eligible": False,
            "feature_names": list(METADATA_FEATURE_NAMES),
            "interpretation": "Weak-supervision fitting and grouped generalization of documented compatibility. Rule-generated labels do not independently establish scientific recommendation quality.",
        } if metadata_mode else {}),
        "checksums": {
            "predictions_lambdamart.jsonl": prediction_checksum,
            "metrics_lambdamart.json": metrics_checksum,
        },
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LambdaMART PBPK baseline.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<ordinal>")
    args = parser.parse_args()
    result = run_lambdamart(
        benchmark_directory=args.benchmark,
        output_directory=args.output,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
