from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from pbpk_kb.recommendation.model_matching.graphs import (
    TypedGraph,
    graph_from_dict,
)
from pbpk_kb.recommendation.model_matching.feature_store import load_embedding_store
from pbpk_kb.recommendation.model_matching.metrics import evaluate_rankings, evaluate_metadata_rankings
from pbpk_kb.recommendation.model_matching.perturbation_evaluation import (
    evaluate_ranking_continuity,
)
from pbpk_kb.recommendation.model_matching.neural import (
    AUXILIARY_NODE_FEATURE_DIMENSIONS,
    ADAPTATION_OPERATION_TYPES,
    METHOD_FIDELITY,
    build_matcher,
    tensorize_graph,
)
from pbpk_kb.recommendation.model_matching.scoring import (
    USE_TIER_ORDER,
    apply_use_tier_gate,
    assess_candidate_use,
    deterministic_use_class,
    score_candidate,
)


TRAINER_VERSION = "pbpk-neural-model-matcher-trainer-1.10.0"
DEFAULT_ARCHITECTURES = tuple(METHOD_FIDELITY)
ADAPTATION_COST_SCALE = 8.0
DECISION_CLASS_COUNT = 4
DECISION_WEIGHTING_POLICY = "train_only_effective_number_focal_loss"
DECISION_EFFECTIVE_NUMBER_BETA = 0.999
DECISION_FOCAL_GAMMA = 2.0
VALIDATION_SELECTION_POLICY = "harmonic_mean_ndcg10_raw_decision_macro_f1"
METADATA_VALIDATION_SELECTION_POLICY = "metadata_relevance_ndcg10_only"
METADATA_LABEL_SEMANTICS = "metadata_compatibility_not_validated_use"


def load_benchmark_metadata(benchmark_directory: Path) -> dict[str, Any]:
    manifest_path = benchmark_directory / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Benchmark manifest is required for claim-boundary propagation: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_level = str(manifest.get("benchmark_level") or "unknown")
    scientific_claim_eligible = bool(manifest.get("scientific_claim_eligible"))
    return {
        "benchmark_level": benchmark_level,
        "scientific_claim_eligible": scientific_claim_eligible,
        "scientific_claim_blockers": list(
            manifest.get("scientific_claim_blockers") or []
        ),
        "adaptation_supervision_available": bool(
            manifest.get("adaptation_supervision_available")
        ),
        "source_manifest": str(manifest_path.resolve()),
        "supervision_mode": str(manifest.get("supervision_mode") or "scientific_use"),
        "relevance_label_semantics": manifest.get("relevance_label_semantics"),
    }


@dataclass(frozen=True, slots=True)
class TrainingVariant:
    name: str
    architecture: str
    ranking_weight: float = 1.0
    adaptation_weight: float = 0.25
    decision_weight: float = 0.25
    uncertainty_weight: float = 0.10
    use_hard_gate: bool = True
    use_relation_typing: bool = True
    use_cross_graph_attention: bool = True
    use_pbpk_hierarchy: bool = True
    use_validation_features: bool = True
    use_provenance_features: bool = True
    use_molecular_representation: bool = True
    use_simulation_verification: bool = True
    # Serialized compatibility fields preserve existing GMN resume hashes.
    # They no longer enable any computation or model variant.
    use_optimal_transport: bool = True
    use_requirement_conditioning: bool = True
    use_late_molecular_fusion: bool = True
    use_strict_type_matching: bool = False
    supervision_mode: str = "scientific_use"


def audited_candidate_ids(
    model_graphs: Mapping[str, TypedGraph], *, allowed_splits: set[str]
) -> list[str]:
    valid_splits = {"train", "validation", "test"}
    invalid = {
        str(graph.metadata.get("candidate_split"))
        for graph in model_graphs.values()
        if graph.metadata.get("candidate_split") not in valid_splits
    }
    if invalid:
        raise RuntimeError(
            "Model graphs have missing or invalid audited candidate splits: "
            + ", ".join(sorted(invalid))
        )
    return sorted(
        model_id
        for model_id, graph in model_graphs.items()
        if graph.metadata.get("candidate_split") in allowed_splits
    )


def default_variants(architectures: Iterable[str]) -> list[TrainingVariant]:
    architectures = list(architectures)
    unknown = sorted(set(architectures) - set(METHOD_FIDELITY))
    if unknown:
        raise ValueError(f"Unknown model architectures: {unknown}")
    variants = [TrainingVariant(name=value, architecture=value) for value in architectures]
    if "cg_hhgm" in architectures:
        variants.extend(
            [
                TrainingVariant(
                    name="cg_hhgm_no_uncertainty_loss",
                    architecture="cg_hhgm",
                    uncertainty_weight=0.0,
                ),
                TrainingVariant(
                    name="cg_hhgm_no_hard_gate",
                    architecture="cg_hhgm",
                    use_hard_gate=False,
                ),
                TrainingVariant(
                    name="cg_hhgm_no_relation_typing",
                    architecture="cg_hhgm",
                    use_relation_typing=False,
                ),
                TrainingVariant(
                    name="cg_hhgm_no_cross_graph_attention",
                    architecture="cg_hhgm",
                    use_cross_graph_attention=False,
                ),
                TrainingVariant(
                    name="cg_hhgm_no_pbpk_hierarchy",
                    architecture="cg_hhgm",
                    use_pbpk_hierarchy=False,
                ),
                TrainingVariant(
                    name="cg_hhgm_no_validation_features",
                    architecture="cg_hhgm",
                    use_validation_features=False,
                ),
                TrainingVariant(
                    name="cg_hhgm_no_provenance_features",
                    architecture="cg_hhgm",
                    use_provenance_features=False,
                ),
                TrainingVariant(
                    name="cg_hhgm_no_adaptation_head_or_cost",
                    architecture="cg_hhgm",
                    adaptation_weight=0.0,
                ),
                TrainingVariant(
                    name="cg_hhgm_no_molecular_representation",
                    architecture="cg_hhgm",
                    use_molecular_representation=False,
                ),
                TrainingVariant(
                    name="cg_hhgm_no_simulation_verification",
                    architecture="cg_hhgm",
                    use_simulation_verification=False,
                ),
            ]
        )
    return variants


def select_variants(
    architectures: Iterable[str], variant_names: Iterable[str] | None = None
) -> list[TrainingVariant]:
    variants = default_variants(architectures)
    if variant_names is None:
        return variants
    requested = list(variant_names)
    if len(requested) != len(set(requested)):
        raise ValueError("Requested training variants contain duplicates")
    available = {variant.name: variant for variant in variants}
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown or non-preregistered variants: {unknown}")
    requested_set = set(requested)
    return [variant for variant in variants if variant.name in requested_set]


def require_revision_training_contract(variant: TrainingVariant, metadata: Mapping[str, Any]) -> None:
    if variant.supervision_mode == "metadata_ranking":
        if (
            metadata.get("supervision_mode") != "metadata_ranking"
            or metadata.get("relevance_label_semantics") != METADATA_LABEL_SEMANTICS
            or metadata.get("scientific_claim_eligible") is not False
            or variant.decision_weight != 0.0
            or variant.adaptation_weight != 0.0
            or variant.uncertainty_weight != 0.0
        ):
            raise ValueError("Metadata ranking requires explicit non-scientific relevance labels and ranking-only losses")
        return
    if metadata.get("supervision_mode") == "metadata_ranking":
        raise ValueError("Metadata relevance cannot supervise scientific use classes; select --supervision-mode metadata_ranking")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            values.append(value)
    return values


def _atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        default=str,
    ).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_torch_save(value: Any, path: Path) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_benchmark(
    benchmark_directory: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, TypedGraph],
    dict[str, TypedGraph],
    dict[str, TypedGraph],
    dict[tuple[str, str], int],
]:
    queries = _read_jsonl(benchmark_directory / "queries.jsonl")
    query_graphs = {
        value["graph_id"]: graph_from_dict(value)
        for value in _read_jsonl(benchmark_directory / "query_graphs.jsonl")
    }
    model_graphs = {
        value["graph_id"]: graph_from_dict(value)
        for value in _read_jsonl(benchmark_directory / "model_graphs.jsonl")
    }
    gate_graph_path = benchmark_directory / "model_gate_graphs.jsonl"
    model_gate_graphs = {
        value["graph_id"]: graph_from_dict(value)
        for value in _read_jsonl(
            gate_graph_path
            if gate_graph_path.is_file()
            else benchmark_directory / "model_graphs.jsonl"
        )
    }
    label_path = benchmark_directory / "labels.jsonl"
    label_rows = _read_jsonl(
        label_path
        if label_path.is_file()
        else benchmark_directory / "predictions_uniform_rule.jsonl"
    )
    labels: dict[tuple[str, str], int] = {}
    for row in label_rows:
        key = (str(row["query_id"]), str(row["candidate_id"]))
        grade = row.get("relevance_grade")
        if row.get("compatibility_label_known") is False or row.get("label_known") is False or grade is None:
            raise ValueError("Dense supervised training requires known relevance labels; unknown labels cannot become zeros")
        if grade not in (0, 1, 2, 3):
            raise ValueError(f"Relevance grade must be an integer in 0..3, received {grade!r}")
        if key in labels:
            raise ValueError(f"Duplicate query/candidate relevance pair: {key}")
        labels[key] = int(grade)
    return queries, query_graphs, model_graphs, model_gate_graphs, labels


def _load_perturbation_queries(
    benchmark_directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, TypedGraph]]:
    queries_path = benchmark_directory / "perturbed_queries.jsonl"
    graphs_path = benchmark_directory / "perturbed_query_graphs.jsonl"
    if not queries_path.is_file() and not graphs_path.is_file():
        return [], {}
    if not queries_path.is_file() or not graphs_path.is_file():
        raise FileNotFoundError(
            "Perturbation evaluation requires both perturbed_queries.jsonl "
            "and perturbed_query_graphs.jsonl."
        )
    queries = _read_jsonl(queries_path)
    graphs = {
        value["graph_id"]: graph_from_dict(value)
        for value in _read_jsonl(graphs_path)
    }
    query_ids = {str(row["query_id"]) for row in queries}
    if len(query_ids) != len(queries):
        raise ValueError("Perturbed query IDs are not unique.")
    if query_ids != set(graphs):
        raise ValueError("Perturbed query rows and graphs do not have identical IDs.")
    return queries, graphs


def _load_adaptation_supervision(
    benchmark_directory: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    label_path = benchmark_directory / "labels.jsonl"
    if not label_path.is_file():
        return {}
    supervision: dict[tuple[str, str], dict[str, Any]] = {}
    allowed = set(ADAPTATION_OPERATION_TYPES)
    for row in _read_jsonl(label_path):
        if not row.get("adaptation_executed"):
            continue
        cost = row.get("adaptation_cost_label")
        if cost is None:
            cost = row.get("adaptation_weighted_cost")
        if cost is None:
            raise ValueError(
                "Executed adaptation is missing a weighted cost label."
            )
        operations = {
            str(value)
            for value in row.get("adaptation_operation_multilabel") or []
        }
        unknown = operations - allowed
        if unknown:
            raise ValueError(
                f"Unknown adaptation operation labels: {sorted(unknown)}."
            )
        key = (str(row["query_id"]), str(row["candidate_id"]))
        if key in supervision:
            raise ValueError(f"Duplicate adaptation supervision for {key}.")
        supervision[key] = {
            "weighted_cost": float(cost),
            "operations": operations,
            "scientific_passed": bool(
                row.get("adaptation_scientific_passed")
            ),
        }
    return supervision


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sample_candidates(
    query_id: str,
    model_ids: list[str],
    labels: Mapping[tuple[str, str], int],
    *,
    maximum: int,
    randomizer: random.Random,
) -> list[str]:
    positives = [model_id for model_id in model_ids if labels[(query_id, model_id)] >= 2]
    weak = [model_id for model_id in model_ids if labels[(query_id, model_id)] == 1]
    negatives = [model_id for model_id in model_ids if labels[(query_id, model_id)] == 0]
    selected = positives[:]
    randomizer.shuffle(weak)
    randomizer.shuffle(negatives)
    selected.extend(weak[: max(2, maximum // 4)])
    selected.extend(negatives[: max(4, maximum - len(selected))])
    return list(dict.fromkeys(selected))[:maximum]


def _decision_label(query: Mapping[str, Any], grade: int) -> int:
    del query
    return 0 if grade == 3 else 1 if grade == 2 else 2 if grade == 1 else 3


def _decision_class_weighting(
    *,
    queries: Iterable[Mapping[str, Any]],
    candidate_ids: Iterable[str],
    labels: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    counts = [0] * DECISION_CLASS_COUNT
    digest = hashlib.sha256()
    query_rows = sorted(queries, key=lambda row: str(row["query_id"]))
    candidates = sorted(str(value) for value in candidate_ids)
    for query in query_rows:
        query_id = str(query["query_id"])
        for candidate_id in candidates:
            decision_class = _decision_label(
                query, int(labels[(query_id, candidate_id)])
            )
            counts[decision_class] += 1
            digest.update(
                f"{query_id}\t{candidate_id}\t{decision_class}\n".encode("utf-8")
            )
    missing = [index for index, count in enumerate(counts) if count == 0]
    if missing:
        raise RuntimeError(
            "All four decision classes must occur in train query x train candidate "
            f"labels; missing classes: {missing}"
        )
    raw_weights = [
        (1.0 - DECISION_EFFECTIVE_NUMBER_BETA)
        / (1.0 - DECISION_EFFECTIVE_NUMBER_BETA**count)
        for count in counts
    ]
    scale = DECISION_CLASS_COUNT / sum(raw_weights)
    weights = [value * scale for value in raw_weights]
    return {
        "policy": DECISION_WEIGHTING_POLICY,
        "scope": "train_query_x_train_candidate_only",
        "class_encoding": {
            "0": "direct_reuse",
            "1": "controlled_adaptation",
            "2": "mechanistic_reference",
            "3": "reject",
        },
        "class_counts": counts,
        "class_weights": weights,
        "effective_number_beta": DECISION_EFFECTIVE_NUMBER_BETA,
        "focal_gamma": DECISION_FOCAL_GAMMA,
        "label_scope_sha256": digest.hexdigest(),
    }


def _training_decision_weighting(
    *,
    supervision_mode: str,
    queries: Iterable[Mapping[str, Any]],
    candidate_ids: Iterable[str],
    labels: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    if supervision_mode == "metadata_ranking":
        return {
            "policy": "not_applicable_metadata_ranking_only",
            "scope": "no_scientific_use_supervision",
            "class_counts": None,
            "class_weights": [0.0] * DECISION_CLASS_COUNT,
        }
    return _decision_class_weighting(queries=queries, candidate_ids=candidate_ids, labels=labels)


def _selection_policy(variant: TrainingVariant) -> str:
    return METADATA_VALIDATION_SELECTION_POLICY if variant.supervision_mode == "metadata_ranking" else VALIDATION_SELECTION_POLICY


def _optimization_policy(variant: TrainingVariant, full_candidate_training: bool = False) -> str:
    if full_candidate_training:
        return "full_candidate_per_query_checkpointed_listwise" if variant.supervision_mode == "metadata_ranking" else "full_candidate_per_query_listwise"
    return "pair_conditioned_per_query_updates" if variant.architecture == "gmn" else "full_batch_listwise_with_batched_heads_and_once_per_epoch_graph_encoding"


def _decision_classification(
    targets: Iterable[int], predictions: Iterable[int]
) -> dict[str, Any]:
    target_values = [int(value) for value in targets]
    prediction_values = [int(value) for value in predictions]
    if len(target_values) != len(prediction_values):
        raise ValueError("Decision targets and predictions must have equal length.")
    confusion = [
        [0 for _ in range(DECISION_CLASS_COUNT)]
        for _ in range(DECISION_CLASS_COUNT)
    ]
    for target, prediction in zip(
        target_values, prediction_values, strict=True
    ):
        if not 0 <= target < DECISION_CLASS_COUNT:
            raise ValueError(f"Invalid decision target class: {target}")
        if not 0 <= prediction < DECISION_CLASS_COUNT:
            raise ValueError(f"Invalid predicted decision class: {prediction}")
        confusion[target][prediction] += 1
    present_classes = sorted(set(target_values) | set(prediction_values))
    f1_by_class: dict[str, float] = {}
    for decision_class in present_classes:
        true_positive = confusion[decision_class][decision_class]
        false_positive = sum(
            confusion[row][decision_class]
            for row in range(DECISION_CLASS_COUNT)
            if row != decision_class
        )
        false_negative = sum(
            confusion[decision_class][column]
            for column in range(DECISION_CLASS_COUNT)
            if column != decision_class
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_by_class[str(decision_class)] = (
            2 * true_positive / denominator if denominator else 0.0
        )
    correct = sum(confusion[index][index] for index in range(DECISION_CLASS_COUNT))
    return {
        "macro_f1": (
            sum(f1_by_class.values()) / len(f1_by_class)
            if f1_by_class
            else 0.0
        ),
        "accuracy": correct / len(target_values) if target_values else 0.0,
        "f1_by_class": f1_by_class,
        "confusion_matrix": confusion,
        "support": len(target_values),
        "present_classes": present_classes,
    }


def _validation_joint_score(ndcg_at_10: float, decision_macro_f1: float) -> float:
    denominator = ndcg_at_10 + decision_macro_f1
    return (
        2.0 * ndcg_at_10 * decision_macro_f1 / denominator
        if denominator > 0.0
        else 0.0
    )


def _validation_selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["validation_joint_score"]),
        float(row["validation_ndcg@10"]),
        float(row["validation_raw_decision_macro_f1"] or 0.0),
        -int(row["epoch"]),
    )


def _build_resume_contract(
    *,
    variant: TrainingVariant,
    seed: int,
    device_type: str,
    epochs: int,
    candidates_per_query: int,
    learning_rate: float,
    feature_dimensions: int,
    decision_weighting: Mapping[str, Any],
    run_input_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "trainer_version": TRAINER_VERSION,
        "variant": asdict(variant),
        "seed": seed,
        "device_type": device_type,
        "epochs": epochs,
        "candidates_per_query": candidates_per_query,
        "learning_rate": learning_rate,
        "feature_dimensions": feature_dimensions,
        "decision_class_weighting": dict(decision_weighting),
        "run_inputs": dict(run_input_contract),
    }


def _batched_encoded_outputs(
    *,
    model: Any,
    query_tensor: Any,
    candidate_ids: Iterable[str],
    model_tensors: Mapping[str, Any],
    encoded_query: Any,
    encoded_models: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    candidates = list(candidate_ids)
    features = torch.stack(
        [
            model.pair_features_encoded(
                query_tensor,
                model_tensors[candidate_id],
                encoded_query,
                encoded_models[candidate_id],
            )
            for candidate_id in candidates
        ]
    )
    batched = model.heads_from_features(features)
    return [
        {key: value[index] for key, value in batched.items()}
        for index in range(len(candidates))
    ]


def _checkpointed_ranking_outputs(
    *, model: Any, query_tensor: Any, candidate_ids: list[str], model_tensors: Mapping[str, Any],
    candidate_batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Keep one full-candidate loss; bound activations by candidate microbatch."""
    from torch.utils.checkpoint import checkpoint

    independent = bool(model.matcher_metadata.get("supports_independent_graph_encoding"))
    query_nodes = model.encode_graph(query_tensor) if independent else query_tensor.x
    outputs = []
    if candidate_batch_size > 1:
        for start in range(0, len(candidate_ids), candidate_batch_size):
            candidates = [model_tensors[key] for key in candidate_ids[start:start + candidate_batch_size]]

            def batch_score(nodes: Any, batch: list[Any] = candidates) -> Any:
                return model.forward_candidates(query_tensor, batch, query_nodes=nodes)["score_logit"]

            scores = checkpoint(batch_score, query_nodes, use_reentrant=False, preserve_rng_state=True)
            outputs.extend({"score_logit": score} for score in scores.unbind(0))
        return outputs
    for candidate_id in candidate_ids:
        candidate_tensor = model_tensors[candidate_id]

        def pair_score(nodes: Any, candidate: Any = candidate_tensor) -> Any:
            if independent:
                return model.forward_encoded(
                    query_tensor, candidate, nodes, model.encode_graph(candidate)
                )["score_logit"]
            return model(query_tensor, candidate)["score_logit"]

        outputs.append({
            "score_logit": checkpoint(pair_score, query_nodes, use_reentrant=False, preserve_rng_state=True)
        })
    return outputs


def _candidate_batch_outputs(
    *, model: Any, query_tensor: Any, candidate_ids: list[str],
    model_tensors: Mapping[str, Any], candidate_batch_size: int,
    query_nodes: Any = None,
) -> list[dict[str, Any]]:
    """Batch inference without retaining per-candidate device-to-host transfers."""
    outputs = []
    for start in range(0, len(candidate_ids), candidate_batch_size):
        candidates = [model_tensors[key] for key in candidate_ids[start:start + candidate_batch_size]]
        batch = model.forward_candidates(query_tensor, candidates, query_nodes=query_nodes)
        cpu_batch = {key: value.cpu() for key, value in batch.items()}
        outputs.extend({key: value[index] for key, value in cpu_batch.items()}
                       for index in range(len(candidates)))
    return outputs


def _without_node_types(
    graph: TypedGraph,
    excluded_types: set[str],
) -> TypedGraph:
    nodes = tuple(
        node for node in graph.nodes if node.node_type not in excluded_types
    )
    node_ids = {node.node_id for node in nodes}
    edges = tuple(
        edge
        for edge in graph.edges
        if edge.source_id in node_ids and edge.target_id in node_ids
    )
    return TypedGraph(
        graph_id=graph.graph_id,
        graph_kind=graph.graph_kind,
        nodes=nodes,
        edges=edges,
        metadata=graph.metadata,
    )


def _without_validation_nodes(graph: TypedGraph) -> TypedGraph:
    return _without_node_types(
        graph,
        {
            "Dataset",
            "CalibrationDataset",
            "QualificationDataset",
            "ModelDataset",
        },
    )


def _without_simulation_verification_nodes(graph: TypedGraph) -> TypedGraph:
    return _without_node_types(
        graph,
        {
            "Dataset",
            "CalibrationDataset",
            "QualificationDataset",
            "ModelDataset",
            "ValidationRun",
            "ValidationMetric",
            "ExecutionStatus",
        },
    )


def _without_molecular_nodes(graph: TypedGraph) -> TypedGraph:
    return _without_node_types(
        graph,
        {
            "MolecularStructure",
            "MolecularFingerprint",
            "MolecularDescriptor",
        },
    )


def _training_loss(
    *,
    variant: TrainingVariant,
    query: Mapping[str, Any],
    query_id: str,
    selected: list[str],
    outputs: list[Mapping[str, Any]],
    labels: Mapping[tuple[str, str], int],
    adaptation_supervision: Mapping[
        tuple[str, str], Mapping[str, Any]
    ],
    decision_class_weights: Any,
    device: Any,
) -> Any:
    import torch

    logits = torch.stack([output["score_logit"] for output in outputs])
    grades = torch.tensor(
        [labels[(query_id, candidate_id)] for candidate_id in selected],
        dtype=torch.float32,
        device=device,
    )
    gains = torch.pow(2.0, grades) - 1.0
    target_distribution = gains / gains.sum().clamp_min(1.0)
    ranking_loss = -(target_distribution * torch.log_softmax(logits, dim=0)).sum()
    if variant.supervision_mode == "metadata_ranking":
        return variant.ranking_weight * ranking_loss
    decisions = torch.stack([output["decision_logits"] for output in outputs])
    decision_target = torch.tensor(
        [_decision_label(query, int(grade)) for grade in grades.tolist()],
        dtype=torch.long,
        device=device,
    )
    decision_log_probabilities = torch.log_softmax(decisions, dim=-1)
    target_log_probabilities = decision_log_probabilities.gather(
        1, decision_target.unsqueeze(1)
    ).squeeze(1)
    target_probabilities = target_log_probabilities.exp()
    target_class_weights = decision_class_weights[decision_target]
    decision_loss = (
        -target_class_weights
        * (1.0 - target_probabilities).pow(DECISION_FOCAL_GAMMA)
        * target_log_probabilities
    ).mean()
    uncertainty = torch.stack([output["uncertainty"] for output in outputs])
    uncertainty_target = torch.tensor(
        [1.0 if grade in {1, 2} else 0.0 for grade in grades.tolist()],
        dtype=torch.float32,
        device=device,
    )
    uncertainty_loss = torch.nn.functional.binary_cross_entropy(
        uncertainty, uncertainty_target
    )
    adaptation_losses = []
    for candidate_id, output in zip(selected, outputs, strict=True):
        supervision = adaptation_supervision.get((query_id, candidate_id))
        if supervision is None:
            continue
        normalized_cost = min(
            1.0,
            float(supervision["weighted_cost"]) / ADAPTATION_COST_SCALE,
        )
        cost_target = torch.tensor(
            normalized_cost,
            dtype=torch.float32,
            device=device,
        )
        cost_loss = torch.nn.functional.smooth_l1_loss(
            output["adaptation_cost"], cost_target
        )
        operations = set(supervision["operations"])
        operation_target = torch.tensor(
            [
                1.0 if operation in operations else 0.0
                for operation in ADAPTATION_OPERATION_TYPES
            ],
            dtype=torch.float32,
            device=device,
        )
        operation_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["adaptation_operation_logits"], operation_target
        )
        adaptation_losses.append(cost_loss + operation_loss)
    adaptation_loss = (
        torch.stack(adaptation_losses).mean()
        if adaptation_losses
        else logits.sum() * 0.0
    )
    return (
        variant.ranking_weight * ranking_loss
        + variant.adaptation_weight * adaptation_loss
        + variant.decision_weight * decision_loss
        + variant.uncertainty_weight * uncertainty_loss
    )


def _train_one(
    *,
    variant: TrainingVariant,
    seed: int,
    queries: list[dict[str, Any]],
    query_graphs: Mapping[str, TypedGraph],
    perturbed_queries: list[dict[str, Any]],
    perturbed_query_graphs: Mapping[str, TypedGraph],
    model_graphs: Mapping[str, TypedGraph],
    model_gate_graphs: Mapping[str, TypedGraph],
    query_tensors: Mapping[str, Any],
    perturbed_query_tensors: Mapping[str, Any],
    model_tensors: Mapping[str, Any],
    labels: Mapping[tuple[str, str], int],
    adaptation_supervision: Mapping[
        tuple[str, str], Mapping[str, Any]
    ],
    output_directory: Path,
    device_name: str,
    epochs: int,
    candidates_per_query: int,
    learning_rate: float,
    feature_dimensions: int,
    benchmark_metadata: Mapping[str, Any],
    run_input_contract: Mapping[str, Any],
    resume_partial: bool,
    full_candidate_training: bool = False,
    disable_early_stopping: bool = False,
    candidate_batch_size: int = 1,
) -> dict[str, Any]:
    import torch

    require_revision_training_contract(variant, benchmark_metadata)
    _seed_everything(seed)
    device = torch.device(device_name)
    model = build_matcher(
        variant.architecture,
        feature_dimensions=feature_dimensions,
        use_relation_typing=variant.use_relation_typing,
        use_cross_graph_attention=variant.use_cross_graph_attention,
        use_pbpk_hierarchy=variant.use_pbpk_hierarchy,
    ).to(device)
    model.candidate_batch_size = candidate_batch_size
    if candidate_batch_size > 1:
        # The outer candidate checkpoint already bounds saved activations.
        model.set_projection_checkpointing(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_queries = [query for query in queries if query["split"] == "train"]
    validation_queries = [query for query in queries if query["split"] == "validation"]
    train_model_ids = audited_candidate_ids(
        model_graphs, allowed_splits={"train"}
    )
    validation_model_ids = audited_candidate_ids(
        model_graphs, allowed_splits={"train", "validation"}
    )
    if not train_model_ids:
        raise RuntimeError("The grouped split produced no training model families.")
    decision_weighting = _training_decision_weighting(
        supervision_mode=variant.supervision_mode,
        queries=train_queries,
        candidate_ids=train_model_ids,
        labels=labels,
    )
    decision_class_weights = torch.tensor(
        decision_weighting["class_weights"],
        dtype=torch.float32,
        device=device,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    progress_path = output_directory / f"seed_{seed}_resume.pt"
    history_path = output_directory / f"seed_{seed}_history.json"
    status_path = output_directory / f"seed_{seed}_training_status.json"
    last_status_write = 0.0
    resume_contract = _build_resume_contract(
        variant=variant,
        seed=seed,
        device_type=device.type,
        epochs=epochs,
        candidates_per_query=candidates_per_query,
        learning_rate=learning_rate,
        feature_dimensions=feature_dimensions,
        decision_weighting=decision_weighting,
        run_input_contract=run_input_contract,
    )
    resume_contract_sha256 = _canonical_sha256(resume_contract)
    history: list[dict[str, Any]] = []
    best_validation = float("-inf")
    best_validation_ndcg = float("-inf")
    best_validation_decision_f1 = None if variant.supervision_mode == "metadata_ranking" else float("-inf")
    best_epoch = 0
    best_selection_key: tuple[float, float, float, int] | None = None
    best_state: dict[str, Any] | None = None
    patience = 0
    randomizer = random.Random(seed)
    start_epoch = 1
    resumed_from_epoch = 0
    if resume_partial and progress_path.is_file():
        progress = torch.load(
            progress_path,
            map_location=device,
            weights_only=False,
        )
        if progress.get("resume_contract_sha256") != resume_contract_sha256:
            raise RuntimeError(
                "Partial checkpoint contract does not match the requested run: "
                f"{progress_path}"
            )
        model.load_state_dict(progress["current_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        best_state = progress["best_state"]
        history = list(progress["history"])
        best_validation = float(progress["best_validation_joint_score"])
        best_validation_ndcg = float(progress["best_validation_ndcg@10"])
        best_validation_decision_f1 = progress["best_validation_raw_decision_macro_f1"]
        best_epoch = int(progress["best_epoch"])
        best_selection_key = tuple(progress["best_selection_key"])
        patience = int(progress["patience"])
        resumed_from_epoch = int(progress["completed_epoch"])
        start_epoch = resumed_from_epoch + 1
        randomizer.setstate(progress["candidate_randomizer_state"])
        random.setstate(progress["python_random_state"])
        torch.set_rng_state(progress["torch_rng_state"].cpu())
        if device.type == "cuda" and progress.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in progress["cuda_rng_state_all"]]
            )

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        _atomic_json(status_path, {
            "state": "training", "seed": seed, "variant": variant.name,
            "epoch": epoch, "epochs": epochs, "queries_completed": 0,
            "training_query_count": len(train_queries),
            "candidate_count": len(train_model_ids),
            "full_candidate_training": full_candidate_training,
            "candidate_batch_size": candidate_batch_size,
            "device": str(device), "resumed_from_epoch": resumed_from_epoch,
            "supervision_mode": variant.supervision_mode,
            "updated_at": datetime.now(UTC).isoformat(),
        })
        model.train()
        train_queries.sort(key=lambda row: str(row["query_id"]))
        randomizer.shuffle(train_queries)
        selected_by_query: list[tuple[dict[str, Any], list[str]]] = []
        for query in train_queries:
            query_id = str(query["query_id"])
            selected = train_model_ids[:] if full_candidate_training else _sample_candidates(
                query_id,
                train_model_ids,
                labels,
                maximum=candidates_per_query,
                randomizer=randomizer,
            )
            selected_by_query.append((query, selected))

        losses = []
        independent = bool(
            model.matcher_metadata.get("supports_independent_graph_encoding")
        )
        if independent and not full_candidate_training:
            encoded_queries = {
                str(query["query_id"]): model.encode_graph(
                    query_tensors[str(query["query_id"])]
                )
                for query, _ in selected_by_query
            }
            selected_model_ids = sorted(
                {
                    candidate_id
                    for _, selected in selected_by_query
                    for candidate_id in selected
                }
            )
            encoded_models = {
                candidate_id: model.encode_graph(model_tensors[candidate_id])
                for candidate_id in selected_model_ids
            }
            query_losses = []
            for query, selected in selected_by_query:
                query_id = str(query["query_id"])
                outputs = _batched_encoded_outputs(
                    model=model,
                    query_tensor=query_tensors[query_id],
                    candidate_ids=selected,
                    model_tensors=model_tensors,
                    encoded_query=encoded_queries[query_id],
                    encoded_models=encoded_models,
                )
                query_loss = _training_loss(
                    variant=variant,
                    query=query,
                    query_id=query_id,
                    selected=selected,
                    outputs=outputs,
                    labels=labels,
                    adaptation_supervision=adaptation_supervision,
                    decision_class_weights=decision_class_weights,
                    device=device,
                )
                query_losses.append(query_loss)
                losses.append(float(query_loss.detach().cpu()))
            loss = torch.stack(query_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            for query_index, (query, selected) in enumerate(selected_by_query, 1):
                query_id = str(query["query_id"])
                if full_candidate_training and variant.supervision_mode == "metadata_ranking":
                    outputs = _checkpointed_ranking_outputs(
                        model=model, query_tensor=query_tensors[query_id],
                        candidate_ids=selected, model_tensors=model_tensors,
                        candidate_batch_size=candidate_batch_size,
                    )
                elif independent:
                    encoded_query = model.encode_graph(query_tensors[query_id])
                    encoded_models = {
                        candidate_id: model.encode_graph(model_tensors[candidate_id])
                        for candidate_id in selected
                    }
                    outputs = _batched_encoded_outputs(
                        model=model,
                        query_tensor=query_tensors[query_id],
                        candidate_ids=selected,
                        model_tensors=model_tensors,
                        encoded_query=encoded_query,
                        encoded_models=encoded_models,
                    )
                else:
                    outputs = [
                        model(query_tensors[query_id], model_tensors[candidate_id])
                        for candidate_id in selected
                    ]
                loss = _training_loss(
                    variant=variant,
                    query=query,
                    query_id=query_id,
                    selected=selected,
                    outputs=outputs,
                    labels=labels,
                    adaptation_supervision=adaptation_supervision,
                    decision_class_weights=decision_class_weights,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if time.monotonic() - last_status_write >= 30.0 or query_index == len(selected_by_query):
                    _atomic_json(status_path, {
                        "state": "training", "seed": seed, "variant": variant.name,
                        "epoch": epoch, "epochs": epochs, "queries_completed": query_index,
                        "training_query_count": len(train_queries),
                        "candidate_count": len(train_model_ids),
                        "full_candidate_training": full_candidate_training,
                        "candidate_batch_size": candidate_batch_size,
                        "device": str(device), "resumed_from_epoch": resumed_from_epoch,
                        "mean_query_seconds": (time.perf_counter() - epoch_started) / query_index,
                        "cuda_memory_allocated_mib": (
                            round(torch.cuda.memory_allocated(device) / 2**20, 1) if device.type == "cuda" else None
                        ),
                        "cuda_peak_memory_allocated_mib": (
                            round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if device.type == "cuda" else None
                        ),
                        "supervision_mode": variant.supervision_mode,
                        "updated_at": datetime.now(UTC).isoformat(),
                    })
                    last_status_write = time.monotonic()

        _atomic_json(status_path, {
            "state": "validating", "seed": seed, "variant": variant.name,
            "epoch": epoch, "epochs": epochs,
            "queries_completed": len(train_queries),
            "training_query_count": len(train_queries),
            "validation_query_count": len(validation_queries),
            "validation_candidate_count": len(validation_model_ids),
            "candidate_batch_size": candidate_batch_size, "device": str(device),
            "supervision_mode": variant.supervision_mode,
            "updated_at": datetime.now(UTC).isoformat(),
        })
        validation_metrics = _quick_validation_metrics(
            model=model,
            variant=variant,
            queries=validation_queries,
            query_graphs=query_graphs,
            query_tensors=query_tensors,
            model_tensors=model_tensors,
            model_gate_graphs=model_gate_graphs,
            labels=labels,
            candidate_ids=validation_model_ids,
        )
        history_row = {
            "epoch": epoch,
            "candidate_batch_size": candidate_batch_size,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "train_loss": sum(losses) / max(1, len(losses)),
            **validation_metrics,
        }
        history.append(history_row)
        print(json.dumps({"variant": variant.name, "seed": seed, **history_row}, sort_keys=True), flush=True)
        selection_key = _validation_selection_key(history_row)
        should_stop = False
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_validation = float(history_row["validation_joint_score"])
            best_validation_ndcg = float(history_row["validation_ndcg@10"])
            best_validation_decision_f1 = history_row["validation_raw_decision_macro_f1"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 8 and not disable_early_stopping:
                should_stop = True
        _atomic_json(history_path, history)
        _atomic_torch_save(
            {
                "resume_contract": resume_contract,
                "resume_contract_sha256": resume_contract_sha256,
                "completed_epoch": epoch,
                "candidate_batch_size": candidate_batch_size,
                "current_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "best_state": best_state,
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_validation_joint_score": best_validation,
                "best_validation_ndcg@10": best_validation_ndcg,
                "best_validation_raw_decision_macro_f1": (
                    best_validation_decision_f1
                ),
                "best_epoch": best_epoch,
                "best_selection_key": best_selection_key,
                "patience": patience,
                "candidate_randomizer_state": randomizer.getstate(),
                "python_random_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else []
                ),
            },
            progress_path,
        )
        if should_stop:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    test_queries = [query for query in queries if query["split"] == "test"]
    _atomic_json(status_path, {
        "state": "evaluating_held_out_queries", "seed": seed, "variant": variant.name,
        "epochs_completed": len(history), "selected_epoch": best_epoch,
        "evaluation_query_count": len(test_queries), "evaluation_candidate_count": len(model_tensors),
        "supervision_mode": variant.supervision_mode,
        "updated_at": datetime.now(UTC).isoformat(),
    })
    predictions = _predict(
        model=model,
        variant=variant,
        queries=test_queries,
        query_graphs=query_graphs,
        query_tensors=query_tensors,
        model_graphs=model_graphs,
        model_gate_graphs=model_gate_graphs,
        model_tensors=model_tensors,
        labels=labels,
        adaptation_supervision=adaptation_supervision,
    )
    metrics = (evaluate_metadata_rankings if variant.supervision_mode == "metadata_ranking" else evaluate_rankings)(predictions)
    perturbation_predictions: list[dict[str, Any]] = []
    perturbation_metrics: dict[str, Any] | None = None
    if perturbed_queries:
        perturbation_predictions = _predict(
            model=model,
            variant=variant,
            queries=[
                query for query in perturbed_queries if query["split"] == "test"
            ],
            query_graphs=perturbed_query_graphs,
            query_tensors=perturbed_query_tensors,
            model_graphs=model_graphs,
            model_gate_graphs=model_gate_graphs,
            model_tensors=model_tensors,
            labels=None,
            adaptation_supervision=adaptation_supervision,
        )
        perturbation_links = {
            str(query["query_id"]): str(query["base_query_id"])
            for query in perturbed_queries
            if query["split"] == "test"
        }
        perturbation_metrics = evaluate_ranking_continuity(
            base_rows=predictions,
            perturbed_rows=perturbation_predictions,
            task_links=perturbation_links,
        )
    checkpoint_path = output_directory / f"seed_{seed}.pt"
    _atomic_torch_save(
        {
            "trainer_version": TRAINER_VERSION,
            "variant": asdict(variant),
            "model_metadata": model.matcher_metadata,
            "state_dict": best_state,
            "seed": seed,
            "benchmark_level": benchmark_metadata["benchmark_level"],
            "scientific_claim_eligible": benchmark_metadata[
                "scientific_claim_eligible"
            ],
            "adaptation_supervision_available": (
                bool(adaptation_supervision)
            ),
            "adaptation_operation_types": list(
                ADAPTATION_OPERATION_TYPES
            ),
            "adaptation_cost_scale": ADAPTATION_COST_SCALE,
            "decision_class_weighting": decision_weighting,
            "validation_selection_policy": _selection_policy(variant),
            "selected_epoch": best_epoch,
            "best_validation_joint_score": best_validation,
            "best_validation_ndcg@10": best_validation_ndcg,
            "best_validation_raw_decision_macro_f1": (
                best_validation_decision_f1
            ),
            "resume_contract_sha256": resume_contract_sha256,
            "resume_contract": resume_contract,
            "resumed_from_epoch": resumed_from_epoch,
            "uncertainty_supervision": (
                "not_trained_metadata_ranking_only"
                if variant.supervision_mode == "metadata_ranking"
                else "execution_derived_outcome_variation"
                if benchmark_metadata["benchmark_level"] == "execution_derived"
                else "candidate_retrieval_label_ambiguity"
            ),
            "optimization_policy": _optimization_policy(variant, full_candidate_training),
            "candidate_batch_size": candidate_batch_size,
            "supervision_mode": variant.supervision_mode,
            "relevance_label_semantics": benchmark_metadata.get("relevance_label_semantics"),
        },
        checkpoint_path,
    )
    _atomic_json(history_path, history)
    training_manifest_checksum = _atomic_json(
        output_directory / f"seed_{seed}_training_manifest.json",
        {
            "trainer_version": TRAINER_VERSION,
            "seed": seed,
            "variant": asdict(variant),
            "decision_class_weighting": decision_weighting,
            "validation_selection_policy": _selection_policy(variant),
            "selected_epoch": best_epoch,
            "best_validation_joint_score": best_validation,
            "best_validation_ndcg@10": best_validation_ndcg,
            "best_validation_raw_decision_macro_f1": (
                best_validation_decision_f1
            ),
            "resume_contract_sha256": resume_contract_sha256,
            "resume_contract": resume_contract,
            "resumed_from_epoch": resumed_from_epoch,
            "candidate_batch_size": candidate_batch_size,
        },
    )
    metrics_checksum = _atomic_json(
        output_directory / f"seed_{seed}_metrics.json", metrics
    )
    predictions_checksum = _atomic_jsonl(
        output_directory / f"seed_{seed}_predictions.jsonl",
        predictions,
    )
    perturbation_predictions_checksum = None
    perturbation_metrics_checksum = None
    if perturbed_queries:
        perturbation_predictions_checksum = _atomic_jsonl(
            output_directory / f"seed_{seed}_perturbation_predictions.jsonl",
            perturbation_predictions,
        )
        perturbation_metrics_checksum = _atomic_json(
            output_directory / f"seed_{seed}_perturbation_metrics.json",
            perturbation_metrics,
        )
    progress_path.unlink(missing_ok=True)
    _atomic_json(status_path, {
        "state": "complete", "seed": seed, "variant": variant.name,
        "epochs_completed": len(history), "selected_epoch": best_epoch,
        "supervision_mode": variant.supervision_mode,
        "scientific_claim_eligible": benchmark_metadata["scientific_claim_eligible"],
        "updated_at": datetime.now(UTC).isoformat(),
    })
    return {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "fidelity": METHOD_FIDELITY[variant.architecture],
        "best_validation_joint_score": best_validation,
        "best_validation_ndcg@10": best_validation_ndcg,
        "best_validation_raw_decision_macro_f1": (
            best_validation_decision_f1
        ),
        "selected_epoch": best_epoch,
        "decision_class_weighting": decision_weighting,
        "training_manifest_sha256": training_manifest_checksum,
        "resume_contract_sha256": resume_contract_sha256,
        "resumed_from_epoch": resumed_from_epoch,
        "test": metrics["aggregate"],
        "epochs_completed": len(history),
        "metrics_sha256": metrics_checksum,
        "predictions_sha256": predictions_checksum,
        "perturbation_predictions_sha256": perturbation_predictions_checksum,
        "perturbation_metrics_sha256": perturbation_metrics_checksum,
        "perturbation_test": (
            {
                key: value
                for key, value in perturbation_metrics.items()
                if key != "details"
            }
            if perturbation_metrics is not None
            else None
        ),
        "adaptation_cost_source": (
            "learned_from_versioned_adaptation_outcomes"
            if variant.adaptation_weight > 0 and adaptation_supervision
            else "not_predicted_by_ablation"
            if variant.adaptation_weight == 0
            else "unavailable_no_executed_adaptations"
        ),
        "uncertainty_supervision": (
            "not_trained_metadata_ranking_only"
            if variant.supervision_mode == "metadata_ranking"
            else "execution_derived_outcome_variation"
            if benchmark_metadata["benchmark_level"] == "execution_derived"
            else "candidate_retrieval_label_ambiguity"
        ),
        "optimization_policy": _optimization_policy(variant, full_candidate_training),
        "candidate_batch_size": candidate_batch_size,
    }


def _completed_seed_result(
    variant: TrainingVariant,
    seed: int,
    variant_directory: Path,
    benchmark_metadata: Mapping[str, Any],
    expect_perturbations: bool,
    expected_resume_contract_sha256: str,
) -> dict[str, Any] | None:
    checkpoint_path = variant_directory / f"seed_{seed}.pt"
    history_path = variant_directory / f"seed_{seed}_history.json"
    metrics_path = variant_directory / f"seed_{seed}_metrics.json"
    predictions_path = variant_directory / f"seed_{seed}_predictions.jsonl"
    perturbation_predictions_path = (
        variant_directory / f"seed_{seed}_perturbation_predictions.jsonl"
    )
    perturbation_metrics_path = (
        variant_directory / f"seed_{seed}_perturbation_metrics.json"
    )
    training_manifest_path = (
        variant_directory / f"seed_{seed}_training_manifest.json"
    )
    paths = [
        checkpoint_path,
        history_path,
        metrics_path,
        predictions_path,
        training_manifest_path,
    ]
    if expect_perturbations:
        paths.extend((perturbation_predictions_path, perturbation_metrics_path))
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return None
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        training_manifest = json.loads(
            training_manifest_path.read_text(encoding="utf-8")
        )
        if training_manifest["trainer_version"] != TRAINER_VERSION:
            return None
        if (
            training_manifest["resume_contract_sha256"]
            != expected_resume_contract_sha256
        ):
            return None
        best_validation_row = max(history, key=_validation_selection_key)
        aggregate = metrics["aggregate"]
        perturbation_metrics = (
            json.loads(perturbation_metrics_path.read_text(encoding="utf-8"))
            if expect_perturbations
            else None
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    return {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "fidelity": METHOD_FIDELITY[variant.architecture],
        "best_validation_joint_score": float(
            best_validation_row["validation_joint_score"]
        ),
        "best_validation_ndcg@10": float(
            best_validation_row["validation_ndcg@10"]
        ),
        "best_validation_raw_decision_macro_f1": best_validation_row["validation_raw_decision_macro_f1"],
        "selected_epoch": int(best_validation_row["epoch"]),
        "decision_class_weighting": training_manifest[
            "decision_class_weighting"
        ],
        "training_manifest_sha256": digest(training_manifest_path),
        "test": aggregate,
        "epochs_completed": len(history),
        "metrics_sha256": digest(metrics_path),
        "predictions_sha256": digest(predictions_path),
        "perturbation_predictions_sha256": (
            digest(perturbation_predictions_path)
            if expect_perturbations
            else None
        ),
        "perturbation_metrics_sha256": (
            digest(perturbation_metrics_path) if expect_perturbations else None
        ),
        "perturbation_test": (
            {
                key: value
                for key, value in perturbation_metrics.items()
                if key != "details"
            }
            if perturbation_metrics is not None
            else None
        ),
        "adaptation_cost_source": (
            "learned_from_versioned_adaptation_outcomes"
            if variant.adaptation_weight > 0
            and benchmark_metadata.get("adaptation_supervision_available")
            else "not_predicted_by_ablation"
            if variant.adaptation_weight == 0
            else "unavailable_no_executed_adaptations"
        ),
        "uncertainty_supervision": (
            "not_trained_metadata_ranking_only"
            if variant.supervision_mode == "metadata_ranking"
            else "execution_derived_outcome_variation"
            if benchmark_metadata["benchmark_level"] == "execution_derived"
            else "candidate_retrieval_label_ambiguity"
        ),
        "optimization_policy": _optimization_policy(
            variant, bool(training_manifest.get("resume_contract", {}).get("run_inputs", {}).get("full_candidate_training"))
        ),
        "resumed_from_verified_artifacts": True,
        "candidate_batch_size": training_manifest.get("candidate_batch_size", 1),
    }


def _quick_validation_metrics(
    *,
    model: Any,
    variant: TrainingVariant,
    queries: list[dict[str, Any]],
    query_graphs: Mapping[str, TypedGraph],
    query_tensors: Mapping[str, Any],
    model_tensors: Mapping[str, Any],
    model_gate_graphs: Mapping[str, TypedGraph],
    labels: Mapping[tuple[str, str], int],
    candidate_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    import torch

    rows = []
    decision_targets: list[int] = []
    decision_predictions: list[int] = []
    model.eval()
    candidate_list = list(candidate_ids or model_tensors)
    batch_size = getattr(model, "candidate_batch_size", 1)
    with torch.inference_mode():
        independent = bool(
            model.matcher_metadata.get("supports_independent_graph_encoding")
        )
        encoded_queries = (
            {
                str(query["query_id"]): model.encode_graph(
                    query_tensors[str(query["query_id"])]
                )
                for query in queries
            }
            if independent
            else {}
        )
        encoded_models = (
            {
                candidate_id: model.encode_graph(model_tensors[candidate_id])
                for candidate_id in candidate_list
            }
            if independent and batch_size == 1
            else {}
        )
        for query in queries:
            query_id = str(query["query_id"])
            outputs = (
                _candidate_batch_outputs(
                    model=model, query_tensor=query_tensors[query_id],
                    candidate_ids=candidate_list, model_tensors=model_tensors,
                    candidate_batch_size=batch_size, query_nodes=encoded_queries.get(query_id),
                )
                if batch_size > 1 else
                _batched_encoded_outputs(
                    model=model,
                    query_tensor=query_tensors[query_id],
                    candidate_ids=candidate_list,
                    model_tensors=model_tensors,
                    encoded_query=encoded_queries[query_id],
                    encoded_models=encoded_models,
                )
                if independent
                else [
                    model(query_tensors[query_id], model_tensors[candidate_id])
                    for candidate_id in candidate_list
                ]
            )
            for candidate_id, output in zip(
                candidate_list, outputs, strict=True
            ):
                raw_score = float(output["score"].cpu())
                relevance_grade = int(labels[(query_id, candidate_id)])
                if variant.supervision_mode != "metadata_ranking":
                    decision_targets.append(_decision_label(query, relevance_grade))
                    decision_predictions.append(int(output["decision_logits"].argmax().cpu()))
                rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": candidate_id,
                        "score": raw_score,
                        "relevance_grade": relevance_grade,
                        "reuse_decision": False,
                    }
                )
    evaluator = evaluate_metadata_rankings if variant.supervision_mode == "metadata_ranking" else evaluate_rankings
    ndcg_at_10 = float(evaluator(rows)["aggregate"].get("ndcg@10") or 0.0)
    if variant.supervision_mode == "metadata_ranking":
        return {
            "validation_ndcg@10": ndcg_at_10,
            "validation_raw_decision_macro_f1": None,
            "validation_raw_decision_accuracy": None,
            "validation_joint_score": ndcg_at_10,
            "validation_selection_policy": METADATA_VALIDATION_SELECTION_POLICY,
            "supervision_mode": "metadata_ranking",
        }
    decision_metrics = _decision_classification(
        decision_targets, decision_predictions
    )
    decision_macro_f1 = float(decision_metrics["macro_f1"])
    return {
        "validation_ndcg@10": ndcg_at_10,
        "validation_raw_decision_macro_f1": decision_macro_f1,
        "validation_raw_decision_accuracy": float(
            decision_metrics["accuracy"]
        ),
        "validation_raw_decision_f1_by_class": decision_metrics[
            "f1_by_class"
        ],
        "validation_raw_decision_confusion_matrix": decision_metrics[
            "confusion_matrix"
        ],
        "validation_raw_decision_present_classes": decision_metrics[
            "present_classes"
        ],
        "validation_joint_score": _validation_joint_score(
            ndcg_at_10, decision_macro_f1
        ),
        "validation_selection_policy": VALIDATION_SELECTION_POLICY,
    }


def _predict(
    *,
    model: Any,
    variant: TrainingVariant,
    queries: list[dict[str, Any]],
    query_graphs: Mapping[str, TypedGraph],
    query_tensors: Mapping[str, Any],
    model_graphs: Mapping[str, TypedGraph],
    model_gate_graphs: Mapping[str, TypedGraph],
    model_tensors: Mapping[str, Any],
    labels: Mapping[tuple[str, str], int] | None,
    adaptation_supervision: Mapping[
        tuple[str, str], Mapping[str, Any]
    ],
) -> list[dict[str, Any]]:
    import torch

    rows = []
    model.eval()
    batch_size = getattr(model, "candidate_batch_size", 1)
    with torch.inference_mode():
        independent = bool(
            model.matcher_metadata.get("supports_independent_graph_encoding")
        )
        encoded_queries = (
            {
                str(query["query_id"]): model.encode_graph(
                    query_tensors[str(query["query_id"])]
                )
                for query in queries
            }
            if independent
            else {}
        )
        encoded_models = (
            {
                candidate_id: model.encode_graph(candidate_tensor)
                for candidate_id, candidate_tensor in model_tensors.items()
            }
            if independent and batch_size == 1
            else {}
        )
        for query in queries:
            query_id = str(query["query_id"])
            candidate_ids = list(model_tensors)
            outputs = (
                _candidate_batch_outputs(
                    model=model, query_tensor=query_tensors[query_id],
                    candidate_ids=candidate_ids, model_tensors=model_tensors,
                    candidate_batch_size=batch_size, query_nodes=encoded_queries.get(query_id),
                )
                if batch_size > 1 else
                _batched_encoded_outputs(
                    model=model,
                    query_tensor=query_tensors[query_id],
                    candidate_ids=candidate_ids,
                    model_tensors=model_tensors,
                    encoded_query=encoded_queries[query_id],
                    encoded_models=encoded_models,
                )
                if independent
                else [
                    model(query_tensors[query_id], model_tensors[candidate_id])
                    for candidate_id in candidate_ids
                ]
            )
            for candidate_id, output in zip(
                candidate_ids, outputs, strict=True
            ):
                deterministic = score_candidate(
                    query_graphs[query_id], model_gate_graphs[candidate_id]
                )
                assessment = assess_candidate_use(
                    query_graphs[query_id],
                    model_gate_graphs[candidate_id],
                    match=deterministic,
                )
                raw_score = float(output["score"].cpu())
                raw_decision_class = None if variant.supervision_mode == "metadata_ranking" else int(output["decision_logits"].argmax().cpu())
                decision_class = (
                    deterministic_use_class(assessment)
                    if variant.supervision_mode == "metadata_ranking"
                    else apply_use_tier_gate(raw_decision_class, assessment, enabled=variant.use_hard_gate)
                )
                supervision = adaptation_supervision.get(
                    (query_id, candidate_id)
                )
                adaptation_enabled = bool(
                    variant.adaptation_weight > 0
                    and adaptation_supervision
                )
                predicted_operation_probabilities = (
                    {
                        operation: float(probability)
                        for operation, probability in zip(
                            ADAPTATION_OPERATION_TYPES,
                            torch.sigmoid(
                                output["adaptation_operation_logits"]
                            )
                            .detach()
                            .cpu()
                            .tolist(),
                            strict=True,
                        )
                    }
                    if adaptation_enabled
                    else {}
                )
                rows.append(
                    {
                        "query_id": query_id,
                        "base_query_id": query.get("base_query_id"),
                        "query_split": query.get("split"),
                        "candidate_id": candidate_id,
                        "supervision_mode": variant.supervision_mode,
                        "scientific_use_label_known": variant.supervision_mode != "metadata_ranking",
                        "relevance_label_semantics": METADATA_LABEL_SEMANTICS if variant.supervision_mode == "metadata_ranking" else "scientific_use_grade",
                        "use_decision_source": "deterministic_evidence_gate" if variant.supervision_mode == "metadata_ranking" else "learned_head_with_gate",
                        "score": raw_score,
                        "raw_score": raw_score,
                        "relevance_grade": (
                            labels[(query_id, candidate_id)]
                            if labels is not None
                            else None
                        ),
                        "hard_gate_passed": (
                            assessment.direct_reuse_gate_passed
                        ),
                        "structural_gate_passed": (
                            assessment.structural_gate_passed
                        ),
                        "direct_reuse_gate_failures": list(
                            assessment.direct_reuse_gate_failures
                        ),
                        "maximum_use_tier": assessment.maximum_use_tier,
                        "raw_decision_class": raw_decision_class,
                        "decision_class": decision_class,
                        "predicted_use_tier": next(
                            tier
                            for tier, value in USE_TIER_ORDER.items()
                            if value == decision_class
                        ),
                        "reuse_decision": decision_class == 0,
                        "adaptation_decision": decision_class == 1,
                        "predicted_adaptation_cost": float(
                            output["adaptation_cost"].cpu()
                            * ADAPTATION_COST_SCALE
                        ) if adaptation_enabled else None,
                        "adaptation_cost_source": (
                            "learned_from_versioned_adaptation_outcomes"
                            if variant.adaptation_weight > 0
                            and adaptation_supervision
                            else "not_predicted_by_ablation"
                            if variant.adaptation_weight == 0
                            else "unavailable_no_executed_adaptations"
                        ),
                        "adaptation_cost_label": (
                            float(supervision["weighted_cost"])
                            if supervision is not None
                            else None
                        ),
                        "adaptation_operation_probabilities": (
                            predicted_operation_probabilities
                        ),
                        "predicted_adaptation_operations": sorted(
                            operation
                            for operation, probability in (
                                predicted_operation_probabilities.items()
                            )
                            if probability >= 0.5
                        ) if adaptation_enabled else None,
                        "adaptation_operation_labels": (
                            sorted(supervision["operations"])
                            if supervision is not None
                            else None
                        ),
                        "predicted_uncertainty": None if variant.supervision_mode == "metadata_ranking" else float(output["uncertainty"].cpu()),
                    }
                )
    return rows


def run_training(
    *,
    benchmark_directory: Path,
    output_directory: Path,
    architectures: list[str],
    seeds: list[int],
    device: str,
    epochs: int,
    candidates_per_query: int,
    learning_rate: float,
    node_embedding_directory: Path | None = None,
    allow_hash_features: bool = False,
    resume: bool = False,
    variant_names: list[str] | None = None,
    supervision_mode: str = "scientific_use",
    full_candidate_training: bool = False,
    disable_early_stopping: bool = False,
    candidate_batch_size: int = 1,
) -> dict[str, Any]:
    started = time.perf_counter()
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be positive")
    benchmark_metadata = load_benchmark_metadata(benchmark_directory)
    variants = select_variants(architectures, variant_names)
    if supervision_mode not in {"scientific_use", "metadata_ranking"}:
        raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
    if supervision_mode == "metadata_ranking":
        variants = [replace(variant, supervision_mode=supervision_mode, decision_weight=0.0,
                            adaptation_weight=0.0, uncertainty_weight=0.0,
                            use_molecular_representation=False, use_late_molecular_fusion=False)
                    for variant in variants]
    for variant in variants:
        require_revision_training_contract(variant, benchmark_metadata)
        if candidate_batch_size > 1 and (
            not full_candidate_training or supervision_mode != "metadata_ranking"
            or variant.architecture != "gmn"
        ):
            raise ValueError("Candidate microbatches require full metadata ranking with GMN")
    queries, query_graphs, model_graphs, model_gate_graphs, labels = _load_benchmark(
        benchmark_directory
    )
    perturbed_queries, perturbed_query_graphs = _load_perturbation_queries(
        benchmark_directory
    )
    base_query_ids = {str(query["query_id"]) for query in queries}
    unknown_base_queries = {
        str(query["base_query_id"])
        for query in perturbed_queries
        if str(query["base_query_id"]) not in base_query_ids
    }
    if unknown_base_queries:
        raise ValueError(
            "Perturbed queries reference unknown base queries: "
            f"{sorted(unknown_base_queries)}"
        )
    adaptation_supervision = {} if supervision_mode == "metadata_ranking" else _load_adaptation_supervision(benchmark_directory)
    node_embeddings: dict[str, list[float]] | None = None
    embedding_manifest: dict[str, Any] | None = None
    if node_embedding_directory is not None:
        node_embeddings, embedding_manifest = load_embedding_store(
            node_embedding_directory
        )
        embedding_dimensions = int(embedding_manifest["dimensions"])
    else:
        embedding_dimensions = 128
        if not allow_hash_features:
            raise ValueError(
                "A versioned biomedical node embedding store is required. "
                "Use --allow-hash-features only for an explicitly named hash baseline."
            )
    feature_dimensions = (
        embedding_dimensions + AUXILIARY_NODE_FEATURE_DIMENSIONS
    )
    benchmark_input_paths = sorted(
        path
        for path in benchmark_directory.iterdir()
        if path.is_file()
        and path.name
        in {
            "experiment_manifest.json",
            "queries.jsonl",
            "query_graphs.jsonl",
            "model_graphs.jsonl",
            "model_gate_graphs.jsonl",
            "labels.jsonl",
            "perturbed_queries.jsonl",
            "perturbed_query_graphs.jsonl",
        }
    )
    run_input_contract = {
        "supervision_mode": supervision_mode,
        "full_candidate_training": full_candidate_training,
        "disable_early_stopping": disable_early_stopping,
        "benchmark_file_sha256": {
            path.name: _file_sha256(path) for path in benchmark_input_paths
        },
        "node_embedding_manifest_sha256": (
            _canonical_sha256(embedding_manifest)
            if embedding_manifest is not None
            else None
        ),
        "node_embeddings_sha256": (
            embedding_manifest.get("node_embeddings_sha256")
            if embedding_manifest is not None
            else None
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=resume)
    contract_decision_weighting = _training_decision_weighting(
        supervision_mode=supervision_mode,
        queries=[query for query in queries if query["split"] == "train"],
        candidate_ids=audited_candidate_ids(
            model_graphs, allowed_splits={"train"}
        ),
        labels=labels,
    )
    tensor_cache: dict[
        tuple[bool, bool, bool, bool],
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = {}

    def graph_for_variant(
        graph: TypedGraph,
        variant: TrainingVariant,
    ) -> TypedGraph:
        value = graph
        if not variant.use_validation_features:
            value = _without_validation_nodes(value)
        if not variant.use_simulation_verification:
            value = _without_simulation_verification_nodes(value)
        if not variant.use_molecular_representation:
            value = _without_molecular_nodes(value)
        return value

    def tensors_for(
        variant: TrainingVariant,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        key = (
            variant.use_validation_features,
            variant.use_provenance_features,
            variant.use_molecular_representation,
            variant.use_simulation_verification,
        )
        if key not in tensor_cache:
            import torch

            tensor_device = torch.device(device)
            tensor_cache[key] = (
                {
                    graph_id: tensorize_graph(
                        graph_for_variant(graph, variant),
                        feature_dimensions=embedding_dimensions,
                        node_embeddings=node_embeddings,
                        include_provenance_features=variant.use_provenance_features,
                    ).to(tensor_device)
                    for graph_id, graph in query_graphs.items()
                },
                {
                    graph_id: tensorize_graph(
                        graph_for_variant(graph, variant),
                        feature_dimensions=embedding_dimensions,
                        node_embeddings=node_embeddings,
                        include_provenance_features=variant.use_provenance_features,
                    ).to(tensor_device)
                    for graph_id, graph in perturbed_query_graphs.items()
                },
                {
                    graph_id: tensorize_graph(
                        graph_for_variant(graph, variant),
                        feature_dimensions=embedding_dimensions,
                        node_embeddings=node_embeddings,
                        include_provenance_features=variant.use_provenance_features,
                    ).to(tensor_device)
                    for graph_id, graph in model_graphs.items()
                },
            )
        return tensor_cache[key]

    results: dict[str, Any] = {}
    for variant in variants:
        variant_directory = output_directory / variant.name
        query_tensors, perturbed_query_tensors, model_tensors = tensors_for(variant)
        seed_results = []
        for seed in seeds:
            expected_resume_contract = _build_resume_contract(
                variant=variant,
                seed=seed,
                device_type=str(device).split(":", 1)[0],
                epochs=epochs,
                candidates_per_query=candidates_per_query,
                learning_rate=learning_rate,
                feature_dimensions=feature_dimensions,
                decision_weighting=contract_decision_weighting,
                run_input_contract=run_input_contract,
            )
            completed = (
                _completed_seed_result(
                    variant,
                    seed,
                    variant_directory,
                    benchmark_metadata,
                    bool(perturbed_queries),
                    _canonical_sha256(expected_resume_contract),
                )
                if resume
                else None
            )
            seed_results.append(
                completed
                or _train_one(
                    variant=variant,
                    seed=seed,
                    queries=queries,
                    query_graphs=query_graphs,
                    perturbed_queries=perturbed_queries,
                    perturbed_query_graphs=perturbed_query_graphs,
                    model_graphs=model_graphs,
                    model_gate_graphs=model_gate_graphs,
                    query_tensors=query_tensors,
                    perturbed_query_tensors=perturbed_query_tensors,
                    model_tensors=model_tensors,
                    labels=labels,
                    adaptation_supervision=adaptation_supervision,
                    output_directory=variant_directory,
                    device_name=device,
                    epochs=epochs,
                    candidates_per_query=candidates_per_query,
                    learning_rate=learning_rate,
                    feature_dimensions=feature_dimensions,
                    benchmark_metadata=benchmark_metadata,
                    run_input_contract=run_input_contract,
                    resume_partial=resume,
                    full_candidate_training=full_candidate_training,
                    disable_early_stopping=disable_early_stopping,
                    candidate_batch_size=candidate_batch_size,
                )
            )
        results[variant.name] = seed_results
    manifest = {
        "trainer_version": TRAINER_VERSION,
        "benchmark_directory": str(benchmark_directory),
        **benchmark_metadata,
        "architectures": architectures,
        "requested_variant_names": variant_names,
        "variants": [asdict(variant) for variant in variants],
        "seeds": seeds,
        "device": device,
        "epochs": epochs,
        "candidates_per_query": candidates_per_query,
        "full_candidate_training": full_candidate_training,
        "candidate_sampling_policy": "all_candidates_within_audited_split" if full_candidate_training else "sampled_candidates_within_audited_split",
        "candidate_batch_size": candidate_batch_size,
        "batching_policy": (
            "candidate_microbatches_one_full_listwise_update_per_query"
            if full_candidate_training else "historical_variant_optimization_policy"
        ),
        "disable_early_stopping": disable_early_stopping,
        "learning_rate": learning_rate,
        "node_feature_source": (
            "biomedical_embedding_store"
            if embedding_manifest is not None
            else "deterministic_hash_baseline"
        ),
        "node_embedding_manifest": embedding_manifest,
        "run_input_contract": run_input_contract,
        "hard_gate_graph_source": (
            "canonical_model_card_graph"
            if (benchmark_directory / "model_gate_graphs.jsonl").is_file()
            else "legacy_model_graph_fallback"
        ),
        "feature_dimensions": feature_dimensions,
        "embedding_dimensions": embedding_dimensions,
        "auxiliary_node_feature_dimensions": (
            AUXILIARY_NODE_FEATURE_DIMENSIONS
        ),
        "tensor_cache_keys": [
            {
                "use_validation_features": validation,
                "use_provenance_features": provenance,
                "use_molecular_representation": molecular,
                "use_simulation_verification": simulation,
            }
            for validation, provenance, molecular, simulation in sorted(
                tensor_cache
            )
        ],
        "adaptation_supervision_available": (
            bool(adaptation_supervision)
        ),
        "adaptation_supervision_pair_count": len(adaptation_supervision),
        "adaptation_operation_types": list(ADAPTATION_OPERATION_TYPES),
        "adaptation_cost_scale": ADAPTATION_COST_SCALE,
        "decision_weighting_policy": contract_decision_weighting["policy"],
        "validation_selection_policy": METADATA_VALIDATION_SELECTION_POLICY if supervision_mode == "metadata_ranking" else VALIDATION_SELECTION_POLICY,
        "perturbation_evaluation_available": bool(perturbed_queries),
        "perturbed_query_count": len(perturbed_queries),
        "uncertainty_semantics": (
            "No uncertainty head is trained or evaluated in metadata ranking."
            if supervision_mode == "metadata_ranking"
            else "Observed execution-outcome variation; not a calibrated clinical "
            "success probability."
            if benchmark_metadata["benchmark_level"] == "execution_derived"
            else "Candidate-retrieval label ambiguity; not epistemic uncertainty "
            "and not a calibrated success probability."
        ),
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
        "resumed": resume,
    }
    _atomic_json(output_directory / "training_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train public PBPK requirement-to-model graph matchers."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        action="append",
        choices=sorted(METHOD_FIDELITY),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--variant",
        action="append",
        help="Execute only the named preregistered variant; may be repeated.",
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--candidates-per-query", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=1,
                        help="Number of candidates computed together; does not subsample candidates.")
    parser.add_argument("--supervision-mode", choices=("scientific_use", "metadata_ranking"), default="scientific_use")
    parser.add_argument("--full-candidate-training", action="store_true")
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--node-embeddings", type=Path)
    parser.add_argument(
        "--allow-hash-features",
        action="store_true",
        help="Run an explicitly non-biomedical hash-feature baseline.",
    )
    args = parser.parse_args()
    manifest = run_training(
        benchmark_directory=args.benchmark,
        output_directory=args.output,
        architectures=args.architecture or list(DEFAULT_ARCHITECTURES),
        seeds=args.seed or [17, 29, 43, 59, 71],
        device=args.device,
        epochs=args.epochs,
        candidates_per_query=args.candidates_per_query,
        learning_rate=args.learning_rate,
        node_embedding_directory=args.node_embeddings,
        allow_hash_features=args.allow_hash_features,
        resume=args.resume,
        variant_names=args.variant,
        supervision_mode=args.supervision_mode,
        full_candidate_training=args.full_candidate_training,
        disable_early_stopping=args.disable_early_stopping,
        candidate_batch_size=args.candidate_batch_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
