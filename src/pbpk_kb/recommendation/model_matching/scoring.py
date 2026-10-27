from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from pbpk_kb.recommendation.model_matching.graphs import GraphNode, TypedGraph


SCORER_VERSION = "pbpk-model-matching-scorer-1.1.0"

USE_TIER_ORDER = {
    "direct_reuse": 0,
    "controlled_adaptation": 1,
    "mechanistic_reference": 2,
    "reject": 3,
}

HIERARCHY_WEIGHTS = {
    "identity": 0.24,
    "context": 0.20,
    "mechanism": 0.24,
    "structure": 0.17,
    "validation": 0.15,
}

EDIT_COSTS = {
    "Compound": 1.00,
    "Species": 1.00,
    "Population": 0.80,
    "Route": 0.75,
    "Formulation": 0.45,
    "ADMEProcess": 0.85,
    "Enzyme": 0.90,
    "Transporter": 0.90,
    "DDIRole": 0.95,
    "Compartment": 0.55,
    "Endpoint": 0.70,
    "ModelStructure": 0.80,
    "Software": 0.50,
    "Dataset": 0.45,
}

HARD_NODE_TYPES = {
    "Compound",
    "Species",
    "Route",
    "ADMEProcess",
    "Enzyme",
    "Transporter",
    "DDIRole",
    "Endpoint",
    "Software",
}


@dataclass(frozen=True, slots=True)
class AdaptationOperation:
    operation: str
    node_type: str
    semantic_key: str
    cost: float
    blocking_for_reuse: bool


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    scorer_version: str
    method: str
    score: float
    hard_gate_passed: bool
    hard_gate_failures: tuple[str, ...]
    hierarchy_scores: Mapping[str, float]
    matched_semantic_keys: tuple[str, ...]
    missing_semantic_keys: tuple[str, ...]
    conflicting_semantic_keys: tuple[str, ...]
    adaptation_cost: float
    adaptation_operations: tuple[AdaptationOperation, ...]
    uncertainty: float
    benchmark_eligible: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["adaptation_operations"] = [
            asdict(item) for item in self.adaptation_operations
        ]
        return value


@dataclass(frozen=True, slots=True)
class CandidateUseAssessment:
    maximum_use_tier: str
    direct_reuse_gate_passed: bool
    direct_reuse_gate_failures: tuple[str, ...]
    structural_gate_passed: bool
    same_active_moiety: bool
    engineering_candidate: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _by_type(graph: TypedGraph) -> dict[str, dict[str, GraphNode]]:
    grouped: dict[str, dict[str, GraphNode]] = defaultdict(dict)
    for node in graph.nodes:
        if node.hierarchy == "root":
            continue
        grouped[node.node_type][node.semantic_key] = node
    return grouped


def _set_score(required: set[str], available: set[str]) -> float:
    if not required:
        return 1.0
    return len(required & available) / len(required)


def _typed_alignment(
    requirement: TypedGraph,
    model: TypedGraph,
) -> tuple[
    dict[str, float],
    list[str],
    list[str],
    list[str],
    list[AdaptationOperation],
]:
    query_by_type = _by_type(requirement)
    model_by_type = _by_type(model)
    hierarchy_values: dict[str, list[float]] = defaultdict(list)
    matched: list[str] = []
    missing: list[str] = []
    conflicts: list[str] = []
    operations: list[AdaptationOperation] = []

    for node_type, query_nodes in query_by_type.items():
        if node_type in {"Objective", "Regimen"}:
            continue
        required = set(query_nodes)
        available = set(model_by_type.get(node_type, {}))
        overlap = sorted(required & available)
        absent = sorted(required - available)
        matched.extend(overlap)
        missing.extend(absent)
        hierarchy = next(iter(query_nodes.values())).hierarchy
        hierarchy_values[hierarchy].append(_set_score(required, available))
        cost = EDIT_COSTS.get(node_type, 0.4)
        for semantic in absent:
            blocking = node_type in HARD_NODE_TYPES and query_nodes[semantic].required
            operations.append(
                AdaptationOperation(
                    operation=(
                        "ADD_OR_VALIDATE_REQUIRED_CAPABILITY"
                        if blocking
                        else "REVIEW_OPTIONAL_CAPABILITY"
                    ),
                    node_type=node_type,
                    semantic_key=semantic,
                    cost=cost,
                    blocking_for_reuse=blocking,
                )
            )
        if required and available and not overlap and node_type in {
            "Compound",
            "Species",
            "Route",
            "Software",
        }:
            conflicts.extend(sorted(required))

    hierarchy_scores = {
        level: (
            sum(hierarchy_values.get(level, []))
            / len(hierarchy_values[level])
            if hierarchy_values.get(level)
            else 1.0
        )
        for level in HIERARCHY_WEIGHTS
    }
    return hierarchy_scores, matched, missing, conflicts, operations


def domain_ged_score(
    requirement: TypedGraph,
    model: TypedGraph,
) -> tuple[float, float, tuple[AdaptationOperation, ...]]:
    _, _, _, _, operations = _typed_alignment(requirement, model)
    query_nodes = [node for node in requirement.nodes if node.hierarchy != "root"]
    denominator = sum(EDIT_COSTS.get(node.node_type, 0.4) for node in query_nodes)
    edit_cost = sum(operation.cost for operation in operations)
    normalized = min(1.0, edit_cost / denominator) if denominator else 0.0
    return 1.0 - normalized, normalized, tuple(operations)


def hierarchical_match_score(
    requirement: TypedGraph,
    model: TypedGraph,
) -> tuple[float, dict[str, float]]:
    hierarchy, _, _, _, _ = _typed_alignment(requirement, model)
    score = sum(HIERARCHY_WEIGHTS[level] * hierarchy[level] for level in HIERARCHY_WEIGHTS)
    return score, hierarchy


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def graph_text(graph: TypedGraph) -> str:
    values = []
    for node in graph.nodes:
        if node.hierarchy == "root":
            continue
        values.extend([node.node_type, node.label, node.hierarchy])
    for edge in graph.edges:
        values.append(edge.predicate)
    return " ".join(values)


def bm25_scores(query: TypedGraph, candidates: Sequence[TypedGraph]) -> list[float]:
    documents = [_tokenize(graph_text(candidate)) for candidate in candidates]
    query_terms = _tokenize(graph_text(query))
    if not candidates:
        return []
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    average_length = sum(map(len, documents)) / max(1, len(documents))
    values: list[float] = []
    for document in documents:
        counts = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.2 * (
                1.0 - 0.75 + 0.75 * len(document) / max(1.0, average_length)
            )
            score += inverse * frequency * 2.2 / denominator
        values.append(score)
    maximum = max(values, default=0.0)
    return [value / maximum if maximum > 0 else 0.0 for value in values]


def cross_graph_attention_score(
    requirement: TypedGraph,
    model: TypedGraph,
) -> float:
    """Deterministic typed cross-attention baseline, not a learned GMN."""

    model_by_type = _by_type(model)
    scores: list[float] = []
    weights: list[float] = []
    for query_node in requirement.nodes:
        if query_node.hierarchy == "root" or query_node.node_type in {"Objective", "Regimen"}:
            continue
        candidates = model_by_type.get(query_node.node_type, {}).values()
        query_tokens = set(_tokenize(query_node.label))
        best = 0.0
        for candidate in candidates:
            candidate_tokens = set(_tokenize(candidate.label))
            union = query_tokens | candidate_tokens
            similarity = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
            if query_node.semantic_key == candidate.semantic_key:
                similarity = 1.0
            best = max(best, similarity)
        weight = EDIT_COSTS.get(query_node.node_type, 0.4)
        scores.append(best * weight)
        weights.append(weight)
    return sum(scores) / sum(weights) if weights else 0.0


def score_candidate(
    requirement: TypedGraph,
    model: TypedGraph,
    *,
    method: str = "cg_hhgm_deterministic_fallback",
) -> CandidateMatch:
    hierarchy, matched, missing, conflicts, operations = _typed_alignment(
        requirement, model
    )
    hard_failures = tuple(
        sorted(
            operation.semantic_key
            for operation in operations
            if operation.blocking_for_reuse
        )
    )
    hierarchy_score = sum(
        HIERARCHY_WEIGHTS[level] * hierarchy[level]
        for level in HIERARCHY_WEIGHTS
    )
    attention_score = cross_graph_attention_score(requirement, model)
    ged_score, adaptation_cost, ged_operations = domain_ged_score(requirement, model)
    uncertainty = min(
        1.0,
        0.65 * (len(missing) / max(1, len(matched) + len(missing)))
        + 0.35 * (len(conflicts) / max(1, len(missing))),
    )
    score = max(
        0.0,
        min(
            1.0,
            0.45 * hierarchy_score
            + 0.35 * attention_score
            + 0.20 * ged_score
            - 0.10 * uncertainty,
        ),
    )
    return CandidateMatch(
        scorer_version=SCORER_VERSION,
        method=method,
        score=score,
        hard_gate_passed=not hard_failures,
        hard_gate_failures=hard_failures,
        hierarchy_scores=hierarchy,
        matched_semantic_keys=tuple(sorted(set(matched))),
        missing_semantic_keys=tuple(sorted(set(missing))),
        conflicting_semantic_keys=tuple(sorted(set(conflicts))),
        adaptation_cost=adaptation_cost,
        adaptation_operations=ged_operations,
        uncertainty=uncertainty,
        notes=(
            "Deterministic fallback; no accepted learned checkpoint was loaded.",
        )
        if method == "cg_hhgm_deterministic_fallback"
        else (),
    )


def _node_labels(graph: TypedGraph, node_type: str) -> set[str]:
    return {
        " ".join(node.label.casefold().split())
        for node in graph.nodes
        if node.node_type == node_type and node.label.strip()
    }


def _metadata_or_node(
    model: TypedGraph,
    metadata_key: str,
    node_type: str,
    default: str = "unknown",
) -> str:
    metadata_value = model.metadata.get(metadata_key)
    if metadata_value is not None and str(metadata_value).strip():
        return " ".join(str(metadata_value).casefold().split())
    values = sorted(_node_labels(model, node_type))
    return values[0] if values else default


def assess_candidate_use(
    requirement: TypedGraph,
    model: TypedGraph,
    *,
    match: CandidateMatch | None = None,
) -> CandidateUseAssessment:
    """Determine the highest safe use tier without changing ranking scores."""

    candidate_match = match or score_candidate(requirement, model)
    failures = list(candidate_match.hard_gate_failures)
    license_status = _metadata_or_node(
        model, "license_status", "License"
    )
    parse_status = _metadata_or_node(
        model, "parse_status", "ParseStatus"
    )
    execution_status = _metadata_or_node(
        model, "execution_probe_status", "ExecutionStatus"
    )
    license_allowed = license_status in {
        "allowed",
        "approved",
        "permitted",
        "open source",
    }
    parse_passed = parse_status in {"parsed", "passed"}
    execution_passed = execution_status == "passed"
    engineering_value = model.metadata.get("engineering_accepted")
    engineering_candidate = (
        bool(engineering_value)
        if engineering_value is not None
        else execution_passed and parse_passed
    )
    reusable_value = model.metadata.get("reusable_candidate")
    reusable_candidate = (
        bool(reusable_value)
        if reusable_value is not None
        else engineering_candidate
    )
    if not license_allowed:
        failures.append(f"LicenseStatus:{license_status}")
    if not parse_passed:
        failures.append(f"ParseStatus:{parse_status}")
    if not execution_passed:
        failures.append(f"ExecutionStatus:{execution_status}")
    if not engineering_candidate:
        failures.append("EngineeringAudit:not_accepted")
    if not reusable_candidate:
        failures.append("ScientificReuse:not_validated")

    required_compounds = _node_labels(requirement, "Compound")
    model_compounds = _node_labels(model, "Compound")
    same_active_moiety = bool(required_compounds) and required_compounds.issubset(
        model_compounds
    )
    direct_reuse_failures = tuple(sorted(set(failures)))
    direct_reuse_passed = not direct_reuse_failures

    if direct_reuse_passed:
        maximum_use_tier = "direct_reuse"
    elif not (license_allowed and parse_passed and execution_passed and engineering_candidate):
        maximum_use_tier = "reject"
    else:
        blocking_types = {
            operation.node_type
            for operation in candidate_match.adaptation_operations
            if operation.blocking_for_reuse
        }
        fundamental_conflicts = blocking_types & {
            "Species",
            "Software",
            "ADMEProcess",
            "Enzyme",
            "Transporter",
            "DDIRole",
        }
        mechanism_types = {"ADMEProcess", "Enzyme", "Transporter", "DDIRole"}
        mechanism_overlap = any(
            _node_labels(requirement, node_type)
            & _node_labels(model, node_type)
            for node_type in mechanism_types
        )
        if same_active_moiety and not fundamental_conflicts:
            maximum_use_tier = "controlled_adaptation"
        elif same_active_moiety or mechanism_overlap:
            maximum_use_tier = "mechanistic_reference"
        else:
            maximum_use_tier = "reject"

    return CandidateUseAssessment(
        maximum_use_tier=maximum_use_tier,
        direct_reuse_gate_passed=direct_reuse_passed,
        direct_reuse_gate_failures=direct_reuse_failures,
        structural_gate_passed=candidate_match.hard_gate_passed,
        same_active_moiety=same_active_moiety,
        engineering_candidate=engineering_candidate,
    )


def apply_use_tier_gate(
    predicted_class: int,
    assessment: CandidateUseAssessment,
    *,
    enabled: bool = True,
) -> int:
    if predicted_class not in set(USE_TIER_ORDER.values()):
        raise ValueError(f"Unknown use-tier class: {predicted_class}")
    if not enabled:
        return predicted_class
    return max(predicted_class, USE_TIER_ORDER[assessment.maximum_use_tier])


def deterministic_use_class(assessment: CandidateUseAssessment) -> int:
    """Separate score/order from a conservative use decision.

    An adaptation ceiling is permission to investigate, not proof that an
    adaptation worked. Without an executed, validated adaptation this policy
    returns reference-only. It does not use rank-score thresholds or test labels.
    """
    if assessment.direct_reuse_gate_passed:
        return USE_TIER_ORDER["direct_reuse"]
    if assessment.maximum_use_tier == "reject":
        return USE_TIER_ORDER["reject"]
    return USE_TIER_ORDER["mechanistic_reference"]


def rank_validated_candidates(
    scores: Mapping[str, float], assessments: Mapping[str, CandidateUseAssessment]
) -> list[str]:
    """Place verified reuse candidates first, retaining score order inside each pool.

    An arbitrary learned rank score is not a success probability. Multiple
    qualifying candidates are retained; neither scores nor a ceiling of possible
    adaptation provide scientific permission to reuse an unvalidated model.
    """
    if set(scores) != set(assessments):
        raise ValueError("Every scored candidate requires exactly one use assessment.")
    if any(not math.isfinite(float(score)) for score in scores.values()):
        raise ValueError("Candidate scores must be finite.")
    return sorted(scores, key=lambda key: (
        not assessments[key].direct_reuse_gate_passed, -float(scores[key]), key
    ))
