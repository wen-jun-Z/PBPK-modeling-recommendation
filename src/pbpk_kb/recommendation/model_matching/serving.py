from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from pbpk_kb.paths import PROJECT_ROOT, resolve_recorded_path
from pbpk_kb.recommendation.model_matching.feature_store import load_embedding_store
from pbpk_kb.recommendation.model_matching.graphs import TypedGraph
from pbpk_kb.recommendation.model_matching.neural import (
    ADAPTATION_OPERATION_TYPES,
    AUXILIARY_NODE_FEATURE_DIMENSIONS,
    build_matcher,
    tensorize_graph,
)
from pbpk_kb.recommendation.model_matching.training import (
    ADAPTATION_COST_SCALE,
)
from pbpk_kb.recommendation.model_matching.scoring import (
    USE_TIER_ORDER,
    apply_use_tier_gate,
    assess_candidate_use,
    score_candidate,
)


SERVING_VERSION = "pbpk-model-matcher-serving-1.4.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    return resolve_recorded_path(value, base=base)


@dataclass(slots=True)
class LearnedServingMatcher:
    manifest: dict[str, Any]
    model: Any
    node_embeddings: Mapping[str, list[float]]
    embedding_dimensions: int
    device: Any

    def score(self, query: TypedGraph, candidate: TypedGraph) -> dict[str, Any]:
        import torch

        missing = sorted(
            {
                node.semantic_key
                for graph in (query, candidate)
                for node in graph.nodes
                if node.semantic_key not in self.node_embeddings
            }
        )
        if missing:
            raise ValueError(
                "Serving feature store does not cover graph nodes: "
                + ", ".join(missing[:10])
            )
        query_tensor = tensorize_graph(
            query,
            feature_dimensions=self.embedding_dimensions,
            node_embeddings=self.node_embeddings,
        ).to(self.device)
        candidate_tensor = tensorize_graph(
            candidate,
            feature_dimensions=self.embedding_dimensions,
            node_embeddings=self.node_embeddings,
        ).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            output = self.model(query_tensor, candidate_tensor)
        deterministic = score_candidate(query, candidate)
        assessment = assess_candidate_use(
            query,
            candidate,
            match=deterministic,
        )
        result = deterministic.to_dict()
        adaptation_supervised = bool(
            self.manifest.get("adaptation_supervision_available")
        )
        operation_probabilities = (
            {
                operation: float(probability)
                for operation, probability in zip(
                    ADAPTATION_OPERATION_TYPES,
                    torch.sigmoid(output["adaptation_operation_logits"])
                    .detach()
                    .cpu()
                    .tolist(),
                    strict=True,
                )
            }
            if adaptation_supervised
            else {}
        )
        raw_decision_class = int(output["decision_logits"].argmax().cpu())
        decision_class = apply_use_tier_gate(
            raw_decision_class,
            assessment,
            enabled=True,
        )
        result.update(
            {
                "score": float(output["score"].cpu()),
                "raw_ranking_score": float(output["score"].cpu()),
                "adaptation_cost": (
                    float(output["adaptation_cost"].cpu())
                    * ADAPTATION_COST_SCALE
                    if adaptation_supervised
                    else result["adaptation_cost"]
                ),
                "adaptation_cost_source": (
                    "learned_from_versioned_adaptation_outcomes"
                    if adaptation_supervised
                    else "pbpk_domain_edit_cost"
                ),
                "adaptation_operation_probabilities": (
                    operation_probabilities
                ),
                "predicted_adaptation_operations": sorted(
                    operation
                    for operation, probability in operation_probabilities.items()
                    if probability >= 0.5
                ),
                "uncertainty": float(output["uncertainty"].cpu()),
                "structural_gate_passed": (
                    assessment.structural_gate_passed
                ),
                "hard_gate_passed": (
                    assessment.direct_reuse_gate_passed
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
                "matching_method": self.manifest["method"],
                "learned": True,
                "checkpoint_sha256": self.manifest["checkpoint_sha256"],
                "benchmark_level": self.manifest["benchmark_level"],
                "serving_scope": self.manifest["serving_scope"],
            }
        )
        return result


def _default_manifest() -> Path | None:
    configured = os.getenv("PBPK_MODEL_MATCHER_SERVING_MANIFEST", "").strip()
    if configured:
        return Path(configured)
    project_root = PROJECT_ROOT
    manifests = sorted(
        (project_root / "data_process/model_matching").glob(
            "serving_*/serving_manifest.json"
        ),
        reverse=True,
    )
    for path in manifests:
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (
                manifest.get("status") == "accepted"
                and manifest.get("benchmark_level") == "execution_derived"
                and manifest.get("serving_scope") == "validated"
                and manifest.get("scientific_claim_eligible") is True
            ):
                return path
        except (OSError, json.JSONDecodeError):
            continue
    return None


@lru_cache(maxsize=4)
def _load(path_string: str) -> LearnedServingMatcher:
    import torch

    path = Path(path_string)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "accepted":
        raise ValueError("Serving manifest status must be accepted.")
    if (
        manifest.get("benchmark_level") != "execution_derived"
        or manifest.get("serving_scope") != "validated"
        or manifest.get("scientific_claim_eligible") is not True
    ):
        raise ValueError(
            "Learned serving requires a scientifically eligible execution-derived benchmark."
        )
    checkpoint_path = _resolve(path.parent, str(manifest["checkpoint_path"]))
    if _sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("Serving checkpoint checksum mismatch.")
    feature_directory = _resolve(path.parent, str(manifest["feature_store_path"]))
    embeddings, feature_manifest = load_embedding_store(feature_directory)
    if feature_manifest["node_embeddings_sha256"] != manifest[
        "feature_store_sha256"
    ]:
        raise ValueError("Serving feature-store checksum mismatch.")
    device = torch.device(
        os.getenv(
            "PBPK_MODEL_MATCHER_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = str(checkpoint["variant"]["architecture"])
    variant = checkpoint["variant"]
    architecture_flags = {
        "cg_hhgm": ("use_cross_graph_attention",),
    }
    if architecture not in architecture_flags or any(
        not bool(variant.get(field, True))
        for field in (
            "use_hard_gate",
            "use_relation_typing",
            "use_pbpk_hierarchy",
            "use_validation_features",
            "use_provenance_features",
            "use_molecular_representation",
            "use_simulation_verification",
        )
        + architecture_flags.get(architecture, ())
    ):
        raise ValueError(
            "Only a full, hard-gated proposed matcher variant may be served."
        )
    model = build_matcher(
        architecture,
        feature_dimensions=(
            int(feature_manifest["dimensions"])
            + AUXILIARY_NODE_FEATURE_DIMENSIONS
        ),
        use_relation_typing=bool(variant.get("use_relation_typing", True)),
        use_cross_graph_attention=bool(
            variant.get("use_cross_graph_attention", True)
        ),
        use_pbpk_hierarchy=bool(variant.get("use_pbpk_hierarchy", True)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return LearnedServingMatcher(
        manifest=manifest,
        model=model,
        node_embeddings=embeddings,
        embedding_dimensions=int(feature_manifest["dimensions"]),
        device=device,
    )


def serving_status() -> dict[str, Any]:
    path = _default_manifest()
    if path is None or not path.is_file():
        return {
            "available": False,
            "serving_version": SERVING_VERSION,
            "method": "cg_hhgm_deterministic_fallback",
            "learned": False,
            "benchmark_level": "not_validated",
            "reason": "No accepted serving manifest is available.",
        }
    try:
        matcher = _load(str(path.resolve()))
    except Exception as exc:
        return {
            "available": False,
            "serving_version": SERVING_VERSION,
            "method": "cg_hhgm_deterministic_fallback",
            "learned": False,
            "benchmark_level": "not_validated",
            "manifest_path": str(path),
            "reason": str(exc),
        }
    return {
        "available": True,
        "serving_version": SERVING_VERSION,
        "method": matcher.manifest["method"],
        "learned": True,
        "benchmark_level": matcher.manifest["benchmark_level"],
        "serving_scope": matcher.manifest["serving_scope"],
        "checkpoint_sha256": matcher.manifest["checkpoint_sha256"],
        "manifest_path": str(path),
    }


def score_for_serving(query: TypedGraph, candidate: TypedGraph) -> dict[str, Any]:
    failure_reason = None
    path = _default_manifest()
    if path is not None and path.is_file():
        try:
            return _load(str(path.resolve())).score(query, candidate)
        except Exception as exc:
            failure_reason = str(exc)
    deterministic = score_candidate(query, candidate)
    assessment = assess_candidate_use(
        query,
        candidate,
        match=deterministic,
    )
    value = deterministic.to_dict()
    reuse_decision = bool(
        assessment.direct_reuse_gate_passed
        and deterministic.score >= 0.90
    )
    value.update(
        {
            "raw_ranking_score": deterministic.score,
            "structural_gate_passed": assessment.structural_gate_passed,
            "hard_gate_passed": assessment.direct_reuse_gate_passed,
            "direct_reuse_gate_failures": list(
                assessment.direct_reuse_gate_failures
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
            "matching_method": "cg_hhgm_deterministic_fallback",
            "learned": False,
            "benchmark_level": "not_validated",
            "serving_scope": "deterministic_fallback",
            "serving_fallback_reason": failure_reason,
        }
    )
    return value
