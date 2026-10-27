from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid5

from pbpk_kb.recommendation.graph_prompt import (
    _components_for_evidence,
)
from pbpk_kb.recommendation.schemas import (
    Citation,
    ComponentRecommendation,
    CompoundRef,
    ModelingRequest,
    RecommendationResponse,
)


SOURCE_RUN_NAMESPACE = UUID("f24e0bb4-069a-4aa8-b270-99f424972604")
COMPOUND_NAMESPACE = UUID("5dfaa680-e80d-4b4f-95d4-804618cda0e8")
SUPPORTED_COMPONENTS = frozenset(
    {
        "physicochemical",
        "absorption",
        "distribution",
        "metabolism",
        "transport",
        "excretion",
        "ddi",
        "special_population",
        "public_model_template",
    }
)


class RequirementProfileSourceError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RequirementProfileSourceError(
                    f"Expected an object at {path}:{line_number}."
                )
            yield value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")
    os.replace(temporary, path)


def _manifest_artifact_sha(
    manifest: dict[str, Any],
    artifact_name: str,
) -> str:
    artifact = (manifest.get("artifacts") or {}).get(artifact_name)
    if not isinstance(artifact, dict):
        raise RequirementProfileSourceError(
            f"Dataset manifest has no {artifact_name!r} artifact."
        )
    value = str(artifact.get("sha256") or "")
    if len(value) != 64:
        raise RequirementProfileSourceError(
            f"Dataset manifest has an invalid {artifact_name!r} checksum."
        )
    return value


class EvidenceKGSnapshot:
    def __init__(self, graph_dir: Path) -> None:
        self.graph_dir = graph_dir
        self.nodes_path = graph_dir / "kg_nodes.jsonl"
        self.edges_path = graph_dir / "kg_edges.jsonl"
        self.manifest_path = graph_dir / "kg_extraction_manifest.json"
        for path in (
            self.nodes_path,
            self.edges_path,
            self.manifest_path,
        ):
            if not path.is_file():
                raise RequirementProfileSourceError(
                    f"Evidence KG artifact is missing: {path}"
                )

        self.node_by_id = {
            str(row["node_id"]): row
            for row in _iter_jsonl(self.nodes_path)
        }
        self.edges_by_evidence: dict[str, list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for edge in _iter_jsonl(self.edges_path):
            provenance = edge.get("provenance") or {}
            evidence_id = str(provenance.get("evidence_id") or "")
            if evidence_id:
                self.edges_by_evidence[evidence_id].append(edge)
        self._component_proxy = SimpleNamespace(
            node_by_id=self.node_by_id,
            edges_by_evidence=self.edges_by_evidence,
        )

    @property
    def manifest_sha256(self) -> str:
        return _sha256_file(self.manifest_path)

    def compound_ref(
        self,
        *,
        canonical_node_id: str,
        request: ModelingRequest,
    ) -> CompoundRef:
        node = self.node_by_id.get(canonical_node_id)
        if node is None or node.get("node_type") != "Compound":
            raise RequirementProfileSourceError(
                f"Compound node is absent from Evidence KG: "
                f"{canonical_node_id}"
            )
        preferred_name = str(node.get("preferred_name") or "").strip()
        if not preferred_name:
            raise RequirementProfileSourceError(
                f"Compound node has no preferred name: {canonical_node_id}"
            )
        standard_identifiers = {
            str(key): str(value)
            for key, value in (
                node.get("standard_identifiers") or {}
            ).items()
            if value is not None and str(value).strip()
        }
        standard_identifiers["EVIDENCE_KG_NODE"] = canonical_node_id
        compound_id: UUID | None = None
        for candidate in (
            standard_identifiers.get("PBPKKB"),
            canonical_node_id.removeprefix("Compound:"),
        ):
            if not candidate:
                continue
            try:
                compound_id = UUID(candidate)
            except ValueError:
                continue
            break
        if compound_id is None:
            compound_id = uuid5(
                COMPOUND_NAMESPACE,
                canonical_node_id,
            )
        return CompoundRef(
            compound_id=compound_id,
            preferred_name=preferred_name,
            active_moiety=request.active_moiety or preferred_name,
            salt_form=request.salt_form,
            external_identifiers=standard_identifiers,
            identity_resolution_method="evidence_kg_canonical_node",
            identity_confidence=1.0,
        )

    def _validated_provenance(
        self,
        evidence_id: str,
    ) -> dict[str, Any]:
        try:
            UUID(evidence_id)
        except ValueError as exc:
            raise RequirementProfileSourceError(
                f"Evidence ID is not a UUID: {evidence_id!r}"
            ) from exc
        node = self.node_by_id.get(evidence_id)
        if node is None:
            raise RequirementProfileSourceError(
                f"Evidence node is absent from Evidence KG: {evidence_id}"
            )
        candidates = [
            edge.get("provenance") or {}
            for edge in self.edges_by_evidence.get(evidence_id, ())
            if (edge.get("provenance") or {}).get("quality_status")
            == "auto_validated"
            and (edge.get("provenance") or {}).get("negation_status")
            != "negated"
        ]
        if not candidates:
            raise RequirementProfileSourceError(
                f"Evidence has no auto-validated provenance: {evidence_id}"
            )
        candidates.sort(
            key=lambda row: (
                -float(row.get("confidence") or 0.0),
                str(row.get("source_id") or ""),
                str(row.get("source_location") or ""),
            )
        )
        provenance = candidates[0]
        for field in ("source_id", "source_location", "supporting_text"):
            if not str(provenance.get(field) or "").strip():
                raise RequirementProfileSourceError(
                    f"Evidence provenance lacks {field}: {evidence_id}"
                )
        return provenance

    def components_for_evidence(self, evidence_id: str) -> set[str]:
        self._validated_provenance(evidence_id)
        return set(
            _components_for_evidence(
                self._component_proxy,
                evidence_id,
            )
        )

    def citation(self, evidence_id: str) -> Citation:
        provenance = self._validated_provenance(evidence_id)
        node = self.node_by_id[evidence_id]
        preferred_name = str(node.get("preferred_name") or "")
        parameter = (
            preferred_name.split(":", 1)[1]
            if ":" in preferred_name
            else None
        )
        return Citation(
            evidence_id=evidence_id,
            source_id=str(provenance["source_id"]),
            source_location=str(provenance["source_location"]),
            source_text=str(provenance["supporting_text"]),
            parameter=parameter,
            confidence=float(provenance.get("confidence") or 0.0),
        )


def _component_recommendation(
    *,
    request: ModelingRequest,
    compound: CompoundRef,
    target: dict[str, Any],
    snapshot: EvidenceKGSnapshot,
) -> RecommendationResponse:
    evidence_ids = sorted(
        {str(value) for value in target.get("evidence_ids") or ()}
    )
    evidence_components = {
        evidence_id: snapshot.components_for_evidence(evidence_id)
        for evidence_id in evidence_ids
    }
    components: list[ComponentRecommendation] = []
    grounded_component_count = 0
    all_confidences: list[float] = []
    for component in sorted(
        {
            str(value)
            for value in target.get("component_ids") or ()
            if str(value) in SUPPORTED_COMPONENTS
        }
    ):
        component_evidence_ids = [
            evidence_id
            for evidence_id in evidence_ids
            if component in evidence_components[evidence_id]
        ]
        citations = [
            snapshot.citation(evidence_id)
            for evidence_id in component_evidence_ids
        ]
        confidences = [
            float(citation.confidence or 0.0)
            for citation in citations
        ]
        if citations:
            grounded_component_count += 1
            all_confidences.extend(confidences)
        components.append(
            ComponentRecommendation(
                component=component,
                status=(
                    "recommended"
                    if citations
                    else "insufficient_evidence"
                ),
                rationale=(
                    "Mapped from auto-validated Evidence KG relations."
                    if citations
                    else "The locked target has no component-specific "
                    "auto-validated evidence reference."
                ),
                citations=citations,
                evidence_count=len(citations),
                source_count=len(
                    {citation.source_id for citation in citations}
                ),
                confidence=(
                    min(confidences) if confidences else 0.0
                ),
            )
        )
    return RecommendationResponse(
        compound_id=str(compound.compound_id),
        compound_name=compound.preferred_name,
        objective=request.objective,
        retrieval_policy=(
            "locked_silver_target_with_auto_validated_evidence_kg"
        ),
        coverage_score=(
            grounded_component_count / len(components)
            if components
            else 0.0
        ),
        overall_confidence=(
            min(all_confidences) if all_confidences else 0.0
        ),
        components=components,
        ranked_subgraphs=[],
        public_model_candidates=[],
        evidence_gaps=[
            str(value) for value in target.get("gap_codes") or ()
        ],
        warnings=[
            "Silver benchmark source; not an expert gold label."
        ],
        generated_from_auto_validated_only=True,
    )


def build_profile_bundles(
    *,
    silver_path: Path,
    dataset_manifest_path: Path,
    graph_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("accepted") is not True:
        raise RequirementProfileSourceError(
            "Requirement profiles require an accepted source dataset."
        )
    if manifest.get("benchmark_level") != "silver":
        raise RequirementProfileSourceError(
            "The source benchmark level must be explicitly silver."
        )
    expected_silver_sha = _manifest_artifact_sha(manifest, "silver")
    actual_silver_sha = _sha256_file(silver_path)
    if actual_silver_sha != expected_silver_sha:
        raise RequirementProfileSourceError(
            "Silver dataset checksum differs from its manifest."
        )

    snapshot = EvidenceKGSnapshot(graph_dir)
    expected_graph_sha = str(
        (manifest.get("input_hashes") or {}).get("graph_manifest") or ""
    )
    if snapshot.manifest_sha256 != expected_graph_sha:
        raise RequirementProfileSourceError(
            "Evidence KG manifest checksum differs from the source dataset."
        )
    source_run_id = uuid5(
        SOURCE_RUN_NAMESPACE,
        f"{actual_silver_sha}:{snapshot.manifest_sha256}",
    )
    bundles: list[dict[str, Any]] = []
    seen_examples: set[str] = set()
    for row in _iter_jsonl(silver_path):
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in seen_examples:
            raise RequirementProfileSourceError(
                f"Missing or duplicate example_id: {example_id!r}"
            )
        seen_examples.add(example_id)
        request = ModelingRequest.model_validate(
            row["modeling_request"]
        )
        compound = snapshot.compound_ref(
            canonical_node_id=str(row["compound_id"]),
            request=request,
        )
        recommendation = _component_recommendation(
            request=request,
            compound=compound,
            target=row["target"],
            snapshot=snapshot,
        )
        bundles.append(
            {
                "source_example_id": example_id,
                "benchmark_level": "silver",
                "benchmark_split": str(row.get("split") or ""),
                "modeling_request": request.model_dump(mode="json"),
                "compound_identity": compound.model_dump(mode="json"),
                "source_evidence_run_id": str(source_run_id),
                "component_recommendation": (
                    recommendation.model_dump(mode="json")
                ),
                "conflicts": [],
            }
        )

    if not bundles:
        raise RequirementProfileSourceError(
            "The source dataset contains no recommendation requests."
        )
    source_manifest = {
        "source_adapter_version": "1.0.0",
        "benchmark_level": "silver",
        "gold_claim": False,
        "clinical_validity_claim": False,
        "source_evidence_run_id": str(source_run_id),
        "bundle_count": len(bundles),
        "compound_count": len(
            {
                bundle["compound_identity"]["compound_id"]
                for bundle in bundles
            }
        ),
        "objective_counts": {
            objective: sum(
                bundle["modeling_request"]["objective"] == objective
                for bundle in bundles
            )
            for objective in sorted(
                {
                    bundle["modeling_request"]["objective"]
                    for bundle in bundles
                }
            )
        },
        "input_artifacts": {
            "silver": {
                "path": str(silver_path),
                "sha256": actual_silver_sha,
            },
            "dataset_manifest": {
                "path": str(dataset_manifest_path),
                "sha256": _sha256_file(dataset_manifest_path),
            },
            "evidence_kg_manifest": {
                "path": str(snapshot.manifest_path),
                "sha256": snapshot.manifest_sha256,
            },
            "evidence_kg_nodes": {
                "path": str(snapshot.nodes_path),
                "sha256": _sha256_file(snapshot.nodes_path),
            },
            "evidence_kg_edges": {
                "path": str(snapshot.edges_path),
                "sha256": _sha256_file(snapshot.edges_path),
            },
        },
    }
    return bundles, source_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt a locked Evidence KG silver dataset into "
            "RequirementProfile source bundles."
        )
    )
    parser.add_argument("--silver", type=Path, required=True)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundles, manifest = build_profile_bundles(
        silver_path=args.silver,
        dataset_manifest_path=args.dataset_manifest,
        graph_dir=args.graph_dir,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    _atomic_jsonl(args.output / "profile_bundles.jsonl", bundles)
    _atomic_json(args.output / "source_manifest.json", manifest)


if __name__ == "__main__":
    main()
