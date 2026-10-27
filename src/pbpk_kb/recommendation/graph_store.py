from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.paths import recommendation_quality_dir

DEFAULT_QUALITY_RUN = recommendation_quality_dir()


class RecommendationGraphStore:
    def __init__(self, quality_dir: Path | None = None) -> None:
        self.quality_dir = quality_dir or configured_quality_dir()
        self.nodes = load_jsonl(self.quality_dir / "kg_ready_nodes.jsonl")
        self.edges = load_jsonl(self.quality_dir / "kg_ready_edges.jsonl")
        self.evidence = load_jsonl(self.quality_dir / "final_auto_validated_evidence.jsonl")
        self.node_by_key = {
            (str(row["node_type"]), str(row["node_id"])): row
            for row in self.nodes
        }
        self.evidence_by_id = {str(row["evidence_id"]): row for row in self.evidence}
        self.outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in self.edges:
            self.outgoing[str(edge["from_id"])].append(edge)
            self.incoming[str(edge["to_id"])].append(edge)
        self.compounds = self._compound_index()

    def resolve_compound(self, value: str) -> dict[str, Any] | None:
        normalized = normalize(value)
        exact = self.compounds.get(normalized)
        if exact:
            return exact
        matches = [row for name, row in self.compounds.items() if normalized in name or name in normalized]
        return matches[0] if len(matches) == 1 else None

    def compound_evidence(self, compound_id: str) -> list[dict[str, Any]]:
        return [
            row
            for row in self.evidence
            if str(row.get("compound_id") or row.get("attributed_compound_id") or "") == compound_id
            and row.get("quality_status") == "auto_validated"
        ]

    def neighborhood(self, center_id: str, max_hops: int = 2) -> tuple[set[str], list[dict[str, Any]]]:
        visited = {center_id}
        frontier = {center_id}
        selected_edges: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node_id in frontier:
                for edge in self.outgoing.get(node_id, []) + self.incoming.get(node_id, []):
                    source = str(edge["from_id"])
                    target = str(edge["to_id"])
                    key = (source, str(edge["relationship_type"]), target, str((edge.get("properties") or {}).get("evidence_id") or ""))
                    selected_edges[key] = edge
                    next_frontier.update((source, target))
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return visited, list(selected_edges.values())

    def node_label(self, node_id: str) -> str:
        for (node_type, candidate_id), row in self.node_by_key.items():
            if candidate_id == node_id:
                return str((row.get("properties") or {}).get("name") or f"{node_type}:{node_id}")
        return node_id

    def _compound_index(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        known_ids = {
            str(row.get("compound_id") or "")
            for row in self.evidence
            if row.get("compound_id")
        }
        for row in self.nodes:
            if row.get("node_type") != "Compound" or str(row.get("node_id")) not in known_ids:
                continue
            name = str((row.get("properties") or {}).get("name") or "")
            if name:
                result[normalize(name)] = {
                    "compound_id": str(row["node_id"]),
                    "compound_name": name,
                }
        return result


def configured_quality_dir() -> Path:
    return recommendation_quality_dir()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Recommendation KG input not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


@lru_cache(maxsize=1)
def get_recommendation_store() -> RecommendationGraphStore:
    return RecommendationGraphStore()


def reset_recommendation_store() -> None:
    get_recommendation_store.cache_clear()
