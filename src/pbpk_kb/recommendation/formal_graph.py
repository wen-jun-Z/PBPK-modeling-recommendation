from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.ontology import (
    ONTOLOGY_VERSION,
    RELATIONS,
    RELATION_GLOSSES,
    RELATION_SEARCH_TERMS,
    pk_parameter_semantics,
)


FORMAL_GRAPH_VERSION = "pbpk-formal-graph-1.1.0"
SUBGRAPH_SCHEMA_VERSION = "pbpk-subgraph-1.4.0"

CENTER_TYPES = frozenset(
    {
        "Compound",
        "Enzyme",
        "Transporter",
        "PKParameter",
        "Population",
        "AdministrationRoute",
        "PBPKModel",
    }
)

DEFAULT_RELATIONS = frozenset(RELATIONS)
OBJECTIVE_RELATIONS: dict[str, frozenset[str]] = {
    "ddi": frozenset(
        {
            "VICTIM_IN",
            "PERPETRATOR_IN",
            "MEDIATED_BY",
            "AFFECTS_PARAMETER",
            "SUBSTRATE_OF",
            "INHIBITS",
            "INDUCES",
            "UNDER_REGIMEN",
            "STUDIED_IN",
            "HAS_VALUE",
            "HAS_UNIT",
            "SUPPORTED_BY",
            "DERIVED_FROM",
        }
    ),
    "ddi_victim": frozenset(
        {
            "VICTIM_IN",
            "PERPETRATOR_IN",
            "MEDIATED_BY",
            "AFFECTS_PARAMETER",
            "SUBSTRATE_OF",
            "INHIBITS",
            "INDUCES",
            "UNDER_REGIMEN",
            "STUDIED_IN",
            "HAS_VALUE",
            "HAS_UNIT",
            "SUPPORTED_BY",
            "DERIVED_FROM",
        }
    ),
    "ddi_perpetrator": frozenset(
        {
            "PERPETRATOR_IN",
            "VICTIM_IN",
            "MEDIATED_BY",
            "AFFECTS_PARAMETER",
            "SUBSTRATE_OF",
            "INHIBITS",
            "INDUCES",
            "UNDER_REGIMEN",
            "STUDIED_IN",
            "HAS_VALUE",
            "HAS_UNIT",
            "SUPPORTED_BY",
            "DERIVED_FROM",
        }
    ),
    "public_model_reuse": frozenset(
        {
            "DESCRIBED_BY_MODEL",
            "HAS_MODEL_VERSION",
            "HAS_SIMULATION",
            "HAS_VALIDATION_RUN",
            "USES_STRUCTURE",
            "HAS_COMPARTMENT",
            "USES_PARAMETER",
            "USES_ENZYME",
            "USES_TRANSPORTER",
            "SUPPORTS_ROUTE",
            "SUPPORTS_FORMULATION",
            "APPLIES_TO",
            "CALIBRATED_WITH",
            "VALIDATED_WITH",
            "HAS_VALIDATION_METRIC",
            "EVALUATES_DATASET",
            "SUPPORTED_BY",
            "DERIVED_FROM",
        }
    ),
    "special_population": frozenset(
        {
            "STUDIED_IN",
            "OF_COMPOUND",
            "MEASURES",
            "HAS_MEASUREMENT",
            "HAS_VALUE",
            "HAS_UNIT",
            "UNDER_REGIMEN",
            "SUPPORTED_BY",
            "DERIVED_FROM",
        }
    ),
    "renal_impairment": frozenset(
        {
            "STUDIED_IN",
            "OF_COMPOUND",
            "MEASURES",
            "HAS_MEASUREMENT",
            "HAS_VALUE",
            "HAS_UNIT",
            "EXCRETED_VIA",
            "UNDER_REGIMEN",
            "SUPPORTED_BY",
            "DERIVED_FROM",
        }
    ),
    "hepatic_impairment": frozenset(
        {
            "STUDIED_IN",
            "OF_COMPOUND",
            "MEASURES",
            "HAS_MEASUREMENT",
            "HAS_VALUE",
            "HAS_UNIT",
            "SUBSTRATE_OF",
            "MEDIATED_BY",
            "UNDER_REGIMEN",
            "SUPPORTED_BY",
            "DERIVED_FROM",
        }
    ),
}

RELATION_PRIORITY = {
    name: index
    for index, name in enumerate(
        (
            "OF_COMPOUND",
            "VICTIM_IN",
            "PERPETRATOR_IN",
            "MEASURES",
            "AFFECTS_PARAMETER",
            "HAS_VALUE",
            "HAS_UNIT",
            "MEDIATED_BY",
            "SUBSTRATE_OF",
            "INHIBITS",
            "INDUCES",
            "TRANSPORTED_BY",
            "ABSORBED_VIA",
            "EXCRETED_VIA",
            "UNDER_REGIMEN",
            "HAS_ROUTE",
            "HAS_FORMULATION",
            "STUDIED_IN",
            "DESCRIBED_BY_MODEL",
            "HAS_MODEL_VERSION",
            "HAS_SIMULATION",
            "HAS_VALIDATION_RUN",
            "USES_STRUCTURE",
            "HAS_COMPARTMENT",
            "USES_PARAMETER",
            "USES_ENZYME",
            "USES_TRANSPORTER",
            "SUPPORTS_ROUTE",
            "SUPPORTS_FORMULATION",
            "APPLIES_TO",
            "CALIBRATED_WITH",
            "VALIDATED_WITH",
            "HAS_VALIDATION_METRIC",
            "EVALUATES_DATASET",
            "HAS_MEASUREMENT",
            "REPORTS",
            "SUPPORTED_BY",
            "DERIVED_FROM",
            "CONTRADICTS",
            "HAS_CONFLICT_MEMBER",
        )
    )
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(namespace: str, *parts: Any) -> str:
    body = "\x1f".join(str(part) for part in parts)
    return f"{namespace}:{hashlib.sha256(body.encode('utf-8')).hexdigest()[:24]}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class TraversalPolicy:
    max_hops: int = 2
    max_neighbors_per_relation: int = 24
    max_evidence_per_source: int = 6
    max_nodes: int = 256
    max_edges: int = 512
    minimum_confidence: float = 0.6

    def __post_init__(self) -> None:
        if self.max_hops not in {0, 1, 2}:
            raise ValueError("Only 0-hop, 1-hop, and 2-hop traversal is supported.")
        for value in (
            self.max_neighbors_per_relation,
            self.max_evidence_per_source,
            self.max_nodes,
            self.max_edges,
        ):
            if value < 1:
                raise ValueError("Traversal limits must be positive.")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1.")


class FormalGraphStore:
    """Validated, immutable view over the formal Gate 1 graph."""

    def __init__(self, graph_dir: Path, gate1_audit_path: Path | None = None) -> None:
        self.graph_dir = graph_dir
        self.paths = {
            "nodes": graph_dir / "kg_nodes.jsonl",
            "edges": graph_dir / "kg_edges.jsonl",
            "events": graph_dir / "kg_events.jsonl",
            "manifest": graph_dir / "kg_extraction_manifest.json",
            "summary": graph_dir / "kg_build_summary.json",
        }
        for path in self.paths.values():
            if not path.exists():
                raise FileNotFoundError(f"Formal KG input not found: {path}")

        self.nodes = load_jsonl(self.paths["nodes"])
        self.edges = load_jsonl(self.paths["edges"])
        self.events = load_jsonl(self.paths["events"])
        self.manifest = load_json(self.paths["manifest"])
        self.summary = load_json(self.paths["summary"])
        self.gate1_audit_path = gate1_audit_path
        self.gate1_audit = load_json(gate1_audit_path) if gate1_audit_path else None

        self.node_by_id = self._unique_index(self.nodes, "node_id")
        self.edge_by_id = self._unique_index(self.edges, "edge_id")
        self.event_by_id = self._unique_index(self.events, "event_id")
        self.outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in self.edges:
            subject_id = str(edge["subject_id"])
            object_id = str(edge["object_id"])
            if subject_id not in self.node_by_id or object_id not in self.node_by_id:
                raise ValueError(f"Edge endpoint is absent: {edge['edge_id']}")
            self.outgoing[subject_id].append(edge)
            self.incoming[object_id].append(edge)
        for values in self.outgoing.values():
            values.sort(key=self._edge_sort_key)
        for values in self.incoming.values():
            values.sort(key=self._edge_sort_key)
        self._validate_gate1_binding()

    @staticmethod
    def _unique_index(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            value = str(row[key])
            if value in result:
                raise ValueError(f"Duplicate {key}: {value}")
            result[value] = row
        return result

    @staticmethod
    def _edge_sort_key(edge: dict[str, Any]) -> tuple[Any, ...]:
        provenance = edge.get("provenance") or {}
        return (
            RELATION_PRIORITY.get(str(edge.get("predicate")), 999),
            -float(provenance.get("confidence") or 0.0),
            str(provenance.get("source_id") or ""),
            str(edge.get("edge_id") or ""),
        )

    def _validate_gate1_binding(self) -> None:
        if not self.summary.get("accepted"):
            raise ValueError("Formal graph builder did not accept its output.")
        if self.gate1_audit is not None:
            if not self.gate1_audit.get("accepted"):
                raise ValueError("Gate 1 audit did not accept the formal graph.")
            expected = (
                self.gate1_audit.get("inputs", {})
                .get("graph_manifest", {})
                .get("sha256")
            )
            observed = sha256_file(self.paths["manifest"])
            if expected != observed:
                raise ValueError("Gate 1 audit is not bound to this graph manifest.")

    def graph_hashes(self) -> dict[str, str]:
        result = {name: sha256_file(path) for name, path in self.paths.items()}
        if self.gate1_audit_path is not None:
            result["gate1_audit"] = sha256_file(self.gate1_audit_path)
        return result

    def eligible_center(self, node: dict[str, Any]) -> bool:
        node_type = str(node.get("node_type") or "")
        if node_type not in CENTER_TYPES:
            return False
        identifiers = node.get("standard_identifiers") or {}
        if node_type == "Compound":
            stable_namespaces = set(identifiers) - {"PBPKKB", "PBPKKB.CONTEXT"}
            return bool(stable_namespaces)
        if node_type == "PBPKModel":
            properties = node.get("properties") or {}
            return bool(
                properties.get("reusable_candidate")
                and (
                    identifiers.get("MODEL_CHECKSUM")
                    or properties.get("model_checksum")
                    or properties.get("repository_path")
                )
            )
        return bool(identifiers)

    def eligible_centers(self, center_types: frozenset[str] = CENTER_TYPES) -> list[dict[str, Any]]:
        return sorted(
            (
                node
                for node in self.nodes
                if node.get("node_type") in center_types and self.eligible_center(node)
            ),
            key=lambda row: (str(row["node_type"]), str(row["node_id"])),
        )

    def resolve_nodes(self, value: str, allowed_types: set[str] | None = None) -> list[dict[str, Any]]:
        normalized = " ".join(value.strip().lower().replace("_", " ").split())
        matches: list[dict[str, Any]] = []
        for node in self.nodes:
            if allowed_types and node.get("node_type") not in allowed_types:
                continue
            candidates = [str(node.get("preferred_name") or "")]
            candidates.extend(str(alias) for alias in (node.get("properties") or {}).get("aliases", []))
            candidates.extend(str(identifier) for identifier in (node.get("standard_identifiers") or {}).values())
            if normalized in {
                " ".join(candidate.strip().lower().replace("_", " ").split())
                for candidate in candidates
                if candidate
            }:
                matches.append(node)
        return sorted(matches, key=lambda row: (str(row["node_type"]), str(row["node_id"])))

    def traverse(
        self,
        center_id: str,
        *,
        policy: TraversalPolicy,
        objective: str | None = None,
    ) -> dict[str, Any]:
        if center_id not in self.node_by_id:
            raise KeyError(f"Unknown center node: {center_id}")
        center = self.node_by_id[center_id]
        if not self.eligible_center(center):
            raise ValueError(f"Node is not eligible as a recommendation center: {center_id}")

        relation_whitelist = OBJECTIVE_RELATIONS.get(objective or "", DEFAULT_RELATIONS)
        distance = {center_id: 0}
        parent: dict[str, tuple[str, str, str]] = {}
        selected_edges: dict[str, dict[str, Any]] = {}
        edge_depth: dict[str, int] = {}
        frontier = [center_id]
        evidence_source_counts: Counter[str] = Counter()
        truncation = Counter()

        for depth in range(1, policy.max_hops + 1):
            next_frontier: list[str] = []
            for current_id in sorted(frontier):
                grouped: dict[tuple[str, str], list[tuple[dict[str, Any], str, str]]] = defaultdict(list)
                for direction, edges in (
                    ("out", self.outgoing.get(current_id, [])),
                    ("in", self.incoming.get(current_id, [])),
                ):
                    for edge in edges:
                        predicate = str(edge["predicate"])
                        if predicate not in relation_whitelist:
                            continue
                        provenance = edge.get("provenance") or {}
                        if float(provenance.get("confidence") or 0.0) < policy.minimum_confidence:
                            truncation["below_minimum_confidence"] += 1
                            continue
                        other_id = (
                            str(edge["object_id"])
                            if direction == "out"
                            else str(edge["subject_id"])
                        )
                        grouped[(direction, predicate)].append((edge, other_id, direction))

                for group_key in sorted(
                    grouped,
                    key=lambda item: (RELATION_PRIORITY.get(item[1], 999), item[0], item[1]),
                ):
                    candidates = sorted(grouped[group_key], key=lambda item: self._edge_sort_key(item[0]))
                    if len(candidates) > policy.max_neighbors_per_relation:
                        truncation["per_relation_neighbor_limit"] += (
                            len(candidates) - policy.max_neighbors_per_relation
                        )
                    for edge, other_id, direction in candidates[: policy.max_neighbors_per_relation]:
                        edge_id = str(edge["edge_id"])
                        other_node = self.node_by_id[other_id]
                        if other_node["node_type"] == "Evidence" and other_id not in distance:
                            source_id = str((other_node.get("properties") or {}).get("source_id") or "unknown")
                            if evidence_source_counts[source_id] >= policy.max_evidence_per_source:
                                truncation["per_source_evidence_limit"] += 1
                                continue
                        if edge_id not in selected_edges and len(selected_edges) >= policy.max_edges:
                            truncation["edge_budget"] += 1
                            continue
                        if other_id not in distance and len(distance) >= policy.max_nodes:
                            truncation["node_budget"] += 1
                            continue

                        if edge_id not in selected_edges:
                            selected_edges[edge_id] = edge
                            edge_depth[edge_id] = depth
                        else:
                            edge_depth[edge_id] = min(edge_depth[edge_id], depth)
                        if other_id not in distance:
                            distance[other_id] = depth
                            parent[other_id] = (current_id, edge_id, direction)
                            next_frontier.append(other_id)
                            if other_node["node_type"] == "Evidence":
                                source_id = str(
                                    (other_node.get("properties") or {}).get("source_id") or "unknown"
                                )
                                evidence_source_counts[source_id] += 1
            frontier = sorted(set(next_frontier))
            if not frontier:
                break

        paths = [
            self._path_record(center_id, target_id, parent, distance[target_id])
            for target_id in sorted(distance, key=lambda value: (distance[value], value))
            if target_id != center_id
        ]
        return {
            "center": center,
            "distance": distance,
            "edges": selected_edges,
            "edge_depth": edge_depth,
            "paths": paths,
            "truncation": dict(sorted(truncation.items())),
            "relation_whitelist": sorted(relation_whitelist),
            "objective": objective,
            "policy": {
                "max_hops": policy.max_hops,
                "max_neighbors_per_relation": policy.max_neighbors_per_relation,
                "max_evidence_per_source": policy.max_evidence_per_source,
                "max_nodes": policy.max_nodes,
                "max_edges": policy.max_edges,
                "minimum_confidence": policy.minimum_confidence,
            },
        }

    @staticmethod
    def _path_record(
        center_id: str,
        target_id: str,
        parent: dict[str, tuple[str, str, str]],
        hop: int,
    ) -> dict[str, Any]:
        node_ids = [target_id]
        edge_ids: list[str] = []
        directions: list[str] = []
        cursor = target_id
        while cursor != center_id:
            previous_id, edge_id, direction = parent[cursor]
            node_ids.append(previous_id)
            edge_ids.append(edge_id)
            directions.append(direction)
            cursor = previous_id
        node_ids.reverse()
        edge_ids.reverse()
        directions.reverse()
        return {
            "path_id": stable_id("path", center_id, target_id, *edge_ids),
            "center_id": center_id,
            "target_id": target_id,
            "hop": hop,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "directions": directions,
        }

    def serialize_subgraph(self, traversal: dict[str, Any], hop_limit: int) -> dict[str, Any]:
        center = traversal["center"]
        distance = traversal["distance"]
        included_node_ids = {
            node_id for node_id, hop in distance.items() if int(hop) <= hop_limit
        }
        nodes = [
            self.node_by_id[node_id]
            for node_id in sorted(
                included_node_ids,
                key=lambda value: (int(distance[value]), self.node_by_id[value]["node_type"], value),
            )
        ]
        edges = []
        for edge_id, edge in sorted(traversal["edges"].items()):
            if (
                traversal["edge_depth"][edge_id] <= hop_limit
                and edge["subject_id"] in included_node_ids
                and edge["object_id"] in included_node_ids
            ):
                serialized = dict(edge)
                serialized["traversal_depth"] = traversal["edge_depth"][edge_id]
                edges.append(serialized)

        events = [
            self.event_by_id[node_id]
            for node_id in sorted(included_node_ids)
            if node_id in self.event_by_id
        ]
        citations_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for edge in edges:
            provenance = edge.get("provenance") or {}
            evidence_id = str(provenance.get("evidence_id") or "")
            source_hash = str(provenance.get("source_text_hash") or "")
            if not evidence_id or not source_hash:
                continue
            citations_by_key[(evidence_id, source_hash)] = {
                "evidence_id": evidence_id,
                "source_id": provenance.get("source_id"),
                "source_location": provenance.get("source_location"),
                "supporting_text": provenance.get("supporting_text"),
                "source_text_hash": source_hash,
                "confidence": provenance.get("confidence"),
                "quality_status": provenance.get("quality_status"),
            }
        citations = [citations_by_key[key] for key in sorted(citations_by_key)]
        paths = [
            path
            for path in traversal["paths"]
            if path["hop"] <= hop_limit and path["target_id"] in included_node_ids
        ]

        subgraph_id = stable_id(
            "subgraph",
            center["node_id"],
            hop_limit,
            SUBGRAPH_SCHEMA_VERSION,
            sha256_file(self.paths["manifest"]),
        )
        row = {
            "subgraph_id": subgraph_id,
            "schema_version": SUBGRAPH_SCHEMA_VERSION,
            "formal_graph_version": FORMAL_GRAPH_VERSION,
            "center_id": center["node_id"],
            "center_type": center["node_type"],
            "center_name": center["preferred_name"],
            "hop_limit": hop_limit,
            "objective": traversal["objective"],
            "relation_whitelist": traversal["relation_whitelist"],
            "policy": traversal["policy"],
            "nodes": nodes,
            "edges": edges,
            "events": events,
            "citations": citations,
            "path_ids": [path["path_id"] for path in paths],
            "node_count": len(nodes),
            "edge_count": len(edges),
            "event_count": len(events),
            "citation_count": len(citations),
            "truncation": traversal["truncation"] if hop_limit == traversal["policy"]["max_hops"] else {},
        }
        center_semantics = (
            pk_parameter_semantics(str(center["preferred_name"]))
            if center["node_type"] == "PKParameter"
            else None
        )
        row["ontology_semantics"] = {
            "center_aliases": list(center_semantics[0]) if center_semantics else [],
            "center_definition": center_semantics[1] if center_semantics else None,
            "relation_definitions": {
                predicate: RELATION_GLOSSES[predicate]
                for predicate in sorted({str(edge["predicate"]) for edge in edges})
                if predicate in RELATION_GLOSSES
            },
        }
        row["embedding_text"] = self._embedding_text(row)
        row["content_sha256"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in row.items() if key != "content_sha256"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return row

    @staticmethod
    def _embedding_text(row: dict[str, Any]) -> str:
        node_names = {node["node_id"]: node["preferred_name"] for node in row["nodes"]}
        node_types = Counter(str(node["node_type"]) for node in row["nodes"])
        relation_types = Counter(str(edge["predicate"]) for edge in row["edges"])
        center_node = next(
            node for node in row["nodes"] if node["node_id"] == row["center_id"]
        )
        lines = [
            f"CENTER [{row['center_type']}] {str(row['center_name'])[:180]}",
            "CENTER IDENTIFIERS "
            + json.dumps(
                center_node.get("standard_identifiers") or {},
                ensure_ascii=False,
                sort_keys=True,
            ),
            f"HOP LIMIT {row['hop_limit']}",
            f"ONTOLOGY VERSION {ONTOLOGY_VERSION}",
            "SOURCE FAMILIES "
            + "; ".join(
                sorted(
                    {
                        str(citation.get("source_id") or "unknown")
                        for citation in row["citations"]
                    }
                )
            ),
        ]
        if row["center_type"] == "PKParameter":
            semantics = pk_parameter_semantics(str(row["center_name"]))
            if semantics is not None:
                aliases, definition = semantics
                lines.append(
                    "CENTER SEMANTICS aliases="
                    + ", ".join(aliases)
                    + f"; definition={definition}"
                )
        semantic_relations = sorted(
            relation_types,
            key=lambda name: (RELATION_PRIORITY.get(name, 999), name),
        )
        lines.extend(
            [
                "RELATION SEARCH TERMS "
                + "; ".join(
                    f"{name}={RELATION_SEARCH_TERMS[name]}"
                    for name in semantic_relations
                    if name in RELATION_SEARCH_TERMS
                ),
                "RELATION TYPES "
                + "; ".join(
                    f"{name}={count}" for name, count in sorted(relation_types.items())
                ),
                "NODE TYPES "
                + "; ".join(
                    f"{name}={count}" for name, count in sorted(node_types.items())
                ),
            ]
        )
        lines.extend(
            f"RELATION GLOSS {name}={RELATION_GLOSSES[name]}"
            for name in semantic_relations
            if name in RELATION_GLOSSES
        )

        # Preserve center-connected and event-role facts before high-degree node
        # descriptions so a BERT-size tokenizer sees graph semantics.
        prioritized_edges = sorted(
            row["edges"],
            key=lambda edge: (
                0
                if row["center_id"] in {edge["subject_id"], edge["object_id"]}
                else 1,
                RELATION_PRIORITY.get(str(edge["predicate"]), 999),
                int(edge.get("traversal_depth") or 0),
                str(edge["edge_id"]),
            ),
        )
        for edge in prioritized_edges[:8]:
            properties = edge.get("properties") or {}
            selected_properties = {
                key: properties.get(key)
                for key in (
                    "value",
                    "canonical_value",
                    "unit",
                    "canonical_unit",
                    "role",
                    "direction",
                    "effect",
                )
                if properties.get(key) is not None
            }
            lines.append(
                "EDGE "
                f"{str(node_names.get(edge['subject_id'], edge['subject_id']))[:96]} "
                f"-[{edge['predicate']}]-> "
                f"{str(node_names.get(edge['object_id'], edge['object_id']))[:96]} "
                f"{json.dumps(selected_properties, ensure_ascii=False, sort_keys=True)}"
            )

        for event in row["events"][:2]:
            context = json.dumps(
                event.get("context") or {},
                ensure_ascii=False,
                sort_keys=True,
            )[:300]
            lines.append(
                f"EVENT [{event['event_type']}] trigger={event['trigger_text']} "
                f"context={context}"
            )

        prioritized_nodes = sorted(
            (
                node
                for node in row["nodes"]
                if node["node_id"] != row["center_id"]
                and node["node_type"]
                in {
                    "PKParameter",
                    "ParameterValue",
                    "Unit",
                    "DoseRegimen",
                    "AdministrationRoute",
                    "Population",
                    "Enzyme",
                    "Transporter",
                    "PBPKModel",
                }
            ),
            key=lambda node: (str(node["node_type"]), str(node["node_id"])),
        )
        for node in prioritized_nodes[:4]:
            properties = node.get("properties") or {}
            selected_properties = {
                key: properties.get(key)
                for key in (
                    "route",
                    "dose",
                    "frequency",
                    "formulation",
                    "value",
                    "canonical_value",
                    "canonical_unit",
                    "availability_status",
                    "reusable_candidate",
                )
                if properties.get(key) is not None
            }
            lines.append(
                f"NODE [{node['node_type']}] {node['preferred_name']} "
                f"{json.dumps(selected_properties, ensure_ascii=False, sort_keys=True)}"
            )

        source_seen: set[str] = set()
        selected_citations: list[dict[str, Any]] = []
        for citation in sorted(
            row["citations"],
            key=lambda value: (
                -float(value.get("confidence") or 0.0),
                str(value.get("source_id") or ""),
                str(value.get("evidence_id") or ""),
            ),
        ):
            source_id = str(citation.get("source_id") or "")
            if source_id in source_seen and len(selected_citations) >= 2:
                continue
            source_seen.add(source_id)
            selected_citations.append(citation)
            if len(selected_citations) >= 2:
                break
        for citation in selected_citations:
            quote = " ".join(str(citation.get("supporting_text") or "").split())[:240]
            lines.append(
                f"CITATION source={citation.get('source_id')} "
                f"location={citation.get('source_location')} quote={quote}"
            )
        return "\n".join(lines)


def build_subgraph_index(
    *,
    graph_dir: Path,
    gate1_audit_path: Path,
    output_dir: Path,
    center_types: frozenset[str] = CENTER_TYPES,
    policy: TraversalPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or TraversalPolicy()
    store = FormalGraphStore(graph_dir, gate1_audit_path)
    centers = store.eligible_centers(center_types)
    subgraphs: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    paths_by_id: dict[str, dict[str, Any]] = {}
    truncation_counts: Counter[str] = Counter()

    for index, center in enumerate(centers, 1):
        traversal = store.traverse(str(center["node_id"]), policy=policy)
        for hop in (0, 1, 2):
            subgraphs[hop].append(store.serialize_subgraph(traversal, hop))
        for path in traversal["paths"]:
            paths_by_id[path["path_id"]] = path
        truncation_counts.update(traversal["truncation"])
        if index % 25 == 0 or index == len(centers):
            print(json.dumps({"processed_centers": index, "total_centers": len(centers)}), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, Path] = {}
    for hop, rows in subgraphs.items():
        path = output_dir / f"subgraphs_{hop}hop.jsonl"
        _atomic_jsonl(path, rows)
        output_files[f"subgraphs_{hop}hop"] = path
    paths_path = output_dir / "traversal_paths.jsonl"
    _atomic_jsonl(paths_path, (paths_by_id[key] for key in sorted(paths_by_id)))
    output_files["traversal_paths"] = paths_path

    manifest = {
        "accepted": True,
        "formal_graph_version": FORMAL_GRAPH_VERSION,
        "subgraph_schema_version": SUBGRAPH_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "graph_dir": str(graph_dir),
        "gate1_audit_path": str(gate1_audit_path),
        "graph_hashes": store.graph_hashes(),
        "center_types": sorted(center_types),
        "eligible_center_count": len(centers),
        "eligible_center_type_counts": dict(
            sorted(Counter(str(center["node_type"]) for center in centers).items())
        ),
        "policy": {
            "max_hops": policy.max_hops,
            "max_neighbors_per_relation": policy.max_neighbors_per_relation,
            "max_evidence_per_source": policy.max_evidence_per_source,
            "max_nodes": policy.max_nodes,
            "max_edges": policy.max_edges,
            "minimum_confidence": policy.minimum_confidence,
        },
        "output_hashes": {
            name: sha256_file(path)
            for name, path in sorted(output_files.items())
        },
        "truncation_counts": dict(sorted(truncation_counts.items())),
    }
    _atomic_json(output_dir / "gate2_manifest.json", manifest)

    summary = {
        "accepted": True,
        "eligible_center_count": len(centers),
        "subgraph_counts": {f"{hop}hop": len(rows) for hop, rows in subgraphs.items()},
        "path_count": len(paths_by_id),
        "node_count_distributions": {
            f"{hop}hop": {
                "min": min((row["node_count"] for row in rows), default=0),
                "max": max((row["node_count"] for row in rows), default=0),
                "mean": (
                    sum(row["node_count"] for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
            }
            for hop, rows in subgraphs.items()
        },
        "edge_count_distributions": {
            f"{hop}hop": {
                "min": min((row["edge_count"] for row in rows), default=0),
                "max": max((row["edge_count"] for row in rows), default=0),
                "mean": (
                    sum(row["edge_count"] for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
            }
            for hop, rows in subgraphs.items()
        },
        "truncation_counts": dict(sorted(truncation_counts.items())),
    }
    _atomic_json(output_dir / "gate2_build_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build formal 0/1/2-hop PBPK KG subgraphs.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--gate1-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--center-types",
        default=",".join(sorted(CENTER_TYPES)),
        help="Comma-separated center node types.",
    )
    parser.add_argument("--max-neighbors-per-relation", type=int, default=24)
    parser.add_argument("--max-evidence-per-source", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--max-edges", type=int, default=512)
    parser.add_argument("--minimum-confidence", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_types = frozenset(
        value.strip() for value in args.center_types.split(",") if value.strip()
    )
    unknown = requested_types - CENTER_TYPES
    if unknown:
        raise ValueError(f"Unsupported center types: {sorted(unknown)}")
    summary = build_subgraph_index(
        graph_dir=args.graph_dir,
        gate1_audit_path=args.gate1_audit,
        output_dir=args.output_dir,
        center_types=requested_types,
        policy=TraversalPolicy(
            max_hops=2,
            max_neighbors_per_relation=args.max_neighbors_per_relation,
            max_evidence_per_source=args.max_evidence_per_source,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            minimum_confidence=args.minimum_confidence,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
