from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pbpk_kb.recommendation.model_matching.graphs import (
    GraphEdge,
    GraphNode,
    TypedGraph,
    node_hierarchy,
    semantic_key,
)


FULL_GRAPH_ADAPTER_VERSION = "pbpk-full-model-kg-adapter-1.0.0"
SEMANTIC_CONDENSATION_VERSION = "pbpk-provenance-preserving-condensation-1.0.0"
EXPAND_OUTGOING_TYPES = {
    "ModelStructure",
    "ModelComponent",
    "SimulationScenario",
    "AdministrationEvent",
    "ValidationRun",
    "ValidationMetric",
    "Evidence",
}
NODE_TYPE_ALIASES = {
    "AdministrationRoute": "Route",
    "CalibrationDataset": "Dataset",
    "QualificationDataset": "Dataset",
    "ModelDataset": "Dataset",
}


def _chunks(values: Iterable[str], size: int = 500) -> Iterable[list[str]]:
    chunk: list[str] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _rows_for_ids(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    values: Iterable[str],
) -> list[sqlite3.Row]:
    output: list[sqlite3.Row] = []
    for chunk in _chunks(sorted(set(values))):
        placeholders = ",".join("?" for _ in chunk)
        output.extend(
            connection.execute(
                f"SELECT * FROM {table} WHERE {column} IN ({placeholders})",
                chunk,
            ).fetchall()
        )
    return output


def load_model_version_graph(
    *,
    connection: sqlite3.Connection,
    model_version_id: str,
) -> TypedGraph:
    root = connection.execute(
        "SELECT * FROM nodes WHERE node_id = ? AND node_type = 'ModelVersion'",
        (model_version_id,),
    ).fetchone()
    if root is None:
        raise KeyError(f"ModelVersion is absent from Model KG: {model_version_id}")
    direct_edges = list(
        connection.execute(
            "SELECT * FROM edges WHERE subject_id = ? OR object_id = ?",
            (model_version_id, model_version_id),
        )
    )
    direct_ids = {
        value
        for edge in direct_edges
        for value in (edge["subject_id"], edge["object_id"])
    }
    direct_nodes = _rows_for_ids(connection, "nodes", "node_id", direct_ids)
    expandable = {
        str(row["node_id"])
        for row in direct_nodes
        if row["node_type"] in EXPAND_OUTGOING_TYPES
    }
    expanded_edges = _rows_for_ids(connection, "edges", "subject_id", expandable)
    edges_by_id = {
        str(row["edge_id"]): row
        for row in [*direct_edges, *expanded_edges]
    }
    node_ids = {
        value
        for edge in edges_by_id.values()
        for value in (edge["subject_id"], edge["object_id"])
    }
    node_rows = _rows_for_ids(connection, "nodes", "node_id", node_ids)
    nodes = tuple(
        GraphNode(
            node_id=str(row["node_id"]),
            node_type=NODE_TYPE_ALIASES.get(
                str(row["node_type"]), str(row["node_type"])
            ),
            semantic_key=semantic_key(
                NODE_TYPE_ALIASES.get(
                    str(row["node_type"]), str(row["node_type"])
                ),
                str(row["name"]),
            ),
            label=str(row["name"]),
            hierarchy=(
                "root"
                if row["node_id"] == model_version_id
                else node_hierarchy(
                    NODE_TYPE_ALIASES.get(
                        str(row["node_type"]), str(row["node_type"])
                    )
                )
            ),
            attributes={
                "source_node_id": str(row["node_id"]),
                "properties": json.loads(row["properties_json"]),
                "provenance": json.loads(row["provenance_json"]),
                "occurrence_count": int(row["occurrence_count"]),
            },
        )
        for row in sorted(node_rows, key=lambda value: str(value["node_id"]))
    )
    edges = tuple(
        GraphEdge(
            source_id=str(row["subject_id"]),
            predicate=str(row["predicate"]),
            target_id=str(row["object_id"]),
            attributes={
                "source_edge_id": str(row["edge_id"]),
                "properties": json.loads(row["properties_json"]),
                "provenance": json.loads(row["provenance_json"]),
                "occurrence_count": int(row["occurrence_count"]),
            },
        )
        for row in sorted(edges_by_id.values(), key=lambda value: str(value["edge_id"]))
    )
    root_properties = json.loads(root["properties_json"])
    return TypedGraph(
        graph_id=model_version_id,
        graph_kind="executable_model_full_kg",
        nodes=nodes,
        edges=edges,
        metadata={
            "adapter_version": FULL_GRAPH_ADAPTER_VERSION,
            "model_family": root_properties.get("model_family"),
            "project_relative_path": root_properties.get("project_relative_path"),
            "repository_commit": root_properties.get("repository_commit"),
            "reusable_candidate": bool(root_properties.get("reusable_candidate")),
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    )


def _id_checksum(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def condense_model_graph(graph: TypedGraph) -> TypedGraph:
    """Merge repeated semantic concepts while retaining source-ID traceability."""

    grouped_nodes: dict[str, list[GraphNode]] = defaultdict(list)
    for node in graph.nodes:
        grouped_nodes[node.semantic_key].append(node)

    node_id_map: dict[str, str] = {}
    condensed_nodes: list[GraphNode] = []
    for semantic_key_value, values in sorted(grouped_nodes.items()):
        ordered = sorted(values, key=lambda value: value.node_id)
        source_ids = [
            str(node.attributes.get("source_node_id") or node.node_id)
            for node in ordered
        ]
        if any(node.node_id == graph.graph_id for node in ordered):
            condensed_id = graph.graph_id
        else:
            digest = hashlib.sha256(semantic_key_value.encode("utf-8")).hexdigest()[:24]
            condensed_id = f"{graph.graph_id}:concept:{digest}"
        for node in ordered:
            node_id_map[node.node_id] = condensed_id
        occurrence_count = sum(
            int(node.attributes.get("occurrence_count") or 1)
            for node in ordered
        )
        provenance_count = sum(
            bool(node.attributes.get("provenance"))
            for node in ordered
        )
        condensed_nodes.append(
            GraphNode(
                node_id=condensed_id,
                node_type=ordered[0].node_type,
                semantic_key=semantic_key_value,
                label=ordered[0].label,
                hierarchy=(
                    "root"
                    if condensed_id == graph.graph_id
                    else ordered[0].hierarchy
                ),
                required=any(node.required for node in ordered),
                attributes={
                    "source_node_ids": source_ids,
                    "source_node_count": len(source_ids),
                    "source_node_ids_sha256": _id_checksum(source_ids),
                    "occurrence_count": occurrence_count,
                    "provenance_record_count": provenance_count,
                    "provenance": bool(provenance_count),
                },
            )
        )

    grouped_edges: dict[tuple[str, str, str], list[GraphEdge]] = defaultdict(list)
    for edge in graph.edges:
        source_id = node_id_map.get(edge.source_id)
        target_id = node_id_map.get(edge.target_id)
        if source_id is None or target_id is None:
            continue
        grouped_edges[(source_id, edge.predicate, target_id)].append(edge)

    condensed_edges: list[GraphEdge] = []
    for (source_id, predicate, target_id), values in sorted(grouped_edges.items()):
        source_edge_ids = [
            str(edge.attributes.get("source_edge_id") or "")
            for edge in values
            if edge.attributes.get("source_edge_id")
        ]
        condensed_edges.append(
            GraphEdge(
                source_id=source_id,
                predicate=predicate,
                target_id=target_id,
                required=any(edge.required for edge in values),
                attributes={
                    "source_edge_ids": sorted(source_edge_ids),
                    "source_edge_count": len(values),
                    "source_edge_ids_sha256": _id_checksum(source_edge_ids),
                    "occurrence_count": sum(
                        int(edge.attributes.get("occurrence_count") or 1)
                        for edge in values
                    ),
                },
            )
        )

    full_node_count = len(graph.nodes)
    full_edge_count = len(graph.edges)
    metadata = dict(graph.metadata)
    metadata.update(
        {
            "condensation_version": SEMANTIC_CONDENSATION_VERSION,
            "full_node_count": full_node_count,
            "full_edge_count": full_edge_count,
            "node_count": len(condensed_nodes),
            "edge_count": len(condensed_edges),
            "node_compression_ratio": (
                len(condensed_nodes) / full_node_count if full_node_count else 1.0
            ),
            "edge_compression_ratio": (
                len(condensed_edges) / full_edge_count if full_edge_count else 1.0
            ),
            "traceability": "source SQLite node/edge IDs retained on every aggregate",
        }
    )
    return TypedGraph(
        graph_id=graph.graph_id,
        graph_kind="executable_model_semantic_condensation",
        nodes=tuple(condensed_nodes),
        edges=tuple(condensed_edges),
        metadata=metadata,
    )
def load_model_graphs(
    *,
    sqlite_path: Path,
    model_version_ids: Iterable[str],
) -> dict[str, TypedGraph]:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return {
            model_version_id: load_model_version_graph(
                connection=connection,
                model_version_id=model_version_id,
            )
            for model_version_id in model_version_ids
        }
    finally:
        connection.close()
