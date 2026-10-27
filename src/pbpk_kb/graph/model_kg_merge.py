from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.graph.model_kg_builder import (
    ModelKGBuildError,
    _GraphStaging,
    _counts,
    _edge_rows,
    _iter_jsonl,
    _node_rows,
    _sha256_file,
    _unresolved_partner_rows,
    _write_gzip_jsonl_atomic,
    _write_json_atomic,
    _write_jsonl_atomic,
)
from pbpk_kb.graph.model_kg_schema import (
    MODEL_KG_SCHEMA_VERSION,
    ModelKGEdge,
    ModelKGNode,
    ModelKGProvenance,
    canonical_json,
    schema_payload,
)


MERGER_VERSION = "pbpk-model-kg-merger-1.0.0"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ModelKGBuildError(f"Expected a JSON object: {path}")
    return value


def _identity_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sum_input_record_counts(
    summaries: Iterable[dict[str, Any]],
) -> dict[str, int]:
    keys = {
        str(key)
        for summary in summaries
        for key in (summary.get("input_record_counts") or {})
    }
    return {
        key: sum(
            int((summary.get("input_record_counts") or {}).get(key, 0))
            for summary in summaries
        )
        for key in sorted(keys)
    }


def merge_model_kgs(
    *,
    graph_directories: list[Path],
    audit_paths: list[Path],
    output_directory: Path,
) -> dict[str, Any]:
    if len(graph_directories) < 2:
        raise ModelKGBuildError("At least two Model KG inputs are required.")
    if len(graph_directories) != len(audit_paths):
        raise ModelKGBuildError(
            "Each Model KG input requires its independent audit report."
        )
    graph_directories = [path.resolve() for path in graph_directories]
    audit_paths = [path.resolve() for path in audit_paths]
    output_directory = output_directory.resolve()
    working = output_directory.with_name(output_directory.name + ".tmp")
    if output_directory.exists() or working.exists():
        raise ModelKGBuildError(
            f"Output or temporary directory already exists: {output_directory}"
        )

    summaries: list[dict[str, Any]] = []
    input_manifests: list[dict[str, Any]] = []
    for graph_directory, audit_path in zip(
        graph_directories,
        audit_paths,
        strict=True,
    ):
        summary_path = graph_directory / "summary.json"
        schema_path = graph_directory / "model_kg_schema.json"
        summary = _read_json(summary_path)
        schema = _read_json(schema_path)
        audit = _read_json(audit_path)
        if audit.get("accepted") is not True:
            raise ModelKGBuildError(
                f"Input Model KG audit is not accepted: {audit_path}"
            )
        if audit.get("graph_summary_sha256") != _sha256_file(summary_path):
            raise ModelKGBuildError(
                f"Audit is not bound to graph summary: {audit_path}"
            )
        checksum_errors = []
        for filename, expected in sorted(
            (summary.get("output_checksums") or {}).items()
        ):
            path = graph_directory / filename
            if not path.is_file():
                checksum_errors.append(f"{filename}:missing")
            elif _sha256_file(path) != expected:
                checksum_errors.append(f"{filename}:checksum_mismatch")
        if checksum_errors:
            raise ModelKGBuildError(
                f"Input Model KG checksum failure in {graph_directory}: "
                + ", ".join(checksum_errors)
            )
        summaries.append(summary)
        input_manifests.append(
            {
                "graph_directory": str(graph_directory),
                "graph_summary_sha256": _sha256_file(summary_path),
                "graph_schema_sha256": _sha256_file(schema_path),
                "graph_schema_version": schema.get("schema_version"),
                "audit_path": str(audit_path),
                "audit_sha256": _sha256_file(audit_path),
                "node_count": summary.get("node_count"),
                "edge_count": summary.get("edge_count"),
            }
        )

    working.mkdir(parents=True)
    staging = _GraphStaging(working / "model_kg.sqlite")
    node_identity: dict[str, str] = {}
    edge_identity: dict[str, str] = {}
    duplicate_node_count = 0
    duplicate_edge_count = 0
    try:
        for graph_directory in graph_directories:
            for raw in _iter_jsonl(
                graph_directory / "model_kg_nodes.jsonl.gz"
            ):
                node = ModelKGNode.model_validate(raw)
                identity = _identity_hash(
                    {
                        "node_type": node.node_type,
                        "name": node.name,
                        "properties": node.properties,
                    }
                )
                previous = node_identity.get(node.node_id)
                if previous is not None:
                    duplicate_node_count += 1
                    if previous != identity:
                        raise ModelKGBuildError(
                            "Node ID collision with non-identical scientific "
                            f"content: {node.node_id}"
                        )
                else:
                    node_identity[node.node_id] = identity
                staging.add_node(node)
            staging.flush()

            for raw in _iter_jsonl(
                graph_directory / "model_kg_edges.jsonl.gz"
            ):
                edge = ModelKGEdge.model_validate(raw)
                identity = _identity_hash(
                    {
                        "subject_id": edge.subject_id,
                        "subject_type": edge.subject_type,
                        "predicate": edge.predicate,
                        "object_id": edge.object_id,
                        "object_type": edge.object_type,
                        "properties": edge.properties,
                    }
                )
                previous = edge_identity.get(edge.edge_id)
                if previous is not None:
                    duplicate_edge_count += 1
                    if previous != identity:
                        raise ModelKGBuildError(
                            "Edge ID collision with non-identical scientific "
                            f"content: {edge.edge_id}"
                        )
                else:
                    edge_identity[edge.edge_id] = identity
                staging.add_edge(edge)
            staging.flush()

            for raw in _iter_jsonl(
                graph_directory / "unresolved_process_partners.jsonl.gz"
            ):
                provenance = ModelKGProvenance.model_validate(
                    raw.get("provenance")
                )
                occurrence_count = int(raw.get("occurrence_count") or 0)
                if occurrence_count < 1:
                    raise ModelKGBuildError(
                        "Unresolved process partner occurrence_count must be "
                        "positive."
                    )
                staging.add_unresolved_partner(
                    str(raw["identifier"]),
                    provenance,
                    occurrence_count=occurrence_count,
                )
            staging.flush()

        staging.finish()
        node_type_counts = _counts(
            staging.connection,
            "nodes",
            "node_type",
        )
        predicate_counts = _counts(
            staging.connection,
            "edges",
            "predicate",
        )
        node_count = sum(node_type_counts.values())
        edge_count = sum(predicate_counts.values())
        unresolved_unique_count = int(
            staging.connection.execute(
                "SELECT COUNT(*) FROM unresolved_process_partners"
            ).fetchone()[0]
        )
        output_checksums = {
            "model_kg_nodes.jsonl.gz": _write_gzip_jsonl_atomic(
                working / "model_kg_nodes.jsonl.gz",
                _node_rows(staging.connection),
            ),
            "model_kg_edges.jsonl.gz": _write_gzip_jsonl_atomic(
                working / "model_kg_edges.jsonl.gz",
                _edge_rows(staging.connection),
            ),
            "model_kg_schema.json": _write_json_atomic(
                working / "model_kg_schema.json",
                schema_payload(),
            ),
            "unresolved_process_partners.jsonl.gz": (
                _write_gzip_jsonl_atomic(
                    working / "unresolved_process_partners.jsonl.gz",
                    _unresolved_partner_rows(staging.connection),
                )
            ),
            "failures.jsonl": _write_jsonl_atomic(
                working / "failures.jsonl",
                [],
            ),
        }
        staging.close()
        output_checksums["model_kg.sqlite"] = _sha256_file(
            working / "model_kg.sqlite"
        )
        input_record_counts = _sum_input_record_counts(summaries)
        summary = {
            "build_schema_version": "1.0.0",
            "builder_version": MERGER_VERSION,
            "model_kg_schema_version": MODEL_KG_SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "accepted_input_gate1_audit": True,
            "source_model_version_directory": "federated_model_kg_union",
            "source_model_version_summary_sha256": _identity_hash(
                [item["graph_summary_sha256"] for item in input_manifests]
            ),
            "gate1_audit_sha256": _identity_hash(
                [item["audit_sha256"] for item in input_manifests]
            ),
            "execution_plan_sha256": _identity_hash(input_manifests),
            "model_count": node_type_counts.get("PBPKModel", 0),
            "model_version_count": node_type_counts.get("ModelVersion", 0),
            "node_count": node_count,
            "edge_count": edge_count,
            "node_type_counts": node_type_counts,
            "predicate_counts": predicate_counts,
            "input_record_counts": input_record_counts,
            "unresolved_process_partner_unique_count": (
                unresolved_unique_count
            ),
            "output_checksums": output_checksums,
            "policy_blockers": sorted(
                {
                    str(reason)
                    for input_summary in summaries
                    for reason in input_summary.get("policy_blockers") or []
                }
            ),
            "input_graphs": input_manifests,
            "merge_counts": {
                "input_graphs": len(graph_directories),
                "duplicate_nodes_merged": duplicate_node_count,
                "duplicate_edges_merged": duplicate_edge_count,
            },
        }
        _write_json_atomic(working / "summary.json", summary)
        working.replace(output_directory)
        return summary
    except Exception as exc:
        try:
            staging.close()
        except Exception:
            pass
        _write_json_atomic(
            working / "failure.json",
            {
                "failed_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "input_graphs": input_manifests,
                "duplicate_nodes_seen": duplicate_node_count,
                "duplicate_edges_seen": duplicate_edge_count,
            },
        )
        raise
