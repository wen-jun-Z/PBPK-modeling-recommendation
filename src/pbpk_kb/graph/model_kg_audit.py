from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from pbpk_kb.graph.model_kg_schema import (
    MODEL_KG_SCHEMA_VERSION,
    NODE_TYPES,
    ModelKGEdge,
    ModelKGNode,
    validate_relation,
)


AUDITOR_VERSION = "pbpk-model-kg-auditor-1.3.0"


class ModelKGAuditError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ModelKGAuditError(f"Expected an object: {path}")
    return value


def _iter_gzip_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ModelKGAuditError(
                    f"Expected object at {path}:{line_number}"
                )
            yield value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def audit_model_kg(graph_directory: Path) -> dict[str, Any]:
    graph_directory = graph_directory.resolve()
    summary = _read_json(graph_directory / "summary.json")
    schema = _read_json(graph_directory / "model_kg_schema.json")
    if schema.get("schema_version") != MODEL_KG_SCHEMA_VERSION:
        raise ModelKGAuditError("Model KG schema version mismatch.")

    checksum_mismatches: list[str] = []
    for filename, expected in sorted(
        (summary.get("output_checksums") or {}).items()
    ):
        path = graph_directory / filename
        if not path.is_file():
            checksum_mismatches.append(f"{filename}:missing")
            continue
        actual = _sha256_file(path)
        if actual != expected:
            checksum_mismatches.append(
                f"{filename}:{expected}:{actual}"
            )

    node_type_counts: Counter[str] = Counter()
    predicate_counts: Counter[str] = Counter()
    duplicate_node_ids: list[str] = []
    duplicate_edge_ids: list[str] = []
    invalid_node_types: list[str] = []
    relation_violations: list[str] = []
    provenance_errors: list[str] = []
    model_version_errors: list[str] = []
    validation_run_errors: list[str] = []
    grade_3_validation_run_ids: set[str] = set()
    grade_3_expected_datasets: dict[str, int] = {}
    concept_projection_errors: list[str] = []
    unresolved_quality_errors: list[str] = []
    parameter_occurrence_count = 0
    component_occurrence_count = 0
    process_occurrence_count = 0

    with tempfile.TemporaryDirectory(
        prefix="pbpk-model-kg-audit-"
    ) as temporary_directory:
        audit_db = Path(temporary_directory) / "audit.sqlite"
        connection = sqlite3.connect(audit_db)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript(
            """
            CREATE TABLE nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                normalized_name TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE edges (
                edge_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                object_type TEXT NOT NULL,
                predicate TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE unresolved_process_partners (
                identifier TEXT PRIMARY KEY,
                normalized_identifier TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        pending = 0
        for row in _iter_gzip_jsonl(
            graph_directory / "model_kg_nodes.jsonl.gz"
        ):
            node = ModelKGNode.model_validate(row)
            node_type_counts[node.node_type] += 1
            if node.node_type not in NODE_TYPES:
                invalid_node_types.append(node.node_id)
            if (
                not node.provenance.source_id
                or not node.provenance.source_location
                or len(node.provenance.asset_sha256) != 64
                or not node.provenance.extraction_method
            ):
                provenance_errors.append(f"node:{node.node_id}")
            if node.node_type == "ModelVersion":
                properties = node.properties
                if (
                    properties.get("execution_probe_status") != "passed"
                    or properties.get("pkml_parse_status") != "parsed"
                    or not properties.get("software")
                    or not properties.get("software_version")
                    or not isinstance(properties.get("license"), dict)
                ):
                    model_version_errors.append(node.node_id)
                license_allowed = (
                    properties.get("license", {}).get("status")
                    == "allowed"
                )
                expected_reusable = (
                    license_allowed
                    and properties.get("execution_probe_status")
                    == "passed"
                    and properties.get("pkml_parse_status") == "parsed"
                )
                if properties.get("reusable_candidate") != expected_reusable:
                    model_version_errors.append(
                        f"{node.node_id}:reuse_policy"
                    )
                if properties.get("reusable_candidate") is True and (
                    properties.get("run_label_grade") != 3
                    or properties.get("validation_grade")
                    != "external_study_validated"
                    or properties.get("ranking_label") != "reuse_positive"
                    or not properties.get("validation_run_id")
                ):
                    model_version_errors.append(
                        f"{node.node_id}:missing_grade_3_validation"
                    )
            if node.node_type == "ValidationRun":
                properties = node.properties
                if properties.get("run_label_grade") == 3:
                    grade_3_validation_run_ids.add(node.node_id)
                    grade_3_expected_datasets[node.node_id] = int(
                        properties.get("verification_dataset_count") or 0
                    )
                    if (
                        properties.get("status")
                        != "external_study_validated"
                        or properties.get("ranking_label")
                        != "reuse_positive"
                        or properties.get("engineering_accepted") is not True
                        or properties.get("scientific_accepted") is not True
                        or properties.get("reusable_candidate") is not True
                        or int(
                            properties.get("verification_dataset_count") or 0
                        )
                        < 1
                        or int(
                            properties.get("pooled_observation_count") or 0
                        )
                        < 1
                    ):
                        validation_run_errors.append(
                            f"{node.node_id}:inconsistent_grade_3_label"
                        )
            if node.node_type == "ModelParameter":
                parameter_occurrence_count += node.occurrence_count
                forbidden = {
                    "parameter_id",
                    "value",
                    "formula_id",
                    "is_default",
                    "value_origin_source",
                    "value_origin_description",
                }
                if (
                    forbidden.intersection(node.properties)
                    or node.properties.get(
                        "value_variants_in_fact_layer"
                    )
                    is not True
                    or node.properties.get("fact_layer")
                    != "parameters.jsonl.gz"
                ):
                    concept_projection_errors.append(node.node_id)
            if node.node_type == "ModelComponent":
                component_kind = node.properties.get("component_kind")
                forbidden = {
                    "component_id",
                    "process_id",
                    "formula_id",
                    "description",
                }
                if forbidden.intersection(node.properties):
                    concept_projection_errors.append(node.node_id)
                if component_kind == "process":
                    process_occurrence_count += node.occurrence_count
                    if (
                        node.properties.get(
                            "process_variants_in_fact_layer"
                        )
                        is not True
                        or node.properties.get("fact_layer")
                        != "processes.jsonl.gz"
                    ):
                        concept_projection_errors.append(node.node_id)
                else:
                    component_occurrence_count += node.occurrence_count
                    if (
                        node.properties.get(
                            "component_variants_in_fact_layer"
                        )
                        is not True
                        or node.properties.get("fact_layer")
                        != "components.jsonl.gz"
                    ):
                        concept_projection_errors.append(node.node_id)
            try:
                connection.execute(
                    """
                    INSERT INTO nodes(
                        node_id, node_type, normalized_name
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        node.node_id,
                        node.node_type,
                        node.name.casefold(),
                    ),
                )
            except sqlite3.IntegrityError:
                duplicate_node_ids.append(node.node_id)
            pending += 1
            if pending >= 50_000:
                connection.commit()
                pending = 0
        connection.commit()

        for row in _iter_gzip_jsonl(
            graph_directory / "model_kg_edges.jsonl.gz"
        ):
            edge = ModelKGEdge.model_validate(row)
            predicate_counts[edge.predicate] += 1
            relation_error = validate_relation(
                edge.predicate,
                edge.subject_type,
                edge.object_type,
            )
            if relation_error:
                relation_violations.append(
                    f"{edge.edge_id}:{relation_error}"
                )
            if (
                not edge.provenance.source_id
                or not edge.provenance.source_location
                or len(edge.provenance.asset_sha256) != 64
                or not edge.provenance.extraction_method
            ):
                provenance_errors.append(f"edge:{edge.edge_id}")
            try:
                connection.execute(
                    """
                    INSERT INTO edges(
                        edge_id, subject_id, subject_type,
                        object_id, object_type, predicate
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge.edge_id,
                        edge.subject_id,
                        edge.subject_type,
                        edge.object_id,
                        edge.object_type,
                        edge.predicate,
                    ),
                )
            except sqlite3.IntegrityError:
                duplicate_edge_ids.append(edge.edge_id)
            pending += 1
            if pending >= 50_000:
                connection.commit()
                pending = 0
        connection.commit()

        unresolved_quality_count = 0
        unresolved_quality_occurrence_count = 0
        for row in _iter_gzip_jsonl(
            graph_directory
            / "unresolved_process_partners.jsonl.gz"
        ):
            identifier = row.get("identifier")
            normalized_identifier = row.get("normalized_identifier")
            provenance = row.get("provenance")
            occurrence_count = row.get("occurrence_count")
            if (
                not isinstance(identifier, str)
                or not identifier
                or normalized_identifier != identifier.casefold()
                or not isinstance(provenance, dict)
                or not provenance.get("source_id")
                or not provenance.get("source_location")
                or len(str(provenance.get("asset_sha256") or "")) != 64
                or not provenance.get("extraction_method")
                or not isinstance(occurrence_count, int)
                or occurrence_count < 1
            ):
                unresolved_quality_errors.append(
                    str(identifier or "<missing>")
                )
                continue
            try:
                connection.execute(
                    """
                    INSERT INTO unresolved_process_partners(
                        identifier, normalized_identifier
                    ) VALUES (?, ?)
                    """,
                    (identifier, normalized_identifier),
                )
            except sqlite3.IntegrityError:
                unresolved_quality_errors.append(
                    f"{identifier}:duplicate"
                )
                continue
            unresolved_quality_count += 1
            unresolved_quality_occurrence_count += occurrence_count
        connection.commit()
        connection.executescript(
            """
            CREATE INDEX edges_subject_audit_idx ON edges(subject_id);
            CREATE INDEX edges_object_audit_idx ON edges(object_id);
            """
        )
        connection.commit()

        missing_endpoint_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM edges e
                LEFT JOIN nodes s ON s.node_id = e.subject_id
                LEFT JOIN nodes o ON o.node_id = e.object_id
                WHERE s.node_id IS NULL OR o.node_id IS NULL
                """
            ).fetchone()[0]
        )
        endpoint_type_mismatch_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM edges e
                JOIN nodes s ON s.node_id = e.subject_id
                JOIN nodes o ON o.node_id = e.object_id
                WHERE s.node_type <> e.subject_type
                   OR o.node_type <> e.object_type
                """
            ).fetchone()[0]
        )
        isolated_scientific_fact_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM nodes n
                WHERE n.node_type NOT IN (
                    'ModelAsset', 'Software', 'SourceDocument'
                )
                AND NOT EXISTS (
                    SELECT 1 FROM edges e WHERE e.subject_id = n.node_id
                )
                AND NOT EXISTS (
                    SELECT 1 FROM edges e WHERE e.object_id = n.node_id
                )
                """
            ).fetchone()[0]
        )
        for run_id in sorted(grade_3_validation_run_ids):
            incoming_model_edges = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM edges
                    WHERE object_id = ?
                      AND predicate = 'HAS_VALIDATION_RUN'
                    """,
                    (run_id,),
                ).fetchone()[0]
            )
            outgoing_counts = {
                str(predicate): int(count)
                for predicate, count in connection.execute(
                    """
                    SELECT predicate, COUNT(*) FROM edges
                    WHERE subject_id = ? GROUP BY predicate
                    """,
                    (run_id,),
                )
            }
            expected_datasets = grade_3_expected_datasets[run_id]
            if incoming_model_edges != 1:
                validation_run_errors.append(
                    f"{run_id}:model_version_binding_count="
                    f"{incoming_model_edges}"
                )
            if outgoing_counts.get("VALIDATED_WITH", 0) != expected_datasets:
                validation_run_errors.append(
                    f"{run_id}:validation_dataset_edge_count="
                    f"{outgoing_counts.get('VALIDATED_WITH', 0)}:"
                    f"expected={expected_datasets}"
                )
            if outgoing_counts.get("REPORTS", 0) < 4:
                validation_run_errors.append(
                    f"{run_id}:validation_metric_count="
                    f"{outgoing_counts.get('REPORTS', 0)}"
                )
            if outgoing_counts.get("SUPPORTED_BY", 0) < 1:
                validation_run_errors.append(
                    f"{run_id}:missing_validation_evidence"
                )
        unresolved_biological_entity_nodes = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT n.node_id
                FROM nodes n
                JOIN unresolved_process_partners u
                  ON u.normalized_identifier = n.normalized_name
                WHERE n.node_type IN (
                    'Compound', 'Metabolite', 'Enzyme', 'Transporter',
                    'Protein', 'MolecularComplex', 'MolecularEntity'
                )
                ORDER BY n.node_id
                """
            )
        ]
        unresolved_biological_entity_count = len(
            unresolved_biological_entity_nodes
        )
        node_count = int(
            connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        )
        edge_count = int(
            connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        )
        connection.close()

    snapshot = sqlite3.connect(
        f"file:{graph_directory / 'model_kg.sqlite'}?mode=ro",
        uri=True,
    )
    sqlite_node_count = int(
        snapshot.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    )
    sqlite_edge_count = int(
        snapshot.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    )
    sqlite_node_type_counts = {
        str(key): int(value)
        for key, value in snapshot.execute(
            """
            SELECT node_type, COUNT(*)
            FROM nodes GROUP BY node_type ORDER BY node_type
            """
        )
    }
    sqlite_predicate_counts = {
        str(key): int(value)
        for key, value in snapshot.execute(
            """
            SELECT predicate, COUNT(*)
            FROM edges GROUP BY predicate ORDER BY predicate
            """
        )
    }
    sqlite_unresolved_quality_count = int(
        snapshot.execute(
            "SELECT COUNT(*) FROM unresolved_process_partners"
        ).fetchone()[0]
    )
    snapshot.close()

    input_record_counts = summary.get("input_record_counts") or {}
    checks = {
        "input_gate1_audit_accepted": bool(
            summary.get("accepted_input_gate1_audit")
        ),
        "output_checksums_match": not checksum_mismatches,
        "node_ids_unique": not duplicate_node_ids,
        "edge_ids_unique": not duplicate_edge_ids,
        "node_types_valid": not invalid_node_types,
        "schema_domain_range_violations_zero": (
            not relation_violations
        ),
        "all_edge_endpoints_exist": missing_endpoint_count == 0,
        "edge_endpoint_types_match": (
            endpoint_type_mismatch_count == 0
        ),
        "isolated_scientific_facts_zero": (
            isolated_scientific_fact_count == 0
        ),
        "all_model_facts_have_provenance": not provenance_errors,
        "model_version_status_complete": not model_version_errors,
        "grade_3_validation_runs_complete": not validation_run_errors,
        "semantic_concept_projection_valid": (
            not concept_projection_errors
            and parameter_occurrence_count
            == input_record_counts.get("parameters")
            and component_occurrence_count
            == input_record_counts.get("components")
            and process_occurrence_count
            == input_record_counts.get("processes")
        ),
        "opaque_component_ids_not_biological_entities": (
            not unresolved_biological_entity_nodes
        ),
        "unresolved_partners_not_biological_entities": (
            unresolved_biological_entity_count == 0
        ),
        "unresolved_quality_manifest_valid": (
            not unresolved_quality_errors
        ),
        "unresolved_quality_parity": (
            unresolved_quality_count
            == summary.get(
                "unresolved_process_partner_unique_count"
            )
            == sqlite_unresolved_quality_count
            and unresolved_quality_occurrence_count
            == input_record_counts.get(
                "unresolved_process_partner_occurrences"
            )
        ),
        "summary_counts_match_jsonl": (
            node_count == summary.get("node_count")
            and edge_count == summary.get("edge_count")
            and dict(sorted(node_type_counts.items()))
            == summary.get("node_type_counts")
            and dict(sorted(predicate_counts.items()))
            == summary.get("predicate_counts")
        ),
        "sqlite_projection_matches_jsonl": (
            sqlite_node_count == node_count
            and sqlite_edge_count == edge_count
            and sqlite_node_type_counts
            == dict(sorted(node_type_counts.items()))
            and sqlite_predicate_counts
            == dict(sorted(predicate_counts.items()))
        ),
        "failures_empty": (
            (graph_directory / "failures.jsonl").stat().st_size == 0
        ),
    }
    return {
        "audit_schema_version": "1.0.0",
        "auditor_version": AUDITOR_VERSION,
        "audited_at": datetime.now(UTC).isoformat(),
        "accepted": all(checks.values()),
        "checks": checks,
        "counts": {
            "nodes": node_count,
            "edges": edge_count,
            "missing_endpoints": missing_endpoint_count,
            "endpoint_type_mismatches": endpoint_type_mismatch_count,
            "isolated_scientific_facts": (
                isolated_scientific_fact_count
            ),
            "provenance_errors": len(provenance_errors),
            "relation_violations": len(relation_violations),
            "model_version_errors": len(model_version_errors),
            "grade_3_validation_runs": len(grade_3_validation_run_ids),
            "validation_run_errors": len(validation_run_errors),
            "unresolved_process_partner_unique": (
                unresolved_quality_count
            ),
            "unresolved_process_partner_occurrences": (
                unresolved_quality_occurrence_count
            ),
            "unresolved_biological_entities": (
                unresolved_biological_entity_count
            ),
            "opaque_biological_entities": len(
                unresolved_biological_entity_nodes
            ),
            "parameter_fact_occurrences": parameter_occurrence_count,
            "component_fact_occurrences": component_occurrence_count,
            "process_fact_occurrences": process_occurrence_count,
        },
        "errors": {
            "checksum_mismatches": checksum_mismatches[:100],
            "duplicate_node_ids": duplicate_node_ids[:100],
            "duplicate_edge_ids": duplicate_edge_ids[:100],
            "invalid_node_types": invalid_node_types[:100],
            "relation_violations": relation_violations[:100],
            "provenance_errors": provenance_errors[:100],
            "model_versions": model_version_errors[:100],
            "validation_runs": validation_run_errors[:100],
            "semantic_concept_projection": (
                concept_projection_errors[:100]
            ),
            "opaque_biological_entities": (
                unresolved_biological_entity_nodes[:100]
            ),
            "unresolved_quality": unresolved_quality_errors[:100],
        },
        "graph_summary_sha256": _sha256_file(
            graph_directory / "summary.json"
        ),
    }


def write_model_kg_audit(
    path: Path,
    report: dict[str, Any],
) -> str:
    return _write_json_atomic(path, report)
