from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.api.evidence_routes import serialize_row


def _add_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    nodes.setdefault(node["id"], node)


def _source_node(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"source:{row['source_id']}",
        "type": "source",
        "label": row.get("source_name") or row["source_id"],
        "properties": {"source_id": row["source_id"], "source_name": row.get("source_name")},
    }


def _compound_node(row: dict[str, Any]) -> dict[str, Any]:
    compound_id = row.get("compound_id")
    return {
        "id": f"compound:{compound_id}",
        "type": "compound",
        "label": row.get("compound_name") or row.get("preferred_name") or "compound",
        "properties": {key: value for key, value in row.items() if key.startswith("compound") or key == "preferred_name"},
    }


def _evidence_node(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"evidence:{row['evidence_type']}:{row['evidence_id']}",
        "type": "evidence",
        "label": row.get("parameter_name") or row.get("evidence_type") or "evidence",
        "properties": row,
    }


def compound_graph(db: Session, compound_id: str) -> dict[str, Any] | None:
    compound = db.execute(
        text(
            """
            SELECT compound_id::text, preferred_name, pubchem_cid, chembl_id,
                   drugcentral_id, rxnorm_rxcui, inchi_key
            FROM compound_identity
            WHERE compound_id = CAST(:compound_id AS uuid)
            """
        ),
        {"compound_id": compound_id},
    ).mappings().first()
    if compound is None:
        return None

    evidence = db.execute(
        text(
            """
            SELECT
                e.evidence_type,
                e.evidence_id::text,
                e.attributed_compound_id::text AS compound_id,
                e.attributed_compound_name AS compound_name,
                e.source_id,
                source.source_name,
                e.parameter AS parameter_name,
                e.curation_status::text,
                e.extraction_confidence::float AS confidence_score,
                e.source_location,
                e.compound_role,
                e.evidence_layer
            FROM auto_validated_evidence e
            LEFT JOIN source_registry source ON source.source_id = e.source_id
            WHERE e.attributed_compound_id = CAST(:compound_id AS uuid)
            ORDER BY e.evidence_type, e.parameter, e.source_id
            LIMIT 500
            """
        ),
        {"compound_id": compound_id},
    ).mappings().all()

    models = db.execute(
        text(
            """
            SELECT model_id::text, compound_id::text, source_id, model_name, software,
                   validation_status, curation_status::text, source_location
            FROM public_pbpk_model_registry
            WHERE compound_id = CAST(:compound_id AS uuid)
              AND model_file_path IS NOT NULL
            ORDER BY model_name
            """
        ),
        {"compound_id": compound_id},
    ).mappings().all()
    relationships = graph_relationship_rows(db, compound_id=compound_id, limit=500)

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    compound_row = serialize_row(dict(compound))
    compound_row["compound_name"] = compound_row["preferred_name"]
    compound_node = _compound_node(compound_row)
    _add_node(nodes, compound_node)

    for raw_row in evidence:
        row = serialize_row(dict(raw_row))
        evidence_node = _evidence_node(row)
        source_node = _source_node(row)
        _add_node(nodes, evidence_node)
        _add_node(nodes, source_node)
        edges.append({"source": compound_node["id"], "target": evidence_node["id"], "type": "HAS_EVIDENCE"})
        edges.append({"source": evidence_node["id"], "target": source_node["id"], "type": "DERIVED_FROM"})

    for raw_row in models:
        row = serialize_row(dict(raw_row))
        model_node_id = f"model:{row['model_id']}"
        source_node_id = f"source:{row['source_id']}"
        _add_node(nodes, {"id": model_node_id, "type": "model", "label": row["model_name"], "properties": row})
        _add_node(nodes, {"id": source_node_id, "type": "source", "label": row["source_id"], "properties": {"source_id": row["source_id"]}})
        edges.append({"source": compound_node["id"], "target": model_node_id, "type": "REPRESENTED_BY"})
        edges.append({"source": model_node_id, "target": source_node_id, "type": "DERIVED_FROM"})

    for row in relationships:
        target_node_id = f"protein_target:{row['protein_target_id']}"
        source_node_id = f"source:{row['source_id']}"
        evidence_node_id = f"evidence:{row['evidence_type']}:{row['evidence_id']}"
        _add_node(nodes, {"id": target_node_id, "type": "protein_target", "label": row.get("target_name") or row.get("gene_symbol") or "target", "properties": row})
        _add_node(nodes, _source_node(row))
        edges.append({"source": compound_node["id"], "target": target_node_id, "type": row["relationship_type"], "properties": row})
        if evidence_node_id in nodes:
            edges.append({"source": evidence_node_id, "target": target_node_id, "type": "SUPPORTS_RELATIONSHIP"})
        edges.append({"source": target_node_id, "target": source_node_id, "type": "DERIVED_FROM"})

    return _graph_response(
        center={"type": "compound", "id": compound_id, "label": compound_row["preferred_name"]},
        nodes=nodes,
        edges=edges,
        evidence_count=len(evidence),
        model_count=len(models),
        relationship_count=len(relationships),
        compound=compound_row,
    )


def target_graph(db: Session, target_id: str, limit: int = 500) -> dict[str, Any] | None:
    target = db.execute(
        text(
            """
            SELECT protein_target_id::text, preferred_name, gene_symbol, uniprot_id,
                   target_type, synonyms, source_id, source_location,
                   confidence_score::float, curation_status::text
            FROM protein_target_identity
            WHERE protein_target_id = CAST(:target_id AS uuid)
            """
        ),
        {"target_id": target_id},
    ).mappings().first()
    if target is None:
        return None
    target_row = serialize_row(dict(target))
    target_node_id = f"protein_target:{target_id}"
    nodes = {
        target_node_id: {
            "id": target_node_id,
            "type": "protein_target",
            "label": target_row.get("preferred_name") or target_row.get("gene_symbol"),
            "properties": target_row,
        }
    }
    edges: list[dict[str, Any]] = []
    relationships = graph_relationship_rows(db, target_id=target_id, limit=limit)
    evidence_ids: set[str] = set()
    compound_ids: set[str] = set()
    for row in relationships:
        compound_node = _compound_node(row)
        source_node = _source_node(row)
        evidence_node = _evidence_node(row)
        _add_node(nodes, compound_node)
        _add_node(nodes, source_node)
        _add_node(nodes, evidence_node)
        compound_ids.add(str(row["compound_id"]))
        evidence_ids.add(str(row["evidence_id"]))
        edges.append({"source": compound_node["id"], "target": target_node_id, "type": row["relationship_type"], "properties": row})
        edges.append({"source": evidence_node["id"], "target": target_node_id, "type": "SUPPORTS_RELATIONSHIP"})
        edges.append({"source": evidence_node["id"], "target": source_node["id"], "type": "DERIVED_FROM"})
    return _graph_response(
        center={"type": "protein_target", "id": target_id, "label": target_row["preferred_name"]},
        nodes=nodes,
        edges=edges,
        evidence_count=len(evidence_ids),
        relationship_count=len(relationships),
        extra_counts={"compounds": len(compound_ids)},
    )


def source_graph(db: Session, source_id: str, limit: int = 500) -> dict[str, Any] | None:
    source = db.execute(
        text("SELECT source_id, source_name, source_family FROM source_registry WHERE source_id = :source_id"),
        {"source_id": source_id},
    ).mappings().first()
    if source is None:
        return None
    source_row = serialize_row(dict(source))
    source_node = _source_node(source_row)
    nodes = {source_node["id"]: source_node}
    edges: list[dict[str, Any]] = []
    evidence = db.execute(
        text(
            """
            SELECT e.evidence_type, e.evidence_id::text,
                   e.attributed_compound_id::text AS compound_id,
                   e.attributed_compound_name AS compound_name,
                   e.source_id, source.source_name, e.parameter AS parameter_name,
                   e.curation_status::text, e.extraction_confidence::float AS confidence_score,
                   e.source_location, e.compound_role, e.evidence_layer
            FROM auto_validated_evidence e
            LEFT JOIN source_registry source ON source.source_id = e.source_id
            WHERE e.source_id = :source_id
            ORDER BY e.attributed_compound_name, e.evidence_type, e.parameter
            LIMIT :limit
            """
        ),
        {"source_id": source_id, "limit": limit},
    ).mappings().all()
    compound_ids: set[str] = set()
    for raw_row in evidence:
        row = serialize_row(dict(raw_row))
        compound_node = _compound_node(row)
        evidence_node = _evidence_node(row)
        _add_node(nodes, compound_node)
        _add_node(nodes, evidence_node)
        compound_ids.add(str(row["compound_id"]))
        edges.append({"source": compound_node["id"], "target": evidence_node["id"], "type": "HAS_EVIDENCE"})
        edges.append({"source": evidence_node["id"], "target": source_node["id"], "type": "DERIVED_FROM"})

    models = db.execute(
        text(
            """
            SELECT model.model_id::text, model.model_name, model.compound_id::text,
                   compound.preferred_name AS compound_name, model.source_id,
                   source.source_name, model.validation_status, model.curation_status::text,
                   model.source_location
            FROM public_pbpk_model_registry model
            LEFT JOIN compound_identity compound ON compound.compound_id = model.compound_id
            LEFT JOIN source_registry source ON source.source_id = model.source_id
            WHERE model.source_id = :source_id AND model.model_file_path IS NOT NULL
            ORDER BY model.model_name
            LIMIT :limit
            """
        ),
        {"source_id": source_id, "limit": limit},
    ).mappings().all()
    for raw_row in models:
        row = serialize_row(dict(raw_row))
        model_node_id = f"model:{row['model_id']}"
        _add_node(nodes, {"id": model_node_id, "type": "model", "label": row["model_name"], "properties": row})
        if row.get("compound_id"):
            compound_node = _compound_node(row)
            _add_node(nodes, compound_node)
            compound_ids.add(str(row["compound_id"]))
            edges.append({"source": compound_node["id"], "target": model_node_id, "type": "REPRESENTED_BY"})
        edges.append({"source": model_node_id, "target": source_node["id"], "type": "DERIVED_FROM"})
    return _graph_response(
        center={"type": "source", "id": source_id, "label": source_row["source_name"]},
        nodes=nodes,
        edges=edges,
        evidence_count=len(evidence),
        model_count=len(models),
        extra_counts={"compounds": len(compound_ids)},
    )


def model_graph(db: Session, model_id: str) -> dict[str, Any] | None:
    model = db.execute(
        text(
            """
            SELECT model.model_id::text, model.model_name, model.compound_id::text,
                   compound.preferred_name AS compound_name, model.model_type, model.software,
                   model.source_id, source.source_name, model.model_file_path,
                   model.parameter_file_path, model.evaluation_report_path,
                   model.validation_status, model.curation_status::text, model.source_location
            FROM public_pbpk_model_registry model
            LEFT JOIN compound_identity compound ON compound.compound_id = model.compound_id
            LEFT JOIN source_registry source ON source.source_id = model.source_id
            WHERE model.model_id = CAST(:model_id AS uuid) AND model.model_file_path IS NOT NULL
            """
        ),
        {"model_id": model_id},
    ).mappings().first()
    if model is None:
        return None
    model_row = serialize_row(dict(model))
    model_node_id = f"model:{model_id}"
    nodes = {model_node_id: {"id": model_node_id, "type": "model", "label": model_row["model_name"], "properties": model_row}}
    edges: list[dict[str, Any]] = []
    if model_row.get("compound_id"):
        compound_node = _compound_node(model_row)
        _add_node(nodes, compound_node)
        edges.append({"source": compound_node["id"], "target": model_node_id, "type": "REPRESENTED_BY"})
    if model_row.get("source_id"):
        source_node = _source_node(model_row)
        _add_node(nodes, source_node)
        edges.append({"source": model_node_id, "target": source_node["id"], "type": "DERIVED_FROM"})

    parameters = db.execute(
        text(
            """
            SELECT model_parameter_evidence_id::text AS evidence_id, parameter_name,
                   parameter_value::float, parameter_value_text, unit, source_id,
                   source_location, curation_status::text
            FROM model_parameter_evidence
            WHERE model_id = CAST(:model_id AS uuid)
            ORDER BY parameter_name
            LIMIT 500
            """
        ),
        {"model_id": model_id},
    ).mappings().all()
    for raw_row in parameters:
        row = serialize_row(dict(raw_row))
        parameter_node_id = f"model_parameter:{row['evidence_id']}"
        _add_node(nodes, {"id": parameter_node_id, "type": "model_parameter", "label": row["parameter_name"], "properties": row})
        edges.append({"source": model_node_id, "target": parameter_node_id, "type": "USES_PARAMETER"})
    return _graph_response(
        center={"type": "model", "id": model_id, "label": model_row["model_name"]},
        nodes=nodes,
        edges=edges,
        model_count=1,
        extra_counts={"parameters": len(parameters)},
    )


def graph_relationship_rows(
    db: Session,
    *,
    compound_id: str | None = None,
    target_id: str | None = None,
    source_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT rel.relationship_id::text, rel.compound_id::text,
                   compound.preferred_name AS compound_name, rel.relationship_type,
                   rel.evidence_direction, rel.source_id, source.source_name,
                   rel.source_location, rel.evidence_text, rel.extraction_method,
                   rel.confidence_score::float, rel.curation_status::text,
                   rel.processed_evidence_id::text AS evidence_id,
                   final.evidence_type, final.parameter AS parameter_name,
                   target.protein_target_id::text, target.preferred_name AS target_name,
                   target.gene_symbol, target.uniprot_id, target.target_type
            FROM enzyme_transporter_relationship rel
            JOIN auto_validated_evidence final ON final.evidence_id = rel.processed_evidence_id
            LEFT JOIN compound_identity compound ON compound.compound_id = rel.compound_id
            LEFT JOIN protein_target_identity target ON target.protein_target_id = rel.protein_target_id
            LEFT JOIN source_registry source ON source.source_id = rel.source_id
            WHERE (CAST(:compound_id AS text) IS NULL OR rel.compound_id::text = CAST(:compound_id AS text))
              AND (CAST(:target_id AS text) IS NULL OR rel.protein_target_id::text = CAST(:target_id AS text))
              AND (CAST(:source_id AS text) IS NULL OR rel.source_id = CAST(:source_id AS text))
            ORDER BY rel.relationship_type, target.preferred_name, compound.preferred_name
            LIMIT :limit
            """
        ),
        {"compound_id": compound_id, "target_id": target_id, "source_id": source_id, "limit": limit},
    ).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


def graph_centers(db: Session, center_type: str) -> list[dict[str, Any]]:
    if center_type == "protein_target":
        sql = """
            SELECT target.protein_target_id::text AS id, target.preferred_name AS label,
                   target.gene_symbol, target.target_type, COUNT(rel.relationship_id) AS relationship_count
            FROM protein_target_identity target
            JOIN enzyme_transporter_relationship rel ON rel.protein_target_id = target.protein_target_id
            JOIN auto_validated_evidence final ON final.evidence_id = rel.processed_evidence_id
            GROUP BY target.protein_target_id, target.preferred_name, target.gene_symbol, target.target_type
            ORDER BY target.preferred_name
        """
    elif center_type == "source":
        sql = """
            WITH active_sources AS (
                SELECT source_id FROM auto_validated_evidence
                UNION SELECT source_id FROM public_pbpk_model_registry WHERE model_file_path IS NOT NULL
            )
            SELECT source.source_id AS id, source.source_name AS label, source.source_family
            FROM active_sources active
            JOIN source_registry source ON source.source_id = active.source_id
            ORDER BY source.source_name
        """
    elif center_type == "model":
        sql = """
            SELECT model.model_id::text AS id, model.model_name AS label,
                   compound.preferred_name AS compound_name, model.software
            FROM public_pbpk_model_registry model
            LEFT JOIN compound_identity compound ON compound.compound_id = model.compound_id
            WHERE model.model_file_path IS NOT NULL
            ORDER BY model.model_name
        """
    else:
        raise ValueError(f"Unsupported graph center type: {center_type}")
    rows = db.execute(text(sql)).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


def graph_summary(db: Session) -> dict[str, Any]:
    row = db.execute(
        text(
            """
            WITH active_sources AS (
                SELECT source_id FROM auto_validated_evidence
                UNION SELECT source_id FROM public_pbpk_model_registry WHERE model_file_path IS NOT NULL
            )
            SELECT
                (SELECT COUNT(*) FROM compound_identity) AS compounds,
                (SELECT COUNT(*) FROM active_sources) AS sources,
                (SELECT COUNT(*) FROM public_pbpk_model_registry WHERE model_file_path IS NOT NULL) AS public_models,
                (SELECT COUNT(*) FROM protein_target_identity target WHERE EXISTS (
                    SELECT 1 FROM enzyme_transporter_relationship rel
                    JOIN auto_validated_evidence final ON final.evidence_id = rel.processed_evidence_id
                    WHERE rel.protein_target_id = target.protein_target_id
                )) AS protein_targets,
                (SELECT COUNT(*) FROM enzyme_transporter_relationship rel
                 JOIN auto_validated_evidence final ON final.evidence_id = rel.processed_evidence_id) AS enzyme_transporter_relationships,
                (SELECT COUNT(*) FROM tissue_expression_evidence) AS tissue_expression_records,
                (SELECT COUNT(*) FROM auto_validated_evidence) AS evidence_records
            """
        )
    ).mappings().one()
    summary = serialize_row(dict(row))
    summary["graph_nodes_minimum"] = summary["compounds"] + summary["sources"] + summary["public_models"] + summary["protein_targets"] + summary["evidence_records"]
    summary["graph_edges_minimum"] = summary["evidence_records"] * 2 + summary["public_models"] * 2 + summary["enzyme_transporter_relationships"] * 2
    return summary


def _graph_response(
    *,
    center: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    evidence_count: int = 0,
    model_count: int = 0,
    relationship_count: int = 0,
    compound: dict[str, Any] | None = None,
    extra_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    counts = {
        "nodes": len(nodes),
        "edges": len(edges),
        "evidence": evidence_count,
        "models": model_count,
        "relationships": relationship_count,
        **(extra_counts or {}),
    }
    return {
        "center": center,
        "compound": compound,
        "nodes": list(nodes.values()),
        "edges": edges,
        "counts": counts,
    }
