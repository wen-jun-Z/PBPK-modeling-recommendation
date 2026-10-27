from __future__ import annotations

from typing import Any


def evidence_citation(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "citation_id": f"E{index}",
        "kind": "structured_evidence",
        "source_id": row.get("source_id"),
        "source_name": row.get("source_name"),
        "source_location": row.get("source_location"),
        "evidence_type": row.get("evidence_type"),
        "evidence_id": row.get("evidence_id"),
        "curation_status": row.get("curation_status"),
        "extraction_method": row.get("extraction_method"),
    }


def model_citation(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "citation_id": f"M{index}",
        "kind": "public_model",
        "source_id": row.get("source_id"),
        "source_name": row.get("source_name"),
        "source_location": row.get("source_location"),
        "model_id": row.get("model_id"),
        "model_name": row.get("model_name"),
        "curation_status": row.get("curation_status"),
        "validation_status": row.get("validation_status"),
    }


def chunk_citation(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "citation_id": f"D{index}",
        "kind": "document_chunk",
        "source_id": row.get("source_id"),
        "source_name": row.get("source_name"),
        "source_location": row.get("source_location"),
        "chunk_id": row.get("chunk_id"),
        "document_id": row.get("document_id"),
        "section_title": row.get("section_title"),
        "embedding_model": row.get("embedding_model"),
    }


def graph_citation(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "citation_id": f"G{index}",
        "kind": "graph_relationship",
        "source_id": row.get("source_id"),
        "source_name": row.get("source_name"),
        "source_location": row.get("source_location"),
        "relationship_id": row.get("relationship_id"),
        "relationship_type": row.get("relationship_type"),
        "target_name": row.get("target_name"),
        "gene_symbol": row.get("gene_symbol"),
        "curation_status": row.get("curation_status"),
        "extraction_method": row.get("extraction_method"),
    }


def attach_citation_ids(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [{**row, "citation_id": f"{prefix}{index}"} for index, row in enumerate(rows, start=1)]
