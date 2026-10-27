from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection


NON_TARGET_NAMES = {
    "no relevant target",
    "rattus norvegicus",
    "cavia porcellus",
    "equus caballus",
    "homo sapiens",
}


def insert_chembl_target_relationships(
    connection: Connection,
    *,
    record: dict[str, Any],
    parsed_document_id: Any,
    compound_id: Any,
) -> int:
    payload = record.get("record") or {}
    inserted = 0
    for activity in payload.get("activities") or []:
        if not _has_relevant_target(activity):
            continue
        target_id = _upsert_protein_target(connection, activity)
        source_location = _activity_source_location(record, activity)
        connection.execute(
            text(
                """
                INSERT INTO enzyme_transporter_relationship (
                    compound_id, protein_target_id, relationship_type, evidence_direction,
                    source_id, parsed_document_id, source_location, evidence_text,
                    value_origin, evidence_origin, extraction_method, extractor_version,
                    confidence_score, curation_status
                )
                VALUES (
                    :compound_id, :protein_target_id, :relationship_type, 'compound_to_target',
                    :source_id, :parsed_document_id, :source_location, :evidence_text,
                    'measured', :evidence_origin, 'machine_extracted', '0.1.0',
                    :confidence_score, 'needs_review'
                )
                """
            ),
            {
                "compound_id": compound_id,
                "protein_target_id": target_id,
                "relationship_type": _relationship_type(activity),
                "source_id": record["source_id"],
                "parsed_document_id": parsed_document_id,
                "source_location": source_location,
                "evidence_text": _activity_evidence_text(activity),
                "evidence_origin": _evidence_origin(activity),
                "confidence_score": _confidence_score(activity),
            },
        )
        inserted += 1
    return inserted


def _has_relevant_target(activity: dict[str, Any]) -> bool:
    target_name = str(activity.get("target_pref_name") or "").strip()
    target_chembl_id = str(activity.get("target_chembl_id") or "").strip()
    if not target_name or not target_chembl_id:
        return False
    normalized = target_name.lower()
    if normalized in NON_TARGET_NAMES:
        return False
    organism = str(activity.get("target_organism") or "").strip().lower()
    return normalized != organism


def _upsert_protein_target(connection: Connection, activity: dict[str, Any]) -> Any:
    target_chembl_id = str(activity.get("target_chembl_id"))
    source_location = f"CHEMBL_TARGET:{target_chembl_id}"
    existing = connection.execute(
        text(
            """
            SELECT protein_target_id
            FROM protein_target_identity
            WHERE source_id = 'chembl' AND source_location = :source_location
            LIMIT 1
            """
        ),
        {"source_location": source_location},
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    row = connection.execute(
        text(
            """
            INSERT INTO protein_target_identity (
                preferred_name, gene_symbol, target_type, synonyms, source_id,
                source_location, confidence_score, curation_status
            )
            VALUES (
                :preferred_name, :gene_symbol, :target_type, :synonyms, 'chembl',
                :source_location, 0.75, 'needs_review'
            )
            RETURNING protein_target_id
            """
        ),
        {
            "preferred_name": activity.get("target_pref_name"),
            "gene_symbol": _infer_gene_symbol(activity),
            "target_type": _target_type(activity),
            "synonyms": [target_chembl_id],
            "source_location": source_location,
        },
    ).scalar_one()
    return row


def _infer_gene_symbol(activity: dict[str, Any]) -> str | None:
    name = str(activity.get("target_pref_name") or "")
    for token in name.replace("/", " ").replace("-", " ").split():
        upper = token.upper()
        if upper.startswith(("CYP", "UGT", "SLCO", "ABCB", "SLC", "CES")):
            return upper
    return None


def _target_type(activity: dict[str, Any]) -> str:
    text_value = " ".join(
        str(value or "")
        for value in [activity.get("target_pref_name"), activity.get("assay_description")]
    ).lower()
    if "transporter" in text_value or "abc" in text_value or "slco" in text_value:
        return "transporter"
    if "cytochrome" in text_value or "cyp" in text_value or "reductase" in text_value or "enzyme" in text_value:
        return "enzyme"
    return "protein_target"


def _relationship_type(activity: dict[str, Any]) -> str:
    text_value = " ".join(
        str(value or "")
        for value in [activity.get("standard_type"), activity.get("type"), activity.get("assay_description")]
    ).lower()
    if any(token in text_value for token in ["inhibition", "inhibit", "ic50", "ki"]):
        return "inhibits_or_binds"
    if "substrate" in text_value:
        return "substrate_of"
    if "induction" in text_value or "inducer" in text_value:
        return "induces"
    return "has_activity_against"


def _evidence_origin(activity: dict[str, Any]) -> str:
    organism = str(activity.get("target_organism") or "").lower()
    if "homo sapiens" in organism:
        return "human"
    if organism:
        return "animal"
    return "unknown"


def _confidence_score(activity: dict[str, Any]) -> float:
    if activity.get("standard_value") is not None and activity.get("standard_type"):
        return 0.72
    return 0.62


def _activity_source_location(record: dict[str, Any], activity: dict[str, Any]) -> str:
    activity_id = activity.get("activity_id") or activity.get("record_id") or activity.get("target_chembl_id")
    return f"{record['source_location']}::CHEMBL_ACTIVITY:{activity_id}"


def _activity_evidence_text(activity: dict[str, Any]) -> str:
    payload = {
        "target_chembl_id": activity.get("target_chembl_id"),
        "target_pref_name": activity.get("target_pref_name"),
        "target_organism": activity.get("target_organism"),
        "assay_description": activity.get("assay_description"),
        "standard_type": activity.get("standard_type"),
        "standard_value": activity.get("standard_value"),
        "standard_units": activity.get("standard_units"),
        "document_chembl_id": activity.get("document_chembl_id"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
