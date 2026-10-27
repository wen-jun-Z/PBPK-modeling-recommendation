from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.api.evidence_routes import serialize_row
from pbpk_kb.storage.db import get_db_session


router = APIRouter(tags=["compounds"])


@router.get("/compounds/{compound_id}")
def get_compound(compound_id: str, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    compound = db.execute(
        text(
            """
            SELECT
                compound_id::text,
                preferred_name,
                normalized_name::text,
                synonyms,
                pubchem_cid,
                chembl_id,
                drugcentral_id,
                rxnorm_rxcui,
                inchi_key,
                cas_number,
                approval_status,
                source_id,
                source_location,
                confidence_score::float,
                curation_status::text,
                created_at,
                updated_at
            FROM compound_identity
            WHERE compound_id = CAST(:compound_id AS uuid)
            """
        ),
        {"compound_id": compound_id},
    ).mappings().first()
    if compound is None:
        raise HTTPException(status_code=404, detail="Compound not found")

    evidence_counts = db.execute(
        text(
            """
            SELECT evidence_type, COUNT(*) AS record_count
            FROM auto_validated_evidence
            WHERE attributed_compound_id = CAST(:compound_id AS uuid)
            GROUP BY evidence_type
            ORDER BY evidence_type
            """
        ),
        {"compound_id": compound_id},
    ).mappings().all()

    source_documents = db.execute(
        text(
            """
            WITH linked_documents AS (
                SELECT parsed_document_id
                FROM auto_validated_evidence
                WHERE attributed_compound_id = CAST(:compound_id AS uuid)
            )
            SELECT DISTINCT
                parsed.parsed_document_id::text,
                parsed.source_id,
                source.source_name,
                parsed.document_type,
                parsed.title,
                parsed.source_location,
                parsed.parser_name,
                parsed.parser_version,
                parsed.created_at
            FROM linked_documents linked
            JOIN parsed_document parsed ON parsed.parsed_document_id = linked.parsed_document_id
            LEFT JOIN source_registry source ON source.source_id = parsed.source_id
            WHERE linked.parsed_document_id IS NOT NULL
            ORDER BY parsed.source_id, parsed.title
            """
        ),
        {"compound_id": compound_id},
    ).mappings().all()

    model_count = db.execute(
        text(
            """
            SELECT COUNT(*)
            FROM public_pbpk_model_registry
            WHERE compound_id = CAST(:compound_id AS uuid)
              AND COALESCE(validation_status, '') <> 'model_mention_only'
            """
        ),
        {"compound_id": compound_id},
    ).scalar_one()

    return {
        "compound": serialize_row(dict(compound)),
        "evidence_counts": [serialize_row(dict(row)) for row in evidence_counts],
        "source_documents": [serialize_row(dict(row)) for row in source_documents],
        "public_model_count": model_count,
    }
