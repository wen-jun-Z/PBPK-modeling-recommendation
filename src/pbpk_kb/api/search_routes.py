from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.storage.db import get_db_session
from pbpk_kb.embeddings.client import EmbeddingServiceError, embed_query
from pbpk_kb.storage.vector_index import vector_literal


router = APIRouter(tags=["search"])


@router.get("/compounds/search")
def search_compounds(
    q: str = Query(default="", min_length=0),
    limit: int = Query(default=10, ge=1, le=250),
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    pattern = f"%{q}%"
    rows = db.execute(
        text(
            """
            WITH evidence_rows AS (
                SELECT
                    attributed_compound_id AS compound_id,
                    source_id,
                    parsed_document_id,
                    evidence_type
                FROM auto_validated_evidence
            ),
            evidence_counts AS (
                SELECT
                    compound_id,
                    COUNT(*) AS evidence_count,
                    COUNT(*) FILTER (WHERE evidence_type = 'clinical_pk') AS clinical_pk_count,
                    COUNT(DISTINCT parsed_document_id) FILTER (WHERE source_id = 'dailymed') AS label_count
                FROM evidence_rows
                WHERE compound_id IS NOT NULL
                GROUP BY compound_id
            ),
            model_counts AS (
                SELECT compound_id, COUNT(*) AS public_model_count
                FROM public_pbpk_model_registry
                WHERE compound_id IS NOT NULL
                  AND COALESCE(validation_status, '') <> 'model_mention_only'
                GROUP BY compound_id
            )
            SELECT
                c.compound_id::text,
                c.preferred_name,
                c.normalized_name::text,
                c.synonyms,
                c.pubchem_cid,
                c.chembl_id,
                c.drugcentral_id,
                c.rxnorm_rxcui,
                c.inchi_key,
                c.approval_status,
                c.source_id,
                c.source_location,
                c.confidence_score::float,
                c.curation_status::text,
                COALESCE(e.evidence_count, 0) AS evidence_count,
                COALESCE(e.clinical_pk_count, 0) AS clinical_pk_count,
                COALESCE(e.label_count, 0) AS label_count,
                COALESCE(m.public_model_count, 0) AS public_model_count
            FROM compound_identity c
            LEFT JOIN evidence_counts e ON e.compound_id = c.compound_id
            LEFT JOIN model_counts m ON m.compound_id = c.compound_id
            WHERE
                CAST(:q AS text) = ''
                OR c.preferred_name ILIKE :pattern
                OR c.normalized_name ILIKE :pattern
                OR c.pubchem_cid = :q
                OR c.chembl_id = :q
                OR c.drugcentral_id = :q
                OR c.rxnorm_rxcui = :q
                OR c.inchi_key = :q
                OR EXISTS (
                    SELECT 1 FROM unnest(c.synonyms) AS synonym
                    WHERE synonym ILIKE :pattern
                )
            ORDER BY c.confidence_score DESC NULLS LAST, c.preferred_name
            LIMIT :limit
            """
        ),
        {"q": q, "pattern": pattern, "limit": limit},
    ).mappings()
    return [dict(row) for row in rows]


@router.get("/evidence/search")
def search_evidence(
    compound_id: str | None = None,
    compound: str | None = None,
    evidence_type: str | None = None,
    source_id: str | None = None,
    source_family: str | None = None,
    species: str | None = None,
    population: str | None = None,
    route: str | None = None,
    dose: str | None = None,
    enzyme_or_transporter: str | None = None,
    source_date_from: str | None = None,
    source_date_to: str | None = None,
    curation_status: str | None = None,
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    q: str | None = None,
    limit: int = Query(default=20, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            WITH evidence AS (
                SELECT
                    e.evidence_type,
                    e.evidence_id::text AS evidence_id,
                    e.attributed_compound_id::text AS compound_id,
                    CASE
                        WHEN e.evidence_type = 'ddi' AND e.compound_role IN ('victim_drug', 'perpetrator_drug')
                            THEN e.compound_id::text
                        ELSE NULL::text
                    END AS related_compound_id,
                    e.attributed_compound_name AS compound_name,
                    e.parameter AS parameter_name,
                    e.value::float AS numeric_value,
                    e.value_text,
                    e.unit,
                    e.canonical_value::float,
                    e.canonical_unit,
                    NULL::text AS unit_dimension,
                    CONCAT_WS(' / ', e.condition, e.population, e.route, e.dose, e.compound_role) AS context,
                    e.source_id,
                    e.source_location,
                    CASE WHEN jsonb_typeof(e.source_record->'species') = 'array'
                        THEN (SELECT string_agg(value, '; ')
                              FROM jsonb_array_elements_text(e.source_record->'species'))
                        ELSE NULLIF(e.source_record->>'species', '') END AS species,
                    e.population,
                    e.route,
                    e.dose,
                    CASE
                        WHEN e.evidence_type IN ('adme', 'ddi') THEN e.parameter
                        ELSE NULL::text
                    END AS enzyme_or_transporter,
                    CASE WHEN e.value IS NOT NULL OR e.value_text IS NOT NULL THEN 'extracted' ELSE 'not_extracted' END AS value_origin,
                    e.extraction_method AS evidence_origin,
                    e.source_text AS evidence_text,
                    e.extraction_method,
                    e.extraction_confidence::float AS confidence_score,
                    e.curation_status::text,
                    e.created_at,
                    COALESCE(parsed.publication_date, e.created_at::date) AS source_date,
                    e.raw_asset_id,
                    e.storage_path,
                    e.checksum,
                    e.compound_role,
                    e.attribution_method,
                    e.attribution_confidence::float,
                    e.attribution_reason,
                    e.matched_text,
                    e.matched_synonym,
                    e.attribution_stage,
                    'auto_validated'::text AS evidence_layer
                FROM auto_validated_evidence e
                LEFT JOIN parsed_document parsed ON parsed.parsed_document_id = e.parsed_document_id
            )
            SELECT evidence.*, source.source_family
            FROM evidence
            LEFT JOIN source_registry source ON source.source_id = evidence.source_id
            WHERE (
                  NULLIF(CAST(:compound_id AS text), '') IS NULL
                  OR compound_id = CAST(:compound_id AS text)
                  OR related_compound_id = CAST(:compound_id AS text)
              )
              AND (NULLIF(CAST(:compound AS text), '') IS NULL OR compound_name ILIKE :compound_pattern)
              AND (NULLIF(CAST(:evidence_type AS text), '') IS NULL OR evidence_type = CAST(:evidence_type AS text))
              AND (NULLIF(CAST(:source_id AS text), '') IS NULL OR evidence.source_id = CAST(:source_id AS text))
              AND (NULLIF(CAST(:source_family AS text), '') IS NULL OR source.source_family = CAST(:source_family AS text))
              AND (NULLIF(CAST(:species AS text), '') IS NULL OR species ILIKE :species_pattern)
              AND (NULLIF(CAST(:population AS text), '') IS NULL OR population ILIKE :population_pattern)
              AND (NULLIF(CAST(:route AS text), '') IS NULL OR route ILIKE :route_pattern)
              AND (NULLIF(CAST(:dose AS text), '') IS NULL OR dose ILIKE :dose_pattern)
              AND (
                  NULLIF(CAST(:enzyme_or_transporter AS text), '') IS NULL
                  OR enzyme_or_transporter ILIKE :enzyme_or_transporter_pattern
              )
              AND (NULLIF(CAST(:source_date_from AS text), '') IS NULL OR source_date >= CAST(NULLIF(CAST(:source_date_from AS text), '') AS date))
              AND (NULLIF(CAST(:source_date_to AS text), '') IS NULL OR source_date <= CAST(NULLIF(CAST(:source_date_to AS text), '') AS date))
              AND (NULLIF(CAST(:curation_status AS text), '') IS NULL OR curation_status = CAST(:curation_status AS text))
              AND (confidence_score IS NULL OR confidence_score >= :min_confidence)
              AND (NULLIF(CAST(:q AS text), '') IS NULL OR parameter_name ILIKE :q_pattern OR evidence_text ILIKE :q_pattern)
            ORDER BY confidence_score DESC NULLS LAST, evidence_type, parameter_name, evidence_id, compound_id
            LIMIT :limit OFFSET :offset
            """
        ),
        {
            "compound_id": compound_id,
            "compound": compound,
            "compound_pattern": f"%{compound or ''}%",
            "evidence_type": evidence_type,
            "source_id": source_id,
            "source_family": source_family,
            "species": species,
            "species_pattern": f"%{species or ''}%",
            "population": population,
            "population_pattern": f"%{population or ''}%",
            "route": route,
            "route_pattern": f"%{route or ''}%",
            "dose": dose,
            "dose_pattern": f"%{dose or ''}%",
            "enzyme_or_transporter": enzyme_or_transporter,
            "enzyme_or_transporter_pattern": f"%{enzyme_or_transporter or ''}%",
            "source_date_from": source_date_from,
            "source_date_to": source_date_to,
            "curation_status": curation_status,
            "min_confidence": min_confidence,
            "q": q,
            "q_pattern": f"%{q or ''}%",
            "limit": limit,
            "offset": offset,
        },
    ).mappings()
    return [dict(row) for row in rows]


@router.get("/search/semantic")
def semantic_search(
    q: str = Query(..., min_length=1),
    source_id: str | None = None,
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    try:
        embedding = vector_literal(embed_query(q))
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    rows = db.execute(
        text(
            """
            WITH eligible AS (
                SELECT record.*
                FROM rag_retrieval_record record
                WHERE record.evidence_layer = 'auto_validated'
            )
            SELECT
                record.retrieval_record_id::text AS chunk_id,
                record.retrieval_record_id::text AS retrieval_record_id,
                record.source_id,
                source.source_name,
                NULL::text AS document_id,
                record.compound_name AS title,
                'final_evidence_rag'::text AS document_type,
                NULL::text AS section_title,
                NULL::integer AS paragraph_number,
                record.source_location,
                record.source_text AS chunk_text,
                record.embedding_model,
                'auto_validated'::text AS evidence_layer,
                'auto_validated'::text AS curation_status,
                record.record_kinds,
                record.compound_id::text,
                record.compound_name,
                (1 - (record.embedding <=> CAST(:embedding AS vector)))::float AS score
            FROM eligible record
            LEFT JOIN source_registry source ON source.source_id = record.source_id
            WHERE (CAST(:source_id AS text) IS NULL OR record.source_id = CAST(:source_id AS text))
            ORDER BY record.embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
            """
        ),
        {"embedding": embedding, "source_id": source_id, "limit": limit},
    ).mappings()
    return [dict(row) for row in rows]
