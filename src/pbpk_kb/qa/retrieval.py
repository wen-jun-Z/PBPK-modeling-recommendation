from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.api.evidence_routes import serialize_row
from pbpk_kb.embeddings.client import EmbeddingServiceError, embed_query
from pbpk_kb.graph.graph_queries import graph_relationship_rows
from pbpk_kb.storage.final_evidence_query import serving_evidence_sql
from pbpk_kb.storage.vector_index import vector_literal


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "available",
    "compound",
    "data",
    "do",
    "does",
    "evidence",
    "exists",
    "for",
    "is",
    "of",
    "on",
    "public",
    "show",
    "the",
    "there",
    "what",
    "which",
    "with",
}


@dataclass(frozen=True)
class RetrievalResult:
    compound: dict[str, Any] | None
    structured_evidence: list[dict[str, Any]]
    model_records: list[dict[str, Any]]
    document_chunks: list[dict[str, Any]]
    graph_relationships: list[dict[str, Any]] = field(default_factory=list)
    retrieval_notes: list[str] = field(default_factory=list)


def query_terms(question: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]+", question.lower())
    terms = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    seen: set[str] = set()
    return [term for term in terms if not (term in seen or seen.add(term))]


def infer_evidence_type(question: str) -> str | None:
    normalized = question.lower()
    if "clinical pk" in normalized or "clinical pharmacokinetic" in normalized:
        return "clinical_pk"
    if "renal impairment" in normalized or "hepatic impairment" in normalized or "special population" in normalized:
        return "special_population"
    if "ddi" in normalized or "drug interaction" in normalized or "drug-drug" in normalized:
        return "ddi"
    if "fraction unbound" in normalized or "protein binding" in normalized or "bioavailability" in normalized:
        return "adme"
    if "molecular weight" in normalized or "logp" in normalized or "physicochemical" in normalized:
        return "physicochemical"
    return None


def infer_compound(db: Session, question: str, explicit_compound: str | None = None) -> dict[str, Any] | None:
    query = (explicit_compound or question).strip()
    if not query:
        return None
    rows = db.execute(
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
                source_id,
                source_location,
                confidence_score::float,
                curation_status::text
            FROM compound_identity
            WHERE
                LOWER(:query) LIKE '%' || LOWER(preferred_name::text) || '%'
                OR LOWER(:query) LIKE '%' || LOWER(normalized_name::text) || '%'
                OR pubchem_cid = :explicit_compound
                OR chembl_id = :explicit_compound
                OR drugcentral_id = :explicit_compound
                OR rxnorm_rxcui = :explicit_compound
                OR inchi_key = :explicit_compound
                OR EXISTS (
                    SELECT 1 FROM unnest(synonyms) AS synonym
                    WHERE LOWER(:query) LIKE '%' || LOWER(synonym) || '%'
                )
            ORDER BY confidence_score DESC NULLS LAST, preferred_name
            LIMIT 1
            """
        ),
        {"query": query, "explicit_compound": explicit_compound or query},
    ).mappings().first()
    return serialize_row(dict(rows)) if rows else None


def retrieve_structured_evidence(
    db: Session,
    *,
    question: str,
    compound: dict[str, Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    terms = query_terms(question)
    evidence_type_hint = infer_evidence_type(question)
    pattern = "%" + "%".join(terms[:4]) + "%" if terms else "%"
    compound_id = compound.get("compound_id") if compound else None
    compound_name = compound.get("preferred_name") if compound else ""
    rows = db.execute(
        text(
            f"""
            WITH evidence AS (
                {serving_evidence_sql()}
            )
            SELECT *
            FROM evidence
            WHERE (
                CAST(:compound_id AS text) IS NULL
                OR compound_id = CAST(:compound_id AS text)
                OR compound_name ILIKE :compound_pattern
            )
              AND (
                CAST(:evidence_type_hint AS text) IS NULL
                OR evidence_type = CAST(:evidence_type_hint AS text)
              )
              AND (
                CAST(:term_count AS integer) = 0
                OR parameter_name ILIKE :pattern
                OR context ILIKE :pattern
                OR evidence_text ILIKE :pattern
                OR EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:terms AS text[])) AS term
                    WHERE parameter_name ILIKE '%' || term || '%'
                       OR context ILIKE '%' || term || '%'
                       OR evidence_text ILIKE '%' || term || '%'
                )
              )
            ORDER BY
                CASE curation_status
                    WHEN 'auto_validated' THEN 0
                    WHEN 'approved_for_modeling' THEN 1
                    WHEN 'reviewed_accepted' THEN 2
                    WHEN 'needs_review' THEN 3
                    WHEN 'machine_extracted' THEN 4
                    ELSE 5
                END,
                confidence_score DESC NULLS LAST,
                created_at DESC
            LIMIT :limit
            """
        ),
        {
            "compound_id": compound_id,
            "compound_pattern": f"%{compound_name}%",
            "terms": terms,
            "term_count": len(terms),
            "evidence_type_hint": evidence_type_hint,
            "pattern": pattern,
            "limit": limit,
        },
    ).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


def retrieve_model_records(
    db: Session,
    *,
    question: str,
    compound: dict[str, Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    compound_id = compound.get("compound_id") if compound else None
    compound_name = compound.get("preferred_name") if compound else question
    rows = db.execute(
        text(
            """
            SELECT
                model.model_id::text,
                model.model_name,
                model.model_type,
                model.software,
                model.species,
                model.population,
                model.route,
                compound.preferred_name AS compound_name,
                model.model_source,
                model.source_id,
                source.source_name,
                model.model_file_path,
                model.parameter_file_path,
                model.evaluation_report_path,
                model.publication_id,
                model.validation_status,
                model.curation_status::text,
                model.source_location,
                model.notes,
                model.created_at
            FROM public_pbpk_model_registry model
            LEFT JOIN compound_identity compound ON compound.compound_id = model.compound_id
            LEFT JOIN source_registry source ON source.source_id = model.source_id
            WHERE
                COALESCE(model.validation_status, '') <> 'model_mention_only'
                AND (
                    CAST(:compound_id AS text) IS NULL
                    OR model.compound_id::text = CAST(:compound_id AS text)
                    OR model.model_name ILIKE :compound_pattern
                    OR compound.preferred_name ILIKE :compound_pattern
                )
            ORDER BY model.model_name
            LIMIT :limit
            """
        ),
        {
            "compound_id": compound_id,
            "compound_pattern": f"%{compound_name}%",
            "limit": limit,
        },
    ).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


def retrieve_document_chunks(db: Session, *, question: str, compound: dict[str, Any] | None, limit: int) -> list[dict[str, Any]]:
    embedding = vector_literal(embed_query(question))
    compound_id = compound.get("compound_id") if compound else None
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
                record.record_kinds,
                'auto_validated'::text AS curation_status,
                (1 - (record.embedding <=> CAST(:embedding AS vector)))::float AS score
            FROM eligible record
            LEFT JOIN source_registry source ON source.source_id = record.source_id
            WHERE (
                CAST(:compound_id AS text) IS NULL
                OR record.compound_id::text = CAST(:compound_id AS text)
            )
            ORDER BY record.embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
            """
        ),
        {"embedding": embedding, "compound_id": compound_id, "limit": limit},
    ).mappings().all()
    return [serialize_row(dict(row)) for row in rows]


def retrieve_graph_relationships(db: Session, *, compound: dict[str, Any] | None, limit: int = 10) -> list[dict[str, Any]]:
    if compound is None:
        return []
    compound_id = compound.get("compound_id")
    if not compound_id:
        return []
    return graph_relationship_rows(db, compound_id=compound_id, limit=limit)


def retrieve_for_question(
    db: Session,
    *,
    question: str,
    compound: str | None = None,
    evidence_limit: int = 8,
    chunk_limit: int = 5,
    model_limit: int = 5,
) -> RetrievalResult:
    matched_compound = infer_compound(db, question, compound)
    structured = retrieve_structured_evidence(db, question=question, compound=matched_compound, limit=evidence_limit)
    models = retrieve_model_records(db, question=question, compound=matched_compound, limit=model_limit)
    notes = []
    try:
        chunks = retrieve_document_chunks(db, question=question, compound=matched_compound, limit=chunk_limit) if chunk_limit else []
    except EmbeddingServiceError as exc:
        chunks = []
        notes.append(f"Biomedical RAG retrieval unavailable: {exc}")
    relationships = retrieve_graph_relationships(db, compound=matched_compound, limit=10)
    if matched_compound is None:
        notes.append("No normalized compound identity was confidently matched.")
    if not structured:
        notes.append("No structured evidence matched the question filters.")
    if not models:
        notes.append("No public model registry records matched the inferred compound/question.")
    if not relationships:
        notes.append("No enzyme/transporter graph relationships matched the inferred compound.")
    return RetrievalResult(
        compound=matched_compound,
        structured_evidence=structured,
        model_records=models,
        document_chunks=chunks,
        graph_relationships=relationships,
        retrieval_notes=notes,
    )
