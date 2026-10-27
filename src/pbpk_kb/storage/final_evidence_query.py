from __future__ import annotations


FINAL_EVIDENCE_LAYERS = {
    "reviewable",
    "auto_validated",
    "accepted",
    "modeling",
}


def final_evidence_sql(evidence_layer: str = "auto_validated") -> str:
    if evidence_layer not in FINAL_EVIDENCE_LAYERS:
        raise ValueError(f"Unsupported final evidence layer: {evidence_layer}")
    view_name = f"{evidence_layer}_evidence"
    return f"""
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
            source.source_name,
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
            e.evidence_layer::text AS evidence_layer
        FROM {view_name} e
        LEFT JOIN parsed_document parsed ON parsed.parsed_document_id = e.parsed_document_id
        LEFT JOIN source_registry source ON source.source_id = e.source_id
    """


def reviewable_evidence_sql() -> str:
    return final_evidence_sql("reviewable")


def serving_evidence_sql() -> str:
    return final_evidence_sql("auto_validated")
