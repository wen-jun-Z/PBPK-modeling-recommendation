CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    CREATE TYPE ingestion_status AS ENUM (
        'queued',
        'running',
        'succeeded',
        'failed',
        'partially_succeeded'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE TYPE curation_status AS ENUM (
        'machine_extracted',
        'needs_review',
        'reviewed_accepted',
        'reviewed_rejected',
        'approved_for_modeling',
        'deprecated',
        'duplicate'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

ALTER TYPE curation_status ADD VALUE IF NOT EXISTS 'superseded';
ALTER TYPE curation_status ADD VALUE IF NOT EXISTS 'auto_validated';

CREATE TABLE IF NOT EXISTS source_registry (
    source_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_family TEXT NOT NULL,
    access_url TEXT NOT NULL,
    api_url TEXT,
    access_method TEXT NOT NULL,
    expected_format TEXT[] NOT NULL DEFAULT '{}',
    refresh_frequency TEXT NOT NULL,
    license_notes TEXT NOT NULL,
    connector TEXT NOT NULL,
    parser TEXT,
    entities TEXT[] NOT NULL DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    source_version TEXT,
    checksum TEXT,
    owner_curator TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingestion_run (
    ingestion_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_run_id TEXT UNIQUE,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    status ingestion_status NOT NULL DEFAULT 'queued',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    source_version TEXT,
    raw_record_count INTEGER NOT NULL DEFAULT 0 CHECK (raw_record_count >= 0),
    parsed_record_count INTEGER NOT NULL DEFAULT 0 CHECK (parsed_record_count >= 0),
    evidence_record_count INTEGER NOT NULL DEFAULT 0 CHECK (evidence_record_count >= 0),
    raw_checksum TEXT,
    log_path TEXT,
    error_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw_document (
    raw_document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    ingestion_run_id UUID REFERENCES ingestion_run(ingestion_run_id) ON DELETE SET NULL,
    source_url TEXT NOT NULL,
    source_record_id TEXT,
    source_version TEXT,
    download_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_storage_path TEXT NOT NULL,
    raw_mime_type TEXT,
    raw_checksum TEXT NOT NULL,
    raw_size_bytes BIGINT CHECK (raw_size_bytes IS NULL OR raw_size_bytes >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS parsed_document (
    parsed_document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_document_id UUID NOT NULL REFERENCES raw_document(raw_document_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    document_type TEXT,
    title TEXT,
    abstract TEXT,
    publication_date DATE,
    source_location TEXT NOT NULL,
    parsed_storage_path TEXT,
    parsed_record JSONB NOT NULL DEFAULT '{}'::jsonb,
    parsed_checksum TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_chunk (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    document_id UUID NOT NULL REFERENCES parsed_document(parsed_document_id) ON DELETE CASCADE,
    section_title TEXT,
    page_number INTEGER CHECK (page_number IS NULL OR page_number >= 0),
    paragraph_number INTEGER CHECK (paragraph_number IS NULL OR paragraph_number >= 0),
    text TEXT NOT NULL,
    token_count INTEGER CHECK (token_count IS NULL OR token_count >= 0),
    embedding vector(1536),
    embedding_model TEXT,
    parser_version TEXT NOT NULL,
    source_location TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS compound_identity (
    compound_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    preferred_name CITEXT NOT NULL,
    normalized_name CITEXT,
    synonyms TEXT[] NOT NULL DEFAULT '{}',
    pubchem_cid TEXT,
    chembl_id TEXT,
    drugcentral_id TEXT,
    rxnorm_rxcui TEXT,
    inchi_key TEXT,
    cas_number TEXT,
    approval_status TEXT,
    source_id TEXT REFERENCES source_registry(source_id),
    source_location TEXT,
    confidence_score NUMERIC(4,3) CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS physicochemical_property_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    property_name TEXT NOT NULL,
    property_value NUMERIC,
    property_value_text TEXT,
    unit TEXT,
    canonical_value NUMERIC,
    canonical_unit TEXT,
    unit_dimension TEXT,
    method TEXT,
    ph NUMERIC,
    temperature NUMERIC,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    raw_document_id UUID REFERENCES raw_document(raw_document_id) ON DELETE SET NULL,
    parsed_document_id UUID REFERENCES parsed_document(parsed_document_id) ON DELETE SET NULL,
    source_location TEXT NOT NULL,
    evidence_text TEXT,
    value_origin TEXT NOT NULL DEFAULT 'unknown',
    evidence_origin TEXT NOT NULL DEFAULT 'unknown',
    extraction_method TEXT NOT NULL DEFAULT 'machine_extracted',
    extractor_version TEXT,
    confidence_score NUMERIC(4,3) CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS adme_parameter_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    parameter_name TEXT NOT NULL,
    value NUMERIC,
    value_text TEXT,
    unit TEXT,
    canonical_value NUMERIC,
    canonical_unit TEXT,
    unit_dimension TEXT,
    matrix TEXT,
    species TEXT,
    system TEXT,
    assay_method TEXT,
    enzyme TEXT,
    transporter TEXT,
    measured_or_predicted TEXT,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    raw_document_id UUID REFERENCES raw_document(raw_document_id) ON DELETE SET NULL,
    parsed_document_id UUID REFERENCES parsed_document(parsed_document_id) ON DELETE SET NULL,
    source_location TEXT NOT NULL,
    evidence_text TEXT,
    value_origin TEXT NOT NULL DEFAULT 'unknown',
    evidence_origin TEXT NOT NULL DEFAULT 'unknown',
    extraction_method TEXT NOT NULL DEFAULT 'machine_extracted',
    extractor_version TEXT,
    confidence_score NUMERIC(4,3) CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS clinical_pk_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    study_context TEXT,
    population TEXT,
    dose TEXT,
    route TEXT,
    regimen TEXT,
    matrix TEXT,
    analyte TEXT,
    pk_parameter TEXT NOT NULL,
    value NUMERIC,
    value_text TEXT,
    unit TEXT,
    canonical_value NUMERIC,
    canonical_unit TEXT,
    unit_dimension TEXT,
    variability TEXT,
    food_effect_flag BOOLEAN,
    renal_impairment_flag BOOLEAN,
    hepatic_impairment_flag BOOLEAN,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    raw_document_id UUID REFERENCES raw_document(raw_document_id) ON DELETE SET NULL,
    parsed_document_id UUID REFERENCES parsed_document(parsed_document_id) ON DELETE SET NULL,
    source_location TEXT NOT NULL,
    evidence_text TEXT,
    value_origin TEXT NOT NULL DEFAULT 'unknown',
    evidence_origin TEXT NOT NULL DEFAULT 'unknown',
    extraction_method TEXT NOT NULL DEFAULT 'machine_extracted',
    extractor_version TEXT,
    confidence_score NUMERIC(4,3) CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ddi_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    victim_compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    perpetrator_compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    enzyme_or_transporter TEXT,
    interaction_type TEXT NOT NULL,
    study_type TEXT,
    auc_ratio NUMERIC,
    cmax_ratio NUMERIC,
    dose_victim TEXT,
    dose_perpetrator TEXT,
    population TEXT,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    raw_document_id UUID REFERENCES raw_document(raw_document_id) ON DELETE SET NULL,
    parsed_document_id UUID REFERENCES parsed_document(parsed_document_id) ON DELETE SET NULL,
    source_location TEXT NOT NULL,
    evidence_text TEXT,
    value_origin TEXT NOT NULL DEFAULT 'unknown',
    evidence_origin TEXT NOT NULL DEFAULT 'unknown',
    extraction_method TEXT NOT NULL DEFAULT 'machine_extracted',
    extractor_version TEXT,
    confidence_score NUMERIC(4,3) CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS special_population_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    population_type TEXT NOT NULL,
    population_definition TEXT,
    covariate TEXT,
    pk_parameter TEXT,
    value NUMERIC,
    value_text TEXT,
    unit TEXT,
    canonical_value NUMERIC,
    canonical_unit TEXT,
    unit_dimension TEXT,
    ratio_to_reference NUMERIC,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    raw_document_id UUID REFERENCES raw_document(raw_document_id) ON DELETE SET NULL,
    parsed_document_id UUID REFERENCES parsed_document(parsed_document_id) ON DELETE SET NULL,
    source_location TEXT NOT NULL,
    evidence_text TEXT,
    value_origin TEXT NOT NULL DEFAULT 'unknown',
    evidence_origin TEXT NOT NULL DEFAULT 'unknown',
    extraction_method TEXT NOT NULL DEFAULT 'machine_extracted',
    extractor_version TEXT,
    confidence_score NUMERIC(4,3) CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS protein_target_identity (
    protein_target_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    preferred_name TEXT NOT NULL,
    gene_symbol TEXT,
    uniprot_id TEXT,
    target_type TEXT,
    synonyms TEXT[] NOT NULL DEFAULT '{}',
    source_id TEXT REFERENCES source_registry(source_id),
    source_location TEXT,
    confidence_score NUMERIC(4,3) CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tissue_expression_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    protein_target_id UUID REFERENCES protein_target_identity(protein_target_id) ON DELETE SET NULL,
    tissue TEXT NOT NULL,
    expression_value NUMERIC,
    expression_value_text TEXT,
    unit TEXT,
    canonical_value NUMERIC,
    canonical_unit TEXT,
    unit_dimension TEXT,
    species TEXT,
    assay_method TEXT,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    raw_document_id UUID REFERENCES raw_document(raw_document_id) ON DELETE SET NULL,
    parsed_document_id UUID REFERENCES parsed_document(parsed_document_id) ON DELETE SET NULL,
    source_location TEXT NOT NULL,
    evidence_text TEXT,
    value_origin TEXT NOT NULL DEFAULT 'unknown',
    evidence_origin TEXT NOT NULL DEFAULT 'unknown',
    extraction_method TEXT NOT NULL DEFAULT 'machine_extracted',
    extractor_version TEXT,
    confidence_score NUMERIC(4,3) CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS processed_evidence_candidate (
    evidence_id UUID PRIMARY KEY,
    run_id TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    compound_name CITEXT,
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    raw_asset_id TEXT NOT NULL,
    raw_document_id UUID REFERENCES raw_document(raw_document_id) ON DELETE SET NULL,
    parsed_document_id UUID REFERENCES parsed_document(parsed_document_id) ON DELETE SET NULL,
    parsed_table_id UUID,
    chunk_id UUID,
    storage_path TEXT NOT NULL,
    source_url TEXT,
    checksum TEXT NOT NULL,
    source_location TEXT NOT NULL,
    source_text TEXT NOT NULL,
    source_text_hash TEXT,
    extraction_method TEXT NOT NULL,
    extractor_name TEXT,
    extractor_version TEXT,
    extraction_confidence NUMERIC(4,3) CHECK (extraction_confidence IS NULL OR extraction_confidence BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'needs_review',
    parameter TEXT,
    value NUMERIC,
    value_text TEXT,
    unit TEXT,
    canonical_value NUMERIC,
    canonical_unit TEXT,
    condition TEXT,
    population TEXT,
    route TEXT,
    dose TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_record JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evidence_compound_attribution (
    attribution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_id UUID NOT NULL REFERENCES processed_evidence_candidate(evidence_id) ON DELETE CASCADE,
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    compound_name TEXT,
    compound_role TEXT NOT NULL CHECK (
        compound_role IN (
            'target_compound',
            'victim_drug',
            'perpetrator_drug',
            'co_medication',
            'comparator',
            'excipient',
            'mentioned_only',
            'shared_or_unlinked'
        )
    ),
    attribution_method TEXT NOT NULL,
    attribution_confidence NUMERIC(4,3) CHECK (attribution_confidence IS NULL OR attribution_confidence BETWEEN 0 AND 1),
    matched_text TEXT,
    matched_synonym TEXT,
    attribution_reason TEXT,
    llm_required BOOLEAN NOT NULL DEFAULT FALSE,
    llm_trigger_reason TEXT,
    attribution_stage TEXT NOT NULL DEFAULT 'rule' CHECK (attribution_stage IN ('rule', 'llm_assisted')),
    llm_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(evidence_id, compound_id, compound_role)
);

CREATE TABLE IF NOT EXISTS evidence_duplicate_group (
    duplicate_group_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_type TEXT NOT NULL CHECK (group_type IN ('evidence_id', 'source', 'content', 'semantic', 'similar_for_review')),
    group_key TEXT NOT NULL,
    evidence_type TEXT,
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    compound_name TEXT,
    representative_evidence_id UUID REFERENCES processed_evidence_candidate(evidence_id) ON DELETE SET NULL,
    duplicate_count INTEGER NOT NULL CHECK (duplicate_count >= 2),
    selection_reason TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(group_type, group_key)
);

CREATE TABLE IF NOT EXISTS evidence_duplicate_member (
    duplicate_group_id UUID NOT NULL REFERENCES evidence_duplicate_group(duplicate_group_id) ON DELETE CASCADE,
    evidence_id UUID NOT NULL REFERENCES processed_evidence_candidate(evidence_id) ON DELETE CASCADE,
    is_representative BOOLEAN NOT NULL DEFAULT FALSE,
    duplicate_rank INTEGER,
    duplicate_reason TEXT NOT NULL,
    similarity_score NUMERIC(6,5),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (duplicate_group_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS evidence_layer_membership (
    evidence_id UUID NOT NULL REFERENCES processed_evidence_candidate(evidence_id) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    evidence_layer TEXT NOT NULL CHECK (
        evidence_layer IN (
            'candidate',
            'excluded',
            'preselected',
            'reviewable',
            'auto_validated',
            'accepted',
            'modeling'
        )
    ),
    exclusion_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    review_blockers JSONB NOT NULL DEFAULT '[]'::jsonb,
    semantic_group_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    conflict_group_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (evidence_id, evidence_layer)
);

ALTER TABLE evidence_layer_membership
    DROP CONSTRAINT IF EXISTS evidence_layer_membership_evidence_layer_check;
ALTER TABLE evidence_layer_membership
    ADD CONSTRAINT evidence_layer_membership_evidence_layer_check CHECK (
        evidence_layer IN (
            'candidate',
            'excluded',
            'preselected',
            'reviewable',
            'auto_validated',
            'accepted',
            'modeling'
        )
    );

CREATE TABLE IF NOT EXISTS evidence_conflict_group (
    conflict_group_id UUID PRIMARY KEY,
    run_id TEXT NOT NULL,
    source_group_id TEXT,
    source_group_type TEXT,
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    compound_name TEXT,
    evidence_type TEXT NOT NULL,
    parameter TEXT,
    canonical_unit TEXT,
    evidence_count INTEGER NOT NULL CHECK (evidence_count >= 2),
    analysis_status TEXT NOT NULL,
    analysis_reason TEXT NOT NULL,
    resolution TEXT NOT NULL,
    requires_human_review BOOLEAN NOT NULL DEFAULT TRUE,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evidence_conflict_member (
    conflict_group_id UUID NOT NULL REFERENCES evidence_conflict_group(conflict_group_id) ON DELETE CASCADE,
    evidence_id UUID NOT NULL REFERENCES processed_evidence_candidate(evidence_id) ON DELETE CASCADE,
    analysis_status TEXT NOT NULL,
    requires_human_review BOOLEAN NOT NULL DEFAULT TRUE,
    source_group_id TEXT,
    source_group_type TEXT,
    source_id TEXT,
    source_location TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (conflict_group_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS rag_retrieval_record (
    retrieval_record_id UUID PRIMARY KEY,
    run_id TEXT NOT NULL,
    embedding_row INTEGER NOT NULL CHECK (embedding_row >= 0),
    compound_id UUID NOT NULL REFERENCES compound_identity(compound_id) ON DELETE CASCADE,
    compound_name TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    source_location TEXT NOT NULL,
    source_url TEXT,
    storage_path TEXT,
    checksum TEXT,
    source_text TEXT NOT NULL,
    source_text_hash TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    embedding_model TEXT NOT NULL,
    evidence_layer TEXT NOT NULL,
    record_kinds JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    parsed_table_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    recovered_table_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    provenance_status TEXT NOT NULL,
    curation_status TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, embedding_row)
);

CREATE TABLE IF NOT EXISTS recommendation_embedding_record (
    embedding_record_id TEXT PRIMARY KEY,
    kg_run_id TEXT NOT NULL,
    object_kind TEXT NOT NULL CHECK (
        object_kind IN ('node', 'edge', 'evidence', 'subgraph')
    ),
    object_id TEXT NOT NULL,
    hop SMALLINT NOT NULL DEFAULT -1 CHECK (hop BETWEEN -1 AND 2),
    embedding_row INTEGER NOT NULL CHECK (embedding_row >= 0),
    embedding_text TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (input_sha256 ~ '^[a-f0-9]{64}$'),
    graph_manifest_sha256 TEXT NOT NULL CHECK (graph_manifest_sha256 ~ '^[a-f0-9]{64}$'),
    embedding vector(768) NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_revision TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension = 768),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kg_run_id, embedding_model, embedding_revision, object_kind, object_id, hop),
    UNIQUE (kg_run_id, embedding_model, embedding_revision, object_kind, hop, embedding_row)
);

CREATE TABLE IF NOT EXISTS recommendation_lexical_record (
    lexical_record_id TEXT PRIMARY KEY,
    kg_run_id TEXT NOT NULL,
    object_kind TEXT NOT NULL CHECK (
        object_kind IN ('node', 'edge', 'evidence', 'subgraph')
    ),
    object_id TEXT NOT NULL,
    hop SMALLINT NOT NULL DEFAULT -1 CHECK (hop BETWEEN -1 AND 2),
    lexical_row INTEGER NOT NULL CHECK (lexical_row >= 0),
    canonical_text TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (input_sha256 ~ '^[a-f0-9]{64}$'),
    graph_manifest_sha256 TEXT NOT NULL CHECK (graph_manifest_sha256 ~ '^[a-f0-9]{64}$'),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english'::regconfig, canonical_text)
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kg_run_id, object_kind, object_id, hop),
    UNIQUE (kg_run_id, object_kind, hop, lexical_row)
);

CREATE TABLE IF NOT EXISTS recommendation_gnn_embedding_record (
    gnn_embedding_record_id TEXT PRIMARY KEY,
    kg_run_id TEXT NOT NULL,
    subgraph_id TEXT NOT NULL,
    center_id TEXT NOT NULL,
    hop SMALLINT NOT NULL CHECK (hop BETWEEN 0 AND 2),
    embedding_row INTEGER NOT NULL CHECK (embedding_row >= 0),
    subgraph_content_sha256 TEXT NOT NULL CHECK (
        subgraph_content_sha256 ~ '^[a-f0-9]{64}$'
    ),
    graph_manifest_sha256 TEXT NOT NULL CHECK (
        graph_manifest_sha256 ~ '^[a-f0-9]{64}$'
    ),
    checkpoint_sha256 TEXT NOT NULL CHECK (
        checkpoint_sha256 ~ '^[a-f0-9]{64}$'
    ),
    embedding vector(256) NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_revision TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension = 256),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (
        kg_run_id, embedding_model, embedding_revision, subgraph_id, hop
    ),
    UNIQUE (
        kg_run_id, embedding_model, embedding_revision, hop, embedding_row
    )
);

CREATE TABLE IF NOT EXISTS recommendation_model_run (
    recommendation_model_run_id TEXT PRIMARY KEY,
    gate5_manifest_sha256 TEXT NOT NULL UNIQUE CHECK (
        gate5_manifest_sha256 ~ '^[a-f0-9]{64}$'
    ),
    gate6_manifest_sha256 TEXT NOT NULL UNIQUE CHECK (
        gate6_manifest_sha256 ~ '^[a-f0-9]{64}$'
    ),
    kg_run_id TEXT NOT NULL,
    evidence_run_id TEXT NOT NULL,
    llm_model TEXT NOT NULL,
    llm_revision TEXT NOT NULL,
    llm_dtype TEXT NOT NULL,
    projector_checkpoint_sha256 TEXT NOT NULL CHECK (
        projector_checkpoint_sha256 ~ '^[a-f0-9]{64}$'
    ),
    benchmark_level TEXT NOT NULL CHECK (benchmark_level = 'silver'),
    engineering_acceptance BOOLEAN NOT NULL,
    generalization_claim_eligible BOOLEAN NOT NULL,
    gold_claim BOOLEAN NOT NULL DEFAULT FALSE,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recommendation_result (
    recommendation_result_id TEXT PRIMARY KEY,
    recommendation_model_run_id TEXT NOT NULL REFERENCES recommendation_model_run(
        recommendation_model_run_id
    ) ON DELETE RESTRICT,
    example_id TEXT NOT NULL,
    compound_node_id TEXT NOT NULL,
    compound_name TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status = 'recommendation_validated'),
    request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[a-f0-9]{64}$'),
    recommendation_sha256 TEXT NOT NULL CHECK (
        recommendation_sha256 ~ '^[a-f0-9]{64}$'
    ),
    overall_confidence DOUBLE PRECISION NOT NULL CHECK (
        overall_confidence BETWEEN 0 AND 1
    ),
    citation_count INTEGER NOT NULL CHECK (citation_count >= 0),
    evidence_gap_count INTEGER NOT NULL CHECK (evidence_gap_count >= 0),
    request_json JSONB NOT NULL,
    recommendation_json JSONB NOT NULL,
    validation_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (recommendation_model_run_id, example_id),
    UNIQUE (recommendation_model_run_id, request_sha256)
);

CREATE OR REPLACE FUNCTION source_priority(source_id TEXT)
RETURNS INTEGER
LANGUAGE SQL
IMMUTABLE
AS $$
    SELECT CASE
        WHEN source_id IN ('dailymed', 'openfda_drug_label', 'drugs_fda', 'ema_epar') THEN 1
        WHEN source_id IN ('pkdb', 'clinicaltrials', 'pubchem', 'rxnorm', 'chembl', 'drugcentral') THEN 2
        WHEN source_id IN ('pubmed', 'pmc_oa') THEN 3
        WHEN source_id IN ('osp_pbpk_model_library', 'osp_tools_docs') THEN 4
        ELSE 5
    END
$$;

CREATE TABLE IF NOT EXISTS enzyme_transporter_relationship (
    relationship_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    protein_target_id UUID REFERENCES protein_target_identity(protein_target_id) ON DELETE SET NULL,
    relationship_type TEXT NOT NULL,
    evidence_direction TEXT,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    raw_document_id UUID REFERENCES raw_document(raw_document_id) ON DELETE SET NULL,
    parsed_document_id UUID REFERENCES parsed_document(parsed_document_id) ON DELETE SET NULL,
    processed_evidence_id UUID REFERENCES processed_evidence_candidate(evidence_id) ON DELETE CASCADE,
    source_location TEXT NOT NULL,
    evidence_text TEXT,
    value_origin TEXT NOT NULL DEFAULT 'unknown',
    evidence_origin TEXT NOT NULL DEFAULT 'unknown',
    extraction_method TEXT NOT NULL DEFAULT 'machine_extracted',
    extractor_version TEXT,
    confidence_score NUMERIC(4,3) CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE IF EXISTS enzyme_transporter_relationship
    ADD COLUMN IF NOT EXISTS processed_evidence_id UUID REFERENCES processed_evidence_candidate(evidence_id) ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS public_pbpk_model_registry (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_model_id TEXT,
    model_name TEXT NOT NULL,
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    model_type TEXT,
    software TEXT,
    species TEXT,
    population TEXT,
    route TEXT,
    model_source TEXT,
    source_id TEXT REFERENCES source_registry(source_id),
    source_variant_id TEXT,
    repository_url TEXT,
    repository_commit TEXT,
    release_tag TEXT,
    license_spdx_id TEXT,
    model_file_path TEXT,
    parameter_file_path TEXT,
    evaluation_report_path TEXT,
    publication_id TEXT,
    validation_status TEXT,
    validation_grade TEXT,
    run_label_grade SMALLINT NOT NULL DEFAULT 0 CHECK (
        run_label_grade BETWEEN 0 AND 3
    ),
    ranking_label TEXT,
    reusable_candidate BOOLEAN NOT NULL DEFAULT FALSE,
    model_checksum TEXT,
    source_snapshot_checksum TEXT,
    static_parse_status TEXT,
    execution_probe_status TEXT,
    latest_validation_run_id TEXT,
    simulation_count INTEGER NOT NULL DEFAULT 0 CHECK (simulation_count >= 0),
    parameter_record_count BIGINT NOT NULL DEFAULT 0 CHECK (
        parameter_record_count >= 0
    ),
    component_record_count BIGINT NOT NULL DEFAULT 0 CHECK (
        component_record_count >= 0
    ),
    process_record_count BIGINT NOT NULL DEFAULT 0 CHECK (
        process_record_count >= 0
    ),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    source_location TEXT,
    notes TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public_pbpk_model_version (
    model_version_id UUID PRIMARY KEY,
    model_id UUID NOT NULL REFERENCES public_pbpk_model_registry(model_id) ON DELETE CASCADE,
    external_model_version_id TEXT NOT NULL UNIQUE,
    version_label TEXT,
    repository_url TEXT NOT NULL,
    repository_commit TEXT NOT NULL,
    project_relative_path TEXT NOT NULL,
    project_sha256 TEXT NOT NULL CHECK (project_sha256 ~ '^[a-f0-9]{64}$'),
    source_snapshot_sha256 TEXT NOT NULL CHECK (
        source_snapshot_sha256 ~ '^[a-f0-9]{64}$'
    ),
    software TEXT NOT NULL,
    software_version TEXT,
    license_status TEXT NOT NULL,
    license_spdx_id TEXT,
    license_source_location TEXT,
    license_source_checksum TEXT CHECK (
        license_source_checksum ~ '^[a-f0-9]{64}$'
    ),
    static_parse_status TEXT NOT NULL,
    execution_probe_status TEXT NOT NULL,
    pkml_parse_status TEXT NOT NULL,
    reusable_candidate BOOLEAN NOT NULL DEFAULT FALSE,
    compounds TEXT[] NOT NULL DEFAULT '{}',
    species TEXT[] NOT NULL DEFAULT '{}',
    populations TEXT[] NOT NULL DEFAULT '{}',
    routes TEXT[] NOT NULL DEFAULT '{}',
    formulations TEXT[] NOT NULL DEFAULT '{}',
    enzymes TEXT[] NOT NULL DEFAULT '{}',
    transporters TEXT[] NOT NULL DEFAULT '{}',
    compartments TEXT[] NOT NULL DEFAULT '{}',
    process_types TEXT[] NOT NULL DEFAULT '{}',
    output_names TEXT[] NOT NULL DEFAULT '{}',
    simulation_count INTEGER NOT NULL CHECK (simulation_count >= 0),
    parameter_record_count BIGINT NOT NULL CHECK (parameter_record_count >= 0),
    component_record_count BIGINT NOT NULL CHECK (component_record_count >= 0),
    process_record_count BIGINT NOT NULL CHECK (process_record_count >= 0),
    report_count INTEGER NOT NULL DEFAULT 0 CHECK (report_count >= 0),
    dataset_reference_count INTEGER NOT NULL DEFAULT 0 CHECK (
        dataset_reference_count >= 0
    ),
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    source_variant_id TEXT NOT NULL,
    source_location TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public_pbpk_model_asset (
    model_asset_id UUID PRIMARY KEY,
    model_version_id UUID NOT NULL REFERENCES public_pbpk_model_version(model_version_id) ON DELETE CASCADE,
    asset_kind TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    source_location TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_version_id, asset_kind, relative_path, sha256)
);

CREATE TABLE IF NOT EXISTS public_pbpk_model_simulation (
    model_simulation_id UUID PRIMARY KEY,
    model_version_id UUID NOT NULL REFERENCES public_pbpk_model_version(model_version_id) ON DELETE CASCADE,
    external_simulation_id TEXT NOT NULL,
    protocol_sequence INTEGER NOT NULL CHECK (protocol_sequence > 0),
    simulation_name TEXT NOT NULL,
    protocol_role TEXT NOT NULL,
    solver TEXT,
    species TEXT[] NOT NULL DEFAULT '{}',
    populations TEXT[] NOT NULL DEFAULT '{}',
    routes TEXT[] NOT NULL DEFAULT '{}',
    formulations TEXT[] NOT NULL DEFAULT '{}',
    validation_dataset_name TEXT,
    pkml_relative_path TEXT NOT NULL,
    pkml_sha256 TEXT NOT NULL CHECK (pkml_sha256 ~ '^[a-f0-9]{64}$'),
    result_relative_path TEXT,
    result_sha256 TEXT CHECK (
        result_sha256 IS NULL OR result_sha256 ~ '^[a-f0-9]{64}$'
    ),
    parameter_count INTEGER NOT NULL DEFAULT 0 CHECK (parameter_count >= 0),
    component_count INTEGER NOT NULL DEFAULT 0 CHECK (component_count >= 0),
    process_count INTEGER NOT NULL DEFAULT 0 CHECK (process_count >= 0),
    administration_count INTEGER NOT NULL DEFAULT 0 CHECK (
        administration_count >= 0
    ),
    source_location TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_version_id, external_simulation_id),
    UNIQUE (model_version_id, protocol_sequence)
);

CREATE TABLE IF NOT EXISTS public_pbpk_model_validation_run (
    validation_run_id UUID PRIMARY KEY,
    external_validation_run_id TEXT NOT NULL UNIQUE,
    model_version_id UUID NOT NULL REFERENCES public_pbpk_model_version(model_version_id) ON DELETE CASCADE,
    protocol_id TEXT NOT NULL,
    protocol_path TEXT NOT NULL,
    protocol_sha256 TEXT NOT NULL CHECK (protocol_sha256 ~ '^[a-f0-9]{64}$'),
    windows_manifest_path TEXT NOT NULL,
    windows_manifest_sha256 TEXT NOT NULL CHECK (
        windows_manifest_sha256 ~ '^[a-f0-9]{64}$'
    ),
    audit_report_path TEXT NOT NULL,
    audit_report_sha256 TEXT NOT NULL CHECK (
        audit_report_sha256 ~ '^[a-f0-9]{64}$'
    ),
    validation_dataset_metrics_path TEXT NOT NULL,
    source_snapshot_sha256 TEXT NOT NULL CHECK (
        source_snapshot_sha256 ~ '^[a-f0-9]{64}$'
    ),
    validation_grade TEXT NOT NULL,
    run_label_grade SMALLINT NOT NULL CHECK (run_label_grade BETWEEN 0 AND 3),
    ranking_label TEXT NOT NULL,
    reusable_candidate BOOLEAN NOT NULL DEFAULT FALSE,
    engineering_accepted BOOLEAN NOT NULL DEFAULT FALSE,
    scientific_accepted BOOLEAN NOT NULL DEFAULT FALSE,
    verification_dataset_count INTEGER NOT NULL CHECK (
        verification_dataset_count >= 0
    ),
    pooled_observation_count INTEGER NOT NULL CHECK (
        pooled_observation_count >= 0
    ),
    pooled_gmfe DOUBLE PRECISION,
    pooled_geometric_bias DOUBLE PRECISION,
    pooled_two_fold_fraction DOUBLE PRECISION CHECK (
        pooled_two_fold_fraction IS NULL
        OR pooled_two_fold_fraction BETWEEN 0 AND 1
    ),
    endpoint_scenario_pass_fraction DOUBLE PRECISION CHECK (
        endpoint_scenario_pass_fraction IS NULL
        OR endpoint_scenario_pass_fraction BETWEEN 0 AND 1
    ),
    validated_at TIMESTAMPTZ NOT NULL,
    source_location TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public_pbpk_model_validation_dataset (
    validation_dataset_id UUID PRIMARY KEY,
    validation_run_id UUID NOT NULL REFERENCES public_pbpk_model_validation_run(validation_run_id) ON DELETE CASCADE,
    simulation_sequence INTEGER NOT NULL CHECK (simulation_sequence > 0),
    dataset_name TEXT NOT NULL,
    simulation_name TEXT NOT NULL,
    reference TEXT NOT NULL,
    source_location TEXT NOT NULL,
    prediction_unit TEXT,
    observation_count INTEGER NOT NULL CHECK (observation_count >= 0),
    positive_pair_count INTEGER NOT NULL CHECK (positive_pair_count >= 0),
    excluded_nonpositive_observation_count INTEGER NOT NULL DEFAULT 0 CHECK (
        excluded_nonpositive_observation_count >= 0
    ),
    auc0_last_ratio DOUBLE PRECISION,
    cmax_ratio DOUBLE PRECISION,
    gmfe DOUBLE PRECISION,
    geometric_bias DOUBLE PRECISION,
    log_rmse DOUBLE PRECISION,
    two_fold_fraction DOUBLE PRECISION CHECK (
        two_fold_fraction IS NULL OR two_fold_fraction BETWEEN 0 AND 1
    ),
    dataset_guard_passed BOOLEAN NOT NULL,
    endpoint_evaluable BOOLEAN NOT NULL,
    endpoint_passed BOOLEAN NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (validation_run_id, simulation_sequence, dataset_name)
);

CREATE TABLE IF NOT EXISTS public_pbpk_model_validation_metric (
    validation_metric_id UUID PRIMARY KEY,
    validation_run_id UUID NOT NULL REFERENCES public_pbpk_model_validation_run(validation_run_id) ON DELETE CASCADE,
    validation_dataset_id UUID REFERENCES public_pbpk_model_validation_dataset(validation_dataset_id) ON DELETE CASCADE,
    metric_scope TEXT NOT NULL CHECK (metric_scope IN ('pooled', 'dataset')),
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    passed BOOLEAN,
    source_location TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_parameter_evidence (
    model_parameter_evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES public_pbpk_model_registry(model_id) ON DELETE CASCADE,
    compound_id UUID REFERENCES compound_identity(compound_id) ON DELETE SET NULL,
    parameter_name TEXT NOT NULL,
    parameter_value NUMERIC,
    parameter_value_text TEXT,
    unit TEXT,
    canonical_value NUMERIC,
    canonical_unit TEXT,
    unit_dimension TEXT,
    parameter_context TEXT,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    raw_document_id UUID REFERENCES raw_document(raw_document_id) ON DELETE SET NULL,
    parsed_document_id UUID REFERENCES parsed_document(parsed_document_id) ON DELETE SET NULL,
    source_location TEXT NOT NULL,
    evidence_text TEXT,
    extraction_method TEXT NOT NULL DEFAULT 'machine_extracted',
    extractor_version TEXT,
    confidence_score NUMERIC(4,3) CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    curation_status curation_status NOT NULL DEFAULT 'machine_extracted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS curation_event (
    curation_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_table TEXT NOT NULL,
    evidence_id UUID NOT NULL,
    previous_status curation_status,
    new_status curation_status NOT NULL,
    curator TEXT NOT NULL,
    curation_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    curator_comment TEXT,
    field_changes JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS export_job (
    export_job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    export_type TEXT NOT NULL,
    filters JSONB NOT NULL DEFAULT '{}'::jsonb,
    requested_by TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    output_path TEXT,
    source_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    included_curation_statuses curation_status[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS user_activity_log (
    user_activity_log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_identifier TEXT,
    activity_type TEXT NOT NULL,
    entity_type TEXT,
    entity_id UUID,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE IF EXISTS ingestion_run ADD COLUMN IF NOT EXISTS external_run_id TEXT;
DO $$
BEGIN
    ALTER TABLE ingestion_run ADD CONSTRAINT ingestion_run_external_run_id_key UNIQUE (external_run_id);
EXCEPTION
    WHEN duplicate_object THEN NULL;
    WHEN duplicate_table THEN NULL;
END $$;
ALTER TABLE IF EXISTS physicochemical_property_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS physicochemical_property_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS physicochemical_property_evidence ADD COLUMN IF NOT EXISTS canonical_value NUMERIC;
ALTER TABLE IF EXISTS physicochemical_property_evidence ADD COLUMN IF NOT EXISTS canonical_unit TEXT;
ALTER TABLE IF EXISTS physicochemical_property_evidence ADD COLUMN IF NOT EXISTS unit_dimension TEXT;
ALTER TABLE IF EXISTS adme_parameter_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS adme_parameter_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS adme_parameter_evidence ADD COLUMN IF NOT EXISTS canonical_value NUMERIC;
ALTER TABLE IF EXISTS adme_parameter_evidence ADD COLUMN IF NOT EXISTS canonical_unit TEXT;
ALTER TABLE IF EXISTS adme_parameter_evidence ADD COLUMN IF NOT EXISTS unit_dimension TEXT;
ALTER TABLE IF EXISTS clinical_pk_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS clinical_pk_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS clinical_pk_evidence ADD COLUMN IF NOT EXISTS canonical_value NUMERIC;
ALTER TABLE IF EXISTS clinical_pk_evidence ADD COLUMN IF NOT EXISTS canonical_unit TEXT;
ALTER TABLE IF EXISTS clinical_pk_evidence ADD COLUMN IF NOT EXISTS unit_dimension TEXT;
ALTER TABLE IF EXISTS ddi_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS ddi_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS special_population_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS special_population_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS special_population_evidence ADD COLUMN IF NOT EXISTS canonical_value NUMERIC;
ALTER TABLE IF EXISTS special_population_evidence ADD COLUMN IF NOT EXISTS canonical_unit TEXT;
ALTER TABLE IF EXISTS special_population_evidence ADD COLUMN IF NOT EXISTS unit_dimension TEXT;
ALTER TABLE IF EXISTS tissue_expression_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS tissue_expression_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS tissue_expression_evidence ADD COLUMN IF NOT EXISTS canonical_value NUMERIC;
ALTER TABLE IF EXISTS tissue_expression_evidence ADD COLUMN IF NOT EXISTS canonical_unit TEXT;
ALTER TABLE IF EXISTS tissue_expression_evidence ADD COLUMN IF NOT EXISTS unit_dimension TEXT;
ALTER TABLE IF EXISTS enzyme_transporter_relationship ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS enzyme_transporter_relationship ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS model_parameter_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS model_parameter_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS evidence_compound_attribution ADD COLUMN IF NOT EXISTS llm_required BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE IF EXISTS evidence_compound_attribution ADD COLUMN IF NOT EXISTS llm_trigger_reason TEXT;
ALTER TABLE IF EXISTS evidence_compound_attribution ADD COLUMN IF NOT EXISTS attribution_stage TEXT NOT NULL DEFAULT 'rule';
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'evidence_compound_attribution_stage_check'
    ) THEN
        ALTER TABLE evidence_compound_attribution
            ADD CONSTRAINT evidence_compound_attribution_stage_check
            CHECK (attribution_stage IN ('rule', 'llm_assisted'));
    END IF;
END $$;
ALTER TABLE IF EXISTS model_parameter_evidence ADD COLUMN IF NOT EXISTS canonical_value NUMERIC;
ALTER TABLE IF EXISTS model_parameter_evidence ADD COLUMN IF NOT EXISTS canonical_unit TEXT;
ALTER TABLE IF EXISTS model_parameter_evidence ADD COLUMN IF NOT EXISTS unit_dimension TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS external_model_id TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS source_variant_id TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS repository_url TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS repository_commit TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS release_tag TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS license_spdx_id TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS validation_grade TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS run_label_grade SMALLINT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS ranking_label TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS reusable_candidate BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS model_checksum TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS source_snapshot_checksum TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS static_parse_status TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS execution_probe_status TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS latest_validation_run_id TEXT;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS simulation_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS parameter_record_count BIGINT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS component_record_count BIGINT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS process_record_count BIGINT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS public_pbpk_model_registry ADD COLUMN IF NOT EXISTS metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE IF EXISTS public_pbpk_model_version
    ALTER COLUMN license_spdx_id DROP NOT NULL,
    ALTER COLUMN license_source_location DROP NOT NULL,
    ALTER COLUMN license_source_checksum DROP NOT NULL;
ALTER TABLE IF EXISTS public_pbpk_model_validation_run ADD COLUMN IF NOT EXISTS validation_dataset_metrics_path TEXT;

ALTER TABLE IF EXISTS evidence_duplicate_group DROP CONSTRAINT IF EXISTS evidence_duplicate_group_group_type_check;
ALTER TABLE IF EXISTS evidence_duplicate_group
    ADD CONSTRAINT evidence_duplicate_group_group_type_check
    CHECK (
        group_type IN (
            'evidence_id', 'source', 'content', 'semantic',
            'semantic_duplicate', 'similar_for_review', 'semantic_conflict_candidate'
        )
    );
ALTER TABLE IF EXISTS evidence_duplicate_group DROP CONSTRAINT IF EXISTS evidence_duplicate_group_group_type_group_key_key;

CREATE INDEX IF NOT EXISTS ingestion_run_source_status_idx ON ingestion_run(source_id, status);
CREATE INDEX IF NOT EXISTS raw_document_source_checksum_idx ON raw_document(source_id, raw_checksum);
CREATE INDEX IF NOT EXISTS parsed_document_source_idx ON parsed_document(source_id);
CREATE INDEX IF NOT EXISTS document_chunk_source_idx ON document_chunk(source_id);
CREATE INDEX IF NOT EXISTS document_chunk_embedding_hnsw_idx ON document_chunk USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS compound_identity_name_idx ON compound_identity(preferred_name);
CREATE INDEX IF NOT EXISTS compound_identity_pubchem_idx ON compound_identity(pubchem_cid);
CREATE INDEX IF NOT EXISTS compound_identity_chembl_idx ON compound_identity(chembl_id);
CREATE INDEX IF NOT EXISTS physicochemical_compound_idx ON physicochemical_property_evidence(compound_id);
CREATE INDEX IF NOT EXISTS adme_compound_idx ON adme_parameter_evidence(compound_id);
CREATE INDEX IF NOT EXISTS clinical_pk_compound_idx ON clinical_pk_evidence(compound_id);
CREATE INDEX IF NOT EXISTS ddi_victim_perpetrator_idx ON ddi_evidence(victim_compound_id, perpetrator_compound_id);
CREATE INDEX IF NOT EXISTS special_population_compound_idx ON special_population_evidence(compound_id);
CREATE INDEX IF NOT EXISTS tissue_expression_target_idx ON tissue_expression_evidence(protein_target_id);
CREATE INDEX IF NOT EXISTS tissue_expression_source_idx ON tissue_expression_evidence(source_id);
CREATE INDEX IF NOT EXISTS enzyme_transporter_evidence_idx ON enzyme_transporter_relationship(processed_evidence_id);
CREATE UNIQUE INDEX IF NOT EXISTS enzyme_transporter_final_fact_unique_idx
    ON enzyme_transporter_relationship(processed_evidence_id, protein_target_id, relationship_type)
    WHERE processed_evidence_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS processed_evidence_candidate_compound_idx ON processed_evidence_candidate(compound_id);
CREATE INDEX IF NOT EXISTS processed_evidence_candidate_type_idx ON processed_evidence_candidate(evidence_type);
CREATE INDEX IF NOT EXISTS processed_evidence_candidate_source_idx ON processed_evidence_candidate(source_id);
CREATE INDEX IF NOT EXISTS processed_evidence_candidate_chunk_idx ON processed_evidence_candidate(chunk_id);
CREATE INDEX IF NOT EXISTS evidence_attribution_evidence_idx ON evidence_compound_attribution(evidence_id);
CREATE INDEX IF NOT EXISTS evidence_attribution_compound_role_idx ON evidence_compound_attribution(compound_id, compound_role);
CREATE INDEX IF NOT EXISTS evidence_duplicate_group_type_idx ON evidence_duplicate_group(group_type);
CREATE INDEX IF NOT EXISTS evidence_duplicate_group_representative_idx ON evidence_duplicate_group(representative_evidence_id);
CREATE INDEX IF NOT EXISTS evidence_duplicate_member_evidence_idx ON evidence_duplicate_member(evidence_id);
CREATE INDEX IF NOT EXISTS evidence_duplicate_group_key_idx ON evidence_duplicate_group(group_type, group_key);
CREATE INDEX IF NOT EXISTS evidence_layer_membership_run_layer_idx ON evidence_layer_membership(run_id, evidence_layer);
CREATE INDEX IF NOT EXISTS evidence_layer_membership_layer_evidence_idx ON evidence_layer_membership(evidence_layer, evidence_id);
CREATE INDEX IF NOT EXISTS evidence_conflict_group_status_idx ON evidence_conflict_group(analysis_status, requires_human_review);
CREATE INDEX IF NOT EXISTS evidence_conflict_member_evidence_idx ON evidence_conflict_member(evidence_id);
CREATE INDEX IF NOT EXISTS rag_retrieval_record_source_idx ON rag_retrieval_record(source_id);
CREATE INDEX IF NOT EXISTS rag_retrieval_record_compound_idx ON rag_retrieval_record(compound_id);
CREATE INDEX IF NOT EXISTS rag_retrieval_record_embedding_hnsw_idx ON rag_retrieval_record USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS recommendation_embedding_object_idx
    ON recommendation_embedding_record(kg_run_id, object_kind, hop, object_id);
CREATE INDEX IF NOT EXISTS recommendation_embedding_model_idx
    ON recommendation_embedding_record(
        kg_run_id, embedding_model, embedding_revision, object_kind, hop
    );
CREATE INDEX IF NOT EXISTS recommendation_lexical_object_idx
    ON recommendation_lexical_record(kg_run_id, object_kind, hop, object_id);
CREATE INDEX IF NOT EXISTS recommendation_lexical_search_idx
    ON recommendation_lexical_record USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS recommendation_gnn_embedding_object_idx
    ON recommendation_gnn_embedding_record(
        kg_run_id, subgraph_id, hop
    );
CREATE INDEX IF NOT EXISTS recommendation_gnn_embedding_model_idx
    ON recommendation_gnn_embedding_record(
        kg_run_id, embedding_model, embedding_revision, hop
    );
CREATE INDEX IF NOT EXISTS recommendation_model_run_acceptance_idx
    ON recommendation_model_run(
        engineering_acceptance, generalization_claim_eligible, created_at
    );
CREATE INDEX IF NOT EXISTS recommendation_result_compound_idx
    ON recommendation_result(
        recommendation_model_run_id, compound_node_id, objective
    );
CREATE INDEX IF NOT EXISTS recommendation_result_request_idx
    ON recommendation_result(request_sha256);
CREATE INDEX IF NOT EXISTS model_registry_compound_idx ON public_pbpk_model_registry(compound_id);
CREATE UNIQUE INDEX IF NOT EXISTS model_registry_external_model_idx
    ON public_pbpk_model_registry(external_model_id)
    WHERE external_model_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS model_registry_reuse_idx
    ON public_pbpk_model_registry(
        reusable_candidate, run_label_grade, compound_id
    );
CREATE INDEX IF NOT EXISTS public_model_version_model_idx
    ON public_pbpk_model_version(model_id, reusable_candidate);
CREATE INDEX IF NOT EXISTS public_model_asset_version_kind_idx
    ON public_pbpk_model_asset(model_version_id, asset_kind);
CREATE INDEX IF NOT EXISTS public_model_simulation_version_role_idx
    ON public_pbpk_model_simulation(model_version_id, protocol_role);
CREATE INDEX IF NOT EXISTS public_model_validation_run_version_idx
    ON public_pbpk_model_validation_run(
        model_version_id, run_label_grade, scientific_accepted
    );
CREATE INDEX IF NOT EXISTS public_model_validation_dataset_run_idx
    ON public_pbpk_model_validation_dataset(validation_run_id);
CREATE INDEX IF NOT EXISTS public_model_validation_metric_run_idx
    ON public_pbpk_model_validation_metric(
        validation_run_id, validation_dataset_id, metric_scope
    );
CREATE UNIQUE INDEX IF NOT EXISTS public_model_validation_metric_unique_idx
    ON public_pbpk_model_validation_metric(
        validation_run_id,
        COALESCE(validation_dataset_id, '00000000-0000-0000-0000-000000000000'::uuid),
        metric_scope,
        metric_name
    );
CREATE INDEX IF NOT EXISTS curation_event_evidence_idx ON curation_event(evidence_table, evidence_id);
CREATE INDEX IF NOT EXISTS raw_document_source_path_idx ON raw_document(source_id, raw_storage_path);
CREATE INDEX IF NOT EXISTS parsed_document_checksum_idx ON parsed_document(source_id, parsed_checksum);
CREATE INDEX IF NOT EXISTS ingestion_run_external_run_id_idx ON ingestion_run(external_run_id);

ALTER TABLE IF EXISTS ingestion_run ADD COLUMN IF NOT EXISTS external_run_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS ingestion_run_external_run_id_unique_idx ON ingestion_run(external_run_id) WHERE external_run_id IS NOT NULL;

ALTER TABLE IF EXISTS physicochemical_property_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS physicochemical_property_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS adme_parameter_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS adme_parameter_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS clinical_pk_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS clinical_pk_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS ddi_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS ddi_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS special_population_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS special_population_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS enzyme_transporter_relationship ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS enzyme_transporter_relationship ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS model_parameter_evidence ADD COLUMN IF NOT EXISTS value_origin TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE IF EXISTS model_parameter_evidence ADD COLUMN IF NOT EXISTS evidence_origin TEXT NOT NULL DEFAULT 'unknown';

DROP VIEW IF EXISTS modeling_evidence;
DROP VIEW IF EXISTS accepted_evidence;
DROP VIEW IF EXISTS auto_validated_evidence;
DROP VIEW IF EXISTS reviewable_evidence;
DROP VIEW IF EXISTS excluded_from_reviewable;
DROP VIEW IF EXISTS preselected_evidence;

CREATE OR REPLACE FUNCTION reviewable_domain_exclusion_reason(
    p_evidence_type TEXT,
    p_parameter TEXT,
    p_source_id TEXT,
    p_source_text TEXT,
    p_value_text TEXT,
    p_unit TEXT,
    p_population TEXT,
    p_condition TEXT,
    p_route TEXT,
    p_dose TEXT,
    p_compound_role TEXT,
    p_attribution_method TEXT
) RETURNS TEXT AS $$
DECLARE
    et TEXT := lower(coalesce(p_evidence_type, ''));
    param TEXT := lower(btrim(coalesce(p_parameter, '')));
    source_id TEXT := lower(coalesce(p_source_id, ''));
    txt TEXT := lower(coalesce(p_source_text, ''));
    role TEXT := lower(coalesce(p_compound_role, ''));
    pop TEXT := lower(coalesce(p_population, ''));
    cond TEXT := lower(coalesce(p_condition, ''));
    route_text TEXT := lower(coalesce(p_route, ''));
    dose_text TEXT := lower(coalesce(p_dose, ''));
    has_ddi_effect BOOLEAN;
    has_special_population BOOLEAN;
BEGIN
    has_ddi_effect := txt ~ '(increase|increased|decrease|decreased|reduce|reduced|lower|lowered|raise|raised|alter|altered|affect|affected|effect|impact|interact|interaction|inhibit|inhibited|inhibition|induce|induced|induction|co.?administr|concomitant|no significant effect|unchanged|auc|cmax|clearance|half[- ]?life|toxicity)';
    has_special_population := (pop || ' ' || cond || ' ' || txt) ~ '(pregnan|lactat|breast.?milk|breastfeed|pediatric|paediatric|neonate|newborn|infant|child|adolescent|geriatric|elderly|hepatic impairment|liver disease|cirrhosis|renal impairment|kidney|genotype|polymorphism|metabolizer|sex difference|male|female|preterm)';

    IF et = 'physicochemical' THEN
        IF param = ANY (ARRAY[
            'molecularformula', 'molecular_formula',
            'molecularweight', 'molecular_weight',
            'inchikey', 'inchi_key',
            'canonicalsmiles', 'canonical_smiles',
            'xlogp', 'logp', 'logd', 'tpsa', 'pka',
            'solubility', 'ionization', 'fraction_unbound', 'protein_binding'
        ]) THEN
            RETURN NULL;
        END IF;
        IF param = ANY (ARRAY['dose_amount', 'formulation', 'product_strength'])
            OR param LIKE '%content%'
            OR txt LIKE '%green tea%'
            OR txt LIKE '%dry weight%'
            OR txt LIKE '%fresh weight%'
            OR txt LIKE '%capsule%'
            OR txt LIKE '%tablet%' THEN
            RETURN 'not_intrinsic_physicochemical_property';
        END IF;
        RETURN 'physicochemical_parameter_not_allowed';
    END IF;

    IF et = 'adme' THEN
        IF param = ANY (ARRAY[
            'activation_by_caffeine', 'ada_activity_inhibition', 'adenosine_antagonist',
            'alpha(2)-ampk_activity', 'calcium_response', 'caffeine_effect',
            'caffeine_inhibition', 'binding', 'biosynthesis', 'content',
            'fa_oxidation', 'insulin_sensitivity', 'lipid_absorption',
            'lymph_flow', 'maternal intake', 'motility_effect',
            'urinary_epinephrine_excretion', 'urinary_volume'
        ])
            OR (param LIKE '%activity%' AND param NOT LIKE '%cyp%')
            OR param LIKE '%effect%'
            OR param LIKE '%response%'
            OR txt LIKE '%plant%'
            OR txt LIKE '%transgenic%'
            OR txt LIKE '%exercise%' THEN
            RETURN 'pharmacodynamic_or_non_adme_endpoint';
        END IF;
        IF param = ANY (ARRAY['absorption', 'distribution', 'bioavailability', 'permeability', 'apparent_permeability'])
            AND (
                txt ~ '(caffeine|egcg).{0,120}(lower|lowered|decrease|decreased|increase|increased|enhance|enhanced|abolish|abolished|alter|alters|affect|affected|inhibit|inhibited).{0,120}(absorption|transport|uptake|bioavailability|exposure|distribution).{0,80}(alpha|cholesterol|lipid|sodium|na[+]|cl-|theacrine|vitamin|egcg|salicylic)'
                OR txt ~ '(caffeine|egcg).{0,120}(lower|lowered|decrease|decreased|increase|increased|enhance|enhanced|abolish|abolished|alter|alters|affect|affected|inhibit|inhibited).{0,120}(alpha|cholesterol|lipid|14c-ch|alpha toh|sodium|na[+]|cl-|theacrine|vitamin|egcg|salicylic).{0,80}(absorption|transport|uptake|bioavailability|exposure|distribution)'
                OR (
                    txt ~ '(14c-ch|alpha toh|theacrine|cholesterol|lipid)'
                    AND txt ~ '(lower|lowered|decrease|decreased|increase|increased|enhance|enhanced|abolish|abolished|alter|alters|affect|affected|inhibit|inhibited)'
                    AND txt LIKE '%caffeine%'
                )
            ) THEN
            RETURN 'compound_acts_on_other_adme_object';
        END IF;
        IF param = ANY (ARRAY[
            'absorption', 'distribution', 'metabolism', 'excretion',
            'bioavailability', 'permeability', 'apparent_permeability',
            'percutaneous_absorption', 'protein_binding', 'fraction_unbound',
            'fraction_unbound_in_plasma', 'renal_clearance', 'hepatic_clearance',
            'metabolism_enzyme', 'metabolism_pathway', 'metabolism_inhibition',
            'caffeine_metabolism', 'caffeine metabolite ratio',
            'paraxanthine/caffeine ratio', 'n3-demethylation_pathway_fraction',
            'c-8 hydroxylation', 'urinary_excretion', 'cyp1a2',
            'cyp1a2_activity', 'cyp1a inhibition', 'cyp2b induction', 'vmax',
            'enzyme', 'brain penetration'
        ])
            OR param LIKE 'cyp%'
            OR param LIKE 'ugt%'
            OR param LIKE '%transporter%'
            OR param LIKE '%metabol%'
            OR param LIKE '%clearance%'
            OR param LIKE '%excretion%'
            OR param LIKE '%permeab%'
            OR param LIKE '%bioavailability%' THEN
            RETURN NULL;
        END IF;
        RETURN 'adme_parameter_not_allowed';
    END IF;

    IF et = 'clinical_pk' THEN
        IF param = ANY (ARRAY['dose', 'route', 'formulation', 'avoiding caffeine'])
            OR param LIKE '%intake%'
            OR param LIKE '%consumption%'
            OR param LIKE '%blood_pressure%'
            OR param LIKE '%weight_loss%'
            OR param LIKE '%energy_intake%'
            OR param LIKE '%time_to_exhaustion%'
            OR param LIKE '%endurance%'
            OR param LIKE '%glycaemic%'
            OR param LIKE '%glucose%'
            OR param LIKE '%insulin%'
            OR param LIKE '%hip fracture%'
            OR param LIKE '%birth_size%'
            OR param LIKE '%acetylcholinesterase%'
            OR param LIKE '%acid_production%'
            OR param LIKE '%ca2+%'
            OR param LIKE '%occupancy%' THEN
            RETURN 'not_core_clinical_pk_parameter';
        END IF;
        IF param = ANY (ARRAY[
            'auc', 'cmax', 'tmax', 'half_life', 'clearance',
            'renal_clearance', 'renal_clearance_unbound',
            'volume_of_distribution', 'bioavailability',
            'fraction_excreted_unchanged', 'ae', 'plasma_concentration',
            'plasma_level', 'plasma_to_urine_ratio', 'salivary_concentration',
            'salivary_level', 'serum concentration', 'urine_concentration',
            'urinary_caffeine_excretion', 'urinary caffeine levels',
            'urinary_excretion', 'excretion_rate', 'excretion_time',
            'total_urinary_recovery', 'metabolite_ratio',
            'paraxanthine/caffeine ratio', 'conversion_ratio',
            'elimination', 'elimination_rate_constant',
            'mean_residence_time', 'concentration', 'metabolism',
            'metabolizer_status', 'iauc', 'tbr', 'flux',
            'follicular_fluid_concentration', 'uterine_fluid_to_plasma_ratio',
            'sodium_lithium_clearance_ratio', 'fractional_excretion_of_lithium',
            'fractional_excretion_of_sodium', 'lithium_clearance'
        ])
            OR param LIKE '%auc%'
            OR param LIKE '%cmax%'
            OR param LIKE '%clearance%'
            OR param LIKE '%concentration%'
            OR param LIKE '%half_life%'
            OR param LIKE '%half-life%'
            OR param LIKE '%excretion%'
            OR param LIKE '%bioavailability%'
            OR param LIKE '%volume_of_distribution%' THEN
            RETURN NULL;
        END IF;
        RETURN 'clinical_pk_parameter_not_allowed';
    END IF;

    IF et = 'ddi' THEN
        IF param ~ '(salicylic|aspirin|midazolam|omeprazole|warfarin|digoxin|dextromethorphan|bupropion|metoprolol|codeine|morphine)'
            AND param NOT LIKE '%caffeine%' THEN
            RETURN 'parameter_points_to_other_drug';
        END IF;
        IF param ~ '(cyp2c19|cyp2c9|cyp2d6|cyp2b6|cyp3a4|cyp3a)'
            AND txt LIKE '%caffeine%'
            AND txt LIKE '%cyp1a2%'
            AND txt ~ '(midazolam|omeprazole|warfarin|digoxin|dextromethorphan|bupropion|metoprolol|losartan)' THEN
            RETURN 'probe_target_mismatch_for_compound';
        END IF;
        IF param = 'caffeine'
            AND NOT (txt ~ '(auc|cmax|clearance|metabolism|pharmacokinetic|exposure|cyp|co.?administr|concomitant|plasma concentration|half[- ]?life)') THEN
            RETURN 'pharmacodynamic_interaction_not_pk_ddi';
        END IF;
        IF role NOT IN ('victim_drug', 'perpetrator_drug') THEN
            RETURN 'missing_victim_or_perpetrator_role';
        END IF;
        IF source_id IN ('dailymed', 'openfda_drug_label')
            AND txt ~ '(codeine|morphine|opioid|butalbital)'
            AND param ~ '(cyp2d6|cyp3a4|cns depressant|respiratory depression)' THEN
            RETURN 'combination_product_interaction_not_caffeine_specific';
        END IF;
        IF txt LIKE '%probe%'
            AND (txt LIKE '%cocktail%' OR txt LIKE '%substrate%')
            AND NOT has_ddi_effect THEN
            RETURN 'probe_or_cocktail_description_not_ddi';
        END IF;
        IF param ~ '^(cyp|ugt|p-gp|pgp|bcrp|oat|oatp|oct|mate)'
            AND NOT has_ddi_effect THEN
            RETURN 'enzyme_or_probe_fact_not_ddi';
        END IF;
        IF NOT has_ddi_effect THEN
            RETURN 'ddi_lacks_interaction_effect_or_direction';
        END IF;
        RETURN NULL;
    END IF;

    IF et = 'special_population' THEN
        IF source_id IN ('dailymed', 'openfda_drug_label')
            AND txt ~ '(codeine|morphine|opioid|butalbital)'
            AND param IN ('labor', 'labor_effect', 'pediatric', 'geriatric') THEN
            RETURN 'combination_product_warning_not_caffeine_specific';
        END IF;
        IF NOT has_special_population THEN
            RETURN 'special_population_context_not_identified';
        END IF;
        RETURN NULL;
    END IF;

    IF et = 'public_pbpk_model' THEN
        IF source_id NOT IN ('osp_pbpk_model_library') THEN
            RETURN 'model_mention_only_without_public_model_files';
        END IF;
        RETURN NULL;
    END IF;

    IF et = 'tissue_expression' THEN
        IF source_id NOT IN ('gtex', 'hpa', 'uniprot')
            OR NOT (txt LIKE '%expression%' OR txt LIKE '%tissue%' OR txt LIKE '%protein%') THEN
            RETURN 'not_tissue_expression_evidence';
        END IF;
        RETURN NULL;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION reviewable_domain_suggested_type(
    p_evidence_type TEXT,
    p_parameter TEXT,
    p_source_id TEXT,
    p_source_text TEXT,
    p_value_text TEXT,
    p_unit TEXT,
    p_population TEXT,
    p_condition TEXT,
    p_route TEXT,
    p_dose TEXT,
    p_compound_role TEXT,
    p_attribution_method TEXT
) RETURNS TEXT AS $$
DECLARE
    reason TEXT;
    param TEXT := lower(btrim(coalesce(p_parameter, '')));
BEGIN
    reason := reviewable_domain_exclusion_reason(
        p_evidence_type, p_parameter, p_source_id, p_source_text, p_value_text,
        p_unit, p_population, p_condition, p_route, p_dose, p_compound_role,
        p_attribution_method
    );
    IF reason IS NULL THEN
        RETURN p_evidence_type;
    END IF;
    IF reason = 'not_intrinsic_physicochemical_property' THEN
        IF param LIKE '%content%' THEN
            RETURN 'sample_or_food_content_candidate';
        END IF;
        RETURN 'dosing_or_product_label_candidate';
    END IF;
    IF reason LIKE 'pharmacodynamic%' THEN
        RETURN 'pharmacodynamic_candidate';
    END IF;
    IF reason = 'compound_acts_on_other_adme_object' THEN
        RETURN 'off_target_adme_or_pharmacodynamic_candidate';
    END IF;
    IF reason LIKE 'not_core_clinical_pk%' THEN
        RETURN 'dosing_outcome_or_safety_candidate';
    END IF;
    IF reason LIKE 'probe_or_cocktail%' OR reason LIKE 'enzyme_or_probe%' THEN
        RETURN 'enzyme_probe_or_cocktail_candidate';
    END IF;
    IF reason = 'probe_target_mismatch_for_compound' THEN
        RETURN 'enzyme_probe_or_cocktail_candidate';
    END IF;
    IF reason = 'parameter_points_to_other_drug' THEN
        RETURN 'other_drug_parameter_candidate';
    END IF;
    IF reason = 'pharmacodynamic_interaction_not_pk_ddi' THEN
        RETURN 'pharmacodynamic_candidate';
    END IF;
    IF reason LIKE 'combination_product%' THEN
        RETURN 'combination_product_warning';
    END IF;
    IF reason LIKE 'model_mention_only%' THEN
        RETURN 'model_mention_only';
    END IF;
    IF reason LIKE 'not_tissue_expression%' THEN
        RETURN 'pharmacogenomic_or_target_candidate';
    END IF;
    RETURN 'candidate_review_required';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE VIEW preselected_evidence AS
SELECT
    pec.*,
    eca.compound_id AS attributed_compound_id,
    eca.compound_name AS attributed_compound_name,
    eca.attribution_id,
    eca.compound_role,
    eca.attribution_method,
    eca.attribution_confidence,
    eca.matched_text,
    eca.matched_synonym,
    eca.attribution_reason,
    eca.llm_required,
    eca.llm_trigger_reason,
    eca.attribution_stage,
    layer.evidence_layer,
    layer.exclusion_reasons,
    layer.review_blockers,
    layer.semantic_group_ids,
    layer.conflict_group_ids,
    layer.metadata_json AS layer_metadata_json
FROM processed_evidence_candidate pec
JOIN evidence_compound_attribution eca ON eca.evidence_id = pec.evidence_id
JOIN evidence_layer_membership layer
  ON layer.evidence_id = pec.evidence_id
 AND layer.evidence_layer = 'preselected';

CREATE OR REPLACE VIEW excluded_from_reviewable AS
SELECT
    pec.*,
    eca.compound_id AS attributed_compound_id,
    eca.compound_name AS attributed_compound_name,
    eca.attribution_id,
    eca.compound_role,
    eca.attribution_method,
    eca.attribution_confidence,
    eca.matched_text,
    eca.matched_synonym,
    eca.attribution_reason,
    eca.llm_required,
    eca.llm_trigger_reason,
    eca.attribution_stage,
    layer.evidence_layer,
    layer.exclusion_reasons,
    layer.review_blockers,
    layer.semantic_group_ids,
    layer.conflict_group_ids,
    layer.metadata_json AS layer_metadata_json,
    array_to_string(ARRAY(SELECT jsonb_array_elements_text(layer.exclusion_reasons)), '; ') AS exclusion_reason,
    NULL::text AS suggested_evidence_type
FROM processed_evidence_candidate pec
JOIN evidence_compound_attribution eca ON eca.evidence_id = pec.evidence_id
JOIN evidence_layer_membership layer
  ON layer.evidence_id = pec.evidence_id
 AND layer.evidence_layer = 'excluded';

CREATE OR REPLACE VIEW reviewable_evidence AS
SELECT
    pec.*,
    eca.compound_id AS attributed_compound_id,
    eca.compound_name AS attributed_compound_name,
    eca.attribution_id,
    eca.compound_role,
    eca.attribution_method,
    eca.attribution_confidence,
    eca.matched_text,
    eca.matched_synonym,
    eca.attribution_reason,
    eca.llm_required,
    eca.llm_trigger_reason,
    eca.attribution_stage,
    layer.evidence_layer,
    layer.exclusion_reasons,
    layer.review_blockers,
    layer.semantic_group_ids,
    layer.conflict_group_ids,
    layer.metadata_json AS layer_metadata_json
FROM processed_evidence_candidate pec
JOIN evidence_compound_attribution eca ON eca.evidence_id = pec.evidence_id
JOIN evidence_layer_membership layer
  ON layer.evidence_id = pec.evidence_id
 AND layer.evidence_layer = 'reviewable'
WHERE pec.curation_status NOT IN ('auto_validated', 'reviewed_rejected', 'duplicate', 'deprecated', 'superseded');

CREATE OR REPLACE VIEW auto_validated_evidence AS
SELECT
    pec.*,
    eca.compound_id AS attributed_compound_id,
    eca.compound_name AS attributed_compound_name,
    eca.attribution_id,
    eca.compound_role,
    eca.attribution_method,
    eca.attribution_confidence,
    eca.matched_text,
    eca.matched_synonym,
    eca.attribution_reason,
    eca.llm_required,
    eca.llm_trigger_reason,
    eca.attribution_stage,
    layer.evidence_layer,
    layer.exclusion_reasons,
    layer.review_blockers,
    layer.semantic_group_ids,
    layer.conflict_group_ids,
    layer.metadata_json AS layer_metadata_json
FROM processed_evidence_candidate pec
JOIN evidence_compound_attribution eca ON eca.evidence_id = pec.evidence_id
JOIN evidence_layer_membership layer
  ON layer.evidence_id = pec.evidence_id
 AND layer.evidence_layer = 'auto_validated'
WHERE pec.curation_status = 'auto_validated';

CREATE OR REPLACE VIEW accepted_evidence AS
SELECT
    pec.*,
    eca.compound_id AS attributed_compound_id,
    eca.compound_name AS attributed_compound_name,
    eca.attribution_id,
    eca.compound_role,
    eca.attribution_method,
    eca.attribution_confidence,
    eca.matched_text,
    eca.matched_synonym,
    eca.attribution_reason,
    eca.llm_required,
    eca.llm_trigger_reason,
    eca.attribution_stage,
    layer.evidence_layer,
    layer.exclusion_reasons,
    layer.review_blockers,
    layer.semantic_group_ids,
    layer.conflict_group_ids,
    layer.metadata_json AS layer_metadata_json
FROM processed_evidence_candidate pec
JOIN evidence_compound_attribution eca
    ON eca.evidence_id = pec.evidence_id
JOIN evidence_layer_membership layer
  ON layer.evidence_id = pec.evidence_id
 AND layer.evidence_layer = 'accepted'
WHERE pec.curation_status IN ('reviewed_accepted', 'approved_for_modeling')
  AND EXISTS (
      SELECT 1
      FROM curation_event ce
      WHERE ce.evidence_table = 'processed_evidence_candidate'
        AND ce.evidence_id = pec.evidence_id
        AND ce.new_status IN ('reviewed_accepted', 'approved_for_modeling')
  );

CREATE OR REPLACE VIEW modeling_evidence AS
SELECT
    pec.*,
    eca.compound_id AS attributed_compound_id,
    eca.compound_name AS attributed_compound_name,
    eca.attribution_id,
    eca.compound_role,
    eca.attribution_method,
    eca.attribution_confidence,
    eca.matched_text,
    eca.matched_synonym,
    eca.attribution_reason,
    eca.llm_required,
    eca.llm_trigger_reason,
    eca.attribution_stage,
    layer.evidence_layer,
    layer.exclusion_reasons,
    layer.review_blockers,
    layer.semantic_group_ids,
    layer.conflict_group_ids,
    layer.metadata_json AS layer_metadata_json
FROM processed_evidence_candidate pec
JOIN evidence_compound_attribution eca
    ON eca.evidence_id = pec.evidence_id
JOIN evidence_layer_membership layer
  ON layer.evidence_id = pec.evidence_id
 AND layer.evidence_layer = 'modeling'
WHERE pec.curation_status = 'approved_for_modeling'
  AND EXISTS (
      SELECT 1
      FROM curation_event ce
      WHERE ce.evidence_table = 'processed_evidence_candidate'
        AND ce.evidence_id = pec.evidence_id
        AND ce.new_status = 'approved_for_modeling'
  );

-- Public-model decisions and executable PBPK blueprint audit trail.
CREATE TABLE IF NOT EXISTS requirement_profile (
    profile_id UUID PRIMARY KEY,
    request_id UUID NOT NULL,
    compound_id UUID NOT NULL REFERENCES compound_identity(compound_id),
    profile_version TEXT NOT NULL,
    ontology_version TEXT NOT NULL,
    ranking_eligibility TEXT NOT NULL CHECK (
        ranking_eligibility IN ('ready', 'blocked')
    ),
    request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[a-f0-9]{64}$'),
    request_json JSONB NOT NULL,
    profile_checksum TEXT NOT NULL UNIQUE CHECK (
        profile_checksum ~ '^[a-f0-9]{64}$'
    ),
    profile_json JSONB NOT NULL,
    source_evidence_run_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE requirement_profile
    ADD COLUMN IF NOT EXISTS request_json JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS model_ranking_run (
    ranking_run_id UUID PRIMARY KEY,
    profile_id UUID NOT NULL REFERENCES requirement_profile(profile_id),
    decision TEXT NOT NULL CHECK (decision IN ('reuse', 'adapt', 'abstain')),
    selected_model_version_id UUID REFERENCES public_pbpk_model_version(model_version_id),
    decision_certainty DOUBLE PRECISION NOT NULL CHECK (
        decision_certainty BETWEEN 0 AND 1
    ),
    confidence_semantics TEXT NOT NULL CHECK (
        confidence_semantics IN ('deterministic_gate', 'probability_calibrated')
    ),
    ranking_schema_version TEXT NOT NULL,
    ranker_version TEXT NOT NULL,
    model_run_manifest_id UUID NOT NULL,
    result_checksum TEXT NOT NULL UNIQUE CHECK (
        result_checksum ~ '^[a-f0-9]{64}$'
    ),
    result_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'model_ranking_run'
          AND column_name = 'calibrated_confidence'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'model_ranking_run'
          AND column_name = 'decision_certainty'
    ) THEN
        ALTER TABLE model_ranking_run
            RENAME COLUMN calibrated_confidence TO decision_certainty;
    END IF;
END
$$;

ALTER TABLE model_ranking_run
    ADD COLUMN IF NOT EXISTS confidence_semantics TEXT
    NOT NULL DEFAULT 'deterministic_gate'
    CHECK (
        confidence_semantics IN ('deterministic_gate', 'probability_calibrated')
    );

CREATE TABLE IF NOT EXISTS model_ranking_candidate (
    ranking_run_id UUID NOT NULL REFERENCES model_ranking_run(ranking_run_id) ON DELETE CASCADE,
    model_version_id UUID NOT NULL REFERENCES public_pbpk_model_version(model_version_id),
    rank INTEGER NOT NULL CHECK (rank > 0),
    candidate_class TEXT NOT NULL,
    eligibility TEXT NOT NULL,
    deterministic_compatibility DOUBLE PRECISION,
    exclusion_reason_codes TEXT[] NOT NULL DEFAULT '{}',
    constraint_matrix_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ranking_run_id, model_version_id),
    UNIQUE (ranking_run_id, rank)
);

CREATE TABLE IF NOT EXISTS pbpk_blueprint_template (
    template_id UUID PRIMARY KEY,
    external_template_id TEXT NOT NULL,
    template_version TEXT NOT NULL,
    compound_id UUID NOT NULL REFERENCES compound_identity(compound_id),
    target_runtime_adapter TEXT NOT NULL,
    runtime_version TEXT NOT NULL,
    source_repository TEXT NOT NULL,
    source_commit TEXT,
    source_manifest_sha256 TEXT CHECK (
        source_manifest_sha256 IS NULL OR
        source_manifest_sha256 ~ '^[a-f0-9]{64}$'
    ),
    source_project_path TEXT NOT NULL,
    source_project_sha256 TEXT NOT NULL CHECK (
        source_project_sha256 ~ '^[a-f0-9]{64}$'
    ),
    source_skeleton_path TEXT NOT NULL,
    source_skeleton_sha256 TEXT NOT NULL CHECK (
        source_skeleton_sha256 ~ '^[a-f0-9]{64}$'
    ),
    license_spdx_id TEXT NOT NULL,
    license_source_location TEXT NOT NULL,
    license_sha256 TEXT NOT NULL CHECK (license_sha256 ~ '^[a-f0-9]{64}$'),
    structural_status TEXT NOT NULL,
    scientific_status TEXT NOT NULL,
    template_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    scientific_registration_allowed BOOLEAN NOT NULL DEFAULT FALSE,
    template_checksum TEXT NOT NULL UNIQUE CHECK (
        template_checksum ~ '^[a-f0-9]{64}$'
    ),
    template_json JSONB NOT NULL,
    asset_root TEXT NOT NULL,
    source_id TEXT REFERENCES source_registry(source_id),
    source_location TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (source_commit IS NOT NULL OR source_manifest_sha256 IS NOT NULL),
    UNIQUE (external_template_id, template_version)
);

CREATE TABLE IF NOT EXISTS pbpk_blueprint (
    blueprint_id UUID PRIMARY KEY,
    compound_id UUID NOT NULL REFERENCES compound_identity(compound_id),
    profile_id UUID NOT NULL REFERENCES requirement_profile(profile_id),
    template_id UUID NOT NULL REFERENCES pbpk_blueprint_template(template_id),
    ranking_run_id UUID REFERENCES model_ranking_run(ranking_run_id),
    objective TEXT NOT NULL,
    current_status TEXT NOT NULL CHECK (
        current_status IN (
            'draft_grounded', 'structurally_validated', 'compiled',
            'executable', 'calibrated', 'externally_validated', 'quarantined'
        )
    ),
    scientific_status TEXT NOT NULL,
    registration_allowed BOOLEAN NOT NULL DEFAULT FALSE,
    blocker_codes TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (profile_id, template_id)
);

CREATE TABLE IF NOT EXISTS blueprint_version (
    blueprint_version_id UUID PRIMARY KEY,
    blueprint_id UUID NOT NULL REFERENCES pbpk_blueprint(blueprint_id) ON DELETE CASCADE,
    blueprint_version TEXT NOT NULL,
    parent_blueprint_checksum TEXT CHECK (
        parent_blueprint_checksum IS NULL OR
        parent_blueprint_checksum ~ '^[a-f0-9]{64}$'
    ),
    status TEXT NOT NULL CHECK (
        status IN (
            'draft_grounded', 'structurally_validated', 'compiled',
            'executable', 'calibrated', 'externally_validated', 'quarantined'
        )
    ),
    blueprint_checksum TEXT NOT NULL UNIQUE CHECK (
        blueprint_checksum ~ '^[a-f0-9]{64}$'
    ),
    blueprint_json JSONB NOT NULL,
    validation_report_json JSONB,
    compilation_report_json JSONB,
    source_location TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (blueprint_id, blueprint_version)
);

CREATE TABLE IF NOT EXISTS blueprint_component (
    blueprint_version_id UUID NOT NULL REFERENCES blueprint_version(blueprint_version_id) ON DELETE CASCADE,
    component_id TEXT NOT NULL,
    component_kind TEXT NOT NULL,
    semantic_id TEXT,
    payload_json JSONB NOT NULL,
    source_location TEXT NOT NULL,
    source_checksum TEXT NOT NULL CHECK (source_checksum ~ '^[a-f0-9]{64}$'),
    PRIMARY KEY (blueprint_version_id, component_kind, component_id)
);

CREATE TABLE IF NOT EXISTS blueprint_parameter (
    blueprint_version_id UUID NOT NULL REFERENCES blueprint_version(blueprint_version_id) ON DELETE CASCADE,
    parameter_id TEXT NOT NULL,
    semantic_id TEXT NOT NULL,
    parameter_name TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('observed', 'derived', 'estimated', 'missing')),
    value DOUBLE PRECISION,
    initial_value DOUBLE PRECISION,
    canonical_unit TEXT,
    blocking BOOLEAN NOT NULL,
    calibration_run_id UUID,
    provenance_json JSONB NOT NULL,
    payload_json JSONB NOT NULL,
    PRIMARY KEY (blueprint_version_id, parameter_id)
);

CREATE TABLE IF NOT EXISTS blueprint_artifact (
    blueprint_artifact_id UUID PRIMARY KEY,
    blueprint_version_id UUID NOT NULL REFERENCES blueprint_version(blueprint_version_id) ON DELETE CASCADE,
    artifact_kind TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    byte_size BIGINT NOT NULL CHECK (byte_size >= 0),
    source_location TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (blueprint_version_id, artifact_kind, relative_path, sha256)
);

CREATE TABLE IF NOT EXISTS runtime_job (
    job_id UUID PRIMARY KEY,
    job_type TEXT NOT NULL CHECK (
        job_type IN (
            'asset_parse', 'execution_probe', 'rank', 'adapt', 'compile',
            'simulate', 'calibrate', 'validate'
        )
    ),
    idempotency_key TEXT NOT NULL UNIQUE CHECK (
        idempotency_key ~ '^[a-f0-9]{64}$'
    ),
    status TEXT NOT NULL CHECK (
        status IN (
            'queued', 'running', 'succeeded', 'failed', 'cancelled', 'timed_out'
        )
    ),
    attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt BETWEEN 1 AND 2),
    progress_current INTEGER NOT NULL DEFAULT 0 CHECK (progress_current >= 0),
    progress_total INTEGER NOT NULL DEFAULT 1 CHECK (progress_total > 0),
    worker_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    input_manifest_sha256 TEXT NOT NULL CHECK (
        input_manifest_sha256 ~ '^[a-f0-9]{64}$'
    ),
    output_manifest_sha256 TEXT CHECK (
        output_manifest_sha256 IS NULL OR output_manifest_sha256 ~ '^[a-f0-9]{64}$'
    ),
    error_code TEXT,
    error_log_location TEXT,
    input_json JSONB NOT NULL,
    output_json JSONB
);

CREATE TABLE IF NOT EXISTS runtime_job_event (
    event_id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES runtime_job(job_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS simulation_run (
    simulation_run_id UUID PRIMARY KEY,
    blueprint_version_id UUID NOT NULL REFERENCES blueprint_version(blueprint_version_id),
    job_id UUID NOT NULL REFERENCES runtime_job(job_id),
    scenario_id TEXT NOT NULL,
    run_ordinal INTEGER NOT NULL CHECK (run_ordinal > 0),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    runtime_json JSONB NOT NULL,
    result_manifest_sha256 TEXT NOT NULL CHECK (
        result_manifest_sha256 ~ '^[a-f0-9]{64}$'
    ),
    source_location TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    UNIQUE (blueprint_version_id, scenario_id, run_ordinal)
);

CREATE TABLE IF NOT EXISTS simulation_output (
    simulation_output_id UUID PRIMARY KEY,
    simulation_run_id UUID NOT NULL REFERENCES simulation_run(simulation_run_id) ON DELETE CASCADE,
    output_kind TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    byte_size BIGINT NOT NULL CHECK (byte_size >= 0),
    row_count BIGINT CHECK (row_count IS NULL OR row_count >= 0),
    source_location TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (simulation_run_id, output_kind, relative_path)
);

CREATE INDEX IF NOT EXISTS ix_blueprint_template_compound
    ON pbpk_blueprint_template (compound_id, template_eligible);
CREATE INDEX IF NOT EXISTS ix_blueprint_compound_status
    ON pbpk_blueprint (compound_id, current_status);
CREATE INDEX IF NOT EXISTS ix_runtime_job_status
    ON runtime_job (status, created_at);
CREATE INDEX IF NOT EXISTS ix_simulation_run_blueprint
    ON simulation_run (blueprint_version_id, scenario_id);
