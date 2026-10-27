// Stage 1 knowledge graph projection schema.
//
// The MVP stores graph facts in PostgreSQL and projects them through FastAPI.
// These labels/relationships document the equivalent graph shape for a future
// Neo4j/RDF loader.

CREATE CONSTRAINT compound_id IF NOT EXISTS FOR (n:Compound) REQUIRE n.compound_id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.source_id IS UNIQUE;
CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:Evidence) REQUIRE n.evidence_id IS UNIQUE;
CREATE CONSTRAINT model_id IF NOT EXISTS FOR (n:PublicPBPKModel) REQUIRE n.model_id IS UNIQUE;
CREATE CONSTRAINT protein_target_id IF NOT EXISTS FOR (n:ProteinTarget) REQUIRE n.protein_target_id IS UNIQUE;

// Compound evidence and provenance.
// (:Compound)-[:HAS_EVIDENCE]->(:Evidence)-[:DERIVED_FROM]->(:Source)

// Public model discovery.
// (:Compound)-[:REPRESENTED_BY]->(:PublicPBPKModel)-[:DERIVED_FROM]->(:Source)

// Enzyme/transporter/target relationships from public sources such as ChEMBL.
// (:Compound)-[:SUBSTRATE_OF|INHIBITS_OR_BINDS|INDUCES|HAS_ACTIVITY_AGAINST]->(:ProteinTarget)-[:DERIVED_FROM]->(:Source)

// Tissue expression context when public tissue-expression connectors are loaded.
// (:ProteinTarget)-[:EXPRESSED_IN]->(:TissueExpression)-[:DERIVED_FROM]->(:Source)
