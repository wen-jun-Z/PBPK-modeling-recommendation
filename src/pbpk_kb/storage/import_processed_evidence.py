from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, UUID, uuid5

from pbpk_kb.config.sources import load_source_registry
from pbpk_kb.config.settings import get_settings
from pbpk_kb.graph.final_evidence_graph import extract_graph_facts
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
from pbpk_kb.storage.db import create_db_engine
from pbpk_kb.storage.source_registry import upsert_source_registry_entries


DOCKER_CONTAINER = "pbpk_stage1_postgres"
DB_USER = "pbpk"
DB_NAME = "pbpk_kb"
CURATION_STATUS = {
    "machine_extracted",
    "needs_review",
    "reviewed_accepted",
    "reviewed_rejected",
    "approved_for_modeling",
    "auto_validated",
    "deprecated",
    "duplicate",
    "superseded",
}


RAW_COLUMNS = [
    "raw_document_id",
    "source_id",
    "source_url",
    "source_record_id",
    "source_version",
    "raw_storage_path",
    "raw_mime_type",
    "raw_checksum",
    "raw_size_bytes",
    "metadata",
]

PARSED_COLUMNS = [
    "parsed_document_id",
    "raw_document_id",
    "source_id",
    "parser_name",
    "parser_version",
    "document_type",
    "title",
    "source_location",
    "parsed_storage_path",
    "parsed_record",
    "parsed_checksum",
]

COMPOUND_COLUMNS = [
    "compound_id",
    "preferred_name",
    "normalized_name",
    "synonyms",
    "pubchem_cid",
    "chembl_id",
    "drugcentral_id",
    "rxnorm_rxcui",
    "inchi_key",
    "source_id",
    "source_location",
    "confidence_score",
    "curation_status",
]

CANDIDATE_COLUMNS = [
    "evidence_id",
    "run_id",
    "evidence_type",
    "compound_name",
    "compound_id",
    "source_id",
    "raw_asset_id",
    "raw_document_id",
    "parsed_document_id",
    "parsed_table_id",
    "chunk_id",
    "storage_path",
    "source_url",
    "checksum",
    "source_location",
    "source_text",
    "source_text_hash",
    "extraction_method",
    "extractor_name",
    "extractor_version",
    "extraction_confidence",
    "curation_status",
    "parameter",
    "value",
    "value_text",
    "unit",
    "canonical_value",
    "canonical_unit",
    "condition",
    "population",
    "route",
    "dose",
    "metadata_json",
    "source_record",
]

PHYS_COLUMNS = [
    "evidence_id",
    "compound_id",
    "property_name",
    "property_value",
    "property_value_text",
    "unit",
    "canonical_value",
    "canonical_unit",
    "source_id",
    "raw_document_id",
    "parsed_document_id",
    "source_location",
    "evidence_text",
    "value_origin",
    "evidence_origin",
    "extraction_method",
    "extractor_version",
    "confidence_score",
    "curation_status",
]

ADME_COLUMNS = [
    "evidence_id",
    "compound_id",
    "parameter_name",
    "value",
    "value_text",
    "unit",
    "canonical_value",
    "canonical_unit",
    "enzyme",
    "transporter",
    "source_id",
    "raw_document_id",
    "parsed_document_id",
    "source_location",
    "evidence_text",
    "value_origin",
    "evidence_origin",
    "extraction_method",
    "extractor_version",
    "confidence_score",
    "curation_status",
]

CLINICAL_COLUMNS = [
    "evidence_id",
    "compound_id",
    "study_context",
    "population",
    "dose",
    "route",
    "pk_parameter",
    "value",
    "value_text",
    "unit",
    "canonical_value",
    "canonical_unit",
    "renal_impairment_flag",
    "hepatic_impairment_flag",
    "source_id",
    "raw_document_id",
    "parsed_document_id",
    "source_location",
    "evidence_text",
    "value_origin",
    "evidence_origin",
    "extraction_method",
    "extractor_version",
    "confidence_score",
    "curation_status",
]

DDI_COLUMNS = [
    "evidence_id",
    "victim_compound_id",
    "perpetrator_compound_id",
    "enzyme_or_transporter",
    "interaction_type",
    "study_type",
    "auc_ratio",
    "cmax_ratio",
    "dose_victim",
    "dose_perpetrator",
    "population",
    "source_id",
    "raw_document_id",
    "parsed_document_id",
    "source_location",
    "evidence_text",
    "value_origin",
    "evidence_origin",
    "extraction_method",
    "extractor_version",
    "confidence_score",
    "curation_status",
]

SPECIAL_COLUMNS = [
    "evidence_id",
    "compound_id",
    "population_type",
    "population_definition",
    "covariate",
    "pk_parameter",
    "value",
    "value_text",
    "unit",
    "canonical_value",
    "canonical_unit",
    "ratio_to_reference",
    "source_id",
    "raw_document_id",
    "parsed_document_id",
    "source_location",
    "evidence_text",
    "value_origin",
    "evidence_origin",
    "extraction_method",
    "extractor_version",
    "confidence_score",
    "curation_status",
]

TISSUE_COLUMNS = [
    "evidence_id",
    "protein_target_id",
    "tissue",
    "expression_value",
    "expression_value_text",
    "unit",
    "canonical_value",
    "canonical_unit",
    "species",
    "assay_method",
    "source_id",
    "raw_document_id",
    "parsed_document_id",
    "source_location",
    "evidence_text",
    "value_origin",
    "evidence_origin",
    "extraction_method",
    "extractor_version",
    "confidence_score",
    "curation_status",
]

MODEL_COLUMNS = [
    "model_id",
    "model_name",
    "compound_id",
    "model_type",
    "software",
    "species",
    "population",
    "route",
    "model_source",
    "source_id",
    "model_file_path",
    "parameter_file_path",
    "evaluation_report_path",
    "publication_id",
    "validation_status",
    "curation_status",
    "source_location",
    "notes",
]

ATTRIBUTION_COLUMNS = [
    "attribution_id",
    "evidence_id",
    "compound_id",
    "compound_name",
    "compound_role",
    "attribution_method",
    "attribution_confidence",
    "matched_text",
    "matched_synonym",
    "attribution_reason",
    "llm_required",
    "llm_trigger_reason",
    "attribution_stage",
    "llm_payload",
]

DUPLICATE_GROUP_COLUMNS = [
    "duplicate_group_id",
    "group_type",
    "group_key",
    "evidence_type",
    "compound_id",
    "compound_name",
    "representative_evidence_id",
    "duplicate_count",
    "selection_reason",
    "metadata_json",
]

DUPLICATE_MEMBER_COLUMNS = [
    "duplicate_group_id",
    "evidence_id",
    "is_representative",
    "duplicate_rank",
    "duplicate_reason",
    "similarity_score",
]

LAYER_COLUMNS = [
    "evidence_id",
    "run_id",
    "evidence_layer",
    "exclusion_reasons",
    "review_blockers",
    "semantic_group_ids",
    "conflict_group_ids",
    "metadata_json",
]

CONFLICT_GROUP_COLUMNS = [
    "conflict_group_id",
    "run_id",
    "source_group_id",
    "source_group_type",
    "compound_id",
    "compound_name",
    "evidence_type",
    "parameter",
    "canonical_unit",
    "evidence_count",
    "analysis_status",
    "analysis_reason",
    "resolution",
    "requires_human_review",
    "metadata_json",
]

CONFLICT_MEMBER_COLUMNS = [
    "conflict_group_id",
    "evidence_id",
    "analysis_status",
    "requires_human_review",
    "source_group_id",
    "source_group_type",
    "source_id",
    "source_location",
]

RAG_COLUMNS = [
    "retrieval_record_id",
    "run_id",
    "embedding_row",
    "compound_id",
    "compound_name",
    "source_id",
    "source_location",
    "source_url",
    "storage_path",
    "checksum",
    "source_text",
    "source_text_hash",
    "embedding_text",
    "embedding",
    "embedding_model",
    "evidence_layer",
    "record_kinds",
    "evidence_ids",
    "chunk_ids",
    "parsed_table_ids",
    "recovered_table_ids",
    "provenance_status",
    "curation_status",
    "metadata_json",
]

TARGET_COLUMNS = [
    "protein_target_id",
    "preferred_name",
    "gene_symbol",
    "uniprot_id",
    "target_type",
    "synonyms",
    "source_id",
    "source_location",
    "confidence_score",
    "curation_status",
]

RELATIONSHIP_COLUMNS = [
    "relationship_id",
    "compound_id",
    "protein_target_id",
    "relationship_type",
    "evidence_direction",
    "source_id",
    "raw_document_id",
    "parsed_document_id",
    "processed_evidence_id",
    "source_location",
    "evidence_text",
    "value_origin",
    "evidence_origin",
    "extraction_method",
    "extractor_version",
    "confidence_score",
    "curation_status",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import processed raw evidence candidates into the Stage 1 PostgreSQL schema.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--final-evidence-dir", help="Import final attribution, deduplication, conflict, layer, and RAG outputs.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--container", default=DOCKER_CONTAINER)
    parser.add_argument("--db-user", default=DB_USER)
    parser.add_argument("--db-name", default=DB_NAME)
    parser.add_argument("--replace", action="store_true", help="Clear generated evidence/raw/parsed/compound rows before importing this processed run.")
    parser.add_argument("--limit", type=int, help="Import at most this many evidence candidates, for smoke tests.")
    args = parser.parse_args()

    importer = EvidenceImporter(
        run_dir=args.run_dir,
        final_evidence_dir=Path(args.final_evidence_dir) if args.final_evidence_dir else None,
        container=args.container,
        db_user=args.db_user,
        db_name=args.db_name,
        database_url=args.database_url,
        replace=args.replace,
        limit=args.limit,
    )
    summary = importer.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


class EvidenceImporter:
    def __init__(
        self,
        *,
        run_dir: Path,
        container: str,
        db_user: str,
        db_name: str,
        replace: bool,
        limit: int | None = None,
        final_evidence_dir: Path | None = None,
        database_url: str | None = None,
    ) -> None:
        self.paths = raw_processing_layout(run_dir)
        self.run_dir = self.paths.root
        self.container = container
        self.db_user = db_user
        self.db_name = db_name
        self.native_database_url = database_url
        self.database_url = database_url or get_settings().database_url
        self.replace = replace
        self.limit = limit
        self.final_evidence_dir = final_evidence_dir
        from pbpk_kb.storage.expansion_identity_mapping import (
            load_expansion_identity_mapping, lookup_registered_identities,
        )
        self.import_identity_mapping = load_expansion_identity_mapping(
            final_evidence_dir, self.run_dir,
            lambda ids: lookup_registered_identities(self.database_url, ids),
        )
        self.compound_ids: dict[str, str] = {}
        self.pubchem_cids = self._discover_pubchem_cids()
        self.counts: Counter[str] = Counter()
        self.seen_candidate_ids: set[str] = set()
        self.seen_typed_ids: dict[str, set[str]] = {}
        self.final_attributions = self._load_final_attributions()
        self.auto_validated_rows = self._load_auto_validated_rows()
        self.serving_ids = set(self.auto_validated_rows)
        self.serving_chunk_ids = {
            str(row["chunk_id"])
            for row in self.auto_validated_rows.values()
            if row.get("chunk_id")
        }
        self.linked_raw_asset_ids, self.linked_parsed_document_ids = self._load_linked_document_ids()
        self.raw_inventory_by_id = self._load_raw_inventory_by_id()

    def run(self) -> dict[str, Any]:
        self._require_files()
        self._exec_sql(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))
        self._seed_source_registry()
        if self.replace:
            self._replace_generated_rows()
        self._import_compounds()
        self._import_raw_documents()
        self._refresh_source_snapshot_metadata()
        self._import_parsed_documents()
        self._import_evidence()
        if self.final_evidence_dir is not None:
            missing_serving_ids = self.serving_ids - self.seen_candidate_ids
            if missing_serving_ids:
                raise RuntimeError(
                    "Final auto-validated evidence is missing from candidate input: "
                    f"{sorted(missing_serving_ids)[:5]}"
                )
            self._import_final_evidence_metadata()
            self._import_graph_relationships()
            self._reconcile_serving_statuses()
        self._analyze()
        summary = self._summary()
        if self.final_evidence_dir is not None:
            self._write_database_import_summary(summary)
        return summary

    def _require_files(self) -> None:
        for name in ["raw_inventory.jsonl", "parsed_documents.jsonl", "evidence_candidates.jsonl"]:
            path = self.paths.path(name)
            if not path.exists():
                raise FileNotFoundError(path)
        if self.final_evidence_dir is not None:
            for relative_path in [
                "00_run_manifest/pipeline_state.json",
                "01_compound_attribution/compound_identity.jsonl",
                "01_compound_attribution/compound_attribution.jsonl",
                "03_deduplication/deterministic_duplicate_groups.jsonl",
                "03_deduplication/deterministic_duplicate_members.jsonl",
                "03_deduplication/semantic_duplicate_groups.jsonl",
                "03_deduplication/semantic_duplicate_members.jsonl",
                "04_conflict_analysis/evidence_conflict_report.jsonl",
                "04_conflict_analysis/conflict_members.jsonl",
                "05_evidence_layers/reviewable_evidence.jsonl",
                "05_evidence_layers/excluded_from_reviewable.jsonl",
                "06_rag_index/retrieval_records.jsonl",
                "06_rag_index/retrieval_embeddings.npy",
                "06_rag_index/rag_index_summary.json",
                "07_llm_quality_validation/final_auto_validated_evidence.jsonl",
                "07_llm_quality_validation/final_quarantined_evidence.jsonl",
                "07_llm_quality_validation/quality_summary.json",
            ]:
                path = self.final_evidence_dir / relative_path
                if not path.exists():
                    raise FileNotFoundError(path)

    def _replace_generated_rows(self) -> None:
        self._exec_sql(
            """
            TRUNCATE TABLE
                curation_event,
                enzyme_transporter_relationship,
                protein_target_identity,
                rag_retrieval_record,
                evidence_conflict_member,
                evidence_conflict_group,
                evidence_layer_membership,
                evidence_duplicate_member,
                evidence_duplicate_group,
                evidence_compound_attribution,
                processed_evidence_candidate,
                model_parameter_evidence,
                public_pbpk_model_registry,
                tissue_expression_evidence,
                ddi_evidence,
                special_population_evidence,
                clinical_pk_evidence,
                adme_parameter_evidence,
                physicochemical_property_evidence,
                document_chunk,
                parsed_document,
                raw_document,
                compound_identity
            RESTART IDENTITY CASCADE;
            """
        )

    def _seed_source_registry(self) -> None:
        source_ids = [entry.source_id for entry in load_source_registry()]
        engine = create_db_engine(self.database_url)
        with engine.begin() as connection:
            upsert_source_registry_entries(connection, source_ids=source_ids)
        self.counts["source_registry_seeded"] = len(source_ids)

    def _load_final_attributions(self) -> dict[str, dict[str, Any]]:
        if self.final_evidence_dir is None:
            return {}
        path = self.final_evidence_dir / "01_compound_attribution" / "compound_attribution.jsonl"
        if not path.exists():
            return {}
        return {str(row["evidence_id"]): self._map_identity_row(row) for row in read_jsonl(path)}

    def _load_auto_validated_rows(self) -> dict[str, dict[str, Any]]:
        if self.final_evidence_dir is None:
            return {}
        output_dir = self.final_evidence_dir / "07_llm_quality_validation"
        evidence_path = output_dir / "final_auto_validated_evidence.jsonl"
        summary_path = output_dir / "quality_summary.json"
        if not evidence_path.exists() or not summary_path.exists():
            return {}
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("accepted") is not True:
            raise RuntimeError("Final LLM quality validation was not accepted.")
        rows: dict[str, dict[str, Any]] = {}
        required_fields = (
            "evidence_id",
            "source_id",
            "source_location",
            "source_text",
            "checksum",
            "raw_asset_id",
        )
        for row in read_jsonl(evidence_path):
            row = self._map_identity_row(row)
            for field in required_fields:
                required(row, field)
            if row.get("quality_status") != "auto_validated":
                raise RuntimeError(
                    f"Non-auto-validated row in serving evidence: {row.get('evidence_id')}"
                )
            if not (row.get("attributed_compound_id") or row.get("compound_id")):
                raise RuntimeError(
                    f"Serving evidence lacks compound attribution: {row.get('evidence_id')}"
                )
            evidence_id = str(row["evidence_id"])
            if evidence_id in rows:
                raise RuntimeError(f"Duplicate serving evidence ID: {evidence_id}")
            rows[evidence_id] = row
        expected = summary.get("final_auto_validated_count")
        if type(expected) is not int or expected < 0:
            raise RuntimeError("Final quality summary lacks a valid serving evidence count.")
        if expected != len(rows):
            raise RuntimeError(
                f"Serving evidence count mismatch: {len(rows)} != {expected}"
            )
        return rows

    def _load_linked_document_ids(self) -> tuple[set[str], set[str]]:
        raw_ids: set[str] = set()
        parsed_ids: set[str] = set()
        candidates_path = self.paths.path("evidence_candidates.jsonl")
        if not candidates_path.exists():
            return raw_ids, parsed_ids
        source_rows: Iterable[dict[str, Any]]
        if self.auto_validated_rows:
            source_rows = self.auto_validated_rows.values()
        else:
            source_rows = read_jsonl(candidates_path)
        for index, row in enumerate(source_rows, start=1):
            if self.limit is not None and index > self.limit:
                break
            if row.get("raw_asset_id"):
                raw_ids.add(str(row["raw_asset_id"]))
            if row.get("parsed_document_id"):
                parsed_ids.add(str(row["parsed_document_id"]))
            metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
            for key in ("model_raw_asset_id", "parameter_file_raw_asset_id"):
                if metadata.get(key):
                    raw_ids.add(str(metadata[key]))
            for key in ("evaluation_report_raw_asset_ids", "parameter_file_raw_asset_ids"):
                for raw_asset_id in metadata.get(key) or []:
                    if raw_asset_id:
                        raw_ids.add(str(raw_asset_id))
        if self.final_evidence_dir is not None:
            rag_path = self.final_evidence_dir / "06_rag_index" / "retrieval_records.jsonl"
            if rag_path.exists():
                for row in read_jsonl(rag_path):
                    evidence_ids = {
                        str(value) for value in row.get("evidence_ids") or [] if value
                    }
                    chunk_ids = {
                        str(value) for value in row.get("chunk_ids") or [] if value
                    }
                    if self.serving_ids and not (
                        evidence_ids & self.serving_ids
                        or chunk_ids & self.serving_chunk_ids
                    ):
                        continue
                    raw_ids.update(str(value) for value in row.get("raw_asset_ids") or [] if value)
                    parsed_ids.update(str(value) for value in row.get("parsed_document_ids") or [] if value)
        return raw_ids, parsed_ids

    def _load_raw_inventory_by_id(self) -> dict[str, dict[str, Any]]:
        path = self.paths.path("raw_inventory.jsonl")
        if not path.exists():
            return {}
        return {
            str(row["raw_asset_id"]): row
            for row in read_jsonl(path)
            if row.get("raw_asset_id") and (not self.linked_raw_asset_ids or str(row["raw_asset_id"]) in self.linked_raw_asset_ids)
        }

    def _import_compounds(self) -> None:
        if self.final_evidence_dir is not None:
            identity_path = self.final_evidence_dir / "01_compound_attribution" / "compound_identity.jsonl"
            identities = list(read_jsonl(identity_path))
            for identity in identities:
                bound = self._map_identity_row(identity)
                self.compound_ids[str(identity["preferred_name"])] = str(bound["compound_id"])
            # A verified alias attaches new evidence to an existing registry row;
            # never replace that row's preferred name or provenance with an alias.
            alias_ids = getattr(self, "import_identity_mapping", None)
            alias_ids = alias_ids.ids if alias_ids else {}
            self._copy_rows(
                "compound_identity",
                COMPOUND_COLUMNS,
                (self._final_compound_row(identity) for identity in identities
                 if str(identity["compound_id"]) not in alias_ids),
            )
            if self.native_database_url:
                import psycopg
                with psycopg.connect(psycopg_dsn(self.native_database_url)) as connection:
                    self.counts["compound_identity:enriched"] = enrich_bound_compound_identities(
                        connection, identities, self.run_dir.name,
                    )
            self.counts["compound_identity"] = len(identities)
            return
        compounds = sorted(self._compound_names())
        for compound in compounds:
            self.compound_ids[compound] = str(uuid5(NAMESPACE_URL, f"pbpk:compound:{compound.lower()}"))
        self._copy_rows(
            "compound_identity",
            COMPOUND_COLUMNS,
            (
                {
                    "compound_id": self.compound_ids[compound],
                    "preferred_name": compound,
                    "normalized_name": compound.lower(),
                    "synonyms": "{}",
                    "pubchem_cid": self.pubchem_cids.get(compound.lower()),
                    "chembl_id": None,
                    "drugcentral_id": None,
                    "rxnorm_rxcui": None,
                    "inchi_key": None,
                    "source_id": "pubchem" if self.pubchem_cids.get(compound.lower()) else None,
                    "source_location": self.paths.relative_path("evidence_candidates.jsonl"),
                    "confidence_score": "0.900",
                    "curation_status": "needs_review",
                }
                for compound in compounds
            ),
        )
        self.counts["compound_identity"] = len(compounds)

    def _final_compound_row(self, identity: dict[str, Any]) -> dict[str, Any]:
        provenance = identity.get("identity_provenance") if isinstance(identity.get("identity_provenance"), list) else []
        primary = provenance[0] if provenance else {}
        synonyms = sorted({str(value) for value in identity.get("synonyms") or [] if value}, key=str.casefold)
        return {
            "compound_id": required(identity, "compound_id"),
            "preferred_name": required(identity, "preferred_name"),
            "normalized_name": identity.get("normalized_name") or str(identity["preferred_name"]).lower(),
            "synonyms": postgres_text_array(synonyms),
            "pubchem_cid": identity.get("pubchem_cid"),
            "chembl_id": identity.get("chembl_id"),
            "drugcentral_id": identity.get("drugcentral_id"),
            "rxnorm_rxcui": identity.get("rxnorm_rxcui"),
            "inchi_key": identity.get("inchikey") or identity.get("inchi_key"),
            "source_id": primary.get("source_id") or ("pubchem" if identity.get("pubchem_cid") else None),
            "source_location": primary.get("source_location") or "01_compound_attribution/compound_identity.jsonl",
            "confidence_score": "0.990" if identity.get("identity_status") == "resolved" else "0.900",
            "curation_status": (
                "auto_validated"
                if identity.get("identity_status") == "resolved"
                else "needs_review"
            ),
        }

    def _import_raw_documents(self) -> None:
        def rows() -> Iterable[dict[str, Any]]:
            for row in read_jsonl(self.paths.path("raw_inventory.jsonl")):
                raw_asset_id = required(row, "raw_asset_id")
                if self.final_evidence_dir is not None and raw_asset_id not in self.linked_raw_asset_ids:
                    continue
                path = str(row.get("storage_path") or "")
                metadata = dict(row.get("asset_metadata_json") or {})
                metadata.update(
                    {
                        "raw_asset_id": raw_asset_id,
                        "compound": row.get("compound"),
                        "group": row.get("group"),
                        "downloaded_at": row.get("downloaded_at"),
                        "request_id": row.get("request_id"),
                        "request_key": row.get("request_key"),
                    }
                )
                self.counts["raw_document"] += 1
                yield {
                    "raw_document_id": raw_document_uuid(raw_asset_id),
                    "source_id": required(row, "source_id"),
                    "source_url": row.get("source_url") or path,
                    "source_record_id": raw_asset_id,
                    "source_version": None,
                    "raw_storage_path": path,
                    "raw_mime_type": row.get("content_type"),
                    "raw_checksum": row.get("checksum") or "unknown",
                    "raw_size_bytes": row.get("file_size"),
                    "metadata": json_dump(metadata),
                }

        self._copy_rows("raw_document", RAW_COLUMNS, rows())

    def _refresh_source_snapshot_metadata(self) -> None:
        self._exec_sql(
            """
            WITH source_aliases(child_source_id, parent_source_id) AS (
                VALUES
                    ('pmc_oa'::text, 'pubmed'::text),
                    ('openfda_drug_label'::text, 'drugs_fda'::text)
            ),
            source_documents AS (
                SELECT
                    COALESCE(alias.parent_source_id, raw.source_id) AS source_id,
                    COALESCE(
                        MAX(NULLIF(raw.metadata->>'snapshot_id', '')),
                        MAX(NULLIF(
                            substring(
                                raw.raw_storage_path
                                FROM '(run_[0-9]{8}T[0-9]{6}Z)'
                            ),
                            ''
                        )),
                        MAX(NULLIF(raw.metadata->>'downloaded_at', ''))
                    ) AS source_version,
                    encode(
                        digest(
                            string_agg(
                                raw.raw_checksum,
                                '|' ORDER BY raw.raw_checksum, raw.raw_document_id
                            ),
                            'sha256'
                        ),
                        'hex'
                    ) AS checksum
                FROM raw_document raw
                LEFT JOIN source_aliases alias
                    ON alias.child_source_id = raw.source_id
                WHERE NULLIF(raw.raw_checksum, '') IS NOT NULL
                  AND raw.raw_checksum <> 'unknown'
                GROUP BY COALESCE(alias.parent_source_id, raw.source_id)
            )
            UPDATE source_registry source
            SET source_version = COALESCE(documents.source_version, source.source_version),
                checksum = documents.checksum,
                updated_at = now()
            FROM source_documents documents
            WHERE documents.source_id = source.source_id;
            """
        )

    def _import_parsed_documents(self) -> None:
        def rows() -> Iterable[dict[str, Any]]:
            parsed_path = self.paths.path("parsed_documents.jsonl")
            for row in read_jsonl(parsed_path):
                parsed_id = row.get("parsed_document_id")
                raw_asset_id = row.get("raw_asset_id")
                if not parsed_id or not raw_asset_id:
                    continue
                if self.final_evidence_dir is not None and str(parsed_id) not in self.linked_parsed_document_ids:
                    continue
                record = {
                    "run_id": row.get("run_id"),
                    "compound": row.get("compound"),
                    "content_kind": row.get("content_kind"),
                    "section_title": row.get("section_title"),
                    "metadata_json": row.get("metadata_json") or {},
                    "text_hash": row.get("text_hash"),
                    "text_preview": str(row.get("text") or "")[:1000],
                }
                self.counts["parsed_document"] += 1
                yield {
                    "parsed_document_id": parsed_id,
                    "raw_document_id": raw_document_uuid(str(raw_asset_id)),
                    "source_id": required(row, "source_id"),
                    "parser_name": row.get("parser_name") or "raw_processing_pipeline",
                    "parser_version": row.get("parser_version") or "raw-processing-v1",
                    "document_type": row.get("content_kind"),
                    "title": row.get("title"),
                    "source_location": required(row, "source_location"),
                    "parsed_storage_path": str(parsed_path),
                    "parsed_record": json_dump(record),
                    "parsed_checksum": row.get("text_hash"),
                }

        self._copy_rows("parsed_document", PARSED_COLUMNS, rows())

    def _import_evidence(self) -> None:
        writers: dict[str, CopyWriter] = {}
        try:
            for table, columns in (
                ("processed_evidence_candidate", CANDIDATE_COLUMNS),
                ("physicochemical_property_evidence", PHYS_COLUMNS),
                ("adme_parameter_evidence", ADME_COLUMNS),
                ("clinical_pk_evidence", CLINICAL_COLUMNS),
                ("ddi_evidence", DDI_COLUMNS),
                ("special_population_evidence", SPECIAL_COLUMNS),
                ("tissue_expression_evidence", TISSUE_COLUMNS),
                ("public_pbpk_model_registry", MODEL_COLUMNS),
            ):
                writers[table] = CopyWriter(self, table, columns)
            for index, row in enumerate(read_jsonl(self.paths.path("evidence_candidates.jsonl")), start=1):
                if self.limit is not None and index > self.limit:
                    break
                evidence_id = required(row, "evidence_id")
                if self.final_evidence_dir is not None and evidence_id not in self.serving_ids:
                    self.counts["candidate_skipped_not_auto_validated"] += 1
                    continue
                serving_row = self.auto_validated_rows.get(evidence_id)
                normalized = self._normalize_evidence_row(
                    {**row, **serving_row} if serving_row else row
                )
                evidence_id = required(normalized, "evidence_id")
                if serving_row:
                    normalized["curation_status"] = "auto_validated"
                if evidence_id in self.seen_candidate_ids:
                    self.counts["duplicate_evidence_id_skipped"] += 1
                    continue
                self.seen_candidate_ids.add(evidence_id)
                writers["processed_evidence_candidate"].write(self._candidate_row(normalized))
                self.counts["processed_evidence_candidate"] += 1
                if self.final_evidence_dir is not None and evidence_id not in self.serving_ids:
                    self.counts["typed_evidence_skipped_not_auto_validated"] += 1
                    continue
                table_row = self._typed_row(normalized)
                if table_row is None:
                    self.counts[f"unsupported:{normalized['evidence_type']}"] += 1
                    continue
                table_name, payload = table_row
                seen_for_table = self.seen_typed_ids.setdefault(table_name, set())
                typed_id = str(payload.get("evidence_id") or payload.get("model_id") or evidence_id)
                if typed_id in seen_for_table:
                    self.counts[f"duplicate_typed_id_skipped:{table_name}"] += 1
                    continue
                seen_for_table.add(typed_id)
                writers[table_name].write(payload)
                self.counts[table_name] += 1
        except BaseException as exc:
            for writer in writers.values():
                try:
                    writer.close(error=exc)
                except BaseException:
                    # Abort every open table and preserve the source failure.
                    pass
            raise
        else:
            pending = list(writers.values())
            try:
                while pending:
                    pending.pop(0).close()
            except BaseException as exc:
                for writer in pending:
                    try:
                        writer.close(error=exc)
                    except BaseException:
                        pass
                raise

    def _normalize_evidence_row(self, row: dict[str, Any]) -> dict[str, Any]:
        attribution = self.final_attributions.get(str(row.get("evidence_id") or ""))
        compound = attribution.get("compound_name") if attribution else row.get("compound")
        compound_name = str(compound).strip() if compound not in (None, "", "shared") else None
        compound_id = str(attribution.get("compound_id")) if attribution and attribution.get("compound_id") else self.compound_ids.get(compound_name)
        raw_asset_id = str(row.get("raw_asset_id") or "")
        curation_status = str(row.get("curation_status") or "needs_review")
        if curation_status not in CURATION_STATUS:
            curation_status = "needs_review"
        confidence = clamp01(row.get("extraction_confidence"))
        return self._map_identity_row({
            **row,
            "compound_name": compound_name,
            "compound_id": compound_id,
            "raw_document_id": raw_document_uuid(raw_asset_id) if raw_asset_id else None,
            "parsed_document_id": row.get("parsed_document_id") or None,
            "parsed_table_id": row.get("parsed_table_id") or None,
            "chunk_id": row.get("chunk_id") or None,
            "curation_status": curation_status,
            "extraction_confidence": confidence,
            "value": numeric(row.get("value")),
            "canonical_value": numeric(row.get("canonical_value")),
            "metadata_json": row.get("metadata_json") or {},
            "parameter": text_or_default(row.get("parameter"), fallback=row.get("evidence_type") or "candidate"),
            "source_text": required(row, "source_text"),
            "source_location": required(row, "source_location"),
            "checksum": row.get("checksum") or "unknown",
            "storage_path": required(row, "storage_path"),
            "source_id": required(row, "source_id"),
            "source_url": row.get("source_url"),
            "compound_role": attribution.get("compound_role") if attribution else "target_compound",
            "attribution": attribution or {},
        })

    def _map_identity_row(self, row: dict[str, Any]) -> dict[str, Any]:
        mapping = getattr(self, "import_identity_mapping", None)
        return mapping.record(row) if mapping is not None else row

    def _candidate_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "evidence_id": required(row, "evidence_id"),
            "run_id": required(row, "run_id"),
            "evidence_type": required(row, "evidence_type"),
            "compound_name": row["compound_name"],
            "compound_id": row["compound_id"],
            "source_id": row["source_id"],
            "raw_asset_id": required(row, "raw_asset_id"),
            "raw_document_id": row["raw_document_id"],
            "parsed_document_id": row["parsed_document_id"],
            "parsed_table_id": row["parsed_table_id"],
            "chunk_id": row["chunk_id"],
            "storage_path": row["storage_path"],
            "source_url": row["source_url"],
            "checksum": row["checksum"],
            "source_location": row["source_location"],
            "source_text": row["source_text"],
            "source_text_hash": row.get("source_text_hash"),
            "extraction_method": row.get("extraction_method") or "machine_extracted",
            "extractor_name": row.get("extractor_name"),
            "extractor_version": row.get("extractor_version"),
            "extraction_confidence": row["extraction_confidence"],
            "curation_status": row["curation_status"],
            "parameter": row["parameter"],
            "value": row["value"],
            "value_text": row.get("value_text"),
            "unit": row.get("unit"),
            "canonical_value": row["canonical_value"],
            "canonical_unit": row.get("canonical_unit"),
            "condition": row.get("condition"),
            "population": row.get("population"),
            "route": row.get("route"),
            "dose": row.get("dose"),
            "metadata_json": json_dump(row["metadata_json"]),
            "source_record": json_dump(row),
        }

    def _typed_row(self, row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        evidence_type = str(row.get("evidence_type") or "")
        common = common_evidence_fields(row)
        if evidence_type == "physicochemical":
            return "physicochemical_property_evidence", {
                **common,
                "compound_id": row["compound_id"],
                "property_name": row["parameter"],
                "property_value": row["value"],
                "property_value_text": row.get("value_text"),
                "unit": row.get("unit"),
                "canonical_value": row["canonical_value"],
                "canonical_unit": row.get("canonical_unit"),
            }
        if evidence_type == "adme":
            parameter = row["parameter"]
            return "adme_parameter_evidence", {
                **common,
                "compound_id": row["compound_id"],
                "parameter_name": parameter,
                "value": row["value"],
                "value_text": row.get("value_text"),
                "unit": row.get("unit"),
                "canonical_value": row["canonical_value"],
                "canonical_unit": row.get("canonical_unit"),
                "enzyme": infer_enzyme(parameter, row["source_text"]),
                "transporter": infer_transporter(parameter, row["source_text"]),
            }
        if evidence_type == "clinical_pk":
            text_blob = f"{row.get('condition') or ''} {row['source_text']}"
            return "clinical_pk_evidence", {
                **common,
                "compound_id": row["compound_id"],
                "study_context": row.get("condition"),
                "population": row.get("population"),
                "dose": row.get("dose"),
                "route": row.get("route"),
                "pk_parameter": row["parameter"],
                "value": row["value"],
                "value_text": row.get("value_text"),
                "unit": row.get("unit"),
                "canonical_value": row["canonical_value"],
                "canonical_unit": row.get("canonical_unit"),
                "renal_impairment_flag": bool_text("renal impairment" in text_blob.lower()),
                "hepatic_impairment_flag": bool_text("hepatic impairment" in text_blob.lower()),
            }
        if evidence_type == "ddi":
            parameter = row["parameter"]
            value = row["value"]
            parameter_lower = parameter.lower()
            compound_role = str(row.get("compound_role") or "target_compound")
            return "ddi_evidence", {
                **common,
                "victim_compound_id": row["compound_id"] if compound_role in {"target_compound", "victim_drug"} else None,
                "perpetrator_compound_id": row["compound_id"] if compound_role == "perpetrator_drug" else None,
                "enzyme_or_transporter": infer_enzyme(parameter, row["source_text"]) or infer_transporter(parameter, row["source_text"]),
                "interaction_type": parameter or "drug_interaction_candidate",
                "study_type": row.get("condition"),
                "auc_ratio": value if "auc" in parameter_lower else None,
                "cmax_ratio": value if "cmax" in parameter_lower else None,
                "dose_victim": row.get("dose"),
                "dose_perpetrator": None,
                "population": row.get("population"),
            }
        if evidence_type == "special_population":
            parameter = row["parameter"]
            return "special_population_evidence", {
                **common,
                "compound_id": row["compound_id"],
                "population_type": population_type(parameter, row.get("condition"), row["source_text"]),
                "population_definition": row.get("condition") or row["source_text"][:500],
                "covariate": parameter,
                "pk_parameter": pk_parameter(parameter, row["source_text"]),
                "value": row["value"],
                "value_text": row.get("value_text"),
                "unit": row.get("unit"),
                "canonical_value": row["canonical_value"],
                "canonical_unit": row.get("canonical_unit"),
                "ratio_to_reference": row["value"] if (row.get("unit") == "ratio" or "ratio" in parameter.lower()) else None,
            }
        if evidence_type == "tissue_expression":
            return "tissue_expression_evidence", {
                **common,
                "evidence_id": required(row, "evidence_id"),
                "protein_target_id": None,
                "tissue": text_or_default(row.get("parameter"), "unknown"),
                "expression_value": row["value"],
                "expression_value_text": row.get("value_text"),
                "unit": row.get("unit"),
                "canonical_value": row["canonical_value"],
                "canonical_unit": row.get("canonical_unit"),
                "species": "human" if "human" in row["source_text"].lower() else None,
                "assay_method": row.get("condition"),
            }
        if evidence_type == "public_pbpk_model":
            metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
            model_name = text_or_default(
                metadata.get("model_name") or row.get("parameter"),
                (row["source_text"][:120] or "public PBPK model candidate"),
            )
            model_file_path = metadata.get("model_storage_path") or metadata.get("model_file_path") or row.get("model_file_path")
            parameter_file_path = self._asset_storage_path(metadata.get("parameter_file_raw_asset_id")) or metadata.get("parameter_file_path")
            evaluation_paths = [
                self._asset_storage_path(raw_asset_id)
                for raw_asset_id in metadata.get("evaluation_report_raw_asset_ids") or []
            ]
            evaluation_paths = [path for path in evaluation_paths if path]
            evaluation_report_path = evaluation_paths[0] if evaluation_paths else None
            has_model_artifact = bool(model_file_path)
            if not has_model_artifact:
                return None
            validation_status = "downloadable_public_model" if has_model_artifact else "model_mention_only"
            return "public_pbpk_model_registry", {
                "model_id": required(row, "evidence_id"),
                "model_name": model_name[:500],
                "compound_id": row["compound_id"],
                "model_type": "PBPK",
                "software": metadata.get("software") or infer_software(row["source_text"]),
                "species": "human" if "human" in row["source_text"].lower() else None,
                "population": row.get("population"),
                "route": row.get("route"),
                "model_source": row["source_url"],
                "source_id": row["source_id"],
                "model_file_path": model_file_path,
                "parameter_file_path": parameter_file_path,
                "evaluation_report_path": evaluation_report_path,
                "publication_id": None,
                "validation_status": validation_status,
                "curation_status": row["curation_status"],
                "source_location": row["source_location"],
                "notes": json_dump(
                    {
                        "source_text": row["source_text"],
                        "model_checksum": metadata.get("model_checksum"),
                        "model_source_url": metadata.get("model_source_url"),
                        "evaluation_report_paths": evaluation_paths or metadata.get("evaluation_report_paths") or [],
                        "model_reproduced": bool(metadata.get("model_reproduced")),
                    }
                ),
            }
        return None

    def _asset_storage_path(self, raw_asset_id: Any) -> str | None:
        if not raw_asset_id:
            return None
        row = self.raw_inventory_by_id.get(str(raw_asset_id)) or {}
        path = row.get("storage_path")
        return str(path) if path else None

    def _import_final_evidence_metadata(self) -> None:
        assert self.final_evidence_dir is not None
        self._import_attributions()
        self._import_duplicate_groups()
        self._import_conflicts()
        self._import_layer_memberships()
        self._import_rag_index()

    def _import_graph_relationships(self) -> None:
        assert self.final_evidence_dir is not None
        serving_path = (
            self.final_evidence_dir
            / "07_llm_quality_validation"
            / "final_auto_validated_evidence.jsonl"
        )
        target_rows: dict[str, dict[str, Any]] = {}
        relationship_rows: dict[tuple[str, str, str], dict[str, Any]] = {}

        for evidence in read_jsonl(serving_path):
            evidence = self._map_identity_row(evidence)
            evidence_id = required(evidence, "evidence_id")
            compound_id = evidence.get("attributed_compound_id") or evidence.get("compound_id")
            if not compound_id:
                continue
            for fact in extract_graph_facts(evidence):
                target_id = str(uuid5(NAMESPACE_URL, f"pbpk:protein_target:{fact.target.key}"))
                target = target_rows.setdefault(
                    fact.target.key,
                    {
                        "protein_target_id": target_id,
                        "preferred_name": fact.target.preferred_name,
                        "gene_symbol": fact.target.gene_symbol,
                        "uniprot_id": None,
                        "target_type": fact.target.target_type,
                        "synonyms": set(),
                        "source_id": required(evidence, "source_id"),
                        "source_location": required(evidence, "source_location"),
                        "confidence_score": "0.900",
                        "curation_status": "auto_validated",
                    },
                )
                target["synonyms"].update({fact.target.key, fact.target.preferred_name, fact.target.matched_text})

                relationship_key = (evidence_id, fact.target.key, fact.relationship_type)
                relationship_rows.setdefault(
                    relationship_key,
                    {
                        "relationship_id": str(
                            uuid5(
                                NAMESPACE_URL,
                                f"pbpk:final_graph:{evidence_id}:{fact.target.key}:{fact.relationship_type}",
                            )
                        ),
                        "compound_id": compound_id,
                        "protein_target_id": target_id,
                        "relationship_type": fact.relationship_type,
                        "evidence_direction": "compound_to_target",
                        "source_id": required(evidence, "source_id"),
                        "raw_document_id": raw_document_uuid(required(evidence, "raw_asset_id")),
                        "parsed_document_id": evidence.get("parsed_document_id"),
                        "processed_evidence_id": evidence_id,
                        "source_location": required(evidence, "source_location"),
                        "evidence_text": fact.evidence_sentence,
                        "value_origin": "qualitative",
                        "evidence_origin": "final_auto_validated_evidence",
                        "extraction_method": "rule_extracted",
                        "extractor_version": "final-evidence-graph-v1",
                        "confidence_score": f"{fact.confidence:.3f}",
                        "curation_status": "auto_validated",
                    },
                )

        def targets() -> Iterable[dict[str, Any]]:
            for key in sorted(target_rows):
                row = dict(target_rows[key])
                row["synonyms"] = postgres_text_array(sorted(row["synonyms"], key=str.lower))
                self.counts["protein_target_identity"] += 1
                yield row

        def relationships() -> Iterable[dict[str, Any]]:
            for key in sorted(relationship_rows):
                self.counts["enzyme_transporter_relationship"] += 1
                yield relationship_rows[key]

        self._copy_rows("protein_target_identity", TARGET_COLUMNS, targets())
        self._copy_rows("enzyme_transporter_relationship", RELATIONSHIP_COLUMNS, relationships())

    def _import_attributions(self) -> None:
        assert self.final_evidence_dir is not None
        path = self.final_evidence_dir / "01_compound_attribution" / "compound_attribution.jsonl"

        def rows() -> Iterable[dict[str, Any]]:
            for row in read_jsonl(path):
                evidence_id = required(row, "evidence_id")
                if self.serving_ids and evidence_id not in self.serving_ids:
                    continue
                compound_id = row.get("compound_id")
                compound_role = required(row, "compound_role")
                attribution_id = str(uuid5(NAMESPACE_URL, f"pbpk:attribution:{evidence_id}:{compound_id}:{compound_role}"))
                self.counts["evidence_compound_attribution"] += 1
                yield {
                    "attribution_id": attribution_id,
                    "evidence_id": evidence_id,
                    "compound_id": compound_id,
                    "compound_name": row.get("compound_name"),
                    "compound_role": compound_role,
                    "attribution_method": row.get("attribution_method") or "unknown",
                    "attribution_confidence": clamp01(row.get("attribution_confidence")),
                    "matched_text": row.get("matched_text"),
                    "matched_synonym": row.get("matched_synonym"),
                    "attribution_reason": row.get("attribution_reason"),
                    "llm_required": bool(row.get("llm_required")),
                    "llm_trigger_reason": row.get("llm_trigger_reason"),
                    "attribution_stage": "llm_assisted" if row.get("attribution_stage") == "llm" else "rule",
                    "llm_payload": json_dump(row if row.get("attribution_stage") == "llm" else {}),
                }

        self._copy_rows("evidence_compound_attribution", ATTRIBUTION_COLUMNS, rows())

    def _import_duplicate_groups(self) -> None:
        assert self.final_evidence_dir is not None
        dedup_dir = self.final_evidence_dir / "03_deduplication"
        group_paths = [
            dedup_dir / "deterministic_duplicate_groups.jsonl",
            dedup_dir / "semantic_duplicate_groups.jsonl",
        ]
        member_paths = [
            dedup_dir / "deterministic_duplicate_members.jsonl",
            dedup_dir / "semantic_duplicate_members.jsonl",
        ]
        serving_members: dict[str, list[dict[str, Any]]] = {}
        for path in member_paths:
            for row in read_jsonl(path):
                evidence_id = required(row, "evidence_id")
                if self.serving_ids and evidence_id not in self.serving_ids:
                    continue
                group_id = required(row, "duplicate_group_id")
                serving_members.setdefault(group_id, []).append(row)
        eligible_group_ids = {
            group_id
            for group_id, rows in serving_members.items()
            if len({str(row["evidence_id"]) for row in rows}) >= 2
        }

        def groups() -> Iterable[dict[str, Any]]:
            seen: set[str] = set()
            for path in group_paths:
                for row in read_jsonl(path):
                    group_id = required(row, "duplicate_group_id")
                    if group_id in seen or group_id not in eligible_group_ids:
                        continue
                    seen.add(group_id)
                    member_ids = sorted(
                        {
                            str(member["evidence_id"])
                            for member in serving_members[group_id]
                        }
                    )
                    representative_id = row.get("representative_evidence_id")
                    if representative_id not in member_ids:
                        representative_id = member_ids[0]
                    self.counts["evidence_duplicate_group"] += 1
                    yield {
                        "duplicate_group_id": group_id,
                        "group_type": required(row, "group_type"),
                        "group_key": required(row, "group_key"),
                        "evidence_type": row.get("evidence_type"),
                        "compound_id": row.get("compound_id"),
                        "compound_name": row.get("compound_name"),
                        "representative_evidence_id": representative_id,
                        "duplicate_count": len(member_ids),
                        "selection_reason": row.get("selection_reason") or "final_evidence_strategy",
                        "metadata_json": json_dump(row.get("metadata") or {}),
                    }

        def members() -> Iterable[dict[str, Any]]:
            seen: set[tuple[str, str]] = set()
            for path in member_paths:
                for row in read_jsonl(path):
                    key = (required(row, "duplicate_group_id"), required(row, "evidence_id"))
                    if (
                        key in seen
                        or key[0] not in eligible_group_ids
                        or (self.serving_ids and key[1] not in self.serving_ids)
                    ):
                        continue
                    seen.add(key)
                    self.counts["evidence_duplicate_member"] += 1
                    yield {
                        "duplicate_group_id": key[0],
                        "evidence_id": key[1],
                        "is_representative": bool(row.get("is_representative")),
                        "duplicate_rank": row.get("duplicate_rank"),
                        "duplicate_reason": row.get("duplicate_reason") or row.get("group_type") or "duplicate",
                        "similarity_score": numeric(row.get("similarity_score")),
                    }

        self._copy_rows("evidence_duplicate_group", DUPLICATE_GROUP_COLUMNS, groups())
        self._copy_rows("evidence_duplicate_member", DUPLICATE_MEMBER_COLUMNS, members())

    def _import_conflicts(self) -> None:
        assert self.final_evidence_dir is not None
        conflict_dir = self.final_evidence_dir / "04_conflict_analysis"
        state = json.loads((self.final_evidence_dir / "00_run_manifest" / "pipeline_state.json").read_text(encoding="utf-8"))
        run_id = str(state.get("run_id") or self.final_evidence_dir.name)
        serving_members: dict[str, list[dict[str, Any]]] = {}
        for row in read_jsonl(conflict_dir / "conflict_members.jsonl"):
            evidence_id = required(row, "evidence_id")
            if self.serving_ids and evidence_id not in self.serving_ids:
                continue
            group_id = required(row, "conflict_group_id")
            serving_members.setdefault(group_id, []).append(row)
        eligible_group_ids = {
            group_id
            for group_id, rows in serving_members.items()
            if len({str(row["evidence_id"]) for row in rows}) >= 2
        }

        def groups() -> Iterable[dict[str, Any]]:
            for row in read_jsonl(conflict_dir / "evidence_conflict_report.jsonl"):
                group_id = required(row, "conflict_group_id")
                if group_id not in eligible_group_ids:
                    continue
                self.counts["evidence_conflict_group"] += 1
                yield {
                    "conflict_group_id": group_id,
                    "run_id": run_id,
                    "source_group_id": row.get("source_group_id"),
                    "source_group_type": row.get("source_group_type"),
                    "compound_id": row.get("compound_id"),
                    "compound_name": row.get("compound_name"),
                    "evidence_type": required(row, "evidence_type"),
                    "parameter": row.get("parameter"),
                    "canonical_unit": row.get("canonical_unit"),
                    "evidence_count": len(
                        {
                            str(member["evidence_id"])
                            for member in serving_members[group_id]
                        }
                    ),
                    "analysis_status": required(row, "analysis_status"),
                    "analysis_reason": row.get("analysis_reason") or "unspecified",
                    "resolution": row.get("resolution") or "human_review",
                    "requires_human_review": bool(row.get("requires_human_review")),
                    "metadata_json": json_dump(row),
                }

        def members() -> Iterable[dict[str, Any]]:
            for row in read_jsonl(conflict_dir / "conflict_members.jsonl"):
                group_id = required(row, "conflict_group_id")
                evidence_id = required(row, "evidence_id")
                if (
                    group_id not in eligible_group_ids
                    or (self.serving_ids and evidence_id not in self.serving_ids)
                ):
                    continue
                self.counts["evidence_conflict_member"] += 1
                yield {
                    "conflict_group_id": group_id,
                    "evidence_id": evidence_id,
                    "analysis_status": required(row, "analysis_status"),
                    "requires_human_review": bool(row.get("requires_human_review")),
                    "source_group_id": row.get("source_group_id"),
                    "source_group_type": row.get("source_group_type"),
                    "source_id": row.get("source_id"),
                    "source_location": row.get("source_location"),
                }

        self._copy_rows("evidence_conflict_group", CONFLICT_GROUP_COLUMNS, groups())
        self._copy_rows("evidence_conflict_member", CONFLICT_MEMBER_COLUMNS, members())

    def _import_layer_memberships(self) -> None:
        assert self.final_evidence_dir is not None
        layers_dir = self.final_evidence_dir / "05_evidence_layers"
        layer_files = {
            "excluded": layers_dir / "excluded_from_reviewable.jsonl",
            "preselected": layers_dir / "preselected_evidence.jsonl",
            "reviewable": layers_dir / "reviewable_evidence.jsonl",
            "auto_validated": (
                self.final_evidence_dir
                / "07_llm_quality_validation"
                / "final_auto_validated_evidence.jsonl"
            ),
            "accepted": layers_dir / "accepted_evidence.jsonl",
            "modeling": layers_dir / "modeling_evidence.jsonl",
        }

        def membership(row: dict[str, Any], layer: str) -> dict[str, Any]:
            metadata = {
                key: row.get(key)
                for key in (
                    "provenance_status",
                    "compound_assignment_status",
                    "domain_gate_status",
                    "domain_classification",
                    "conflict_statuses",
                    "preselected",
                    "quality_status",
                    "validation_method",
                    "quality_checked_at",
                )
                if key in row
            }
            return {
                "evidence_id": required(row, "evidence_id"),
                "run_id": required(row, "run_id"),
                "evidence_layer": layer,
                "exclusion_reasons": json_dump(row.get("exclusion_reasons") or []),
                "review_blockers": json_dump(row.get("review_blockers") or []),
                "semantic_group_ids": json_dump(row.get("semantic_group_ids") or []),
                "conflict_group_ids": json_dump(row.get("conflict_group_ids") or []),
                "metadata_json": json_dump(metadata),
            }

        def rows() -> Iterable[dict[str, Any]]:
            for row in self.auto_validated_rows.values():
                self.counts["evidence_layer_membership:candidate"] += 1
                yield {
                    "evidence_id": required(row, "evidence_id"),
                    "run_id": required(row, "run_id"),
                    "evidence_layer": "candidate",
                    "exclusion_reasons": "[]",
                    "review_blockers": "[]",
                    "semantic_group_ids": "[]",
                    "conflict_group_ids": "[]",
                    "metadata_json": "{}",
                }
            for row in read_jsonl(layer_files["auto_validated"]):
                self.counts["evidence_layer_membership:auto_validated"] += 1
                yield membership(row, "auto_validated")

        self._copy_rows("evidence_layer_membership", LAYER_COLUMNS, rows())

    def _import_rag_index(self) -> None:
        assert self.final_evidence_dir is not None
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to import RAG embeddings") from exc
        index_dir = self.final_evidence_dir / "06_rag_index"
        records_path = index_dir / "retrieval_records.jsonl"
        embeddings_path = index_dir / "retrieval_embeddings.npy"
        summary = json.loads((index_dir / "rag_index_summary.json").read_text(encoding="utf-8"))
        if summary.get("accepted") is not True or summary.get("embedding_dimension") != 768:
            raise RuntimeError("The RAG index must be completed, accepted, and 768-dimensional before import.")
        if summary.get("records_sha256") != sha256_file(records_path):
            raise RuntimeError("RAG retrieval record checksum mismatch.")
        if summary.get("embeddings_sha256") != sha256_file(embeddings_path):
            raise RuntimeError("RAG embedding checksum mismatch.")
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if tuple(embeddings.shape) != (int(summary["record_count"]), 768):
            raise RuntimeError(f"Unexpected RAG embedding shape: {embeddings.shape}")

        def rows() -> Iterable[dict[str, Any]]:
            input_count = 0
            for record in read_jsonl(records_path):
                embedding_row = int(record["embedding_row"])
                if embedding_row != input_count:
                    raise RuntimeError(
                        f"Non-contiguous RAG embedding row at {input_count}"
                    )
                input_count += 1
                evidence_ids = {
                    str(value)
                    for value in record.get("evidence_ids") or []
                    if value
                }
                chunk_ids = {
                    str(value)
                    for value in record.get("chunk_ids") or []
                    if value
                }
                if self.serving_ids and not (
                    evidence_ids & self.serving_ids
                    or chunk_ids & self.serving_chunk_ids
                ):
                    self.counts["rag_retrieval_record_skipped_not_auto_validated"] += 1
                    continue
                serving_evidence_ids = sorted(evidence_ids & self.serving_ids)
                metadata = {
                    key: record.get(key)
                    for key in (
                        "evidence_layers",
                        "evidence_types",
                        "parameters",
                        "values",
                        "raw_asset_ids",
                        "parsed_document_ids",
                        "upstream_text_hashes",
                    )
                }
                metadata["upstream_evidence_layer"] = record.get("evidence_layer")
                metadata["serving_evidence_ids"] = serving_evidence_ids
                self.counts["rag_retrieval_record"] += 1
                yield {
                    "retrieval_record_id": required(record, "retrieval_record_id"),
                    "run_id": required(record, "run_id"),
                    "embedding_row": embedding_row,
                    "compound_id": required(record, "compound_id"),
                    "compound_name": required(record, "compound_name"),
                    "source_id": required(record, "source_id"),
                    "source_location": required(record, "source_location"),
                    "source_url": record.get("source_url"),
                    "storage_path": record.get("storage_path"),
                    "checksum": record.get("checksum"),
                    "source_text": required(record, "source_text"),
                    "source_text_hash": required(record, "source_text_hash"),
                    "embedding_text": required(record, "embedding_text"),
                    "embedding": vector_text(embeddings[embedding_row]),
                    "embedding_model": summary["embedding_model"],
                    "evidence_layer": "auto_validated",
                    "record_kinds": json_dump(record.get("record_kinds") or []),
                    "evidence_ids": json_dump(serving_evidence_ids),
                    "chunk_ids": json_dump(record.get("chunk_ids") or []),
                    "parsed_table_ids": json_dump(record.get("parsed_table_ids") or []),
                    "recovered_table_ids": json_dump(record.get("recovered_table_ids") or []),
                    "provenance_status": record.get("provenance_status") or "validated",
                    "curation_status": "auto_validated",
                    "metadata_json": json_dump(metadata),
                }
            if input_count != int(summary["record_count"]):
                raise RuntimeError(
                    f"RAG record count mismatch: "
                    f"{input_count} != {summary['record_count']}"
                )

        self._copy_rows("rag_retrieval_record", RAG_COLUMNS, rows())

    def _compound_names(self) -> set[str]:
        compounds: set[str] = set()
        for row in read_jsonl(self.paths.path("evidence_candidates.jsonl")):
            compound = row.get("compound")
            if compound and compound != "shared":
                compounds.add(str(compound))
        return compounds

    def _discover_pubchem_cids(self) -> dict[str, str]:
        cids: dict[str, str] = {}
        inventory = self.paths.path("raw_inventory.jsonl")
        if not inventory.exists():
            return cids
        for row in read_jsonl(inventory):
            if row.get("source_id") != "pubchem" or not row.get("compound"):
                continue
            path = Path(str(row.get("storage_path") or ""))
            if not path.exists() or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            cid = find_first_cid(payload)
            if cid:
                cids[str(row["compound"]).lower()] = str(cid)
        return cids

    def _copy_rows(self, table: str, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
        writer = CopyWriter(self, table, columns)
        try:
            for row in rows:
                writer.write(row)
        except BaseException as exc:
            writer.close(error=exc)
            raise
        else:
            writer.close()

    def _exec_sql(self, sql: str) -> None:
        if self.native_database_url:
            from sqlalchemy.engine import make_url

            url = make_url(self.native_database_url)
            psql = "psql"
            if os.environ.get("PBPK_PROJECT_BOUNDARY") == "1":
                psql_path = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages/pgserver/pginstall/bin/psql"
                if not psql_path.is_file():
                    raise FileNotFoundError("Project-owned PostgreSQL client is missing.")
                psql = str(psql_path)
            env = os.environ.copy()
            if url.password:
                env["PGPASSWORD"] = url.password
            subprocess.run(
                [
                    psql,
                    "--no-psqlrc",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-h",
                    url.host or "localhost",
                    "-p",
                    str(url.port or 5432),
                    "-U",
                    url.username or self.db_user,
                    "-d",
                    url.database or self.db_name,
                ],
                input=sql,
                text=True,
                check=True,
                env=env,
            )
            return
        subprocess.run(
            ["docker", "exec", "-i", self.container, "psql", "-v", "ON_ERROR_STOP=1", "-U", self.db_user, "-d", self.db_name],
            input=sql,
            text=True,
            check=True,
        )

    def _analyze(self) -> None:
        self._exec_sql(
            """
            ANALYZE compound_identity;
            ANALYZE raw_document;
            ANALYZE parsed_document;
            ANALYZE processed_evidence_candidate;
            ANALYZE physicochemical_property_evidence;
            ANALYZE adme_parameter_evidence;
            ANALYZE clinical_pk_evidence;
            ANALYZE ddi_evidence;
            ANALYZE special_population_evidence;
            ANALYZE tissue_expression_evidence;
            ANALYZE protein_target_identity;
            ANALYZE enzyme_transporter_relationship;
            ANALYZE public_pbpk_model_registry;
            ANALYZE evidence_compound_attribution;
            ANALYZE evidence_duplicate_group;
            ANALYZE evidence_duplicate_member;
            ANALYZE evidence_layer_membership;
            ANALYZE evidence_conflict_group;
            ANALYZE evidence_conflict_member;
            ANALYZE rag_retrieval_record;
            """
        )

    def _reconcile_serving_statuses(self) -> dict[str, Any]:
        engine = create_db_engine(self.database_url)
        try:
            with engine.begin() as connection:
                rows = {eid: self._normalize_evidence_row(row) for eid, row in self.auto_validated_rows.items()}
                result = reconcile_auto_validated_statuses(connection, rows)
            self.counts["processed_evidence_candidate:status_reconciled"] += len(result["promoted_ids"])
            self.counts["serving_evidence_verified"] = result["verified_count"]
            return result
        finally:
            engine.dispose()

    def _summary(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "final_evidence_dir": str(self.final_evidence_dir) if self.final_evidence_dir else None,
            "replace": self.replace,
            "linked_only": self.final_evidence_dir is not None,
            "serving_evidence_layer": (
                "auto_validated" if self.final_evidence_dir is not None else None
            ),
            "serving_evidence_count": len(self.serving_ids),
            "counts": dict(sorted(self.counts.items())),
            "identity_mapping": getattr(self, "import_identity_mapping", None).receipt
            if getattr(self, "import_identity_mapping", None) is not None
            else {"version": 1, "mappings": []},
        }

    def _write_database_import_summary(self, summary: dict[str, Any]) -> None:
        assert self.final_evidence_dir is not None
        manifest_dir = self.final_evidence_dir / "00_run_manifest"
        summary_path = manifest_dir / "database_import_summary.json"
        payload = {
            **summary,
            "generated_at": datetime_utc_now(),
            "accepted": True,
        }
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        state_path = manifest_dir / "pipeline_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["updated_at"] = payload["generated_at"]
        state["current_stage"] = "database_ready"
        state.setdefault("stages", {})["database_import"] = {
            "status": "completed",
            "accepted": True,
            "candidate_count": self.counts.get("processed_evidence_candidate", 0),
            "reviewable_count": self.counts.get("evidence_layer_membership:reviewable", 0),
            "auto_validated_count": self.counts.get(
                "evidence_layer_membership:auto_validated",
                0,
            ),
            "rag_record_count": self.counts.get("rag_retrieval_record", 0),
            "summary_path": "00_run_manifest/database_import_summary.json",
        }
        state["stages"]["database_ui_export"] = {"status": "in_progress"}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def reconcile_auto_validated_statuses(connection, incoming_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Upgrade only identical machine-reviewed facts; never overwrite human decisions.

    A raw asset gets a new acquisition ID on each run. Its identical checksum,
    quote, scientific fields and attributed role provide the cross-run identity.
    Source records and provenance links remain unchanged.
    """
    from decimal import Decimal
    from sqlalchemy import bindparam, text

    def selected(query, ids):
        return connection.execute(text(query).bindparams(bindparam("ids", expanding=True)), {"ids": sorted(ids)})

    ids = set(incoming_rows)
    if not ids:
        return {"verified_count": 0, "promoted_ids": []}
    available = set(selected("SELECT evidence_id::text FROM auto_validated_evidence WHERE evidence_id::text IN :ids", ids).scalars())
    missing = ids - available
    if not missing:
        return {"verified_count": len(ids), "promoted_ids": []}
    existing = {str(row["evidence_id"]): dict(row) for row in selected(
        "SELECT * FROM processed_evidence_candidate WHERE evidence_id::text IN :ids FOR UPDATE", missing).mappings()}
    curated = set(selected("SELECT evidence_id::text FROM curation_event WHERE evidence_id::text IN :ids", missing).scalars())
    proven = {(str(row[0]), str(row[1])) for row in selected(
        "SELECT evidence_id::text, run_id FROM evidence_layer_membership WHERE evidence_layer='auto_validated' AND evidence_id::text IN :ids", missing)}
    roles = {(str(row[0]), str(row[1]), str(row[2])) for row in selected(
        "SELECT evidence_id::text, compound_id::text, compound_role FROM evidence_compound_attribution WHERE evidence_id::text IN :ids", missing)}
    fields = ("evidence_type", "compound_name", "compound_id", "source_id", "checksum", "source_location", "source_text",
              "parameter", "value", "unit", "canonical_value", "canonical_unit", "condition", "population", "route", "dose",
              "value_low", "value_high", "value_qualifier")
    def value(row, key):
        return row.get(key) if key in row else (row.get("source_record") or {}).get(key)
    def same(a, b, key):
        if a is None or b is None:
            return a is b
        if key in {"value", "canonical_value", "value_low", "value_high"}:
            return Decimal(str(a)) == Decimal(str(b))
        return str(a) == str(b)
    # Validate the entire set before the first update; a single conflict blocks
    # reconciliation and the enclosing transaction preserves the old state.
    for eid in sorted(missing):
        old = existing.get(eid); new = incoming_rows[eid]
        if not old or old.get("curation_status") not in {"needs_review", "machine_extracted"} or eid in curated:
            raise RuntimeError(f"Serving reconciliation requires untouched machine evidence: {eid}")
        if new.get("quality_status") != "auto_validated" or (eid, str(new.get("run_id"))) not in proven:
            raise RuntimeError(f"Serving reconciliation lacks current validation layer: {eid}")
        if (eid, str(new.get("compound_id")), str(new.get("compound_role"))) not in roles:
            raise RuntimeError(f"Serving reconciliation attribution differs: {eid}")
        differences = [key for key in fields if not same(value(old, key), value(new, key), key)]
        if differences or not new.get("checksum") or new.get("checksum") == "unknown":
            raise RuntimeError(f"Serving reconciliation scientific fields differ: {eid}: {differences}")
    promoted = []
    for eid in sorted(missing):
        old = existing[eid]; new = incoming_rows[eid]
        changed = connection.execute(text("UPDATE processed_evidence_candidate SET curation_status='auto_validated', updated_at=now() WHERE evidence_id::text=:eid AND curation_status::text=:previous"),
                                     {"eid": eid, "previous": old["curation_status"]}).rowcount
        if changed != 1:
            raise RuntimeError(f"Serving status changed concurrently: {eid}")
        changes = {"curation_status": {"old": old["curation_status"], "new": "auto_validated"},
                   "validation_run_id": new["run_id"], "validation_method": new.get("validation_method"),
                   "checksum": new["checksum"], "scientific_fields_unchanged": True, "provenance_links_unchanged": True}
        connection.execute(text("""INSERT INTO curation_event
            (evidence_table,evidence_id,previous_status,new_status,curator,curator_comment,field_changes)
            VALUES ('processed_evidence_candidate',CAST(:eid AS uuid),CAST(:previous AS curation_status),
                    'auto_validated','system:final_auto_validation',
                    'Automatic status alignment after current-run validation of identical scientific fields; no human approval implied.',CAST(:changes AS jsonb))"""),
            {"eid":eid,"previous":old["curation_status"],"changes":json.dumps(changes,ensure_ascii=False)})
        promoted.append(eid)
    available = set(selected("SELECT evidence_id::text FROM auto_validated_evidence WHERE evidence_id::text IN :ids", ids).scalars())
    if available != ids:
        raise RuntimeError("Serving verification failed after status reconciliation")
    return {"verified_count": len(ids), "promoted_ids": promoted}


def enrich_bound_compound_identities(connection, identities, run_id: str) -> int:
    """Fill missing machine identity fields, preserving IDs, human decisions and provenance."""
    from psycopg import sql
    from psycopg.rows import dict_row
    from pbpk_kb.processing.final_evidence_attribution_stage import normalize_name

    changed = 0
    fields = (("pubchem_cid", "pubchem_cid", "pubchem_cids"),
              ("chembl_id", "chembl_id", "chembl_ids"),
              ("rxnorm_rxcui", "rxnorm_rxcui", "rxnorm_rxcuis"),
              ("inchi_key", "inchikey", "inchikeys"))
    with connection.cursor(row_factory=dict_row) as cursor:
        for identity in sorted(identities, key=lambda item: str(item["compound_id"])):
            binding = identity.get("existing_registry_binding")
            if not binding or not identity.get("identity_provenance"):
                continue
            if str(binding["compound_id"]) != str(identity["compound_id"]):
                raise RuntimeError("Compound identity binding changed before import.")
            current = cursor.execute("SELECT * FROM compound_identity WHERE compound_id=%s FOR UPDATE",
                                     (identity["compound_id"],)).fetchone()
            if not current or normalize_name(current["preferred_name"]) != normalize_name(identity["preferred_name"]):
                raise RuntimeError("Registered compound identity changed before import.")
            if current["curation_status"] not in {"machine_extracted", "needs_review"}:
                continue
            human = cursor.execute("SELECT 1 FROM curation_event WHERE evidence_table='compound_identity' "
                                   "AND evidence_id=%s AND curator NOT LIKE 'system:%%' LIMIT 1",
                                   (identity["compound_id"],)).fetchone()
            if human:
                continue
            additions = {}
            for column, key, plural in fields:
                values = {str(value) for value in identity.get(plural, []) if value}
                incoming = str(identity[key]) if identity.get(key) else None
                if len(values) != 1 or incoming not in values:
                    continue
                if current[column] and str(current[column]) != incoming:
                    raise RuntimeError("Compound stable identifier changed before import.")
                if current[column] is None:
                    additions[column] = incoming
            if not additions:
                continue
            cursor.execute(sql.SQL("UPDATE compound_identity SET {},updated_at=now() WHERE compound_id=%s").format(
                sql.SQL(", ").join(sql.SQL("{}=%s").format(sql.Identifier(key)) for key in additions)),
                (*additions.values(), identity["compound_id"]))
            details = {"run_id": run_id, "fields": {key: {"old": None, "new": value} for key, value in additions.items()},
                       "identity_provenance": identity["identity_provenance"], "compound_id_unchanged": True}
            cursor.execute("INSERT INTO curation_event (evidence_table,evidence_id,previous_status,new_status,curator,curator_comment,field_changes) "
                           "VALUES ('compound_identity',%s,%s,%s,'system:identity_enrichment',%s,%s::jsonb)",
                           (identity["compound_id"], current["curation_status"], current["curation_status"],
                            "Missing stable identifiers filled from unambiguous, source-traceable identity records.", json.dumps(details)))
            changed += 1
    return changed


class CopyWriter:
    def __init__(self, importer: EvidenceImporter, table: str, columns: list[str]) -> None:
        self.importer = importer
        self.table = table
        self.columns = columns
        self.native_connection = None
        self.native_cursor = None
        self.native_copy_context = None
        self.native_copy = None
        self.staging_table = None
        self.rows_written = 0
        if importer.native_database_url:
            import psycopg
            from psycopg import sql

            self.native_connection = psycopg.connect(
                psycopg_dsn(importer.native_database_url)
            )
            self.native_cursor = self.native_connection.cursor()
            # Staging makes interrupted append imports safe to resume without
            # overwriting previously collected evidence or experiment records.
            self.staging_table = "pbpk_evidence_import_staging"
            self.native_cursor.execute(sql.SQL(
                "CREATE TEMP TABLE {} (LIKE {} INCLUDING DEFAULTS) ON COMMIT DROP"
            ).format(sql.Identifier(self.staging_table), sql.Identifier(table)))
            self.native_copy_context = self.native_cursor.copy(
                sql.SQL("COPY {} ({}) FROM STDIN").format(
                    sql.Identifier(self.staging_table),
                    sql.SQL(', ').join(map(sql.Identifier, columns)),
                )
            )
            self.native_copy = self.native_copy_context.__enter__()
            self.process = None
            self.writer = None
            return
        copy_sql = f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, HEADER true)"
        self.process = subprocess.Popen(
            ["docker", "exec", "-i", importer.container, "psql", "-v", "ON_ERROR_STOP=1", "-U", importer.db_user, "-d", importer.db_name, "-c", copy_sql],
            stdin=subprocess.PIPE,
            text=True,
        )
        if self.process.stdin is None:
            raise RuntimeError("Could not open COPY stdin")
        self.writer = csv.writer(self.process.stdin, lineterminator="\n")
        self.writer.writerow(columns)

    def write(self, row: dict[str, Any]) -> None:
        row = self.importer._map_identity_row(row)
        if self.native_copy is not None:
            self.native_copy.write_row(
                [csv_value(row.get(column)) for column in self.columns]
            )
            self.rows_written += 1
            return
        assert self.writer is not None
        self.writer.writerow([csv_value(row.get(column)) for column in self.columns])

    def close(self, error: BaseException | None = None) -> None:
        if self.native_copy_context is not None:
            assert self.native_connection is not None
            assert self.native_cursor is not None
            try:
                self.native_copy_context.__exit__(
                    type(error) if error else None, error, error.__traceback__ if error else None)
                if error is not None:
                    self.native_connection.rollback()
                    return
                from psycopg import sql
                columns = sql.SQL(', ').join(map(sql.Identifier, self.columns))
                self.native_cursor.execute(sql.SQL(
                    "INSERT INTO {} ({}) SELECT {} FROM {} ON CONFLICT DO NOTHING"
                ).format(sql.Identifier(self.table), columns, columns, sql.Identifier(self.staging_table)))
                inserted = self.native_cursor.rowcount
                self.native_connection.commit()
                self.importer.counts[f"{self.table}:inserted"] += inserted
                self.importer.counts[f"{self.table}:already_present"] += self.rows_written - inserted
            except BaseException:
                self.native_connection.rollback()
                raise
            finally:
                self.native_cursor.close()
                self.native_connection.close()
            return
        assert self.process is not None
        if error is not None:
            self.process.terminate()
        if self.process.stdin:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"COPY failed for {self.table} with exit code {return_code}")


def common_evidence_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": required(row, "evidence_id"),
        "source_id": row["source_id"],
        "raw_document_id": row["raw_document_id"],
        "parsed_document_id": row["parsed_document_id"],
        "source_location": row["source_location"],
        "evidence_text": row["source_text"],
        "value_origin": "machine_extracted" if row.get("value") is not None or row.get("value_text") else "unknown",
        "evidence_origin": "source_text",
        "extraction_method": row.get("extraction_method") or "machine_extracted",
        "extractor_version": row.get("extractor_version"),
        "confidence_score": row["extraction_confidence"],
        "curation_status": row["curation_status"],
    }


def raw_document_uuid(raw_asset_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"pbpk:raw_document:{raw_asset_id}"))


def psycopg_dsn(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def postgres_text_array(values: Iterable[Any]) -> str:
    escaped = []
    for value in values:
        text = clean_text(str(value)).replace("\\", "\\\\").replace('"', '\\"')
        escaped.append(f'"{text}"')
    return "{" + ",".join(escaped) + "}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_text(vector: Any) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"


def normalize_rag_curation_status(value: Any) -> str | None:
    if value in (None, ""):
        return None
    status = str(value)
    if status == "accepted":
        return "reviewed_accepted"
    return status


def datetime_utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.strip():
                continue
            yield json.loads(line)


def required(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required field {key}")
    return str(value)


def numeric(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return f"{number:.12g}"


def clamp01(value: Any) -> str | None:
    parsed = numeric(value)
    if parsed is None:
        return None
    number = max(0.0, min(1.0, float(parsed)))
    return f"{number:.3f}"


def text_or_default(value: Any, fallback: Any = "unknown") -> str:
    text = str(value).strip() if value not in (None, "") else ""
    return text or str(fallback)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def csv_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json_dump(value)
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, str):
        return clean_text(value)
    return value


def json_dump(value: Any) -> str:
    return json.dumps(clean_json_value(value if value is not None else {}), ensure_ascii=False, sort_keys=True, default=str)


def clean_text(value: str) -> str:
    return value.replace("\x00", "")


def clean_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, dict):
        return {clean_text(str(key)): clean_json_value(item) for key, item in value.items()}
    return value


def find_first_cid(payload: Any) -> str | None:
    if isinstance(payload, dict):
        if "cid" in payload and payload["cid"] not in (None, ""):
            return str(payload["cid"])
        if "CID" in payload and payload["CID"] not in (None, ""):
            value = payload["CID"]
            if isinstance(value, list) and value:
                return str(value[0])
            return str(value)
        for value in payload.values():
            cid = find_first_cid(value)
            if cid:
                return cid
    elif isinstance(payload, list):
        for item in payload:
            cid = find_first_cid(item)
            if cid:
                return cid
    return None


def infer_enzyme(parameter: str, text: str) -> str | None:
    haystack = f"{parameter} {text}".lower()
    for enzyme in ["CYP3A4", "CYP3A", "CYP2C9", "CYP2C19", "CYP2D6", "CYP1A2", "UGT1A1", "UGT"]:
        if enzyme.lower() in haystack:
            return enzyme
    return None


def infer_transporter(parameter: str, text: str) -> str | None:
    haystack = f"{parameter} {text}".lower()
    for transporter in ["P-gp", "P-glycoprotein", "OATP1B1", "OATP1B3", "OATP", "BCRP", "OCT2", "OCT", "MATE1", "MATE"]:
        if transporter.lower() in haystack:
            return transporter
    return None


def population_type(parameter: str, condition: Any, text: str) -> str:
    haystack = f"{parameter} {condition or ''} {text}".lower()
    if "renal" in haystack:
        return "renal_impairment"
    if "hepatic" in haystack or "liver" in haystack:
        return "hepatic_impairment"
    if "pediatric" in haystack or "paediatric" in haystack or "children" in haystack:
        return "pediatric"
    if "geriatric" in haystack or "elderly" in haystack:
        return "geriatric"
    if "pregnan" in haystack:
        return "pregnancy"
    if "lactation" in haystack or "breastfeeding" in haystack:
        return "lactation"
    return text_or_default(parameter, "special_population_candidate")


def pk_parameter(parameter: str, text: str) -> str | None:
    haystack = f"{parameter} {text}".lower()
    for pk in ["Cmax", "AUC", "Tmax", "half_life", "clearance", "bioavailability", "volume_of_distribution"]:
        if pk.lower() in haystack:
            return pk
    return None


def infer_software(text: str) -> str | None:
    lowered = text.lower()
    for software in ["PK-Sim", "MoBi", "Simcyp", "GastroPlus", "Open Systems Pharmacology"]:
        if software.lower() in lowered:
            return software
    return None


if __name__ == "__main__":
    main()
