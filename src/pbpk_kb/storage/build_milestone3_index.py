from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from pbpk_kb.graph.graph_loader import insert_chembl_target_relationships
from pbpk_kb.normalization.compound_resolver import CompoundResolver, ResolvedCompound, normalize_name
from pbpk_kb.normalization.unit_normalizer import normalize_unit, to_canonical_value
from pbpk_kb.storage.db import create_db_engine
from pbpk_kb.storage.source_registry import upsert_source_registry_entries
from pbpk_kb.storage.vector_index import EMBEDDING_MODEL, chunk_text, hash_embedding, vector_literal


MILESTONE2_SOURCES = ["pubmed", "dailymed", "pubchem", "osp_pbpk_model_library", "pkdb"]
STAGE_A_SOURCES = ["drugs_fda", "rxnorm", "drugcentral", "chembl", "clinicaltrials"]
STAGE1_MVP_SOURCES = [*MILESTONE2_SOURCES, *STAGE_A_SOURCES]
DEFAULT_STAGE_B_COMPOUNDS = [
    "midazolam",
    "warfarin",
    "caffeine",
    "metformin",
    "digoxin",
    "omeprazole",
    "theophylline",
    "dextromethorphan",
    "rosuvastatin",
    "repaglinide",
]

STAGE3_RESET_TABLES = [
    "document_chunk",
    "model_parameter_evidence",
    "public_pbpk_model_registry",
    "special_population_evidence",
    "ddi_evidence",
    "clinical_pk_evidence",
    "adme_parameter_evidence",
    "physicochemical_property_evidence",
    "tissue_expression_evidence",
    "enzyme_transporter_relationship",
    "protein_target_identity",
    "compound_identity",
    "parsed_document",
    "raw_document",
]


def load_latest_records(root: Path, source_ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    logs: dict[str, Any] = {}
    for source_id in source_ids:
        log_path = root / "ingestion_logs" / source_id / "latest.json"
        if not log_path.exists():
            continue
        log_payload = json.loads(log_path.read_text(encoding="utf-8"))
        logs[source_id] = log_payload
        parsed_records_path = localize_path(root, log_payload["parsed_records_path"])
        with parsed_records_path.open("r", encoding="utf-8") as records_file:
            for line in records_file:
                if line.strip():
                    record = json.loads(line)
                    record["_ingestion_log"] = log_payload
                    records.append(record)
    return records, logs


def replace_dailymed_placeholders_with_history(
    root: Path,
    records: list[dict[str, Any]],
    logs: dict[str, Any],
    query: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use the latest real DailyMed label for a compound when live SPL fetch failed.

    The connector can legitimately write a clearly marked sample placeholder when
    DailyMed is transiently unavailable. Indexing that placeholder would erase the
    usable public label evidence on a reset rebuild, even though versioned raw and
    parsed records still exist locally.
    """
    has_dailymed_placeholder = any(
        record.get("source_id") == "dailymed" and _is_sample_placeholder(record) for record in records
    )
    if not has_dailymed_placeholder:
        return records, logs

    historical = _latest_matching_dailymed_label(root, query)
    if historical is None:
        return records, logs

    parsed_path, historical_record = historical
    historical_record["_ingestion_log"] = {"parsed_records_path": str(parsed_path)}

    raw_metadata = _raw_metadata_for_record(root, historical_record)
    if raw_metadata:
        dailymed_log = logs.setdefault(
            "dailymed",
            {
                "source_id": "dailymed",
                "raw_documents": [],
                "parsed_records_path": str(parsed_path),
                "source_version": raw_metadata.get("source_version"),
                "source_checksum": raw_metadata.get("raw_checksum"),
            },
        )
        raw_documents = dailymed_log.setdefault("raw_documents", [])
        raw_paths = {raw_document.get("raw_storage_path") for raw_document in raw_documents}
        if raw_metadata.get("raw_storage_path") not in raw_paths:
            raw_documents.append(raw_metadata)

    replaced = [
        record
        for record in records
        if not (record.get("source_id") == "dailymed" and _is_sample_placeholder(record))
    ]
    replaced.append(historical_record)
    return replaced, logs


def _is_sample_placeholder(record: dict[str, Any]) -> bool:
    payload = record.get("record") or {}
    return record.get("record_type") == "sample_placeholder" or bool(payload.get("is_sample_placeholder"))


def _latest_matching_dailymed_label(root: Path, query: str) -> tuple[Path, dict[str, Any]] | None:
    parsed_root = root / "parsed" / "dailymed"
    if not parsed_root.exists():
        return None

    latest: tuple[Path, dict[str, Any]] | None = None
    for records_path in sorted(parsed_root.glob("*/records.jsonl")):
        with records_path.open("r", encoding="utf-8") as records_file:
            for line in records_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("record_type") != "label_detail":
                    continue
                if _dailymed_record_matches_query(record, query):
                    latest = (records_path, record)
    return latest


def _dailymed_record_matches_query(record: dict[str, Any], query: str) -> bool:
    query_key = normalize_name(query)
    if not query_key:
        return False
    payload = record.get("record") or {}
    text_values = [payload.get("title") or ""]
    text_values.extend(str(ingredient.get("name") or "") for ingredient in payload.get("active_ingredients") or [])
    return any(query_key in normalize_name(value) for value in text_values)


def _raw_metadata_for_record(root: Path, record: dict[str, Any]) -> dict[str, Any] | None:
    raw_path_value = record.get("raw_document_path")
    if not raw_path_value:
        return None
    raw_path = localize_path(root, raw_path_value)
    candidates = [Path(str(raw_path) + ".metadata.json")]
    candidates.extend(raw_path.parent.glob(f"{raw_path.name}*.metadata.json"))
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def localize_path(root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    if path.is_absolute() and len(path.parts) > 2:
        candidate = root / Path(*path.parts[2:])
        if candidate.exists():
            return candidate
    candidate = root / path_value
    return candidate


def compound_queries_from_args(query_values: list[str] | None, compounds_value: str | None) -> list[str]:
    values: list[str] = []
    for query_value in query_values or []:
        values.extend(query_value.split(","))
    if compounds_value:
        values.extend(compounds_value.split(","))
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = value.strip()
        key = normalize_name(query)
        if not query or key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
    return cleaned or ["midazolam"]


class Milestone3Indexer:
    def __init__(self, *, root: str | Path = ".", database_url: str | None = None) -> None:
        self.root = Path(root).resolve()
        self.engine = create_db_engine(database_url)
        self.resolver = CompoundResolver()

    def build(self, *, source_ids: list[str] | None = None, query: str = "midazolam", reset: bool = False) -> dict[str, Any]:
        source_ids = source_ids or STAGE1_MVP_SOURCES
        records, logs = load_latest_records(self.root, source_ids)
        records, logs = replace_dailymed_placeholders_with_history(self.root, records, logs, query)
        with self.engine.begin() as connection:
            self._ensure_vector_index(connection)
            if reset:
                self._reset_stage3_rows(connection)
            self._upsert_source_registry(connection, source_ids, logs)
            raw_by_path = self._insert_raw_documents(connection, logs)
            parsed_by_key = self._insert_parsed_documents(connection, records, raw_by_path)
            compound = self.resolver.resolve(records, query=query)
            compound_id = self._insert_compound(connection, compound)
            evidence_counts = self._insert_evidence(connection, records, parsed_by_key, compound_id, compound)
            chunk_count = self._insert_chunks(connection, records, parsed_by_key)
        return {
            "source_count": len(logs),
            "parsed_record_count": len(records),
            "compound": {
                "compound_id": str(compound_id),
                "preferred_name": compound.preferred_name,
                "pubchem_cid": compound.pubchem_cid,
                "inchi_key": compound.inchi_key,
            },
            "evidence_counts": evidence_counts,
            "document_chunk_count": chunk_count,
        }

    def build_many(
        self,
        *,
        queries: list[str],
        source_ids: list[str] | None = None,
        reset: bool = False,
        fetch_connectors: bool = False,
        retmax: int = 1,
        allow_network: bool = True,
        sync_source_registry: bool = True,
        connector_timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        source_ids = source_ids or STAGE1_MVP_SOURCES
        results: list[dict[str, Any]] = []
        aggregate_evidence_counts: dict[str, int] = {}
        total_chunks = 0
        total_parsed_records = 0
        if reset:
            with self.engine.begin() as connection:
                self._ensure_vector_index(connection)
                self._reset_stage3_rows(connection)
        for query in queries:
            connector_results: list[dict[str, Any]] | None = None
            if fetch_connectors:
                from pbpk_kb.connectors.run_connectors import run_sources

                connector_results = run_sources(
                    sources=source_ids,
                    query=query,
                    retmax=retmax,
                    output_root=str(self.root),
                    allow_network=allow_network,
                    sync_source_registry=sync_source_registry,
                    timeout_seconds=connector_timeout_seconds,
                )
            index_result = self.build(
                source_ids=source_ids,
                query=query,
                reset=False,
            )
            for table_name, count in index_result["evidence_counts"].items():
                aggregate_evidence_counts[table_name] = aggregate_evidence_counts.get(table_name, 0) + count
            total_chunks += index_result["document_chunk_count"]
            total_parsed_records += index_result["parsed_record_count"]
            results.append(
                {
                    "query": query,
                    "connector_results": self._connector_summary(connector_results or []),
                    "index_result": index_result,
                }
            )
        return {
            "compound_query_count": len(queries),
            "source_count": len(source_ids),
            "parsed_record_count": total_parsed_records,
            "document_chunk_count": total_chunks,
            "evidence_counts": aggregate_evidence_counts,
            "results": results,
        }

    @staticmethod
    def _connector_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "source_id": result["source_id"],
                "status": result["status"],
                "raw_document_count": result["raw_document_count"],
                "raw_record_count": result["raw_record_count"],
                "parsed_record_count": result["parsed_record_count"],
                "used_sample_placeholder": result["used_sample_placeholder"],
                "source_version": result.get("source_version"),
                "source_checksum_present": bool(result.get("source_checksum")),
                "errors": [error.get("message") for error in result.get("errors", [])],
            }
            for result in results
        ]

    def _ensure_vector_index(self, connection: Connection) -> None:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS document_chunk_embedding_hnsw_idx "
                "ON document_chunk USING hnsw (embedding vector_cosine_ops)"
            )
        )

    def _reset_stage3_rows(self, connection: Connection) -> None:
        for table_name in STAGE3_RESET_TABLES:
            connection.execute(text(f"DELETE FROM {table_name}"))

    def _upsert_source_registry(self, connection: Connection, source_ids: list[str], logs: dict[str, Any]) -> None:
        upsert_source_registry_entries(connection, source_ids=source_ids, metadata_by_source=logs)

    def _insert_raw_documents(self, connection: Connection, logs: dict[str, Any]) -> dict[str, Any]:
        raw_by_path: dict[str, Any] = {}
        for log_payload in logs.values():
            for raw_document in log_payload.get("raw_documents", []):
                raw_path = str(localize_path(self.root, raw_document["raw_storage_path"]))
                existing = connection.execute(
                    text("SELECT raw_document_id FROM raw_document WHERE source_id = :source_id AND raw_storage_path = :path"),
                    {"source_id": raw_document["source_id"], "path": raw_path},
                ).scalar_one_or_none()
                if existing:
                    raw_by_path[raw_path] = existing
                    continue
                raw_id = connection.execute(
                    text(
                        """
                        INSERT INTO raw_document (
                            source_id, source_url, source_record_id, source_version, download_timestamp,
                            raw_storage_path, raw_mime_type, raw_checksum, raw_size_bytes, metadata
                        )
                        VALUES (
                            :source_id, :source_url, :source_record_id, :source_version, :download_timestamp,
                            :raw_storage_path, :raw_mime_type, :raw_checksum, :raw_size_bytes, CAST(:metadata AS jsonb)
                        )
                        RETURNING raw_document_id
                        """
                    ),
                    {
                        "source_id": raw_document["source_id"],
                        "source_url": raw_document["source_url"],
                        "source_record_id": raw_document.get("extra_metadata", {}).get("set_id"),
                        "source_version": raw_document.get("source_version"),
                        "download_timestamp": raw_document["download_timestamp"],
                        "raw_storage_path": raw_path,
                        "raw_mime_type": raw_document.get("content_type"),
                        "raw_checksum": raw_document["raw_checksum"],
                        "raw_size_bytes": Path(raw_path).stat().st_size if Path(raw_path).exists() else None,
                        "metadata": json.dumps(raw_document, ensure_ascii=False),
                    },
                ).scalar_one()
                raw_by_path[raw_path] = raw_id
        return raw_by_path

    def _insert_parsed_documents(
        self,
        connection: Connection,
        records: list[dict[str, Any]],
        raw_by_path: dict[str, Any],
    ) -> dict[tuple[str, str], Any]:
        parsed_by_key: dict[tuple[str, str], Any] = {}
        for record in records:
            raw_path = str(localize_path(self.root, record["raw_document_path"]))
            parsed_checksum = sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            key = (record["source_id"], record["source_record_id"])
            existing = connection.execute(
                text("SELECT parsed_document_id FROM parsed_document WHERE source_id = :source_id AND parsed_checksum = :checksum"),
                {"source_id": record["source_id"], "checksum": parsed_checksum},
            ).scalar_one_or_none()
            if existing:
                parsed_by_key[key] = existing
                continue
            parsed_id = connection.execute(
                text(
                    """
                    INSERT INTO parsed_document (
                        raw_document_id, source_id, parser_name, parser_version, document_type, title,
                        abstract, source_location, parsed_storage_path, parsed_record, parsed_checksum
                    )
                    VALUES (
                        :raw_document_id, :source_id, :parser_name, :parser_version, :document_type, :title,
                        :abstract, :source_location, :parsed_storage_path, CAST(:parsed_record AS jsonb), :parsed_checksum
                    )
                    RETURNING parsed_document_id
                    """
                ),
                {
                    "raw_document_id": raw_by_path[raw_path],
                    "source_id": record["source_id"],
                    "parser_name": record["parser_name"],
                    "parser_version": record["parser_version"],
                    "document_type": record["record_type"],
                    "title": self._record_title(record),
                    "abstract": self._record_abstract(record),
                    "source_location": record["source_location"],
                    "parsed_storage_path": str(localize_path(self.root, record.get("_ingestion_log", {}).get("parsed_records_path", ""))),
                    "parsed_record": json.dumps(record, ensure_ascii=False),
                    "parsed_checksum": parsed_checksum,
                },
            ).scalar_one()
            parsed_by_key[key] = parsed_id
        return parsed_by_key

    def _insert_compound(self, connection: Connection, compound: ResolvedCompound) -> Any:
        existing = None
        if compound.pubchem_cid:
            existing = connection.execute(
                text("SELECT compound_id FROM compound_identity WHERE pubchem_cid = :pubchem_cid"),
                {"pubchem_cid": compound.pubchem_cid},
            ).scalar_one_or_none()
        if existing is None:
            existing = connection.execute(
                text("SELECT compound_id FROM compound_identity WHERE normalized_name = :normalized_name LIMIT 1"),
                {"normalized_name": compound.normalized_name},
            ).scalar_one_or_none()
        if existing:
            return existing
        return connection.execute(
            text(
                """
                INSERT INTO compound_identity (
                    preferred_name, normalized_name, synonyms, pubchem_cid, chembl_id, drugcentral_id,
                    rxnorm_rxcui, inchi_key, cas_number, approval_status, source_id, source_location,
                    confidence_score, curation_status
                )
                VALUES (
                    :preferred_name, :normalized_name, :synonyms, :pubchem_cid, :chembl_id, :drugcentral_id,
                    :rxnorm_rxcui, :inchi_key, :cas_number, :approval_status, :source_id, :source_location,
                    :confidence_score, :curation_status
                )
                RETURNING compound_id
                """
            ),
            {
                "preferred_name": compound.preferred_name,
                "normalized_name": compound.normalized_name,
                "synonyms": compound.synonyms,
                "pubchem_cid": compound.pubchem_cid,
                "chembl_id": compound.chembl_id,
                "drugcentral_id": compound.drugcentral_id,
                "rxnorm_rxcui": compound.rxnorm_rxcui,
                "inchi_key": compound.inchi_key,
                "cas_number": compound.cas_number,
                "approval_status": compound.approval_status,
                "source_id": compound.source_id,
                "source_location": compound.source_location,
                "confidence_score": compound.confidence_score,
                "curation_status": "needs_review" if compound.needs_review else "machine_extracted",
            },
        ).scalar_one()

    def _insert_evidence(
        self,
        connection: Connection,
        records: list[dict[str, Any]],
        parsed_by_key: dict[tuple[str, str], Any],
        compound_id: Any,
        compound: ResolvedCompound,
    ) -> dict[str, int]:
        counts = {
            "physicochemical_property_evidence": 0,
            "adme_parameter_evidence": 0,
            "clinical_pk_evidence": 0,
            "special_population_evidence": 0,
            "enzyme_transporter_relationship": 0,
            "public_pbpk_model_registry": 0,
        }
        for record in records:
            parsed_document_id = parsed_by_key[(record["source_id"], record["source_record_id"])]
            payload = record["record"]
            if record["source_id"] == "pubchem" and record["record_type"] == "compound_metadata":
                counts["physicochemical_property_evidence"] += self._insert_pubchem_properties(
                    connection, record, payload, parsed_document_id, compound_id
                )
            elif record["source_id"] == "dailymed" and record["record_type"] == "label_detail":
                result_counts = self._insert_dailymed_evidence(connection, record, payload, parsed_document_id, compound_id)
                for key, value in result_counts.items():
                    counts[key] += value
            elif record["source_id"] == "pkdb" and record["record_type"] == "pk_study_metadata":
                self._insert_pkdb_clinical_candidate(connection, record, payload, parsed_document_id, compound_id)
                counts["clinical_pk_evidence"] += 1
            elif record["source_id"] == "chembl" and record["record_type"] == "chembl_molecule":
                counts["enzyme_transporter_relationship"] += insert_chembl_target_relationships(
                    connection,
                    record=record,
                    parsed_document_id=parsed_document_id,
                    compound_id=compound_id,
                )
            elif record["source_id"] == "osp_pbpk_model_library" and record["record_type"] == "public_pbpk_model_folder":
                linked_compound_id = compound_id if self._model_matches_compound(payload, compound) else None
                self._insert_model_registry_record(connection, record, payload, linked_compound_id)
                counts["public_pbpk_model_registry"] += 1
        return counts

    def _insert_pubchem_properties(self, connection: Connection, record: dict[str, Any], payload: dict[str, Any], parsed_document_id: Any, compound_id: Any) -> int:
        inserted = 0
        for property_name, property_value_text in [
            ("MolecularFormula", payload.get("molecular_formula")),
            ("MolecularWeight", payload.get("molecular_weight")),
            ("CanonicalSMILES", payload.get("canonical_smiles")),
            ("InChIKey", payload.get("inchi_key")),
        ]:
            if property_value_text is None:
                continue
            property_value, unit = self._pubchem_numeric_property(property_name, property_value_text)
            canonical = self._canonical_fields(property_value, unit)
            connection.execute(
                text(
                    """
                    INSERT INTO physicochemical_property_evidence (
                        compound_id, property_name, property_value, property_value_text, unit,
                        canonical_value, canonical_unit, unit_dimension,
                        source_id, parsed_document_id,
                        source_location, evidence_text, extraction_method, extractor_version,
                        confidence_score, curation_status
                    )
                    VALUES (
                        :compound_id, :property_name, :property_value, :property_value_text, :unit,
                        :canonical_value, :canonical_unit, :unit_dimension,
                        :source_id, :parsed_document_id,
                        :source_location, :evidence_text, 'machine_extracted', '0.1.0',
                        0.85, 'needs_review'
                    )
                    """
                ),
                {
                    "compound_id": compound_id,
                    "property_name": property_name,
                    "property_value": property_value,
                    "property_value_text": str(property_value_text),
                    "unit": unit,
                    **canonical,
                    "source_id": record["source_id"],
                    "parsed_document_id": parsed_document_id,
                    "source_location": record["source_location"],
                    "evidence_text": json.dumps({property_name: property_value_text}, ensure_ascii=False),
                },
            )
            inserted += 1
        return inserted

    @staticmethod
    def _canonical_fields(value: float | None, unit: str | None) -> dict[str, Any]:
        normalized = normalize_unit(unit)
        canonical_value, canonical_unit = to_canonical_value(value, unit)
        return {
            "canonical_value": canonical_value,
            "canonical_unit": canonical_unit or normalized.canonical_unit,
            "unit_dimension": normalized.dimension,
        }

    def _insert_dailymed_evidence(self, connection: Connection, record: dict[str, Any], payload: dict[str, Any], parsed_document_id: Any, compound_id: Any) -> dict[str, int]:
        counts = {"clinical_pk_evidence": 0, "adme_parameter_evidence": 0, "special_population_evidence": 0}
        for candidate in payload.get("pk_section_candidates") or []:
            title = candidate.get("title") or candidate.get("display_name") or "PK section candidate"
            evidence_text = candidate.get("text") or ""
            source_location = f"{record['source_location']}::{candidate.get('source_location')}"
            connection.execute(
                text(
                    """
                    INSERT INTO clinical_pk_evidence (
                        compound_id, study_context, pk_parameter, source_id, parsed_document_id,
                        source_location, evidence_text, extraction_method, extractor_version,
                        confidence_score, curation_status,
                        renal_impairment_flag, hepatic_impairment_flag
                    )
                    VALUES (
                        :compound_id, :study_context, :pk_parameter, :source_id, :parsed_document_id,
                        :source_location, :evidence_text, 'section_candidate', '0.1.0',
                        0.55, 'needs_review',
                        :renal_impairment_flag, :hepatic_impairment_flag
                    )
                    """
                ),
                {
                    "compound_id": compound_id,
                    "study_context": "DailyMed Clinical Pharmacology / PK section-level candidate; numeric values require rule-based extraction or human review",
                    "pk_parameter": title,
                    "source_id": record["source_id"],
                    "parsed_document_id": parsed_document_id,
                    "source_location": source_location,
                    "evidence_text": evidence_text,
                    "renal_impairment_flag": "renal" in title.lower(),
                    "hepatic_impairment_flag": "hepatic" in title.lower(),
                },
            )
            counts["clinical_pk_evidence"] += 1

            lowered_title = title.lower()
            if lowered_title in {"absorption", "distribution", "metabolism", "elimination", "pharmacokinetics"}:
                connection.execute(
                    text(
                        """
                        INSERT INTO adme_parameter_evidence (
                            compound_id, parameter_name, value_text, source_id, parsed_document_id,
                            source_location, evidence_text, extraction_method, extractor_version,
                            confidence_score, curation_status
                        )
                        VALUES (
                            :compound_id, :parameter_name, :value_text, :source_id, :parsed_document_id,
                            :source_location, :evidence_text, 'section_candidate', '0.1.0',
                            0.50, 'needs_review'
                        )
                        """
                    ),
                    {
                        "compound_id": compound_id,
                        "parameter_name": title,
                        "value_text": None,
                        "source_id": record["source_id"],
                        "parsed_document_id": parsed_document_id,
                        "source_location": source_location,
                        "evidence_text": evidence_text,
                    },
                )
                counts["adme_parameter_evidence"] += 1
            if "renal" in lowered_title or "hepatic" in lowered_title or "special population" in lowered_title:
                connection.execute(
                    text(
                        """
                        INSERT INTO special_population_evidence (
                            compound_id, population_type, population_definition, source_id,
                            parsed_document_id, source_location, evidence_text, extraction_method,
                            extractor_version, confidence_score, curation_status
                        )
                        VALUES (
                            :compound_id, :population_type, :population_definition, :source_id,
                            :parsed_document_id, :source_location, :evidence_text, 'section_candidate',
                            '0.1.0', 0.50, 'needs_review'
                        )
                        """
                    ),
                    {
                        "compound_id": compound_id,
                        "population_type": title,
                        "population_definition": title,
                        "source_id": record["source_id"],
                        "parsed_document_id": parsed_document_id,
                        "source_location": source_location,
                        "evidence_text": evidence_text,
                    },
                )
                counts["special_population_evidence"] += 1
            numeric_counts = self._insert_dailymed_numeric_candidates(
                connection,
                record,
                parsed_document_id,
                compound_id,
                title,
                evidence_text,
                source_location,
            )
            for key, value in numeric_counts.items():
                counts[key] += value
        return counts

    @staticmethod
    def _pubchem_numeric_property(property_name: str, property_value: Any) -> tuple[float | None, str | None]:
        if property_name != "MolecularWeight":
            return None, None
        try:
            return float(property_value), "g/mol"
        except (TypeError, ValueError):
            return None, "g/mol"

    def _insert_dailymed_numeric_candidates(
        self,
        connection: Connection,
        record: dict[str, Any],
        parsed_document_id: Any,
        compound_id: Any,
        section_title: str,
        section_text: str,
        source_location: str,
    ) -> dict[str, int]:
        counts = {"clinical_pk_evidence": 0, "adme_parameter_evidence": 0}
        for candidate in self._extract_dailymed_numeric_candidates(section_title, section_text):
            canonical = self._canonical_fields(candidate["value"], candidate["unit"])
            if candidate["table"] == "clinical_pk_evidence":
                connection.execute(
                    text(
                        """
                        INSERT INTO clinical_pk_evidence (
                            compound_id, study_context, pk_parameter, value, value_text, unit,
                            canonical_value, canonical_unit, unit_dimension,
                            source_id, parsed_document_id, source_location, evidence_text,
                            extraction_method, extractor_version, confidence_score, curation_status
                        )
                        VALUES (
                            :compound_id, :study_context, :pk_parameter, :value, :value_text, :unit,
                            :canonical_value, :canonical_unit, :unit_dimension,
                            :source_id, :parsed_document_id, :source_location, :evidence_text,
                            'rule_based', '0.1.0', :confidence_score, 'needs_review'
                        )
                        """
                    ),
                    {
                        "compound_id": compound_id,
                        "study_context": f"DailyMed rule-based numeric candidate from {section_title}",
                        "pk_parameter": candidate["parameter_name"],
                        "value": candidate["value"],
                        "value_text": candidate["value_text"],
                        "unit": candidate["unit"],
                        **canonical,
                        "source_id": record["source_id"],
                        "parsed_document_id": parsed_document_id,
                        "source_location": source_location,
                        "evidence_text": candidate["evidence_text"],
                        "confidence_score": candidate["confidence_score"],
                    },
                )
                counts["clinical_pk_evidence"] += 1
            elif candidate["table"] == "adme_parameter_evidence":
                connection.execute(
                    text(
                        """
                        INSERT INTO adme_parameter_evidence (
                            compound_id, parameter_name, value, value_text, unit, measured_or_predicted,
                            canonical_value, canonical_unit, unit_dimension,
                            source_id, parsed_document_id, source_location, evidence_text,
                            extraction_method, extractor_version, confidence_score, curation_status
                        )
                        VALUES (
                            :compound_id, :parameter_name, :value, :value_text, :unit, 'reported',
                            :canonical_value, :canonical_unit, :unit_dimension,
                            :source_id, :parsed_document_id, :source_location, :evidence_text,
                            'rule_based', '0.1.0', :confidence_score, 'needs_review'
                        )
                        """
                    ),
                    {
                        "compound_id": compound_id,
                        "parameter_name": candidate["parameter_name"],
                        "value": candidate["value"],
                        "value_text": candidate["value_text"],
                        "unit": candidate["unit"],
                        **canonical,
                        "source_id": record["source_id"],
                        "parsed_document_id": parsed_document_id,
                        "source_location": source_location,
                        "evidence_text": candidate["evidence_text"],
                        "confidence_score": candidate["confidence_score"],
                    },
                )
                counts["adme_parameter_evidence"] += 1
        return counts

    @classmethod
    def _extract_dailymed_numeric_candidates(cls, section_title: str, text_value: str) -> list[dict[str, Any]]:
        lowered_title = section_title.lower()
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str | None, str, str]] = set()

        def add_candidate(
            *,
            table: str,
            parameter_name: str,
            value: float | None,
            value_text: str | None,
            unit: str,
            evidence_text: str,
            confidence_score: float = 0.72,
        ) -> None:
            compact_text = " ".join(evidence_text.split())
            key = (table, parameter_name.lower(), value_text or str(value), unit.lower(), compact_text.lower())
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                {
                    "table": table,
                    "parameter_name": parameter_name,
                    "value": value,
                    "value_text": value_text,
                    "unit": unit,
                    "evidence_text": compact_text,
                    "confidence_score": confidence_score,
                }
            )

        def add_single(
            *,
            match: re.Match[str],
            table: str,
            parameter_name: str,
            unit: str | None = None,
            confidence_score: float = 0.72,
        ) -> None:
            value = float(match.group("value"))
            unit_value = unit or match.group("unit")
            add_candidate(
                table=table,
                parameter_name=parameter_name,
                value=value,
                value_text=str(value),
                unit=cls._normalize_extracted_unit(unit_value),
                evidence_text=cls._sentence_around_match(text_value, match),
                confidence_score=confidence_score,
            )

        def add_range(
            *,
            match: re.Match[str],
            table: str,
            parameter_name: str,
            unit: str | None = None,
            confidence_score: float = 0.72,
        ) -> None:
            unit_value = unit or match.group("unit")
            add_candidate(
                table=table,
                parameter_name=parameter_name,
                value=None,
                value_text=f"{match.group('low')} to {match.group('high')}",
                unit=cls._normalize_extracted_unit(unit_value),
                evidence_text=cls._sentence_around_match(text_value, match),
                confidence_score=confidence_score,
            )

        time_unit = r"hours?|hrs?|h|minutes?|mins?|min|days?|d"
        concentration_unit = r"ng\s*/\s*mL|ng\s*/\s*ml|ug\s*/\s*mL|mcg\s*/\s*mL|mg\s*/\s*L"
        auc_unit = r"ng\s*(?:\*|·|x)?\s*h(?:r)?\s*/\s*mL|ug\s*(?:\*|·|x)?\s*h(?:r)?\s*/\s*mL|mcg\s*(?:\*|·|x)?\s*h(?:r)?\s*/\s*mL|mg\s*(?:\*|·|x)?\s*h(?:r)?\s*/\s*L"
        clearance_unit = r"mL\s*/\s*min\s*/\s*kg|ml\s*/\s*min\s*/\s*kg|mL\s*/\s*min|ml\s*/\s*min|L\s*/\s*h\s*/\s*kg|L\s*/\s*h|L\s*/\s*hr"
        volume_unit = r"L\s*/\s*kg|l\s*/\s*kg|mL\s*/\s*kg|ml\s*/\s*kg|L|l"

        for match in re.finditer(
            r"(?:absolute\s+)?bioavailability[^.]{0,200}?\b(?:about|approximately|is|was|of)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)(?!\w)",
            text_value,
            re.IGNORECASE,
        ):
            add_single(match=match, table="adme_parameter_evidence", parameter_name="absolute bioavailability")

        for match in re.finditer(
            r"(?:T\s*max|Tmax|time to (?:reach )?(?:peak|max(?:imum)?) (?:plasma )?concentration)[^.]{0,160}?(?:range(?:d)? from|ranges from|between|was|is|of|:)?\s*(?P<low>\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(?P<high>\d+(?:\.\d+)?)\s*(?P<unit>"
            + time_unit
            + r")\b",
            text_value,
            re.IGNORECASE,
        ):
            add_range(match=match, table="clinical_pk_evidence", parameter_name="Tmax")

        for match in re.finditer(
            r"(?:T\s*max|Tmax|time to (?:reach )?(?:peak|max(?:imum)?) (?:plasma )?concentration)[^.]{0,160}?(?:was|is|of|approximately|about|:)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>"
            + time_unit
            + r")\b",
            text_value,
            re.IGNORECASE,
        ):
            add_single(match=match, table="clinical_pk_evidence", parameter_name="Tmax")

        for match in re.finditer(
            r"(?:protein binding|plasma protein binding|bound to (?:plasma )?proteins?)[^.]{0,160}?(?:is|was|approximately|about|of)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)(?!\w)",
            text_value,
            re.IGNORECASE,
        ):
            add_single(match=match, table="adme_parameter_evidence", parameter_name="plasma protein binding")

        for match in re.finditer(
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)(?!\w)[^.]{0,100}?bound to (?:plasma )?proteins?",
            text_value,
            re.IGNORECASE,
        ):
            add_single(match=match, table="adme_parameter_evidence", parameter_name="plasma protein binding")

        for match in re.finditer(
            r"(?:steady-state |apparent )?(?:volume of distribution|Vd|Vss)[^.]{0,180}?(?:range(?:d)? from|ranges from|between|was|is|of|approximately|about)?\s*(?P<low>\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(?P<high>\d+(?:\.\d+)?)\s*(?P<unit>"
            + volume_unit
            + r")\b",
            text_value,
            re.IGNORECASE,
        ):
            add_range(match=match, table="clinical_pk_evidence", parameter_name="volume of distribution")

        for match in re.finditer(
            r"(?:steady-state |apparent )?(?:volume of distribution|Vd|Vss)[^.]{0,180}?(?:was|is|of|approximately|about|:)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>"
            + volume_unit
            + r")\b",
            text_value,
            re.IGNORECASE,
        ):
            add_single(match=match, table="clinical_pk_evidence", parameter_name="volume of distribution")

        for match in re.finditer(
            r"(?:terminal\s+)?(?:elimination\s+)?half[- ]life(?: of [^.]{0,80}?)?[^.]{0,140}?(?:range(?:d)? from|ranges from|between|was|is|of|approximately|about)?\s*(?P<low>\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(?P<high>\d+(?:\.\d+)?)\s*(?P<unit>"
            + time_unit
            + r")\b",
            text_value,
            re.IGNORECASE,
        ):
            add_range(match=match, table="clinical_pk_evidence", parameter_name="elimination half-life")

        for match in re.finditer(
            r"(?:terminal\s+)?(?:elimination\s+)?half[- ]life(?: of [^.]{0,80}?)?[^.]{0,140}?(?:was|is|of|approximately|about|:)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>"
            + time_unit
            + r")\b",
            text_value,
            re.IGNORECASE,
        ):
            add_single(match=match, table="clinical_pk_evidence", parameter_name="elimination half-life")

        for match in re.finditer(
            r"(?:mean\s+)?(?:total\s+|body\s+|oral\s+|renal\s+)?clearance[^.]{0,180}?(?:range(?:d)? from|ranges from|between|was|is|of|approximately|about)?\s*(?P<low>\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(?P<high>\d+(?:\.\d+)?)\s*(?P<unit>"
            + clearance_unit
            + r")\b",
            text_value,
            re.IGNORECASE,
        ):
            add_range(match=match, table="clinical_pk_evidence", parameter_name="total clearance")

        for match in re.finditer(
            r"(?:mean\s+)?(?:total\s+|body\s+|oral\s+|renal\s+)?clearance[^.]{0,180}?(?:was|is|of|approximately|about|:)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>"
            + clearance_unit
            + r")\b",
            text_value,
            re.IGNORECASE,
        ):
            add_single(match=match, table="clinical_pk_evidence", parameter_name="total clearance")

        for match in re.finditer(
            r"\bC\s*max\b[^.]{0,160}?(?:was|is|of|approximately|about|:)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>"
            + concentration_unit
            + r")\b",
            text_value,
            re.IGNORECASE,
        ):
            add_single(match=match, table="clinical_pk_evidence", parameter_name="Cmax", confidence_score=0.68)

        for match in re.finditer(
            r"\bAUC(?:0[-–]?\w+)?\b[^.]{0,160}?(?:was|is|of|approximately|about|:)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>"
            + auc_unit
            + r")\b",
            text_value,
            re.IGNORECASE,
        ):
            add_single(match=match, table="clinical_pk_evidence", parameter_name="AUC", confidence_score=0.68)

        return candidates

    @staticmethod
    def _normalize_extracted_unit(unit: str) -> str:
        return (
            " ".join(unit.split())
            .replace("µ", "u")
            .replace("μ", "u")
            .replace("·", "*")
            .replace("×", "*")
            .replace(" / ", "/")
            .replace("/ ", "/")
            .replace(" /", "/")
        )

    @staticmethod
    def _sentence_around_match(text_value: str, match: re.Match[str]) -> str:
        start = Milestone3Indexer._previous_sentence_boundary(text_value, match.start()) + 1
        end = Milestone3Indexer._next_sentence_boundary(text_value, match.end())
        if end == -1:
            end = min(len(text_value), match.end() + 240)
        return " ".join(text_value[start : end + 1].split())

    @staticmethod
    def _previous_sentence_boundary(text_value: str, position: int) -> int:
        index = position - 1
        while index >= 0:
            if text_value[index] == "." and not Milestone3Indexer._is_decimal_point(text_value, index):
                return index
            index -= 1
        return -1

    @staticmethod
    def _next_sentence_boundary(text_value: str, position: int) -> int:
        index = position
        while index < len(text_value):
            if text_value[index] == "." and not Milestone3Indexer._is_decimal_point(text_value, index):
                return index
            index += 1
        return -1

    @staticmethod
    def _is_decimal_point(text_value: str, index: int) -> bool:
        return (
            index > 0
            and index + 1 < len(text_value)
            and text_value[index - 1].isdigit()
            and text_value[index + 1].isdigit()
        )

    def _insert_pkdb_clinical_candidate(self, connection: Connection, record: dict[str, Any], payload: dict[str, Any], parsed_document_id: Any, compound_id: Any) -> None:
        evidence_text = json.dumps(
            {
                "study_sid": payload.get("study_sid"),
                "study_name": payload.get("study_name"),
                "filter_counts": payload.get("filter_counts"),
            },
            ensure_ascii=False,
        )
        connection.execute(
            text(
                """
                INSERT INTO clinical_pk_evidence (
                    compound_id, study_context, pk_parameter, value_text, source_id, parsed_document_id,
                    source_location, evidence_text, extraction_method, extractor_version, confidence_score,
                    curation_status
                )
                VALUES (
                    :compound_id, :study_context, 'pkdb_study_metadata', :value_text, :source_id, :parsed_document_id,
                    :source_location, :evidence_text, 'machine_extracted', '0.1.0', 0.70,
                    'needs_review'
                )
                """
            ),
            {
                "compound_id": compound_id,
                "study_context": payload.get("study_name") or payload.get("study_sid"),
                "value_text": None,
                "source_id": record["source_id"],
                "parsed_document_id": parsed_document_id,
                "source_location": record["source_location"],
                "evidence_text": evidence_text,
            },
        )

    def _insert_model_registry_record(self, connection: Connection, record: dict[str, Any], payload: dict[str, Any], compound_id: Any) -> None:
        model_link_note = (
            "Model folder name matched the normalized compound identity."
            if compound_id
            else "Indexed from public OSP model folder inventory; compound linkage not asserted until curator review."
        )
        connection.execute(
            text(
                """
                INSERT INTO public_pbpk_model_registry (
                    model_name, compound_id, model_type, software, model_source, source_id,
                    model_file_path, evaluation_report_path, validation_status, curation_status,
                    source_location, notes
                )
                VALUES (
                    :model_name, :compound_id, 'public_repository_folder', 'Open Systems Pharmacology',
                    :model_source, :source_id, :model_file_path, :evaluation_report_path,
                    :validation_status, 'needs_review', :source_location, :notes
                )
                """
            ),
            {
                "model_name": payload.get("model_folder_name") or record["source_record_id"],
                "compound_id": compound_id,
                "model_source": payload.get("github_html_url"),
                "source_id": record["source_id"],
                "model_file_path": payload.get("model_folder_path"),
                "evaluation_report_path": None,
                "validation_status": "metadata_indexed",
                "source_location": record["source_location"],
                "notes": model_link_note,
            },
        )

    @staticmethod
    def _model_matches_compound(payload: dict[str, Any], compound: ResolvedCompound) -> bool:
        model_name = normalize_name(payload.get("model_folder_name"))
        if not model_name:
            return False
        accepted_names = {compound.normalized_name}
        accepted_names.update(normalize_name(synonym) for synonym in compound.synonyms)
        accepted_names.discard("")
        return model_name in accepted_names

    def _insert_chunks(
        self,
        connection: Connection,
        records: list[dict[str, Any]],
        parsed_by_key: dict[tuple[str, str], Any],
    ) -> int:
        inserted = 0
        for record in records:
            parsed_document_id = parsed_by_key[(record["source_id"], record["source_record_id"])]
            for section_title, text_value, source_location in self._text_sources(record):
                for chunk in chunk_text(text_value, section_title=section_title):
                    connection.execute(
                        text(
                            """
                            INSERT INTO document_chunk (
                                source_id, document_id, section_title, paragraph_number, text,
                                token_count, embedding, embedding_model, parser_version, source_location
                            )
                            VALUES (
                                :source_id, :document_id, :section_title, :paragraph_number, :text,
                                :token_count, CAST(:embedding AS vector), :embedding_model, :parser_version,
                                :source_location
                            )
                            """
                        ),
                        {
                            "source_id": record["source_id"],
                            "document_id": parsed_document_id,
                            "section_title": section_title,
                            "paragraph_number": chunk.paragraph_number,
                            "text": chunk.text,
                            "token_count": chunk.token_count,
                            "embedding": vector_literal(hash_embedding(chunk.text)),
                            "embedding_model": EMBEDDING_MODEL,
                            "parser_version": record["parser_version"],
                            "source_location": source_location,
                        },
                    )
                    inserted += 1
        return inserted

    def _text_sources(self, record: dict[str, Any]) -> list[tuple[str | None, str, str]]:
        payload = record["record"]
        source_location = record["source_location"]
        if record["source_id"] == "dailymed" and record["record_type"] == "label_detail":
            sources: list[tuple[str | None, str, str]] = []
            clinical = payload.get("clinical_pharmacology_section")
            if clinical and clinical.get("text"):
                sources.append((clinical.get("title"), clinical["text"], f"{source_location}::{clinical.get('source_location')}"))
            for candidate in payload.get("pk_section_candidates") or []:
                if candidate.get("text"):
                    sources.append((candidate.get("title"), candidate["text"], f"{source_location}::{candidate.get('source_location')}"))
            return sources
        if record["source_id"] == "pubchem":
            text_value = json.dumps(payload, ensure_ascii=False)
            return [("PubChem compound metadata", text_value, source_location)]
        if record["source_id"] == "pubmed":
            text_value = " ".join(
                str(value)
                for value in [payload.get("title"), payload.get("journal"), payload.get("publication_date"), payload.get("doi")]
                if value
            )
            return [("PubMed metadata", text_value, source_location)]
        if record["source_id"] == "pkdb":
            return [("PK-DB study metadata", json.dumps(payload, ensure_ascii=False), source_location)]
        if record["source_id"] == "osp_pbpk_model_library":
            return [("OSP model folder metadata", json.dumps(payload, ensure_ascii=False), source_location)]
        return [(record["record_type"], json.dumps(payload, ensure_ascii=False), source_location)]

    @staticmethod
    def _record_title(record: dict[str, Any]) -> str | None:
        payload = record.get("record") or {}
        return (
            payload.get("title")
            or payload.get("model_folder_name")
            or payload.get("study_name")
            or payload.get("pubchem_cid")
        )

    @staticmethod
    def _record_abstract(record: dict[str, Any]) -> str | None:
        payload = record.get("record") or {}
        if record.get("source_id") == "dailymed":
            clinical = payload.get("clinical_pharmacology_section") or {}
            return clinical.get("text")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Milestone 3 structured and vector indexes.")
    parser.add_argument("--root", default=".", help="Workspace root containing ingestion_logs/, raw/, and parsed/.")
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Compound query/name used for identity resolution. Repeat for multiple compounds.",
    )
    parser.add_argument(
        "--compounds",
        default=None,
        help="Comma-separated compound queries. Can be combined with repeated --query.",
    )
    parser.add_argument("--source", action="append", default=None, help="Source ID to index. Defaults to the 10 Stage 1 MVP sources.")
    parser.add_argument(
        "--fetch-connectors",
        action="store_true",
        help="Run selected connectors for each compound before indexing its latest parsed records.",
    )
    parser.add_argument("--retmax", type=int, default=1, help="Maximum records per connector when --fetch-connectors is used.")
    parser.add_argument(
        "--connector-timeout-seconds",
        type=int,
        default=30,
        help="Per-request connector timeout in seconds when --fetch-connectors is used.",
    )
    parser.add_argument("--no-network", action="store_true", help="Disable live connector retrieval when --fetch-connectors is used.")
    parser.add_argument(
        "--no-source-registry-sync",
        action="store_true",
        help="Do not sync connector source versions/checksums into source_registry during connector fetches.",
    )
    parser.add_argument("--reset-stage3", action="store_true", help="Clear generated DB rows before indexing. Raw files are not deleted.")
    args = parser.parse_args()

    queries = compound_queries_from_args(args.query, args.compounds)
    indexer = Milestone3Indexer(root=args.root)
    if args.reset_stage3:
        print(
            "WARNING: --reset-stage3 deletes generated structured evidence/index rows before rebuilding. "
            "It is a development rebuild operation, not an immutable evidence-history workflow.",
            file=sys.stderr,
        )
    if len(queries) == 1 and not args.fetch_connectors:
        result = indexer.build(
            source_ids=args.source or STAGE1_MVP_SOURCES,
            query=queries[0],
            reset=args.reset_stage3,
        )
    else:
        result = indexer.build_many(
            source_ids=args.source or STAGE1_MVP_SOURCES,
            queries=queries,
            reset=args.reset_stage3,
            fetch_connectors=args.fetch_connectors,
            retmax=args.retmax,
            allow_network=not args.no_network,
            sync_source_registry=not args.no_source_registry_sync,
            connector_timeout_seconds=args.connector_timeout_seconds,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
