from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, TextIO
from uuid import NAMESPACE_URL, uuid5

from pbpk_kb.extraction.ddi_extractor import extract_ddi_statements
from pbpk_kb.normalization.unit_normalizer import normalize_unit, to_canonical_value
from pbpk_kb.processing.chunk_classifier import EVIDENCE_SIGNAL_RE, classify_chunk, should_run_rule_extraction
from pbpk_kb.processing.raw_processing_paths import RawProcessingLayout, raw_processing_layout
from pbpk_kb.processing.raw_processor import (
    ADME_ACRONYM_TERMS,
    ADME_WORD_TERMS,
    PBPK_TERMS,
    PK_PATTERNS,
    SECTION_CANDIDATE_SOURCES,
    SPECIAL_POP_TERMS,
    _earliest_match,
    _looks_like_numeric_context,
    _normalize_parameter,
    _sentence_around,
)


EXTRACTOR_VERSION = "deterministic-extract-v1"
PROGRESS_INTERVAL = 10_000

STAGE_OUTPUTS = (
    "structured_record_processing_status.jsonl",
    "structured_table_processing_status.jsonl",
    "chunk_processing_status.jsonl",
    "evidence_candidates_deterministic.jsonl",
    "evidence_candidates.jsonl",
    "evidence_pointers.jsonl",
    "processing_failures.jsonl",
)

IMMUTABLE_DOWNSTREAM_FILES = (
    "embedding_index.jsonl",
    "evidence_candidates_llm.jsonl",
    "llm_extraction_requests.jsonl",
    "llm_extraction_responses.jsonl",
    "llm_extraction_requests_live.jsonl",
    "llm_extraction_responses_live.jsonl",
)

PUBCHEM_FIELDS: dict[str, tuple[bool, str | None]] = {
    "CID": (False, None),
    "MolecularFormula": (False, None),
    "MolecularWeight": (True, "g/mol"),
    "ConnectivitySMILES": (False, None),
    "CanonicalSMILES": (False, None),
    "IsomericSMILES": (False, None),
    "InChIKey": (False, None),
    "XLogP": (True, None),
    "TPSA": (True, None),
}

CHEMBL_FIELDS: dict[str, tuple[bool, str | None]] = {
    "full_mwt": (True, "g/mol"),
    "alogp": (True, None),
    "psa": (True, None),
    "hba": (True, None),
    "hbd": (True, None),
}

PARAMETER_UNIT_DIMENSIONS: dict[str, set[str]] = {
    "Cmax": {"concentration", "molar_concentration"},
    "AUC": {"auc"},
    "Tmax": {"time"},
    "half_life": {"time"},
    "clearance": {"clearance", "clearance_per_weight", "clearance_bsa_normalized"},
    "volume_of_distribution": {"volume", "volume_per_weight"},
    "bioavailability": {"fraction_percent"},
}

RENAL_FUNCTION_CLEARANCE_RE = re.compile(
    r"(?:\bcreatinine\s+clearance\b|\bCrCl\b|\bCL\s*[_ ]?\s*cr\b|\beGFR\b|\bestimated glomerular filtration rate\b)",
    flags=re.IGNORECASE,
)
DIALYSIS_CLEARANCE_RE = re.compile(r"\b(?:dialyzable|dialysable|hemodialysis|haemodialysis|dialysis)\b", flags=re.IGNORECASE)
RELATIVE_BIOAVAILABILITY_RE = re.compile(
    r"(?:\b(?:increased|decreased)\s+by\b|\b[0-9]+(?:\.[0-9]+)?\s*%[^.;]{0,30}\b(?:higher|lower)\b)",
    flags=re.IGNORECASE,
)
METABOLITE_SUBJECT_RE = re.compile(
    r"(?:\b(?:metabolites?|glucuronides?|hydroxy[- ]?[A-Za-z0-9-]*|desmethyl[A-Za-z0-9-]*)\b|"
    r"(?<![A-Za-z0-9])(?:\d+[\u2032'\u2019]?-?)?OH[- ]+[A-Za-z][A-Za-z0-9-]*)",
    flags=re.IGNORECASE,
)
PK_LOCAL_RELATION_MAX_BRIDGE = 72
PK_PARAMETER_TOKEN_RE = re.compile(
    r"\b(?:C\s*max|AUC(?:0-[a-z0-9]+|inf|last)?|T\s*max|half[- ]life|t1/2|t½|clearance|CL|"
    r"volume of distribution|Vd|Vz/F|Vz|Vss|bioavailability)\b",
    flags=re.IGNORECASE,
)
INTERVENING_PK_FIELD_RE = re.compile(
    r"\b(?:concentration\s+at|C\s*\d+|creatinine|urinary|renal\s+(?:clearance|excretion)|"
    r"AUC|C\s*max|T\s*max|half[- ]life|clearance|volume of distribution|bioavailability)\b",
    flags=re.IGNORECASE,
)
PK_SUBJECT_SHIFT_BRIDGE_RE = re.compile(
    r"\b(?:did\s+not\s+alter|does\s+not\s+alter|altered?|affects?|affected|effect\s+on|"
    r"compared\s+(?:with|to)|versus|vs\.?|than|coadministered|concomitantly\s+administered)\b",
    flags=re.IGNORECASE,
)
MODEL_INTERNAL_LOCATION_RE = re.compile(r"(?:\.pksim5(?:[#!]|$)|#sqlite_table=(?:COMMANDS|PARAMETERS?))", flags=re.IGNORECASE)
NON_OBSERVED_PK_CONTEXT_RE = re.compile(
    r"\b(?:threshold|assumed|assumption|value\s+of\s+parameter|set\s+from|set\s+to|parameter[- ]setting)\b",
    flags=re.IGNORECASE,
)


@dataclass
class MappingResult:
    mapper_name: str
    status: str
    reason: str
    evidence: list[dict[str, Any]]
    pointers: list[dict[str, Any]]


class AtomicJSONLBundle:
    def __init__(self, layout: RawProcessingLayout, names: tuple[str, ...]) -> None:
        self.layout = layout
        self.names = names
        self.targets: dict[str, Path] = {}
        self.temporary: dict[str, Path] = {}
        self.handles: dict[str, TextIO] = {}

    def __enter__(self) -> "AtomicJSONLBundle":
        for name in self.names:
            target = self.layout.path(name, for_write=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.deterministic.tmp")
            temporary.unlink(missing_ok=True)
            self.targets[name] = target
            self.temporary[name] = temporary
            self.handles[name] = temporary.open("w", encoding="utf-8")
        return self

    def write(self, name: str, row: dict[str, Any]) -> None:
        self.handles[name].write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
        self.handles[name].write("\n")

    def write_raw(self, name: str, line: str) -> None:
        self.handles[name].write(line.rstrip("\n"))
        self.handles[name].write("\n")

    def commit(self) -> None:
        for handle in self.handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self.handles.clear()
        for name in self.names:
            os.replace(self.temporary[name], self.targets[name])

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        for path in self.temporary.values():
            path.unlink(missing_ok=True)


class DeterministicExtractStage:
    def __init__(self, *, run_root: Path, output_dir: Path, classifier_min_chars: int = 160) -> None:
        self.run_root = run_root.resolve()
        self.run_id = self.run_root.name
        self.output_dir = output_dir.resolve()
        self.layout = raw_processing_layout(self.output_dir)
        self.classifier_min_chars = classifier_min_chars
        self.started_at = datetime.now(timezone.utc).isoformat()

        self.inventory: dict[str, dict[str, Any]] = {}
        self.alias_compounds: dict[str, set[str]] = defaultdict(set)
        self.target_compounds: set[str] = set()
        self.repository_assets_by_path: dict[str, dict[str, Any]] = {}
        self.repository_reports_by_compound: dict[str, list[dict[str, Any]]] = defaultdict(list)

        self.counts: Counter[str] = Counter()
        self.record_status_counts: Counter[str] = Counter()
        self.table_status_counts: Counter[str] = Counter()
        self.chunk_status_counts: Counter[str] = Counter()
        self.evidence_type_counts: Counter[str] = Counter()
        self.evidence_source_counts: Counter[str] = Counter()
        self.pointer_topic_counts: Counter[str] = Counter()
        self.seen_evidence_ids: set[str] = set()
        self.seen_pointer_ids: set[str] = set()
        self.provenance_failures = 0
        self.bundle: AtomicJSONLBundle | None = None

    def run(self) -> dict[str, Any]:
        parse_summary, pipeline_state = self._validate_prerequisite()
        downstream_before = snapshot_files(self.layout, IMMUTABLE_DOWNSTREAM_FILES)
        self._load_context()
        progress_path = self.layout.path("deterministic_extract_progress.json", for_write=True)
        self._write_progress(progress_path, "records", 0, int(parse_summary["parsed_record_count"]))

        try:
            with AtomicJSONLBundle(self.layout, STAGE_OUTPUTS) as bundle:
                self.bundle = bundle
                self._preserve_non_stage_failures(bundle)
                self._process_records(parse_summary, progress_path)
                self._process_tables(parse_summary, progress_path)
                self._process_chunks(parse_summary, progress_path)
                bundle.commit()
                self.bundle = None
        except Exception as exc:
            self.bundle = None
            self._write_failed_state(pipeline_state, progress_path, exc)
            raise

        downstream_after = snapshot_files(self.layout, IMMUTABLE_DOWNSTREAM_FILES)
        summary = self._build_summary(parse_summary, downstream_before, downstream_after)
        write_json_atomic(self.layout.path("deterministic_extract_summary.json", for_write=True), summary)
        self._update_processing_summary(summary, parse_summary)
        self._update_pipeline_state(pipeline_state, summary)
        self._write_progress(
            progress_path,
            "completed",
            self.counts["chunk_rows"],
            int(parse_summary["text_chunk_count"]),
            status="completed" if summary["accepted"] else "failed",
        )
        return summary

    def _validate_prerequisite(self) -> tuple[dict[str, Any], dict[str, Any]]:
        parse_summary_path = self.layout.path("parse_summary.json")
        state_path = self.layout.path("pipeline_state.json")
        if not parse_summary_path.exists() or not state_path.exists():
            raise FileNotFoundError("parse_summary.json and pipeline_state.json are required")
        parse_summary = json.loads(parse_summary_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        parse_state = (state.get("stages") or {}).get("parse") or {}
        if parse_summary.get("status") != "completed" or parse_summary.get("accepted") is not True:
            raise ValueError("Parse summary is not accepted")
        if parse_state.get("status") != "completed" or parse_state.get("accepted") is not True:
            raise ValueError("pipeline_state parse prerequisite is not accepted")
        if parse_summary.get("run_id") != self.run_id or state.get("run_id") != self.run_id:
            raise ValueError("Run ID mismatch between raw run and parsed outputs")
        for downstream in ("llm_extract", "completeness", "embedding"):
            status = ((state.get("stages") or {}).get(downstream) or {}).get("status", "not_started")
            if status != "not_started":
                raise ValueError(f"Downstream stage {downstream} already started; reset it before deterministic rerun")
        required = (
            "raw_inventory.jsonl",
            "raw_asset_aliases.jsonl",
            "parsed_records.jsonl",
            "parsed_tables.jsonl",
            "text_chunks.jsonl",
        )
        missing = [name for name in required if not self.layout.path(name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing deterministic extraction inputs: {', '.join(missing)}")
        return parse_summary, state

    def _load_context(self) -> None:
        for row in iter_jsonl(self.layout.path("raw_inventory.jsonl")):
            raw_asset_id = str(row.get("raw_asset_id") or "")
            if not raw_asset_id:
                continue
            self.inventory[raw_asset_id] = row
            compound = clean_optional(row.get("compound"))
            if compound:
                self.target_compounds.add(compound)
                self.alias_compounds[raw_asset_id].add(compound)
            metadata = row.get("asset_metadata_json") if isinstance(row.get("asset_metadata_json"), dict) else {}
            nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
            repository_path = clean_optional(nested.get("repository_path"))
            if row.get("source_id") == "osp_pbpk_model_library" and repository_path:
                self.repository_assets_by_path[repository_path] = row
                repository_compound = repository_compound_name(repository_path, self.target_compounds)
                if repository_compound and re.search(r"evaluation[_ -]?report\.(?:md|pdf)$", repository_path, flags=re.I):
                    self.repository_reports_by_compound[repository_compound].append(row)
        for row in iter_jsonl(self.layout.path("raw_asset_aliases.jsonl")):
            canonical = clean_optional(row.get("canonical_raw_asset_id"))
            compound = clean_optional(row.get("alias_compound"))
            if canonical and compound:
                self.alias_compounds[canonical].add(compound)
                self.target_compounds.add(compound)

    def _process_records(self, parse_summary: dict[str, Any], progress_path: Path) -> None:
        expected = int(parse_summary["parsed_record_count"])
        seen: set[str] = set()
        for index, row in enumerate(iter_jsonl(self.layout.path("parsed_records.jsonl")), start=1):
            record_id = str(row.get("parsed_record_id") or "")
            if not record_id or record_id in seen:
                self._failure("record_input_id_invalid", row, f"Missing or duplicate parsed_record_id: {record_id}")
                result = MappingResult("input_validation", "failed", "missing or duplicate parsed_record_id", [], [])
            else:
                seen.add(record_id)
                try:
                    self._validate_input_provenance(row)
                    result = self._map_record(row)
                except Exception as exc:
                    self._failure("record_mapping_failed", row, str(exc))
                    result = MappingResult("source_specific_record_mapper", "failed", str(exc), [], [])
            evidence_count, pointer_count = self._emit_mapping(result)
            status = self._status_row(
                row,
                input_id_field="parsed_record_id",
                mapper_name=result.mapper_name,
                status=result.status,
                reason=result.reason,
                evidence_count=evidence_count,
                pointer_count=pointer_count,
            )
            self._write("structured_record_processing_status.jsonl", status)
            self.counts["record_rows"] += 1
            self.record_status_counts[result.status] += 1
            if index % PROGRESS_INTERVAL == 0:
                self._write_progress(progress_path, "records", index, expected)
        self.counts["record_unique_ids"] = len(seen)
        self._write_progress(progress_path, "records", self.counts["record_rows"], expected)

    def _process_tables(self, parse_summary: dict[str, Any], progress_path: Path) -> None:
        expected = int(parse_summary["parsed_table_count"])
        seen: set[str] = set()
        for index, row in enumerate(iter_jsonl(self.layout.path("parsed_tables.jsonl")), start=1):
            table_id = str(row.get("parsed_table_id") or "")
            if not table_id or table_id in seen:
                self._failure("table_input_id_invalid", row, f"Missing or duplicate parsed_table_id: {table_id}")
                status_name = "failed"
                reason = "missing or duplicate parsed_table_id"
            else:
                seen.add(table_id)
                try:
                    self._validate_input_provenance(row)
                    metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
                    table_text = str(row.get("table_text") or "")
                    if metadata.get("llm_eligible") and EVIDENCE_SIGNAL_RE.search(table_text):
                        status_name = "deferred_complex_table"
                        reason = "evidence signal present but no approved source-specific deterministic column mapping"
                    else:
                        status_name = "not_evidence"
                        reason = "table has no approved deterministic evidence mapping"
                except Exception as exc:
                    self._failure("table_mapping_failed", row, str(exc))
                    status_name = "failed"
                    reason = str(exc)
            status = self._status_row(
                row,
                input_id_field="parsed_table_id",
                mapper_name="source_specific_table_router",
                status=status_name,
                reason=reason,
                evidence_count=0,
                pointer_count=0,
            )
            self._write("structured_table_processing_status.jsonl", status)
            self.counts["table_rows"] += 1
            self.table_status_counts[status_name] += 1
            if index % PROGRESS_INTERVAL == 0:
                self._write_progress(progress_path, "tables", index, expected)
        self.counts["table_unique_ids"] = len(seen)
        self._write_progress(progress_path, "tables", self.counts["table_rows"], expected)

    def _process_chunks(self, parse_summary: dict[str, Any], progress_path: Path) -> None:
        expected = int(parse_summary["text_chunk_count"])
        seen: set[str] = set()
        for index, row in enumerate(iter_jsonl(self.layout.path("text_chunks.jsonl")), start=1):
            chunk_id = str(row.get("chunk_id") or "")
            evidence: list[dict[str, Any]] = []
            pointers: list[dict[str, Any]] = []
            if not chunk_id or chunk_id in seen:
                self._failure("chunk_input_id_invalid", row, f"Missing or duplicate chunk_id: {chunk_id}")
                decision = None
                status_name = "failed"
                reason = "missing or duplicate chunk_id"
            else:
                seen.add(chunk_id)
                try:
                    self._validate_input_provenance(row)
                    decision = classify_chunk(row, min_chars=self.classifier_min_chars, existing_evidence_count=0)
                    if should_run_rule_extraction(row):
                        evidence, pointers = self._extract_chunk(row)
                    emitted_evidence, emitted_pointers = self._emit_candidates(evidence, pointers)
                    if emitted_evidence:
                        status_name = "deterministic_evidence_extracted"
                        reason = "high-precision deterministic evidence rule matched"
                    elif emitted_pointers:
                        status_name = "topic_pointer_only"
                        reason = "keyword-only signal retained as a non-evidence routing pointer"
                    else:
                        status_name = decision.status
                        reason = decision.reason
                    status = self._status_row(
                        row,
                        input_id_field="chunk_id",
                        mapper_name="chunk_classifier_and_deterministic_rules",
                        status=status_name,
                        reason=reason,
                        evidence_count=emitted_evidence,
                        pointer_count=emitted_pointers,
                    )
                    status.update(
                        {
                            "classifier_status": decision.status,
                            "classifier_action": decision.action,
                            "llm_required": decision.llm_required,
                            "evidence_signal": decision.evidence_signal,
                            "natural_language_score": decision.natural_language_score,
                            "machine_json_score": decision.machine_json_score,
                        }
                    )
                    self._write("chunk_processing_status.jsonl", status)
                    self.counts["chunk_rows"] += 1
                    self.chunk_status_counts[status_name] += 1
                    if decision.llm_required:
                        self.counts["llm_required_chunks"] += 1
                    if index % PROGRESS_INTERVAL == 0:
                        self._write_progress(progress_path, "chunks", index, expected)
                    continue
                except Exception as exc:
                    self._failure("chunk_mapping_failed", row, str(exc))
                    decision = None
                    status_name = "failed"
                    reason = str(exc)
            status = self._status_row(
                row,
                input_id_field="chunk_id",
                mapper_name="chunk_classifier_and_deterministic_rules",
                status=status_name,
                reason=reason,
                evidence_count=0,
                pointer_count=0,
            )
            status["llm_required"] = False
            self._write("chunk_processing_status.jsonl", status)
            self.counts["chunk_rows"] += 1
            self.chunk_status_counts[status_name] += 1
            if index % PROGRESS_INTERVAL == 0:
                self._write_progress(progress_path, "chunks", index, expected)
        self.counts["chunk_unique_ids"] = len(seen)
        self._write_progress(progress_path, "chunks", self.counts["chunk_rows"], expected)

    def _map_record(self, row: dict[str, Any]) -> MappingResult:
        source_id = str(row.get("source_id") or "")
        if source_id == "pubchem":
            return self._map_pubchem(row)
        if source_id == "chembl":
            return self._map_chembl(row)
        if source_id == "osp_pbpk_model_library":
            return self._map_osp_models(row)
        if source_id == "dailymed" and row.get("record_type") == "dailymed_spl":
            return self._map_dailymed_label_context(row)
        if source_id == "rxnorm":
            return self._map_rxnorm_identity_context(row)
        if source_id == "pkdb":
            return self._map_pkdb_context(row)
        return MappingResult(
            "source_specific_record_router",
            "not_evidence",
            "parsed metadata or registry record has no approved deterministic evidence mapping",
            [],
            [],
        )

    def _map_pubchem(self, row: dict[str, Any]) -> MappingResult:
        payload = row.get("record_json")
        properties = ((payload.get("PropertyTable") or {}).get("Properties") or []) if isinstance(payload, dict) else []
        evidence: list[dict[str, Any]] = []
        compound = clean_optional(row.get("compound"))
        for property_index, prop in enumerate(properties):
            if not isinstance(prop, dict):
                continue
            for field, (numeric, unit) in PUBCHEM_FIELDS.items():
                raw_value = prop.get(field)
                if raw_value in (None, ""):
                    continue
                value = as_float(raw_value) if numeric else None
                evidence.append(
                    self._make_evidence(
                        row,
                        evidence_type="physicochemical",
                        source_text=str(raw_value),
                        source_location=json_field_location(row, f"$.PropertyTable.Properties[{property_index}].{field}"),
                        extraction_method="source_structured_field",
                        extractor_name="pubchem_property_mapper",
                        parameter=field,
                        value=value,
                        value_text=str(raw_value),
                        unit=unit,
                        confidence=0.95,
                        compound=compound,
                        compound_assignment_status="structured_single_compound_name_resolution",
                        metadata={"pubchem_cid": prop.get("CID"), "json_field": field},
                    )
                )
        if evidence:
            return MappingResult("pubchem_property_mapper", "mapped_to_evidence", "explicit PubChem property fields mapped", evidence, [])
        return MappingResult("pubchem_property_mapper", "not_evidence", "record is not a PubChem property response", [], [])

    def _map_chembl(self, row: dict[str, Any]) -> MappingResult:
        payload = row.get("record_json")
        molecules = payload.get("molecules") if isinstance(payload, dict) else None
        compound = clean_optional(row.get("compound"))
        evidence: list[dict[str, Any]] = []
        exact_count = 0
        if isinstance(molecules, list) and compound:
            target = normalize_identity(compound)
            for molecule_index, molecule in enumerate(molecules):
                if not isinstance(molecule, dict) or target not in chembl_identity_names(molecule):
                    continue
                exact_count += 1
                props = molecule.get("molecule_properties") if isinstance(molecule.get("molecule_properties"), dict) else {}
                for field, (numeric, unit) in CHEMBL_FIELDS.items():
                    raw_value = props.get(field)
                    if raw_value in (None, ""):
                        continue
                    evidence.append(
                        self._make_evidence(
                            row,
                            evidence_type="physicochemical",
                            source_text=str(raw_value),
                            source_location=json_field_location(row, f"$.molecules[{molecule_index}].molecule_properties.{field}"),
                            extraction_method="source_structured_field",
                            extractor_name="chembl_exact_molecule_mapper",
                            parameter=field,
                            value=as_float(raw_value) if numeric else None,
                            value_text=str(raw_value),
                            unit=unit,
                            confidence=0.9,
                            compound=compound,
                            compound_assignment_status="structured_exact_chembl_identity",
                            metadata={
                                "chembl_identity_gate": "exact",
                                "molecule_chembl_id": molecule.get("molecule_chembl_id"),
                                "pref_name": molecule.get("pref_name"),
                                "json_field": field,
                            },
                        )
                    )
        if evidence:
            return MappingResult(
                "chembl_exact_molecule_mapper",
                "mapped_to_evidence",
                f"mapped properties from {exact_count} exact ChEMBL molecule match(es)",
                evidence,
                [],
            )
        reason = "no exact ChEMBL identity match" if isinstance(molecules, list) else "record is not a ChEMBL molecule search response"
        return MappingResult("chembl_exact_molecule_mapper", "not_evidence", reason, [], [])

    def _map_osp_models(self, row: dict[str, Any]) -> MappingResult:
        payload = row.get("record_json")
        files = payload.get("target_compound_files") if isinstance(payload, dict) else None
        if not isinstance(files, list):
            return MappingResult("osp_public_model_manifest_mapper", "not_evidence", "record is not the target model repository manifest", [], [])
        evidence: list[dict[str, Any]] = []
        for file_index, item in enumerate(files):
            if not isinstance(item, dict):
                continue
            repository_path = clean_optional(item.get("path"))
            if not repository_path or not repository_path.lower().endswith(".pksim5"):
                continue
            compound = repository_compound_name(repository_path, self.target_compounds)
            actual_asset = self.repository_assets_by_path.get(repository_path)
            if not compound or actual_asset is None:
                continue
            reports = self.repository_reports_by_compound.get(compound, [])
            evidence.append(
                self._make_evidence(
                    row,
                    evidence_type="public_pbpk_model",
                    source_text=repository_path,
                    source_location=json_field_location(row, f"$.target_compound_files[{file_index}].path"),
                    extraction_method="source_structured_field",
                    extractor_name="osp_public_model_manifest_mapper",
                    parameter="model_file",
                    value=None,
                    value_text=repository_path,
                    unit=None,
                    confidence=0.99,
                    compound=compound,
                    compound_assignment_status="structured_exact_repository_directory",
                    metadata={
                        "model_name": PurePosixPath(repository_path).stem,
                        "model_file_path": repository_path,
                        "model_raw_asset_id": actual_asset.get("raw_asset_id"),
                        "model_storage_path": actual_asset.get("storage_path"),
                        "model_checksum": actual_asset.get("checksum"),
                        "model_source_url": actual_asset.get("source_url") or item.get("raw_url"),
                        "software": "PK-Sim",
                        "evaluation_report_paths": [repository_path_for(asset) for asset in reports],
                        "evaluation_report_raw_asset_ids": [asset.get("raw_asset_id") for asset in reports],
                        "compound_match": "exact_repository_directory",
                        "model_reproduced": False,
                    },
                )
            )
        if evidence:
            return MappingResult("osp_public_model_manifest_mapper", "mapped_to_evidence", "downloaded exact-match PK-Sim model files mapped", evidence, [])
        return MappingResult("osp_public_model_manifest_mapper", "not_evidence", "no downloaded exact-match model file in this manifest", [], [])

    def _map_dailymed_label_context(self, row: dict[str, Any]) -> MappingResult:
        payload = row.get("record_json") if isinstance(row.get("record_json"), dict) else {}
        pointers: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for field in ("active_ingredients", "active_moieties"):
            values = payload.get(field)
            if not isinstance(values, list):
                continue
            for index, item in enumerate(values):
                name = clean_optional(item.get("name")) if isinstance(item, dict) else clean_optional(item)
                if not name or normalize_identity(name) in seen_names:
                    continue
                seen_names.add(normalize_identity(name))
                compound = exact_target_name(name, self.target_compounds)
                if not compound:
                    continue
                pointers.append(
                    self._make_pointer(
                        row,
                        topic="label",
                        matched_term=name,
                        source_text=name,
                        source_location=json_field_location(row, f"$.{field}[{index}].name"),
                        extractor_name="dailymed_active_ingredient_mapper",
                        confidence=0.99,
                        compound=compound,
                        compound_assignment_status="structured_exact_active_ingredient",
                        metadata={"set_id": payload.get("set_id"), "title": payload.get("title"), "field": field},
                    )
                )
        if pointers:
            return MappingResult("dailymed_active_ingredient_mapper", "mapped_to_pointer", "exact active ingredient retained as label context", [], pointers)
        return MappingResult("dailymed_active_ingredient_mapper", "not_evidence", "no exact target active ingredient in label record", [], [])

    def _map_rxnorm_identity_context(self, row: dict[str, Any]) -> MappingResult:
        payload = row.get("record_json") if isinstance(row.get("record_json"), dict) else {}
        identifiers = ((payload.get("idGroup") or {}).get("rxnormId") or []) if isinstance(payload.get("idGroup"), dict) else []
        compound = clean_optional(row.get("compound"))
        pointers: list[dict[str, Any]] = []
        for index, identifier in enumerate(identifiers if isinstance(identifiers, list) else []):
            if identifier in (None, ""):
                continue
            pointers.append(
                self._make_pointer(
                    row,
                    topic="identity",
                    matched_term="RxCUI",
                    source_text=str(identifier),
                    source_location=json_field_location(row, f"$.idGroup.rxnormId[{index}]"),
                    extractor_name="rxnorm_identifier_mapper",
                    confidence=0.99,
                    compound=compound,
                    compound_assignment_status="structured_single_compound_name_resolution",
                    metadata={"identifier_system": "RxNorm", "rxcui": str(identifier)},
                )
            )
        if pointers:
            return MappingResult("rxnorm_identifier_mapper", "mapped_to_pointer", "RxCUI retained as identity context", [], pointers)
        return MappingResult("rxnorm_identifier_mapper", "not_evidence", "record is not an RxCUI response", [], [])

    def _map_pkdb_context(self, row: dict[str, Any]) -> MappingResult:
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        request_key = str(metadata.get("request_key") or "")
        payload = row.get("record_json") if isinstance(row.get("record_json"), dict) else {}
        if not request_key.startswith("interventions_"):
            return MappingResult("pkdb_study_context_mapper", "not_evidence", "PK-DB record is not an intervention context response", [], [])
        data_wrapper = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        items = data_wrapper.get("data") if isinstance(data_wrapper.get("data"), list) else []
        pointers: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            substance = item.get("substance") if isinstance(item.get("substance"), dict) else {}
            compound = exact_target_name(str(substance.get("name") or substance.get("label") or ""), self.target_compounds)
            value = as_float(item.get("value"))
            unit = clean_optional(item.get("unit"))
            if not compound or value is None or not unit:
                continue
            pointers.append(
                self._make_pointer(
                    row,
                    topic="clinical_pk",
                    matched_term="administered_dose",
                    source_text=str(item.get("value")),
                    source_location=json_field_location(row, f"$.data.data[{index}].value"),
                    extractor_name="pkdb_intervention_context_mapper",
                    confidence=0.98,
                    compound=compound,
                    compound_assignment_status="structured_exact_intervention_substance",
                    metadata={
                        "dose_value": value,
                        "dose_unit": unit,
                        "route": nested_label(item.get("route")),
                        "formulation": nested_label(item.get("form")),
                        "application": nested_label(item.get("application")),
                        "study": nested_label(item.get("study")),
                        "pkdb_intervention_id": item.get("pk"),
                    },
                )
            )
        if pointers:
            return MappingResult("pkdb_intervention_context_mapper", "mapped_to_pointer", "structured dose retained as study context, not PK outcome evidence", [], pointers)
        return MappingResult("pkdb_intervention_context_mapper", "not_evidence", "no exact target intervention dose in record", [], [])

    def _extract_chunk(self, row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        text = str(row.get("chunk_text") or "")
        evidence: list[dict[str, Any]] = []
        pointers: list[dict[str, Any]] = []
        local_evidence: set[str] = set()
        local_pointers: set[str] = set()

        for parameter, pattern in PK_PATTERNS:
            for match in pattern.finditer(text):
                source_text = _sentence_around(text, match.start(), match.end())
                if not source_text or source_text not in text or _looks_like_numeric_context(source_text):
                    continue
                value = as_float(match.group(1))
                unit = clean_unit(match.group(2) if len(match.groups()) >= 2 else None)
                if not valid_pk_value_and_unit(parameter, value, unit):
                    continue
                compound, assignment = self._assign_narrative_compound(source_text)
                rejection_reason = pk_narrative_rejection_reason(
                    parameter,
                    source_text,
                    compound,
                    value_text=match.group(1),
                    unit=unit,
                    rule_match=match.group(0),
                    source_location=str(row.get("source_location") or ""),
                )
                if rejection_reason or compound is None:
                    pointer = self._make_pointer(
                        row,
                        topic="clinical_pk",
                        matched_term=parameter,
                        source_text=source_text,
                        source_location=str(row.get("source_location") or ""),
                        extractor_name="pk_numeric_rule_review_pointer",
                        confidence=0.45,
                        compound=compound,
                        compound_assignment_status=assignment,
                        metadata={
                            "candidate_value": value,
                            "candidate_unit": unit,
                            "routing_reason": rejection_reason or assignment,
                            "value_qualifier": pk_value_qualifier(match.group(0)),
                        },
                    )
                    if pointer["pointer_id"] not in local_pointers:
                        local_pointers.add(pointer["pointer_id"])
                        pointers.append(pointer)
                    continue
                candidate = self._make_evidence(
                    row,
                    evidence_type="clinical_pk",
                    source_text=source_text,
                    source_location=str(row.get("source_location") or ""),
                    extraction_method="rule_based",
                    extractor_name="pk_numeric_regex_high_precision",
                    parameter=parameter,
                    value=value,
                    value_text=match.group(1),
                    unit=unit,
                    confidence=0.62,
                    compound=compound,
                    compound_assignment_status=assignment,
                    metadata={
                        "rule_match": match.group(0),
                        "unit_gate": "parameter_dimension_match",
                        "value_qualifier": pk_value_qualifier(match.group(0)),
                    },
                )
                if candidate["evidence_id"] not in local_evidence:
                    local_evidence.add(candidate["evidence_id"])
                    evidence.append(candidate)

        for ddi in extract_ddi_statements(text, source_id=str(row.get("source_id") or "")):
            if not ddi.evidence_text or ddi.evidence_text not in text:
                continue
            compound, assignment = self._assign_narrative_compound(ddi.evidence_text)
            validated_relation_role = (
                explicit_compound_target_relation(ddi.evidence_text, compound, ddi.enzyme_or_transporter)
                if compound and ddi.enzyme_or_transporter and not ddi.ambiguous_target
                else None
            )
            relation_rejection_reason = ddi_relation_rejection_reason(
                ddi.evidence_text,
                compound,
                ddi.enzyme_or_transporter,
                validated_relation_role,
            )
            if validated_relation_role is None or relation_rejection_reason is not None:
                pointer = self._make_pointer(
                    row,
                    topic="ddi",
                    matched_term=ddi.enzyme_or_transporter or ddi.interaction_type,
                    source_text=ddi.evidence_text,
                    source_location=str(row.get("source_location") or ""),
                    extractor_name="ddi_relation_rule_review_pointer",
                    confidence=0.4,
                    compound=compound,
                    compound_assignment_status=assignment,
                    metadata={
                        "interaction_type": ddi.interaction_type,
                        "all_targets": list(ddi.all_targets),
                        "ambiguous_target": ddi.ambiguous_target,
                        "routing_reason": relation_rejection_reason
                        or ddi_pointer_reason(compound, ddi.enzyme_or_transporter, ddi.ambiguous_target),
                    },
                )
                if pointer["pointer_id"] not in local_pointers:
                    local_pointers.add(pointer["pointer_id"])
                    pointers.append(pointer)
                continue
            ratio = ddi.auc_ratio if ddi.auc_ratio is not None else ddi.cmax_ratio
            candidate = self._make_evidence(
                row,
                evidence_type="ddi",
                source_text=ddi.evidence_text,
                source_location=str(row.get("source_location") or ""),
                extraction_method="rule_based",
                extractor_name="ddi_relation_regex",
                parameter=ddi.enzyme_or_transporter or ddi.interaction_type,
                value=ratio,
                value_text=str(ratio) if ratio is not None else None,
                unit="ratio" if ratio is not None else None,
                confidence=0.35 if ddi.ambiguous_target else 0.55,
                compound=compound,
                compound_assignment_status=assignment,
                condition=ddi.population,
                metadata={
                    "interaction_type": f"{ddi.enzyme_or_transporter} {validated_relation_role}",
                    "raw_rule_interaction_type": ddi.interaction_type,
                    "validated_relation_role": validated_relation_role,
                    "auc_ratio": ddi.auc_ratio,
                    "cmax_ratio": ddi.cmax_ratio,
                    "study_type": ddi.study_type,
                    "all_targets": list(ddi.all_targets),
                    "ambiguous_target": ddi.ambiguous_target,
                    "requires_relation_review": True,
                },
            )
            if candidate["evidence_id"] not in local_evidence:
                local_evidence.add(candidate["evidence_id"])
                evidence.append(candidate)

        source_id = str(row.get("source_id") or "")
        pointer_specs: list[tuple[str, str, re.Match[str] | None, float]] = []
        if source_id in SECTION_CANDIDATE_SOURCES:
            pointer_specs.append(("adme", "adme_topic_pointer", _earliest_match(text, ADME_ACRONYM_TERMS, ADME_WORD_TERMS), 0.45))
            pointer_specs.append(("special_population", "special_population_topic_pointer", SPECIAL_POP_TERMS.search(text), 0.45))
        if source_id in {"pubmed", "pmc_oa", "osp_pbpk_model_library", "osp_tools_docs"}:
            pointer_specs.append(("public_pbpk_model", "public_model_topic_pointer", PBPK_TERMS.search(text), 0.4))
        for topic, extractor_name, match, confidence in pointer_specs:
            if match is None:
                continue
            source_text = _sentence_around(text, match.start(), match.end())
            if not source_text or source_text not in text:
                continue
            compound, assignment = self._assign_narrative_compound(source_text)
            pointer = self._make_pointer(
                row,
                topic=topic,
                matched_term=_normalize_parameter(match.group(0)) or match.group(0),
                source_text=source_text,
                source_location=str(row.get("source_location") or ""),
                extractor_name=extractor_name,
                confidence=confidence,
                compound=compound,
                compound_assignment_status=assignment,
                metadata={"note": "topic pointer only; no numeric value or asserted relation"},
            )
            if pointer["pointer_id"] not in local_pointers:
                local_pointers.add(pointer["pointer_id"])
                pointers.append(pointer)
        return evidence, pointers

    def _make_evidence(
        self,
        row: dict[str, Any],
        *,
        evidence_type: str,
        source_text: str,
        source_location: str,
        extraction_method: str,
        extractor_name: str,
        parameter: str | None,
        value: float | None,
        value_text: str | None,
        unit: str | None,
        confidence: float,
        compound: str | None,
        compound_assignment_status: str,
        condition: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_text = source_text.strip()
        if not source_text:
            raise ValueError("evidence source_text is empty")
        canonical_value, canonical_unit = to_canonical_value(value, unit)
        normalized_unit = normalize_unit(unit)
        provenance = self._provenance(row)
        upstream_id = str(
            row.get("parsed_record_id")
            or row.get("parsed_table_id")
            or row.get("chunk_id")
            or row.get("parsed_document_id")
            or ""
        )
        evidence_id = stable_id(
            "evidence",
            evidence_type,
            str(row.get("raw_asset_id") or ""),
            upstream_id,
            source_location,
            parameter or "",
            value_text or "",
            hash_text(source_text),
        )
        return {
            **provenance,
            "evidence_id": evidence_id,
            "evidence_type": evidence_type,
            "compound": compound,
            "retrieval_compounds": self._retrieval_compounds(row),
            "compound_assignment_status": compound_assignment_status,
            "source_location": source_location,
            "source_text": source_text,
            "source_text_hash": hash_text(source_text),
            "evidence_tier": "quantitative" if finite(value) else "qualitative",
            "extraction_method": extraction_method,
            "extractor_name": extractor_name,
            "extractor_version": EXTRACTOR_VERSION,
            "extraction_confidence": confidence,
            "curation_status": "needs_review",
            "parameter": parameter,
            "value": value if finite(value) else None,
            "value_text": value_text,
            "unit": unit,
            "normalized_unit": normalized_unit.normalized_unit,
            "canonical_value": canonical_value if finite(canonical_value) else None,
            "canonical_unit": canonical_unit,
            "unit_dimension": normalized_unit.dimension,
            "condition": condition,
            "population": None,
            "route": None,
            "dose": None,
            "metadata_json": metadata or {},
            "created_at": self.started_at,
        }

    def _make_pointer(
        self,
        row: dict[str, Any],
        *,
        topic: str,
        matched_term: str,
        source_text: str,
        source_location: str,
        extractor_name: str,
        confidence: float,
        compound: str | None,
        compound_assignment_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_text = source_text.strip()
        provenance = self._provenance(row)
        upstream_id = str(
            row.get("parsed_record_id")
            or row.get("parsed_table_id")
            or row.get("chunk_id")
            or row.get("parsed_document_id")
            or ""
        )
        return {
            **provenance,
            "pointer_id": stable_id(
                "pointer",
                topic,
                str(row.get("raw_asset_id") or ""),
                upstream_id,
                source_location,
                matched_term,
                hash_text(source_text),
            ),
            "evidence_tier": "pointer",
            "topic": topic,
            "matched_term": matched_term,
            "compound": compound,
            "retrieval_compounds": self._retrieval_compounds(row),
            "compound_assignment_status": compound_assignment_status,
            "source_location": source_location,
            "source_text": source_text,
            "source_text_hash": hash_text(source_text),
            "extractor_name": extractor_name,
            "extractor_version": EXTRACTOR_VERSION,
            "confidence": confidence,
            "metadata_json": metadata or {},
            "created_at": self.started_at,
        }

    def _provenance(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_id": row.get("source_id"),
            "raw_asset_id": row.get("raw_asset_id"),
            "parsed_record_id": row.get("parsed_record_id"),
            "parsed_document_id": row.get("parsed_document_id"),
            "parsed_table_id": row.get("parsed_table_id"),
            "chunk_id": row.get("chunk_id"),
            "storage_path": row.get("storage_path"),
            "source_url": row.get("source_url"),
            "checksum": row.get("checksum"),
        }

    def _retrieval_compounds(self, row: dict[str, Any]) -> list[str]:
        compounds = set(self.alias_compounds.get(str(row.get("raw_asset_id") or ""), set()))
        row_compound = clean_optional(row.get("compound"))
        if row_compound:
            compounds.add(row_compound)
        return sorted(compounds, key=str.lower)

    def _assign_narrative_compound(self, source_text: str) -> tuple[str | None, str]:
        matches = [compound for compound in self.target_compounds if contains_exact_name(source_text, compound)]
        if len(matches) == 1:
            return matches[0], "exact_target_name_in_source_text"
        if len(matches) > 1:
            return None, "multiple_target_compounds_in_source_text"
        return None, "unresolved_text_subject"

    def _emit_mapping(self, result: MappingResult) -> tuple[int, int]:
        return self._emit_candidates(result.evidence, result.pointers)

    def _emit_candidates(self, evidence: list[dict[str, Any]], pointers: list[dict[str, Any]]) -> tuple[int, int]:
        evidence_count = 0
        pointer_count = 0
        for row in evidence:
            if not self._valid_candidate_provenance(row):
                self._failure("invalid_evidence_provenance", row, "deterministic evidence is missing required provenance")
                continue
            evidence_id = str(row["evidence_id"])
            if evidence_id in self.seen_evidence_ids:
                self.counts["duplicate_evidence_suppressed"] += 1
                continue
            self.seen_evidence_ids.add(evidence_id)
            self._write("evidence_candidates_deterministic.jsonl", row)
            self._write("evidence_candidates.jsonl", row)
            evidence_count += 1
            self.counts["evidence_rows"] += 1
            self.evidence_type_counts[str(row.get("evidence_type") or "unknown")] += 1
            self.evidence_source_counts[str(row.get("source_id") or "unknown")] += 1
        for row in pointers:
            if not self._valid_candidate_provenance(row):
                self._failure("invalid_pointer_provenance", row, "deterministic pointer is missing required provenance")
                continue
            pointer_id = str(row["pointer_id"])
            if pointer_id in self.seen_pointer_ids:
                self.counts["duplicate_pointer_suppressed"] += 1
                continue
            self.seen_pointer_ids.add(pointer_id)
            self._write("evidence_pointers.jsonl", row)
            pointer_count += 1
            self.counts["pointer_rows"] += 1
            self.pointer_topic_counts[str(row.get("topic") or "unknown")] += 1
        return evidence_count, pointer_count

    def _valid_candidate_provenance(self, row: dict[str, Any]) -> bool:
        required = ("raw_asset_id", "checksum", "source_location", "source_text")
        valid = all(row.get(field) not in (None, "") for field in required)
        if not valid:
            self.provenance_failures += 1
        return valid

    def _validate_input_provenance(self, row: dict[str, Any]) -> None:
        raw_asset_id = clean_optional(row.get("raw_asset_id"))
        if not raw_asset_id or not row.get("checksum") or not row.get("source_location"):
            raise ValueError("parsed input is missing raw_asset_id, checksum, or source_location")
        inventory_row = self.inventory.get(raw_asset_id)
        if inventory_row is None:
            raise ValueError(f"raw_asset_id not found in inventory: {raw_asset_id}")
        if inventory_row.get("checksum") != row.get("checksum"):
            raise ValueError(f"checksum mismatch for parsed input: {raw_asset_id}")

    def _status_row(
        self,
        row: dict[str, Any],
        *,
        input_id_field: str,
        mapper_name: str,
        status: str,
        reason: str,
        evidence_count: int,
        pointer_count: int,
    ) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            input_id_field: row.get(input_id_field),
            "raw_asset_id": row.get("raw_asset_id"),
            "source_id": row.get("source_id"),
            "compound": row.get("compound"),
            "retrieval_compounds": self._retrieval_compounds(row),
            "storage_path": row.get("storage_path"),
            "checksum": row.get("checksum"),
            "source_location": row.get("source_location"),
            "mapper_name": mapper_name,
            "extractor_version": EXTRACTOR_VERSION,
            "processing_status": status,
            "reason": reason,
            "evidence_count": evidence_count,
            "pointer_count": pointer_count,
            "processed_at": self.started_at,
        }

    def _failure(self, failure_type: str, row: dict[str, Any], error: str) -> None:
        failure = {
            "run_id": self.run_id,
            "stage": "deterministic_extract",
            "failure_id": stable_id(
                "deterministic_failure",
                failure_type,
                str(row.get("raw_asset_id") or ""),
                str(row.get("parsed_record_id") or row.get("parsed_table_id") or row.get("chunk_id") or ""),
                error,
            ),
            "failure_type": failure_type,
            "source_id": row.get("source_id"),
            "compound": row.get("compound"),
            "raw_asset_id": row.get("raw_asset_id"),
            "parsed_record_id": row.get("parsed_record_id"),
            "parsed_table_id": row.get("parsed_table_id"),
            "chunk_id": row.get("chunk_id"),
            "storage_path": row.get("storage_path"),
            "checksum": row.get("checksum"),
            "source_location": row.get("source_location"),
            "error": error,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write("processing_failures.jsonl", failure)
        self.counts["stage_failures"] += 1

    def _preserve_non_stage_failures(self, bundle: AtomicJSONLBundle) -> None:
        path = self.layout.path("processing_failures.jsonl")
        if not path.exists():
            return
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bundle.write_raw("processing_failures.jsonl", line)
                    self.counts["preserved_non_stage_failures"] += 1
                    continue
                if row.get("stage") == "deterministic_extract":
                    continue
                bundle.write_raw("processing_failures.jsonl", line)
                self.counts["preserved_non_stage_failures"] += 1

    def _write(self, name: str, row: dict[str, Any]) -> None:
        if self.bundle is None:
            raise RuntimeError("deterministic output bundle is not open")
        self.bundle.write(name, row)

    def _build_summary(
        self,
        parse_summary: dict[str, Any],
        downstream_before: dict[str, dict[str, Any]],
        downstream_after: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        expected_records = int(parse_summary["parsed_record_count"])
        expected_tables = int(parse_summary["parsed_table_count"])
        expected_chunks = int(parse_summary["text_chunk_count"])
        downstream_unchanged = downstream_before == downstream_after
        accepted = bool(
            self.counts["record_rows"] == expected_records
            and self.counts["record_unique_ids"] == expected_records
            and self.counts["table_rows"] == expected_tables
            and self.counts["table_unique_ids"] == expected_tables
            and self.counts["chunk_rows"] == expected_chunks
            and self.counts["chunk_unique_ids"] == expected_chunks
            and self.counts["stage_failures"] == 0
            and self.provenance_failures == 0
            and downstream_unchanged
        )
        return {
            "stage": "deterministic_extract",
            "status": "completed" if accepted else "failed",
            "accepted": accepted,
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "output_dir": str(self.output_dir),
            "extractor_version": EXTRACTOR_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "expected_parsed_record_count": expected_records,
            "structured_record_status_count": self.counts["record_rows"],
            "unique_parsed_record_id_count": self.counts["record_unique_ids"],
            "expected_parsed_table_count": expected_tables,
            "structured_table_status_count": self.counts["table_rows"],
            "unique_parsed_table_id_count": self.counts["table_unique_ids"],
            "expected_text_chunk_count": expected_chunks,
            "chunk_processing_status_count": self.counts["chunk_rows"],
            "unique_chunk_id_count": self.counts["chunk_unique_ids"],
            "record_status_counts": dict(sorted(self.record_status_counts.items())),
            "table_status_counts": dict(sorted(self.table_status_counts.items())),
            "chunk_status_counts": dict(sorted(self.chunk_status_counts.items())),
            "evidence_candidate_count": self.counts["evidence_rows"],
            "evidence_counts_by_type": dict(sorted(self.evidence_type_counts.items())),
            "evidence_counts_by_source": dict(sorted(self.evidence_source_counts.items())),
            "evidence_pointer_count": self.counts["pointer_rows"],
            "evidence_pointer_counts_by_topic": dict(sorted(self.pointer_topic_counts.items())),
            "llm_required_chunk_count": self.counts["llm_required_chunks"],
            "duplicate_evidence_suppressed_count": self.counts["duplicate_evidence_suppressed"],
            "duplicate_pointer_suppressed_count": self.counts["duplicate_pointer_suppressed"],
            "deterministic_failure_count": self.counts["stage_failures"],
            "provenance_failure_count": self.provenance_failures,
            "downstream_files_unchanged": downstream_unchanged,
            "llm_requests_created": 0,
            "llm_responses_created": 0,
            "embeddings_created": 0,
            "downstream_file_snapshot_before": downstream_before,
            "downstream_file_snapshot_after": downstream_after,
        }

    def _update_processing_summary(self, summary: dict[str, Any], parse_summary: dict[str, Any]) -> None:
        path = self.layout.path("processing_summary.json", for_write=True)
        payload: dict[str, Any] = {}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(
            {
                "run_id": self.run_id,
                "run_root": str(self.run_root),
                "output_dir": str(self.output_dir),
                "current_stage": "deterministic_extract",
                "raw_asset_count": parse_summary.get("expected_raw_asset_count"),
                "parsed_document_count": parse_summary.get("parsed_document_count"),
                "parsed_record_count": parse_summary.get("parsed_record_count"),
                "parsed_table_count": parse_summary.get("parsed_table_count"),
                "text_chunk_count": parse_summary.get("text_chunk_count"),
                "evidence_candidate_count": summary["evidence_candidate_count"],
                "evidence_counts_by_type": summary["evidence_counts_by_type"],
                "evidence_pointer_count": summary["evidence_pointer_count"],
                "evidence_pointer_counts_by_topic": summary["evidence_pointer_counts_by_topic"],
                "deterministic_extract_accepted": summary["accepted"],
                "generated_at": summary["generated_at"],
            }
        )
        write_json_atomic(path, payload)

    def _update_pipeline_state(self, state: dict[str, Any], summary: dict[str, Any]) -> None:
        stages = state.setdefault("stages", {})
        stages["deterministic_extract"] = {
            "status": "completed" if summary["accepted"] else "failed",
            "accepted": summary["accepted"],
            "summary_path": str(self.layout.path("deterministic_extract_summary.json", for_write=True)),
            "completed_at": summary["generated_at"],
        }
        for downstream in ("llm_extract", "completeness", "embedding"):
            stages[downstream] = {"status": "not_started"}
        state["current_stage"] = "deterministic_extract"
        write_json_atomic(self.layout.path("pipeline_state.json", for_write=True), state)

    def _write_failed_state(self, state: dict[str, Any], progress_path: Path, exc: Exception) -> None:
        failed_at = datetime.now(timezone.utc).isoformat()
        stages = state.setdefault("stages", {})
        stages["deterministic_extract"] = {
            "status": "failed",
            "accepted": False,
            "error": str(exc),
            "completed_at": failed_at,
        }
        state["current_stage"] = "deterministic_extract"
        write_json_atomic(self.layout.path("pipeline_state.json", for_write=True), state)
        write_json_atomic(
            progress_path,
            {
                "stage": "deterministic_extract",
                "status": "failed",
                "run_id": self.run_id,
                "error": str(exc),
                "updated_at": failed_at,
            },
        )

    def _write_progress(self, path: Path, phase: str, processed: int, expected: int, *, status: str = "in_progress") -> None:
        write_json_atomic(
            path,
            {
                "stage": "deterministic_extract",
                "status": status,
                "phase": phase,
                "run_id": self.run_id,
                "processed": processed,
                "expected": expected,
                "record_rows": self.counts["record_rows"],
                "table_rows": self.counts["table_rows"],
                "chunk_rows": self.counts["chunk_rows"],
                "evidence_rows": self.counts["evidence_rows"],
                "pointer_rows": self.counts["pointer_rows"],
                "failure_rows": self.counts["stage_failures"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="strict") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def snapshot_files(layout: RawProcessingLayout, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = layout.path(name)
        if path.exists():
            stat = path.stat()
            result[name] = {"exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        else:
            result[name] = {"exists": False, "size": 0, "mtime_ns": None}
    return result


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_id(*parts: str) -> str:
    return str(uuid5(NAMESPACE_URL, "|".join(parts)))


def hash_text(text: str) -> str:
    return sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        result = float(str(value).replace(",", ""))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def clean_unit(unit: Any) -> str | None:
    cleaned = clean_optional(unit)
    if cleaned is None:
        return None
    return cleaned.strip(".,;:)]}")


def valid_pk_value_and_unit(parameter: str, value: float | None, unit: str | None) -> bool:
    if value is None or value <= 0 or unit is None:
        return False
    normalized = normalize_unit(unit)
    allowed = PARAMETER_UNIT_DIMENSIONS.get(parameter, set())
    if normalized.dimension not in allowed:
        return False
    if parameter == "bioavailability" and value > 100:
        return False
    return True


def normalize_identity(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def chembl_identity_names(molecule: dict[str, Any]) -> set[str]:
    names: set[Any] = {molecule.get("pref_name"), molecule.get("molecule_chembl_id")}
    synonyms = molecule.get("molecule_synonyms")
    if isinstance(synonyms, list):
        for synonym in synonyms:
            if isinstance(synonym, dict):
                names.add(synonym.get("molecule_synonym"))
                names.add(synonym.get("synonyms"))
            else:
                names.add(synonym)
    return {normalize_identity(str(name)) for name in names if name}


def exact_target_name(value: str, targets: set[str]) -> str | None:
    normalized = normalize_identity(value)
    matches = [target for target in targets if normalize_identity(target) == normalized]
    return sorted(matches, key=str.lower)[0] if len(matches) == 1 else None


def contains_exact_name(text: str, compound: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9-]){re.escape(compound)}(?![A-Za-z0-9-])", text, flags=re.IGNORECASE) is not None


def pk_narrative_rejection_reason(
    parameter: str,
    source_text: str,
    compound: str | None,
    *,
    value_text: str | None = None,
    unit: str | None = None,
    rule_match: str | None = None,
    source_location: str | None = None,
) -> str | None:
    if compound is None:
        return "target_compound_not_uniquely_resolved_in_source_text"
    if parameter == "clearance" and RENAL_FUNCTION_CLEARANCE_RE.search(source_text):
        return "subject_renal_function_not_drug_clearance"
    if parameter == "clearance" and DIALYSIS_CLEARANCE_RE.search(source_text):
        return "dialysis_context_not_systemic_drug_clearance"
    if source_location and MODEL_INTERNAL_LOCATION_RE.search(source_location):
        return "model_configuration_not_clinical_observation"
    if NON_OBSERVED_PK_CONTEXT_RE.search(source_text):
        return "model_or_threshold_context_not_clinical_observation"
    if relative_pk_change(source_text, value_text=value_text, unit=unit):
        return "relative_change_not_absolute_pk_parameter"
    if parameter == "bioavailability" and RELATIVE_BIOAVAILABILITY_RE.search(source_text):
        return "relative_change_not_absolute_pk_parameter"
    if METABOLITE_SUBJECT_RE.search(source_text):
        return "metabolite_subject_requires_resolution"
    if rule_match is not None and not explicit_compound_pk_relation(
        source_text,
        compound,
        parameter=parameter,
        rule_match=rule_match,
        value_text=value_text,
    ):
        return "pk_subject_parameter_value_relation_unresolved"
    return None


def explicit_compound_pk_relation(
    source_text: str,
    compound: str,
    *,
    parameter: str,
    rule_match: str,
    value_text: str | None,
) -> bool:
    """Require a local compound -> parameter -> value relation, not sentence co-occurrence."""
    if not source_text or not compound or not rule_match or not value_text:
        return False
    match_occurrences = list(re.finditer(re.escape(rule_match), source_text, flags=re.IGNORECASE))
    if not match_occurrences:
        return False
    compound_occurrences = list(
        re.finditer(
            rf"(?<![A-Za-z0-9-]){re.escape(compound)}(?![A-Za-z0-9-])",
            source_text,
            flags=re.IGNORECASE,
        )
    )
    if not compound_occurrences:
        return False
    value_offset = rule_match.lower().rfind(value_text.lower())
    if value_offset < 0:
        return False
    prefix_after_parameter = PK_PARAMETER_TOKEN_RE.sub("", rule_match[:value_offset], count=1)
    if INTERVENING_PK_FIELD_RE.search(prefix_after_parameter):
        return False

    for rule_occurrence in match_occurrences:
        preceding_context = source_text[max(0, rule_occurrence.start() - 80) : rule_occurrence.start()]
        if parallel_pk_parameter_context(preceding_context):
            continue
        value_start = rule_occurrence.start() + value_offset
        for compound_occurrence in compound_occurrences:
            if rule_occurrence.start() <= compound_occurrence.start() < value_start:
                return True
            if compound_occurrence.end() <= rule_occurrence.start():
                bridge = source_text[compound_occurrence.end() : rule_occurrence.start()]
                if len(bridge) > PK_LOCAL_RELATION_MAX_BRIDGE:
                    continue
                if re.search(r"[.;]", bridge):
                    continue
                if PK_PARAMETER_TOKEN_RE.search(bridge):
                    continue
                if PK_SUBJECT_SHIFT_BRIDGE_RE.search(bridge):
                    continue
                return True
    return False


def parallel_pk_parameter_context(preceding_context: str) -> bool:
    matches = list(PK_PARAMETER_TOKEN_RE.finditer(preceding_context))
    if not matches:
        return False
    tail = preceding_context[matches[-1].start() :]
    return len(matches) > 1 or re.search(r"(?:,|\band\b|\bor\b)", tail, flags=re.IGNORECASE) is not None


def pk_value_qualifier(rule_match: str) -> str:
    if re.search(r"\b(?:under|below|less than|up to)\b|[<≤]", rule_match, flags=re.IGNORECASE):
        return "upper_bound"
    if re.search(r"\b(?:over|above|more than|at least)\b|[>≥]", rule_match, flags=re.IGNORECASE):
        return "lower_bound"
    if re.search(r"\b(?:about|approximately|around|roughly)\b|~", rule_match, flags=re.IGNORECASE):
        return "approximate"
    return "reported_value"


def relative_pk_change(source_text: str, *, value_text: str | None, unit: str | None) -> bool:
    if not value_text or not unit:
        return False
    numeric = rf"{re.escape(value_text)}\s*{re.escape(unit)}"
    change_words = r"(?:increase|increased|decrease|decreased|reduction|reduced|higher|lower|later|earlier|difference|change|changed|prolonged|shortened)"
    after = rf"{numeric}[^.;]{{0,30}}\b{change_words}\b"
    before = rf"\b{change_words}\b[^.;]{{0,45}}(?:\b(?:by|from|to)\b[^.;]{{0,20}})?{numeric}"
    return re.search(after, source_text, flags=re.IGNORECASE) is not None or re.search(before, source_text, flags=re.IGNORECASE) is not None


def explicit_compound_target_relation(source_text: str, compound: str, target: str) -> str | None:
    compound_pattern = rf"(?<![A-Za-z0-9-]){re.escape(compound)}(?![A-Za-z0-9-])"
    target_pattern = ddi_target_pattern(target)
    flags = re.IGNORECASE
    qualifier = (
        r"(?:(?:well[- ]recogni[sz]ed|recogni[sz]ed|known|sensitive|probe|model|phenotyping|specific|"
        r"highly\s+permeable|primary|major)\s+)*"
    )
    role_patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "substrate",
            (
                rf"{compound_pattern}\s*[,(:-]?\s*(?:(?:(?:is|was)\s+used\s+as|acts?\s+as|is|was|as)\s+)?(?:a|an)?\s*{qualifier}{target_pattern}\s+{qualifier}substrates?\b",
                rf"{compound_pattern}\s*[,(:-]?\s*(?:(?:(?:is|was)\s+used\s+as|acts?\s+as|is|was|as)\s+)?(?:a|an)?\s*{qualifier}substrates?\s+(?:of|for)\s+(?:intestinal\s+)?{target_pattern}\b",
                rf"{compound_pattern}\s+(?:is|was)\s+one\s+of\s+(?:the\s+)?(?:many\s+)?{target_pattern}\s+substrates?\b",
                rf"{compound_pattern}\s+has\s+(?:the\s+)?characteristics\s+of\s+(?:a|an)\s+{qualifier}{target_pattern}\s+{qualifier}substrate\b",
                rf"{compound_pattern}\s+(?:is|was)?\s*(?:primarily\s+|mainly\s+)?metaboli[sz]ed\s+by\s+(?:the\s+)?{target_pattern}\b",
                rf"{compound_pattern}\s+(?:is|was)?\s*(?:primarily\s+|mainly\s+)?(?:transported|cleared)\s+by\s+(?:the\s+)?{target_pattern}\b",
                rf"(?:metabolism|transport|clearance)\s+of\s+{compound_pattern}\s+(?:is|was)\s+(?:primarily\s+|mainly\s+)?(?:mediated|performed)\s+by\s+(?:the\s+)?{target_pattern}\b",
            ),
        ),
        (
            "inhibitor",
            (
                rf"{compound_pattern}\s*[,(:-]?\s*(?:is|was|acts?\s+as)\s+(?:a|an)?\s*(?:potent\s+|strong\s+|weak\s+)?inhibitor\s+of\s+(?:the\s+)?{target_pattern}\b",
                rf"{compound_pattern}\s*,?\s*(?:a|an)\s+(?:potent\s+|strong\s+|weak\s+)?inhibitor\s+of\s+(?:the\s+)?{target_pattern}\b",
                rf"{compound_pattern}\s+(?:potently\s+|strongly\s+|weakly\s+)?inhibits?\s+(?:the\s+)?{target_pattern}\b",
                rf"inhibition\s+of\s+(?:the\s+)?{target_pattern}(?:\s+enzyme)?\s+by\s+{compound_pattern}\b",
                rf"(?:ability|capacity)\s+of\s+{compound_pattern}\s+to\s+inhibit\s+(?:the\s+)?{target_pattern}\b",
            ),
        ),
        (
            "inducer",
            (
                rf"{compound_pattern}\s*[,(:-]?\s*(?:is|was|acts?\s+as)\s+(?:a|an)?\s*(?:potent\s+|strong\s+|weak\s+)?inducer\s+of\s+(?:the\s+)?{target_pattern}\b",
                rf"{compound_pattern}\s*,?\s*(?:a|an)\s+(?:potent\s+|strong\s+|weak\s+)?inducer\s+of\s+(?:the\s+)?{target_pattern}\b",
                rf"{compound_pattern}\s+(?:potently\s+|strongly\s+|weakly\s+)?induces?\s+(?:the\s+)?{target_pattern}\b",
                rf"induction\s+of\s+(?:the\s+)?{target_pattern}(?:\s+enzyme)?\s+by\s+{compound_pattern}\b",
                rf"(?:ability|capacity)\s+of\s+{compound_pattern}\s+to\s+induce\s+(?:the\s+)?{target_pattern}\b",
            ),
        ),
    )
    for role, patterns in role_patterns:
        if any(re.search(pattern, source_text, flags=flags) for pattern in patterns):
            return role
    return None


def ddi_relation_rejection_reason(
    source_text: str,
    compound: str | None,
    target: str | None,
    role: str | None,
) -> str | None:
    if not compound or not target or not role:
        return None
    compound_pattern = rf"(?<![A-Za-z0-9-]){re.escape(compound)}(?![A-Za-z0-9-])"
    target_pattern = ddi_target_pattern(target)
    relation_terms = {
        "substrate": r"(?:substrates?|metaboli[sz]ed|transported|cleared)",
        "inhibitor": r"(?:inhibitors?|inhibits?|inhibition)",
        "inducer": r"(?:inducers?|induces?|induction)",
    }[role]
    flags = re.IGNORECASE
    negation = r"(?:not(?!\s+only)|neither|no\s+(?:evidence|indication))"
    relation_order = rf"(?:{target_pattern}.{{0,30}}{relation_terms}|{relation_terms}.{{0,30}}{target_pattern})"
    negated_patterns = (rf"{compound_pattern}.{{0,70}}\b{negation}\b.{{0,45}}{relation_order}",)
    if any(re.search(pattern, source_text, flags=flags) for pattern in negated_patterns):
        return "negated_ddi_relation"
    if re.search(rf"\bwhether(?:\s+or\s+not)?\b.{{0,60}}{compound_pattern}.{{0,70}}{relation_order}", source_text, flags=flags):
        return "question_or_hypothesis_not_asserted_relation"
    uncertainty = r"(?:may|might|could|possibly|potentially)"
    if re.search(rf"{compound_pattern}.{{0,45}}\b{uncertainty}\b.{{0,45}}{relation_order}", source_text, flags=flags):
        return "uncertain_ddi_relation"
    if re.search(
        rf"{compound_pattern}\s+(?:and/or|or)\s+(?:its\s+)?metabolites?",
        source_text,
        flags=flags,
    ):
        return "parent_metabolite_relation_subject_ambiguous"
    return None


def ddi_target_pattern(target: str) -> str:
    aliases = {
        "P-glycoprotein": r"(?:P-gp|P-glycoprotein|ABCB1|MDR1)",
        "BCRP": r"(?:BCRP|ABCG2)",
        "OATP1B1": r"(?:OATP1B1|SLCO1B1)",
        "OATP1B3": r"(?:OATP1B3|SLCO1B3)",
        "OCT1": r"(?:OCT1|SLC22A1)",
        "OCT2": r"(?:OCT2|SLC22A2)",
        "MATE1": r"(?:MATE1|SLC47A1)",
        "MATE2": r"(?:MATE2(?:-K)?|SLC47A2)",
    }
    return aliases.get(target, re.escape(target))


def ddi_pointer_reason(compound: str | None, target: str | None, ambiguous_target: bool) -> str:
    if compound is None:
        return "target_compound_not_uniquely_resolved_in_source_text"
    if target is None:
        return "ddi_target_not_resolved"
    if ambiguous_target:
        return "multiple_ddi_targets_require_relation_resolution"
    return "compound_target_relation_direction_not_explicit"


def repository_compound_name(repository_path: str, targets: set[str]) -> str | None:
    parts = PurePosixPath(repository_path).parts
    if not parts:
        return None
    return exact_target_name(parts[0], targets)


def repository_path_for(asset: dict[str, Any]) -> str | None:
    metadata = asset.get("asset_metadata_json") if isinstance(asset.get("asset_metadata_json"), dict) else {}
    nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    return clean_optional(nested.get("repository_path"))


def json_field_location(row: dict[str, Any], path: str) -> str:
    location = str(row.get("source_location") or "")
    prefix = location.split("#json_path=", 1)[0]
    return f"{prefix}#json_path={path}"


def nested_label(value: Any) -> str | None:
    if not isinstance(value, dict):
        return clean_optional(value)
    return clean_optional(value.get("label") or value.get("name") or value.get("sid"))
