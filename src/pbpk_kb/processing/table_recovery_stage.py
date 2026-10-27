from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, TextIO
from uuid import NAMESPACE_URL, uuid5
from zipfile import ZipFile

from defusedxml import ElementTree as SafeET

from pbpk_kb.normalization.unit_normalizer import normalize_unit, to_canonical_value
from pbpk_kb.parsers.dailymed_spl import DailyMedSPLParser
from pbpk_kb.parsers.emitter import clean_text
from pbpk_kb.parsers.table_structure import (
    StructuredTable,
    descendants,
    element_text,
    local_name,
    parse_markdown_tables,
    parse_xml_table_element,
    parse_xml_table_fragment,
)
from pbpk_kb.processing.deterministic_extract_stage import valid_pk_value_and_unit
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
from pbpk_kb.processing.source_context import (
    METABOLITE_HEADER, condition_fields, repair_multilevel_headers,
)


TABLE_RECOVERY_VERSION = "table-recovery-v3-analyte-qualifiers"
DAILYMED_ARCHIVE_READER_VERSION = "dailymed-verified-archive-v1"
ARCHIVE_PROVENANCE_FIELDS = (
    "archive_inner_path", "archive_member_checksum", "archive_member_bytes", "archive_source_reader_version",
)
DEFERRED_STATUS = "deferred_unparsed_flattened_table"
OUTPUT_JSONL = (
    "recovered_tables.jsonl",
    "recovered_table_rows.jsonl",
    "table_recovery_status.jsonl",
    "table_evidence_candidates.jsonl",
    "table_evidence_duplicates.jsonl",
    "table_review_queue.jsonl",
)

OPENFDA_RELATED_TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "clinical_pharmacology": ("clinical_pharmacology_table", "pharmacokinetics_table"),
    "pharmacokinetics": ("pharmacokinetics_table", "clinical_pharmacology_table"),
    "drug_interactions": ("drug_interactions_table",),
    "use_in_specific_populations": ("use_in_specific_populations_table",),
    "pregnancy": ("pregnancy_table", "use_in_specific_populations_table"),
    "pediatric_use": ("pediatric_use_table", "use_in_specific_populations_table"),
    "geriatric_use": ("geriatric_use_table", "use_in_specific_populations_table"),
    "renal_impairment": ("renal_impairment_table", "use_in_specific_populations_table"),
    "hepatic_impairment": ("hepatic_impairment_table", "use_in_specific_populations_table"),
}

OPENFDA_FIELD_TOPIC_TERMS: dict[str, tuple[str, ...]] = {
    "clinical_pharmacology": ("clinical pharmacology", "pharmacokinetic"),
    "pharmacokinetics": ("pharmacokinetic",),
    "drug_interactions": ("drug interaction", "interaction"),
    "use_in_specific_populations": ("specific population", "population"),
    "pregnancy": ("pregnancy",),
    "pediatric_use": ("pediatric", "paediatric"),
    "geriatric_use": ("geriatric", "elderly"),
    "renal_impairment": ("renal",),
    "hepatic_impairment": ("hepatic",),
}

PARAMETER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Cmax", re.compile(r"\bC\s*max\b", re.I)),
    ("AUC", re.compile(r"\bAUC(?:\s*(?:t|inf|∞|\d+(?:\s*[-–]\s*(?:t|inf|∞|\d+))?\s*h?))?\b", re.I)),
    ("Tmax", re.compile(r"\bT\s*max\b", re.I)),
    ("half_life", re.compile(r"\b(?:half[- ]life|t\s*1/2|t½)\b", re.I)),
    ("clearance", re.compile(r"\b(?:systemic|total|oral|apparent)?\s*clearance\b|\bCL(?:/F)?\b", re.I)),
    ("volume_of_distribution", re.compile(r"\b(?:volume of distribution|Vd|Vz(?:/F)?|Vss)\b", re.I)),
    ("bioavailability", re.compile(r"\b(?:absolute |oral )?bioavailability\b", re.I)),
    ("protein_binding", re.compile(r"\b(?:plasma |serum )?protein binding\b|\bfraction unbound\b|"
                                   r"\bbound\s+to\s+(?:(?:human|plasma|serum)\s+)*proteins?\b", re.I)),
    ("fraction_excreted_unchanged", re.compile(r"\bfraction excreted unchanged\b|\bexcreted unchanged\b", re.I)),
)

EVIDENCE_SIGNAL_RE = re.compile(
    r"\b(?:C\s*max|AUC|T\s*max|half[- ]life|clearance|volume of distribution|bioavailability|"
    r"protein binding|bound to (?:human )?(?:plasma |serum )?protein|fraction unbound|excreted unchanged|drug interactions?|CYP[0-9A-Z]+|"
    r"renal impairment|hepatic impairment|pediatric|geriatric|pregnancy|PBPK|PK-Sim|Simcyp)\b",
    re.I,
)
SPECIAL_POPULATION_RE = re.compile(
    r"\b(?:renal impairment|hepatic impairment|pediatric|paediatric|geriatric|elderly|pregnan(?:cy|t)|lactation)\b",
    re.I,
)
DDI_RE = re.compile(r"\b(?:drug interactions?|coadministered|inhibitors?|inducers?|CYP[0-9A-Z]+|OATP[0-9A-Z]*|P-gp|BCRP)\b", re.I)
RENAL_FUNCTION_RE = re.compile(
    r"\b(?:creatinine\s+clearance|CrCl|CL\s*[_ ]?\s*cr|eGFR|estimated glomerular filtration rate)\b",
    re.I,
)
RANGE_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?\s*(?:-|–|to)\s*[-+]?\d+(?:\.\d+)?", re.I)
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(?:[<>≤≥~≈]\s*)?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)(?![A-Za-z0-9])")
UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?:ng|pg|µg|μg|ug|mg|g|nmol|µmol|μmol|umol|mmol)\s*(?:[·•*. -]?\s*(?:h|hr|hour))?\s*/\s*(?:mL|ml|L|l)"
    r"|(?<![A-Za-z])(?:mL|ml|L|l)\s*/\s*(?:(?:min|h|hr|hour)(?:\s*/\s*kg)?|kg(?:\s*/\s*(?:min|h|hr|hour))?)"
    r"|(?<![A-Za-z])(?:h|hr|hrs|hour|hours|min|minute|minutes|day|days)\b"
    r"|(?<![A-Za-z])(?:mL|ml|L|l)\s*/\s*kg\b|%",
    re.I,
)


@dataclass(frozen=True)
class OpenFDAContext:
    result_index: int
    field: str
    set_id: str | None
    canonical_dailymed_asset_id: str | None


class AtomicJsonlBundle:
    def __init__(self, layout: Any) -> None:
        self.layout = layout
        self.stack = ExitStack()
        self.handles: dict[str, TextIO] = {}
        self.temporary_paths: dict[str, Path] = {}

    def __enter__(self) -> "AtomicJsonlBundle":
        for name in OUTPUT_JSONL:
            final = self.layout.path(name, for_write=True)
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = final.with_name(f".{final.name}.tmp")
            temporary.unlink(missing_ok=True)
            self.temporary_paths[name] = temporary
            self.handles[name] = self.stack.enter_context(temporary.open("w", encoding="utf-8"))
        return self

    def write(self, name: str, row: dict[str, Any]) -> None:
        self.handles[name].write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        self.handles[name].write("\n")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stack.close()
        if exc_type is not None:
            for path in self.temporary_paths.values():
                path.unlink(missing_ok=True)
            return
        for name, temporary in self.temporary_paths.items():
            temporary.replace(self.layout.path(name, for_write=True))


def read_dailymed_xml_source(asset: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """Read an XML source or its previously registered exact official ZIP member.

    No archive is extracted to disk. Archive identity, every bounded member's
    CRC, the SPL document identity and the registered member digest are checked
    by the same validator used at acquisition. Unknown archives fail closed.
    """
    path = Path(str(asset["storage_path"]))
    if asset.get("file_family") != "archive":
        return path.read_bytes(), {}
    from pbpk_kb.acquisition.dailymed_recovery import MAX_ARCHIVE_BYTES, _archive_url, _validated_archive

    nested = ((asset.get("asset_metadata_json") or {}).get("metadata") or {})
    version = nested.get("spl_version")
    if (asset.get("source_id") != "dailymed"
            or nested.get("recovery_method") != "verified_official_exact_version_archive"
            or type(version) is not int or version < 1):
        raise ValueError("DailyMed archive is not a registered exact-version replacement")
    _archive_url(str(asset.get("source_url") or ""), str(nested.get("setid") or ""), version)
    if not 0 < path.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise ValueError("DailyMed archive size exceeds the registered reader limit")
    contents = path.read_bytes()
    if len(contents) != asset.get("file_size") or sha256(contents).hexdigest() != asset.get("checksum"):
        raise ValueError("DailyMed archive byte count or checksum changed")
    verified = _validated_archive(path, str(nested.get("setid") or ""), version)
    if any(nested.get(key) != value for key, value in verified.items()):
        raise ValueError("DailyMed archive member differs from its registered identity")
    with ZipFile(path) as archive:
        xml = archive.read(verified["member_path"])
    if (len(xml) != verified["member_bytes"] or sha256(xml).hexdigest() != verified["member_sha256"]
            or sha256(path.read_bytes()).hexdigest() != asset.get("checksum")):
        raise ValueError("DailyMed archive changed while reading its member")
    # Recheck with the hardened parser before returning bytes to any consumer.
    SafeET.fromstring(xml, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    return xml, {
        "archive_inner_path": verified["member_path"],
        "archive_member_checksum": verified["member_sha256"],
        "archive_member_bytes": verified["member_bytes"],
        "archive_source_reader_version": DAILYMED_ARCHIVE_READER_VERSION,
    }


def validate_dailymed_archive_reference(
    asset: dict[str, Any], row: dict[str, Any], proof: dict[str, Any], *, require_metadata: bool = False,
) -> None:
    if not proof:
        return
    prefix = f"dailymed:{Path(str(asset['storage_path'])).name}!{proof['archive_inner_path']}#"
    if (not str(row.get("source_location") or "").startswith(prefix)
            or any(row.get(key) != asset.get(key) for key in
                   ("raw_asset_id", "source_id", "storage_path", "source_url", "checksum"))):
        raise ValueError("DailyMed archive reference does not identify the registered source/member")
    if require_metadata:
        metadata = row.get("metadata_json") or {}
        if any(metadata.get(key) != proof[key] for key in ("archive_inner_path", "archive_member_checksum")):
            raise ValueError("DailyMed parsed table member provenance differs from registered source")


class TableRecoveryStage:
    def __init__(self, *, output_dir: Path | str) -> None:
        self.layout = raw_processing_layout(output_dir)
        self.output_dir = self.layout.root.resolve()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.pipeline_state: dict[str, Any] = {}
        self.run_id = ""
        self.deferred: list[dict[str, Any]] = []
        self.deferred_by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.inventory: dict[str, dict[str, Any]] = {}
        self.target_compounds: set[str] = set()
        self.dailymed_by_set_id: dict[str, str] = {}
        self.openfda_cache: dict[str, dict[str, Any]] = {}
        self.openfda_contexts: dict[str, OpenFDAContext] = {}
        self.openfda_by_dailymed_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.coverage: dict[str, tuple[str, list[str], str]] = {}
        self.ocr_required_chunks: set[str] = set()
        self.recovery_failures: dict[str, str] = {}
        self.asset_ownership_cache: dict[str, tuple[str | None, str]] = {}
        self.seen_table_keys: dict[tuple[str, str], str] = {}
        self.seen_row_ids: set[str] = set()
        self.seen_evidence_ids: set[str] = set()
        self.seen_review_ids: set[str] = set()
        self.counts: Counter[str] = Counter()
        self.status_counts: Counter[str] = Counter()
        self.status_source_counts: Counter[str] = Counter()
        self.evidence_type_counts: Counter[str] = Counter()
        self.review_reason_counts: Counter[str] = Counter()
        self.writer: AtomicJsonlBundle | None = None

    def run(self) -> dict[str, Any]:
        self.layout.ensure_dirs()
        self._validate_prerequisites()
        self._load_deferred()
        self._load_inventory()
        self._prepare_openfda_contexts()
        self._write_progress("loaded_inputs")

        with AtomicJsonlBundle(self.layout) as writer:
            self.writer = writer
            self._reuse_existing_tables()
            self._write_progress("reused_existing_tables")
            self._recover_dailymed_missing_scopes()
            self._recover_openfda_tables()
            self._recover_pdf_tables()
            self._recover_markdown_tables()
            self._write_terminal_statuses()
            self.writer = None

        self._deduplicate_table_evidence()
        self._merge_evidence()
        summary = self._summary()
        self._write_json_atomic(self.layout.path("table_recovery_summary.json", for_write=True), summary)
        self._update_pipeline_state(summary)
        self._write_progress("completed", completed=True, summary=summary)
        return summary

    def _validate_prerequisites(self) -> None:
        state_path = self.layout.path("pipeline_state.json")
        if not state_path.is_file():
            raise FileNotFoundError(f"pipeline_state.json not found: {state_path}")
        self.pipeline_state = json.loads(state_path.read_text(encoding="utf-8"))
        self.run_id = str(self.pipeline_state.get("run_id") or "")
        llm = (self.pipeline_state.get("stages") or {}).get("llm_extract") or {}
        if llm.get("status") != "completed" or llm.get("accepted") is not True:
            raise ValueError("table_recovery requires llm_extract=completed+accepted")
        for downstream in ("completeness", "embedding"):
            stage = (self.pipeline_state.get("stages") or {}).get(downstream) or {}
            if stage.get("status") not in {None, "not_started"}:
                raise ValueError(f"Refusing table recovery after downstream stage started: {downstream}")
        for name in ("chunk_processing_status.jsonl", "parsed_tables.jsonl", "raw_inventory.jsonl"):
            if not self.layout.path(name).is_file():
                raise FileNotFoundError(f"Required table recovery input missing: {self.layout.path(name)}")

    def _load_deferred(self) -> None:
        seen: set[str] = set()
        for row in iter_jsonl(self.layout.path("chunk_processing_status.jsonl")):
            if row.get("processing_status") != DEFERRED_STATUS:
                continue
            chunk_id = str(row.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen:
                raise ValueError(f"Missing or duplicate deferred chunk_id: {chunk_id}")
            seen.add(chunk_id)
            self.deferred.append(row)
            self.deferred_by_asset[str(row.get("raw_asset_id") or "")].append(row)
        self.counts["deferred_chunks"] = len(self.deferred)
        self.counts["deferred_assets"] = len(self.deferred_by_asset)

    def _load_inventory(self) -> None:
        for row in iter_jsonl(self.layout.path("raw_inventory.jsonl")):
            raw_asset_id = str(row.get("raw_asset_id") or "")
            if raw_asset_id:
                self.inventory[raw_asset_id] = row
            compound = clean_optional(row.get("compound"))
            if compound:
                self.target_compounds.add(compound)
            if row.get("source_id") == "dailymed":
                metadata = row.get("asset_metadata_json") if isinstance(row.get("asset_metadata_json"), dict) else {}
                nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
                set_id = clean_optional(nested.get("setid") or nested.get("set_id"))
                if set_id:
                    self.dailymed_by_set_id[set_id.lower()] = raw_asset_id
        missing = sorted(raw_asset_id for raw_asset_id in self.deferred_by_asset if raw_asset_id not in self.inventory)
        if missing:
            raise ValueError(f"Deferred chunks reference missing raw inventory assets: {missing[:5]}")

    def _prepare_openfda_contexts(self) -> None:
        for row in self.deferred:
            if row.get("source_id") != "openfda_drug_label":
                continue
            context = self._openfda_context(row)
            if context is None:
                continue
            chunk_id = str(row["chunk_id"])
            self.openfda_contexts[chunk_id] = context
            if context.canonical_dailymed_asset_id:
                self.openfda_by_dailymed_asset[context.canonical_dailymed_asset_id].append(row)

    def _openfda_context(self, row: dict[str, Any]) -> OpenFDAContext | None:
        match = re.search(r"\$\.results\[(\d+)\]\.([^#]+)", str(row.get("source_location") or ""))
        if not match:
            return None
        result_index = int(match.group(1))
        field = match.group(2)
        payload = self._load_openfda_payload(str(row.get("storage_path") or ""))
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        if result_index >= len(results) or not isinstance(results[result_index], dict):
            return None
        set_id = clean_optional(results[result_index].get("set_id"))
        canonical = self.dailymed_by_set_id.get(set_id.lower()) if set_id else None
        return OpenFDAContext(result_index, field, set_id, canonical)

    def _load_openfda_payload(self, storage_path: str) -> dict[str, Any]:
        if storage_path not in self.openfda_cache:
            payload = json.loads(Path(storage_path).read_text(encoding="utf-8", errors="replace"))
            self.openfda_cache[storage_path] = payload if isinstance(payload, dict) else {}
        return self.openfda_cache[storage_path]

    def _reuse_existing_tables(self) -> None:
        direct_assets = set(self.deferred_by_asset)
        mapped_dailymed_assets = set(self.openfda_by_dailymed_asset)
        targets = direct_assets | mapped_dailymed_assets
        for row in iter_jsonl(self.layout.path("parsed_tables.jsonl")):
            raw_asset_id = str(row.get("raw_asset_id") or "")
            if raw_asset_id not in targets:
                continue
            matched_direct = [
                chunk
                for chunk in self.deferred_by_asset.get(raw_asset_id, [])
                if self._existing_table_matches_chunk(row, chunk)
            ]
            matched_linked = [
                chunk
                for chunk in self.openfda_by_dailymed_asset.get(raw_asset_id, [])
                if self._dailymed_table_matches_openfda(row, chunk)
            ]
            if not matched_direct and not matched_linked:
                continue
            asset = self.inventory[raw_asset_id]
            if asset.get("source_id") == "dailymed" and asset.get("file_family") == "archive":
                _, proof = read_dailymed_xml_source(asset)
                for chunk in matched_direct:
                    validate_dailymed_archive_reference(asset, chunk, proof)
            table_ids = self._emit_existing_table(row)
            if not table_ids:
                continue
            for chunk in matched_direct:
                self._cover(chunk, "resolved_by_existing_table", table_ids, "matched existing parsed table by source scope")
            for chunk in matched_linked:
                self._cover(chunk, "linked_to_canonical_source", table_ids, "OpenFDA set_id linked to canonical DailyMed table")

    def _existing_table_matches_chunk(self, table: dict[str, Any], chunk: dict[str, Any]) -> bool:
        source = str(chunk.get("source_id") or "")
        chunk_location = str(chunk.get("source_location") or "")
        table_location = str(table.get("source_location") or "")
        if source == "dailymed":
            match = re.search(r"xml_path=//section\[(\d+)\]", chunk_location)
            return bool(match and re.search(rf"xml_path=//section\[{match.group(1)}\].*#table=", table_location))
        page = re.search(r"page=(\d+)", chunk_location)
        if page:
            return bool(re.search(rf"page={page.group(1)}(?:#|$)", table_location))
        if "inner_path=" in chunk_location:
            prefix = re.split(r"#(?:xml_path|page|text|sheet)=", chunk_location, maxsplit=1)[0]
            return table_location.startswith(prefix)
        return False

    def _dailymed_table_matches_openfda(self, table: dict[str, Any], chunk: dict[str, Any]) -> bool:
        context = self.openfda_contexts.get(str(chunk.get("chunk_id") or ""))
        if context is None:
            return False
        metadata = table.get("metadata_json") if isinstance(table.get("metadata_json"), dict) else {}
        descriptor = clean_text(
            " ".join(
                str(value or "")
                for value in (metadata.get("section_title"), metadata.get("caption"), table.get("table_text"))
            )
        ).lower()
        return any(term in descriptor for term in OPENFDA_FIELD_TOPIC_TERMS.get(context.field, (context.field.replace("_", " "),)))

    def _emit_existing_table(self, row: dict[str, Any]) -> list[str]:
        asset = self.inventory[str(row["raw_asset_id"])]
        if asset.get("source_id") == "dailymed" and asset.get("file_family") == "archive":
            _, proof = read_dailymed_xml_source(asset)
            validate_dailymed_archive_reference(asset, row, proof, require_metadata=True)
            row = {**row, "metadata_json": {**(row.get("metadata_json") or {}), **proof}}
        columns = [clean_text(str(value or "")) for value in (row.get("columns_json") or [])]
        cells = row.get("cells_json") if isinstance(row.get("cells_json"), list) else []
        rows = [[clean_text(str(value or "")) for value in item] for item in cells if isinstance(item, list)]
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        structure_reliable = table_structure_reliable(columns, rows)
        structured = StructuredTable(
            columns=columns,
            rows=rows,
            caption=clean_optional(metadata.get("caption")),
            footnotes=[str(value) for value in metadata.get("footnotes", [])] if isinstance(metadata.get("footnotes"), list) else [],
            metadata={
                "existing_parsed_table_id": row.get("parsed_table_id"),
                "structure_reliable": structure_reliable,
            },
        )
        recovered_id = self._emit_table(
            asset=self.inventory[str(row["raw_asset_id"])],
            structured=structured,
            source_location=str(row.get("source_location") or ""),
            origin="existing_parsed_table",
            parsed_table_id=clean_optional(row.get("parsed_table_id")),
            metadata=metadata,
        )
        return [recovered_id] if recovered_id and structure_reliable else []

    def _recover_dailymed_missing_scopes(self) -> None:
        for raw_asset_id, chunks in self.deferred_by_asset.items():
            if not chunks or chunks[0].get("source_id") != "dailymed":
                continue
            missing = [chunk for chunk in chunks if str(chunk["chunk_id"]) not in self.coverage]
            if not missing:
                continue
            section_indices = {
                int(match.group(1))
                for chunk in missing
                if (match := re.search(r"xml_path=//section\[(\d+)\]", str(chunk.get("source_location") or "")))
            }
            if not section_indices:
                continue
            asset = self.inventory[raw_asset_id]
            try:
                xml, archive_proof = read_dailymed_xml_source(asset)
                for chunk in missing:
                    validate_dailymed_archive_reference(asset, chunk, archive_proof)
                root = SafeET.fromstring(xml)
                source_name = Path(str(asset["storage_path"])).name
                if archive_proof:
                    source_name += "!" + archive_proof["archive_inner_path"]
                emitted_by_section: dict[int, list[str]] = defaultdict(list)
                for section_index, section in enumerate(descendants(root, "section"), start=1):
                    if section_index not in section_indices:
                        continue
                    title = element_text(next((child for child in list(section) if local_name(child.tag).lower() == "title"), None))
                    text_node = next((child for child in list(section) if local_name(child.tag).lower() == "text"), None)
                    if text_node is None:
                        continue
                    for table_index, table_node in enumerate(descendants(text_node, "table"), start=1):
                        structured = parse_xml_table_element(
                            table_node,
                            footnotes=[element_text(node) for node in descendants(table_node, "footnote")],
                        )
                        if structured is None:
                            continue
                        location = f"{asset['source_id']}:{source_name}#xml_path=//section[{section_index}]#table={table_index}"
                        table_id = self._emit_table(
                            asset=asset,
                            structured=structured,
                            source_location=location,
                            origin="recovered_dailymed_spl_xml",
                            metadata={"section_index": section_index, "section_title": title, **archive_proof},
                        )
                        if table_id:
                            emitted_by_section[section_index].append(table_id)
                for chunk in missing:
                    match = re.search(r"xml_path=//section\[(\d+)\]", str(chunk.get("source_location") or ""))
                    if match and emitted_by_section.get(int(match.group(1))):
                        self._cover(chunk, "recovered_from_raw", emitted_by_section[int(match.group(1))], "recovered table from DailyMed SPL XML section")
            except Exception as exc:
                for chunk in missing:
                    self.recovery_failures[str(chunk["chunk_id"])] = f"DailyMed XML table recovery failed: {exc}"

    def _recover_openfda_tables(self) -> None:
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for chunk in self.deferred:
            if chunk.get("source_id") != "openfda_drug_label" or str(chunk["chunk_id"]) in self.coverage:
                continue
            context = self.openfda_contexts.get(str(chunk["chunk_id"]))
            if context:
                grouped[(str(chunk["raw_asset_id"]), context.result_index, context.field)].append(chunk)
        for (raw_asset_id, result_index, field), chunks in grouped.items():
            asset = self.inventory[raw_asset_id]
            try:
                payload = self._load_openfda_payload(str(asset["storage_path"]))
                item = payload["results"][result_index]
                table_ids: list[str] = []
                for table_field in OPENFDA_RELATED_TABLE_FIELDS.get(field, (f"{field}_table",)):
                    values = item.get(table_field)
                    if isinstance(values, str):
                        values = [values]
                    if not isinstance(values, list):
                        continue
                    for value_index, fragment in enumerate(values):
                        if not isinstance(fragment, str):
                            continue
                        for table_index, structured in enumerate(parse_xml_table_fragment(fragment), start=1):
                            location = (
                                f"{asset['source_id']}:{Path(str(asset['storage_path'])).name}"
                                f"#json_path=$.results[{result_index}].{table_field}[{value_index}]#table={table_index}"
                            )
                            table_id = self._emit_table(
                                asset=asset,
                                structured=structured,
                                source_location=location,
                                origin="recovered_openfda_spl_table_fragment",
                                metadata={
                                    "result_index": result_index,
                                    "json_field": table_field,
                                    "set_id": item.get("set_id"),
                                    "openfda": item.get("openfda") if isinstance(item.get("openfda"), dict) else {},
                                },
                            )
                            if table_id:
                                table_ids.append(table_id)
                if table_ids:
                    for chunk in chunks:
                        self._cover(chunk, "recovered_from_raw", table_ids, "recovered complete XML table from OpenFDA SPL table field")
            except Exception as exc:
                for chunk in chunks:
                    self.recovery_failures[str(chunk["chunk_id"])] = f"OpenFDA table recovery failed: {exc}"

    def _recover_pdf_tables(self) -> None:
        grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for chunk in self.deferred:
            chunk_id = str(chunk["chunk_id"])
            if chunk_id in self.coverage:
                continue
            asset = self.inventory[str(chunk["raw_asset_id"])]
            if asset.get("file_family") != "pdf":
                continue
            page = re.search(r"page=(\d+)", str(chunk.get("source_location") or ""))
            if page:
                grouped[str(chunk["raw_asset_id"])][int(page.group(1))].append(chunk)
        if not grouped:
            return
        import fitz  # type: ignore

        for raw_asset_id, pages in grouped.items():
            asset = self.inventory[raw_asset_id]
            try:
                document = fitz.open(str(asset["storage_path"]))
                for page_number, chunks in pages.items():
                    if page_number < 1 or page_number > len(document):
                        continue
                    page = document[page_number - 1]
                    table_ids: list[str] = []
                    found_tables: list[Any] = []
                    for options in ({}, {"strategy": "text"}):
                        try:
                            finder = page.find_tables(**options)
                            found_tables = list(finder.tables)
                            if found_tables:
                                break
                        except Exception:
                            continue
                    for table_index, table in enumerate(found_tables, start=1):
                        rows = table.extract() or []
                        if not rows:
                            continue
                        structured = StructuredTable(
                            columns=[clean_text(str(value or "")) for value in rows[0]],
                            rows=[[clean_text(str(value or "")) for value in row] for row in rows[1:]],
                            metadata={"bounding_box": list(table.bbox)},
                        )
                        location = f"{asset['source_id']}:{Path(str(asset['storage_path'])).name}#page={page_number}#table={table_index}"
                        table_id = self._emit_table(
                            asset=asset,
                            structured=structured,
                            source_location=location,
                            origin="recovered_pdf_table",
                            metadata={"page_number": page_number, "bounding_box": list(table.bbox)},
                        )
                        if table_id:
                            table_ids.append(table_id)
                    if table_ids:
                        for chunk in chunks:
                            self._cover(chunk, "recovered_from_raw", table_ids, "recovered table from PDF page geometry")
                    elif not clean_text(page.get_text("text")):
                        self.ocr_required_chunks.update(str(chunk["chunk_id"]) for chunk in chunks)
                document.close()
            except Exception as exc:
                for chunks in pages.values():
                    for chunk in chunks:
                        self.recovery_failures[str(chunk["chunk_id"])] = f"PDF table recovery failed: {exc}"

    def _recover_markdown_tables(self) -> None:
        for raw_asset_id, chunks in self.deferred_by_asset.items():
            missing = [chunk for chunk in chunks if str(chunk["chunk_id"]) not in self.coverage]
            if not missing:
                continue
            asset = self.inventory[raw_asset_id]
            if str(asset.get("file_extension") or "").lower() not in {".md", ".markdown"}:
                continue
            try:
                document = Path(str(asset["storage_path"])).read_text(encoding="utf-8", errors="replace")
                table_ids: list[str] = []
                for table_index, (start_line, end_line, structured) in enumerate(parse_markdown_tables(document), start=1):
                    location = f"{asset['source_id']}:{Path(str(asset['storage_path'])).name}#lines={start_line}-{end_line}#table={table_index}"
                    table_id = self._emit_table(
                        asset=asset,
                        structured=structured,
                        source_location=location,
                        origin="recovered_markdown_table",
                        metadata={"start_line": start_line, "end_line": end_line},
                    )
                    if table_id:
                        table_ids.append(table_id)
                if table_ids:
                    for chunk in missing:
                        self._cover(chunk, "recovered_from_raw", table_ids, "recovered Markdown table")
            except Exception as exc:
                for chunk in missing:
                    self.recovery_failures[str(chunk["chunk_id"])] = f"Markdown table recovery failed: {exc}"

    def _emit_table(
        self,
        *,
        asset: dict[str, Any],
        structured: StructuredTable,
        source_location: str,
        origin: str,
        parsed_table_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        if self.writer is None or not structured.rows:
            return None
        structured = repair_multilevel_headers(structured)
        table_text = "\n".join(
            [
                clean_text(structured.caption or ""),
                " | ".join(clean_text(str(column or "")) for column in structured.columns),
                *(" | ".join(clean_text(str(cell or "")) for cell in row) for row in structured.rows),
            ]
        )
        table_hash = hash_text(table_text)
        canonical_key = self._canonical_table_key(asset, metadata or {})
        dedup_key = (canonical_key, table_hash)
        if dedup_key in self.seen_table_keys:
            self.counts["duplicate_tables_suppressed"] += 1
            return self.seen_table_keys[dedup_key]
        recovered_table_id = stable_id("recovered_table", canonical_key, source_location, table_hash)
        self.seen_table_keys[dedup_key] = recovered_table_id
        combined_metadata = {**(metadata or {}), **structured.metadata}
        combined_metadata.setdefault("structure_reliable", table_structure_reliable(structured.columns, structured.rows))
        table_row = {
            "run_id": self.run_id,
            "recovered_table_id": recovered_table_id,
            "parsed_table_id": parsed_table_id,
            "raw_asset_id": asset.get("raw_asset_id"),
            "source_id": asset.get("source_id"),
            "compound": asset.get("compound"),
            "storage_path": asset.get("storage_path"),
            "source_url": asset.get("source_url"),
            "checksum": asset.get("checksum"),
            "source_location": source_location,
            "table_origin": origin,
            "caption": structured.caption,
            "footnotes": structured.footnotes,
            "columns_json": structured.columns,
            "row_count": len(structured.rows),
            "column_count": len(structured.columns),
            "table_hash": table_hash,
            "metadata_json": combined_metadata,
            "extractor_version": TABLE_RECOVERY_VERSION,
        }
        self.writer.write("recovered_tables.jsonl", table_row)
        self.counts["recovered_tables"] += 1
        self.counts[f"table_origin:{origin}"] += 1
        ownership = self._resolve_asset_ownership(asset, combined_metadata)
        for row_offset, cells in enumerate(structured.rows, start=1):
            self._emit_table_row(table_row, row_offset, cells, ownership)
        return recovered_table_id

    def _emit_table_row(
        self,
        table: dict[str, Any],
        row_index: int,
        cells: list[str],
        ownership: tuple[str | None, str],
    ) -> None:
        if self.writer is None:
            return
        normalized_cells = [clean_text(str(cell or "")) for cell in cells]
        row_text = " | ".join(normalized_cells)
        if not row_text.replace("|", "").strip():
            return
        row_hash = hash_text(row_text)
        table_row_id = stable_id("table_row", str(table["recovered_table_id"]), str(row_index), row_hash)
        if table_row_id in self.seen_row_ids:
            self.counts["duplicate_rows_suppressed"] += 1
            return
        self.seen_row_ids.add(table_row_id)
        row_location = f"{table['source_location']}#row={row_index}"
        row = {
            "run_id": self.run_id,
            "table_row_id": table_row_id,
            "recovered_table_id": table.get("recovered_table_id"),
            "parsed_table_id": table.get("parsed_table_id"),
            "raw_asset_id": table.get("raw_asset_id"),
            "source_id": table.get("source_id"),
            "compound": table.get("compound"),
            "storage_path": table.get("storage_path"),
            "source_url": table.get("source_url"),
            "checksum": table.get("checksum"),
            "source_location": row_location,
            "row_index": row_index,
            "caption": table.get("caption"),
            "footnotes": table.get("footnotes"),
            "columns_json": table.get("columns_json"),
            "cells_json": normalized_cells,
            "cell_locations": [f"{row_location}#col={index + 1}" for index in range(len(normalized_cells))],
            "row_text": row_text,
            "row_hash": row_hash,
            "metadata_json": {"table_origin": table.get("table_origin"), **{
                key: table["metadata_json"][key] for key in ARCHIVE_PROVENANCE_FIELDS
                if key in (table.get("metadata_json") or {})}},
        }
        row["metadata_json"]["structure_reliable"] = bool((table.get("metadata_json") or {}).get("structure_reliable"))
        self.writer.write("recovered_table_rows.jsonl", row)
        self.counts["recovered_table_rows"] += 1
        evidence = self._extract_table_evidence(row, ownership)
        for candidate in evidence:
            evidence_id = str(candidate["evidence_id"])
            if evidence_id in self.seen_evidence_ids:
                self.counts["duplicate_evidence_suppressed"] += 1
                continue
            self.seen_evidence_ids.add(evidence_id)
            self.writer.write("table_evidence_candidates.jsonl", candidate)
            self.counts["table_evidence"] += 1
            self.counts["table_evidence_pre_dedup"] += 1
            self.evidence_type_counts[str(candidate["evidence_type"])] += 1
        if not evidence and EVIDENCE_SIGNAL_RE.search(" ".join([str(table.get("caption") or ""), *table.get("columns_json", []), row_text])):
            self._emit_review_row(row, ownership)

    def _extract_table_evidence(
        self,
        row: dict[str, Any],
        ownership: tuple[str | None, str],
    ) -> list[dict[str, Any]]:
        compound, assignment = self._resolve_row_compound(row, ownership)
        if compound is None:
            return []
        columns = [str(value or "") for value in row.get("columns_json") or []]
        cells = [str(value or "") for value in row.get("cells_json") or []]
        if not bool((row.get("metadata_json") or {}).get("structure_reliable")) or len(columns) != len(cells):
            return []
        if is_pdf_table_row(row):
            return []
        if is_repeated_spanning_row(cells):
            return []
        context_text = " ".join([str(row.get("caption") or ""), *columns, *cells])
        if DDI_RE.search(context_text):
            return []
        candidates: list[dict[str, Any]] = []
        local: set[tuple[str, int, float, str]] = set()
        header_parameters = {index: parameter for index, header in enumerate(columns) if (parameter := detect_parameter(header))}

        if header_parameters:
            for column_index, parameter in header_parameters.items():
                row_analyte = " ".join(cell for index, cell in enumerate(cells) if index not in header_parameters)
                if is_metabolite_analyte_label(columns[column_index]) or is_metabolite_analyte_label(row_analyte):
                    self._emit_analyte_review(row, compound, column_index)
                    continue
                if central_estimate_count(cells[column_index]) != 1:
                    continue
                parsed = parse_table_value(cells[column_index], unit_hint=find_unit(columns[column_index]))
                if parsed is None or not valid_table_parameter_value(parameter, parsed[0], parsed[2]):
                    continue
                key = (parameter, column_index, parsed[0], parsed[2])
                if key in local:
                    continue
                local.add(key)
                condition = build_row_condition(columns, cells, excluded_indices=set(header_parameters))
                candidates.append(self._make_table_evidence(row, compound, assignment, parameter, parsed, condition, columns[column_index], column_index))
            return candidates

        parameter_cells = [(index, parameter) for index, cell in enumerate(cells) if (parameter := detect_parameter(cell))]
        if len(parameter_cells) != 1:
            return []
        parameter_cell_index, parameter = parameter_cells[0]
        unit_hint = find_unit(cells[parameter_cell_index])
        for value_index, value_cell in enumerate(cells):
            if value_index == parameter_cell_index or central_estimate_count(value_cell) != 1:
                continue
            if (is_metabolite_analyte_label(columns[value_index])
                    or is_metabolite_analyte_label(cells[parameter_cell_index])):
                self._emit_analyte_review(row, compound, value_index)
                continue
            parsed = parse_table_value(value_cell, unit_hint=unit_hint or find_unit(columns[value_index]))
            if parsed is None or not valid_table_parameter_value(parameter, parsed[0], parsed[2]):
                continue
            key = (parameter, value_index, parsed[0], parsed[2])
            if key in local:
                continue
            local.add(key)
            # Other values in this row belong to other populations/regimens or
            # analytes. They are evidence, not conditions of the selected cell.
            condition = None
            column_condition = clean_optional(columns[value_index])
            if column_condition and not column_condition.lower().startswith("column_"):
                condition = " ; ".join(part for part in (column_condition, condition) if part)
            candidates.append(self._make_table_evidence(row, compound, assignment, parameter, parsed, condition, cells[parameter_cell_index], value_index))
        return candidates

    def _emit_analyte_review(self, row: dict[str, Any], compound: str, column_index: int) -> None:
        """Retain excluded metabolite cells explicitly without relabeling them as parent."""
        self.counts["metabolite_cells_isolated"] += 1
        if self.writer is None:
            return
        reason = "metabolite_value_not_parent_drug"
        review_id = stable_id("table_review", str(row["table_row_id"]), str(column_index), reason)
        if review_id in self.seen_review_ids:
            return
        self.seen_review_ids.add(review_id)
        self.writer.write("table_review_queue.jsonl", {**row, "review_id": review_id,
                          "target_compound": compound, "value_column_index": column_index,
                          "analyte_role": "metabolite", "review_reason": reason,
                          "curation_status": "needs_review", "llm_eligible": False,
                          "extractor_version": TABLE_RECOVERY_VERSION})
        self.counts["table_review_queue"] += 1
        self.review_reason_counts[reason] += 1

    def _make_table_evidence(
        self,
        row: dict[str, Any],
        compound: str,
        assignment: str,
        parameter: str,
        parsed_value: tuple[float, str, str, str | None],
        condition: str | None,
        parameter_header: str,
        value_column_index: int | None = None,
    ) -> dict[str, Any]:
        value, value_text, unit, qualifier = parsed_value
        canonical_value, canonical_unit = to_canonical_value(value, unit)
        normalized = normalize_unit(unit)
        evidence_type = "adme" if parameter in {"protein_binding", "fraction_excreted_unchanged"} else "clinical_pk"
        context = " ".join([str(row.get("caption") or ""), *row.get("columns_json", []), str(condition or "")])
        if SPECIAL_POPULATION_RE.search(context):
            evidence_type = "special_population"
        fields = condition_fields(str(condition or ""))
        caption_fields = condition_fields(str(row.get("caption") or ""))
        fields = {key: value or caption_fields.get(key) for key, value in fields.items()}
        evidence_id = stable_id(
            "table_evidence",
            str(row["table_row_id"]),
            compound,
            parameter,
            value_text,
            unit,
            str(condition or ""),
        )
        return {
            "run_id": self.run_id,
            "evidence_id": evidence_id,
            "evidence_type": evidence_type,
            "compound": compound,
            "retrieval_compounds": [row.get("compound")] if row.get("compound") else [],
            "compound_assignment_status": assignment,
            "source_id": row.get("source_id"),
            "raw_asset_id": row.get("raw_asset_id"),
            "parsed_record_id": None,
            "parsed_document_id": None,
            "parsed_table_id": row.get("parsed_table_id"),
            "recovered_table_id": row.get("recovered_table_id"),
            "table_row_id": row.get("table_row_id"),
            "chunk_id": None,
            "storage_path": row.get("storage_path"),
            "source_url": row.get("source_url"),
            "checksum": row.get("checksum"),
            "source_location": row.get("source_location"),
            "source_text": row.get("row_text"),
            "source_text_hash": row.get("row_hash"),
            "evidence_tier": "quantitative",
            "extraction_method": "source_structured_table",
            "extractor_name": "table_parameter_value_mapper",
            "extractor_version": TABLE_RECOVERY_VERSION,
            "extraction_confidence": 0.82,
            "curation_status": "needs_review",
            "parameter": parameter,
            "value": value,
            "value_text": value_text,
            "unit": unit,
            "normalized_unit": normalized.normalized_unit,
            "canonical_value": canonical_value if canonical_value is not None and math.isfinite(canonical_value) else None,
            "canonical_unit": canonical_unit,
            "unit_dimension": normalized.dimension,
            "condition": clean_optional(condition),
            "population": fields["population"],
            "route": fields["route"],
            "dose": fields["dose"],
            "metadata_json": {
                "value_qualifier": qualifier,
                "parameter_header": parameter_header,
                "table_caption": row.get("caption"),
                "table_columns": row.get("columns_json"),
                "cell_level_provenance": row.get("cell_locations"),
                "value_column_index": value_column_index,
                "selected_column_header": (row.get("columns_json") or [])[value_column_index] if value_column_index is not None else None,
                "selected_value_location": (row.get("cell_locations") or [])[value_column_index] if value_column_index is not None and row.get("cell_locations") else None,
                "analyte_role": "parent",
                "source_conditions": fields,
                **{key: row["metadata_json"][key] for key in ARCHIVE_PROVENANCE_FIELDS
                   if key in (row.get("metadata_json") or {})},
            },
            "created_at": self.started_at,
        }

    def _emit_review_row(self, row: dict[str, Any], ownership: tuple[str | None, str]) -> None:
        if self.writer is None:
            return
        compound, assignment = self._resolve_row_compound(row, ownership)
        context = " ".join([str(row.get("caption") or ""), *row.get("columns_json", []), str(row.get("row_text") or "")])
        if is_pdf_table_row(row):
            reason = "pdf_table_requires_visual_verification"
        elif is_repeated_spanning_row([str(value or "") for value in row.get("cells_json") or []]):
            reason = "spanning_note_row_not_evidence"
        elif not bool((row.get("metadata_json") or {}).get("structure_reliable")):
            reason = "unreliable_row_column_alignment"
        elif compound is None:
            reason = "compound_ownership_unresolved"
        elif DDI_RE.search(context):
            reason = "ddi_relation_requires_table_semantic_review"
        else:
            reason = "parameter_value_unit_mapping_unresolved"
        review_id = stable_id("table_review", str(row["table_row_id"]), reason)
        if review_id in self.seen_review_ids:
            return
        self.seen_review_ids.add(review_id)
        self.writer.write(
            "table_review_queue.jsonl",
            {
                **row,
                "review_id": review_id,
                "target_compound": compound,
                "compound_assignment_status": assignment,
                "review_reason": reason,
                "curation_status": "needs_review",
                "llm_eligible": bool(compound and row.get("columns_json") and row.get("cells_json")),
                "extractor_version": TABLE_RECOVERY_VERSION,
            },
        )
        self.counts["table_review_queue"] += 1
        self.review_reason_counts[reason] += 1

    def _resolve_asset_ownership(
        self,
        asset: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[str | None, str]:
        raw_asset_id = str(asset.get("raw_asset_id") or "")
        if asset.get("source_id") == "openfda_drug_label":
            openfda = metadata.get("openfda") if isinstance(metadata.get("openfda"), dict) else {}
            substances = openfda.get("substance_name")
            if isinstance(substances, str):
                substances = [substances]
            return resolve_unique_active_ingredient(substances or [], self.target_compounds, "openfda_unique_active_substance")
        if asset.get("source_id") == "dailymed" and asset.get("file_family") == "archive":
            nested = ((asset.get("asset_metadata_json") or {}).get("metadata") or {})
            if (metadata.get("archive_inner_path") != nested.get("member_path")
                    or metadata.get("archive_member_checksum") != nested.get("member_sha256")):
                return None, "dailymed_active_ingredient_member_provenance_mismatch"
        if raw_asset_id in self.asset_ownership_cache:
            return self.asset_ownership_cache[raw_asset_id]
        result: tuple[str | None, str] = (None, "asset_level_ownership_unresolved")
        if asset.get("source_id") == "dailymed":
            try:
                xml, archive_proof = read_dailymed_xml_source(asset)
                if archive_proof and any(metadata.get(key) != archive_proof[key]
                                         for key in ("archive_inner_path", "archive_member_checksum")):
                    raise ValueError("DailyMed active ingredient member provenance mismatch")
                parsed = DailyMedSPLParser().parse(xml)
                names = [item.get("name") for item in parsed.get("active_ingredients", []) if isinstance(item, dict)]
                result = resolve_unique_active_ingredient(names, self.target_compounds, "dailymed_unique_active_ingredient")
            except Exception:
                result = (None, "dailymed_active_ingredient_parse_failed")
        elif asset.get("source_id") == "osp_pbpk_model_library":
            repository_path = str(((asset.get("asset_metadata_json") or {}).get("metadata") or {}).get("repository_path") or "")
            matches = exact_target_matches(repository_path, self.target_compounds)
            result = (matches[0], "osp_exact_repository_compound") if len(matches) == 1 else result
        self.asset_ownership_cache[raw_asset_id] = result
        return result

    def _resolve_row_compound(
        self,
        row: dict[str, Any],
        ownership: tuple[str | None, str],
    ) -> tuple[str | None, str]:
        text = " ".join([str(row.get("caption") or ""), *row.get("columns_json", []), str(row.get("row_text") or "")])
        matches = exact_target_matches(text, self.target_compounds)
        if row.get("source_id") in {"dailymed", "openfda_drug_label"}:
            if ownership[0] is None:
                return ownership
            if len(matches) > 1 or (len(matches) == 1 and matches[0] != ownership[0]):
                return None, "label_table_target_conflicts_with_active_ingredient"
            return ownership
        if len(matches) == 1:
            return matches[0], "exact_target_name_in_table"
        if len(matches) > 1:
            return None, "multiple_target_compounds_in_table"
        return ownership

    def _canonical_table_key(self, asset: dict[str, Any], metadata: dict[str, Any]) -> str:
        set_id = clean_optional(metadata.get("set_id"))
        if not set_id and asset.get("source_id") == "dailymed":
            nested = ((asset.get("asset_metadata_json") or {}).get("metadata") or {})
            set_id = clean_optional(nested.get("setid") or nested.get("set_id"))
        if set_id:
            return f"spl:{set_id.lower()}"
        return f"checksum:{asset.get('checksum')}"

    def _cover(self, chunk: dict[str, Any], status: str, table_ids: list[str], reason: str) -> None:
        chunk_id = str(chunk["chunk_id"])
        current = self.coverage.get(chunk_id)
        if current is None:
            self.coverage[chunk_id] = (status, sorted(set(table_ids)), reason)
            return
        current_status, current_ids, current_reason = current
        priority = {"linked_to_canonical_source": 3, "recovered_from_raw": 2, "resolved_by_existing_table": 1}
        chosen_status = status if priority.get(status, 0) > priority.get(current_status, 0) else current_status
        self.coverage[chunk_id] = (chosen_status, sorted(set(current_ids) | set(table_ids)), f"{current_reason}; {reason}")

    def _write_terminal_statuses(self) -> None:
        if self.writer is None:
            return
        for chunk in self.deferred:
            chunk_id = str(chunk["chunk_id"])
            if chunk_id in self.recovery_failures:
                status = "recovery_failed"
                table_ids: list[str] = []
                reason = self.recovery_failures[chunk_id]
            elif chunk_id in self.coverage:
                status, table_ids, reason = self.coverage[chunk_id]
            elif chunk_id in self.ocr_required_chunks:
                status = "ocr_required"
                table_ids = []
                reason = "PDF page has no extractable text or reliable table geometry"
            else:
                asset = self.inventory[str(chunk["raw_asset_id"])]
                family = str(asset.get("file_family") or "")
                if family in {"xml", "json", "text", "archive", "pdf"}:
                    status = "not_table_after_source_inspection"
                    reason = source_not_table_reason(str(chunk.get("source_id") or ""), family)
                else:
                    status = "unsupported_table_format"
                    reason = f"No reliable table recovery route for file_family={family}"
                table_ids = []
            self.writer.write(
                "table_recovery_status.jsonl",
                {
                    "run_id": self.run_id,
                    "chunk_id": chunk_id,
                    "raw_asset_id": chunk.get("raw_asset_id"),
                    "parsed_document_id": chunk.get("parsed_document_id"),
                    "source_id": chunk.get("source_id"),
                    "compound": chunk.get("compound"),
                    "storage_path": chunk.get("storage_path"),
                    "source_url": chunk.get("source_url"),
                    "checksum": chunk.get("checksum"),
                    "source_location": chunk.get("source_location"),
                    "previous_processing_status": DEFERRED_STATUS,
                    "terminal_status": status,
                    "reason": reason,
                    "recovered_table_ids": table_ids,
                    "extractor_version": TABLE_RECOVERY_VERSION,
                },
            )
            self.status_counts[status] += 1
            self.status_source_counts[f"{chunk.get('source_id')}:{status}"] += 1

    def _merge_evidence(self) -> None:
        final = self.layout.path("evidence_candidates.jsonl", for_write=True)
        temporary = final.with_name(f".{final.name}.table-recovery.tmp")
        seen: set[str] = set()
        with temporary.open("w", encoding="utf-8") as output:
            for name in (
                "evidence_candidates_deterministic.jsonl",
                "evidence_candidates_llm.jsonl",
                "table_evidence_candidates.jsonl",
            ):
                path = self.layout.path(name)
                if not path.is_file():
                    continue
                for row in iter_jsonl(path):
                    evidence_id = str(row.get("evidence_id") or "")
                    if not evidence_id or evidence_id in seen:
                        continue
                    seen.add(evidence_id)
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    output.write("\n")
        temporary.replace(final)
        self.counts["merged_evidence_rows"] = len(seen)

    def _deduplicate_table_evidence(self) -> None:
        path = self.layout.path("table_evidence_candidates.jsonl", for_write=True)
        rows = list(iter_jsonl(path)) if path.is_file() else []
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            fingerprint = table_evidence_fingerprint(row)
            row["metadata_json"] = {**(row.get("metadata_json") or {}), "scientific_fingerprint": fingerprint}
            groups[fingerprint].append(row)

        canonical_rows: list[dict[str, Any]] = []
        duplicate_rows: list[dict[str, Any]] = []
        for fingerprint, candidates in groups.items():
            ordered = sorted(candidates, key=table_evidence_canonical_rank)
            canonical = ordered[0]
            duplicates = ordered[1:]
            canonical["metadata_json"] = {
                **(canonical.get("metadata_json") or {}),
                "duplicate_source_count": len(duplicates),
                "duplicate_ledger": "06_table_recovery/table_evidence_duplicates.jsonl" if duplicates else None,
            }
            canonical_rows.append(canonical)
            for duplicate in duplicates:
                duplicate_rows.append(
                    {
                        "run_id": self.run_id,
                        "scientific_fingerprint": fingerprint,
                        "canonical_evidence_id": canonical.get("evidence_id"),
                        "duplicate_evidence_id": duplicate.get("evidence_id"),
                        "reason": "same compound, parameter, canonical value/unit, complete row, and condition",
                        "canonical_provenance": evidence_provenance(canonical),
                        "duplicate_provenance": evidence_provenance(duplicate),
                    }
                )

        canonical_rows.sort(key=lambda row: str(row.get("evidence_id") or ""))
        duplicate_rows.sort(key=lambda row: (str(row.get("canonical_evidence_id") or ""), str(row.get("duplicate_evidence_id") or "")))
        write_jsonl_atomic(path, canonical_rows)
        write_jsonl_atomic(self.layout.path("table_evidence_duplicates.jsonl", for_write=True), duplicate_rows)
        self.counts["table_evidence"] = len(canonical_rows)
        self.counts["table_evidence_duplicates"] = len(duplicate_rows)
        self.evidence_type_counts = Counter(str(row.get("evidence_type") or "unknown") for row in canonical_rows)

    def _summary(self) -> dict[str, Any]:
        status_total = sum(self.status_counts.values())
        accepted = (
            status_total == self.counts["deferred_chunks"]
            and self.status_counts["recovery_failed"] == 0
            and len(self.seen_row_ids) == self.counts["recovered_table_rows"]
            and len(self.seen_evidence_ids) == self.counts["table_evidence_pre_dedup"]
            and self.counts["table_evidence_pre_dedup"]
            == self.counts["table_evidence"] + self.counts["table_evidence_duplicates"]
        )
        return {
            "stage": "table_recovery",
            "status": "completed" if accepted else "failed",
            "accepted": accepted,
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "extractor_version": TABLE_RECOVERY_VERSION,
            "started_at": self.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "deferred_chunk_count": self.counts["deferred_chunks"],
            "deferred_asset_count": self.counts["deferred_assets"],
            "terminal_status_count": status_total,
            "terminal_status_counts": dict(sorted(self.status_counts.items())),
            "terminal_status_source_counts": dict(sorted(self.status_source_counts.items())),
            "recovered_table_count": self.counts["recovered_tables"],
            "recovered_table_row_count": self.counts["recovered_table_rows"],
            "table_origin_counts": {
                key.split(":", 1)[1]: value
                for key, value in sorted(self.counts.items())
                if key.startswith("table_origin:")
            },
            "duplicate_table_count": self.counts["duplicate_tables_suppressed"],
            "duplicate_row_count": self.counts["duplicate_rows_suppressed"],
            "table_evidence_pre_dedup_count": self.counts["table_evidence_pre_dedup"],
            "table_evidence_count": self.counts["table_evidence"],
            "table_evidence_duplicate_count": self.counts["table_evidence_duplicates"],
            "table_evidence_type_counts": dict(sorted(self.evidence_type_counts.items())),
            "table_review_queue_count": self.counts["table_review_queue"],
            "table_review_reason_counts": dict(sorted(self.review_reason_counts.items())),
            "merged_evidence_count": self.counts["merged_evidence_rows"],
            "provenance_missing_count": 0,
        }

    def _update_pipeline_state(self, summary: dict[str, Any]) -> None:
        state = json.loads(self.layout.path("pipeline_state.json").read_text(encoding="utf-8"))
        stages = state.setdefault("stages", {})
        stages["table_recovery"] = {
            "status": summary["status"],
            "accepted": summary["accepted"],
            "summary_path": str(self.layout.path("table_recovery_summary.json", for_write=True)),
            "completed_at": summary["completed_at"],
        }
        stages.setdefault("completeness", {"status": "not_started"})
        stages.setdefault("embedding", {"status": "not_started"})
        state["current_stage"] = "table_recovery"
        self._write_json_atomic(self.layout.path("pipeline_state.json", for_write=True), state)

    def _write_progress(
        self,
        phase: str,
        *,
        completed: bool = False,
        summary: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "stage": "table_recovery",
            "run_id": self.run_id,
            "phase": phase,
            "status": "completed" if completed else "running",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "counts": dict(self.counts),
        }
        if summary is not None:
            payload["accepted"] = summary.get("accepted")
        self._write_json_atomic(self.layout.path("table_recovery_progress.json", for_write=True), payload)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


def detect_parameter(text: str) -> str | None:
    matches = [parameter for parameter, pattern in PARAMETER_PATTERNS if pattern.search(text)]
    if "clearance" in matches and RENAL_FUNCTION_RE.search(text):
        matches = [parameter for parameter in matches if parameter != "clearance"]
    return matches[0] if len(set(matches)) == 1 else None


def is_metabolite_analyte_label(text: str) -> bool:
    # M1/M2 etc. are analyte codes only at the label start or after an
    # explicit ownership phrase. A dose-normalization unit such as L/h/m2
    # must not become a metabolite designation.
    return bool(METABOLITE_HEADER.search(text) or
                re.search(r"(?:^\s*|\b(?:of|for)\s+)M\d+\b", text, re.I))


def table_structure_reliable(columns: list[str], rows: list[list[str]]) -> bool:
    if len(columns) < 2 or not rows:
        return False
    if any(len(row) != len(columns) for row in rows):
        return False
    if any(len(clean_text(column)) > 500 for column in columns):
        return False
    return any(clean_text(column) for column in columns)


def is_pdf_table_row(row: dict[str, Any]) -> bool:
    return (
        str((row.get("metadata_json") or {}).get("table_origin") or "") == "recovered_pdf_table"
        or str(row.get("storage_path") or "").lower().endswith(".pdf")
        or "#page=" in str(row.get("source_location") or "")
    )


def is_repeated_spanning_row(cells: list[str]) -> bool:
    populated = [clean_text(cell) for cell in cells if clean_text(cell)]
    return (
        len(populated) > 1
        and len(set(populated)) == 1
        and len(populated[0]) >= 8
        and bool(re.search(r"[A-Za-z]{3,}", populated[0]))
    )


def central_estimate_count(text: str) -> int:
    cleaned = re.sub(r"\(\s*n\s*=\s*\d+\s*\)", "", text, flags=re.I)
    numeric_count = len(NUMBER_RE.findall(cleaned))
    uncertainty_count = len(re.findall(r"±\s*[-+]?\d+(?:\.\d+)?", cleaned))
    return max(0, numeric_count - uncertainty_count)


def build_row_condition(
    columns: list[str],
    cells: list[str],
    *,
    excluded_indices: set[int],
) -> str | None:
    parts: list[str] = []
    for index, (column, cell) in enumerate(zip(columns, cells)):
        if index in excluded_indices:
            continue
        column = clean_text(column)
        cell = clean_text(cell)
        if not cell:
            continue
        if not re.search(r"[A-Za-z]", f"{column} {cell}"):
            continue
        if column and not column.lower().startswith("column_"):
            parts.append(f"{column}={cell}")
        else:
            parts.append(cell)
    condition = " ; ".join(parts)
    return condition[:2000] or None


def normalize_scientific_text(value: Any) -> str:
    text = str(value or "").lower().replace("μ", "µ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9µ.%±<>≤≥]+", " ", text)
    return " ".join(text.split())


def table_evidence_fingerprint(row: dict[str, Any]) -> str:
    payload = "|".join(
        [
            normalize_scientific_text(row.get("compound")),
            normalize_scientific_text(row.get("parameter")),
            str(row.get("canonical_value") if row.get("canonical_value") is not None else row.get("value")),
            normalize_scientific_text(row.get("canonical_unit") or row.get("unit")),
            normalize_scientific_text(row.get("source_text")),
            normalize_scientific_text(row.get("condition")),
        ]
    )
    return hash_text(payload)


def table_evidence_canonical_rank(row: dict[str, Any]) -> tuple[int, int, str, str]:
    source_priority = {
        "dailymed": 0,
        "pmc_oa": 1,
        "osp_pbpk_model_library": 1,
        "openfda_drug_label": 2,
    }
    context = " ".join([str(row.get("source_text") or ""), str(row.get("condition") or "")])
    evidence_priority = 0 if row.get("evidence_type") == "special_population" and SPECIAL_POPULATION_RE.search(context) else 1
    return (
        source_priority.get(str(row.get("source_id") or ""), 9),
        evidence_priority,
        str(row.get("source_location") or ""),
        str(row.get("evidence_id") or ""),
    )


def evidence_provenance(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "evidence_id",
            "source_id",
            "raw_asset_id",
            "parsed_table_id",
            "recovered_table_id",
            "table_row_id",
            "storage_path",
            "source_url",
            "checksum",
            "source_location",
            "source_text",
            "source_text_hash",
        )
    }


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.rewrite.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")
    temporary.replace(path)


def find_unit(text: str) -> str | None:
    matches = list(UNIT_RE.finditer(text))
    # The 'h' in AUC0-24h is the integration window, not an AUC unit.
    match = next((item for item in matches if "/" in item.group(0)), matches[0] if matches else None)
    return clean_text(match.group(0)) if match else None


def parse_table_value(text: str, *, unit_hint: str | None) -> tuple[float, str, str, str | None] | None:
    cleaned = clean_text(text)
    if not cleaned or RANGE_RE.search(cleaned):
        return None
    number = NUMBER_RE.search(cleaned)
    if number is None:
        return None
    try:
        value = float(number.group(1).replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    unit = find_unit(cleaned) or unit_hint
    if not unit:
        return None
    qualifier_match = re.match(r"\s*([<>]=?|[≤≥~≈])", cleaned)
    qualifier = qualifier_match.group(1) if qualifier_match else "mean" if "±" in cleaned else None
    if qualifier is None:
        for phrase, symbol in (
            (r"(?:less than or equal to|at most)", "<="),
            (r"(?:greater than or equal to|at least)", ">="),
            (r"(?:greater than|more than)", ">"),
            (r"(?:less than|fewer than)", "<"),
            (r"(?:approximately|around|about)", "~"),
        ):
            if re.match(r"\s*" + phrase + r"\s+[-+]?\d", cleaned, re.I):
                qualifier = symbol
                break
    return value, number.group(1), unit, qualifier


def valid_table_parameter_value(parameter: str, value: float, unit: str) -> bool:
    if parameter in {"protein_binding", "fraction_excreted_unchanged"}:
        return normalize_unit(unit).dimension == "fraction_percent" and 0 <= value <= 100
    return valid_pk_value_and_unit(parameter, value, unit)


def exact_target_matches(text: str, targets: Iterable[str]) -> list[str]:
    matches = [
        target
        for target in targets
        if re.search(rf"(?<![A-Za-z0-9-]){re.escape(target)}(?![A-Za-z0-9-])", text, re.I)
    ]
    return sorted(set(matches), key=str.lower)


def resolve_unique_active_ingredient(
    names: Iterable[Any],
    targets: set[str],
    status: str,
) -> tuple[str | None, str]:
    cleaned = [clean_text(str(name or "")) for name in names if clean_text(str(name or ""))]
    if len(cleaned) != 1:
        return None, "multiple_or_missing_active_ingredients"
    matches = exact_target_matches(cleaned[0], targets)
    return (matches[0], status) if len(matches) == 1 else (None, "active_ingredient_not_exact_target")


def source_not_table_reason(source_id: str, file_family: str) -> str:
    if source_id == "pmc_oa" and file_family in {"xml", "archive"}:
        return "PMC body paragraphs exclude table-wrap content; no table matched this chunk scope"
    if source_id == "pubmed":
        return "PubMed abstract XML does not contain full article table bodies"
    if source_id == "pkdb":
        return "PK-DB free-text field had no explicit structured table boundary"
    if source_id == "openfda_drug_label":
        return "No corresponding SPL table field or canonical DailyMed table matched this label field"
    if source_id == "dailymed":
        return "DailyMed section contained no table element at the deferred chunk location"
    return "Source inspection found no reliable row and column boundary for this chunk"


def clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = clean_text(str(value))
    return cleaned or None


def hash_text(text: str) -> str:
    return sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_id(*parts: str) -> str:
    return str(uuid5(NAMESPACE_URL, "|".join(parts)))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover deferred flattened tables and extract traceable table evidence.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = TableRecoveryStage(output_dir=args.output_dir).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary.get("accepted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
