from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from uuid import uuid5, NAMESPACE_URL

from pbpk_kb.extraction.ddi_extractor import extract_ddi_statements
from pbpk_kb.normalization.unit_normalizer import normalize_unit, to_canonical_value
from pbpk_kb.parsers.file_types import extension_for, file_family_for, sniff_file_family
from pbpk_kb.processing.chunk_classifier import classify_chunk, should_run_rule_extraction
from pbpk_kb.processing.llm_agent import LLMExtractionAgent, LLM_EXTRACTION_PROMPT
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
from pbpk_kb.storage.vector_index import chunk_text, hash_embedding


RUN_STATUS_FILE = "RUN_STATUS.json"
PARSER_VERSION = "raw-processing-v1"
MAX_TEXT_BYTES = 4_000_000
MAX_CHUNK_LLM_CHARS = 6000
MAX_XML_SECTIONS_PER_ASSET = 20
MAX_PDF_PAGES_PER_ASSET = 5

# [FIX 2026-07-02 · claude-opus-4-8] tightened PK value proximity {0,80}->{0,25}; removed bare "F"
# (it matched numeric arrays like "0000 F 0 0 0" -> bioavailability=0) and added a strict "F=NN%" shorthand.
PK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Cmax", re.compile(r"\bC\s*max\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Zµμ/%*.\-]+(?:/[a-zA-Zµμ0-9*.\-]+)?)?", re.I)),
    ("AUC", re.compile(r"\bAUC(?:0-[a-z0-9]+|inf|last)?\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Zµμ/%*.\-]+(?:/[a-zA-Zµμ0-9*.\-]+)?)?", re.I)),
    ("Tmax", re.compile(r"\bT\s*max\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|min|minutes)?", re.I)),
    ("half_life", re.compile(r"\b(?:half[- ]life|t1/2|t½)\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|min|minutes|day|days)?", re.I)),
    ("clearance", re.compile(r"\bclearance\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*(mL/min/kg|mL/min|L/h/kg|L/h|L/hr)?", re.I)),
    # [FIX 2026-07-02 · claude-opus-4-8] abbreviation "CL" must be followed by a clearance unit (else it matches "Cl"/chlorine, etc.)
    ("clearance", re.compile(r"\bCL\b[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\s*(mL/min/kg|mL/min|L/h/kg|L/h|L/hr)", re.I)),
    ("volume_of_distribution", re.compile(r"\bvolume of distribution\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*(L/kg|L|mL/kg)?", re.I)),
    # [FIX 2026-07-02 · claude-opus-4-8] abbreviation "Vd"/"Vz"/"Vss" must be followed by a volume unit.
    ("volume_of_distribution", re.compile(r"\b(?:Vd|Vz/F|Vz|Vss)\b[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\s*(L/kg|L|mL/kg)", re.I)),
    ("bioavailability", re.compile(r"\b(?:bioavailability|absolute bioavailability|oral bioavailability)\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*(%)?", re.I)),
    ("bioavailability", re.compile(r"(?<![A-Za-z])F\s*[=:]?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*(%)")),
]

# [FIX 2026-07-02 · claude-opus-4-8] ADME split into case-insensitive prose words + case-sensitive acronyms.
# OCT/MATE now require a digit (OCT1/MATE1...) so the month "Oct" and the word "mate" no longer match.
ADME_WORD_TERMS = re.compile(r"\b(absorption|distribution|metabolism|excretion|bioavailability|protein binding|renal clearance|hepatic clearance|transporter|P-glycoprotein|P-gp)\b", re.I)
ADME_ACRONYM_TERMS = re.compile(r"\b(CYP3A4|CYP3A|CYP2C9|CYP2C19|CYP2D6|CYP1A2|OATP1B1|OATP1B3|OATP2B1|OATP|BCRP|OCT1|OCT2|OCT3|OCTN1|OCTN2|MATE1|MATE2|UGT1A1|UGT2B7)\b")
SPECIAL_POP_TERMS = re.compile(r"\b(renal impairment|hepatic impairment|pediatric|paediatric|geriatric|elderly|pregnancy|lactation|breastfeeding|obesity|neonate|infant)\b", re.I)
# [FIX 2026-07-02 · claude-opus-4-8] removed generic "model"/"simulation" (produced parameter=model x2049 from non-PBPK text)
PBPK_TERMS = re.compile(r"\b(PBPK|physiologically[- ]based pharmacokinetic|PK-Sim|MoBi|Simcyp|GastroPlus)\b", re.I)

# [FIX 2026-07-02 · claude-opus-4-8] section candidates only fire on prose-bearing sources; excludes structured/registry
# sources (clinicaltrials/aact/chembl/pubchem/rxnorm/gtex/hpa/nhanes) whose metadata produced false ADME/special-pop flags.
SECTION_CANDIDATE_SOURCES = {"dailymed", "pubmed", "pmc_oa", "ema_epar", "openfda_drug_label", "drugcentral", "osp_tools_docs", "pkdb"}

JSON_FREE_TEXT_KEYS_BY_SOURCE = {
    "clinicaltrials": {
        "briefsummary",
        "detaileddescription",
    },
    "openfda_drug_label": {
        "clinicalpharmacology",
        "pharmacokinetics",
        "druginteractions",
        "useinspecificpopulations",
        "pregnancy",
        "nursingmothers",
        "lactation",
        "pediatricuse",
        "geriatricuse",
        "renalimpairment",
        "hepaticimpairment",
        "precautions",
    },
    "pkdb": {
        "abstract",
        "text",
    },
    "pubmed": {
        "abstract",
        "abstracttext",
    },
    "drugcentral": {
        "description",
        "pharmacology",
        "indication",
    },
}

JSON_FREE_TEXT_MAX_RECORDS_BY_SOURCE = {
    "clinicaltrials": 800,
    "openfda_drug_label": 800,
    "pkdb": 800,
    "pubmed": 1000,
}

# [FIX 2026-07-02 · claude-opus-4-8] canonical forms for parameter normalization (fixes case-split duplicates).
_KNOWN_ACRONYMS = {
    "cyp3a4": "CYP3A4", "cyp3a": "CYP3A", "cyp2c9": "CYP2C9", "cyp2c19": "CYP2C19", "cyp2d6": "CYP2D6", "cyp1a2": "CYP1A2",
    "oatp": "OATP", "oatp1b1": "OATP1B1", "oatp1b3": "OATP1B3", "oatp2b1": "OATP2B1", "bcrp": "BCRP",
    "oct1": "OCT1", "oct2": "OCT2", "oct3": "OCT3", "octn1": "OCTN1", "octn2": "OCTN2", "mate1": "MATE1", "mate2": "MATE2",
    "ugt1a1": "UGT1A1", "ugt2b7": "UGT2B7", "p-gp": "P-gp", "p-glycoprotein": "P-glycoprotein",
    "pbpk": "PBPK", "pk-sim": "PK-Sim", "mobi": "MoBi", "simcyp": "Simcyp", "gastroplus": "GastroPlus",
}


def _normalize_parameter(term: str | None) -> str | None:
    # [FIX 2026-07-02 · claude-opus-4-8] known acronyms -> canonical form; generic prose words -> lowercase
    # (fixes metabolism/Metabolism, pregnancy/Pregnancy being counted as different parameters).
    if term is None:
        return None
    key = term.strip().lower()
    return _KNOWN_ACRONYMS.get(key, key)


def _earliest_match(text: str, *patterns: re.Pattern[str]) -> re.Match[str] | None:
    # [FIX 2026-07-02 · claude-opus-4-8] return the earliest-position match across several patterns.
    best: re.Match[str] | None = None
    for pattern in patterns:
        m = pattern.search(text)
        if m and (best is None or m.start() < best.start()):
            best = m
    return best


def _looks_like_numeric_context(text: str) -> bool:
    # [FIX 2026-07-02 · claude-opus-4-8] reject PK matches whose window is a numeric array / non-prose (e.g. "0000 F 0 0 0").
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text)
    return len(nums) >= 8 and len(nums) > max(4, len(words) * 2)


@dataclass(frozen=True)
class RawAsset:
    run_id: str
    raw_asset_id: str
    source_id: str
    compound: str | None
    group: str | None
    storage_path: str
    source_url: str | None
    downloaded_at: str | None
    checksum: str | None
    content_type: str | None
    file_extension: str
    file_family: str
    file_size: int | None
    request_id: str | None
    request_key: str | None
    asset_metadata_json: dict[str, Any]


class RawProcessingPipeline:
    def __init__(
        self,
        *,
        run_root: Path,
        output_root: Path,
        checksum_mode: str = "manifest",
        max_assets: int | None = None,
        llm_mode: str = "auto",
        llm_max_requests: int = 200,
        llm_min_chunk_chars: int = 160,
        embedding_mode: str = "hash",
        stage: str = "legacy_all",
        shared_snapshot_root: Path | None = None,
    ) -> None:
        self.run_root = run_root.resolve()
        self.run_id = self.run_root.name
        self.output_dir = (output_root / self.run_id).resolve()
        self.output_paths = raw_processing_layout(self.output_dir)
        self.checksum_mode = checksum_mode
        self.max_assets = max_assets
        self.llm_mode = llm_mode
        self.llm_max_requests = llm_max_requests
        self.llm_min_chunk_chars = llm_min_chunk_chars
        self.embedding_mode = embedding_mode
        self.stage = stage
        self.shared_snapshot_root = shared_snapshot_root.resolve() if shared_snapshot_root else None
        self.agent: LLMExtractionAgent | None = None
        self.raw_assets: list[RawAsset] = []
        self.processing_failures: list[dict[str, Any]] = []
        self.completeness_rows: list[dict[str, Any]] = []
        self.llm_candidate_chunks: list[dict[str, Any]] = []
        self.llm_request_count = 0
        self.counts = Counter()
        self.source_counts: Counter[str] = Counter()
        self.compound_counts: Counter[str] = Counter()
        self.topic_counts: Counter[str] = Counter()
        self.inventory_validation_counts: Counter[str] = Counter()
        self.acquisition_status_counts: Counter[str] = Counter()
        self.inventory_ledger_counts: Counter[str] = Counter()
        # [FIX 2026-07-02 · claude-opus-4-8] pointers (T3 topic-flags) are counted separately and kept OUT of the evidence layer.
        self.pointer_type_counts: Counter[str] = Counter()
        self.completeness_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"raw": 0, "parsed": 0, "candidates": 0, "failures": 0, "raw_ids": set(), "parsed_ids": set(), "evidence_ids": set(), "failure_ids": set()}
        )

    def process(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_paths.ensure_dirs()
        if self.stage == "parse":
            from pbpk_kb.processing.parse_stage import RawParseStage

            return RawParseStage(
                run_root=self.run_root,
                output_dir=self.output_dir,
                max_assets=self.max_assets,
            ).run()
        if self.stage == "deterministic_extract":
            from pbpk_kb.processing.deterministic_extract_stage import DeterministicExtractStage

            return DeterministicExtractStage(
                run_root=self.run_root,
                output_dir=self.output_dir,
                classifier_min_chars=self.llm_min_chunk_chars,
            ).run()
        self._clear_previous_outputs()
        self._validate_run()

        self.raw_assets = self._load_raw_inventory()
        self._write_jsonl("raw_inventory.jsonl", [asset.__dict__ for asset in self.raw_assets])
        self._write_acquisition_failure_statuses()

        if self.stage == "inventory":
            failures_path = self._output_path("processing_failures.jsonl", for_write=True)
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            failures_path.touch(exist_ok=True)
            summary = self._inventory_summary()
            self._write_json("inventory_summary.json", summary)
            self._write_json("pipeline_state.json", self._inventory_pipeline_state(summary))
            return summary

        self.agent = LLMExtractionAgent(mode=self.llm_mode)
        self._write_text(self._output_path("llm_agent_prompt.md", for_write=True), LLM_EXTRACTION_PROMPT)

        for index, asset in enumerate(self.raw_assets, start=1):
            if self.max_assets is not None and index > self.max_assets:
                break
            self._process_asset(asset)

        self._run_llm_extraction()
        self._build_completeness()
        self._write_outputs()
        return self._summary()

    def _clear_previous_outputs(self) -> None:
        for name in [
            "raw_inventory.jsonl",
            "inventory_summary.json",
            "parse_summary.json",
            "parse_progress.json",
            "pipeline_state.json",
            "acquisition_failure_status.jsonl",
            "asset_parse_status.jsonl",
            "raw_asset_aliases.jsonl",
            "parsed_documents.jsonl",
            "parsed_records.jsonl",
            "parsed_tables.jsonl",
            "archive_members.jsonl",
            "asset_links.jsonl",
            "text_chunks.jsonl",
            "llm_extraction_requests.jsonl",
            "llm_extraction_responses.jsonl",
            "evidence_candidates.jsonl",
            "evidence_pointers.jsonl",  # [FIX 2026-07-02 · claude-opus-4-8] T3 pointer stream
            "embedding_index.jsonl",
            "processing_failures.jsonl",
            "completeness_matrix.csv",
            "processing_summary.json",
        ]:
            for path in self.output_paths.candidate_paths(name):
                if path.exists():
                    path.unlink()

    def _validate_run(self) -> None:
        status_path = self.run_root / RUN_STATUS_FILE
        finalization_root = self.run_root / "finalization"
        required = [
            status_path,
            self.run_root / "summary.json",
            self.run_root / "assets.jsonl",
            self.run_root / "failures.jsonl",
            self.run_root / "download_plan.json",
            finalization_root / "FINAL_ACQUISITION_RESULT.json",
            finalization_root / "FINAL_INTEGRITY_AUDIT.json",
            finalization_root / "failure_resolutions.jsonl",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required raw run file missing: {path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed" or status.get("accepted_as_final_result") is not True:
            raise ValueError(f"Raw run is not accepted as completed: {status_path}")

        final_result = json.loads((finalization_root / "FINAL_ACQUISITION_RESULT.json").read_text(encoding="utf-8"))
        if final_result.get("accepted_as_complete_raw_content") is not True:
            raise ValueError("Final acquisition result is not accepted as complete raw content")
        integrity = json.loads((finalization_root / "FINAL_INTEGRITY_AUDIT.json").read_text(encoding="utf-8"))
        if integrity.get("status") != "passed" or int(integrity.get("issue_count") or 0) != 0:
            raise ValueError("Final raw integrity audit has not passed")

        if self.shared_snapshot_root is None:
            snapshot_root = (final_result.get("osp_pbpk_model_library_snapshot") or {}).get("snapshot_root")
            if snapshot_root:
                self.shared_snapshot_root = Path(str(snapshot_root)).resolve()
        if self.shared_snapshot_root is not None:
            shared_status_path = self.shared_snapshot_root / RUN_STATUS_FILE
            shared_assets_path = self.shared_snapshot_root / "assets.jsonl"
            if not shared_status_path.exists() or not shared_assets_path.exists():
                raise FileNotFoundError(f"Final shared snapshot is incomplete: {self.shared_snapshot_root}")
            shared_status = json.loads(shared_status_path.read_text(encoding="utf-8"))
            if shared_status.get("status") != "completed" or shared_status.get("accepted_as_final_result") is not True:
                raise ValueError(f"Shared snapshot is not accepted: {shared_status_path}")

    def _load_raw_inventory(self) -> list[RawAsset]:
        assets: list[RawAsset] = []
        ledgers = [("primary_run", self.run_root / "assets.jsonl")]
        if self.shared_snapshot_root is not None:
            ledgers.append(("final_shared_snapshot", self.shared_snapshot_root / "assets.jsonl"))
        seen_asset_ids: set[str] = set()
        primary_assets: dict[str, dict[str, Any]] = {}
        for inventory_origin, ledger_path in ledgers:
            with ledger_path.open("r", encoding="utf-8", errors="replace") as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if inventory_origin == "primary_run":
                        primary_assets.setdefault(str(row.get("asset_id") or ""), row)
                    elif primary_assets.get(str(row.get("asset_id") or "")) == row:
                        self.inventory_validation_counts["shared_asset_reference_reused"] += 1
                        continue
                    raw_asset_id = str(row.get("asset_id") or _stable_id("raw", str(row.get("storage_path") or "")))
                    if raw_asset_id in seen_asset_ids:
                        self.inventory_validation_counts["duplicate_asset_id"] += 1
                        self._failure("duplicate_raw_asset_id", row, f"Duplicate asset_id across input ledgers: {raw_asset_id}")
                        continue
                    seen_asset_ids.add(raw_asset_id)
                    self.inventory_ledger_counts[inventory_origin] += 1
                    self._append_inventory_asset(assets, row, raw_asset_id, inventory_origin)
        return assets

    def _append_inventory_asset(
        self,
        assets: list[RawAsset],
        row: dict[str, Any],
        raw_asset_id: str,
        inventory_origin: str,
    ) -> None:
        path = Path(str(row.get("storage_path") or ""))
        extension = _extension(path, row.get("content_type"), row.get("url"))
        family = _file_family(path, extension, row.get("content_type"), row.get("source_id"))
        file_size = path.stat().st_size if path.exists() and path.is_file() else None
        checksum = str(row.get("checksum") or "") or None
        if not path.exists() or not path.is_file():
            self.inventory_validation_counts["missing_storage_path"] += 1
            self._failure("missing_raw_file", row, f"storage_path does not exist: {path}")
        else:
            expected_size = row.get("byte_count")
            if expected_size is not None and file_size != int(expected_size):
                self.inventory_validation_counts["file_size_mismatch"] += 1
                self._failure("raw_file_size_mismatch", row, f"manifest={expected_size} actual={file_size}")
            if self.checksum_mode == "verify" or self.stage == "inventory":
                actual = _file_checksum(path)
                self.inventory_validation_counts["checksum_verified"] += 1
                if not checksum or actual != checksum:
                    self.inventory_validation_counts["checksum_mismatch"] += 1
                    self._failure("checksum_mismatch", row, f"manifest={checksum} actual={actual}")
        metadata_path = row.get("metadata_path")
        if metadata_path and not Path(str(metadata_path)).is_file():
            self.inventory_validation_counts["missing_metadata_sidecar"] += 1
            self._failure("missing_metadata_sidecar", row, f"metadata_path does not exist: {metadata_path}")
        asset_metadata = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "asset_id",
                "source_id",
                "compound",
                "group",
                "storage_path",
                "url",
                "downloaded_at",
                "checksum",
                "content_type",
                "request_id",
                "request_key",
            }
        }
        asset_metadata["inventory_origin"] = inventory_origin
        asset = RawAsset(
            run_id=self.run_id,
            raw_asset_id=raw_asset_id,
            source_id=str(row.get("source_id") or "unknown"),
            compound=row.get("compound"),
            group=row.get("group"),
            storage_path=str(path),
            source_url=row.get("url"),
            downloaded_at=row.get("downloaded_at"),
            checksum=checksum,
            content_type=row.get("content_type"),
            file_extension=extension,
            file_family=family,
            file_size=file_size,
            request_id=row.get("request_id"),
            request_key=row.get("request_key"),
            asset_metadata_json=asset_metadata,
        )
        assets.append(asset)
        self.source_counts[asset.source_id] += 1
        self.compound_counts[asset.compound or "shared"] += 1
        stats = self.completeness_stats[(asset.compound or "shared", asset.source_id)]
        stats["raw"] += 1
        stats["raw_ids"].add(asset.raw_asset_id)

    def _write_acquisition_failure_statuses(self) -> None:
        failures_path = self.run_root / "failures.jsonl"
        resolutions_path = self.run_root / "finalization" / "failure_resolutions.jsonl"
        failure_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with failures_path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                request_id = str(row.get("request_id") or _stable_id("acquisition_failure", json.dumps(row, sort_keys=True)))
                failure_records[request_id].append(row)

        resolutions: dict[str, dict[str, Any]] = {}
        with resolutions_path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                request_id = str(row.get("original_failure_request_id") or "")
                if request_id:
                    resolutions[request_id] = row

        statuses: list[dict[str, Any]] = []
        for request_id, records in sorted(failure_records.items()):
            original = records[-1]
            resolution = resolutions.get(request_id)
            terminal_status = str(resolution.get("status")) if resolution else "unresolved"
            statuses.append(
                {
                    "run_id": self.run_id,
                    "request_id": request_id,
                    "source_id": original.get("source_id"),
                    "compound": original.get("compound"),
                    "request_key": original.get("request_key"),
                    "source_url": original.get("url"),
                    "historical_failure_record_count": len(records),
                    "historical_error": original.get("error"),
                    "terminal_status": terminal_status,
                    "resolution_id": resolution.get("resolution_id") if resolution else None,
                    "resolution_method": resolution.get("resolution_method") if resolution else None,
                    "resolved_at": resolution.get("resolved_at") if resolution else None,
                    "resolution_detail": resolution.get("detail") if resolution else None,
                    "raw_only": True,
                }
            )
            self.acquisition_status_counts[terminal_status] += 1
            if resolution is None:
                self._failure(
                    "failed_to_acquire",
                    {
                        "run_id": self.run_id,
                        "failure_type": "failed_to_acquire",
                        "source_id": original.get("source_id"),
                        "compound": original.get("compound"),
                        "request_id": request_id,
                        "source_url": original.get("url"),
                        "error": original.get("error"),
                        "raw_failure": original,
                    },
                    str(original.get("error") or "acquisition failed"),
                )
        self._write_jsonl("acquisition_failure_status.jsonl", statuses)

    def _process_asset(self, asset: RawAsset) -> None:
        path = Path(asset.storage_path)
        if not path.exists():
            return
        try:
            if asset.file_family == "json":
                self._parse_json_asset(asset)
            elif asset.file_family == "xml":
                self._parse_xml_asset(asset)
            elif asset.file_family == "html":
                self._parse_html_asset(asset)
            elif asset.file_family == "pdf":
                self._parse_pdf_asset(asset)
            elif asset.file_family == "table":
                self._parse_table_asset(asset)
            elif asset.file_family == "archive":
                self._parse_archive_asset(asset)
            elif asset.file_family == "repo":
                self._parse_repo_asset(asset)
            elif asset.file_family == "text":
                self._parse_text_asset(asset)
            else:
                self._parse_unknown_asset(asset)
        except Exception as exc:
            self._failure("parse_exception", asset.__dict__, str(exc), asset=asset)

    def _parse_json_asset(self, asset: RawAsset) -> None:
        payload = _read_json(Path(asset.storage_path))
        title = f"{asset.source_id} {asset.request_key or 'json'}"
        text = _json_text(payload)
        doc = self._add_parsed_document(asset, "json", title, text, _source_location(asset, "json_path=$"), {"json_keys": _json_keys(payload)})
        for record in _json_free_text_records(asset.source_id, payload):
            free_text_doc = self._add_parsed_document(
                asset,
                "json_free_text",
                f"{title} {record['label']}",
                record["text"],
                _source_location(asset, f"json_path={record['path']}"),
                {"json_field": record["key"], "json_path": record["path"], "source_document_id": doc["parsed_document_id"]},
            )
            self._add_chunks(asset, free_text_doc, record["text"], section_title=record["label"])
        self._extract_structured_json(asset, doc, payload)

    def _parse_xml_asset(self, asset: RawAsset) -> None:
        text = _read_text(Path(asset.storage_path), max_bytes=MAX_TEXT_BYTES)
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            if _looks_json(text):
                asset_json = RawAsset(**{**asset.__dict__, "file_family": "json", "file_extension": ".json"})
                self._parse_json_asset(asset_json)
                return
            doc = self._add_parsed_document(asset, "xml_text", f"{asset.source_id} xml text", _clean_text(text), _source_location(asset, "xml_path=/"), {})
            self._add_chunks(asset, doc, doc["text"])
            return
        title = _first_xml_text(root, {"title", "name"}) or f"{asset.source_id} XML"
        all_text = _clean_text(" ".join(root.itertext()))
        doc = self._add_parsed_document(asset, "xml", title, all_text, _source_location(asset, "xml_path=/"), {"root_tag": _local(root.tag)})
        sections = _xml_sections(root, limit=MAX_XML_SECTIONS_PER_ASSET)
        if not sections:
            self._add_chunks(asset, doc, all_text)
        for index, section in enumerate(sections, start=1):
            section_text = _clean_text(section["text"])
            if len(section_text) < 40:
                continue
            section_doc = self._add_parsed_document(
                asset,
                "xml_section",
                section.get("title") or title,
                section_text,
                _source_location(asset, f"xml_path={section.get('path') or f'/section[{index}]'}"),
                {"section_index": index, "section_title": section.get("title")},
            )
            self._add_chunks(asset, section_doc, section_text, section_title=section.get("title"))

    def _parse_html_asset(self, asset: RawAsset) -> None:
        text = _read_text(Path(asset.storage_path), max_bytes=MAX_TEXT_BYTES)
        parser = _HTMLTextExtractor()
        parser.feed(text)
        body = _clean_text(" ".join(parser.text_parts))
        title = parser.title or f"{asset.source_id} HTML"
        page_type = _html_page_type(asset, body, parser.links)
        doc = self._add_parsed_document(asset, "html", title, body, _source_location(asset, "css=body"), {"page_type": page_type, "link_count": len(parser.links)})
        self._add_chunks(asset, doc, body)

    def _parse_pdf_asset(self, asset: RawAsset) -> None:
        path = Path(asset.storage_path)
        parsed_any = False
        try:
            import fitz  # type: ignore

            document = fitz.open(str(path))
            for page_index in range(min(len(document), MAX_PDF_PAGES_PER_ASSET)):
                page = document[page_index]
                text = _clean_text(page.get_text("text"))
                if not text:
                    continue
                doc = self._add_parsed_document(asset, "pdf_page", f"{path.name} page {page_index + 1}", text, _source_location(asset, f"page={page_index + 1}"), {"page_number": page_index + 1})
                self._add_chunks(asset, doc, text)
                parsed_any = True
            document.close()
        except Exception as fitz_exc:
            try:
                from pypdf import PdfReader  # type: ignore

                reader = PdfReader(str(path))
                for page_index, page in enumerate(reader.pages[:MAX_PDF_PAGES_PER_ASSET]):
                    text = _clean_text(page.extract_text() or "")
                    if not text:
                        continue
                    doc = self._add_parsed_document(asset, "pdf_page", f"{path.name} page {page_index + 1}", text, _source_location(asset, f"page={page_index + 1}"), {"page_number": page_index + 1})
                    self._add_chunks(asset, doc, text)
                    parsed_any = True
            except Exception as pypdf_exc:
                self._failure("pdf_parser_unavailable_or_failed", {"fitz_error": str(fitz_exc), "pypdf_error": str(pypdf_exc)}, "PDF text extraction failed", asset=asset)
        if not parsed_any:
            self._failure("ocr_required_or_no_pdf_text", asset.__dict__, "No extractable PDF text was produced", asset=asset)

    def _parse_table_asset(self, asset: RawAsset) -> None:
        path = Path(asset.storage_path)
        if asset.file_extension in {".csv", ".tsv", ".txt"}:
            delimiter = "\t" if asset.file_extension in {".tsv", ".txt"} else ","
            self._parse_delimited_table(asset, path, delimiter)
        elif asset.file_extension in {".gz", ".tsv_gz"} or path.name.endswith(".tsv.gz"):
            try:
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as file:
                    sample = "".join([file.readline() for _ in range(30)])
                self._add_table_manifest(asset, "gzip_table_sample", sample, {"compressed": True})
            except Exception as exc:
                self._failure("table_gzip_parse_failed", asset.__dict__, str(exc), asset=asset)
        else:
            self._add_table_manifest(asset, "table_metadata_only", "", {"note": "No optional table parser available for this file type"})

    def _parse_delimited_table(self, asset: RawAsset, path: Path, delimiter: str) -> None:
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.reader(file, delimiter=delimiter)
            for index, row in enumerate(reader):
                if index >= 100:
                    break
                rows.append(row)
        columns = rows[0] if rows else []
        table_text = "\n".join([" | ".join(row) for row in rows[:20]])
        self._add_parsed_table(asset, None, None, 1, len(rows), len(columns), columns, rows[:100], table_text, {"sampled_rows": len(rows)})

    def _parse_archive_asset(self, asset: RawAsset) -> None:
        path = Path(asset.storage_path)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                inner = [{"path": info.filename, "size": info.file_size} for info in archive.infolist()[:2000]]
            text = "\n".join(f"{row['path']}\t{row['size']}" for row in inner[:200])
            self._add_parsed_document(asset, "archive_manifest", path.name, text, _source_location(asset, "archive_manifest=/"), {"inner_files": inner})
        else:
            self._add_parsed_document(asset, "archive_metadata", path.name, "", _source_location(asset, "archive_manifest=/"), {"note": "Archive type not expanded"})

    def _parse_repo_asset(self, asset: RawAsset) -> None:
        repo = Path(asset.storage_path)
        entries = []
        if repo.exists() and repo.is_dir():
            for path in repo.rglob("*"):
                if path.is_file() and ".git" not in path.parts:
                    rel = str(path.relative_to(repo))
                    entries.append({"path": rel, "size": path.stat().st_size, "extension": path.suffix})
        text = "\n".join(f"{row['path']}\t{row['size']}" for row in entries[:500])
        doc = self._add_parsed_document(asset, "repository_inventory", repo.name, text, _source_location(asset, "file=/"), {"file_count": len(entries), "files": entries[:2000]})
        self._add_chunks(asset, doc, text)

    def _parse_text_asset(self, asset: RawAsset) -> None:
        text = _read_text(Path(asset.storage_path), max_bytes=MAX_TEXT_BYTES)
        doc = self._add_parsed_document(asset, "text", Path(asset.storage_path).name, _clean_text(text), _source_location(asset, "text=/"), {})
        self._add_chunks(asset, doc, doc["text"])

    def _parse_unknown_asset(self, asset: RawAsset) -> None:
        path = Path(asset.storage_path)
        try:
            text = _read_text(path, max_bytes=MAX_TEXT_BYTES)
            if _looks_json(text):
                asset_json = RawAsset(**{**asset.__dict__, "file_family": "json"})
                self._parse_json_asset(asset_json)
            elif _looks_xml(text):
                asset_xml = RawAsset(**{**asset.__dict__, "file_family": "xml"})
                self._parse_xml_asset(asset_xml)
            elif len(text.strip()) > 40:
                doc = self._add_parsed_document(asset, "unknown_text", path.name, _clean_text(text), _source_location(asset, "text=/"), {})
                self._add_chunks(asset, doc, doc["text"])
            else:
                self._failure("unknown_binary", asset.__dict__, "Unknown binary or empty text", asset=asset)
        except Exception as exc:
            self._failure("unknown_parse_failed", asset.__dict__, str(exc), asset=asset)

    def _add_parsed_document(self, asset: RawAsset, content_kind: str, title: str, text: str, source_location: str, metadata: dict[str, Any]) -> dict[str, Any]:
        text = text[:200_000]
        doc = {
            "run_id": self.run_id,
            "parsed_document_id": _stable_id("parsed_document", asset.raw_asset_id, source_location, content_kind),
            "raw_asset_id": asset.raw_asset_id,
            "source_id": asset.source_id,
            "compound": asset.compound,
            "storage_path": asset.storage_path,
            "source_url": asset.source_url,
            "checksum": asset.checksum,
            "parser_name": "raw_processing_pipeline",
            "parser_version": PARSER_VERSION,
            "content_kind": content_kind,
            "title": title[:500],
            "section_title": metadata.get("section_title"),
            "source_location": source_location,
            "text": text,
            "text_hash": _hash_text(text),
            "metadata_json": metadata,
        }
        self.counts["parsed_documents"] += 1
        self._stream_jsonl("parsed_documents.jsonl", doc)
        stats = self.completeness_stats[(asset.compound or "shared", asset.source_id)]
        stats["parsed"] += 1
        stats["parsed_ids"].add(doc["parsed_document_id"])
        return doc

    def _add_parsed_table(self, asset: RawAsset, sheet_name: str | None, page_number: int | None, table_index: int, row_count: int, column_count: int, columns: list[Any], cells: list[Any], table_text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        suffix = f"sheet={sheet_name or ''}#page={page_number or ''}#table={table_index}"
        table = {
            "run_id": self.run_id,
            "parsed_table_id": _stable_id("parsed_table", asset.raw_asset_id, suffix),
            "raw_asset_id": asset.raw_asset_id,
            "source_id": asset.source_id,
            "compound": asset.compound,
            "storage_path": asset.storage_path,
            "source_url": asset.source_url,
            "checksum": asset.checksum,
            "parser_name": "raw_processing_pipeline",
            "parser_version": PARSER_VERSION,
            "source_location": _source_location(asset, suffix),
            "sheet_name": sheet_name,
            "page_number": page_number,
            "table_index": table_index,
            "row_count": row_count,
            "column_count": column_count,
            "columns_json": columns,
            "cells_json": cells,
            "table_text": table_text[:100_000],
            "table_hash": _hash_text(table_text),
            "metadata_json": metadata,
        }
        self.counts["parsed_tables"] += 1
        self._stream_jsonl("parsed_tables.jsonl", table)
        return table

    def _add_table_manifest(self, asset: RawAsset, title: str, sample: str, metadata: dict[str, Any]) -> None:
        self._add_parsed_table(asset, None, None, 1, 0, 0, [], [], sample, {"title": title, **metadata})

    def _add_chunks(self, asset: RawAsset, doc: dict[str, Any], text: str, section_title: str | None = None) -> None:
        for index, chunk in enumerate(chunk_text(text, section_title=section_title, max_tokens=180), start=1):
            if len(chunk.text.strip()) < 20:
                continue
            record = {
                "run_id": self.run_id,
                "chunk_id": _stable_id("chunk", doc["parsed_document_id"], str(index)),
                "raw_asset_id": asset.raw_asset_id,
                "parsed_document_id": doc["parsed_document_id"],
                "parsed_table_id": None,
                "source_id": asset.source_id,
                "compound": asset.compound,
                "storage_path": asset.storage_path,
                "source_url": asset.source_url,
                "checksum": asset.checksum,
                "source_location": f"{doc['source_location']}#chunk={index}",
                "chunk_index": index,
                "chunk_text": chunk.text,
                "chunk_text_hash": _hash_text(chunk.text),
                "token_count": chunk.token_count,
                "metadata_json": {"section_title": chunk.section_title, "paragraph_number": chunk.paragraph_number},
            }
            self.counts["text_chunks"] += 1
            self._stream_jsonl("text_chunks.jsonl", record)
            if self.embedding_mode == "hash":
                self._stream_embedding(record)
            if self._is_llm_candidate(record) and len(self.llm_candidate_chunks) < self.llm_max_requests:
                self.llm_candidate_chunks.append(record)
            self._extract_rule_evidence_from_chunk(asset, doc, record)

    def _extract_structured_json(self, asset: RawAsset, doc: dict[str, Any], payload: Any) -> None:
        if asset.source_id == "pubchem":
            for prop in _find_dicts_with_keys(payload, {"MolecularFormula", "MolecularWeight", "InChIKey"}):
                for key in ["MolecularFormula", "MolecularWeight", "CanonicalSMILES", "IsomericSMILES", "InChIKey", "XLogP", "TPSA"]:
                    if key in prop and prop[key] not in (None, ""):
                        self._add_evidence(
                            evidence_type="physicochemical",
                            compound=asset.compound,
                            source_id=asset.source_id,
                            raw_asset_id=asset.raw_asset_id,
                            parsed_document_id=doc["parsed_document_id"],
                            parsed_table_id=None,
                            chunk_id=None,
                            storage_path=asset.storage_path,
                            source_url=asset.source_url,
                            checksum=asset.checksum,
                            source_location=_source_location(asset, f"json_path=PropertyTable.Properties.{key}"),
                            source_text=f"{key}: {prop[key]}",
                            extraction_method="source_structured_field",
                            extractor_name="pubchem_json_mapper",
                            parameter=key,
                            value=_to_float(prop[key]),
                            value_text=str(prop[key]),
                            unit="g/mol" if key == "MolecularWeight" else None,
                            confidence=0.95,
                        )
        if asset.source_id == "chembl":
            for molecule_index, molecule in enumerate(_chembl_exact_molecules(payload, asset.compound)):
                props = molecule.get("molecule_properties") if isinstance(molecule.get("molecule_properties"), dict) else {}
                molecule_id = str(molecule.get("molecule_chembl_id") or "")
                pref_name = str(molecule.get("pref_name") or "")
                identity_label = pref_name or molecule_id or (asset.compound or "unknown")
                for key in ["full_mwt", "alogp", "psa", "hba", "hbd"]:
                    if key in props and props[key] not in (None, ""):
                        self._add_evidence(
                            evidence_type="physicochemical",
                            compound=asset.compound,
                            source_id=asset.source_id,
                            raw_asset_id=asset.raw_asset_id,
                            parsed_document_id=doc["parsed_document_id"],
                            parsed_table_id=None,
                            chunk_id=None,
                            storage_path=asset.storage_path,
                            source_url=asset.source_url,
                            checksum=asset.checksum,
                            source_location=_source_location(asset, f"json_path=molecules[{molecule_index}].molecule_properties.{key}"),
                            source_text=f"{identity_label} ({molecule_id}) {key}: {props[key]}",
                            extraction_method="source_structured_field",
                            extractor_name="chembl_json_mapper",
                            parameter=key,
                            value=_to_float(props[key]),
                            value_text=str(props[key]),
                            unit="g/mol" if key == "full_mwt" else None,
                            confidence=0.9,
                            metadata={
                                "chembl_identity_gate": "exact",
                                "molecule_chembl_id": molecule_id,
                                "pref_name": pref_name,
                                "query_compound": asset.compound,
                            },
                        )

    def _extract_rule_evidence_from_chunk(self, asset: RawAsset, doc: dict[str, Any], chunk: dict[str, Any]) -> None:
        if not should_run_rule_extraction(chunk):
            return
        text = chunk["chunk_text"]
        for parameter, pattern in PK_PATTERNS:
            for match in pattern.finditer(text):
                source_text = _sentence_around(text, match.start(), match.end())
                # [FIX 2026-07-02 · claude-opus-4-8] skip PK matches sitting inside a numeric array / non-prose window.
                if _looks_like_numeric_context(source_text):
                    continue
                value = _to_float(match.group(1))
                unit = match.group(2) if len(match.groups()) >= 2 else None
                canonical_value, canonical_unit = to_canonical_value(value, unit)
                self._add_evidence(
                    evidence_type="clinical_pk",
                    compound=asset.compound,
                    source_id=asset.source_id,
                    raw_asset_id=asset.raw_asset_id,
                    parsed_document_id=doc["parsed_document_id"],
                    parsed_table_id=None,
                    chunk_id=chunk["chunk_id"],
                    storage_path=asset.storage_path,
                    source_url=asset.source_url,
                    checksum=asset.checksum,
                    source_location=chunk["source_location"],
                    source_text=source_text,
                    extraction_method="rule_based",
                    extractor_name="pk_numeric_regex",
                    parameter=parameter,
                    value=value,
                    value_text=match.group(1),
                    unit=unit,
                    canonical_value=canonical_value,
                    canonical_unit=canonical_unit,
                    confidence=0.55,
                )
        # [FIX 2026-07-02 · claude-opus-4-8] keyword-only section matches are now T3 topic-pointers, routed OUT of the
        # evidence layer (evidence_pointers.jsonl), not evidence rows. Only from prose sources; source_text around the match.
        if asset.source_id in SECTION_CANDIDATE_SOURCES:
            adme_match = _earliest_match(text, ADME_ACRONYM_TERMS, ADME_WORD_TERMS)
            if adme_match:
                self._add_pointer(asset, doc, chunk, "adme", "adme_text_candidate", adme_match, 0.45)
            pop_match = SPECIAL_POP_TERMS.search(text)
            if pop_match:
                self._add_pointer(asset, doc, chunk, "special_population", "special_population_text_candidate", pop_match, 0.45)
        pbpk_match = PBPK_TERMS.search(text)
        if pbpk_match and asset.source_id in {"pubmed", "pmc_oa", "osp_pbpk_model_library", "osp_tools_docs"}:
            self._add_pointer(asset, doc, chunk, "public_pbpk_model", "public_model_text_candidate", pbpk_match, 0.4)
        # [FIX 2026-07-02 · claude-opus-4-8] pass source_id (drives study_type) and downgrade confidence when the
        # sentence is target-ambiguous (>1 enzyme/transporter), so proximity-guessed attributions are flagged for review.
        for ddi in extract_ddi_statements(text, source_id=asset.source_id):
            ratio = ddi.auc_ratio or ddi.cmax_ratio
            self._add_evidence(
                evidence_type="ddi",
                compound=asset.compound,
                source_id=asset.source_id,
                raw_asset_id=asset.raw_asset_id,
                parsed_document_id=doc["parsed_document_id"],
                parsed_table_id=None,
                chunk_id=chunk["chunk_id"],
                storage_path=asset.storage_path,
                source_url=asset.source_url,
                checksum=asset.checksum,
                source_location=chunk["source_location"],
                source_text=ddi.evidence_text,
                extraction_method="rule_based",
                extractor_name="ddi_regex",
                parameter=ddi.enzyme_or_transporter or ddi.interaction_type,
                value=ratio,
                value_text=str(ratio) if ratio else None,
                unit="ratio" if ratio else None,
                condition=ddi.population,
                confidence=0.35 if ddi.ambiguous_target else 0.55,
                metadata={
                    "interaction_type": ddi.interaction_type,
                    "auc_ratio": ddi.auc_ratio,
                    "cmax_ratio": ddi.cmax_ratio,
                    "study_type": ddi.study_type,
                    "all_targets": list(ddi.all_targets),
                    "ambiguous_target": ddi.ambiguous_target,
                },
            )

    def _add_pointer(self, asset: RawAsset, doc: dict[str, Any], chunk: dict[str, Any], topic: str, extractor_name: str, match: "re.Match[str]", confidence: float) -> None:
        # Keyword-only topic pointers have no value or asserted relation, so they
        # remain retrieval/LLM-routing hints and never enter the evidence stream.
        text = chunk["chunk_text"]
        source_text = _clean_text(_sentence_around(text, match.start(), match.end()))
        if not asset.raw_asset_id or not chunk["source_location"] or not source_text or not asset.checksum:
            return
        pointer = {
            "run_id": self.run_id,
            "pointer_id": _stable_id("pointer", topic, asset.raw_asset_id, chunk["source_location"], match.group(0), _hash_text(source_text)),
            "evidence_tier": "pointer",
            "topic": topic,
            "matched_term": _normalize_parameter(match.group(0)),
            "compound": asset.compound,
            "source_id": asset.source_id,
            "raw_asset_id": asset.raw_asset_id,
            "parsed_document_id": doc["parsed_document_id"],
            "chunk_id": chunk["chunk_id"],
            "storage_path": asset.storage_path,
            "source_url": asset.source_url,
            "checksum": asset.checksum,
            "source_location": chunk["source_location"],
            "source_text": source_text[:1000],
            "source_text_hash": _hash_text(source_text),
            "extractor_name": extractor_name,
            "extractor_version": PARSER_VERSION,
            "confidence": confidence,
            "note": "topic pointer only; not evidence (no value, no asserted relation)",
        }
        self._stream_jsonl("evidence_pointers.jsonl", pointer)
        self.counts["evidence_pointers"] += 1
        self.pointer_type_counts[topic] += 1

    def _add_evidence(
        self,
        *,
        evidence_type: str,
        compound: str | None,
        source_id: str,
        raw_asset_id: str,
        parsed_document_id: str | None,
        parsed_table_id: str | None,
        chunk_id: str | None,
        storage_path: str,
        source_url: str | None,
        checksum: str | None,
        source_location: str,
        source_text: str,
        extraction_method: str,
        extractor_name: str,
        parameter: str | None = None,
        value: float | None = None,
        value_text: str | None = None,
        unit: str | None = None,
        canonical_value: float | None = None,
        canonical_unit: str | None = None,
        condition: str | None = None,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        source_text = _clean_text(source_text)
        if not raw_asset_id or not source_location or not source_text or not checksum:
            self._failure("invalid_evidence_provenance", {"source_location": source_location, "source_text": source_text[:200]}, "Evidence missing required provenance")
            return
        if canonical_value is None and value is not None:
            canonical_value, canonical_unit = to_canonical_value(value, unit)
        normalized_unit = normalize_unit(unit)
        evidence = {
            "run_id": self.run_id,
            # [FIX 2026-07-02 · claude-opus-4-8] include value_text + full source_text hash (was source_text[:160] only),
            # which collided across distinct matches in the same chunk (12,334 duplicate evidence_ids in the prior run).
            "evidence_id": _stable_id("evidence", evidence_type, raw_asset_id, source_location, parameter or "", value_text or "", _hash_text(source_text)),
            "evidence_type": evidence_type,
            "compound": compound,
            "source_id": source_id,
            "raw_asset_id": raw_asset_id,
            "parsed_document_id": parsed_document_id,
            "parsed_table_id": parsed_table_id,
            "chunk_id": chunk_id,
            "storage_path": storage_path,
            "source_url": source_url,
            "checksum": checksum,
            "source_location": source_location,
            "source_text": source_text[:4000],
            "source_text_hash": _hash_text(source_text),
            # [FIX 2026-07-02 · claude-opus-4-8] tier real evidence: T1 quantitative (has numeric value) vs T2 qualitative
            # (relation/assertion, no number). T3 pointers never reach here (see _add_pointer).
            "evidence_tier": "quantitative" if _finite(value) else "qualitative",
            "extraction_method": extraction_method,
            "extractor_name": extractor_name,
            "extractor_version": PARSER_VERSION,
            "extraction_confidence": confidence,
            "curation_status": "needs_review",
            "parameter": parameter,
            "value": value if _finite(value) else None,
            "value_text": value_text,
            "unit": unit,
            "canonical_value": canonical_value if _finite(canonical_value) else None,
            "canonical_unit": canonical_unit or normalized_unit.canonical_unit,
            "condition": condition,
            "population": None,
            "route": None,
            "dose": None,
            "metadata_json": metadata or {},
        }
        self._stream_jsonl("evidence_candidates.jsonl", evidence)
        self.topic_counts[evidence_type] += 1
        self.counts["evidence_candidates"] += 1
        stats = self.completeness_stats[(compound or "shared", source_id)]
        stats["candidates"] += 1
        stats["evidence_ids"].add(evidence["evidence_id"])

    def _run_llm_extraction(self) -> None:
        if self.agent is None:
            raise RuntimeError("LLM agent is not initialized for this stage")
        request_rows: list[dict[str, Any]] = []
        response_rows: list[dict[str, Any]] = []
        for chunk in self.llm_candidate_chunks:
            if self.llm_request_count >= self.llm_max_requests:
                break
            payload = {
                "request_id": _stable_id("llm_request", chunk["chunk_id"]),
                "run_id": self.run_id,
                "source_id": chunk["source_id"],
                "compound": chunk["compound"],
                "raw_asset_id": chunk["raw_asset_id"],
                "storage_path": chunk["storage_path"],
                "source_url": chunk["source_url"],
                "parsed_document_id": chunk["parsed_document_id"],
                "parsed_table_id": chunk["parsed_table_id"],
                "chunk_id": chunk["chunk_id"],
                "content_kind": "text_chunk",
                "source_location": chunk["source_location"],
                "checksum": chunk["checksum"],
                "input_text_or_table": chunk["chunk_text"][:MAX_CHUNK_LLM_CHARS],
            }
            request_record = self.agent.build_request_record(payload)
            request_rows.append(request_record)
            result = self.agent.extract(request_record)
            response_record = {
                "request_id": request_record["request_id"],
                "run_id": self.run_id,
                "source_id": chunk["source_id"],
                "raw_asset_id": chunk["raw_asset_id"],
                "parsed_document_id": chunk["parsed_document_id"],
                "parsed_table_id": chunk["parsed_table_id"],
                "chunk_id": chunk["chunk_id"],
                "source_location": chunk["source_location"],
                "model_name": result.model_name,
                "status": result.status,
                "error": result.error,
                "latency_seconds": result.latency_seconds,
                "output_hash": _hash_text(result.raw_response or json.dumps(result.candidates, sort_keys=True)),
                "candidate_count": len(result.candidates),
                "raw_response": result.raw_response,
                "candidates": result.candidates,
            }
            response_rows.append(response_record)
            self.llm_request_count += 1
            if result.status == "ok":
                self._add_llm_evidence(chunk, result.candidates)
        self._write_jsonl("llm_extraction_requests.jsonl", request_rows)
        self._write_jsonl("llm_extraction_responses.jsonl", response_rows)

    def _is_llm_candidate(self, chunk: dict[str, Any]) -> bool:
        return classify_chunk(chunk, min_chars=self.llm_min_chunk_chars).llm_required

    def _add_llm_evidence(self, chunk: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
        for candidate in candidates:
            source_text = str(candidate.get("source_text") or "").strip()
            if not source_text or source_text not in chunk["chunk_text"]:
                self._failure("llm_source_text_not_found", candidate, "LLM source_text was not found in chunk", chunk=chunk)
                continue
            evidence_type = str(candidate.get("evidence_type") or "clinical_pk")
            unit = candidate.get("unit")
            value = _to_float(candidate.get("value"))
            canonical_value, canonical_unit = to_canonical_value(value, unit)
            self._add_evidence(
                evidence_type=evidence_type,
                compound=chunk["compound"],
                source_id=chunk["source_id"],
                raw_asset_id=chunk["raw_asset_id"],
                parsed_document_id=chunk["parsed_document_id"],
                parsed_table_id=chunk["parsed_table_id"],
                chunk_id=chunk["chunk_id"],
                storage_path=chunk["storage_path"],
                source_url=chunk["source_url"],
                checksum=chunk["checksum"],
                source_location=chunk["source_location"],
                source_text=source_text,
                extraction_method="llm_assisted",
                extractor_name="pbpk_llm_evidence_agent",
                parameter=str(candidate.get("parameter") or ""),
                value=value,
                value_text=str(candidate.get("value")) if candidate.get("value") is not None else None,
                unit=unit,
                canonical_value=canonical_value,
                canonical_unit=canonical_unit,
                condition=candidate.get("condition"),
                confidence=_to_float(candidate.get("confidence")) or 0.5,
                metadata={"llm_candidate": candidate},
            )

    def _stream_embedding(self, chunk: dict[str, Any]) -> None:
        if self.embedding_mode != "hash":
            return
        vector = hash_embedding(chunk["chunk_text"])
        record = {
            "embedding_id": _stable_id("embedding", chunk["chunk_id"]),
            "run_id": self.run_id,
            "source_id": chunk["source_id"],
            "raw_asset_id": chunk["raw_asset_id"],
            "parsed_document_id": chunk["parsed_document_id"],
            "parsed_table_id": chunk["parsed_table_id"],
            "chunk_id": chunk["chunk_id"],
            "evidence_id": None,
            "storage_path": chunk["storage_path"],
            "source_url": chunk["source_url"],
            "checksum": chunk["checksum"],
            "source_location": chunk["source_location"],
            "text_hash": chunk["chunk_text_hash"],
            "embedding_model": "hashing-embedding-v0",
            "embedding_model_version": "local",
            "embedding_vector": vector,
        }
        self.counts["embeddings"] += 1
        self._stream_jsonl("embedding_index.jsonl", record)

    def _build_completeness(self) -> None:
        for (compound, source_id), stats in sorted(self.completeness_stats.items()):
            status = "candidate_extracted" if stats["candidates"] else "parsed" if stats["parsed"] else "acquired" if stats["raw"] else "failed_to_acquire"
            if stats["failures"] and not stats["raw"]:
                status = "failed_to_acquire"
            self.completeness_rows.append(
                {
                    "run_id": self.run_id,
                    "compound": compound,
                    "source_id": source_id,
                    "topic": "raw_processing",
                    "status": status,
                    "status_reason": f"raw={stats['raw']} parsed={stats['parsed']} candidates={stats['candidates']} failures={stats['failures']}",
                    "raw_asset_count": stats["raw"],
                    "parsed_count": stats["parsed"],
                    "candidate_count": stats["candidates"],
                    "failure_count": stats["failures"],
                    "raw_asset_ids": sorted(stats["raw_ids"])[:1000],
                    "parsed_document_ids": sorted(stats["parsed_ids"])[:1000],
                    "evidence_ids": sorted(stats["evidence_ids"])[:1000],
                    "failure_ids": sorted(stats["failure_ids"])[:1000],
                }
            )

    def _write_outputs(self) -> None:
        for name in ["parsed_documents.jsonl", "parsed_tables.jsonl", "text_chunks.jsonl", "evidence_candidates.jsonl", "evidence_pointers.jsonl", "embedding_index.jsonl", "processing_failures.jsonl", "llm_extraction_requests.jsonl", "llm_extraction_responses.jsonl"]:
            path = self._output_path(name, for_write=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        self._write_csv("completeness_matrix.csv", self.completeness_rows)
        self._write_json("processing_summary.json", self._summary())

    def _summary(self) -> dict[str, Any]:
        if self.agent is None:
            raise RuntimeError("LLM agent is not initialized for legacy processing summary")
        return {
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "output_dir": str(self.output_dir),
            "raw_asset_count": len(self.raw_assets),
            "parsed_document_count": self.counts["parsed_documents"],
            "parsed_table_count": self.counts["parsed_tables"],
            "text_chunk_count": self.counts["text_chunks"],
            "evidence_candidate_count": self.counts["evidence_candidates"],
            # [FIX 2026-07-02 · claude-opus-4-8] T3 pointers reported separately; they are not part of the evidence layer.
            "evidence_pointer_count": self.counts["evidence_pointers"],
            "evidence_pointer_counts_by_topic": dict(self.pointer_type_counts),
            "embedding_count": self.counts["embeddings"],
            "processing_failure_count": len(self.processing_failures),
            "llm_request_count": self.llm_request_count,
            "llm_mode": self.llm_mode,
            "llm_model": self.agent.model_name,
            "llm_api_key_present": bool(self.agent.api_key),
            "source_counts": dict(self.source_counts),
            "compound_counts": dict(self.compound_counts),
            "evidence_counts_by_type": dict(self.topic_counts),
        }

    def _inventory_summary(self) -> dict[str, Any]:
        family_counts = Counter(asset.file_family for asset in self.raw_assets)
        unique_asset_count = len({asset.raw_asset_id for asset in self.raw_assets})
        unresolved_count = self.acquisition_status_counts.get("unresolved", 0)
        accepted = len(self.processing_failures) == 0 and unique_asset_count == len(self.raw_assets) and unresolved_count == 0
        return {
            "stage": "inventory",
            "status": "completed" if accepted else "failed",
            "accepted": accepted,
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "shared_snapshot_root": str(self.shared_snapshot_root) if self.shared_snapshot_root else None,
            "output_dir": str(self.output_dir),
            "raw_inventory_path": str(self._output_path("raw_inventory.jsonl", for_write=True)),
            "raw_asset_count": len(self.raw_assets),
            "unique_raw_asset_id_count": unique_asset_count,
            "raw_asset_bytes": sum(int(asset.file_size or 0) for asset in self.raw_assets),
            "input_ledger_counts": dict(sorted(self.inventory_ledger_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "compound_counts": dict(sorted(self.compound_counts.items())),
            "file_family_counts": dict(sorted(family_counts.items())),
            "inventory_validation_counts": dict(sorted(self.inventory_validation_counts.items())),
            "processing_failure_count": len(self.processing_failures),
            "historical_failure_unique_request_count": sum(self.acquisition_status_counts.values()),
            "acquisition_terminal_status_counts": dict(sorted(self.acquisition_status_counts.items())),
            "unresolved_acquisition_failure_count": unresolved_count,
            "checksum_mode": "verify" if self.stage == "inventory" else self.checksum_mode,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _inventory_pipeline_state(self, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "output_dir": str(self.output_dir),
            "current_stage": "inventory",
            "stages": {
                "inventory": {
                    "status": "completed" if summary["accepted"] else "failed",
                    "accepted": summary["accepted"],
                    "summary_path": str(self._output_path("inventory_summary.json", for_write=True)),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                "parse": {"status": "not_started"},
                "deterministic_extract": {"status": "not_started"},
                "llm_extract": {"status": "not_started"},
                "completeness": {"status": "not_started"},
                "embedding": {"status": "not_started"},
            },
        }

    def _write_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        path = self._output_path(name, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default))
                file.write("\n")

    def _stream_jsonl(self, name: str, row: dict[str, Any]) -> None:
        path = self._output_path(name, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default))
            file.write("\n")

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        path = self._output_path(name, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _write_csv(self, name: str, rows: list[dict[str, Any]]) -> None:
        path = self._output_path(name, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0])
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})

    def _output_path(self, name: str, *, for_write: bool = False) -> Path:
        return self.output_paths.path(name, for_write=for_write)

    def _failure(self, failure_type: str, payload: dict[str, Any], error: str, *, asset: RawAsset | None = None, chunk: dict[str, Any] | None = None) -> None:
        record = {
            "run_id": self.run_id,
            "failure_type": failure_type,
            "source_id": asset.source_id if asset else payload.get("source_id"),
            "compound": asset.compound if asset else payload.get("compound"),
            "raw_asset_id": asset.raw_asset_id if asset else payload.get("raw_asset_id"),
            "storage_path": asset.storage_path if asset else payload.get("storage_path"),
            "source_url": asset.source_url if asset else payload.get("source_url") or payload.get("url"),
            "checksum": asset.checksum if asset else payload.get("checksum"),
            "source_location": chunk.get("source_location") if chunk else payload.get("source_location"),
            "error": error,
            "payload": payload,
        }
        self.processing_failures.append(record)
        self._stream_jsonl("processing_failures.jsonl", record)
        key = (record.get("compound") or "shared", record.get("source_id") or "unknown")
        stats = self.completeness_stats[key]
        stats["failures"] += 1
        stats["failure_ids"].add(_stable_id("failure", json.dumps(record, sort_keys=True, default=str)[:500]))


def _source_location(asset: RawAsset, suffix: str) -> str:
    return f"{asset.source_id}:{Path(asset.storage_path).name}#{suffix}"


def _chembl_exact_molecules(payload: Any, compound: str | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not compound:
        return []
    molecules = payload.get("molecules")
    if not isinstance(molecules, list):
        return []
    target = _normalize_chembl_identity_name(compound)
    exact: list[dict[str, Any]] = []
    for molecule in molecules:
        if not isinstance(molecule, dict):
            continue
        names = {molecule.get("pref_name"), molecule.get("molecule_chembl_id")}
        synonyms = molecule.get("molecule_synonyms")
        if isinstance(synonyms, list):
            for synonym in synonyms:
                if isinstance(synonym, dict):
                    names.add(synonym.get("molecule_synonym"))
                    names.add(synonym.get("synonyms"))
                else:
                    names.add(synonym)
        normalized_names = {_normalize_chembl_identity_name(str(name)) for name in names if name}
        if target and target in normalized_names:
            exact.append(molecule)
    return exact


def _normalize_chembl_identity_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _file_checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _stable_id(*parts: str) -> str:
    return str(uuid5(NAMESPACE_URL, "|".join(parts)))


def _extension(path: Path, content_type: str | None, url: str | None) -> str:
    return extension_for(path, content_type, url)


def _file_family(path: Path, extension: str, content_type: str | None, source_id: str | None) -> str:
    return file_family_for(path, extension, content_type, source_id)


def _sniff_file_family(path: Path) -> str:
    return sniff_file_family(path)


def _read_text(path: Path, max_bytes: int = MAX_TEXT_BYTES) -> str:
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    text = _read_text(path, max_bytes=MAX_TEXT_BYTES)
    return json.loads(text)


def _json_text(payload: Any) -> str:
    return _clean_text(json.dumps(payload, ensure_ascii=False, indent=2)[:200_000])


def _json_free_text_records(source_id: str, payload: Any) -> list[dict[str, str]]:
    allowed_keys = JSON_FREE_TEXT_KEYS_BY_SOURCE.get(source_id, set())
    if not allowed_keys:
        return []
    max_records = JSON_FREE_TEXT_MAX_RECORDS_BY_SOURCE.get(source_id, 500)
    rows: list[dict[str, str]] = []

    def walk(value: Any, path: str, key: str | None = None) -> None:
        normalized_key = _normalize_json_key(key)
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                child_path = f"{path}.{child_key}" if path else f"$.{child_key}"
                walk(child_value, child_path, str(child_key))
            return
        if isinstance(value, list):
            if normalized_key in allowed_keys and _json_free_text_path_allowed(source_id, normalized_key, path) and all(isinstance(item, str) for item in value):
                text = _clean_text("\n".join(str(item) for item in value))
                add_row(normalized_key, path, text)
                return
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", key)
            return
        if isinstance(value, str) and normalized_key in allowed_keys and _json_free_text_path_allowed(source_id, normalized_key, path):
            add_row(normalized_key, path, value)

    def add_row(key: str, path: str, text: str) -> None:
        cleaned = _clean_text(text)
        if len(cleaned) < 40:
            return
        rows.append(
            {
                "key": key,
                "label": key.replace("_", " "),
                "path": path or "$",
                "text": cleaned[:100_000],
            }
        )

    walk(payload, "$")
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        dedup_key = (row["path"], _hash_text(row["text"]))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        unique.append(row)
    return unique[:max_records]


def _normalize_json_key(key: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (key or "").lower())


def _json_free_text_path_allowed(source_id: str, key: str, path: str) -> bool:
    normalized_path = re.sub(r"[^a-z0-9]+", "", path.lower())
    if source_id == "clinicaltrials":
        return any(term in normalized_path for term in ["descriptionmodule", "eligibilitymodule"])
    if source_id == "pkdb":
        if key == "abstract":
            return "reference" in normalized_path
        if key == "text":
            return any(term in normalized_path for term in ["descriptions", "comments", "interventions", "groupset"])
        return False
    if source_id == "pubmed":
        return key in {"abstract", "abstracttext"}
    return True


def _json_keys(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        return sorted(str(key) for key in payload.keys())[:100]
    if isinstance(payload, list):
        return ["[]"]
    return [type(payload).__name__]


def _find_dicts_with_keys(payload: Any, keys: set[str]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if keys.issubset(payload.keys()) or any(key in payload for key in keys):
            found.append(payload)
        for value in payload.values():
            found.extend(_find_dicts_with_keys(value, keys))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_find_dicts_with_keys(item, keys))
    return found[:50]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _looks_json(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _looks_xml(text: str) -> bool:
    return text.lstrip().startswith("<")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _first_xml_text(root: ET.Element, names: set[str]) -> str | None:
    for element in root.iter():
        if _local(element.tag).lower() in names and element.text:
            return _clean_text(element.text)
    return None


def _xml_sections(root: ET.Element, *, limit: int) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for index, element in enumerate(root.iter(), start=1):
        tag = _local(element.tag).lower()
        if tag not in {"section", "component", "abstract"}:
            continue
        text = _clean_text(" ".join(element.itertext()))
        if len(text) < 80:
            continue
        title = None
        for child in list(element):
            if _local(child.tag).lower() in {"title", "caption"}:
                title = _clean_text(" ".join(child.itertext()))
                break
        sections.append({"path": f"//{tag}[{index}]", "title": title, "text": text})
        if len(sections) >= limit:
            break
    return sections


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self.title: str | None = None
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "nav", "footer"}:
            self._skip_depth += 1
        if lowered == "title":
            self._in_title = True
        if lowered in {"a", "link"}:
            attrs_dict = {key.lower(): value for key, value in attrs}
            href = attrs_dict.get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "nav", "footer"} and self._skip_depth:
            self._skip_depth -= 1
        if lowered == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = _clean_text(data)
        if not cleaned:
            return
        if self._in_title:
            self.title = cleaned
        self.text_parts.append(cleaned)


def _html_page_type(asset: RawAsset, body: str, links: list[str]) -> str:
    lowered = body.lower()
    if "download" in lowered or any("download" in link.lower() for link in links):
        return "download_page"
    if asset.compound:
        return "record_page"
    if len(body) > 2000:
        return "documentation_page"
    return "landing_page"


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _sentence_around(text: str, start: int, end: int) -> str:
    left = -1
    for index in range(start - 1, -1, -1):
        if text[index] in ";\n" or (text[index] == "." and _is_sentence_terminal_period(text, index)):
            left = index
            break
    right = -1
    for index in range(end, len(text)):
        if text[index] in ";\n" or (text[index] == "." and _is_sentence_terminal_period(text, index)):
            right = index
            break
    if right == -1:
        right = min(len(text), end + 220)
    return _clean_text(text[left + 1 : right + 1])[:1000]


def _is_sentence_terminal_period(text: str, index: int) -> bool:
    if index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False
    if text[max(0, index - 1) : index + 3].lower() in {"e.g.", "i.e."}:
        return False
    prefix = text[max(0, index - 12) : index + 1].lower()
    if re.search(r"(?:\be\.g|\bi\.e|\bvs|\bet\s+al|\bfig|\bdr|\bmr|\bmrs|\bprof)\.$", prefix):
        return False
    return True


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one explicit PBPK raw-processing stage.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shared-snapshot-root", default=None)
    parser.add_argument("--stage", choices=["inventory", "parse", "deterministic_extract", "legacy_all"], default="inventory")
    parser.add_argument("--checksum-mode", choices=["manifest", "verify"], default="verify")
    parser.add_argument("--max-assets", type=int, default=None)
    parser.add_argument("--llm-mode", choices=["off", "auto", "live", "audit"], default="off")
    parser.add_argument("--llm-max-requests", type=int, default=200)
    parser.add_argument("--embedding-mode", choices=["off", "hash"], default="off")
    args = parser.parse_args()
    pipeline = RawProcessingPipeline(
        run_root=args.run_root,
        output_root=args.output_root,
        checksum_mode=args.checksum_mode,
        max_assets=args.max_assets,
        llm_mode=args.llm_mode,
        llm_max_requests=args.llm_max_requests,
        embedding_mode=args.embedding_mode,
        stage=args.stage,
        shared_snapshot_root=Path(args.shared_snapshot_root) if args.shared_snapshot_root else None,
    )
    summary = pipeline.process()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.stage in {"inventory", "parse", "deterministic_extract"} and not summary.get("accepted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
