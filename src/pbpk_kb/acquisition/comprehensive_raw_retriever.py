from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
import re
import ssl
import subprocess
import threading
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html.parser import HTMLParser
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from pbpk_kb.acquisition.precision import (
    CLINICALTRIAL_CANDIDATE_FIELDS,
    HIGH_PRECISION_QUERY_FAMILIES,
    classify_clinical_trial,
    clinicaltrial_candidate_query,
    classify_dailymed_listing,
    classify_openfda_label,
    classify_pubmed_record,
    high_precision_literature_query,
    pubmed_records_from_xml,
)
from pbpk_kb.acquisition.pkdb_bulk_export import (
    PKDBBulkExportError,
    inspect_pkdb_bulk_export,
)
from pbpk_kb.paths import raw_output_root

DEFAULT_OUTPUT_ROOT = raw_output_root()
DEFAULT_COMPOUNDS = [
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
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
# NCBI moved the legacy PMC Article Dataset tree under this prefix on 2026-04-13.
# The OA Web Service still returns legacy FTP locations during the transition.
PMC_LEGACY_FTP_PREFIX = "/pub/pmc/"
PMC_DEPRECATED_HTTPS_PREFIX = "/pub/pmc/deprecated/"
CURL_FIRST_HOST_SUFFIXES = (
    "api.github.com",
    "clinicaltrials.gov",
    "dailymed.nlm.nih.gov",
    "pk-db.com",
    "ncbi.nlm.nih.gov",
)

DOWNLOAD_EXTENSIONS = (
    ".zip",
    ".gz",
    ".tgz",
    ".tar",
    ".tar.gz",
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".xml",
    ".json",
    ".pdf",
    ".sdf",
    ".xpt",
)

SOURCE_MODE = {
    "pubmed": "targeted_retrieval",
    "pmc_oa": "linked_resource_expansion",
    "dailymed": "targeted_retrieval",
    "fda_label_search": "targeted_retrieval",
    "drugs_fda": "corpus_snapshot",
    "fda_ddi": "corpus_snapshot",
    "pubchem": "targeted_retrieval",
    "chembl": "targeted_retrieval",
    "bindingdb": "targeted_retrieval",
    "drugcentral": "corpus_snapshot",
    "rxnorm": "targeted_retrieval",
    "uniprot": "linked_resource_expansion",
    "hpa": "corpus_snapshot",
    "gtex": "corpus_snapshot",
    "clinicaltrials": "targeted_retrieval",
    "aact": "corpus_snapshot",
    "nhanes": "corpus_snapshot",
    "ema_epar": "corpus_snapshot",
    "fda_orange_book": "corpus_snapshot",
    "fda_inactive_ingredient_database": "corpus_snapshot",
    "openfda_drug_label": "targeted_retrieval",
    "epa_comptox_httk": "targeted_retrieval",
    "osp_pbpk_model_library": "repository_mirror",
    "osp_tools_docs": "repository_mirror",
    "pkdb": "targeted_retrieval",
}


class DailyMedSPLContentError(ValueError):
    """The finite attempts returned content that is not the requested SPL."""


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "compound"


def load_source_registry(path: Path) -> list[dict[str, str | None]]:
    entries: list[dict[str, str | None]] = []
    current: dict[str, str | None] | None = None
    scalar_field = re.compile(r"^\s{2}([a-zA-Z0-9_]+):\s*(.*)\s*$")
    with path.open("r", encoding="utf-8") as source_file:
        for line in source_file:
            if line.startswith("- source_id:"):
                if current:
                    entries.append(current)
                current = {"source_id": clean_yaml_scalar(line.split(":", 1)[1])}
                continue
            if current is None:
                continue
            match = scalar_field.match(line)
            if not match:
                continue
            key, value = match.groups()
            if key in {"source_name", "source_family", "access_url", "api_url", "access_method", "refresh_frequency", "license_notes", "connector", "parser", "enabled"}:
                current[key] = clean_yaml_scalar(value)
    if current:
        entries.append(current)
    return [entry for entry in entries if str(entry.get("enabled", "true")).lower() != "false"]


def clean_yaml_scalar(value: str) -> str | None:
    value = value.strip()
    if not value or value == "null":
        return None
    if value in {"true", "false"}:
        return value
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() not in {"a", "link"}:
            return
        attrs_dict = {key.lower(): value for key, value in attrs}
        href = attrs_dict.get("href")
        if href:
            self.links.append(href)


class ComprehensiveRawRetriever:
    def __init__(
        self,
        *,
        output_root: Path,
        compounds: list[str],
        source_registry_path: Path,
        timeout_seconds: int,
        delay_seconds: float,
        ncbi_batch_size: int,
        page_size: int,
        dailymed_workers: int,
        pkdb_workers: int,
        download_oa_packages: bool,
        mirror_repositories: bool,
        include_shared_sources: bool,
        include_combination_labels: bool,
        max_pmc_fulltext_per_compound: int,
        max_pmc_oa_packages_per_compound: int,
        official_run: bool,
        clinicaltrials_workers: int = 2,
        shared_source_ids: list[str] | None = None,
        shared_cache_root: Path | None = None,
    ) -> None:
        self.output_root = output_root
        self.compounds = compounds
        self.source_registry_path = source_registry_path
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds
        self.ncbi_batch_size = ncbi_batch_size
        self.page_size = page_size
        self.dailymed_workers = dailymed_workers
        self.pkdb_workers = pkdb_workers
        self.download_oa_packages = download_oa_packages
        self.mirror_repositories = mirror_repositories
        self.include_shared_sources = include_shared_sources
        self.include_combination_labels = include_combination_labels
        self.max_pmc_fulltext_per_compound = max_pmc_fulltext_per_compound
        self.max_pmc_oa_packages_per_compound = max_pmc_oa_packages_per_compound
        self.official_run = official_run
        self.clinicaltrials_workers = clinicaltrials_workers
        self.shared_source_ids = set(shared_source_ids or [])
        self.shared_cache_root = shared_cache_root or output_root / "source_cache"
        self.run_id = f"run_{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
        self.run_root = output_root / "compound_runs" / self.run_id
        self.status_path = self.run_root / "RUN_STATUS.json"
        self.acquisition_plan_path = self.run_root / "acquisition_plan.json"
        self.download_plan_path = self.run_root / "download_plan.json"
        self.requests_path = self.run_root / "requests.jsonl"
        self.failures_path = self.run_root / "failures.jsonl"
        self.deferred_path = self.run_root / "deferred.jsonl"
        self.assets_path = self.run_root / "assets.jsonl"
        self.coverage: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
        self.pmcids_by_compound: dict[str, set[str]] = defaultdict(set)
        self.pubmed_linked_pmcids_by_compound: dict[str, set[str]] = defaultdict(set)
        self.pmcid_to_pmids_by_compound: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.pubmed_pmc_link_lookup_succeeded_pmids_by_compound: dict[str, set[str]] = defaultdict(set)
        self.pmc_fulltext_selected_by_compound: dict[str, set[str]] = defaultdict(set)
        self.pmc_oa_selected_by_compound: dict[str, set[str]] = defaultdict(set)
        self.pmids_by_compound: dict[str, set[str]] = defaultdict(set)
        self.selected_pmids_by_compound: dict[str, set[str]] = defaultdict(set)
        self.pubmed_selection_scores_by_pmid: dict[str, dict[str, int]] = defaultdict(dict)
        self.pubmed_family_ids_by_pmid: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.literature_selection_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.dailymed_selection_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.clinicaltrial_selection_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.openfda_selection_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pkdb_selection_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pubmed_pmids_with_pmcid_by_compound: dict[str, set[str]] = defaultdict(set)
        self.identity_coverage_paths: dict[str, Path] = {}
        self.shared_assets_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.write_lock = threading.Lock()
        self.sources = load_source_registry(source_registry_path)
        shared_source_ids = {
            str(source.get("source_id"))
            for source in self.sources
            if SOURCE_MODE.get(str(source.get("source_id"))) in {"corpus_snapshot", "repository_mirror"}
        }
        if self.include_shared_sources and not self.shared_source_ids:
            raise ValueError("Shared-source refresh requires at least one explicit shared source ID.")
        unknown_shared_sources = self.shared_source_ids - shared_source_ids
        if unknown_shared_sources:
            raise ValueError(
                "Unknown or non-shared source IDs requested for shared refresh: " + ", ".join(sorted(unknown_shared_sources))
            )

    def _bind_run_root(self, run_root: Path) -> None:
        run_root = run_root.resolve()
        if not run_root.is_relative_to(self.output_root.resolve()):
            raise ValueError("Run must be inside the configured raw output root.")
        self.run_root = run_root
        self.run_id = run_root.name
        for attribute, filename in (
            ("status_path", "RUN_STATUS.json"), ("acquisition_plan_path", "acquisition_plan.json"),
            ("download_plan_path", "download_plan.json"), ("requests_path", "requests.jsonl"),
            ("failures_path", "failures.jsonl"), ("deferred_path", "deferred.jsonl"),
            ("assets_path", "assets.jsonl"),
        ):
            setattr(self, attribute, run_root / filename)

    def bind_new_run_root(self, run_root: Path) -> None:
        if run_root.exists() and (not run_root.is_dir() or any(run_root.iterdir())):
            raise ValueError("A new raw run directory must be empty; existing acquisition is immutable.")
        self._bind_run_root(run_root)

    def run(self) -> dict[str, Any]:
        self.run_root.mkdir(parents=True, exist_ok=True)
        for ledger in (self.requests_path, self.failures_path, self.deferred_path, self.assets_path):
            ledger.touch(exist_ok=True)
        self.write_status("running", "Formal raw acquisition run created; writing acquisition and download plans.")
        write_json(self.acquisition_plan_path, self.build_acquisition_plan())
        write_json(self.download_plan_path, self.build_download_plan())
        self.write_status("running", "download_plan.json written; network download execution has started.")
        try:
            for compound in self.compounds:
                self.retrieve_compound(compound)
            self.retrieve_pmc_fulltext()
            if self.include_shared_sources:
                self.retrieve_shared_sources()
            summary = self.write_summary()
            self.write_status("completed", "All configured raw acquisition stages finished.", summary=summary)
            return summary
        except KeyboardInterrupt:
            self.write_status("interrupted", "Run interrupted before completion; this run is not an accepted final result.")
            raise
        except Exception as exc:
            self.write_status("failed", "Run failed before completion; this run is not an accepted final result.", error=str(exc))
            raise

    def resume_after_discovery(self, run_root: Path) -> dict[str, Any]:
        """Restore a completed discovery phase from raw records, then finish PMC.

        Never bless an interrupted discovery run merely because its ledgers exist.
        Existing raw bytes and generated manifests remain immutable on resume.
        """
        if not run_root.resolve().is_relative_to(self.output_root.resolve()):
            raise ValueError("Resume run must be inside the configured raw output root.")
        for compound in self.compounds:
            gap = run_root / "compounds" / slugify(compound) / "pubmed/open_access_gap_manifest.json"
            if not gap.is_file():
                raise RuntimeError(f"Discovery is unfinished for {compound}; cannot resume at PMC.")
        self._bind_run_root(run_root)
        assets = [json.loads(line) for line in self.assets_path.read_text().splitlines() if line.strip()]
        self._resume_assets = {}
        for asset in assets:
            path = Path(asset["storage_path"]).resolve()
            if not path.is_relative_to(self.output_root.resolve()):
                raise ValueError("Resume asset is outside this acquisition root.")
            if not path.is_file() or file_checksum(path) != asset["checksum"]:
                raise RuntimeError("Resume asset checksum verification failed.")
            self._resume_assets[asset["request_id"]] = asset
            source = asset["source_id"]
            self.coverage[source]["assets"] += 1
            self.coverage[source]["bytes"] += int(asset.get("byte_count") or 0)
            compound = asset.get("compound")
            if compound in self.compounds and str(asset.get("group", "")).startswith("pubmed/elink_pmc"):
                pmcids, pmids, mapping = pmcids_and_pmids_from_elink(load_json_file(path))
                self.pmcids_by_compound[compound].update(pmcids)
                self.pubmed_linked_pmcids_by_compound[compound].update(pmcids)
                self.pubmed_pmids_with_pmcid_by_compound[compound].update(pmids)
                for pmcid, source_pmids in mapping.items():
                    self.pmcid_to_pmids_by_compound[compound][pmcid].update(source_pmids)
        for compound in self.compounds:
            base = run_root / "compounds" / slugify(compound)
            self.identity_coverage_paths[compound] = base / "identity/identity_coverage_manifest.json"
            for source, counts in (
                ("pubmed", self.literature_selection_counts), ("dailymed", self.dailymed_selection_counts),
                ("clinicaltrials", self.clinicaltrial_selection_counts),
                ("openfda_drug_label", self.openfda_selection_counts), ("pkdb", self.pkdb_selection_counts),
            ):
                selection = base / "selection" / source / "candidate_records.jsonl"
                if not selection.exists():
                    continue
                for line in selection.read_text().splitlines():
                    row = json.loads(line)
                    counts[compound][row["decision"]] += 1
                    if source == "pubmed":
                        pmid = str(row["source_record_id"])
                        self.pmids_by_compound[compound].add(pmid)
                        if row["decision"] == "fulltext_selected":
                            self.selected_pmids_by_compound[compound].add(pmid)
                            self.pubmed_selection_scores_by_pmid[compound][pmid] = int(row.get("selection_score") or 0)
        self.write_status("running", "Resuming PMC and shared-source retrieval from verified discovery records.")
        self.retrieve_pmc_fulltext()
        if self.include_shared_sources:
            self.retrieve_shared_sources()
        summary = self.write_summary()
        self.write_status("completed", "Resumed acquisition stages finished; finalization and reconciliation still required.", summary=summary)
        return summary

    def build_acquisition_plan(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": iso_now(),
            "official_run": self.official_run,
            "output_root": str(self.output_root),
            "run_root": str(self.run_root),
            "shared_cache_root": str(self.shared_cache_root),
            "compounds": self.compounds,
            "requirements": [
                "write download_plan.json before network downloads",
                "discover candidate records before requesting full-text or child resources",
                "use exact compound identity and Title/Abstract topic queries for literature",
                "write fulltext_selected, metadata_only, and excluded candidate manifests",
                "expand PubMed-to-PMC links only for selected PMIDs",
                "use field-limited ClinicalTrials candidate pages and download full study JSON only for selected NCT IDs",
                "use PK-DB studies endpoint with substance=<compound>, official open-licence bulk export, and study metadata child endpoints",
                "keep shared source snapshots outside compound runs unless explicitly enabled",
                "final result accepted only when RUN_STATUS.json status is completed",
            ],
            "raw_only": True,
        }

    def build_download_plan(self) -> dict[str, Any]:
        source_modes = {
            str(source.get("source_id")): SOURCE_MODE.get(str(source.get("source_id")), "registered")
            for source in self.sources
        }
        shared_jobs = []
        for source in self.sources:
            source_id = str(source.get("source_id"))
            mode = SOURCE_MODE.get(source_id)
            if mode not in {"corpus_snapshot", "repository_mirror"}:
                continue
            shared_jobs.append(
                {
                    "source_id": source_id,
                    "mode": mode,
                    "start_url": source.get("access_url") or source.get("api_url"),
                    "selected_for_refresh": source_id in self.shared_source_ids,
                    "rule": "not executed unless include_shared_sources is enabled and the source is explicitly selected",
                }
            )
        compound_jobs = [
            {
                "compound": compound,
                "stages": [
                    "compound_public_api_requests",
                    "dailymed_spl_listing_then_exact_single_ingredient_label_selection",
                    "clinicaltrials_field_limited_intervention_and_pk_topic_search_then_selected_nct_detail_download",
                    "pkdb_filter_counts_then_open_bulk_export_then_studies_by_substance_then_metadata_children",
                    "openfda_skip_pagination",
                    "pubmed_title_target_and_title_topic_query_then_metadata_abstract_selection",
                    "selected_pubmed_pmids_then_pmc_link_validation_then_oa_download",
                ],
                "pkdb": {
                    "filter_counts": {
                        "url": "https://pk-db.com/api/v1/filter/",
                        "params": {"studies__substance": compound, "concise": "true"},
                        "purpose": "count/index discovery only",
                    },
                    "study_listing": {
                        "url": "https://pk-db.com/api/v1/studies/",
                        "params": {"substance": compound},
                        "purpose": "authoritative paginated study list for this compound",
                    },
                    "bulk_export_rule": "download the official open-licence ZIP and require unique-object counts to match the same filter response before observation processing",
                    "child_endpoint_rule": "for each discovered study sid, request study detail plus group/intervention/individual metadata; output and timecourse observations come from the audited bulk export",
                },
            }
            for compound in self.compounds
        ]
        return {
            "run_id": self.run_id,
            "created_at": iso_now(),
            "official_run": self.official_run,
            "raw_only": True,
            "two_stage_execution": [
                "discover record/file lists, cursors, PMIDs, PMCIDs, set IDs, study IDs, release files, and repository roots",
                "write candidate metadata with fulltext_selected, metadata_only, or excluded decision",
                "expand to full text, PMC OA packages, SPL XML, and trial details only for fulltext_selected records",
            ],
            "output_root": str(self.output_root),
            "run_root": str(self.run_root),
            "shared_cache_root": str(self.shared_cache_root),
            "accepted_final_status": "completed",
            "manifest_files": {
                "status": str(self.status_path),
                "acquisition_plan": str(self.acquisition_plan_path),
                "download_plan": str(self.download_plan_path),
                "requests": str(self.requests_path),
                "assets": str(self.assets_path),
                "failures": str(self.failures_path),
                "deferred": str(self.deferred_path),
                "summary": str(self.run_root / "summary.json"),
            },
            "raw_file_policy": {
                "stored_formats": ["json", "xml", "html", "pdf", "zip", "tsv/csv", "xpt", "sdf", "git_repository", "binary"],
                "assets_jsonl_role": "index only; each row points to the real raw file and checksum",
                "overwrite_policy": "append-only run directory; identical bytes may share checksum but requests remain recorded",
            },
            "literature_query_policy": {
                "profile": "high_precision_v3",
                "pubmed_compound_clause": "exact target compound in Title",
                "pubmed_topic_clause": "selected Clinical PK, ADME, DDI, special-population, PBPK, or formulation/food-effect topic in Title; special-population and formulation queries also require strong PK support in Title/Abstract",
                "selection_rule": "full text requires target and selected topic in Title plus strong PK/ADME/DDI/PBPK support; ambiguous records remain metadata_only",
                "pmc_expansion": "only validated PubMed-to-PMC links from fulltext_selected PMIDs",
                "reason": "avoid broad full-text retrieval for common compounds and preserve a complete decision trail",
            },
            "clinicaltrials_query_policy": {
                "candidate_endpoint": "https://clinicaltrials.gov/api/v2/studies",
                "candidate_params": {
                    "query.intr": "target compound",
                    "query.term": clinicaltrial_candidate_query(),
                    "fields": CLINICALTRIAL_CANDIDATE_FIELDS,
                    "pageSize": 100,
                },
                "selection_rule": "only exact target interventions or probes with direct PK topic terms are eligible for complete NCT JSON download",
                "detail_endpoint": "https://clinicaltrials.gov/api/v2/studies/{NCTId}",
            },
            "compounds": self.compounds,
            "source_modes": source_modes,
            "compound_jobs": compound_jobs,
            "shared_source_jobs": shared_jobs,
            "limits": {
                "timeout_seconds": self.timeout_seconds,
                "delay_seconds": self.delay_seconds,
                "ncbi_batch_size": self.ncbi_batch_size,
                "page_size": self.page_size,
                "dailymed_workers": self.dailymed_workers,
                "clinicaltrials_workers": self.clinicaltrials_workers,
                "pkdb_workers": self.pkdb_workers,
                "download_oa_packages": self.download_oa_packages,
                "mirror_repositories": self.mirror_repositories,
                "include_shared_sources": self.include_shared_sources,
                "shared_source_ids": sorted(self.shared_source_ids),
                "include_combination_labels": self.include_combination_labels,
                "max_pmc_fulltext_per_compound": self.max_pmc_fulltext_per_compound,
                "max_pmc_oa_packages_per_compound": self.max_pmc_oa_packages_per_compound,
            },
        }

    def write_status(
        self,
        status: str,
        message: str,
        *,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "run_id": self.run_id,
            "status": status,
            "message": message,
            "official_run": self.official_run,
            "accepted_as_final_result": status == "completed" and self.official_run,
            "raw_only": True,
            "updated_at": iso_now(),
            "run_root": str(self.run_root),
            "download_plan_path": str(self.download_plan_path),
            "summary_path": str(self.run_root / "summary.json"),
        }
        if error:
            payload["error"] = error
        if summary:
            payload["summary"] = summary
        write_json(self.status_path, payload)

    def record_selection(
        self,
        *,
        source_id: str,
        compound: str,
        source_record_id: str,
        decision: dict[str, Any],
        record: dict[str, Any],
        source_url: str | None = None,
        source_location: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Persist a retrieval decision without turning it into scientific evidence."""

        selection_dir = self.run_root / "compounds" / slugify(compound) / "selection" / source_id
        row = {
            "compound": compound,
            "source_id": source_id,
            "source_record_id": source_record_id,
            "decision": decision.get("decision"),
            "selection_reason": decision.get("selection_reason"),
            "selection_score": decision.get("selection_score"),
            "query_family": decision.get("query_families") or [],
            "identity_match_type": decision.get("identity_match_type") or "unknown",
            "source_url": source_url,
            "source_location": source_location or source_record_id,
            "request_id": request_id,
            "selected_at": iso_now(),
            "raw_only": True,
            "record": record,
            "decision_metadata": {key: value for key, value in decision.items() if key not in {"decision", "selection_reason", "selection_score"}},
        }
        decision_name = str(decision.get("decision") or "unknown")
        with self.write_lock:
            append_jsonl(selection_dir / "candidate_records.jsonl", row)
            append_jsonl(selection_dir / f"{decision_name}_records.jsonl", row)
            if source_id == "pubmed":
                self.literature_selection_counts[compound][decision_name] += 1
            elif source_id == "dailymed":
                self.dailymed_selection_counts[compound][decision_name] += 1
            elif source_id == "clinicaltrials":
                self.clinicaltrial_selection_counts[compound][decision_name] += 1
            elif source_id == "openfda_drug_label":
                self.openfda_selection_counts[compound][decision_name] += 1
            elif source_id == "pkdb":
                self.pkdb_selection_counts[compound][decision_name] += 1

    def retrieve_compound(self, compound: str) -> None:
        self.retrieve_compound_public_apis(compound)
        self.retrieve_pubmed(compound)

    def retrieve_compound_public_apis(self, compound: str) -> None:
        encoded = quote(compound)
        requests = [
            ("pubchem", "compound_by_name", f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/JSON"),
            ("pubchem", "synonyms_by_name", f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/synonyms/JSON"),
            (
                "pubchem",
                "properties_by_name",
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,InChIKey,XLogP,TPSA/JSON",
            ),
            ("rxnorm", "rxcui_by_name", "https://rxnav.nlm.nih.gov/REST/rxcui.json", {"name": compound, "search": "2"}),
            ("rxnorm", "drugs_by_name", "https://rxnav.nlm.nih.gov/REST/drugs.json", {"name": compound}),
            ("chembl", "molecule_search", "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json", {"q": compound, "limit": 25, "offset": 0}),
            ("pkdb", "filter_by_substance", "https://pk-db.com/api/v1/filter/", {"studies__substance": compound, "concise": "true"}),
        ]
        identity_assets: list[dict[str, Any]] = []
        for source_id, request_key, url, *params in requests:
            asset = self.fetch_and_store(
                source_id=source_id,
                compound=compound,
                group="compound_api",
                request_key=request_key,
                url=url,
                params=params[0] if params else None,
            )
            if asset and source_id in {"pubchem", "rxnorm", "chembl"}:
                identity_assets.append(asset)
        self.write_identity_coverage_manifest(compound, identity_assets)
        self.retrieve_dailymed_spl(compound)
        self.retrieve_clinicaltrials_pages(compound)
        self.retrieve_pkdb_pages_and_children(compound)
        self.retrieve_openfda_pages(compound)

    def write_identity_coverage_manifest(self, compound: str, assets: list[dict[str, Any]]) -> None:
        """Record the exact raw identity lookups used by this compound run."""

        manifest_path = self.run_root / "compounds" / slugify(compound) / "identity" / "identity_coverage_manifest.json"
        payload = {
            "compound": compound,
            "query_name": compound,
            "status": "raw_identity_discovery_only",
            "note": "This manifest records source responses used for identity discovery. It is not a canonical identity merge or scientific evidence.",
            "required_lookup_sources": ["pubchem", "rxnorm", "chembl"],
            "assets": [
                {
                    "source_id": asset.get("source_id"),
                    "request_id": asset.get("request_id"),
                    "request_key": asset.get("request_key"),
                    "url": asset.get("url"),
                    "storage_path": asset.get("storage_path"),
                    "checksum": asset.get("checksum"),
                }
                for asset in assets
            ],
            "missing_lookup_sources": sorted({"pubchem", "rxnorm", "chembl"} - {str(asset.get("source_id")) for asset in assets}),
            "raw_only": True,
            "created_at": iso_now(),
        }
        write_json(manifest_path, payload)
        self.identity_coverage_paths[compound] = manifest_path
        asset = {
            "asset_id": f"asset:{uuid4().hex}",
            "request_id": self.request_id("identity_discovery", compound, "identity_coverage_manifest", str(manifest_path), None),
            "source_id": "identity_discovery",
            "compound": compound,
            "group": "identity/coverage",
            "url": None,
            "storage_path": str(manifest_path),
            "metadata_path": str(manifest_path),
            "checksum": file_checksum(manifest_path),
            "byte_count": manifest_path.stat().st_size,
            "content_type": "application/json",
            "status_code": None,
            "downloaded_at": iso_now(),
            "raw_only": True,
        }
        with self.write_lock:
            append_jsonl(self.assets_path, asset)
            self.coverage["identity_discovery"]["assets"] += 1
            self.coverage["identity_discovery"]["bytes"] += asset["byte_count"]

    def retrieve_dailymed_spl(self, compound: str) -> None:
        page = 1
        seen_setids: set[str] = set()
        selected_spl_rows: list[dict[str, Any]] = []
        while True:
            asset = self.fetch_and_store(
                source_id="dailymed",
                compound=compound,
                group="dailymed",
                request_key=f"spls_page_{page}",
                url="https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
                params={"drug_name": compound, "pagesize": 100, "page": page},
            )
            payload = load_json_file(asset["storage_path"]) if asset else {}
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            for row in rows:
                setid = row.get("setid")
                if not setid or setid in seen_setids:
                    continue
                seen_setids.add(setid)
                decision = classify_dailymed_listing(
                    compound=compound,
                    row=row,
                    include_combination_labels=self.include_combination_labels,
                )
                self.record_selection(
                    source_id="dailymed",
                    compound=compound,
                    source_record_id=str(setid),
                    decision=decision,
                    record={
                        "setid": setid,
                        "title": row.get("title"),
                        "spl_version": row.get("spl_version"),
                        "published_date": row.get("published_date"),
                    },
                    source_url=str(asset.get("url") or ""),
                    source_location=f"setid:{setid}",
                    request_id=str(asset.get("request_id") or ""),
                )
                if decision["decision"] == "fulltext_selected":
                    selected_spl_rows.append(row)
            if len(rows) < 100:
                break
            page += 1
        with ThreadPoolExecutor(max_workers=max(1, self.dailymed_workers)) as executor:
            futures = {executor.submit(self.retrieve_dailymed_spl_xml, compound, row): row.get("setid") for row in selected_spl_rows}
            for future in as_completed(futures):
                setid = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.log_failure(
                        self.request_id("dailymed", compound, f"spl_xml_unhandled_{setid}", f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml", None),
                        "dailymed",
                        compound,
                        f"spl_xml_unhandled_{setid}",
                        f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml",
                        f"Unhandled DailyMed SPL XML retrieval error: {exc}",
                    )

    def retrieve_dailymed_spl_xml(self, compound: str, row: dict[str, Any]) -> None:
        setid = row.get("setid")
        if not setid:
            return
        self.fetch_and_store(
            source_id="dailymed",
            compound=compound,
            group="dailymed/spl_xml",
            request_key=f"spl_xml_{setid}",
            url=f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml",
            expected_extension="xml",
            metadata={"setid": setid, "title": row.get("title"), "spl_version": row.get("spl_version")},
        )

    def retrieve_clinicaltrials_pages(self, compound: str) -> None:
        page_token: str | None = None
        page = 1
        seen_nct_ids: set[str] = set()
        selected_trials: list[tuple[str, str]] = []
        while True:
            params: dict[str, Any] = {
                "query.intr": compound,
                "query.term": clinicaltrial_candidate_query(),
                "fields": CLINICALTRIAL_CANDIDATE_FIELDS,
                "pageSize": 100,
                "format": "json",
            }
            if page_token:
                params["pageToken"] = page_token
            asset = None
            for attempt in range(3):
                asset = self.fetch_and_store(
                    source_id="clinicaltrials",
                    compound=compound,
                    group="clinicaltrials/candidate_pages",
                    request_key=f"studies_page_{page}" if attempt == 0 else f"studies_page_{page}_retry_{attempt}",
                    url="https://clinicaltrials.gov/api/v2/studies",
                    params=params,
                )
                if asset:
                    break
                time.sleep(2 + attempt)
            if not asset:
                self.log_failure(
                    self.request_id("clinicaltrials", compound, f"studies_page_{page}_pagination_incomplete", "https://clinicaltrials.gov/api/v2/studies", params),
                    "clinicaltrials",
                    compound,
                    f"studies_page_{page}_pagination_incomplete",
                    url_with_params("https://clinicaltrials.gov/api/v2/studies", params),
                    "ClinicalTrials.gov pagination stopped after repeated failures for this page token.",
                    params=params,
                )
                break
            payload = load_json_file(asset["storage_path"]) if asset else {}
            for study in payload.get("studies", []) if isinstance(payload, dict) else []:
                if not isinstance(study, dict):
                    continue
                protocol = study.get("protocolSection") or {}
                identification = protocol.get("identificationModule") if isinstance(protocol, dict) else {}
                nct_id = str((identification or {}).get("nctId") or "")
                if not nct_id or nct_id in seen_nct_ids:
                    continue
                seen_nct_ids.add(nct_id)
                decision = classify_clinical_trial(compound, study)
                self.record_selection(
                    source_id="clinicaltrials",
                    compound=compound,
                    source_record_id=nct_id,
                    decision=decision,
                    record={
                        "nct_id": nct_id,
                        "brief_title": (identification or {}).get("briefTitle"),
                        "official_title": (identification or {}).get("officialTitle"),
                    },
                    source_url=str(asset.get("url") or ""),
                    source_location=f"nct_id:{nct_id}",
                    request_id=str(asset.get("request_id") or ""),
                )
                if decision["decision"] == "fulltext_selected":
                    selected_trials.append((nct_id, str(asset.get("request_id") or "")))
            page_token = payload.get("nextPageToken") if isinstance(payload, dict) else None
            if not page_token:
                break
            page += 1
        with ThreadPoolExecutor(max_workers=max(1, self.clinicaltrials_workers)) as executor:
            futures = {
                executor.submit(self.retrieve_clinicaltrial_detail, compound, nct_id, source_request_id=source_request_id): nct_id
                for nct_id, source_request_id in selected_trials
            }
            for future in as_completed(futures):
                nct_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.log_failure(
                        self.request_id(
                            "clinicaltrials",
                            compound,
                            f"study_{nct_id}_unhandled",
                            f"https://clinicaltrials.gov/api/v2/studies/{quote(nct_id)}",
                            None,
                        ),
                        "clinicaltrials",
                        compound,
                        f"study_{nct_id}_unhandled",
                        f"https://clinicaltrials.gov/api/v2/studies/{quote(nct_id)}",
                        f"Unhandled ClinicalTrials.gov selected-study retrieval error: {exc}",
                    )

    def retrieve_clinicaltrial_detail(self, compound: str, nct_id: str, *, source_request_id: str) -> None:
        self.fetch_and_store(
            source_id="clinicaltrials",
            compound=compound,
            group="clinicaltrials/selected_study",
            request_key=f"study_{nct_id}",
            url=f"https://clinicaltrials.gov/api/v2/studies/{quote(nct_id)}",
            metadata={"nct_id": nct_id, "selection_request_id": source_request_id},
        )

    def retrieve_pkdb_pages_and_children(self, compound: str) -> None:
        self.retrieve_pkdb_bulk_export(compound)
        next_url: str | None = "https://pk-db.com/api/v1/studies/?" + urlencode({"substance": compound})
        page = 1
        seen_studies: set[str] = set()
        study_ids: list[str] = []
        while next_url:
            asset = None
            for attempt in range(3):
                asset = self.fetch_and_store(
                    source_id="pkdb",
                    compound=compound,
                    group="pkdb/studies",
                    request_key=f"studies_page_{page}" if attempt == 0 else f"studies_page_{page}_retry_{attempt}",
                    url=next_url,
                )
                if asset:
                    break
                time.sleep(2 + attempt)
            if not asset:
                self.log_failure(
                    self.request_id("pkdb", compound, f"studies_page_{page}_pagination_incomplete", next_url, None),
                    "pkdb",
                    compound,
                    f"studies_page_{page}_pagination_incomplete",
                    next_url,
                    "PK-DB study pagination stopped after repeated failures for this page URL.",
                )
                break
            payload = load_json_file(asset["storage_path"]) if asset else {}
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            studies = data.get("data", []) if isinstance(data, dict) else []
            for study in studies:
                sid = study.get("sid")
                if not sid or sid in seen_studies:
                    continue
                seen_studies.add(sid)
                study_ids.append(str(sid))
                self.record_selection(
                    source_id="pkdb",
                    compound=compound,
                    source_record_id=str(sid),
                    decision={
                        "decision": "fulltext_selected",
                        "selection_reason": "exact_pkdb_substance_query",
                        "selection_score": 1,
                        "identity_match_type": "exact_target",
                    },
                    record={
                        "study_sid": sid,
                        "name": study.get("name"),
                        "description": study.get("description"),
                    },
                    source_url=str(asset.get("url") or ""),
                    source_location=f"study_sid:{sid}",
                    request_id=str(asset.get("request_id") or ""),
                )
            next_url = payload.get("next_page_url") if isinstance(payload, dict) else None
            page += 1
        if not study_ids:
            return
        with ThreadPoolExecutor(max_workers=max(1, self.pkdb_workers)) as executor:
            futures = {executor.submit(self.retrieve_pkdb_child_endpoints, compound, sid): sid for sid in study_ids}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.log_failure(
                        self.request_id("pkdb", compound, f"children_unhandled_{sid}", f"https://pk-db.com/api/v1/studies/{sid}/", None),
                        "pkdb",
                        compound,
                        f"children_unhandled_{sid}",
                        f"https://pk-db.com/api/v1/studies/{sid}/",
                        f"Unhandled PK-DB child retrieval error: {exc}",
                    )

    def retrieve_pkdb_bulk_export(self, compound: str) -> None:
        filter_params = {
            "format": "json",
            "studies__substance": compound,
            "studies__licence__in": "open",
            "concise": "false",
        }
        filter_asset = self.fetch_and_store_with_retries(
            source_id="pkdb",
            compound=compound,
            group="pkdb/official_bulk_export",
            request_key="open_filter_counts",
            url="https://pk-db.com/api/v1/filter/",
            params=filter_params,
            expected_extension="json",
            retries=3,
        )
        if filter_asset is None:
            return
        filter_payload = load_json_file(filter_asset["storage_path"])
        if not isinstance(filter_payload, dict):
            self.log_failure(
                self.request_id("pkdb", compound, "open_filter_invalid", str(filter_asset.get("url") or ""), None),
                "pkdb",
                compound,
                "open_filter_invalid",
                str(filter_asset.get("url") or ""),
                "PK-DB open filter response is not a JSON object.",
            )
            return

        export_params = {**filter_params, "download": "true"}
        export_asset = self.fetch_and_store_with_retries(
            source_id="pkdb",
            compound=compound,
            group="pkdb/official_bulk_export",
            request_key="open_bulk_export",
            url="https://pk-db.com/api/v1/filter/",
            params=export_params,
            expected_extension="zip",
            metadata={
                "filter_request_id": filter_asset.get("request_id"),
                "filter_uuid": filter_payload.get("uuid"),
                "filter_counts": {
                    key: filter_payload.get(key)
                    for key in ("studies", "groups", "individuals", "interventions", "outputs", "timecourses", "scatters")
                },
                "licence_scope": "open",
            },
            retries=3,
        )
        if export_asset is None:
            return

        inspection_directory = Path(export_asset["storage_path"]).parent / "open_bulk_export_inspection"
        try:
            audit = inspect_pkdb_bulk_export(
                archive_path=Path(export_asset["storage_path"]),
                output_directory=inspection_directory,
                filter_payload=filter_payload,
                compound=compound,
                source_url=str(export_asset.get("url") or ""),
                response_headers={
                    "content-type": str(export_asset.get("content_type") or ""),
                    "status-code": str(export_asset.get("status_code") or ""),
                },
            )
        except PKDBBulkExportError as exc:
            self.log_failure(
                self.request_id("pkdb", compound, "open_bulk_export_invalid", str(export_asset.get("url") or ""), None),
                "pkdb",
                compound,
                "open_bulk_export_invalid",
                str(export_asset.get("url") or ""),
                str(exc),
            )
            return

        audit_path = inspection_directory / "export_audit.json"
        audit_asset = {
            "asset_id": f"asset:{uuid4().hex}",
            "request_id": self.request_id("pkdb", compound, "open_bulk_export_audit", str(export_asset.get("url") or ""), None),
            "source_id": "pkdb",
            "compound": compound,
            "group": "pkdb/official_bulk_export_audit",
            "request_key": "open_bulk_export_audit",
            "url": export_asset.get("url"),
            "storage_path": str(audit_path),
            "metadata_path": str(audit_path),
            "checksum": file_checksum(audit_path),
            "byte_count": audit_path.stat().st_size,
            "content_type": "application/json",
            "status_code": None,
            "downloaded_at": iso_now(),
            "metadata": {
                "export_asset_id": export_asset.get("asset_id"),
                "accepted_for_observation_processing": audit["accepted_for_observation_processing"],
                "gap_codes": audit["gap_codes"],
            },
            "storage_scope": "compound_run",
            "raw_only": True,
        }
        with self.write_lock:
            append_jsonl(self.assets_path, audit_asset)
            self.coverage["pkdb"]["assets"] += 1
            self.coverage["pkdb"]["bytes"] += audit_asset["byte_count"]
        if not audit["accepted_for_observation_processing"]:
            self.log_failure(
                self.request_id("pkdb", compound, "open_bulk_export_incomplete", str(export_asset.get("url") or ""), None),
                "pkdb",
                compound,
                "open_bulk_export_incomplete",
                str(export_asset.get("url") or ""),
                "PK-DB returned an archive, but unique object counts did not match the filter response: "
                + ", ".join(audit["gap_codes"]),
            )

    def retrieve_pkdb_child_endpoints(self, compound: str, sid: str) -> None:
        child_specs = [
            ("study_detail", f"https://pk-db.com/api/v1/studies/{sid}/"),
            ("interventions", "https://pk-db.com/api/v1/interventions/", {"search_multi_match": sid}),
            ("groups", "https://pk-db.com/api/v1/groups/", {"search_multi_match": sid}),
            ("individuals", "https://pk-db.com/api/v1/individuals/", {"search_multi_match": sid}),
        ]
        for request_key, url, *params in child_specs:
            self.fetch_and_store(
                source_id="pkdb",
                compound=compound,
                group=f"pkdb/children/{sid}",
                request_key=f"{request_key}_{sid}",
                url=url,
                params=params[0] if params else None,
                metadata={"study_sid": sid},
            )

    def retrieve_openfda_pages(self, compound: str) -> None:
        skip = 0
        total: int | None = None
        seen_label_ids: set[str] = set()
        while total is None or skip < total:
            asset = self.fetch_and_store(
                source_id="openfda_drug_label",
                compound=compound,
                group="openfda_drug_label",
                request_key=f"label_skip_{skip}",
                url="https://api.fda.gov/drug/label.json",
                params={"search": f'openfda.generic_name:"{compound}"', "limit": 100, "skip": skip},
            )
            payload = load_json_file(asset["storage_path"]) if asset else {}
            if isinstance(payload, dict):
                total = payload.get("meta", {}).get("results", {}).get("total")
                for index, label in enumerate(payload.get("results", []) or []):
                    if not isinstance(label, dict):
                        continue
                    label_id = str(label.get("id") or label.get("set_id") or f"skip_{skip}_row_{index}")
                    if label_id in seen_label_ids:
                        continue
                    seen_label_ids.add(label_id)
                    decision = classify_openfda_label(compound, label)
                    self.record_selection(
                        source_id="openfda_drug_label",
                        compound=compound,
                        source_record_id=label_id,
                        decision=decision,
                        record={
                            "label_id": label_id,
                            "set_id": label.get("set_id"),
                            "openfda": label.get("openfda") or {},
                            "effective_time": label.get("effective_time"),
                        },
                        source_url=str(asset.get("url") or ""),
                        source_location=f"label_id:{label_id}",
                        request_id=str(asset.get("request_id") or ""),
                    )
            if not total or skip + 100 >= total:
                break
            skip += 100

    def retrieve_pubmed(self, compound: str) -> None:
        all_pmids: set[str] = set()
        pmid_family_ids: dict[str, set[str]] = defaultdict(set)
        for family_id, _category, terms in HIGH_PRECISION_QUERY_FAMILIES:
            query = literature_query(compound, terms, family_id=family_id)
            pmids = self.ncbi_esearch_all(
                db="pubmed",
                term=query,
                source_id="pubmed",
                compound=compound,
                group=f"pubmed/esearch/{family_id}",
                request_key=f"esearch_{family_id}",
            )
            all_pmids.update(pmids)
            for pmid in pmids:
                pmid_family_ids[pmid].add(family_id)
        self.pmids_by_compound[compound].update(all_pmids)
        self.pubmed_family_ids_by_pmid[compound].update(pmid_family_ids)
        pmid_list = sorted(all_pmids, key=int)
        # Keep XML payloads bounded. A single failed PubMed batch must not
        # silently remove its candidate PMIDs from the decision manifests.
        for batch in batched(pmid_list, min(self.ncbi_batch_size, 100)):
            batch_key = f"{batch[0]}_{batch[-1]}_{len(batch)}"
            ids = ",".join(batch)
            self.fetch_and_store_with_retries(
                source_id="pubmed",
                compound=compound,
                group="pubmed/esummary",
                request_key=f"esummary_{batch_key}",
                url=f"{NCBI_BASE}/esummary.fcgi",
                params={"db": "pubmed", "id": ids, "retmode": "json"},
            )
            efetch_asset = self.fetch_and_store_with_retries(
                source_id="pubmed",
                compound=compound,
                group="pubmed/efetch_xml",
                request_key=f"efetch_{batch_key}",
                url=f"{NCBI_BASE}/efetch.fcgi",
                params={"db": "pubmed", "id": ids, "retmode": "xml"},
                expected_extension="xml",
            )
            if not efetch_asset:
                source_url = url_with_params(
                    f"{NCBI_BASE}/efetch.fcgi",
                    {"db": "pubmed", "id": ids, "retmode": "xml"},
                )
                for pmid in batch:
                    self.record_selection(
                        source_id="pubmed",
                        compound=compound,
                        source_record_id=pmid,
                        decision={
                            "decision": "metadata_only",
                            "selection_reason": "pubmed_metadata_fetch_failed_pending_retry",
                            "identity_match_type": "unknown",
                            "query_families": sorted(pmid_family_ids.get(pmid, set())),
                            "metadata_fetch_failed": True,
                        },
                        record={"pmid": pmid, "title": None, "abstract": None},
                        source_url=source_url,
                        source_location=f"pmid:{pmid}",
                        request_id=self.request_id("pubmed", compound, f"efetch_{batch_key}", f"{NCBI_BASE}/efetch.fcgi", {"db": "pubmed", "id": ids, "retmode": "xml"}),
                    )
                continue
            for record in pubmed_records_from_xml(Path(efetch_asset["storage_path"]).read_bytes()):
                pmid = record["pmid"]
                decision = classify_pubmed_record(
                    compound=compound,
                    title=record["title"],
                    abstract=record["abstract"],
                    query_families=pmid_family_ids.get(pmid, set()),
                )
                self.record_selection(
                    source_id="pubmed",
                    compound=compound,
                    source_record_id=pmid,
                    decision=decision,
                    record=record,
                    source_url=str(efetch_asset.get("url") or ""),
                    source_location=f"pmid:{pmid}",
                    request_id=str(efetch_asset.get("request_id") or ""),
                )
                if decision["decision"] == "fulltext_selected":
                    self.selected_pmids_by_compound[compound].add(pmid)
                    self.pubmed_selection_scores_by_pmid[compound][pmid] = int(decision.get("selection_score") or 0)
        self.retrieve_selected_pubmed_pmc_links(compound)
        self.write_pubmed_open_access_gap_manifest(compound)

    def retrieve_selected_pubmed_pmc_links(self, compound: str) -> None:
        selected_pmids = sorted(self.selected_pmids_by_compound[compound], key=int)
        # NCBI ELink can truncate larger multi-ID responses over this network.
        # Smaller batches plus explicit retries preserve a precise distinction
        # between a confirmed absence of a PMC link and a failed link lookup.
        def consume_asset(asset: dict[str, Any], batch: list[str]) -> bool:
            payload = load_json_file(asset["storage_path"])
            if not elink_payload_covers_pmids(payload, batch):
                self.log_failure(
                    str(asset.get("request_id") or ""),
                    "pubmed",
                    compound,
                    str(asset.get("request_key") or "elink_pmc_source_error"),
                    str(asset.get("url") or f"{NCBI_BASE}/elink.fcgi"),
                    "HTTP 200 source_error_payload: ELink JSON was invalid, contained ERROR, or omitted requested PMIDs",
                    params={"id": batch},
                )
                return False
            linked_pmcids, linked_pmids, pmcid_to_pmids = pmcids_and_pmids_from_elink(payload)
            self.pubmed_pmc_link_lookup_succeeded_pmids_by_compound[compound].update(batch)
            self.pmcids_by_compound[compound].update(linked_pmcids)
            self.pubmed_linked_pmcids_by_compound[compound].update(linked_pmcids)
            self.pubmed_pmids_with_pmcid_by_compound[compound].update(linked_pmids)
            for pmcid, pmids in pmcid_to_pmids.items():
                self.pmcid_to_pmids_by_compound[compound][pmcid].update(pmids)
            return True

        for batch in batched(selected_pmids, min(self.ncbi_batch_size, 20)):
            batch_key = f"{batch[0]}_{batch[-1]}_{len(batch)}"
            asset = self.fetch_and_store_with_retries(
                source_id="pubmed",
                compound=compound,
                group="pubmed/elink_pmc",
                request_key=f"elink_pmc_{batch_key}",
                url=f"{NCBI_BASE}/elink.fcgi",
                params={"dbfrom": "pubmed", "db": "pmc", "linkname": "pubmed_pmc", "id": batch, "retmode": "json"},
            )
            if asset and consume_asset(asset, batch):
                continue

            # A transport-successful ELink response can still contain a source
            # error or be truncated. Retry each PMID independently so a bad
            # batch cannot be mistaken for a confirmed absence of PMC links.
            for pmid in batch:
                single_asset = self.fetch_and_store_with_retries(
                    source_id="pubmed",
                    compound=compound,
                    group="pubmed/elink_pmc_single",
                    request_key=f"elink_pmc_single_{pmid}",
                    url=f"{NCBI_BASE}/elink.fcgi",
                    params={"dbfrom": "pubmed", "db": "pmc", "linkname": "pubmed_pmc", "id": [pmid], "retmode": "json"},
                )
                if single_asset:
                    consume_asset(single_asset, [pmid])

    def retrieve_pmc_fulltext(self) -> None:
        for compound in self.compounds:
            pmcids = self.pmcids_by_compound.get(compound, set())
            selected_pmcids = self.select_pmcids_for_fulltext(compound, pmcids)
            self.write_pmc_fulltext_gap_manifest(compound, pmcids, selected_pmcids)
            self.pmc_fulltext_selected_by_compound[compound].update(selected_pmcids)
            for batch in batched(sorted(selected_pmcids, key=pmcid_sort_key), min(self.ncbi_batch_size, 20)):
                ids = ",".join(batch)
                batch_key = f"{batch[0]}_{batch[-1]}_{len(batch)}"
                self.fetch_and_store_with_retries(
                    source_id="pmc_oa",
                    compound=compound,
                    group="pmc/efetch_xml",
                    request_key=f"pmc_efetch_{batch_key}",
                    url=f"{NCBI_BASE}/efetch.fcgi",
                    params={"db": "pmc", "id": ids, "retmode": "xml"},
                    expected_extension="xml",
                )
            if self.download_oa_packages:
                selected_oa_pmcids = self.select_pmcids_for_oa_packages(compound, selected_pmcids)
                self.pmc_oa_selected_by_compound[compound].update(selected_oa_pmcids)
                if len(selected_pmcids) > len(selected_oa_pmcids):
                    self.log_deferred(
                        self.request_id("pmc_oa", compound, "oa_package_threshold", "pmc_oa_package_selection", None),
                        "pmc_oa",
                        compound,
                        "pmc/oa_package_threshold",
                        "pmc_oa_package_selection",
                        (
                            f"OA package download was limited to {len(selected_oa_pmcids)} of "
                            f"{len(selected_pmcids)} selected PMCIDs by max_pmc_oa_packages_per_compound="
                            f"{self.max_pmc_oa_packages_per_compound}; skipped PMCIDs are listed in the gap manifest."
                        ),
                    )
                for pmcid in sorted(selected_oa_pmcids, key=pmcid_sort_key):
                    self.retrieve_pmc_oa_package(compound, pmcid)

    def select_pmcids_for_fulltext(self, compound: str, pmcids: set[str]) -> set[str]:
        if self.max_pmc_fulltext_per_compound <= 0:
            return set(pmcids)
        if len(pmcids) <= self.max_pmc_fulltext_per_compound:
            return set(pmcids)
        selected = set(self.ranked_pmcids(compound, pmcids)[: self.max_pmc_fulltext_per_compound])
        self.log_deferred(
            self.request_id("pmc_oa", compound, "pmc_fulltext_threshold", "pmc_fulltext_selection", None),
            "pmc_oa",
            compound,
            "pmc/fulltext_threshold",
            "pmc_fulltext_selection",
            (
                f"Discovered {len(pmcids)} PMCIDs, above max_pmc_fulltext_per_compound="
                f"{self.max_pmc_fulltext_per_compound}. Downloading the top {len(selected)} PMCIDs ranked by "
                "their selected PubMed metadata score and recording the remainder as not downloaded."
            ),
        )
        return selected

    def select_pmcids_for_oa_packages(self, compound: str, pmcids: set[str]) -> set[str]:
        if self.max_pmc_oa_packages_per_compound <= 0:
            return set(pmcids)
        if len(pmcids) <= self.max_pmc_oa_packages_per_compound:
            return set(pmcids)
        return set(self.ranked_pmcids(compound, pmcids)[: self.max_pmc_oa_packages_per_compound])

    def ranked_pmcids(self, compound: str, pmcids: set[str]) -> list[str]:
        pmid_scores = self.pubmed_selection_scores_by_pmid[compound]
        pmcid_to_pmids = self.pmcid_to_pmids_by_compound[compound]

        def rank_key(pmcid: str) -> tuple[int, int]:
            score = max((pmid_scores.get(pmid, 0) for pmid in pmcid_to_pmids.get(pmcid, set())), default=0)
            return (-score, pmcid_sort_key(pmcid))

        return sorted(pmcids, key=rank_key)

    def write_pmc_fulltext_gap_manifest(self, compound: str, discovered: set[str], selected: set[str]) -> None:
        pubmed_linked = self.pubmed_linked_pmcids_by_compound.get(compound, set())
        skipped = set(discovered) - set(selected)
        manifest_path = self.run_root / "compounds" / slugify(compound) / "pmc_oa" / "pmc_fulltext_gap_manifest.json"
        if hasattr(self, "_resume_assets") and manifest_path.is_file():
            previous = load_json_file(manifest_path)
            if set(previous.get("selected_for_fulltext_download") or []) != selected:
                raise RuntimeError("Resumed PMC selection differs from its immutable discovery manifest.")
            return
        write_json(
            manifest_path,
            {
                "compound": compound,
                "source_id": "pmc_oa",
                "discovered_pmcid_count": len(discovered),
                "selected_pubmed_linked_pmcid_count": len(pubmed_linked),
                "selected_for_fulltext_download_count": len(selected),
                "skipped_pmcid_count": len(skipped),
                "max_pmc_fulltext_per_compound": self.max_pmc_fulltext_per_compound,
                "max_pmc_oa_packages_per_compound": self.max_pmc_oa_packages_per_compound,
                "selected_for_fulltext_download": sorted(selected, key=pmcid_sort_key),
                "selected_pmcid_rank": [
                    {
                        "pmcid": pmcid,
                        "source_pmids": sorted(
                            self.pmcid_to_pmids_by_compound[compound].get(pmcid, set()),
                            key=lambda value: int(re.sub(r"\D", "", value) or "0"),
                        ),
                        "max_pubmed_selection_score": max(
                            (
                                self.pubmed_selection_scores_by_pmid[compound].get(pmid, 0)
                                for pmid in self.pmcid_to_pmids_by_compound[compound].get(pmcid, set())
                            ),
                            default=0,
                        ),
                    }
                    for pmcid in self.ranked_pmcids(compound, selected)
                ],
                "skipped_pmcids": sorted(skipped, key=pmcid_sort_key),
                "reason": (
                    "Only PMCIDs resolved from fulltext_selected PubMed records are eligible. The selected "
                    "PMCID queue exceeded the configured per-compound operational threshold; skipped IDs "
                    "remain retrieval metadata, not scientific evidence."
                )
                if skipped
                else "All discovered PMCIDs were selected for full-text download.",
                "raw_only": True,
                "created_at": iso_now(),
            },
        )
        asset = {
            "asset_id": f"asset:{uuid4().hex}",
            "request_id": self.request_id("pmc_oa", compound, "pmc_fulltext_gap_manifest", str(manifest_path), None),
            "source_id": "pmc_oa",
            "compound": compound,
            "group": "pmc/fulltext_gaps",
            "url": None,
            "storage_path": str(manifest_path),
            "metadata_path": str(manifest_path),
            "checksum": file_checksum(manifest_path),
            "byte_count": manifest_path.stat().st_size,
            "content_type": "application/json",
            "status_code": None,
            "downloaded_at": iso_now(),
            "raw_only": True,
        }
        append_jsonl(self.assets_path, asset)
        self.coverage["pmc_oa"]["assets"] += 1
        self.coverage["pmc_oa"]["bytes"] += asset["byte_count"]

    def retrieve_pmc_oa_package(self, compound: str, pmcid: str) -> None:
        # The legacy OA API was retired in August 2026. Discover official
        # version prefixes here; finalization plans all files and static transfer
        # downloads each object once, shared across compounds.
        self.fetch_and_store(
            source_id="pmc_oa", compound=compound,
            group=f"pmc/oa_s3_listing/{pmcid}",
            request_key=f"oa_s3_versions_{pmcid}",
            url="https://pmc-oa-opendata.s3.amazonaws.com/",
            params={"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"},
            expected_extension="xml",
            metadata={"pmcid": pmcid, "retrieval_method": "ncbi_pmc_aws_s3_list_versions"},
        )

    def retrieve_shared_sources(self) -> None:
        for source in self.sources:
            source_id = str(source.get("source_id"))
            mode = SOURCE_MODE.get(source_id)
            if mode not in {"corpus_snapshot", "repository_mirror"}:
                continue
            if self.shared_source_ids and source_id not in self.shared_source_ids:
                continue
            url = source.get("access_url") or source.get("api_url")
            if not url:
                continue
            snapshot_root = self.shared_snapshot_root(source_id)
            if mode == "repository_mirror" and source_id == "osp_pbpk_model_library":
                self.retrieve_repository(source_id, str(url), snapshot_root)
            else:
                self.retrieve_shared_source_pages(source_id, str(url), snapshot_root)
            self.write_shared_snapshot_manifest(source, mode, snapshot_root)

    def shared_snapshot_root(self, source_id: str) -> Path:
        return self.shared_cache_root / source_id / self.run_id

    def retrieve_shared_source_pages(self, source_id: str, start_url: str, snapshot_root: Path) -> None:
        queue: list[tuple[str, int]] = [(start_url, 0)]
        seen_pages: set[str] = set()
        seen_downloads: set[str] = set()
        page_limit = 120
        while queue and len(seen_pages) < page_limit:
            url, depth = queue.pop(0)
            if url in seen_pages:
                continue
            seen_pages.add(url)
            asset = self.fetch_and_store(
                source_id=source_id,
                compound=None,
                group=f"shared/{source_id}/pages",
                request_key=f"official_page_depth_{depth}_{len(seen_pages)}",
                url=url,
                expected_extension=extension_from_url(url) or "html",
                storage_root=snapshot_root,
            )
            if not asset:
                continue
            path = Path(asset["storage_path"])
            for link in downloadable_links(path, url):
                if link in seen_downloads:
                    continue
                seen_downloads.add(link)
                self.fetch_and_store(
                    source_id=source_id,
                    compound=None,
                    group=f"shared/{source_id}/downloads",
                    request_key=f"official_download_{len(seen_downloads)}",
                    url=link,
                    expected_extension=extension_from_url(link) or "bin",
                    storage_root=snapshot_root,
                )
            if depth < 2:
                for link in page_links(path, url):
                    if link not in seen_pages and same_site(start_url, link):
                        queue.append((link, depth + 1))
        if queue:
            self.log_failure(
                self.request_id(source_id, None, "shared_crawl_page_limit", start_url, None),
                source_id,
                None,
                "shared_crawl_page_limit",
                start_url,
                f"Shared source crawl stopped at {page_limit} HTML pages; remaining links were not fetched in this run.",
            )

    def write_pubmed_open_access_gap_manifest(self, compound: str) -> None:
        selected_pmids = self.selected_pmids_by_compound[compound]
        lookup_succeeded_pmids = self.pubmed_pmc_link_lookup_succeeded_pmids_by_compound[compound]
        confirmed_missing = sorted(
            lookup_succeeded_pmids - self.pubmed_pmids_with_pmcid_by_compound[compound],
            key=lambda value: int(re.sub(r"\D", "", value) or "0"),
        )
        lookup_failed = sorted(
            selected_pmids - lookup_succeeded_pmids,
            key=lambda value: int(re.sub(r"\D", "", value) or "0"),
        )
        manifest_path = self.run_root / "compounds" / slugify(compound) / "pubmed" / "open_access_gap_manifest.json"
        write_json(
            manifest_path,
            {
                "compound": compound,
                "source_id": "pubmed",
                "candidate_pmid_count": len(self.pmids_by_compound[compound]),
                "fulltext_selected_pmid_count": len(selected_pmids),
                "selection_counts": dict(self.literature_selection_counts[compound]),
                "selected_pmids_with_pmc_open_fulltext_link": len(self.pubmed_pmids_with_pmcid_by_compound[compound]),
                "pmc_link_lookup_succeeded_pmid_count": len(lookup_succeeded_pmids),
                "selected_pmids_confirmed_without_pmc_open_fulltext_link": len(confirmed_missing),
                "selected_pmids_confirmed_without_pmc_open_fulltext_link_list": confirmed_missing,
                "selected_pmids_with_pmc_link_lookup_failure": len(lookup_failed),
                "selected_pmids_with_pmc_link_lookup_failure_list": lookup_failed,
                "note": "Only fulltext_selected PubMed records are expanded to PMC. A PMID is listed as confirmed_without_pmc_link only after a successful ELink batch; failed ELink batches are recorded separately for retry.",
                "raw_only": True,
                "created_at": iso_now(),
            },
        )
        asset = {
            "asset_id": f"asset:{uuid4().hex}",
            "request_id": self.request_id("pubmed", compound, "open_access_gap_manifest", str(manifest_path), None),
            "source_id": "pubmed",
            "compound": compound,
            "group": "pubmed/open_access_gaps",
            "url": None,
            "storage_path": str(manifest_path),
            "metadata_path": str(manifest_path),
            "checksum": file_checksum(manifest_path),
            "byte_count": manifest_path.stat().st_size,
            "content_type": "application/json",
            "status_code": None,
            "downloaded_at": iso_now(),
            "raw_only": True,
        }
        append_jsonl(self.assets_path, asset)
        self.coverage["pubmed"]["assets"] += 1
        self.coverage["pubmed"]["bytes"] += asset["byte_count"]

    def retrieve_repository(self, source_id: str, url: str, snapshot_root: Path) -> None:
        repo_dir = snapshot_root / "repository"
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        request_id = self.request_id(source_id, None, "repository_mirror", url, None)
        self.log_request(request_id, source_id, None, "repository_mirror", url, None, "started")
        if repo_dir.exists() and (repo_dir / ".git").exists():
            self.log_request(request_id, source_id, None, "repository_mirror", url, None, "skipped_existing")
        elif self.mirror_repositories:
            try:
                # A full clone can block an acquisition run for an hour when a
                # public repository is rate-limited or unreachable. The API
                # fallback below retains the authoritative tree and a precise
                # download plan, so use a bounded shallow attempt first.
                subprocess.run(
                    ["git", "clone", "--depth", "1", url, str(repo_dir)],
                    check=True,
                    cwd=str(snapshot_root),
                    timeout=180,
                )
                if (repo_dir / ".git").exists():
                    try:
                        subprocess.run(["git", "lfs", "pull"], check=False, cwd=str(repo_dir), timeout=1800)
                    except Exception:
                        pass
            except Exception as exc:
                self.log_failure(request_id, source_id, None, "repository_mirror", url, str(exc))
                if self.retrieve_github_repository_fallback(source_id, url, snapshot_root):
                    return
                return
        else:
            self.log_failure(
                request_id,
                source_id,
                None,
                "repository_mirror",
                url,
                "Repository mirroring is disabled by --skip-repository-mirror; no source snapshot was downloaded.",
            )
            return
        inventory = []
        if repo_dir.exists():
            for path in repo_dir.rglob("*"):
                if path.is_file() and ".git" not in path.parts:
                    inventory.append(
                        {
                            "path": str(path.relative_to(repo_dir)),
                            "size": path.stat().st_size,
                            "checksum": file_checksum(path),
                        }
                    )
        inventory_path = repo_dir.parent / "repository_inventory.json"
        write_json(inventory_path, {"source_id": source_id, "repository_url": url, "file_count": len(inventory), "files": inventory})
        asset = {
            "asset_id": f"asset:{uuid4().hex}",
            "request_id": request_id,
            "source_id": source_id,
            "compound": None,
            "group": "repository_mirror",
            "url": url,
            "storage_path": str(repo_dir),
            "metadata_path": str(inventory_path),
            "checksum": file_checksum(inventory_path),
            "byte_count": sum(item["size"] for item in inventory),
            "content_type": "git/repository",
            "status_code": None,
            "downloaded_at": iso_now(),
            "storage_scope": "shared_source_cache",
            "snapshot_id": self.run_id,
            "raw_only": True,
        }
        with self.write_lock:
            append_jsonl(self.assets_path, asset)
            self.coverage[source_id]["assets"] += 1
            self.coverage[source_id]["bytes"] += asset["byte_count"]
            self.shared_assets_by_source[source_id].append(asset)
        self.log_request(request_id, source_id, None, "repository_mirror", url, None, "fetched")

    def retrieve_github_repository_fallback(self, source_id: str, url: str, snapshot_root: Path) -> bool:
        """Create an API-backed source snapshot when a public Git clone fails.

        The fallback records the authoritative repository metadata and recursive
        tree, downloads files whose paths exactly match this run's compounds,
        and writes a plan for remaining repository objects. It is intentionally
        source-cache-only and does not create evidence records.
        """

        parsed = urlparse(url)
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.netloc.lower() != "github.com" or len(parts) < 2:
            return False
        owner, repository = parts[0], parts[1].removesuffix(".git")
        api_base = f"https://api.github.com/repos/{quote(owner)}/{quote(repository)}"
        group = f"shared/{source_id}/github_api"
        metadata_asset = self.fetch_and_store(
            source_id=source_id,
            compound=None,
            group=group,
            request_key="github_repository_metadata",
            url=api_base,
            expected_extension="json",
            metadata={"fallback": "github_api", "repository_url": url},
            storage_root=snapshot_root,
        )
        if metadata_asset is None:
            return False
        repository_metadata = load_json_file(metadata_asset["storage_path"])
        default_branch = str(repository_metadata.get("default_branch") or "main")
        tree_asset = self.fetch_and_store(
            source_id=source_id,
            compound=None,
            group=group,
            request_key=f"github_recursive_tree_{slugify(default_branch)}",
            url=f"{api_base}/git/trees/{quote(default_branch)}",
            params={"recursive": "1"},
            expected_extension="json",
            metadata={"fallback": "github_api", "default_branch": default_branch},
            storage_root=snapshot_root,
        )
        if tree_asset is None:
            return False
        tree_payload = load_json_file(tree_asset["storage_path"])
        tree_entries = [item for item in tree_payload.get("tree", []) if isinstance(item, dict)]
        extensions = (".pkml", ".pksim5", ".pksim", ".pdf", ".md", ".xlsx", ".xls", ".csv", ".json")
        compound_terms = {slugify(compound) for compound in self.compounds}
        selected_entries = [
            item
            for item in tree_entries
            if item.get("type") == "blob"
            and str(item.get("path") or "").lower().endswith(extensions)
            and any(term in slugify(str(item.get("path") or "")) for term in compound_terms)
        ]
        # OSP currently publishes executable model projects as .pksim5 files.
        # They are source data, not derived media, and several are larger than
        # the generic document threshold; retain exact compound matches up to
        # 100 MB in the source-cache snapshot.
        max_download_bytes = 100_000_000
        selected_paths = {str(item.get("path") or "") for item in selected_entries}
        download_plan: list[dict[str, Any]] = []
        complete_object_plan: list[dict[str, Any]] = []
        for item in tree_entries:
            if item.get("type") != "blob":
                continue
            path = str(item.get("path") or "")
            size = int(item.get("size") or 0)
            complete_object_plan.append(
                {
                    "path": path,
                    "size": size,
                    "raw_url": f"https://raw.githubusercontent.com/{quote(owner)}/{quote(repository)}/{quote(default_branch)}/{quote(path, safe='/')}",
                    "sha": item.get("sha"),
                    "selected_for_compound_retrieval": path in selected_paths,
                }
            )
        immediate_downloads: list[dict[str, Any]] = []
        for index, item in enumerate(selected_entries, start=1):
            path = str(item.get("path") or "")
            size = int(item.get("size") or 0)
            raw_url = f"https://raw.githubusercontent.com/{quote(owner)}/{quote(repository)}/{quote(default_branch)}/{quote(path, safe='/')}"
            plan_item = {"path": path, "size": size, "raw_url": raw_url, "sha": item.get("sha")}
            download_plan.append(plan_item)
            if size > max_download_bytes:
                continue
            immediate_downloads.append(
                {
                    "source_id": source_id,
                    "compound": None,
                    "group": f"shared/{source_id}/github_selected_files",
                    "request_key": f"github_selected_{index}_{slugify(path)}",
                    "url": raw_url,
                    "expected_extension": extension_from_url(path) or "bin",
                    "metadata": {
                        "fallback": "github_api",
                        "repository_path": path,
                        "repository_blob_sha": item.get("sha"),
                        "size": size,
                    },
                    "storage_root": snapshot_root,
                }
            )
        # Repository files are independent. A small bounded pool avoids one
        # large public model project serially delaying every other exact match.
        with ThreadPoolExecutor(max_workers=min(3, max(1, len(immediate_downloads)))) as executor:
            futures = [executor.submit(self.fetch_and_store, **job) for job in immediate_downloads]
            for future in as_completed(futures):
                future.result()
        inventory_path = snapshot_root / "repository_inventory.json"
        write_json(
            inventory_path,
            {
                "source_id": source_id,
                "repository_url": url,
                "fallback": "github_api",
                "default_branch": default_branch,
                "repository_id": repository_metadata.get("id"),
                "repository_updated_at": repository_metadata.get("updated_at"),
                "tree_truncated": bool(tree_payload.get("truncated")),
                "tree_file_count": len(tree_entries),
                "target_compound_file_count": len(selected_entries),
                "target_compound_files": download_plan,
            },
        )
        self.record_shared_generated_asset(source_id, url, snapshot_root, inventory_path, "repository_inventory")
        plan_path = snapshot_root / "repository_download_plan.json"
        write_json(
            plan_path,
            {
                "source_id": source_id,
                "repository_url": url,
                "default_branch": default_branch,
                "max_immediate_download_bytes": max_download_bytes,
                "tree_truncated": bool(tree_payload.get("truncated")),
                "complete_repository_blob_count": len(complete_object_plan),
                "target_compound_file_count": len(download_plan),
                "entries": complete_object_plan,
            },
        )
        self.record_shared_generated_asset(source_id, url, snapshot_root, plan_path, "repository_download_plan")
        return True

    def record_shared_generated_asset(
        self,
        source_id: str,
        source_url: str,
        snapshot_root: Path,
        path: Path,
        request_key: str,
    ) -> None:
        asset = {
            "asset_id": f"asset:{uuid4().hex}",
            "request_id": self.request_id(source_id, None, request_key, str(path), None),
            "source_id": source_id,
            "compound": None,
            "group": f"shared/{source_id}/{request_key}",
            "url": source_url,
            "storage_path": str(path),
            "metadata_path": str(path),
            "checksum": file_checksum(path),
            "byte_count": path.stat().st_size,
            "content_type": "application/json",
            "status_code": None,
            "downloaded_at": iso_now(),
            "storage_scope": "shared_source_cache",
            "snapshot_id": self.run_id,
            "raw_only": True,
        }
        with self.write_lock:
            append_jsonl(self.assets_path, asset)
            self.coverage[source_id]["assets"] += 1
            self.coverage[source_id]["bytes"] += asset["byte_count"]
            self.shared_assets_by_source[source_id].append(asset)

    def write_shared_snapshot_manifest(
        self,
        source: dict[str, str | None],
        mode: str,
        snapshot_root: Path,
    ) -> None:
        source_id = str(source.get("source_id"))
        assets = list(self.shared_assets_by_source[source_id])
        checksums = sorted(str(asset.get("checksum") or "") for asset in assets)
        aggregate_checksum = sha256("\n".join(checksums).encode("utf-8")).hexdigest()
        manifest_path = snapshot_root / "snapshot_manifest.json"
        payload = {
            "source_id": source_id,
            "source_name": source.get("source_name"),
            "source_mode": mode,
            "snapshot_id": self.run_id,
            "source_url": source.get("access_url") or source.get("api_url"),
            "source_version": f"snapshot:{self.run_id}",
            "asset_count": len(assets),
            "asset_checksum_aggregate": aggregate_checksum,
            "assets": [
                {
                    "asset_id": asset.get("asset_id"),
                    "request_id": asset.get("request_id"),
                    "group": asset.get("group"),
                    "storage_path": asset.get("storage_path"),
                    "checksum": asset.get("checksum"),
                    "byte_count": asset.get("byte_count"),
                }
                for asset in assets
            ],
            "raw_only": True,
            "created_at": iso_now(),
        }
        write_json(manifest_path, payload)
        manifest_asset = {
            "asset_id": f"asset:{uuid4().hex}",
            "request_id": self.request_id(source_id, None, "shared_snapshot_manifest", str(manifest_path), None),
            "source_id": source_id,
            "compound": None,
            "group": "shared/snapshot_manifest",
            "url": source.get("access_url") or source.get("api_url"),
            "storage_path": str(manifest_path),
            "metadata_path": str(manifest_path),
            "checksum": file_checksum(manifest_path),
            "byte_count": manifest_path.stat().st_size,
            "content_type": "application/json",
            "status_code": None,
            "downloaded_at": iso_now(),
            "metadata": {"snapshot_id": self.run_id, "source_version": f"snapshot:{self.run_id}"},
            "storage_scope": "shared_source_cache",
            "snapshot_id": self.run_id,
            "raw_only": True,
        }
        with self.write_lock:
            append_jsonl(self.assets_path, manifest_asset)
            self.coverage[source_id]["assets"] += 1
            self.coverage[source_id]["bytes"] += manifest_asset["byte_count"]

    def ncbi_esearch_all(
        self,
        *,
        db: str,
        term: str,
        source_id: str,
        compound: str,
        group: str,
        request_key: str,
    ) -> list[str]:
        retstart = 0
        ids: list[str] = []
        count: int | None = None
        while count is None or retstart < count:
            asset = self.fetch_and_store(
                source_id=source_id,
                compound=compound,
                group=group,
                request_key=f"{request_key}_retstart_{retstart}",
                url=f"{NCBI_BASE}/esearch.fcgi",
                params={"db": db, "term": term, "retmode": "json", "retmax": self.page_size, "retstart": retstart},
            )
            if not asset:
                break
            payload = load_json_file(asset["storage_path"])
            result = payload.get("esearchresult", {}) if isinstance(payload, dict) else {}
            if count is None:
                count = safe_int(result.get("count")) or 0
            page_ids = [str(item) for item in result.get("idlist", [])]
            ids.extend(page_ids)
            if not page_ids:
                break
            retstart += len(page_ids)
        return sorted(set(ids), key=lambda value: int(re.sub(r"\D", "", value) or "0"))

    def fetch_and_store(
        self,
        *,
        source_id: str,
        compound: str | None,
        group: str,
        request_key: str,
        url: str,
        params: dict[str, Any] | None = None,
        expected_extension: str | None = None,
        metadata: dict[str, Any] | None = None,
        storage_root: Path | None = None,
    ) -> dict[str, Any] | None:
        request_id = self.request_id(source_id, compound, request_key, url, params)
        validate_spl = source_id == "dailymed" and group == "dailymed/spl_xml"
        spl_setid = str((metadata or {}).get("setid") or "")
        cached = getattr(self, "_resume_assets", {}).get(request_id)
        if cached is not None:
            if validate_spl:
                try:
                    self.validate_dailymed_spl(Path(cached["storage_path"]).read_bytes(), url, spl_setid)
                except (OSError, ValueError) as exc:
                    self.log_failure(request_id, source_id, compound, request_key, url,
                                     f"Cached DailyMed SPL failed validation: {exc}", params=params)
                    return None
            return cached
        final_url = url_with_params(url, params)
        self.log_request(request_id, source_id, compound, request_key, final_url, params, "started")
        try:
            if validate_spl:
                try:
                    payload, content_type, status_code, warning = self.fetch_validated_dailymed_spl(final_url, spl_setid)
                except DailyMedSPLContentError:
                    version = (metadata or {}).get("spl_version")
                    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
                        raise
                    asset = self.fetch_dailymed_archive_asset(
                        compound=compound, metadata=metadata or {}, original_url=final_url,
                        original_request_id=request_id,
                    )
                    self.log_request(request_id, source_id, compound, request_key, final_url, params,
                                     "recovered_by_verified_archive", asset=asset)
                    return asset
            else:
                payload, content_type, status_code, warning = self.fetch_bytes(final_url)
            checksum = sha256(payload).hexdigest()
            extension = clean_extension(expected_extension or extension_from_content_type(content_type) or extension_from_url(final_url) or "bin")
            folder = storage_root / group if storage_root else self.run_root / (f"compounds/{slugify(compound)}" if compound else "shared") / group / source_id
            folder.mkdir(parents=True, exist_ok=True)
            filename = f"{slugify(request_key)}_{checksum[:12]}.{extension}"
            path = folder / filename
            if not path.exists():
                path.write_bytes(payload)
            metadata_path = path.with_suffix(path.suffix + ".metadata.json")
            asset = {
                "asset_id": f"asset:{uuid4().hex}",
                "request_id": request_id,
                "source_id": source_id,
                "compound": compound,
                "group": group,
                "request_key": request_key,
                "url": final_url,
                "storage_path": str(path),
                "metadata_path": str(metadata_path),
                "checksum": checksum,
                "byte_count": len(payload),
                "content_type": content_type,
                "status_code": status_code,
                "downloaded_at": iso_now(),
                "transport_warning": warning,
                "metadata": metadata or {},
                "storage_scope": "shared_source_cache" if storage_root else "compound_run",
                "raw_only": True,
            }
            if storage_root:
                asset["snapshot_id"] = self.run_id
            write_json(metadata_path, asset)
            with self.write_lock:
                append_jsonl(self.assets_path, asset)
                self.coverage[source_id]["assets"] += 1
                self.coverage[source_id]["bytes"] += len(payload)
                if storage_root:
                    self.shared_assets_by_source[source_id].append(asset)
            self.log_request(request_id, source_id, compound, request_key, final_url, params, "fetched", asset=asset)
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            return asset
        except Exception as exc:
            self.log_failure(request_id, source_id, compound, request_key, final_url, str(exc), params=params)
            return None

    @staticmethod
    def validate_dailymed_spl(payload: bytes, url: str, expected_setid: str) -> None:
        """Require the requested SPL document before accepting transport success."""
        parsed_url = urlparse(url)
        expected_path = f"/dailymed/services/v2/spls/{expected_setid}.xml"
        if (not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", expected_setid)
                or parsed_url.scheme != "https" or parsed_url.netloc != "dailymed.nlm.nih.gov"
                or parsed_url.path.lower() != expected_path.lower() or parsed_url.query or parsed_url.fragment):
            raise ValueError("DailyMed SPL request identity is invalid or mismatched")
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise ValueError(f"DailyMed SPL response is not valid XML: {exc}") from exc
        if root.tag != "{urn:hl7-org:v3}document":
            raise ValueError("DailyMed SPL response is not an HL7 v3 document")
        identifiers = root.findall("{urn:hl7-org:v3}setId")
        if len(identifiers) != 1 or identifiers[0].get("root", "").lower() != expected_setid.lower():
            raise ValueError("DailyMed SPL document setId does not match the requested label")

    def fetch_validated_dailymed_spl(self, url: str, expected_setid: str) -> tuple[bytes, str | None, int, str | None]:
        # DailyMed has returned HTTP 200 with the four bytes b'None'.  Retry
        # content failures finitely; never register those bytes as a valid SPL.
        for attempt in range(3):
            result = self.fetch_bytes(url)
            payload, _, status_code, _ = result
            try:
                self.validate_dailymed_spl(payload, url, expected_setid)
            except ValueError as exc:
                delay = float(attempt + 1) if attempt < 2 else 0.0
                with self.write_lock:
                    append_jsonl(self.run_root / "transport_attempts.jsonl", {
                        "url": url, "status_code": status_code, "attempt": attempt + 1,
                        "max_attempts": 3, "retry_delay_seconds": delay,
                        "status": "retry_pending" if attempt < 2 else "retry_exhausted",
                        "failure_type": "dailymed_spl_content_invalid", "error": str(exc),
                        "byte_count": len(payload), "checksum": sha256(payload).hexdigest(),
                        "timestamp": iso_now(), "raw_only": True,
                    })
                if attempt == 2:
                    raise DailyMedSPLContentError(str(exc)) from exc
                time.sleep(delay)
            else:
                return result
        raise RuntimeError("DailyMed SPL validation attempts exhausted")

    def fetch_dailymed_archive_bytes(self, url: str) -> tuple[bytes, str | None, int, str | None]:
        """One bounded official archive request; never disable TLS validation."""
        from pbpk_kb.acquisition.dailymed_recovery import MAX_ARCHIVE_BYTES

        self.run_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".dailymed-archive-", dir=self.run_root) as temporary:
            target = Path(temporary) / "response.zip"
            result = subprocess.run([
                "curl", "--disable", "--fail", "--silent", "--show-error", "--retry", "0",
                "--proto", "=https", "--proto-redir", "=https", "--max-redirs", "0",
                "--max-time", str(self.timeout_seconds), "--max-filesize", str(MAX_ARCHIVE_BYTES),
                "-A", "pbpk-public-raw-retrieval/0.2 (public research data acquisition)",
                "--output", str(target), "--write-out", "%{http_code}\n%{url_effective}\n%{content_type}\n", url,
            ], capture_output=True, check=True, timeout=self.timeout_seconds + 15)
            response = result.stdout.decode("utf-8", errors="strict").splitlines()
            if len(response) < 2 or response[0] != "200" or response[1] != url:
                raise ValueError("DailyMed archive did not return HTTP 200 from the exact official URL")
            if not target.is_file() or not 0 < target.stat().st_size <= MAX_ARCHIVE_BYTES:
                raise ValueError("DailyMed archive exceeds the bounded download limit or is empty")
            return target.read_bytes(), response[2] if len(response) > 2 else None, 200, None

    def fetch_dailymed_archive_asset(
        self, *, compound: str | None, metadata: dict[str, Any], original_url: str,
        original_request_id: str,
    ) -> dict[str, Any]:
        # The official API documents getFile.cfm with an exact SPL version:
        # https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
        # Import locally: the offline recovery module also uses iso_now here.
        from pbpk_kb.acquisition.dailymed_recovery import (
            MAX_ARCHIVE_BYTES, _archive_url, _validated_archive,
        )

        setid, version = str(metadata["setid"]), metadata["spl_version"]
        if (not isinstance(version, int) or isinstance(version, bool) or version < 1
                or not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", setid)
                or original_url != f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"):
            raise ValueError("DailyMed archive fallback request identity is invalid")
        url = f"https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?type=zip&setid={setid}&version={version}"
        _archive_url(url, setid, version)
        key = f"spl_archive_{setid}_v{version}"
        request_id = self.request_id("dailymed", compound, key, url, None)
        self.log_request(request_id, "dailymed", compound, key, url, None, "started")
        try:
            payload, content_type, status_code, warning = self.fetch_dailymed_archive_bytes(url)
            if status_code != 200 or not 0 < len(payload) <= MAX_ARCHIVE_BYTES:
                raise ValueError("DailyMed archive response status or byte count is invalid")
            member = _validated_archive(BytesIO(payload), setid, version)
            checksum = sha256(payload).hexdigest()
            folder = self.run_root / (f"compounds/{slugify(compound)}" if compound else "shared") / "dailymed/spl_archive/dailymed"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{slugify(key)}_{checksum[:12]}.zip"
            if not path.exists():
                path.write_bytes(payload)
            metadata_path = path.with_suffix(".zip.metadata.json")
            asset = {
                "asset_id": f"asset:{uuid4().hex}", "request_id": request_id,
                "source_id": "dailymed", "compound": compound, "group": "dailymed/spl_archive",
                "request_key": key, "url": url, "storage_path": str(path),
                "metadata_path": str(metadata_path), "checksum": checksum, "byte_count": len(payload),
                "content_type": content_type, "status_code": status_code, "downloaded_at": iso_now(),
                "transport_warning": warning, "storage_scope": "compound_run", "raw_only": True,
                "metadata": {**metadata, "original_xml_url": original_url,
                             "original_xml_request_id": original_request_id,
                             "recovery_method": "verified_official_exact_version_archive", **member},
            }
            write_json(metadata_path, asset)
            with self.write_lock:
                append_jsonl(self.assets_path, asset)
                self.coverage["dailymed"]["assets"] += 1
                self.coverage["dailymed"]["bytes"] += len(payload)
            self.log_request(request_id, "dailymed", compound, key, url, None, "fetched", asset=asset)
            return asset
        except Exception as exc:
            self.log_failure(request_id, "dailymed", compound, key, url, str(exc))
            raise

    def fetch_and_store_with_retries(
        self,
        *,
        retries: int = 3,
        retry_delay_seconds: float = 2.0,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Retry a raw request while preserving every attempt in the request log."""

        request_key = str(kwargs["request_key"])
        for attempt in range(max(1, retries)):
            attempt_kwargs = dict(kwargs)
            if attempt:
                attempt_kwargs["request_key"] = f"{request_key}_retry_{attempt}"
            asset = self.fetch_and_store(**attempt_kwargs)
            if asset:
                return asset
            if attempt < retries - 1 and retry_delay_seconds:
                time.sleep(retry_delay_seconds + attempt)
        return None

    def fetch_bytes(self, url: str) -> tuple[bytes, str | None, int, str | None]:
        user_agent = "pbpk-public-raw-retrieval/0.2 (public research data acquisition)"
        headers = {"User-Agent": user_agent}
        last_error: Exception | None = None
        use_curl_first = should_use_curl_first(url)
        if use_curl_first:
            try:
                return self.fetch_bytes_with_curl(url, user_agent=user_agent, insecure=False)
            except Exception as exc:
                last_error = exc
        for attempt in range(3):
            try:
                req = Request(url, headers=headers)
                with urlopen(req, timeout=self.timeout_seconds) as response:
                    return response.read(), response.headers.get("content-type"), int(getattr(response, "status", 200) or 200), None
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise
                retry_delay = float(1 + attempt)
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        retry_delay = float(int(retry_after))
                    except (ValueError, OverflowError):
                        try:
                            retry_at = parsedate_to_datetime(retry_after)
                            if retry_at.tzinfo is None:
                                retry_at = retry_at.replace(tzinfo=timezone.utc)
                            retry_delay = (retry_at - utc_now()).total_seconds()
                        except (TypeError, ValueError, OverflowError):
                            pass
                retry_delay = min(60.0, max(0.0, retry_delay)) if attempt < 2 else 0.0
                # These are bounded transport attempts, not accepted source gaps.
                # Only an exhausted request becomes a failure in requests.jsonl,
                # so the watchdog remains strict without interrupting its retry.
                with self.write_lock:
                    append_jsonl(self.run_root / "transport_attempts.jsonl", {
                        "url": url, "status_code": exc.code, "attempt": attempt + 1,
                        "max_attempts": 3, "retry_delay_seconds": retry_delay,
                        "status": "retry_pending" if attempt < 2 else "retry_exhausted",
                        "timestamp": iso_now(), "raw_only": True,
                    })
                exc.close()
                if attempt == 2:
                    raise
                time.sleep(retry_delay)
            except (URLError, TimeoutError, ssl.SSLError, IncompleteRead, RemoteDisconnected, ConnectionResetError) as exc:
                last_error = exc
                time.sleep(1 + attempt)
        if last_error and ("CERTIFICATE_VERIFY_FAILED" in str(last_error) or "EOF occurred" in str(last_error)):
            try:
                req = Request(url, headers=headers)
                with urlopen(req, timeout=self.timeout_seconds, context=ssl._create_unverified_context()) as response:
                    return (
                        response.read(),
                        response.headers.get("content-type"),
                        int(getattr(response, "status", 200) or 200),
                        "TLS verification/handshake failed in local runtime; retried with verification disabled and recorded warning.",
                    )
            except Exception as exc:
                last_error = exc
        if not use_curl_first:
            for insecure in (False, True):
                try:
                    return self.fetch_bytes_with_curl(url, user_agent=user_agent, insecure=insecure)
                except Exception as exc:
                    last_error = exc
        raise last_error or RuntimeError(f"Failed fetching {url}")

    def fetch_bytes_with_curl(self, url: str, *, user_agent: str, insecure: bool) -> tuple[bytes, str | None, int, str | None]:
        command = [
            "curl",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--max-time",
            str(self.timeout_seconds),
            "-A",
            user_agent,
        ]
        if insecure:
            command.append("--insecure")
        command.append(url)
        result = subprocess.run(command, capture_output=True, check=True, timeout=self.timeout_seconds + 15)
        warning = "urllib failed; downloaded with curl fallback"
        if insecure:
            warning += " using --insecure"
        return result.stdout, None, 200, warning

    def request_id(self, source_id: str, compound: str | None, request_key: str, url: str, params: dict[str, Any] | None) -> str:
        digest = sha256(f"{source_id}|{compound}|{request_key}|{url_with_params(url, params)}".encode("utf-8")).hexdigest()[:16]
        return f"{source_id}:{request_key}:{digest}"

    def log_request(
        self,
        request_id: str,
        source_id: str,
        compound: str | None,
        request_key: str,
        url: str,
        params: dict[str, Any] | None,
        status: str,
        *,
        asset: dict[str, Any] | None = None,
    ) -> None:
        with self.write_lock:
            append_jsonl(
                self.requests_path,
                {
                    "request_id": request_id,
                    "source_id": source_id,
                    "compound": compound,
                    "request_key": request_key,
                    "url": url,
                    "params": params or {},
                    "status": status,
                    "timestamp": iso_now(),
                    "asset_id": asset.get("asset_id") if asset else None,
                    "raw_only": True,
                },
            )

    def log_failure(
        self,
        request_id: str,
        source_id: str,
        compound: str | None,
        request_key: str,
        url: str,
        error: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "request_id": request_id,
            "source_id": source_id,
            "compound": compound,
            "request_key": request_key,
            "url": url,
            "params": params or {},
            "status": "failed",
            "error": error,
            "timestamp": iso_now(),
            "raw_only": True,
        }
        with self.write_lock:
            append_jsonl(self.failures_path, row)
            append_jsonl(self.requests_path, row)
            self.coverage[source_id]["failures"] += 1

    def log_deferred(
        self,
        request_id: str,
        source_id: str,
        compound: str | None,
        request_key: str,
        url: str,
        reason: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Record a planned limit deferral without reporting a transport failure."""

        row = {
            "request_id": request_id,
            "source_id": source_id,
            "compound": compound,
            "request_key": request_key,
            "url": url,
            "params": params or {},
            "status": "deferred",
            "reason": reason,
            "timestamp": iso_now(),
            "raw_only": True,
        }
        with self.write_lock:
            append_jsonl(self.deferred_path, row)
            append_jsonl(self.requests_path, row)
            self.coverage[source_id]["deferred"] += 1

    def write_summary(self) -> dict[str, Any]:
        assets = count_jsonl(self.assets_path)
        failures = count_jsonl(self.failures_path)
        deferred = count_jsonl(self.deferred_path)
        summary = {
            "run_id": self.run_id,
            "output_root": str(self.output_root),
            "run_root": str(self.run_root),
            "created_at": iso_now(),
            "compound_count": len(self.compounds),
            "asset_count": assets,
            "failure_count": failures,
            "deferred_count": deferred,
            "pmid_count_by_compound": {compound: len(values) for compound, values in self.pmids_by_compound.items()},
            "identity_coverage_manifest_by_compound": {
                compound: str(self.identity_coverage_paths.get(compound) or "") for compound in self.compounds
            },
            "pubmed_selection_by_compound": {
                compound: dict(self.literature_selection_counts[compound]) for compound in self.compounds
            },
            "dailymed_selection_by_compound": {
                compound: dict(self.dailymed_selection_counts[compound]) for compound in self.compounds
            },
            "clinicaltrials_selection_by_compound": {
                compound: dict(self.clinicaltrial_selection_counts[compound]) for compound in self.compounds
            },
            "openfda_selection_by_compound": {
                compound: dict(self.openfda_selection_counts[compound]) for compound in self.compounds
            },
            "pkdb_selection_by_compound": {
                compound: dict(self.pkdb_selection_counts[compound]) for compound in self.compounds
            },
            "pmcid_count_by_compound": {compound: len(values) for compound, values in self.pmcids_by_compound.items()},
            "pubmed_linked_pmcid_count_by_compound": {
                compound: len(values) for compound, values in self.pubmed_linked_pmcids_by_compound.items()
            },
            "pmc_fulltext_selected_count_by_compound": {
                compound: len(values) for compound, values in self.pmc_fulltext_selected_by_compound.items()
            },
            "pmc_oa_package_selected_count_by_compound": {
                compound: len(values) for compound, values in self.pmc_oa_selected_by_compound.items()
            },
            "pmc_fulltext_download_policy": {
                "max_pmc_fulltext_per_compound": self.max_pmc_fulltext_per_compound,
                "max_pmc_oa_packages_per_compound": self.max_pmc_oa_packages_per_compound,
                "overflow_behavior": "retain selected PMID-to-PMCID links in the gap manifest; only fulltext_selected PubMed records can enter the PMC queue; deferred PMCIDs are ranked by the source PubMed metadata score",
            },
            "retrieval_profile": "high_precision_v3",
            "shared_sources_included": self.include_shared_sources,
            "shared_source_ids": sorted(self.shared_source_ids),
            "shared_cache_root": str(self.shared_cache_root),
            "combination_labels_included": self.include_combination_labels,
            "coverage_by_source": {source_id: dict(values) for source_id, values in sorted(self.coverage.items())},
            "raw_only": True,
        }
        write_json(self.run_root / "summary.json", summary)
        write_markdown_summary(self.run_root / "README.md", summary)
        write_json(self.output_root / "latest.json", summary)
        return summary


def literature_query(
    compound: str,
    terms: tuple[str, ...] | list[str],
    *,
    family_id: str | None = None,
) -> str:
    return high_precision_literature_query(compound, terms, family_id=family_id)


def batched(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def url_with_params(url: str, params: dict[str, Any] | None) -> str:
    if not params:
        return url
    return url + ("&" if "?" in url else "?") + urlencode(params, doseq=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        file.write("\n")


def load_json_file(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_pmcid(value: str) -> str:
    value = str(value)
    return value if value.upper().startswith("PMC") else f"PMC{value}"


def pmcid_sort_key(value: str) -> int:
    return int(re.sub(r"\D", "", value) or "0")


def pmcids_from_elink(payload: dict[str, Any]) -> set[str]:
    pmcids: set[str] = set()
    for linkset in payload.get("linksets", []) if isinstance(payload, dict) else []:
        for linksetdb in linkset.get("linksetdbs", []) or []:
            for link in linksetdb.get("links", []) or []:
                pmcids.add(normalize_pmcid(str(link)))
    return pmcids


def pmcids_and_pmids_from_elink(payload: dict[str, Any]) -> tuple[set[str], set[str], dict[str, set[str]]]:
    pmcids: set[str] = set()
    pmids_with_links: set[str] = set()
    pmcid_to_pmids: dict[str, set[str]] = defaultdict(set)
    for linkset in payload.get("linksets", []) if isinstance(payload, dict) else []:
        linkset_pmids = {str(item) for item in linkset.get("ids", []) or []}
        has_links = False
        for linksetdb in linkset.get("linksetdbs", []) or []:
            for link in linksetdb.get("links", []) or []:
                pmcid = normalize_pmcid(str(link))
                pmcids.add(pmcid)
                pmcid_to_pmids[pmcid].update(linkset_pmids)
                has_links = True
        if has_links:
            pmids_with_links.update(linkset_pmids)
    return pmcids, pmids_with_links, dict(pmcid_to_pmids)


def elink_payload_covers_pmids(payload: dict[str, Any], expected_pmids: list[str] | set[str]) -> bool:
    """Return true only for a complete, source-successful ELink response."""

    if not isinstance(payload, dict) or payload.get("ERROR") or not isinstance(payload.get("linksets"), list):
        return False
    returned_pmids: set[str] = set()
    for linkset in payload["linksets"]:
        if isinstance(linkset, dict):
            returned_pmids.update(str(value) for value in linkset.get("ids", []) or [])
    return {str(value) for value in expected_pmids} <= returned_pmids


def pmc_oa_links(path: Path) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    try:
        root = ET.fromstring(path.read_bytes())
    except Exception:
        return links
    for link in root.iter("link"):
        href = link.attrib.get("href")
        if not href:
            continue
        links.append({"href": href, "format": link.attrib.get("format", ""), "updated": link.attrib.get("updated", "")})
    return links


def pmc_official_download_url(url: str) -> str:
    """Map the OA service's legacy FTP URI to NCBI's current HTTPS layout.

    NCBI moved the legacy PMC dataset directories below ``deprecated`` in April
    2026, while OA service responses can still contain the former FTP location.
    Keep all other URLs untouched so provenance remains source-defined.
    """

    parsed = urlparse(url)
    if parsed.hostname != "ftp.ncbi.nlm.nih.gov" or not parsed.path.startswith(PMC_LEGACY_FTP_PREFIX):
        return url
    path = parsed.path
    if not path.startswith(PMC_DEPRECATED_HTTPS_PREFIX):
        path = PMC_DEPRECATED_HTTPS_PREFIX + path[len(PMC_LEGACY_FTP_PREFIX) :]
    return parsed._replace(scheme="https", path=path).geturl()


def downloadable_links(path: Path, base_url: str) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    parser = LinkExtractor()
    parser.feed(text)
    links: list[str] = []
    seen: set[str] = set()
    for href in parser.links:
        url = urljoin(base_url, href)
        lowered = urlparse(url).path.lower()
        if not lowered.endswith(DOWNLOAD_EXTENSIONS):
            continue
        if url in seen:
            continue
        seen.add(url)
        links.append(url)
    return links


def page_links(path: Path, base_url: str) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    parser = LinkExtractor()
    parser.feed(text)
    links: list[str] = []
    seen: set[str] = set()
    for href in parser.links:
        url = urljoin(base_url, href)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.path.lower().endswith(DOWNLOAD_EXTENSIONS):
            continue
        clean_url = parsed._replace(fragment="").geturl()
        if clean_url in seen:
            continue
        seen.add(clean_url)
        links.append(clean_url)
    return links


def same_site(root_url: str, candidate_url: str) -> bool:
    root = urlparse(root_url)
    candidate = urlparse(candidate_url)
    return candidate.netloc == root.netloc


def should_use_curl_first(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return any(hostname == suffix or hostname.endswith(f".{suffix}") for suffix in CURL_FIRST_HOST_SUFFIXES)


def extension_from_content_type(content_type: str | None) -> str | None:
    lowered = (content_type or "").lower()
    if "json" in lowered:
        return "json"
    if "xml" in lowered:
        return "xml"
    if "html" in lowered:
        return "html"
    if "pdf" in lowered:
        return "pdf"
    if "csv" in lowered:
        return "csv"
    if "zip" in lowered:
        return "zip"
    if "gzip" in lowered:
        return "gz"
    return None


def extension_from_url(url: str) -> str | None:
    path = urlparse(url).path
    lowered = path.lower()
    for ext in [".tar.gz", ".tsv.gz", ".csv.gz", ".json.gz", *DOWNLOAD_EXTENSIONS]:
        if lowered.endswith(ext):
            return ext.lstrip(".").replace(".", "_")
    suffix = Path(path).suffix
    return suffix.lstrip(".") if suffix else None


def clean_extension(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip(".") or "bin")


def file_checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Comprehensive Raw Retrieval Run",
        "",
        f"Run ID: `{summary['run_id']}`",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Compounds | {summary['compound_count']} |",
        f"| Raw assets | {summary['asset_count']} |",
        f"| Failures / blocked attempts | {summary['failure_count']} |",
        f"| Deferred by configured limits | {summary.get('deferred_count', 0)} |",
        "",
        "## Literature Selection",
        "",
        "| Compound | Candidate PMIDs | Full-text selected | Metadata only | Excluded | Selected PMC full text |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for compound in sorted(summary["pmid_count_by_compound"]):
        lines.append(
            f"| {compound} | {summary['pmid_count_by_compound'].get(compound, 0)} | "
            f"{summary.get('pubmed_selection_by_compound', {}).get(compound, {}).get('fulltext_selected', 0)} | "
            f"{summary.get('pubmed_selection_by_compound', {}).get(compound, {}).get('metadata_only', 0)} | "
            f"{summary.get('pubmed_selection_by_compound', {}).get(compound, {}).get('excluded', 0)} | "
            f"{summary.get('pmc_fulltext_selected_count_by_compound', {}).get(compound, 0)} |"
        )
    policy = summary.get("pmc_fulltext_download_policy", {})
    lines.extend(
        [
            "",
            "## PMC Full Text Download Policy",
            "",
            f"- Max PMC full-text EFetch per compound: `{policy.get('max_pmc_fulltext_per_compound')}`",
            f"- Max PMC OA package downloads per compound: `{policy.get('max_pmc_oa_packages_per_compound')}`",
            f"- Overflow behavior: {policy.get('overflow_behavior')}",
        ]
    )
    lines.extend(["", "## Coverage By Source", "", "| Source | Assets | Failures | Deferred | Bytes |", "|---|---:|---:|---:|---:|"])
    for source_id, row in summary["coverage_by_source"].items():
        lines.append(
            f"| {source_id} | {row.get('assets', 0)} | {row.get('failures', 0)} | "
            f"{row.get('deferred', 0)} | {row.get('bytes', 0)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive raw retrieval for PBPK public sources.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sources-yaml", default="pbpk_kb/config/sources.yaml")
    parser.add_argument("--compounds", default=",".join(DEFAULT_COMPOUNDS))
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--delay-seconds", type=float, default=0.34)
    parser.add_argument("--ncbi-batch-size", type=int, default=200)
    parser.add_argument("--page-size", type=int, default=10000)
    parser.add_argument("--dailymed-workers", type=int, default=8)
    parser.add_argument("--clinicaltrials-workers", type=int, default=2)
    parser.add_argument("--pkdb-workers", type=int, default=4)
    parser.add_argument("--skip-oa-packages", action="store_true")
    parser.add_argument("--skip-repository-mirror", action="store_true")
    parser.add_argument(
        "--include-shared-sources",
        action="store_true",
        help="Refresh explicitly selected source-wide snapshots/repositories outside the compound run.",
    )
    parser.add_argument(
        "--shared-source",
        action="append",
        default=[],
        help="Source ID to refresh with --include-shared-sources; repeat for multiple independent source snapshots.",
    )
    parser.add_argument(
        "--include-combination-labels",
        action="store_true",
        help="Allow DailyMed combination-product labels into the full-label queue. Disabled by default.",
    )
    parser.add_argument(
        "--max-pmc-fulltext-per-compound",
        type=int,
        default=250,
        help="Maximum selected PMCIDs per compound to EFetch as full-text XML; use 0 for no operational cap.",
    )
    parser.add_argument(
        "--max-pmc-oa-packages-per-compound",
        type=int,
        default=100,
        help="Maximum selected PMCIDs per compound for PMC OA package downloads; use 0 for no operational cap.",
    )
    parser.add_argument("--debug-run", action="store_true", help="Mark this run as non-official debug output.")
    args = parser.parse_args()
    if args.include_shared_sources and not args.shared_source:
        parser.error("--include-shared-sources requires at least one --shared-source <source_id>.")
    if args.shared_source and not args.include_shared_sources:
        parser.error("--shared-source requires --include-shared-sources.")
    registered_shared_sources = {
        str(source.get("source_id"))
        for source in load_source_registry(Path(args.sources_yaml))
        if SOURCE_MODE.get(str(source.get("source_id"))) in {"corpus_snapshot", "repository_mirror"}
    }
    invalid_shared_sources = sorted(set(args.shared_source) - registered_shared_sources)
    if invalid_shared_sources:
        parser.error("Unknown or non-shared source ID(s): " + ", ".join(invalid_shared_sources))
    compounds = [item.strip() for item in args.compounds.split(",") if item.strip()]
    retriever = ComprehensiveRawRetriever(
        output_root=Path(args.output_root),
        compounds=compounds,
        source_registry_path=Path(args.sources_yaml),
        timeout_seconds=args.timeout_seconds,
        delay_seconds=args.delay_seconds,
        ncbi_batch_size=args.ncbi_batch_size,
        page_size=args.page_size,
        dailymed_workers=args.dailymed_workers,
        pkdb_workers=args.pkdb_workers,
        download_oa_packages=not args.skip_oa_packages,
        mirror_repositories=not args.skip_repository_mirror,
        include_shared_sources=args.include_shared_sources,
        include_combination_labels=args.include_combination_labels,
        max_pmc_fulltext_per_compound=args.max_pmc_fulltext_per_compound,
        max_pmc_oa_packages_per_compound=args.max_pmc_oa_packages_per_compound,
        official_run=not args.debug_run,
        clinicaltrials_workers=args.clinicaltrials_workers,
        shared_source_ids=args.shared_source,
    )
    summary = retriever.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
