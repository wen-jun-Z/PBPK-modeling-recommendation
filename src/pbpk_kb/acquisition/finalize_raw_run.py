from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlparse
from uuid import uuid4

from pbpk_kb.acquisition.comprehensive_raw_retriever import (
    DEFAULT_COMPOUNDS,
    DEFAULT_OUTPUT_ROOT,
    NCBI_BASE,
    PMC_OA_BASE,
    SOURCE_MODE,
    ComprehensiveRawRetriever,
    append_jsonl,
    batched,
    clean_extension,
    count_jsonl,
    extension_from_content_type,
    extension_from_url,
    file_checksum,
    elink_payload_covers_pmids,
    iso_now,
    load_json_file,
    load_source_registry,
    pmc_official_download_url,
    pmc_oa_links,
    pmcids_and_pmids_from_elink,
    slugify,
    url_with_params,
    write_json,
    write_markdown_summary,
)


def load_assets(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def ensure_failure_resolution_ledger(run_root: Path) -> Path:
    """Materialize an empty history without inventing or replacing resolutions.

    Zero failures is a valid upstream outcome. Inventory still requires the
    explicit ledger, so both finalization and final reconciliation provide it.
    Exclusive creation leaves every existing record and its mtime unchanged.
    """
    path = run_root / "finalization" / "failure_resolutions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8"):
            pass
    except FileExistsError:
        if not path.is_file():
            raise
    return path


def selection_summary_from_ledgers(run_root: Path, compounds: list[str]) -> dict[str, Any]:
    """Rebuild selection counts, including recoveries made outside the live retriever.

    A source record counts once per compound/source. Repeated transport attempts
    may have different request IDs, URLs or timestamps; conflicting decisions or
    record contents are not silently resolved by choosing the latest row.
    """
    root = run_root.resolve()
    result: dict[str, Any] = {}
    for source in ("pubmed", "dailymed", "clinicaltrials", "openfda_drug_label", "pkdb"):
        counts_by_compound = {}
        for compound in compounds:
            path = root / "compounds" / slugify(compound) / "selection" / source / "candidate_records.jsonl"
            if not path.resolve().is_relative_to(root):
                raise ValueError("Selection ledger escapes this acquisition run")
            seen: dict[str, dict[str, Any]] = {}
            counts: Counter[str] = Counter()
            if path.exists():
                with path.open("r", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Malformed selection ledger: {source}/{compound}:{line_number}") from exc
                        if not isinstance(row, dict) or row.get("compound") != compound or row.get("source_id") != source:
                            raise ValueError(f"Selection ledger identity mismatch: {source}/{compound}:{line_number}")
                        record_id = row.get("source_record_id")
                        decision = row.get("decision")
                        if not isinstance(record_id, str) or not record_id.strip() or not isinstance(decision, str) or not decision.strip():
                            raise ValueError(f"Incomplete selection identity or decision: {source}/{compound}:{line_number}")
                        content = {key: value for key, value in row.items()
                                   if key not in {"selected_at", "request_id", "source_url"}}
                        if record_id in seen:
                            if seen[record_id] != content:
                                raise ValueError(f"Conflicting selection records: {source}/{compound}/{record_id}")
                            continue
                        seen[record_id] = content
                        counts[decision] += 1
            counts_by_compound[compound] = dict(sorted(counts.items()))
        prefix = "openfda" if source == "openfda_drug_label" else source
        result[f"{prefix}_selection_by_compound"] = counts_by_compound
    return result


def pmcid_number(value: str) -> int:
    return int(re.sub(r"\D", "", value) or "0")


def pmid_number(value: str) -> int:
    return int(re.sub(r"\D", "", value) or "0")


PMC_AWS_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"


def s3_common_prefixes(path: Path) -> list[str]:
    root = ET.fromstring(path.read_bytes())
    if root.tag.rsplit("}", 1)[-1] != "ListBucketResult":
        raise ValueError("Invalid S3 version listing; cannot classify it as unavailable.")
    if any(n.tag.rsplit("}", 1)[-1] == "IsTruncated" and n.text == "true" for n in root):
        raise ValueError("Truncated S3 version listing; explicit pagination is required.")
    prefixes: list[str] = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "CommonPrefixes":
            continue
        for child in node:
            if child.tag.rsplit("}", 1)[-1] == "Prefix" and child.text:
                prefixes.append(child.text)
    return prefixes


def s3_contents(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    root = ET.fromstring(path.read_bytes())
    if root.tag.rsplit("}", 1)[-1] != "ListBucketResult":
        raise ValueError("Invalid S3 object listing.")
    objects: list[dict[str, Any]] = []
    next_token: str | None = None
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "Contents":
            item: dict[str, Any] = {}
            for child in node:
                child_tag = child.tag.rsplit("}", 1)[-1]
                if child_tag == "Key" and child.text:
                    item["key"] = child.text
                elif child_tag == "Size" and child.text:
                    item["size"] = int(child.text)
                elif child_tag == "ETag" and child.text:
                    item["etag"] = child.text.strip('"')
            if item.get("key"):
                objects.append(item)
        elif tag == "NextContinuationToken" and node.text:
            next_token = node.text
    if any(n.tag.rsplit("}", 1)[-1] == "IsTruncated" and n.text == "true" for n in root) and not next_token:
        raise ValueError("Truncated S3 object listing lacks a continuation token.")
    return objects, next_token


class RawRunFinalizer:
    def __init__(
        self,
        *,
        run_root: Path,
        output_root: Path,
        sources_yaml: Path,
        compounds: list[str],
        timeout_seconds: int,
        delay_seconds: float,
        pmc_download_workers: int,
        max_pmc_object_bytes: int | None,
        finish_shared_sources: bool,
        shared_source_ids: list[str],
    ) -> None:
        self.run_root = run_root
        self.output_root = output_root
        self.sources_yaml = sources_yaml
        self.compounds = compounds
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds
        self.pmc_download_workers = max(1, pmc_download_workers)
        self.max_pmc_object_bytes = max_pmc_object_bytes
        self.finish_shared_sources = finish_shared_sources
        self.shared_source_ids = shared_source_ids
        self.run_id = run_root.name
        self.assets_path = run_root / "assets.jsonl"
        self.failures_path = run_root / "failures.jsonl"
        self.requests_path = run_root / "requests.jsonl"
        self.status_path = run_root / "RUN_STATUS.json"
        self.finalization_root = run_root / "finalization"
        self.resolutions_path = self.finalization_root / "failure_resolutions.jsonl"
        self.pmc_object_plan_path = self.finalization_root / "pmc_object_download_plan.jsonl"
        self.progress_path = self.finalization_root / "progress.json"
        self.historical_failures: list[dict[str, Any]] = []
        self.resolved_failure_request_ids: set[str] = set()
        self.pubmed_link_retry_succeeded_pmids: dict[str, set[str]] = defaultdict(set)
        self.write_lock = threading.Lock()
        self.planned_pmc_object_keys = {
            str(row.get("s3_key"))
            for row in load_assets(self.pmc_object_plan_path)
            if row.get("s3_key")
        }

    def finalize(self) -> dict[str, Any]:
        ensure_failure_resolution_ledger(self.run_root)
        self.historical_failures = load_assets(self.failures_path)
        self.resolved_failure_request_ids = {
            str(row.get("original_failure_request_id"))
            for row in load_assets(self.resolutions_path)
            if row.get("original_failure_request_id")
        }
        message = "Finalizer is completing PMC OA downloads, deferred PMC full text, and failed PubMed link lookups."
        if self.finish_shared_sources:
            message = "Finalizer is completing PMC/PubMed recovery and explicitly requested shared-source snapshots."
        self.write_status("running", message)
        self.complete_remaining_pmc_oa()
        self.retry_pubmed_pmc_links()
        self.resolve_retired_pkdb_timecourse_requests()
        if self.finish_shared_sources:
            self.complete_shared_sources()
        from pbpk_kb.acquisition.source_recovery import record_unavailable_pkdb_observations
        record_unavailable_pkdb_observations(self)
        summary = self.write_summary()
        self.write_status("completed", "Existing raw acquisition run finalized successfully.", summary=summary)
        return summary

    def complete_remaining_pmc_oa(self) -> None:
        existing = self.existing_assets_by_request()
        for compound in self.compounds:
            manifest = self.run_root / "compounds" / slugify(compound) / "pmc_oa" / "pmc_fulltext_gap_manifest.json"
            if not manifest.exists():
                continue
            payload = load_json_file(manifest)
            selected = {str(pmcid) for pmcid in payload.get("selected_for_fulltext_download") or []}
            deferred = {str(pmcid) for pmcid in payload.get("skipped_pmcids") or []}
            self.complete_pmcids(
                compound=compound,
                pmcids=selected | deferred,
                missing_xml_pmcids=deferred,
                existing=existing,
                origin="selected_or_previously_deferred_pmc_fulltext",
            )
            self.write_pmc_finalization_manifest(compound, selected | deferred)

    def complete_pmcids(
        self,
        *,
        compound: str,
        pmcids: set[str],
        missing_xml_pmcids: set[str],
        existing: dict[tuple[Any, ...], dict[str, Any]],
        origin: str,
    ) -> None:
        """Fetch PMC XML plus OA package assets without repeating prior successes."""

        known_xml = self.existing_pmcids_from_efetch(existing, compound)
        pending_xml = sorted((pmcids | missing_xml_pmcids) - known_xml, key=pmcid_number)
        for batch in batched(pending_xml, 20):
            ids = ",".join(batch)
            batch_key = f"pmc_efetch_finalizer_{batch[0]}_{batch[-1]}_{len(batch)}"
            asset = self.fetch_and_store(
                source_id="pmc_oa",
                compound=compound,
                group="pmc/efetch_xml_finalizer",
                request_key=batch_key,
                url=f"{NCBI_BASE}/efetch.fcgi",
                params={"db": "pmc", "id": ids, "retmode": "xml"},
                expected_extension="xml",
                metadata={"pmcids": batch, "origin": origin, "finalizer": True},
            )
            if asset:
                existing[self.asset_key(asset)] = asset

        legacy_file_jobs: list[dict[str, Any]] = []
        s3_file_jobs: list[dict[str, Any]] = []
        expected_s3_keys_by_pmcid: dict[str, set[str]] = {}
        for pmcid in sorted(pmcids, key=pmcid_number):
            package_asset = self.successful_oa_package_asset(existing, compound, pmcid)
            if package_asset is not None:
                self.resolve_covered_by_oa_package(compound, pmcid, package_asset)
                continue

            s3_objects = self.discover_pmc_s3_objects(compound, pmcid, existing, origin)
            if s3_objects:
                expected_s3_keys_by_pmcid[pmcid] = {str(item["key"]) for item in s3_objects}
                for item in s3_objects:
                    object_key = str(item["key"])
                    object_size = int(item.get("size") or 0)
                    if self.max_pmc_object_bytes is not None and object_size > self.max_pmc_object_bytes:
                        self.record_pmc_object_deferred(pmcid, object_key, item, origin)
                        continue
                    request_key = f"oa_s3_{pmcid}_{sha256(object_key.encode('utf-8')).hexdigest()[:16]}"
                    group = f"pmc/oa_s3_files/{pmcid}"
                    key = (compound, "pmc_oa", group, request_key)
                    if key in existing:
                        continue
                    object_url = f"{PMC_AWS_BASE}/{quote(object_key, safe='/')}"
                    s3_file_jobs.append(
                        {
                            "kind": "s3",
                            "pmcid": pmcid,
                            "object_key": object_key,
                            "request_key": request_key,
                            "download_url": object_url,
                            "fetch_kwargs": {
                                "source_id": "pmc_oa",
                                "compound": compound,
                                "group": group,
                                "request_key": request_key,
                                "url": object_url,
                                "expected_extension": extension_from_url(object_key) or "bin",
                                "metadata": {
                                    "pmcid": pmcid,
                                    "s3_bucket": "pmc-oa-opendata",
                                    "s3_key": object_key,
                                    "s3_etag": item.get("etag"),
                                    "s3_size": item.get("size"),
                                    "origin": origin,
                                    "finalizer": True,
                                },
                            },
                        }
                    )
                continue

            # Empty official listings remain explicit availability gaps. Never
            # fall back to the retired OA API or legacy FTP package service.

        self.download_pmc_file_jobs(compound, legacy_file_jobs + s3_file_jobs, existing)
        self.resolve_s3_article_mirrors(compound, expected_s3_keys_by_pmcid, existing)

    def download_pmc_file_jobs(
        self,
        compound: str,
        file_jobs: list[dict[str, Any]],
        existing: dict[tuple[Any, ...], dict[str, Any]],
    ) -> None:
        # OA metadata remains rate-limited above. Static article files are
        # independent, so controlled parallel retrieval prevents a single large
        # supplement from serially blocking the remaining official downloads.
        with ThreadPoolExecutor(max_workers=self.pmc_download_workers) as executor:
            futures = {
                executor.submit(self.fetch_and_store, **job["fetch_kwargs"]): job
                for job in file_jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    file_asset = future.result()
                except Exception as exc:
                    self.log_failure(
                        self.request_id("pmc_oa", compound, str(job["request_key"]), str(job["download_url"]), None),
                        "pmc_oa",
                        compound,
                        str(job["request_key"]),
                        str(job["download_url"]),
                        f"Unhandled concurrent PMC OA retrieval error: {exc}",
                    )
                    continue
                if file_asset:
                    existing[self.asset_key(file_asset)] = file_asset
                    if job["kind"] == "legacy":
                        self.record_download_resolution(
                            source_id="pmc_oa",
                            compound=compound,
                            request_key=str(job["request_key"]),
                            original_url=str(job["original_url"]),
                            resolved_url=str(job["download_url"]),
                            asset=file_asset,
                        )

    def existing_assets_by_request(self) -> dict[tuple[Any, ...], dict[str, Any]]:
        return {self.asset_key(row): row for row in load_assets(self.assets_path) if row.get("request_key")}

    @staticmethod
    def asset_key(asset: dict[str, Any]) -> tuple[Any, ...]:
        return (
            asset.get("compound"),
            asset.get("source_id"),
            asset.get("group"),
            asset.get("request_key"),
        )

    @staticmethod
    def existing_pmcids_from_efetch(existing: dict[tuple[Any, ...], dict[str, Any]], compound: str) -> set[str]:
        pmcids: set[str] = set()
        for asset in existing.values():
            if asset.get("compound") != compound or asset.get("source_id") != "pmc_oa":
                continue
            if not str(asset.get("group") or "").startswith("pmc/efetch_xml"):
                continue
            query = parse_qs(urlparse(str(asset.get("url") or "")).query)
            for value in query.get("id", []):
                pmcids.update(part for part in value.split(",") if part.startswith("PMC"))
        return pmcids

    @staticmethod
    def successful_oa_package_asset(
        existing: dict[tuple[Any, ...], dict[str, Any]], compound: str, pmcid: str
    ) -> dict[str, Any] | None:
        group = f"pmc/oa_files/{pmcid}"
        for asset in existing.values():
            if asset.get("compound") != compound or asset.get("source_id") != "pmc_oa" or asset.get("group") != group:
                continue
            if str(asset.get("request_key") or "").endswith("_tgz"):
                return asset
        return None

    def discover_pmc_s3_objects(
        self,
        compound: str,
        pmcid: str,
        existing: dict[tuple[Any, ...], dict[str, Any]],
        origin: str,
    ) -> list[dict[str, Any]]:
        """List all official PMC Open Data article objects for one PMCID."""

        listing_group = f"pmc/oa_s3_listing/{pmcid}"
        version_key = (compound, "pmc_oa", listing_group, f"oa_s3_versions_{pmcid}")
        version_asset = existing.get(version_key)
        if version_asset is None:
            version_asset = self.fetch_and_store(
                source_id="pmc_oa",
                compound=compound,
                group=listing_group,
                request_key=f"oa_s3_versions_{pmcid}",
                url=f"{PMC_AWS_BASE}/",
                params={"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"},
                expected_extension="xml",
                metadata={"pmcid": pmcid, "retrieval_method": "ncbi_pmc_aws_s3_list_versions", "origin": origin, "finalizer": True},
            )
            if version_asset is None:
                raise RuntimeError("S3 version listing failed; article availability is unresolved.")
            existing[self.asset_key(version_asset)] = version_asset

        version_prefixes = s3_common_prefixes(Path(str(version_asset["storage_path"])))
        objects_by_key: dict[str, dict[str, Any]] = {}
        for version_prefix in version_prefixes:
            if not re.fullmatch(re.escape(pmcid) + r"\.\d+/", version_prefix):
                raise ValueError("S3 listing returned a different article prefix.")
            prefix_hash = sha256(version_prefix.encode("utf-8")).hexdigest()[:12]
            page = 1
            continuation_token: str | None = None
            seen_tokens: set[str] = set()
            while True:
                if page > 100:
                    raise RuntimeError("S3 article pagination exceeded 100 pages; stopping downloads.")
                request_key = f"oa_s3_objects_{pmcid}_{prefix_hash}_{page}"
                object_key = (compound, "pmc_oa", listing_group, request_key)
                object_asset = existing.get(object_key)
                if object_asset is None:
                    params: dict[str, Any] = {"list-type": "2", "prefix": version_prefix}
                    if continuation_token:
                        params["continuation-token"] = continuation_token
                    object_asset = self.fetch_and_store(
                        source_id="pmc_oa",
                        compound=compound,
                        group=listing_group,
                        request_key=request_key,
                        url=f"{PMC_AWS_BASE}/",
                        params=params,
                        expected_extension="xml",
                        metadata={
                            "pmcid": pmcid,
                            "s3_prefix": version_prefix,
                            "page": page,
                            "retrieval_method": "ncbi_pmc_aws_s3_list_objects",
                            "origin": origin,
                            "finalizer": True,
                        },
                    )
                    if object_asset is None:
                        raise RuntimeError("S3 object listing failed; refusing a partial article plan.")
                    existing[self.asset_key(object_asset)] = object_asset
                objects, continuation_token = s3_contents(Path(str(object_asset["storage_path"])))
                for item in objects:
                    if not str(item["key"]).startswith(version_prefix) or ".." in str(item["key"]).split("/"):
                        raise ValueError("S3 object key escaped the requested article version.")
                    objects_by_key[str(item["key"])] = item
                if not continuation_token:
                    break
                if continuation_token in seen_tokens:
                    raise RuntimeError("S3 repeated a continuation token; stopping pagination.")
                seen_tokens.add(continuation_token)
                page += 1
        return list(objects_by_key.values())

    def resolve_covered_by_oa_package(self, compound: str, pmcid: str, package_asset: dict[str, Any]) -> None:
        for row in self.historical_failures:
            if row.get("source_id") != "pmc_oa" or row.get("compound") != compound:
                continue
            if not str(row.get("request_key") or "").startswith(f"oa_file_{pmcid}_"):
                continue
            self.record_resolution(
                row,
                status="resolved",
                resolution_method="existing_ncbI_pmc_oa_article_package",
                detail={
                    "package_asset_id": package_asset.get("asset_id"),
                    "package_path": package_asset.get("storage_path"),
                    "package_checksum": package_asset.get("checksum"),
                    "note": "The verified official article package contains the article XML, PDF when present, media, and supplementary files.",
                },
            )

    def resolve_s3_article_mirrors(
        self,
        compound: str,
        expected_s3_keys_by_pmcid: dict[str, set[str]],
        existing: dict[tuple[Any, ...], dict[str, Any]],
    ) -> None:
        for pmcid, expected_keys in expected_s3_keys_by_pmcid.items():
            s3_assets = [
                asset
                for asset in existing.values()
                if asset.get("compound") == compound
                and asset.get("source_id") == "pmc_oa"
                and asset.get("group") == f"pmc/oa_s3_files/{pmcid}"
            ]
            downloaded_by_key = {
                str((asset.get("metadata") or {}).get("s3_key")): asset
                for asset in s3_assets
                if (asset.get("metadata") or {}).get("s3_key")
            }
            if not expected_keys or not expected_keys <= set(downloaded_by_key):
                continue
            article_pdf_assets = [asset for key, asset in downloaded_by_key.items() if key.lower().endswith(".pdf")]
            representative_asset = next(iter(downloaded_by_key.values()))
            for row in self.historical_failures:
                if row.get("source_id") != "pmc_oa" or row.get("compound") != compound:
                    continue
                if not str(row.get("request_key") or "").startswith(f"oa_file_{pmcid}_"):
                    continue
                if "/oa_pdf/" in str(row.get("url") or "") and not article_pdf_assets:
                    continue
                self.record_resolution(
                    row,
                    status="resolved",
                    resolution_method="ncbi_pmc_aws_article_object_mirror",
                    detail={
                        "s3_bucket": "pmc-oa-opendata",
                        "pmcid": pmcid,
                        "object_count": len(expected_keys),
                        "representative_asset_id": representative_asset.get("asset_id"),
                        "representative_path": representative_asset.get("storage_path"),
                        "note": "The official NCBI PMC Open Data article mirror was downloaded as individual raw XML, PDF, media, and supplementary objects instead of the legacy tar bundle.",
                    },
                )

    def record_pmc_object_deferred(
        self,
        pmcid: str,
        object_key: str,
        item: dict[str, Any],
        origin: str,
    ) -> None:
        with self.write_lock:
            if object_key in self.planned_pmc_object_keys:
                return
            append_jsonl(
                self.pmc_object_plan_path,
                {
                    "status": "deferred",
                    "reason": "object_exceeds_configured_local_download_budget",
                    "source_id": "pmc_oa",
                    "pmcid": pmcid,
                    "s3_bucket": "pmc-oa-opendata",
                    "s3_key": object_key,
                    "url": f"{PMC_AWS_BASE}/{quote(object_key, safe='/')}",
                    "byte_count": int(item.get("size") or 0),
                    "etag": item.get("etag"),
                    "origin": origin,
                    "recorded_at": iso_now(),
                    "raw_only": True,
                },
            )
            self.planned_pmc_object_keys.add(object_key)

    def record_pmc_legacy_object_deferred(
        self,
        pmcid: str,
        original_url: str,
        link: dict[str, Any],
        origin: str,
    ) -> None:
        plan_key = f"legacy:{original_url}"
        with self.write_lock:
            if plan_key in self.planned_pmc_object_keys:
                return
            append_jsonl(
                self.pmc_object_plan_path,
                {
                    "status": "deferred",
                    "reason": "article_not_present_in_current_pmc_aws_dataset_and_legacy_large_object_not_downloaded_in_plan_only_mode",
                    "source_id": "pmc_oa",
                    "pmcid": pmcid,
                    "url": pmc_official_download_url(original_url),
                    "original_url": original_url,
                    "format": link.get("format"),
                    "origin": origin,
                    "recorded_at": iso_now(),
                    "raw_only": True,
                },
            )
            self.planned_pmc_object_keys.add(plan_key)

    def retry_pubmed_pmc_links(self) -> None:
        """Retry prior ELink transport failures in small batches, then expand new PMC links."""

        failed_pmids_by_compound: dict[str, set[str]] = defaultdict(set)
        failure_rows_by_compound: dict[str, list[dict[str, Any]]] = defaultdict(list)
        known_failure_request_ids = {str(row.get("request_id") or "") for row in self.historical_failures}

        # HTTP 200 does not guarantee a valid ELink result. Promote malformed,
        # source-error, or partial payloads into the append-only failure ledger
        # so they are retried and auditable like transport failures.
        for asset in load_assets(self.assets_path):
            if asset.get("source_id") != "pubmed" or not str(asset.get("group") or "").startswith("pubmed/elink_pmc"):
                continue
            query = parse_qs(urlparse(str(asset.get("url") or "")).query)
            pmids = [str(value) for value in query.get("id", [])]
            if not pmids:
                metadata = asset.get("metadata") or {}
                pmids = [str(value) for value in metadata.get("pmids", [])] if isinstance(metadata, dict) else []
            payload = load_json_file(asset["storage_path"])
            if not pmids or elink_payload_covers_pmids(payload, pmids):
                continue
            request_id = str(asset.get("request_id") or "")
            if request_id in known_failure_request_ids:
                continue
            failure_row = {
                "request_id": request_id,
                "source_id": "pubmed",
                "compound": asset.get("compound"),
                "request_key": asset.get("request_key"),
                "url": asset.get("url"),
                "params": {"id": pmids},
                "status": "failed",
                "error": "HTTP 200 source_error_payload: ELink JSON was invalid, contained ERROR, or omitted requested PMIDs",
                "timestamp": iso_now(),
                "raw_only": True,
                "finalizer": True,
            }
            append_jsonl(self.failures_path, failure_row)
            append_jsonl(self.requests_path, failure_row)
            self.historical_failures.append(failure_row)
            known_failure_request_ids.add(request_id)

        for row in self.historical_failures:
            if row.get("source_id") != "pubmed" or not str(row.get("request_key") or "").startswith("elink_pmc"):
                continue
            if str(row.get("request_id") or "") in self.resolved_failure_request_ids:
                continue
            compound = str(row.get("compound") or "")
            if not compound:
                continue
            pmids = {str(value) for value in (row.get("params") or {}).get("id", [])}
            if not pmids:
                continue
            failed_pmids_by_compound[compound].update(pmids)
            failure_rows_by_compound[compound].append(row)

        existing = self.existing_assets_by_request()
        for compound, pmids in failed_pmids_by_compound.items():
            new_pmcids: set[str] = set()
            pmcid_to_pmids: dict[str, set[str]] = defaultdict(set)
            pending_jobs: list[tuple[list[str], str]] = []
            for batch in batched(sorted(pmids, key=pmid_number), 1):
                batch_key = f"elink_pmc_finalizer_single_{batch[0]}"
                asset = existing.get((compound, "pubmed", "pubmed/elink_pmc_finalizer", batch_key))
                if asset is None:
                    pending_jobs.append((batch, batch_key))
                    continue
                payload = load_json_file(asset["storage_path"])
                if not elink_payload_covers_pmids(payload, batch):
                    pending_jobs.append((batch, f"{batch_key}_recovery"))
                    continue
                self.pubmed_link_retry_succeeded_pmids[compound].update(batch)
                pmcids, _linked_pmids, mapping = pmcids_and_pmids_from_elink(payload)
                new_pmcids.update(pmcids)
                for pmcid, linked_pmids in mapping.items():
                    pmcid_to_pmids[pmcid].update(linked_pmids)

            # Reuse successful batches from an interrupted attempt. Network
            # retries are bounded to three concurrent NCBI requests, which is
            # faster than serial recovery without turning a transient issue
            # into a burst of duplicate calls.
            with ThreadPoolExecutor(max_workers=min(3, max(1, len(pending_jobs)))) as executor:
                futures = {
                    executor.submit(
                        self.fetch_and_store,
                        source_id="pubmed",
                        compound=compound,
                        group="pubmed/elink_pmc_finalizer",
                        request_key=batch_key,
                        url=f"{NCBI_BASE}/elink.fcgi",
                        params={"dbfrom": "pubmed", "db": "pmc", "linkname": "pubmed_pmc", "id": batch, "retmode": "json"},
                        expected_extension="json",
                        metadata={"pmids": batch, "finalizer": True},
                    ): (batch, batch_key)
                    for batch, batch_key in pending_jobs
                }
                for future in as_completed(futures):
                    batch, completed_batch_key = futures[future]
                    try:
                        asset = future.result()
                    except Exception:
                        asset = None
                    if asset is None:
                        continue
                    existing[self.asset_key(asset)] = asset
                    payload = load_json_file(asset["storage_path"])
                    if not elink_payload_covers_pmids(payload, batch):
                        self.log_failure(
                            str(asset.get("request_id") or ""),
                            "pubmed",
                            compound,
                            str(asset.get("request_key") or completed_batch_key),
                            str(asset.get("url") or f"{NCBI_BASE}/elink.fcgi"),
                            "HTTP 200 source_error_payload during single-PMID recovery",
                            params={"id": batch},
                        )
                        continue
                    self.pubmed_link_retry_succeeded_pmids[compound].update(batch)
                    pmcids, _linked_pmids, mapping = pmcids_and_pmids_from_elink(payload)
                    new_pmcids.update(pmcids)
                    for pmcid, linked_pmids in mapping.items():
                        pmcid_to_pmids[pmcid].update(linked_pmids)

            if new_pmcids:
                self.complete_pmcids(
                    compound=compound,
                    pmcids=new_pmcids,
                    missing_xml_pmcids=new_pmcids,
                    existing=existing,
                    origin="retried_pubmed_elink_pmc_link",
                )

            for row in failure_rows_by_compound[compound]:
                requested_pmids = {str(value) for value in (row.get("params") or {}).get("id", [])}
                if requested_pmids and requested_pmids <= self.pubmed_link_retry_succeeded_pmids[compound]:
                    self.record_resolution(
                        row,
                        status="resolved",
                        resolution_method="retry_pubmed_elink_in_small_batches",
                        detail={"retried_pmid_count": len(requested_pmids), "retried_pmcid_count": len(new_pmcids)},
                    )

            write_json(
                self.finalization_root / "pubmed_recovery" / f"{slugify(compound)}.json",
                {
                    "compound": compound,
                    "source_id": "pubmed",
                    "retried_pmids": sorted(pmids, key=pmid_number),
                    "successful_pmids": sorted(self.pubmed_link_retry_succeeded_pmids[compound], key=pmid_number),
                    "recovered_pmcids": sorted(new_pmcids, key=pmcid_number),
                    "pmcid_to_pmids": {
                        pmcid: sorted(values, key=pmid_number) for pmcid, values in sorted(pmcid_to_pmids.items())
                    },
                    "complete": pmids <= self.pubmed_link_retry_succeeded_pmids[compound],
                    "raw_only": True,
                    "recovered_at": iso_now(),
                },
            )

    def record_download_resolution(
        self,
        *,
        source_id: str,
        compound: str,
        request_key: str,
        original_url: str,
        resolved_url: str,
        asset: dict[str, Any],
    ) -> None:
        for row in self.historical_failures:
            if (
                row.get("source_id") == source_id
                and row.get("compound") == compound
                and row.get("request_key") == request_key
                and row.get("url") in {original_url, resolved_url}
            ):
                self.record_resolution(
                    row,
                    status="resolved",
                    resolution_method="ncbi_pmc_legacy_https_deprecated_path",
                    detail={
                        "original_url": original_url,
                        "resolved_url": resolved_url,
                        "asset_id": asset.get("asset_id"),
                        "storage_path": asset.get("storage_path"),
                        "checksum": asset.get("checksum"),
                    },
                )

    def resolve_retired_pkdb_timecourse_requests(self) -> None:
        """Close legacy direct-timecourse requests; current runs audit the official bulk ZIP."""

        existing_assets = self.existing_assets_by_request()
        for row in self.historical_failures:
            if row.get("source_id") != "pkdb" or not str(row.get("request_key") or "").startswith("timecourses"):
                continue
            sid = str((row.get("params") or {}).get("search_multi_match") or "")
            detail_key = (str(row.get("compound") or ""), "pkdb", f"pkdb/children/{sid}", f"study_detail_{sid}")
            detail_asset = existing_assets.get(detail_key)
            if detail_asset is None:
                continue
            self.record_resolution(
                row,
                status="not_available_from_current_api",
                resolution_method="removed_nonexistent_pkdb_timecourses_endpoint",
                detail={
                    "study_detail_asset_id": detail_asset.get("asset_id"),
                    "study_detail_path": detail_asset.get("storage_path"),
                    "note": "The public API has no direct /timecourses/ route. Current acquisition uses and audits the official filter download ZIP; this resolution applies only to legacy direct-endpoint requests.",
                },
            )

    def record_resolution(
        self,
        original_failure: dict[str, Any],
        *,
        status: str,
        resolution_method: str,
        detail: dict[str, Any],
    ) -> None:
        original_request_id = str(original_failure.get("request_id") or "")
        with self.write_lock:
            if not original_request_id or original_request_id in self.resolved_failure_request_ids:
                return
            append_jsonl(
                self.resolutions_path,
                {
                    "resolution_id": f"resolution:{uuid4().hex}",
                    "original_failure_request_id": original_request_id,
                    "source_id": original_failure.get("source_id"),
                    "compound": original_failure.get("compound"),
                    "original_request_key": original_failure.get("request_key"),
                    "original_url": original_failure.get("url"),
                    "status": status,
                    "resolution_method": resolution_method,
                    "detail": detail,
                    "resolved_at": iso_now(),
                    "raw_only": True,
                },
            )
            self.resolved_failure_request_ids.add(original_request_id)

    def write_pmc_finalization_manifest(self, compound: str, pmcids: set[str]) -> None:
        assets = self.existing_assets_by_request()
        package_assets = [
            asset
            for asset in assets.values()
            if asset.get("compound") == compound
            and asset.get("source_id") == "pmc_oa"
            and str(asset.get("group") or "").startswith("pmc/oa_files/")
        ]
        write_json(
            self.finalization_root / "pmc" / f"{slugify(compound)}.json",
            {
                "compound": compound,
                "pmcid_count_targeted_for_finalization": len(pmcids),
                "pmcids_targeted_for_finalization": sorted(pmcids, key=pmcid_number),
                "oa_file_asset_count_after_finalization": len(package_assets),
                "download_layout": "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/...",
                "raw_only": True,
                "created_at": iso_now(),
            },
        )

    def complete_shared_sources(self) -> None:
        from pbpk_kb.acquisition.source_recovery import MODEL_REPOSITORIES, repair_model_snapshot
        if self.shared_source_ids == ["osp_pbpk_model_library"] and set(self.compounds) <= set(MODEL_REPOSITORIES):
            repair_model_snapshot(self.run_root, self.output_root, self.compounds)
            return
        retriever = ComprehensiveRawRetriever(
            output_root=self.output_root,
            compounds=self.compounds,
            source_registry_path=self.sources_yaml,
            timeout_seconds=self.timeout_seconds,
            delay_seconds=self.delay_seconds,
            ncbi_batch_size=200,
            page_size=10000,
            dailymed_workers=1,
            pkdb_workers=1,
            download_oa_packages=True,
            mirror_repositories=True,
            include_shared_sources=True,
            include_combination_labels=False,
            max_pmc_fulltext_per_compound=250,
            max_pmc_oa_packages_per_compound=100,
            official_run=True,
            shared_source_ids=self.shared_source_ids,
        )
        retriever.run_id = self.run_id
        retriever.run_root = self.run_root
        retriever.status_path = self.status_path
        retriever.acquisition_plan_path = self.run_root / "acquisition_plan.json"
        retriever.download_plan_path = self.run_root / "download_plan.json"
        retriever.requests_path = self.requests_path
        retriever.failures_path = self.failures_path
        retriever.assets_path = self.assets_path
        retriever.retrieve_shared_sources()

    def existing_asset_groups(self) -> set[tuple[Any, ...]]:
        groups: set[tuple[Any, ...]] = set()
        for row in load_assets(self.assets_path):
            compound = row.get("compound")
            source_id = row.get("source_id")
            group = row.get("group")
            request_key = row.get("request_key")
            groups.add((compound, source_id, group))
            groups.add((compound, source_id, group, request_key))
        return groups

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
    ) -> dict[str, Any] | None:
        request_id = self.request_id(source_id, compound, request_key, url, params)
        final_url = url_with_params(url, params)
        self.log_request(request_id, source_id, compound, request_key, final_url, params, "started")
        try:
            payload, content_type, status_code = self.fetch_bytes(final_url)
            checksum = sha256(payload).hexdigest()
            extension = clean_extension(expected_extension or extension_from_content_type(content_type) or extension_from_url(final_url) or "bin")
            folder = self.run_root / (f"compounds/{slugify(compound)}" if compound else "shared") / group / source_id
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{slugify(request_key)}_{checksum[:12]}.{extension}"
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
                "transport_warning": "finalizer fast retrieval",
                "metadata": metadata or {},
                "raw_only": True,
            }
            write_json(metadata_path, asset)
            with self.write_lock:
                append_jsonl(self.assets_path, asset)
            self.log_request(request_id, source_id, compound, request_key, final_url, params, "fetched", asset=asset)
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            return asset
        except Exception as exc:
            self.log_failure(request_id, source_id, compound, request_key, final_url, str(exc), params=params)
            return None

    def fetch_bytes(self, url: str) -> tuple[bytes, str | None, int]:
        command = [
            "curl",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "1",
            "--retry-delay",
            "1",
            "--max-time",
            str(self.timeout_seconds),
            "-A",
            "pbpk-public-raw-retrieval-finalizer/0.1",
        ]
        if urlparse(url).hostname == "eutils.ncbi.nlm.nih.gov":
            command.append("--http1.1")
        command.append(url)
        result = subprocess.run(command, capture_output=True, check=True, timeout=self.timeout_seconds + 5)
        return result.stdout, None, 200

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
                    "finalizer": True,
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
            "finalizer": True,
        }
        with self.write_lock:
            append_jsonl(self.failures_path, row)
            append_jsonl(self.requests_path, row)

    def write_summary(self) -> dict[str, Any]:
        assets = load_assets(self.assets_path)
        failure_rows = load_assets(self.failures_path)
        failures = len(failure_rows)
        resolution_rows = load_assets(self.resolutions_path)
        resolution_status_counts = Counter(str(row.get("status") or "unknown") for row in resolution_rows)
        resolved_failure_ids = {
            str(row.get("original_failure_request_id"))
            for row in resolution_rows
            if row.get("status") == "resolved" and row.get("original_failure_request_id")
        }
        known_api_limit_ids = {
            str(row.get("original_failure_request_id"))
            for row in resolution_rows
            if row.get("status") == "not_available_from_current_api" and row.get("original_failure_request_id")
        }
        historical_failure_ids = {str(row.get("request_id") or "") for row in self.historical_failures}
        unresolved_historical = [
            row
            for row in self.historical_failures
            if str(row.get("request_id") or "") not in resolved_failure_ids
            and str(row.get("request_id") or "") not in known_api_limit_ids
        ]
        post_finalization_failures = [
            row for row in failure_rows if row.get("finalizer") and str(row.get("request_id") or "") not in historical_failure_ids
        ]
        coverage: dict[str, dict[str, int]] = defaultdict(lambda: {"assets": 0, "failures": 0, "bytes": 0})
        for row in assets:
            source_id = row.get("source_id") or "unknown"
            coverage[source_id]["assets"] += 1
            coverage[source_id]["bytes"] += int(row.get("byte_count") or 0)
        if self.failures_path.exists():
            with self.failures_path.open("r", encoding="utf-8", errors="replace") as file:
                for line in file:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    source_id = row.get("source_id") or "unknown"
                    coverage[source_id]["failures"] += 1
        pmid_counts: dict[str, int] = {}
        pmcid_counts: dict[str, int] = {}
        linked_counts: dict[str, int] = {}
        selected_counts: dict[str, int] = {}
        oa_selected_counts: dict[str, int] = {}
        for compound in self.compounds:
            pubmed_gap = self.run_root / "compounds" / slugify(compound) / "pubmed" / "open_access_gap_manifest.json"
            if pubmed_gap.exists():
                payload = load_json_file(pubmed_gap)
                pmid_counts[compound] = int(payload.get("candidate_pmid_count") or payload.get("pmid_count") or 0)
            pmc_gap = self.run_root / "compounds" / slugify(compound) / "pmc_oa" / "pmc_fulltext_gap_manifest.json"
            if pmc_gap.exists():
                payload = load_json_file(pmc_gap)
                pmcid_counts[compound] = int(payload.get("discovered_pmcid_count") or 0)
                linked_counts[compound] = int(
                    payload.get("selected_pubmed_linked_pmcid_count")
                    or payload.get("pubmed_elink_pmcid_count")
                    or 0
                )
                selected_counts[compound] = len(payload.get("selected_for_fulltext_download") or [])
                oa_selected_counts[compound] = len(payload.get("selected_for_fulltext_download") or [])
        shared_snapshots = {
            source_id: str(self.output_root / "source_cache" / source_id / self.run_id / "snapshot_manifest.json")
            for source_id in self.shared_source_ids
            if (self.output_root / "source_cache" / source_id / self.run_id / "snapshot_manifest.json").exists()
        }
        summary = {
            "run_id": self.run_id,
            "output_root": str(self.output_root),
            "run_root": str(self.run_root),
            "created_at": iso_now(),
            "compound_count": len(self.compounds),
            "asset_count": len(assets),
            "failure_count": failures,
            "historical_failure_attempt_count": len(self.historical_failures),
            "failure_resolution_count": len(resolution_rows),
            "failure_resolution_status_counts": dict(sorted(resolution_status_counts.items())),
            "remaining_unresolved_historical_failure_count": len(unresolved_historical),
            "post_finalization_failure_count": len(post_finalization_failures),
            "known_public_api_data_limit_count": len(known_api_limit_ids),
            "pmid_count_by_compound": pmid_counts,
            "pmcid_count_by_compound": pmcid_counts,
            "pubmed_linked_pmcid_count_by_compound": linked_counts,
            "pmc_fulltext_selected_count_by_compound": selected_counts,
            "pmc_oa_package_selected_count_by_compound": oa_selected_counts,
            "pmc_fulltext_download_policy": {
                "max_pmc_fulltext_per_compound": 0,
                "max_pmc_oa_packages_per_compound": 0,
                "overflow_behavior": "The finalizer expands all previously selected and operationally deferred PMCIDs while retaining the original selection and deferral audit records.",
                "finalizer_timeout_seconds": self.timeout_seconds,
                "max_pmc_object_bytes": self.max_pmc_object_bytes,
            },
            "historical_deferred_count": count_jsonl(self.run_root / "deferred.jsonl"),
            "pmc_object_download_plan_count": count_jsonl(self.pmc_object_plan_path),
            "pmc_object_download_plan_path": str(self.pmc_object_plan_path),
            "shared_source_snapshots": shared_snapshots,
            "coverage_by_source": {source_id: values for source_id, values in sorted(coverage.items())},
            **selection_summary_from_ledgers(self.run_root, self.compounds),
            "raw_only": True,
            "finalized_existing_run": True,
        }
        write_json(self.run_root / "summary.json", summary)
        write_markdown_summary(self.run_root / "README.md", summary)
        write_json(self.output_root / "latest.json", summary)
        return summary

    def write_status(self, status: str, message: str, *, summary: dict[str, Any] | None = None) -> None:
        payload = {
            "run_id": self.run_id,
            "status": status,
            "message": message,
            "official_run": True,
            "accepted_as_final_result": status == "completed",
            "raw_only": True,
            "updated_at": iso_now(),
            "run_root": str(self.run_root),
            "download_plan_path": str(self.run_root / "download_plan.json"),
            "summary_path": str(self.run_root / "summary.json"),
            "finalized_existing_run": True,
        }
        if summary:
            payload["summary"] = summary
        write_json(self.status_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize an existing comprehensive raw retrieval run.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sources-yaml", default="pbpk_kb/config/sources.yaml")
    parser.add_argument("--compounds", default=",".join(DEFAULT_COMPOUNDS))
    parser.add_argument("--timeout-seconds", type=int, default=8)
    parser.add_argument("--delay-seconds", type=float, default=0.05)
    parser.add_argument(
        "--pmc-download-workers",
        type=int,
        default=6,
        help="Maximum concurrent static PMC OA package/PDF downloads; NCBI metadata requests remain serial.",
    )
    parser.add_argument(
        "--max-pmc-object-bytes",
        type=int,
        default=None,
        help="Defer individual PMC AWS objects above this size into a source-backed download plan; use 0 for manifest-only mode.",
    )
    parser.add_argument("--include-shared-sources", action="store_true")
    parser.add_argument("--shared-source", action="append", default=[])
    parser.add_argument("--skip-shared-sources", action="store_true", help=argparse.SUPPRESS)
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
    finalizer = RawRunFinalizer(
        run_root=Path(args.run_root),
        output_root=Path(args.output_root),
        sources_yaml=Path(args.sources_yaml),
        compounds=compounds,
        timeout_seconds=args.timeout_seconds,
        delay_seconds=args.delay_seconds,
        pmc_download_workers=args.pmc_download_workers,
        max_pmc_object_bytes=args.max_pmc_object_bytes,
        finish_shared_sources=args.include_shared_sources and not args.skip_shared_sources,
        shared_source_ids=args.shared_source,
    )
    summary = finalizer.finalize()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
