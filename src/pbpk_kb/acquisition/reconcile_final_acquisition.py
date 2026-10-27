"""Reconcile a completed raw acquisition run into a traceable final report."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pbpk_kb.acquisition.comprehensive_raw_retriever import (
    DEFAULT_COMPOUNDS,
    elink_payload_covers_pmids,
    slugify,
    write_markdown_summary,
)
from pbpk_kb.acquisition.finalize_raw_run import ensure_failure_resolution_ledger


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pmcid_from_asset(asset: dict[str, Any]) -> str | None:
    metadata = asset.get("metadata") or {}
    value = metadata.get("pmcid") if isinstance(metadata, dict) else None
    if isinstance(value, str) and value.startswith("PMC"):
        return value
    group = str(asset.get("group") or "")
    for part in reversed(group.split("/")):
        if part.startswith("PMC"):
            return part
    return None


def confirmed_absent_from_public_cloud(asset: dict[str, Any], pmcid: str) -> bool:
    """Only a verified, complete, exactly scoped empty listing proves absence."""
    if asset.get("request_key") != f"oa_s3_versions_{pmcid}":
        return False
    url = urlparse(str(asset.get("url") or ""))
    if url.hostname != "pmc-oa-opendata.s3.amazonaws.com" or parse_qs(url.query).get("prefix") != [pmcid + "."]:
        return False
    path = Path(str(asset.get("storage_path") or ""))
    try:
        content = path.read_bytes()
        if sha256(content).hexdigest() != asset.get("checksum"):
            return False
        root = ET.fromstring(content)
    except (OSError, ET.ParseError):
        return False
    values = {node.tag.rsplit("}", 1)[-1]: node.text for node in root}
    return (
        root.tag.rsplit("}", 1)[-1] == "ListBucketResult"
        and values.get("Prefix") == pmcid + "."
        and values.get("IsTruncated") == "false"
        and values.get("KeyCount") == "0"
        and "CommonPrefixes" not in values
        and "Contents" not in values
    )


def pmc_object_key_from_plan(item: dict[str, Any]) -> str | None:
    s3_key = item.get("s3_key")
    if s3_key:
        return f"s3:{s3_key}"
    url = item.get("url")
    return f"url:{url}" if url else None


def pmc_object_key_from_asset(asset: dict[str, Any]) -> str | None:
    metadata = asset.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    s3_key = metadata.get("s3_key")
    if s3_key:
        return f"s3:{s3_key}"
    url = asset.get("url")
    return f"url:{url}" if url else None


def target_pmcids_by_compound(run_root: Path, compounds: list[str]) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = defaultdict(set)
    for compound in compounds:
        path = run_root / "compounds" / slugify(compound) / "pmc_oa" / "pmc_fulltext_gap_manifest.json"
        payload = read_json(path)
        targets[compound].update(str(value) for value in payload.get("selected_for_fulltext_download") or [])
        targets[compound].update(str(value) for value in payload.get("skipped_pmcids") or [])
        recovery = read_json(run_root / "finalization" / "pubmed_recovery" / f"{slugify(compound)}.json")
        targets[compound].update(str(value) for value in recovery.get("recovered_pmcids") or [])
    return targets


def pmcid_from_failure(row: dict[str, Any]) -> str | None:
    text = " ".join(str(row.get(field) or "") for field in ("request_id", "request_key", "url"))
    match = re.search(r"PMC\d+", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def pmcids_from_efetch_asset(asset: dict[str, Any]) -> set[str]:
    metadata = asset.get("metadata") or {}
    values = metadata.get("pmcids", []) if isinstance(metadata, dict) else []
    pmcids = {str(value) for value in values if str(value).startswith("PMC")}
    query = parse_qs(urlparse(str(asset.get("url") or "")).query)
    for value in query.get("id", []):
        pmcids.update(part for part in value.split(",") if part.startswith("PMC"))
    return pmcids


def oa_metadata_has_links(asset: dict[str, Any]) -> bool | None:
    path = Path(str(asset.get("storage_path") or ""))
    if not path.is_file():
        return None
    try:
        root = ET.fromstring(path.read_bytes())
    except Exception:
        return None
    return any(node.tag.rsplit("}", 1)[-1] == "link" and node.attrib.get("href") for node in root.iter())


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def expected_elink_pmids(asset: dict[str, Any]) -> list[str]:
    query = parse_qs(urlparse(str(asset.get("url") or "")).query)
    pmids: list[str] = []
    for value in query.get("id", []):
        pmids.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(pmids))


def validate_pubmed_elink_payloads(
    run_root: Path,
    assets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recovery_successes: dict[str, set[str]] = defaultdict(set)
    recovery_root = run_root / "finalization" / "pubmed_recovery"
    for path in recovery_root.glob("*.json") if recovery_root.exists() else []:
        if path.name.startswith("._"):
            continue
        recovery = read_json(path)
        compound = str(recovery.get("compound") or "")
        recovery_successes[compound].update(str(value) for value in recovery.get("successful_pmids") or [])

    rows: list[dict[str, Any]] = []
    for asset in assets:
        if asset.get("source_id") != "pubmed" or "elink_pmc" not in str(asset.get("group") or ""):
            continue
        path = Path(str(asset.get("storage_path") or ""))
        expected_pmids = expected_elink_pmids(asset)
        validation_error: str | None = None
        valid = False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            valid = isinstance(payload, dict) and elink_payload_covers_pmids(payload, expected_pmids)
            if not valid:
                validation_error = "source_error_or_incomplete_pmid_coverage"
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            validation_error = f"{type(exc).__name__}: {exc}"

        compound = str(asset.get("compound") or "")
        recovered = bool(expected_pmids) and set(expected_pmids).issubset(recovery_successes.get(compound, set()))
        if valid:
            status = "valid"
        elif recovered:
            status = "recovered_source_error_payload"
        else:
            status = "invalid_unrecovered"
        rows.append(
            {
                "asset_id": asset.get("asset_id"),
                "request_id": asset.get("request_id"),
                "compound": asset.get("compound"),
                "storage_path": asset.get("storage_path"),
                "expected_pmids": expected_pmids,
                "status": status,
                "validation_error": validation_error,
                "raw_response_preserved": True,
                "raw_only": True,
            }
        )

    status_counts = Counter(str(row["status"]) for row in rows)
    report_path = run_root / "finalization" / "pubmed_elink_payload_validation.jsonl"
    write_jsonl(report_path, rows)
    summary = {
        "path": str(report_path),
        "payload_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "unrecovered_payload_count": status_counts.get("invalid_unrecovered", 0),
        "raw_only": True,
    }
    return rows, summary


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_final_integrity(
    run_root: Path,
    assets: list[dict[str, Any]],
    object_plan: list[dict[str, Any]],
    osp_manifest: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    digest_by_path: dict[str, str] = {}
    asset_bytes = 0
    asset_ids = [str(asset.get("asset_id") or "") for asset in assets]
    storage_paths = [str(asset.get("storage_path") or "") for asset in assets]
    duplicate_asset_id_count = len(asset_ids) - len(set(asset_ids))
    duplicate_storage_path_count = len(storage_paths) - len(set(storage_paths))
    if duplicate_asset_id_count:
        issues.append({"type": "duplicate_asset_id", "count": duplicate_asset_id_count})
    if duplicate_storage_path_count:
        issues.append({"type": "duplicate_storage_path", "count": duplicate_storage_path_count})

    for asset in assets:
        storage_path = str(asset.get("storage_path") or "")
        path = Path(storage_path)
        if not path.is_file():
            issues.append({"type": "missing_asset_file", "asset_id": asset.get("asset_id"), "path": storage_path})
            continue
        size = path.stat().st_size
        asset_bytes += size
        if size != int(asset.get("byte_count") or -1):
            issues.append(
                {
                    "type": "asset_size_mismatch",
                    "asset_id": asset.get("asset_id"),
                    "path": storage_path,
                    "expected": asset.get("byte_count"),
                    "actual": size,
                }
            )
        digest = file_sha256(path)
        digest_by_path[storage_path] = digest
        if digest != str(asset.get("checksum") or ""):
            issues.append(
                {
                    "type": "asset_checksum_mismatch",
                    "asset_id": asset.get("asset_id"),
                    "path": storage_path,
                    "expected": asset.get("checksum"),
                    "actual": digest,
                }
            )
        metadata_path = Path(str(asset.get("metadata_path") or ""))
        if not metadata_path.is_file():
            issues.append(
                {"type": "missing_metadata_sidecar", "asset_id": asset.get("asset_id"), "path": str(metadata_path)}
            )
        else:
            try:
                json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                issues.append(
                    {
                        "type": "invalid_metadata_sidecar",
                        "asset_id": asset.get("asset_id"),
                        "path": str(metadata_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    assets_by_s3_key = {
        str((asset.get("metadata") or {}).get("s3_key")): asset
        for asset in assets
        if (asset.get("metadata") or {}).get("s3_key")
    }
    plan_checksum_compared_count = 0
    for item in object_plan:
        s3_key = str(item.get("s3_key") or "")
        asset = assets_by_s3_key.get(s3_key)
        if not asset:
            issues.append({"type": "missing_planned_pmc_object", "s3_key": s3_key, "url": item.get("url")})
            continue
        path_string = str(asset.get("storage_path") or "")
        path = Path(path_string)
        if path.is_file() and path.stat().st_size != int(item.get("byte_count") or -1):
            issues.append(
                {
                    "type": "planned_pmc_object_size_mismatch",
                    "s3_key": s3_key,
                    "expected": item.get("byte_count"),
                    "actual": path.stat().st_size,
                }
            )
        expected_digest = str(item.get("sha256") or item.get("checksum") or "")
        if expected_digest:
            plan_checksum_compared_count += 1
            if digest_by_path.get(path_string) != expected_digest:
                issues.append(
                    {
                        "type": "planned_pmc_object_checksum_mismatch",
                        "s3_key": s3_key,
                        "expected": expected_digest,
                        "actual": digest_by_path.get(path_string),
                    }
                )

    osp_assets = osp_manifest.get("assets") or []
    osp_checksums: list[str] = []
    for asset in osp_assets:
        path = Path(str(asset.get("storage_path") or ""))
        if not path.is_file():
            issues.append({"type": "missing_osp_snapshot_asset", "asset_id": asset.get("asset_id"), "path": str(path)})
            continue
        if path.stat().st_size != int(asset.get("byte_count") or -1):
            issues.append(
                {
                    "type": "osp_snapshot_asset_size_mismatch",
                    "asset_id": asset.get("asset_id"),
                    "path": str(path),
                }
            )
        digest = file_sha256(path)
        osp_checksums.append(digest)
        if digest != str(asset.get("checksum") or ""):
            issues.append(
                {
                    "type": "osp_snapshot_asset_checksum_mismatch",
                    "asset_id": asset.get("asset_id"),
                    "path": str(path),
                }
            )
    if osp_assets:
        aggregate = sha256("\n".join(sorted(osp_checksums)).encode("utf-8")).hexdigest()
        if aggregate != osp_manifest.get("asset_checksum_aggregate"):
            issues.append(
                {
                    "type": "osp_snapshot_aggregate_checksum_mismatch",
                    "expected": osp_manifest.get("asset_checksum_aggregate"),
                    "actual": aggregate,
                }
            )

    issue_counts = Counter(str(issue["type"]) for issue in issues)
    report = {
        "status": "passed" if not issues else "failed",
        "accepted": not issues,
        "checksum_algorithm": "sha256",
        "raw_asset_count": len(assets),
        "raw_asset_bytes": asset_bytes,
        "verified_raw_asset_count": len(assets) - sum(issue["type"] == "missing_asset_file" for issue in issues),
        "metadata_sidecar_count": len(assets),
        "pmc_plan_object_count": len(object_plan),
        "pmc_plan_checksum_compared_count": plan_checksum_compared_count,
        "osp_snapshot_asset_count": len(osp_assets),
        "issue_count": len(issues),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues": issues,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "raw_only": True,
    }
    report_path = run_root / "finalization" / "FINAL_INTEGRITY_AUDIT.json"
    write_json(report_path, report)
    report["path"] = str(report_path)
    return report


def completed_run_osp_snapshot(output_root: Path, run_id: str) -> Path | None:
    """A new compound run must never inherit another batch's model sources."""
    cache = output_root / "source_cache" / "osp_pbpk_model_library" / run_id
    for snapshot in (cache / "verified", cache):
        status = read_json(snapshot / "RUN_STATUS.json")
        if status.get("status") == "completed" and (snapshot / "snapshot_manifest.json").exists():
            return snapshot
    return None


def reconcile(run_root: Path, output_root: Path, compounds: list[str]) -> dict[str, Any]:
    finalization_root = run_root / "finalization"
    ensure_failure_resolution_ledger(run_root)
    assets = read_jsonl(run_root / "assets.jsonl")
    failures = read_jsonl(run_root / "failures.jsonl")
    resolutions = read_jsonl(finalization_root / "failure_resolutions.jsonl")
    object_plan = read_jsonl(finalization_root / "pmc_object_download_plan.jsonl")
    _, source_payload_validation = validate_pubmed_elink_payloads(run_root, assets)

    targets = target_pmcids_by_compound(run_root, compounds)
    pmcid_compounds: dict[str, set[str]] = defaultdict(set)
    for compound, pmcids in targets.items():
        for pmcid in pmcids:
            pmcid_compounds[pmcid].add(compound)
    for asset in assets:
        if asset.get("source_id") != "pmc_oa" or not asset.get("compound"):
            continue
        pmcid = pmcid_from_asset(asset)
        if pmcid:
            pmcid_compounds[pmcid].add(str(asset["compound"]))

    links: list[dict[str, Any]] = []
    for plan_index, item in enumerate(object_plan, start=1):
        pmcid = str(item.get("pmcid") or "")
        compounds_for_object = sorted(pmcid_compounds.get(pmcid) or {None}, key=lambda value: value or "")
        object_id = sha256(
            json.dumps({"pmcid": pmcid, "s3_key": item.get("s3_key"), "url": item.get("url")}, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        for compound in compounds_for_object:
            links.append(
                {
                    "object_plan_index": plan_index,
                    "object_id": f"pmc-object:{object_id}",
                    "compound": compound,
                    "pmcid": pmcid or None,
                    "source_id": item.get("source_id"),
                    "s3_key": item.get("s3_key"),
                    "url": item.get("url"),
                    "byte_count": item.get("byte_count"),
                    "status": item.get("status"),
                    "reason": item.get("reason"),
                    "origin": item.get("origin"),
                    "raw_only": True,
                }
            )
    links_path = finalization_root / "pmc_object_compound_links.jsonl"
    write_jsonl(links_path, links)

    plan_by_compound: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        if link.get("compound"):
            plan_by_compound[str(link["compound"])].append(link)
    physical_s3_assets = [
        asset
        for asset in assets
        if asset.get("source_id") == "pmc_oa" and str(asset.get("group") or "").startswith("pmc/oa_s3_files/")
    ]
    canonical_static_assets = [
        asset
        for asset in assets
        if asset.get("source_id") == "pmc_oa" and str(asset.get("group") or "").startswith("pmc/oa_s3_objects/")
    ]
    physical_s3_assets.extend(canonical_static_assets)
    physical_assets_by_key: dict[str, dict[str, Any]] = {}
    for asset in physical_s3_assets:
        path = Path(str(asset.get("storage_path") or ""))
        key = pmc_object_key_from_asset(asset)
        if not key or not path.is_file() or path.stat().st_size != int(asset.get("byte_count") or -1):
            continue
        physical_assets_by_key[key] = asset
    physical_object_keys = set(physical_assets_by_key)
    physical_package_assets = [
        asset
        for asset in assets
        if asset.get("source_id") == "pmc_oa" and str(asset.get("group") or "").startswith("pmc/oa_files/")
    ]

    object_plan_by_pmcid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in object_plan:
        object_plan_by_pmcid[str(item.get("pmcid") or "")].append(item)
    efetch_pmcids_by_compound: dict[str, set[str]] = defaultdict(set)
    oa_metadata_by_compound_pmcid: dict[tuple[str, str], dict[str, Any]] = {}
    cloud_absence_by_compound_pmcid: dict[tuple[str, str], dict[str, Any]] = {}
    package_pmcids_by_compound: dict[str, set[str]] = defaultdict(set)
    for asset in assets:
        compound = str(asset.get("compound") or "")
        if asset.get("source_id") != "pmc_oa" or not compound:
            continue
        group = str(asset.get("group") or "")
        if group.startswith("pmc/efetch_xml"):
            efetch_pmcids_by_compound[compound].update(pmcids_from_efetch_asset(asset))
        pmcid = pmcid_from_asset(asset)
        if not pmcid:
            continue
        if group.startswith("pmc/oa_metadata/"):
            oa_metadata_by_compound_pmcid[(compound, pmcid)] = asset
        if group.startswith("pmc/oa_s3_listing/") and confirmed_absent_from_public_cloud(asset, pmcid):
            cloud_absence_by_compound_pmcid[(compound, pmcid)] = asset
        if group.startswith("pmc/oa_files/") and str(asset.get("request_key") or "").endswith("_tgz"):
            path = Path(str(asset.get("storage_path") or ""))
            if path.is_file() and path.stat().st_size == int(asset.get("byte_count") or -1):
                package_pmcids_by_compound[compound].add(pmcid)

    article_coverage: list[dict[str, Any]] = []
    for compound in compounds:
        for pmcid in sorted(targets.get(compound, set())):
            plan_rows = object_plan_by_pmcid.get(pmcid, [])
            plan_keys = [key for row in plan_rows if (key := pmc_object_key_from_plan(row))]
            static_complete = bool(plan_keys) and all(key in physical_object_keys for key in plan_keys)
            package_complete = pmcid in package_pmcids_by_compound[compound]
            efetch_complete = pmcid in efetch_pmcids_by_compound[compound]
            oa_status = oa_metadata_has_links(oa_metadata_by_compound_pmcid.get((compound, pmcid), {}))
            cloud_absence = cloud_absence_by_compound_pmcid.get((compound, pmcid))
            if oa_status is True:
                complete = package_complete or static_complete
            elif oa_status is False:
                complete = efetch_complete
            else:
                complete = package_complete or static_complete
            if cloud_absence and not plan_keys:
                # Completion means the availability decision is audited, not
                # that a PDF/full article was downloaded or scientific evidence exists.
                complete = True
            modes = [
                mode
                for mode, enabled in (
                    ("complete_oa_package", package_complete),
                    ("complete_s3_static_objects", static_complete),
                    ("pmc_efetch_xml", efetch_complete),
                )
                if enabled
            ]
            article_coverage.append(
                {
                    "compound": compound,
                    "pmcid": pmcid,
                    "open_access": oa_status,
                    "availability": "not_in_public_cloud_dataset" if cloud_absence else "public_objects_or_legacy_metadata",
                    "availability_asset_id": cloud_absence.get("asset_id") if cloud_absence else None,
                    "full_article_files_complete": package_complete or static_complete,
                    "coverage_modes": modes,
                    "planned_static_object_count": len(plan_keys),
                    "verified_static_object_count": sum(key in physical_object_keys for key in plan_keys),
                    "complete": complete,
                    "raw_only": True,
                }
            )
    article_coverage_path = finalization_root / "pmc_article_coverage.jsonl"
    write_jsonl(article_coverage_path, article_coverage)
    incomplete_articles = [row for row in article_coverage if not row["complete"]]

    per_compound: dict[str, dict[str, Any]] = {}
    for compound in compounds:
        target_pmcids = targets.get(compound, set())
        compound_assets = [asset for asset in assets if asset.get("compound") == compound and asset.get("source_id") == "pmc_oa"]
        listing_count = sum(1 for asset in compound_assets if str(asset.get("group") or "").startswith("pmc/oa_s3_listing/"))
        metadata_count = sum(1 for asset in compound_assets if str(asset.get("group") or "").startswith("pmc/oa_metadata/"))
        object_links = plan_by_compound.get(compound, [])
        completed_links = [link for link in object_links if pmc_object_key_from_plan(link) in physical_object_keys]
        pending_links = [link for link in object_links if pmc_object_key_from_plan(link) not in physical_object_keys]
        completed_keys = {key for link in completed_links if (key := pmc_object_key_from_plan(link))}
        actual_s3 = [physical_assets_by_key[key] for key in completed_keys]
        actual_packages = [asset for asset in physical_package_assets if asset.get("compound") == compound]
        payload = {
            "compound": compound,
            "pmcid_count_targeted_for_finalization": len(target_pmcids),
            "pmcids_targeted_for_finalization": sorted(target_pmcids),
            "oa_metadata_asset_count": metadata_count,
            "s3_listing_asset_count": listing_count,
            "oa_package_asset_count": len(actual_packages),
            "s3_object_asset_count": len(actual_s3),
            "s3_object_asset_bytes": sum(int(asset.get("byte_count") or 0) for asset in actual_s3),
            "planned_object_count": len(object_links),
            "completed_object_count": len(completed_links),
            "remaining_object_count": len(pending_links),
            "remaining_object_bytes": sum(int(link.get("byte_count") or 0) for link in pending_links),
            "object_compound_links_path": str(links_path),
            "physical_object_transfer_complete": len(pending_links) == 0,
            "article_coverage_complete": not any(
                row["compound"] == compound and not row["complete"] for row in article_coverage
            ),
            "raw_only": True,
            "reconciled_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(finalization_root / "pmc" / f"{slugify(compound)}.json", payload)
        per_compound[compound] = payload

    pending_object_plan = [item for item in object_plan if pmc_object_key_from_plan(item) not in physical_object_keys]
    extension_counts = Counter(Path(str(item.get("s3_key") or item.get("url") or "")).suffix.lower() or "unknown" for item in object_plan)
    total_planned_bytes = sum(int(item.get("byte_count") or 0) for item in object_plan)
    remaining_object_bytes = sum(int(item.get("byte_count") or 0) for item in pending_object_plan)
    plan_summary = {
        "object_count": len(object_plan),
        "remaining_object_count": len(pending_object_plan),
        "completed_object_count": len(object_plan) - len(pending_object_plan),
        "object_bytes": total_planned_bytes,
        "remaining_object_bytes": remaining_object_bytes,
        "object_compound_link_count": len(links),
        "unlinked_object_count": sum(1 for item in object_plan if not pmcid_compounds.get(str(item.get("pmcid") or ""))),
        "by_extension": dict(sorted(extension_counts.items())),
        "physical_s3_object_asset_count": len(physical_s3_assets),
        "physical_s3_object_asset_bytes": sum(int(asset.get("byte_count") or 0) for asset in physical_s3_assets),
        "plan_path": str(finalization_root / "pmc_object_download_plan.jsonl"),
        "compound_links_path": str(links_path),
        "article_coverage_path": str(article_coverage_path),
        "article_compound_link_count": len(article_coverage),
        "unique_article_count": len({row["pmcid"] for row in article_coverage}),
        "incomplete_article_count": len(incomplete_articles),
        "raw_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(finalization_root / "pmc_object_download_plan_summary.json", plan_summary)

    osp_snapshot = completed_run_osp_snapshot(output_root, run_root.name)
    shared_limits = read_json(run_root / "download_plan.json").get("limits", {})
    osp_required = (shared_limits.get("include_shared_sources") is True
                    and "osp_pbpk_model_library" in shared_limits.get("shared_source_ids", []))
    osp_manifest = read_json(osp_snapshot / "snapshot_manifest.json") if osp_snapshot else {}
    integrity_audit = validate_final_integrity(run_root, assets, object_plan, osp_manifest)
    covered_pmcids = {str(row["pmcid"]) for row in article_coverage if row["complete"]}
    for values in package_pmcids_by_compound.values():
        covered_pmcids.update(values)
    for pmcid, rows in object_plan_by_pmcid.items():
        keys = [key for row in rows if (key := pmc_object_key_from_plan(row))]
        if keys and all(key in physical_object_keys for key in keys):
            covered_pmcids.add(pmcid)

    existing_resolution_ids = {
        str(row.get("original_failure_request_id") or "") for row in resolutions if row.get("original_failure_request_id")
    }
    for failure in failures:
        request_id = str(failure.get("request_id") or "")
        if not request_id or request_id in existing_resolution_ids:
            continue
        source_id = str(failure.get("source_id") or "")
        pmcid = pmcid_from_failure(failure)
        method: str | None = None
        detail: dict[str, Any] = {}
        if source_id == "pmc_oa" and pmcid in covered_pmcids:
            method = "reconciled_verified_pmc_raw_coverage"
            detail = {"pmcid": pmcid, "coverage_manifest": str(article_coverage_path)}
        elif source_id == "osp_pbpk_model_library" and osp_snapshot:
            method = "reconciled_verified_osp_repository_snapshot"
            detail = {
                "snapshot_root": str(osp_snapshot),
                "snapshot_manifest": str(osp_snapshot / "snapshot_manifest.json"),
                "aggregate_checksum": osp_manifest.get("asset_checksum_aggregate"),
            }
        if not method:
            continue
        resolution = {
            "resolution_id": f"resolution:reconcile:{sha256(request_id.encode('utf-8')).hexdigest()[:20]}",
            "original_failure_request_id": request_id,
            "source_id": source_id,
            "compound": failure.get("compound"),
            "original_request_key": failure.get("request_key"),
            "original_url": failure.get("url"),
            "status": "resolved",
            "resolution_method": method,
            "detail": detail,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "raw_only": True,
        }
        append_jsonl(finalization_root / "failure_resolutions.jsonl", resolution)
        resolutions.append(resolution)
        existing_resolution_ids.add(request_id)

    resolved_failure_ids = {
        str(row.get("original_failure_request_id") or "") for row in resolutions if row.get("original_failure_request_id")
    }
    unresolved_failures = [
        row for row in failures if str(row.get("request_id") or "") not in resolved_failure_ids
    ]
    unique_failure_request_count = len({str(row.get("request_id") or "") for row in failures})
    resolution_status_counts = Counter(str(row.get("status") or "unknown") for row in resolutions)
    accepted = (
        not pending_object_plan
        and not incomplete_articles
        and not unresolved_failures
        and source_payload_validation["unrecovered_payload_count"] == 0
        and integrity_audit["accepted"]
        and (not osp_required or osp_snapshot is not None)
    )
    result = {
        "run_id": run_root.name,
        "status": "completed" if accepted else "incomplete",
        "accepted_as_complete_raw_content": accepted,
        "missing_requested_shared_sources": ["osp_pbpk_model_library"] if osp_required and osp_snapshot is None else [],
        "run_root": str(run_root),
        "raw_asset_count": len(assets),
        "historical_failure_count": len(failures),
        "historical_failure_record_count": len(failures),
        "historical_failure_unique_request_count": unique_failure_request_count,
        "duplicate_historical_failure_record_count": len(failures) - unique_failure_request_count,
        "failure_resolution_count": len(resolutions),
        "failure_resolution_status_counts": dict(sorted(resolution_status_counts.items())),
        "remaining_unresolved_failure_count": len(unresolved_failures),
        "remaining_unresolved_failures": [
            {
                "request_id": row.get("request_id"),
                "source_id": row.get("source_id"),
                "compound": row.get("compound"),
                "request_key": row.get("request_key"),
                "error": row.get("error"),
            }
            for row in unresolved_failures
        ],
        "pmc_object_transfer": plan_summary,
        "pmc_article_coverage": {
            "path": str(article_coverage_path),
            "article_compound_link_count": len(article_coverage),
            "unique_article_count": len({row["pmcid"] for row in article_coverage}),
            "incomplete_article_count": len(incomplete_articles),
        },
        "source_payload_validation": source_payload_validation,
        "integrity_audit": integrity_audit,
        "pmc_by_compound": per_compound,
        "osp_pbpk_model_library_snapshot": {
            "snapshot_root": str(osp_snapshot) if osp_snapshot else None,
            "snapshot_manifest": str(osp_snapshot / "snapshot_manifest.json") if osp_snapshot else None,
            "asset_count": osp_manifest.get("asset_count"),
            "aggregate_checksum": osp_manifest.get("asset_checksum_aggregate"),
        },
        "statement": (
            "The run is incomplete; inspect remaining_unresolved_failures, incomplete PMC articles, and pending static objects."
            if not accepted
            else "All selected source assets and PMC article-level raw coverage were physically transferred and validated."
        ),
        "raw_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = finalization_root / "FINAL_ACQUISITION_RESULT.json"
    write_json(result_path, result)

    summary_path = run_root / "summary.json"
    summary = read_json(summary_path)
    source_coverage: dict[str, dict[str, int]] = defaultdict(lambda: {"assets": 0, "bytes": 0, "failures": 0})
    for asset in assets:
        source_id = str(asset.get("source_id") or "unknown")
        source_coverage[source_id]["assets"] += 1
        source_coverage[source_id]["bytes"] += int(asset.get("byte_count") or 0)
    for failure in failures:
        source_id = str(failure.get("source_id") or "unknown")
        source_coverage[source_id]["failures"] += 1
    summary.update(
        {
            "completion_status": result["status"],
            "accepted_as_complete_raw_content": result["accepted_as_complete_raw_content"],
            "asset_count": len(assets),
            "failure_count": len(failures),
            "failure_unique_request_count": unique_failure_request_count,
            "duplicate_failure_record_count": len(failures) - unique_failure_request_count,
            "failure_resolution_count": len(resolutions),
            "failure_resolution_status_counts": dict(sorted(resolution_status_counts.items())),
            "remaining_unresolved_historical_failure_count": len(unresolved_failures),
            "final_acquisition_result_path": str(result_path),
            "pmc_object_download_plan_count": len(object_plan),
            "pmc_object_transfer": plan_summary,
            "pmc_article_coverage": result["pmc_article_coverage"],
            "source_payload_validation": source_payload_validation,
            "integrity_audit": integrity_audit,
            "osp_pbpk_model_library_snapshot": result["osp_pbpk_model_library_snapshot"],
            "coverage_by_source": {key: value for key, value in sorted(source_coverage.items())},
        }
    )
    write_json(summary_path, summary)
    write_markdown_summary(run_root / "README.md", summary)
    write_json(output_root / "latest.json", summary)

    status_path = run_root / "RUN_STATUS.json"
    status = read_json(status_path)
    status.update(
        {
            "status": result["status"],
            "accepted_as_final_result": accepted,
            "message": result["statement"],
            "final_acquisition_result_path": str(result_path),
            "summary_path": str(summary_path),
            "integrity_audit_path": integrity_audit["path"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(status_path, status)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reconciled final acquisition result.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compounds", default=",".join(DEFAULT_COMPOUNDS))
    args = parser.parse_args()
    compounds = [item.strip() for item in args.compounds.split(",") if item.strip()]
    print(json.dumps(reconcile(args.run_root, args.output_root, compounds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
