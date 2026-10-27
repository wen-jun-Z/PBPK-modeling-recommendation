"""Register an exact-version official SPL archive after a literal ``None`` response.

This is an offline, provenance-preserving repair. The caller must pause the
pipeline before changing its canonical asset ledger; no parser output is edited.
"""
from __future__ import annotations

import copy
import fcntl
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from urllib.parse import parse_qs, urlsplit
from uuid import UUID
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from pbpk_kb.paths import PROJECT_ROOT
from pbpk_kb.acquisition.comprehensive_raw_retriever import iso_now

REASON = "invalid_response_recovered_from_exact_version_archive"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 256
NS = "{urn:hl7-org:v3}"


def _path(value, root: Path) -> Path:
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("DailyMed recovery path escaped the project")
    return path


def _json_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, contents: str) -> None:
    temporary = path.with_name(path.name + ".recovery.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _json_write(path: Path, value: dict) -> None:
    _atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _archive_url(url: str, setid: str, version: int) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "dailymed.nlm.nih.gov"
            or parsed.path != "/dailymed/getFile.cfm" or parsed.fragment
            or parse_qs(parsed.query, keep_blank_values=True) != {
                "type": ["zip"], "setid": [setid], "version": [str(version)]}):
        raise ValueError("Archive URL is not the exact official setId/version URL")


def _validated_archive(path: Path, setid: str, version: int) -> dict:
    total = 0
    documents = []
    names = set()
    with ZipFile(path) as archive:
        members = archive.infolist()
        if not members or len(members) > MAX_MEMBERS:
            raise ValueError("Archive member count exceeds the bounded recovery limit")
        for info in members:
            name = info.filename
            parts = PurePosixPath(name).parts
            mode = info.external_attr >> 16
            if (not name or "\\" in name or "\x00" in name or ":" in name
                    or PurePosixPath(name).is_absolute() or ".." in parts
                    or name in names or stat.S_ISLNK(mode) or info.flag_bits & 1):
                raise ValueError("Unsafe or duplicate archive member")
            names.add(name)
            total += info.file_size
            if info.file_size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("Archive expanded size exceeds the bounded recovery limit")
            if info.is_dir():
                continue
            # Reading every bounded member validates its CRC as well as size.
            contents = archive.read(info)
            if len(contents) != info.file_size:
                raise ValueError("Archive member size mismatch")
            if not name.lower().endswith(".xml"):
                continue
            if b"<!DOCTYPE" in contents.upper() or b"<!ENTITY" in contents.upper():
                raise ValueError("XML declarations are not supported in recovery")
            document = ET.fromstring(contents)
            if document.tag != NS + "document":
                raise ValueError("Archive XML is not a main HL7 SPL document")
            doc_id = document.find(NS + "id")
            doc_setid = document.find(NS + "setId")
            doc_version = document.find(NS + "versionNumber")
            if (doc_id is None or doc_setid is None or doc_version is None
                    or doc_setid.get("root") != setid
                    or doc_version.get("value") != str(version)):
                raise ValueError("Archive SPL identity or version mismatch")
            document_id = str(UUID(doc_id.get("root", "")))
            documents.append({"member_path": name, "member_sha256": sha256(contents).hexdigest(),
                              "member_bytes": len(contents), "document_id": document_id,
                              "setid": setid, "spl_version": version})
    if len(documents) != 1:
        raise ValueError("Expected exactly one matching main SPL XML")
    return {**documents[0], "archive_member_count": len(members), "archive_expanded_bytes": total}


def register_verified_archive_replacement(raw_run, original_asset_id, probe_report, root=PROJECT_ROOT):
    """Validate and register downloaded bytes; return a durable recovery receipt.

    Original response bytes and metadata sidecar remain untouched. The canonical
    ledger records the invalid response as metadata only and appends a new asset
    for the actual official ZIP URL. A before-ledger snapshot and a sidecar copy
    are retained. Repeating this call verifies and returns the same receipt.
    """
    root = Path(root).resolve()
    raw_run = _path(raw_run, root)
    ledger = raw_run / "assets.jsonl"
    report_path = _path(probe_report, root)
    report = json.loads(report_path.read_text())
    # Do not create recovery files before basic input validation succeeds.
    assets = _json_rows(ledger)
    matching = [asset for asset in assets if asset.get("asset_id") == original_asset_id]
    if len(matching) != 1:
        raise ValueError("Original asset must occur exactly once")
    original_current = matching[0]
    metadata = original_current.get("metadata", {})
    setid = str(UUID(metadata.get("setid", "")))
    version = metadata.get("spl_version")
    compound = original_current.get("compound", "")
    if (not isinstance(version, int) or isinstance(version, bool) or version < 1
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", compound)):
        raise ValueError("Invalid original compound or SPL version")
    expected_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
    if (original_current.get("source_id") != "dailymed"
            or original_current.get("group") != "dailymed/spl_xml"
            or original_current.get("url") != expected_url
            or original_current.get("request_key") != f"spl_xml_{setid}"
            or original_current.get("status_code") != 200
            or not str(original_current.get("request_id", "")).startswith(f"dailymed:spl_xml_{setid}:")):
        raise ValueError("Original asset source identity mismatch")
    invalid_path = _path(original_current["storage_path"], root)
    sidecar = _path(original_current["metadata_path"], root)
    if not invalid_path.is_relative_to(raw_run) or not sidecar.is_relative_to(raw_run):
        raise ValueError("Original asset must be inside its raw run")
    if (invalid_path.stat().st_size != 4 or invalid_path.read_bytes() != b"None"
            or original_current.get("byte_count") != 4
            or _digest(invalid_path) != original_current.get("checksum")):
        raise ValueError("Original asset is not the verified literal None response")
    original = json.loads(sidecar.read_text())
    if original.get("asset_id") != original_asset_id:
        raise ValueError("Original sidecar identity mismatch")
    if report.get("status_code") != 200:
        raise ValueError("Archive retrieval was not HTTP 200")
    for name in ("url", "final_url"):
        _archive_url(report.get(name, ""), setid, version)
    archive_path = _path(report["path"], root)
    archive_bytes = archive_path.stat().st_size
    archive_sha = _digest(archive_path)
    if (not 0 < archive_bytes <= MAX_ARCHIVE_BYTES or archive_bytes != report.get("bytes")
            or archive_sha != report.get("sha256")):
        raise ValueError("Archive byte count or SHA256 mismatch")
    member = _validated_archive(archive_path, setid, version)
    selection = raw_run / "compounds" / compound / "selection/dailymed/fulltext_selected_records.jsonl"
    selected = [item for item in _json_rows(selection)
                if item.get("source_record_id") == setid]
    if len(selected) != 1:
        raise ValueError("Expected one original selected SPL record")
    selected = selected[0]
    if (selected.get("source_id") != "dailymed" or selected.get("compound") != compound
            or selected.get("decision") != "fulltext_selected"
            or selected.get("record", {}).get("setid") != setid
            or selected.get("record", {}).get("spl_version") != version
            or selected.get("record", {}).get("title") != metadata.get("title")):
        raise ValueError("Original selected SPL identity/version mismatch")
    recovery_key = sha256((original_asset_id + archive_sha).encode()).hexdigest()[:32]
    work = raw_run / "finalization/dailymed_recovery" / recovery_key
    work.mkdir(parents=True, exist_ok=True)
    with (work.parent / "registration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _commit(ledger, original, original_current, sidecar, archive_path, report,
                       member, selection, work, recovery_key, root)


def _commit(ledger, original, original_current, sidecar, archive_path, report,
            member, selection, work, recovery_key, root):
    receipt_path = work / "receipt.json"
    old_ledger = work / "assets.before.jsonl"
    old_sidecar = work / "original.metadata.json"
    replacement_id = "asset:dailymed-recovered:" + recovery_key
    amended = copy.deepcopy(original)
    amended.update(processing_policy="metadata_only", processing_policy_reason=REASON)
    amended.setdefault("metadata", {}).update(recovered_by_asset_id=replacement_id,
                                             recovery_receipt=str(receipt_path))
    if original_current not in (original, amended):
        raise ValueError("Original ledger and immutable sidecar disagree")
    assets = _json_rows(ledger)
    # Revalidate after taking the lock, so concurrent/repeated registration is safe.
    current = [item for item in assets if item.get("asset_id") == original["asset_id"]]
    if len(current) != 1 or current[0] not in (original, amended):
        raise ValueError("Original ledger changed during recovery")
    destination = ledger.parent / "compounds" / original["compound"] / "dailymed/spl_archive/recovery" / (
            f"{member['setid']}_v{member['spl_version']}_{report['sha256'][:12]}_{recovery_key[:8]}.zip")
    destination = _path(destination, root)
    replacement = {
        "asset_id": replacement_id, "request_id": "dailymed:archive_recovery:" + recovery_key,
        "request_key": f"spl_archive_{member['setid']}_v{member['spl_version']}",
        "source_id": "dailymed", "compound": original["compound"],
        "group": "dailymed/spl_archive/recovery", "url": report["url"],
        "status_code": 200, "content_type": "application/zip", "raw_only": True,
        "storage_scope": "compound_run", "storage_path": str(destination),
        "metadata_path": str(destination.with_suffix(".zip.metadata.json")),
        "byte_count": report["bytes"], "checksum": report["sha256"],
        "metadata": {**member, "title": original["metadata"].get("title"),
                     "replaces_invalid_asset_id": original["asset_id"],
                     "recovery_receipt": str(receipt_path), "recovery_reason": REASON},
    }
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("original_asset") != original or receipt.get("replacement_asset_id") != replacement_id
                or receipt.get("archive_sha256") != report["sha256"]
                or receipt.get("selection_sha256") != _digest(selection)
                or receipt.get("original_ledger_sha256") != _digest(old_ledger)
                or old_sidecar.read_bytes() != sidecar.read_bytes()):
            raise ValueError("Existing recovery receipt or preserved provenance changed")
    else:
        if current[0] != original:
            raise ValueError("Cannot accept amended ledger without a recovery receipt")
        # Existing snapshots may be from an interrupted pre-commit attempt.
        for source, target in ((ledger, old_ledger), (sidecar, old_sidecar)):
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError("Existing recovery backup differs from original")
            if not target.exists():
                shutil.copyfile(source, target)
        receipt = {"status": "prepared", "created_at": iso_now(), "method": REASON,
                   "original_asset": original, "replacement_asset_id": replacement_id,
                   "archive_sha256": report["sha256"], "archive_url": report["url"],
                   "original_ledger_sha256": _digest(old_ledger), "selection_sha256": _digest(selection),
                   "original_ledger_backup": str(old_ledger), "original_sidecar_backup": str(old_sidecar),
                   "verified_member": member}
        _json_write(receipt_path, receipt)
    replacement["downloaded_at"] = receipt["created_at"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _digest(destination) != report["sha256"]:
            raise ValueError("Recovery destination changed")
    else:
        shutil.copyfile(archive_path, destination)
    replacement_sidecar = Path(replacement["metadata_path"])
    if replacement_sidecar.exists():
        if json.loads(replacement_sidecar.read_text()) != replacement:
            raise ValueError("Replacement sidecar changed")
    else:
        _json_write(replacement_sidecar, replacement)
    existing = [item for item in assets if item.get("asset_id") == replacement_id]
    if existing and existing != [replacement]:
        raise ValueError("Replacement ledger identity collision")
    if not existing or current[0] != amended:
        updated = [amended if item.get("asset_id") == original["asset_id"] else item for item in assets]
        if not existing:
            updated.append(replacement)
        _atomic(ledger, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in updated))
    if receipt.get("status") != "completed":
        receipt["status"] = "completed"
        _json_write(receipt_path, receipt)
    return receipt
