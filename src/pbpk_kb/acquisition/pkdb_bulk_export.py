from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import stat
import tempfile
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile, ZipInfo


PKDB_FILTER_URL = "https://pk-db.com/api/v1/filter/"
USER_AGENT = "pbpk-public-kb/0.1 (public research data acquisition)"
EXPECTED_TABLES = {
    "studies": ("studies.csv", "sid"),
    "groups": ("groups.csv", "group_pk"),
    "individuals": ("individuals.csv", "individual_pk"),
    "interventions": ("interventions.csv", "intervention_pk"),
    "outputs": ("outputs.csv", "output_pk"),
    "timecourses": ("timecourses.csv", "subset_pk"),
    "scatters": ("scatters.csv", "subset_pk"),
}
REQUIRED_MEMBERS = {
    *(table[0] for table in EXPECTED_TABLES.values()),
    "info_nodes.csv",
    "README.md",
    "TERMS_OF_USE.md",
}


class PKDBBulkExportError(RuntimeError):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _is_symlink(member: ZipInfo) -> bool:
    return stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK


def _validated_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts:
        raise PKDBBulkExportError(f"Unsafe absolute or empty ZIP member path: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise PKDBBulkExportError(f"Unsafe ZIP member traversal path: {name!r}")
    if ":" in path.parts[0]:
        raise PKDBBulkExportError(f"Unsafe drive-qualified ZIP member path: {name!r}")
    return path


def _copy_and_hash(source: BinaryIO, destination: Path) -> str:
    digest = sha256()
    with destination.open("wb") as output:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            output.write(block)
    return digest.hexdigest()


def _csv_unique_count(path: Path, key: str) -> dict[str, Any]:
    row_count = 0
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(field or "") for field in (reader.fieldnames or [])]
        if key not in fieldnames:
            return {
                "row_count": 0,
                "unique_object_count": 0,
                "key": key,
                "key_present": False,
                "fieldnames": fieldnames,
            }
        for row in reader:
            row_count += 1
            value = str(row.get(key) or "").strip()
            if value:
                identifiers.add(value)
    return {
        "row_count": row_count,
        "unique_object_count": len(identifiers),
        "key": key,
        "key_present": True,
        "fieldnames": fieldnames,
    }


def inspect_pkdb_bulk_export(
    *,
    archive_path: Path,
    output_directory: Path,
    filter_payload: dict[str, Any],
    compound: str,
    source_url: str,
    response_headers: dict[str, str] | None = None,
    archive_reference: str | None = None,
    max_members: int = 1_000,
    max_uncompressed_bytes: int = 4 * 1024**3,
    max_compression_ratio: float = 2_000.0,
) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise PKDBBulkExportError(f"Immutable inspection directory already exists: {output_directory}")
    if not archive_path.is_file():
        raise PKDBBulkExportError(f"PK-DB ZIP does not exist: {archive_path}")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent))
    try:
        extracted_root = temporary / "members"
        extracted_root.mkdir()
        inventory: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        try:
            with ZipFile(archive_path) as archive:
                members = archive.infolist()
                if len(members) > max_members:
                    raise PKDBBulkExportError(
                        f"PK-DB ZIP has {len(members)} members; maximum is {max_members}."
                    )
                total_uncompressed = sum(member.file_size for member in members)
                if total_uncompressed > max_uncompressed_bytes:
                    raise PKDBBulkExportError(
                        f"PK-DB ZIP expands to {total_uncompressed} bytes; maximum is {max_uncompressed_bytes}."
                    )
                for member in members:
                    relative = _validated_member_path(member.filename)
                    casefolded = relative.as_posix().casefold()
                    if casefolded in seen_names:
                        raise PKDBBulkExportError(f"Duplicate ZIP member path: {member.filename!r}")
                    seen_names.add(casefolded)
                    if _is_symlink(member):
                        raise PKDBBulkExportError(f"ZIP symlink is not allowed: {member.filename!r}")
                    if member.file_size and member.compress_size:
                        ratio = member.file_size / member.compress_size
                        if ratio > max_compression_ratio:
                            raise PKDBBulkExportError(
                                f"ZIP member compression ratio is too high ({ratio:.1f}): {member.filename!r}"
                            )
                    destination = extracted_root.joinpath(*relative.parts)
                    if member.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source:
                        checksum = _copy_and_hash(source, destination)
                    inventory.append(
                        {
                            "member_path": relative.as_posix(),
                            "byte_count": member.file_size,
                            "compressed_byte_count": member.compress_size,
                            "sha256": checksum,
                        }
                    )
        except BadZipFile as exc:
            raise PKDBBulkExportError(f"Invalid PK-DB ZIP: {archive_path}") from exc

        present_members = {row["member_path"] for row in inventory}
        missing_members = sorted(REQUIRED_MEMBERS - present_members)
        table_audits: dict[str, dict[str, Any]] = {}
        count_mismatches: list[dict[str, Any]] = []
        for resource, (filename, key) in EXPECTED_TABLES.items():
            expected = int(filter_payload.get(resource) or 0)
            table_path = extracted_root / filename
            if not table_path.is_file():
                audit = {
                    "filename": filename,
                    "key": key,
                    "expected_unique_object_count": expected,
                    "actual_unique_object_count": 0,
                    "status": "missing_table",
                }
            else:
                details = _csv_unique_count(table_path, key)
                actual = int(details["unique_object_count"])
                audit = {
                    "filename": filename,
                    **details,
                    "expected_unique_object_count": expected,
                    "actual_unique_object_count": actual,
                    "status": "matched" if actual == expected else "count_mismatch",
                }
            table_audits[resource] = audit
            if audit["status"] != "matched":
                count_mismatches.append(
                    {
                        "resource": resource,
                        "expected": expected,
                        "actual": audit["actual_unique_object_count"],
                        "status": audit["status"],
                    }
                )

        gap_codes: list[str] = []
        if missing_members:
            gap_codes.append("required_export_members_missing")
        for resource in ("outputs", "timecourses"):
            if int(filter_payload.get(resource) or 0) > 0 and table_audits[resource]["actual_unique_object_count"] == 0:
                gap_codes.append(f"{resource}_observation_table_missing")
        if count_mismatches:
            gap_codes.append("filter_export_count_mismatch")

        accepted = not missing_members and not count_mismatches
        audit = {
            "schema_version": "1.0.0",
            "created_at": iso_now(),
            "source_id": "pkdb",
            "compound": compound,
            "source_url": source_url,
            "archive_path": archive_reference or str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "archive_byte_count": archive_path.stat().st_size,
            "response_headers": response_headers or {},
            "filter_payload": filter_payload,
            "member_count": len(inventory),
            "members": sorted(inventory, key=lambda row: row["member_path"].casefold()),
            "missing_required_members": missing_members,
            "table_audits": table_audits,
            "count_mismatches": count_mismatches,
            "gap_codes": sorted(set(gap_codes)),
            "accepted_for_observation_processing": accepted,
            "raw_only": True,
        }
        audit["audit_sha256"] = sha256(
            json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        write_json(temporary / "export_audit.json", audit)
        temporary.replace(output_directory)
        return audit
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _download(
    *,
    url: str,
    destination: Path,
    timeout_seconds: int,
    retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        part = destination.with_suffix(destination.suffix + ".part")
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            digest = sha256()
            byte_count = 0
            with urlopen(request, timeout=timeout_seconds) as response, part.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    byte_count += len(block)
                    output.write(block)
                headers = {str(key): str(value) for key, value in response.headers.items()}
                status_code = int(getattr(response, "status", 200) or 200)
            os.replace(part, destination)
            return {
                "url": url,
                "status_code": status_code,
                "headers": headers,
                "byte_count": byte_count,
                "sha256": digest.hexdigest(),
                "downloaded_at": iso_now(),
            }
        except Exception as exc:
            last_error = exc
            if part.exists():
                part.unlink()
            if attempt < retries - 1:
                time.sleep(2 + attempt)
    raise PKDBBulkExportError(f"Failed to download {url}: {last_error}") from last_error


def acquire_pkdb_bulk_export(
    *,
    compound: str,
    output_directory: Path,
    timeout_seconds: int = 1_800,
    open_licence_only: bool = True,
) -> dict[str, Any]:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise PKDBBulkExportError(f"Immutable acquisition directory already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent))
    try:
        params: dict[str, str] = {
            "format": "json",
            "studies__substance": compound,
            "concise": "false",
        }
        if open_licence_only:
            params["studies__licence__in"] = "open"
        filter_url = f"{PKDB_FILTER_URL}?{urlencode(params)}"
        filter_path = temporary / "filter_response.json"
        filter_download = _download(
            url=filter_url,
            destination=filter_path,
            timeout_seconds=timeout_seconds,
        )
        filter_payload = json.loads(filter_path.read_text(encoding="utf-8"))
        for resource in EXPECTED_TABLES:
            if resource not in filter_payload:
                raise PKDBBulkExportError(f"PK-DB filter response is missing count: {resource}")

        export_params = {**params, "download": "true"}
        export_url = f"{PKDB_FILTER_URL}?{urlencode(export_params)}"
        archive_path = temporary / "pkdb_export.zip"
        archive_download = _download(
            url=export_url,
            destination=archive_path,
            timeout_seconds=timeout_seconds,
        )
        audit = inspect_pkdb_bulk_export(
            archive_path=archive_path,
            output_directory=temporary / "inspected_export",
            filter_payload=filter_payload,
            compound=compound,
            source_url=export_url,
            response_headers=archive_download["headers"],
            archive_reference="pkdb_export.zip",
        )
        filter_download["local_path"] = "filter_response.json"
        archive_download["local_path"] = "pkdb_export.zip"
        manifest = {
            "schema_version": "1.0.0",
            "created_at": iso_now(),
            "source_id": "pkdb",
            "compound": compound,
            "open_licence_only": open_licence_only,
            "filter_request": filter_download,
            "export_request": archive_download,
            "filter_counts": {resource: int(filter_payload.get(resource) or 0) for resource in EXPECTED_TABLES},
            "filter_uuid": filter_payload.get("uuid"),
            "export_audit_sha256": audit["audit_sha256"],
            "accepted_for_observation_processing": audit["accepted_for_observation_processing"],
            "gap_codes": audit["gap_codes"],
            "raw_only": True,
        }
        manifest["manifest_sha256"] = sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        write_json(temporary / "acquisition_manifest.json", manifest)
        temporary.replace(output_directory)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire and audit an official PK-DB bulk export.")
    parser.add_argument("--compound", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    parser.add_argument("--include-closed-licence-metadata", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = acquire_pkdb_bulk_export(
        compound=args.compound,
        output_directory=args.output_directory,
        timeout_seconds=args.timeout_seconds,
        open_licence_only=not args.include_closed_licence_metadata,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if manifest["accepted_for_observation_processing"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
