from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any


class SecureZipExtractionError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.removeprefix("\\\\")
    return "\\\\?\\" + resolved


def _safe_member_path(name: str, expected_root_name: str) -> PurePosixPath:
    if "\\" in name:
        raise SecureZipExtractionError(
            f"ZIP member uses a non-canonical path separator: {name!r}"
        )
    member = PurePosixPath(name)
    if member.is_absolute() or not member.parts or ".." in member.parts:
        raise SecureZipExtractionError(f"Unsafe ZIP member path: {name!r}")
    if member.parts[0] != expected_root_name:
        raise SecureZipExtractionError(
            f"ZIP member is outside expected root {expected_root_name!r}: {name!r}"
        )
    if any(part in {"", "."} for part in member.parts):
        raise SecureZipExtractionError(f"Invalid ZIP member path: {name!r}")
    return member


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_reusable_manifest(
    *,
    destination: Path,
    manifest_path: Path,
    archive_sha256: str,
    expected_root_name: str,
) -> dict[str, Any] | None:
    if not destination.exists() and not manifest_path.exists():
        return None
    if not destination.is_dir() or not manifest_path.is_file():
        raise SecureZipExtractionError(
            "Repository destination and extraction manifest are inconsistent."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SecureZipExtractionError(
            f"Unable to read extraction manifest: {manifest_path}"
        ) from exc
    if (
        manifest.get("status") != "complete"
        or manifest.get("archive_sha256") != archive_sha256
        or manifest.get("expected_root_name") != expected_root_name
        or manifest.get("destination") != str(destination.resolve())
    ):
        raise SecureZipExtractionError(
            "Existing repository extraction does not match the locked archive."
        )
    manifest["reused"] = True
    return manifest


def extract_repository_zip(
    *,
    archive_path: Path,
    destination: Path,
    manifest_path: Path,
    expected_root_name: str,
    expected_sha256: str,
) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    destination = destination.resolve()
    manifest_path = manifest_path.resolve()
    if not archive_path.is_file():
        raise SecureZipExtractionError(f"Archive does not exist: {archive_path}")
    archive_sha256 = _sha256_file(archive_path)
    if archive_sha256 != expected_sha256.casefold():
        raise SecureZipExtractionError(
            "Repository archive checksum does not match the bundle manifest."
        )
    reusable = _load_reusable_manifest(
        destination=destination,
        manifest_path=manifest_path,
        archive_sha256=archive_sha256,
        expected_root_name=expected_root_name,
    )
    if reusable is not None:
        return reusable

    staging = destination.with_name(destination.name + ".tmp")
    if staging.exists():
        shutil.rmtree(_windows_long_path(staging))
    staging.mkdir(parents=True)

    member_count = 0
    file_count = 0
    total_uncompressed_bytes = 0
    seen_windows_paths: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                member = _safe_member_path(info.filename, expected_root_name)
                normalized = member.as_posix().rstrip("/").casefold()
                if normalized in seen_windows_paths:
                    raise SecureZipExtractionError(
                        f"Duplicate case-insensitive ZIP path: {info.filename!r}"
                    )
                seen_windows_paths.add(normalized)
                if _is_symlink(info):
                    raise SecureZipExtractionError(
                        f"Symlink ZIP members are not permitted: {info.filename!r}"
                    )
                target = staging.joinpath(*member.parts)
                target_long = _windows_long_path(target)
                if info.is_dir():
                    os.makedirs(target_long, exist_ok=True)
                else:
                    os.makedirs(
                        _windows_long_path(target.parent),
                        exist_ok=True,
                    )
                    copied = 0
                    with archive.open(info, "r") as source, open(
                        target_long,
                        "xb",
                    ) as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                            copied += len(chunk)
                    if copied != info.file_size:
                        raise SecureZipExtractionError(
                            f"Extracted size mismatch for {info.filename!r}."
                        )
                    file_count += 1
                    total_uncompressed_bytes += copied
                member_count += 1

        extracted_root = staging / expected_root_name
        top_level = list(staging.iterdir())
        if top_level != [extracted_root] or not extracted_root.is_dir():
            raise SecureZipExtractionError(
                "Repository archive did not produce exactly one expected root."
            )
        os.replace(
            _windows_long_path(extracted_root),
            _windows_long_path(destination),
        )
        staging.rmdir()
        manifest = {
            "manifest_schema_version": "1.0.0",
            "status": "complete",
            "archive": str(archive_path),
            "archive_sha256": archive_sha256,
            "destination": str(destination),
            "expected_root_name": expected_root_name,
            "member_count": member_count,
            "file_count": file_count,
            "total_uncompressed_bytes": total_uncompressed_bytes,
            "completed_at": datetime.now(UTC).isoformat(),
            "reused": False,
        }
        _write_json_atomic(manifest_path, manifest)
        return manifest
    except Exception:
        shutil.rmtree(_windows_long_path(staging), ignore_errors=True)
        if destination.exists() and not manifest_path.exists():
            shutil.rmtree(_windows_long_path(destination), ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely extract a checksum-locked OSP repository ZIP on Windows."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-root-name", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    result = extract_repository_zip(
        archive_path=args.archive,
        destination=args.destination,
        manifest_path=args.manifest,
        expected_root_name=args.expected_root_name,
        expected_sha256=args.expected_sha256,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
