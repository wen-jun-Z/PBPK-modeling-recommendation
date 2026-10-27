from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.windows_extract import (
    SecureZipExtractionError,
    extract_repository_zip,
)


def test_extract_repository_zip_supports_long_paths_and_resume(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "repository.zip"
    root_name = "OSP-PBPK-Model-Library-v12.3.1"
    long_relative = (
        f"{root_name}/Omeprazole/images/"
        + ("long-section-" * 8)
        + "/"
        + ("long-file-name-" * 8)
        + ".png"
    )
    assert len(long_relative) > 260
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
        archive.writestr(long_relative, b"image")

    destination = tmp_path / root_name
    manifest_path = tmp_path / "repository_extraction_manifest.json"
    first = extract_repository_zip(
        archive_path=archive_path,
        destination=destination,
        manifest_path=manifest_path,
        expected_root_name=root_name,
        expected_sha256=sha256_file(archive_path),
    )
    second = extract_repository_zip(
        archive_path=archive_path,
        destination=destination,
        manifest_path=manifest_path,
        expected_root_name=root_name,
        expected_sha256=sha256_file(archive_path),
    )

    assert first["reused"] is False
    assert first["file_count"] == 1
    assert destination.joinpath(*Path(long_relative).parts[1:]).read_bytes() == b"image"
    assert second["reused"] is True


def test_extract_repository_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    root_name = "OSP-PBPK-Model-Library-v12.3.1"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{root_name}/../../outside.txt", b"unsafe")

    destination = tmp_path / root_name
    with pytest.raises(SecureZipExtractionError, match="Unsafe ZIP member"):
        extract_repository_zip(
            archive_path=archive_path,
            destination=destination,
            manifest_path=tmp_path / "manifest.json",
            expected_root_name=root_name,
            expected_sha256=sha256_file(archive_path),
        )

    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_extract_repository_zip_rejects_checksum_mismatch(tmp_path: Path) -> None:
    archive_path = tmp_path / "repository.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("root/file.txt", b"value")

    with pytest.raises(SecureZipExtractionError, match="checksum"):
        extract_repository_zip(
            archive_path=archive_path,
            destination=tmp_path / "root",
            manifest_path=tmp_path / "manifest.json",
            expected_root_name="root",
            expected_sha256="0" * 64,
        )
